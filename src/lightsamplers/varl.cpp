// Copyright @TwoCookingMice 2020

#include "lightsamplers/varl.h"
#include "geometry.h"
#include "materials/ltc.h"
#include "reflection.h"
#include "stats.h"

#include <math.h>
#include <vector>

namespace pbrt {

STAT_COUNTER("Lightsampler/Total Hash Table Iterations", totalIterations);
STAT_MEMORY_COUNTER("Memory/Hash Table Size", totalHashTableSize);
STAT_PERCENT("Lightsampler/Hash Table Sparsity", liveLightCutCount, totalLightCutCount);
STAT_RATIO("Lightsampler/Avg Light Cut Size", totalLightCutNodeSize, lightCutCount);

const Float noChangeIterationLimit = 128.0f;

void VARLLightSampler::Preprocess(const Scene& scene, 
    const std::vector<std::shared_ptr<Light>>& lights, bool isVpl) {
  LightTreeSampler::Preprocess(scene, lights, isVpl);
  scene.ComputeShadingPointClusters(_shadingPointClusters);

  if (_initLightCutSize == 0) {
    _initLightCutSize = lights.size() > 8 ? 8 : lights.size();
    _maxLightCutSize = _initLightCutSize * 8 > lights.size() ? lights.size() : _initLightCutSize * 8;
  }
  _globalLightCut = _lightTree->generateLightcut(0.125f / _initLightCutSize, _initLightCutSize, 
      [&](const LinearLightTreeNode& node) {
      return node.power;
  });

  // Our Light Field is encoded as ((Phi, Theta), (shadingPointClusterIndex))
  _directionalGridSize[0] = Pi / _directionDivideNumber;
  _directionalGridSize[1] = 2.0f * Pi / _directionDivideNumber;

  uint32_t hashTableSize = _shadingPointClusters * _directionDivideNumber * _directionDivideNumber;
  _hashMap = std::vector<std::shared_ptr<HashTableEntry> >(hashTableSize, std::shared_ptr<HashTableEntry>(nullptr));
  // totalHashTableSize += _hashMap.size() * sizeof(HashTableEntry*);

  totalLightCutCount = hashTableSize;
}

Spectrum VARLLightSampler::Sample(const Vector3f& wo, const Interaction& it, const Scene& scene,
    MemoryArena& arena, Sampler& sampler,
    bool handleMedia, uint32_t nSamples) {
  if (!it.IsSurfaceInteraction()) {
    return Spectrum(0.0f);
  }
  SurfaceInteraction its = static_cast<const SurfaceInteraction&>(it);
  CHECK_GE(its.shadingPointCluster, 0);

  Vector3f position(its.p);
  Vector3f normal(its.shading.n);

  // Calculate hash table coordinates
  Float lightFieldEncoding[2] = { SphericalTheta(wo), SphericalPhi(wo) };
  uint32_t index = 0;
  uint32_t factor = 1;
  for (uint32_t idx = 0; idx < 2; ++idx) {
    Float gridMin;
    Float gridSize;
    int divideNumber;
    gridMin = 0.0f;
    gridSize = _directionalGridSize[idx];
    divideNumber = _directionDivideNumber;

    int hashTableCoordinate = std::floor((lightFieldEncoding[idx] - gridMin) / gridSize);
    hashTableCoordinate = std::min(hashTableCoordinate, divideNumber - 1);
    index += (factor * hashTableCoordinate);
    factor *= divideNumber;
  }
  index += its.shadingPointCluster * factor;

  // bool haveBRDF = false;
  // BRDFRecord record;
  // if (its.IsSurfaceInteraction()) {
  //   haveBRDF = LTC::GetBRDFRecord((const SurfaceInteraction&) its, record);
  // }

  std::vector<Float> weights(_lightTree->TotalNodes(), -1.0);
  auto weightingFunction = [&](const Vector3f& wo, 
      const Interaction& its, 
      const LinearLightTreeNode& n) {
    uint32_t nodeID = n.nodeID;
    if (weights[nodeID] >= 0.0f) {
      return weights[nodeID];
    }

    Float res;
    if (!its.IsSurfaceInteraction()) {
      res = n.power;
    } else {
      const SurfaceInteraction& iit = (const SurfaceInteraction&) its;
      Vector3f wi = Normalize(LinearLightTreeNodeCentroid(&n) - its.p);
      Vector3f point(iit.p);
      Vector3f normal(iit.shading.n);
      Float G = LinearLightTreeNodeGeoTermBound(&n, its.p, normal);
      Float brdfIntegral = 1.0f;

      // BoundingSphere bs;
      // n.bound.BoundingSphere(&bs.c, &bs.r);
      // if (haveBRDF) {
      //   brdfIntegral = LTC::EvaluatePivotIntegral(iit, record, bs);
      // }

      res = n.power * G * brdfIntegral;
    }

    weights[nodeID] = res;
    return res;
  };

  {
    std::lock_guard<std::mutex> _(_hashMapMutex[index&3]);
    if (_hashMap[index] == nullptr) {
      std::shared_ptr<LightCut> lightCut = _globalLightCut->Clone();
      uint32_t lightCutSize = lightCut->Size();

      for (uint32_t i = 0; i < lightCutSize; ++i) {
        uint32_t lightTreeNode = lightCut->_cut[i].lightTreeNodeIndex;
        const LinearLightTreeNode* node = _lightTree->getNodeByIndex(lightTreeNode);
        lightCut->Update(i, lightTreeNode, [&](LightCutNode* n) {
            n->weight = weightingFunction(wo, its, *node);
        });
     }
      lightCut->UpdateDistribution();
     _hashMap[index] = std::make_shared<HashTableEntry>(lightCut, _maxLightCutSize);

      ++liveLightCutCount;
      ++lightCutCount;
      totalLightCutNodeSize += lightCut->Size();
      totalHashTableSize += _hashMap[index]->MemoryCost();
    }
  }

  const std::shared_ptr<LightCut>& lightCut = _hashMap[index]->lightCut->Clone();
  nSamples = SampleAllocation(lightCut.get(), nSamples);
  Float lr = 1.0f / (4.0f * std::pow((_hashMap[index]->iteration), 0.857f));
  if (_learningRate >= 0.0f) {
    lr = _learningRate;
  }
  ++_hashMap[index]->iteration;
  auto callback = [&](uint32_t lightTreeIndex, const LightTreeSamplingResult& result) {};

  Spectrum l(0.0f);
  std::vector<SamplingRecord> samplingRecords;
  samplingRecords.reserve(nSamples);
  {
    Float uLightCut = sampler.Get1D();
    for (uint32_t i = 0; i < nSamples; ++i) {
      int initTreeNode;
      int cutNode;
      Float cutPdf;
      Float uLightTree = sampler.Get1D();
      lightCut->UniformSample(uLightCut, cutNode, initTreeNode, cutPdf, &uLightCut);

      Float reward;
      auto estimateDirectFunction = [&](const std::shared_ptr<Light>& light) {
        Point2f uLight = sampler.Get2D();
        Point2f uScattering = sampler.Get2D();
        Spectrum ld = EstimateSingleLightMIS(its, uScattering, *light, uLight, scene, sampler,
            arena, reward, handleMedia);

        return ld;
      };
      LightTreeSamplingResult result = _lightTree->Sample(wo, its, initTreeNode, 
          uLightTree, std::move(weightingFunction), 
          std::move(estimateDirectFunction), std::move(callback));

      if (cutPdf > 0.0f && result.second > 0.0f) {
        const Spectrum ld = result.first / result.second;
        reward = std::sqrt(reward * ShadingPointEstimation(_lightTree->getNodeByIndex(initTreeNode)) / 
            result.second);
        l += (ld / (cutPdf * nSamples));
        _hashMap[index]->samplingResults->Append(cutNode, result);
        samplingRecords.push_back(SamplingRecord(cutNode, initTreeNode, result, reward));

        if (_biased) {
          Float curMaxQValue = lightCut->MeanWeight();
          lightCut->Update(cutNode, initTreeNode, [&](LightCutNode* node) {
            Float wt = node->weight;
            const LinearLightTreeNode* lightTreeNode = _lightTree->getNodeByIndex(node->lightTreeNodeIndex);
            wt = (1.0f - lr) * wt + lr * (reward + _gamma * curMaxQValue);
            node->weight = wt;
          });
          lightCut->UpdateDistribution();
        }
      }
    }
  }

  {
    std::lock_guard<std::mutex> _(_hashMapMutex[index&3]);
    uint32_t lightCutSize = _hashMap[index]->lightCut->Size();
    if (lightCutSize < _maxLightCutSize && 
        _hashMap[index]->noChangeIterations < noChangeIterationLimit) {
    
      Float sumVariance = 1e-4f;
      Float sumMean = 1e-2f;
      for (int i = 0; i < lightCutSize; ++i) {
        sumVariance += _hashMap[index]->samplingResults->Var(i);
      }

      Float pE = 1.0f / (1.0f + (lightCut->Size() / _initLightCutSize) *  std::exp(-1.0f * sumVariance));
    
      Float curMaxQValue = _hashMap[index]->futureValue(_vStar);
      Float uSplit = sampler.Get1D();
      bool changed = false;
      for (const auto& samplingRecord : samplingRecords) {
        uint32_t lightCutIndex = samplingRecord.lightCutIndex;
        uint32_t lightTreeIndex = samplingRecord.lightTreeIndex;

        if (_hashMap[index]->lightCut->Verify(lightCutIndex, lightTreeIndex)) {
          const LinearLightTreeNode* node = _lightTree->getNodeByIndex(lightTreeIndex);
          Float variance = _hashMap[index]->samplingResults->Var(lightCutIndex);
          Float mean = _hashMap[index]->samplingResults->E[lightCutIndex];
          Float nSamples = _hashMap[index]->samplingResults->SampleCount(lightCutIndex);
          if (nSamples == 0) continue;
          Float pT = 1.0f - (1.0f / nSamples);
          Float pV = variance / sumVariance;
          // Float pE = 1.0f / (1 + std::exp(-1.0 * variance / (mean * mean + 0.0001f)));
          Float pSplit = pT * pE * pV;
          if (!LinearLightTreeNodeIsLeaf(node) && uSplit <= pSplit) {
            const LinearLightTreeNode* lchild = _lightTree->getNodeByIndex(lightTreeIndex+1);
            const LinearLightTreeNode* rchild = _lightTree->getNodeByIndex(node->secondChildOffset);
            Float lErrBound = weightingFunction(wo, it, *lchild);
            Float rErrBound = weightingFunction(wo, it, *rchild);
            Float lw = lErrBound / (lErrBound + rErrBound);
            Float rw = rErrBound / (lErrBound + rErrBound);
            Float lSampleCount = lw * nSamples;
            Float rSampleCount = rw * nSamples;
            Float lE = lw * _hashMap[index]->samplingResults->Exp2(lightCutIndex);
            Float rE = rw * _hashMap[index]->samplingResults->Exp2(lightCutIndex);
            Float lC = std::pow(1.0f - lr, lSampleCount);
            Float rC = std::pow(1.0f - lr, rSampleCount);
            Float lWeight = (std::sqrt(lE * ShadingPointEstimation(lchild)) + _gamma * curMaxQValue) * 
              (1.0f - lC) + lErrBound * lC;
            Float rWeight = (std::sqrt(rE * ShadingPointEstimation(rchild)) + _gamma * curMaxQValue) * 
              (1.0f - rC) + rErrBound * rC;
            _hashMap[index]->lightCut->Update(lightCutIndex, lightTreeIndex, [&](LightCutNode* n) {
                n->lightTreeNodeIndex += 1;
                n->weight = lWeight;
                _hashMap[index]->samplingResults->Reset(lightCutIndex);
              });
            _hashMap[index]->lightCut->Append(LightCutNode(node->secondChildOffset, rWeight));
            uSplit = uSplit / pSplit;
            ++totalLightCutNodeSize;
            changed |= true;
            totalHashTableSize += (4 * sizeof(Float) + sizeof(LightCutNode));
          } else if (uSplit <= pSplit) {
            uSplit = uSplit / pSplit;
          } else {
            uSplit = (uSplit - pSplit) / (1.0f - pSplit);
          }
        }

       if (_hashMap[index]->lightCut->Size() >= _maxLightCutSize) {
         break;
       }
      }

      if (changed) {
        _hashMap[index]->noChangeIterations = 0.0f;
      } else {
        _hashMap[index]->noChangeIterations += std::ceil(nSamples / lightCutSize);
      }
    }
  }

  Float curMaxQValue = _hashMap[index]->futureValue(_vStar);
  for (const auto& samplingRecord : samplingRecords) {
    _hashMap[index]->lightCut->Update(samplingRecord.lightCutIndex, samplingRecord.lightTreeIndex, 
        [&](LightCutNode* node) {
      Float wt = node->weight;
      const LinearLightTreeNode* lightTreeNode = _lightTree->getNodeByIndex(node->lightTreeNodeIndex);
      wt = (1.0f - lr) * wt + lr * (samplingRecord.reward + _gamma * curMaxQValue);
      node->weight = wt;
    });
  }

  _hashMap[index]->lightCut->UpdateDistribution();
  ++totalIterations;

  return l;
}

uint32_t VARLLightSampler::SampleAllocation(const LightCut* lightCut, uint32_t proposedSampleCount) const {
  if (!_isAdaptive) {
    return proposedSampleCount;
  } else {
    Float ratio = Clamp((float) lightCut->Size() / _initLightCutSize, 0.99f, 2.0f);
    uint32_t res = std::floor(proposedSampleCount * ratio);
    // Float ratio = std::min(std::pow((float)lightCut->Size() / _initLightCutSize, 3.0f), 3.375f);
    // uint32_t res = std::ceil(proposedSampleCount * std::max(ratio, 1.0f));
    return std::max(res, 1u);
  }
}

inline Float VARLLightSampler::ShadingPointEstimation(const LinearLightTreeNode* node) const {
  if (_isVpl) {
    return static_cast<Float>(node->nLight);
  } else {
    return static_cast<Float>(node->nLight);
    // Float angle = node->thetaO > 0.0f ? node->thetaO / 2.0f : Pi / 2.0f;
    // return node->bound.SurfaceArea() * angle / Pi;
  }
}

Spectrum VARLLightSampler::EstimateSingleLightMIS(const Interaction &it, const Point2f &uScattering,
                                                  const Light &light, const Point2f &uLight,
                                                  const Scene &scene, Sampler &sampler,
                                                  MemoryArena &arena, Float& reward,
                                                  bool handleMedia, bool specular) const {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0.0f;
    Float scatteringPdf = 0.0f;
    reward = 0.0f;
    Float lReward = 1.0f;
    Float bReward = 1.0f;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li << ", wi: "
            << wi << ", pdf: " << lightPdf;
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
            scatteringPdf = isect.bsdf->Pdf(isect.wo, wi, bsdfFlags);
            VLOG(2) << "  surf f*dot :" << f << ", scatteringPdf: " << scatteringPdf;
        } else {
            // Evaluate phase function for light sampling strategy
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->p(mi.wo, wi);
            f = Spectrum(p);
            scatteringPdf = p;
            VLOG(2) << "  medium p: " << p;
        }
        if (!f.IsBlack()) {
            // Compute effect of visibility for light source sample
            if (handleMedia) {
                Li *= visibility.Tr(scene, sampler);
                VLOG(2) << "  after Tr, Li: " << Li;
            } else {
              if (!visibility.Unoccluded(scene)) {
                VLOG(2) << "  shadow ray blocked";
                Li = Spectrum(0.f);
              } else
                VLOG(2) << "  shadow ray unoccluded";
            }

            // Add light's contribution to reflected radiance
            if (!Li.IsBlack()) {
                if (IsDeltaLight(light.flags)) {
                    Ld += f * Li / lightPdf;
                    lReward = Spectrum(f * Li).y();
                    reward += (lReward * lReward / lightPdf);
                } else {
                    Float weight =
                        PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                    lReward = Spectrum(f * Li).y() * weight;
                    reward += (lReward * lReward * light.WorldBound().SurfaceArea() / lightPdf);
                    lReward /= lightPdf;
                    Ld += f * Li * weight / lightPdf;
                }
            }
        }
    }

    // Sample BSDF with multiple importance sampling
    if (!IsDeltaLight(light.flags)) {
        Spectrum f;
        Float cosTheta = 1.0f;
        bool sampledSpecular = false;
        if (it.IsSurfaceInteraction()) {
            // Sample scattered direction for surface interactions
            BxDFType sampledType;
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->Sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
                                     bsdfFlags, &sampledType);
            f *= AbsDot(wi, isect.shading.n);
            cosTheta = AbsDot(wi, isect.shading.n);
            sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
        } else {
            // Sample scattered direction for medium interactions
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->Sample_p(mi.wo, &wi, uScattering);
            f = Spectrum(p);
            cosTheta = AbsDot(wi, mi.n);
            scatteringPdf = p;
        }
        VLOG(2) << "  BSDF / phase sampling f: " << f << ", scatteringPdf: " <<
            scatteringPdf;
        if (!f.IsBlack() && scatteringPdf > 0) {
            // Account for light contributions along sampled direction _wi_
            Float weight = 1;
            if (!sampledSpecular) {
                lightPdf = light.Pdf_Li(it, wi);
                if (lightPdf == 0) return Ld;
                weight = PowerHeuristic(1, scatteringPdf, 1, lightPdf);
            }

            // Find intersection and compute transmittance
            SurfaceInteraction lightIsect;
            Ray ray = it.SpawnRay(wi);
            Spectrum Tr(1.f);
            bool foundSurfaceInteraction =
                handleMedia ? scene.IntersectTr(ray, sampler, &lightIsect, &Tr)
                            : scene.Intersect(ray, &lightIsect);

            // Add light contribution from material sampling
            Spectrum Li(0.f);
            if (foundSurfaceInteraction) {
                if (lightIsect.primitive->GetAreaLight() == &light)
                    Li = lightIsect.Le(-wi);
            } else
                Li = light.Le(ray);
            if (!Li.IsBlack()) {
              Ld += f * Li * Tr * weight / scatteringPdf;
              bReward = Spectrum(f * Li * Tr).y() * weight / scatteringPdf;
              reward += (bReward * bReward);
              reward += (2.0f * lReward * bReward);
            }
        }
    }

    return Ld;
}

Spectrum VARLLightSampler::EstimateSingleLight(const Interaction &it, const Light &light, 
                                                  const Point2f &uLight, const Scene &scene, 
                                                  Sampler &sampler, MemoryArena &arena, 
                                                  Float& reward, bool handleMedia, bool specular) const {
    BxDFType bsdfFlags = BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float lightPdf = 0.0f;
    reward = 0.0f;
    Float lReward = 1.0f;
    Float bReward = 1.0f;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li << ", wi: "
            << wi << ", pdf: " << lightPdf;
    if (lightPdf > 0 && !Li.IsBlack()) {
        // Compute BSDF or phase function's value for light sample
        Spectrum f;
        if (it.IsSurfaceInteraction()) {
            // Evaluate BSDF for light sampling strategy
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = isect.bsdf->f(isect.wo, wi, bsdfFlags) *
                AbsDot(wi, isect.shading.n);
        } else {
            // Evaluate phase function for light sampling strategy
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase->p(mi.wo, wi);
            f = Spectrum(p);
            VLOG(2) << "  medium p: " << p;
        }
        if (!f.IsBlack()) {
            // Compute effect of visibility for light source sample
            if (handleMedia) {
                Li *= visibility.Tr(scene, sampler);
                VLOG(2) << "  after Tr, Li: " << Li;
            } else {
              if (!visibility.Unoccluded(scene)) {
                VLOG(2) << "  shadow ray blocked";
                Li = Spectrum(0.f);
              } else
                VLOG(2) << "  shadow ray unoccluded";
            }

            // Add light's contribution to reflected radiance
            if (!Li.IsBlack()) {
              Ld += f * Li / lightPdf;
              lReward = Spectrum(f * Li).y();
              reward += (lReward * lReward / lightPdf);
            }
        }
    }

    return Ld;
}

LightSampler* CreateVARLLightSampler(const ParamSet& params) {
  uint32_t directionalGridNumber = params.FindOneInt("directionalgridnumber", 8);
  // Defaultly we apply 1/T for learning rate.
  Float learningRate = params.FindOneFloat("learningrate", -1.0);
  Float gamma = params.FindOneFloat("gamma", 0.7f);
  uint32_t maxLightCutSize = params.FindOneInt("maxlightcutsize", 0);
  uint32_t initLightCutSize = params.FindOneInt("initlightcutsize", 0);
  bool isAdaptive = params.FindOneBool("adaptive", true);
  uint32_t shadingPointClusters = params.FindOneInt("shadingpointclusters", 32768);
  bool biased = params.FindOneBool("biased", false);
  std::string strategy = params.FindOneString("vstar", "zero");
  return new VARLLightSampler(directionalGridNumber, learningRate, gamma, 
      initLightCutSize, maxLightCutSize, isAdaptive, shadingPointClusters, biased, strategy);
}

} // namespace pbrt
