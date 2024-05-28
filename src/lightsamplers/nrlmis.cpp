// Copyright @TwoCookingMice 2020

#include "lightsamplers/nrlmis.h"
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
STAT_COUNTER("Lightsampler/Merge Collapse Count", updateCount);

const Float noChangeIterationLimit = 128.0f;

void NRLMISLightSampler::Preprocess(const Scene& scene, 
    const std::vector<std::shared_ptr<Light>>& lights, bool isVpl) {
  LightTreeSampler::Preprocess(scene, lights, isVpl);
  scene.ComputeShadingPointClusters(_shadingPointClusters);

  _globalLightCut = _lightTree->generateLightcut(0.0001f, _maxLightCutSize, 
      [&](const LinearLightTreeNode& node) {
      return node.power;
  });

  // Our Light Field is encoded as ((Phi, Theta), (shadingPointClusterIndex))
  _directionalGridSize[0] = Pi / _directionDivideNumber;
  _directionalGridSize[1] = 2.0f * Pi / _directionDivideNumber;

  uint32_t hashTableSize = _shadingPointClusters * _directionDivideNumber * _directionDivideNumber;
  _hashMap = std::vector<std::shared_ptr<HashTableEntry> >(hashTableSize, std::shared_ptr<HashTableEntry>(nullptr));
  // totalHashTableSize += _hashMap.size() * sizeof(HashTableEntry*);

  std::cout << hashTableSize << std::endl;
  totalLightCutCount = hashTableSize;
}

Spectrum NRLMISLightSampler::Sample(const Vector3f& wo, const Interaction& it, const Scene& scene,
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

  // // bool haveBRDF = false;
  // // BRDFRecord record;
  // // if (its.IsSurfaceInteraction()) {
  // //   haveBRDF = LTC::GetBRDFRecord((const SurfaceInteraction&) its, record);
  // // }

  auto weightingFunction = [&](const Vector3f& wo, 
      const Interaction& its, 
      const LinearLightTreeNode& n) {
    return n.power;
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
     _hashMap[index] = std::make_shared<HashTableEntry>(lightCut, _lightTree->TotalNodes());

      ++liveLightCutCount;
      ++lightCutCount;
      totalLightCutNodeSize += lightCut->Size();
      totalHashTableSize += _hashMap[index]->MemoryCost();
    }
  }

  const std::shared_ptr<LightCut>& lightCut = _hashMap[index]->lightCut->Clone();
  Float lr = _learningRate;
  if (lr <= 0.0f) {
    lr = 1.0f / (4.0f * std::pow((_hashMap[index]->iteration), 0.857f));
  }
  ++_hashMap[index]->iteration;
  auto callback = [&](uint32_t lightTreeIndex, const LightTreeSamplingResult& result) {};

  Spectrum l(0.0f);
  std::vector<SamplingRecord> samplingRecords;
  std::vector<Float> misWeights;
  samplingRecords.reserve(nSamples);
  misWeights.reserve(nSamples);
  {
    Float uLightCut = sampler.Get1D();
    Float pBRDF = 1.0f;
    Float uBRDF = sampler.Get1D();
    uint32_t nBRDFSamples = 0;
    uint32_t nVARLSamples = 0;
    std::vector<LightSamplingResult> results;
    results.reserve(nSamples);
    for (uint32_t i = 0; i < nSamples; ++i) {
      if (uBRDF < pBRDF) {
        ++nBRDFSamples;
        uBRDF = uBRDF / pBRDF;
        Float brdfScatteringPdf = 0.0f;
        Float brdfLightPdf = 0.0f;
        auto lightTreePdfFunc = [&](const Interaction& isect, const Vector3f& wi, const Light* light) {
          Float lightCutPdf = 1.0f;
          Float lightTreeSamplingPdf = 1.0f;
          uint32_t lightcutIndex;

          for (uint32_t idx = 0; idx < lightCut->Size(); ++idx) {
            int lightTreeNodeIndex = lightCut->_cut[idx].lightTreeNodeIndex;
            const LinearLightTreeNode* lightTreeNode = _lightTree->getNodeByIndex(lightTreeNodeIndex);
            if (light->index >= lightTreeNode->lightsOffset &&
                light->index < lightTreeNode->lightsOffset + lightTreeNode->nLight) {
              lightcutIndex = idx;
              break;
            }
          }
          int lightTreeNodeIndex = lightCut->_cut[lightcutIndex].lightTreeNodeIndex;
          const LinearLightTreeNode* lightTreeNode = _lightTree->getNodeByIndex(lightTreeNodeIndex);
          lightCutPdf = lightCut->Pdf(lightcutIndex);
          lightTreeSamplingPdf = _lightTree->Pdf(
              lightTreeNodeIndex, wi, its, light, std::move(weightingFunction));

          return lightCutPdf * lightTreeSamplingPdf;
        };
        Spectrum lBRDF = NaiveBRDFSampling(its, sampler.Get2D(), scene, sampler, brdfScatteringPdf, 
            brdfLightPdf, std::move(lightTreePdfFunc), false, false);
        if (!lBRDF.IsBlack() && brdfScatteringPdf > 0.0f && brdfLightPdf > 0.0f) {
          results.push_back(LightSamplingResult(0, lBRDF, brdfLightPdf, 
                brdfScatteringPdf, 0.0f, SamplingMethod::BRDF));
          pBRDF = std::min(pBRDF + 4.0f / nSamples, 0.99f);
        } else {
          pBRDF = std::max(pBRDF - 2.0f / nSamples, 0.01f);
        }
      } else {
        uBRDF = (uBRDF - pBRDF) / (1.0f - pBRDF);
      }

      {
        ++nVARLSamples;
        int initTreeNode;
        int cutNode;
        Float cutPdf;
        Float uLightTree = sampler.Get1D();
        lightCut->UniformSample(uLightCut, cutNode, initTreeNode, cutPdf, &uLightCut);

        Float reward;
        Float lightPdf;
        Float scatteringPdf;
        auto estimateDirectFunction = [&](const std::shared_ptr<Light>& light) {
          Point2f uLight = sampler.Get2D();
          Point2f uScattering = sampler.Get2D();
          Spectrum ld = EstimateSingleLight(its, *light, uLight, scene, sampler,
              arena, reward, lightPdf, scatteringPdf, handleMedia);

          return ld;
        };
        LightTreeSamplingResult result = _lightTree->Sample(wo, its, initTreeNode, 
            uLightTree, std::move(weightingFunction), 
            std::move(estimateDirectFunction), std::move(callback));

        if (cutPdf > 0.0f && result.second > 0.0f && lightPdf > 0.0f) {
          const Spectrum ld = result.first / result.second;
          reward = std::sqrt(reward * 
              ShadingPointEstimation(_lightTree->getNodeByIndex(initTreeNode)) / 
              result.second);
          LightSamplingResult sresult = LightSamplingResult(cutNode, result.first, 
                lightPdf * result.second * cutPdf, scatteringPdf, 
                reward, SamplingMethod::VARL);
          sresult.lightTreeIndex = initTreeNode;
          results.push_back(sresult);
          result.first /= lightPdf;
          samplingRecords.push_back(SamplingRecord(cutNode, initTreeNode, result, reward));
        }
      }
    }

    for (const auto& result : results) {
      Spectrum ld;
      Float lightPdf = result.lightPdf;
      Float scatteringPdf = result.scatteringPdf;
      if (result.method == SamplingMethod::BRDF) {
        Float weight = PowerHeuristic(nBRDFSamples, scatteringPdf, nVARLSamples, lightPdf);
        ld = result.l * weight / result.scatteringPdf;
        l += (ld / nBRDFSamples);
      } else {
        Float weight = PowerHeuristic(nVARLSamples, lightPdf, nBRDFSamples, scatteringPdf);
        ld = result.l * weight / lightPdf;
        l += (ld / nVARLSamples);
        misWeights.push_back(weight);
      }
    }
  }

  {
    if (_hashMap[index]->lightCut->Size() > 2) {
      std::lock_guard<std::mutex> _(_hashMapMutex[index&3]);
      uint32_t lightCutSize = _hashMap[index]->lightCut->Size();

      uint32_t maxWeightLightCutIndex = -1;
      Float maxWeight = -1.0;
      for (uint32_t idx = 0; idx < lightCutSize; ++idx) {
        int lightTreeNodeIndex = _hashMap[index]->lightCut->_cut[idx].lightTreeNodeIndex;
        Float curWeight = _hashMap[index]->lightCut->_cut[idx].weight;
        if (!LinearLightTreeNodeIsLeaf(_lightTree->getNodeByIndex(lightTreeNodeIndex)) && 
            curWeight > maxWeight) {
          maxWeight = curWeight;
          maxWeightLightCutIndex = idx;
        }
      } 

      if (maxWeightLightCutIndex >= 0) {
        uint32_t minWeightLightTreeNodeIndex = -1;
        Float minWeight = 1e+9;
        for (uint32_t idx = 0; idx < lightCutSize; ++idx) {
          int lightTreeNodeIndex = _hashMap[index]->lightCut->_cut[idx].lightTreeNodeIndex;
          const LinearLightTreeNode* node = _lightTree->getNodeByIndex(lightTreeNodeIndex);
          const LinearLightTreeNode* parentNode = _lightTree->getNodeByIndex(node->parent);

          float lWeight = _hashMap[index]->Weight(node->parent+1);
          float rWeight = _hashMap[index]->Weight(parentNode->secondChildOffset);

          if (lWeight >= 0.0f && rWeight >= 0.0f && lWeight + rWeight < minWeight) {
            minWeight = lWeight + rWeight;
            minWeightLightTreeNodeIndex = lightTreeNodeIndex;
          }
        }

        if (minWeightLightTreeNodeIndex >= 0 && minWeight < maxWeight * 0.01) {
          const LinearLightTreeNode* mergeNode = _lightTree->getNodeByIndex(minWeightLightTreeNodeIndex);
          const LinearLightTreeNode* mergeParentNode = _lightTree->getNodeByIndex(mergeNode->parent);
          int mergeNodeCutIndex = _hashMap[index]->treeNodeToCutNode[mergeNode->parent+1];
          int mergeNodeCutIndex2 = _hashMap[index]->treeNodeToCutNode[mergeParentNode->secondChildOffset];
          int splitNodeTreeIndex = _hashMap[index]->lightCut->_cut[maxWeightLightCutIndex].lightTreeNodeIndex;
          const LinearLightTreeNode* splitNode = _lightTree->getNodeByIndex(splitNodeTreeIndex);
          int splitNodeSecondChild = splitNode->secondChildOffset;

          _hashMap[index]->lightCut->Update(mergeNodeCutIndex, mergeNode->parent+1, 
              [&](LightCutNode* cutNode) {
              cutNode->lightTreeNodeIndex = mergeNode->parent;
              cutNode->weight = minWeight;
              _hashMap[index]->treeNodeToCutNode[mergeNode->parent+1] = -1;
              _hashMap[index]->treeNodeToCutNode[mergeNode->parent] = mergeNodeCutIndex;
              });
          _hashMap[index]->lightCut->Update(maxWeightLightCutIndex, splitNodeTreeIndex, 
              [&](LightCutNode* cutNode) {
              cutNode->lightTreeNodeIndex += 1;
              cutNode->weight = _lightTree->getNodeByIndex(cutNode->lightTreeNodeIndex)->power;
              _hashMap[index]->treeNodeToCutNode[splitNodeTreeIndex+1] = maxWeightLightCutIndex;
              _hashMap[index]->treeNodeToCutNode[splitNodeTreeIndex] = -1;
              });
          _hashMap[index]->lightCut->Update(mergeNodeCutIndex2, mergeParentNode->secondChildOffset, 
              [&](LightCutNode* cutNode) {
              cutNode->lightTreeNodeIndex = splitNodeSecondChild;
              cutNode->weight = _lightTree->getNodeByIndex(cutNode->lightTreeNodeIndex)->power;
              _hashMap[index]->treeNodeToCutNode[mergeParentNode->secondChildOffset] = -1;
              _hashMap[index]->treeNodeToCutNode[splitNodeSecondChild] = mergeNodeCutIndex2;
              });
          ++updateCount;
        }
      }

    }
  }

  {
    uint32_t misIndex = 0;
    for (const auto& samplingRecord : samplingRecords) {
      _hashMap[index]->lightCut->Update(samplingRecord.lightCutIndex, samplingRecord.lightTreeIndex, 
          [&](LightCutNode* node) {
        Float wt = node->weight;
        const LinearLightTreeNode* lightTreeNode = _lightTree->getNodeByIndex(node->lightTreeNodeIndex);
        wt = (1.0f - lr) * wt + lr * (samplingRecord.reward * misWeights[misIndex++]);
        node->weight = wt;
      });
    }
  }

  _hashMap[index]->lightCut->UpdateDistribution();
  ++totalIterations;

  return l;
}

inline Float NRLMISLightSampler::ShadingPointEstimation(const LinearLightTreeNode* node) const {
  if (_isVpl) {
    return static_cast<Float>(node->nLight);
  } else {
    return static_cast<Float>(node->nLight);
    // Float angle = node->thetaO > 0.0f ? node->thetaO / 2.0f : Pi / 2.0f;
    // return node->bound.SurfaceArea() * angle / Pi;
  }
}

Spectrum NRLMISLightSampler::EstimateSingleLight(const Interaction &it, const Light &light, 
                                                  const Point2f &uLight, const Scene &scene, 
                                                  Sampler &sampler, MemoryArena &arena, 
                                                  Float& reward, Float& lightPdf, 
                                                  Float& scatteringPdf, bool handleMedia, bool specular) const {
    BxDFType bsdfFlags = BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    lightPdf = 0.0f;
    reward = 0.0f;
    scatteringPdf = 1.0f;
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
              Ld += f * Li;
              lReward = Spectrum(f * Li).y();
              reward += (lReward * lReward / lightPdf);
            }
        }
    }

    return Ld;
}

LightSampler* CreateNRLMISLightSampler(const ParamSet& params) {
  uint32_t directionalGridNumber = params.FindOneInt("directionalgridnumber", 8);
  // Defaultly we apply 1/T for learning rate.
  Float learningRate = params.FindOneFloat("learningrate", -1.0);
  uint32_t maxLightCutSize = params.FindOneInt("maxlightcutsize", 0);
  uint32_t shadingPointClusters = params.FindOneInt("shadingpointclusters", 32768);
  bool biased = params.FindOneBool("biased", false);
  return new NRLMISLightSampler(directionalGridNumber, learningRate, maxLightCutSize, shadingPointClusters, biased);
}

} // namespace pbrt
