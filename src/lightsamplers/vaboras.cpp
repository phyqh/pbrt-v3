/*************************************************************************
    > File Name: src/lightsamplers/vaboras.cpp
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Sunday, February 21, 2021 PM04:45:40 HKT
 ************************************************************************/

#include "lightsamplers/vaboras.h"

#include "light.h"
#include "reflection.h"
#include "stats.h"

namespace pbrt {

STAT_PERCENT("Lightsampler/Hash Table Sparsity", liveLightCutCount, totalLightCutCount);
STAT_RATIO("Lightsampler/Avg Light Cut Size", totalLightCutNodeSize, lightCutCount);

void VarianceAwareBayesianOnlineRegressionLightSampler::Preprocess(const Scene &scene, 
    const std::vector<std::shared_ptr<Light>> &lights, bool isVpl) {
  LightTreeSampler::Preprocess(scene, lights, isVpl);
  scene.ComputeShadingPointClusters(_shadingpointclusters);

  _worldBound = scene.WorldBound();
  _directionalGridSize[0] = Pi / _directionalGridNumber;
  _directionalGridSize[1] = 2 * Pi / _directionalGridNumber;

  uint32_t hashTableSize = _shadingpointclusters * _directionalGridNumber * _directionalGridNumber;
  _hashMap = std::vector<std::shared_ptr<HashTableEntry> >(hashTableSize, 
      std::shared_ptr<HashTableEntry>(nullptr));

  totalLightCutCount = hashTableSize;
}

Spectrum VarianceAwareBayesianOnlineRegressionLightSampler::Sample(const Vector3f &wi, const Interaction &it, 
    const Scene &scene, MemoryArena &arena, Sampler &sampler, bool handleMedia, uint32_t nSamples) {
  if (!it.IsSurfaceInteraction()) {
    return Spectrum(0.0f);
  }
  SurfaceInteraction its = static_cast<const SurfaceInteraction&>(it);
  CHECK_GE(its.shadingPointCluster, 0);

  Vector3f p(its.p);
  Vector3f n(its.shading.n);

  Float lightFieldEncoding[2] = { SphericalTheta(n), SphericalPhi(n) };
  uint32_t index = 0;
  uint32_t factor = 1;
  for (uint32_t idx = 0; idx < 2; ++idx) {
    Float gridMin;
    Float gridSize;
    int divideNumber;
    gridMin = 0.0f;
    gridSize = _directionalGridSize[idx];
    divideNumber = _directionalGridNumber;

    int hashTableCoordinate = std::floor((lightFieldEncoding[idx] - gridMin) / gridSize);
    hashTableCoordinate = Clamp(hashTableCoordinate, 0, divideNumber - 1);
    index += (factor * hashTableCoordinate);
    factor *= divideNumber;
  }
  index += its.shadingPointCluster * factor;

  auto lcBar = [&](const LinearLightTreeNode& node) {
    Vector3f wo = Normalize(LinearLightTreeNodeCentroid(&node) - its.p);
    Vector3f n(its.shading.n);
    Float G = LinearLightTreeNodeGeoTermBound(&node, its.p, n);

    if (_isVpl) {
      return node.power * G;
    } else {
      return node.power * G * node.bound.SurfaceArea() / node.nLight;
    }
  };


  auto weightingFunction = [&](const Vector3f& wi, const Interaction& its,
      const LinearLightTreeNode& n) {
    return n.power;
  };

  std::shared_ptr<LightCut> lightCut;
  {
    std::lock_guard<std::mutex> _(_hashMapMutex[index&3]);
    if (_hashMap[index] == nullptr) {
      auto lightCut = _lightTree->generateLightcut(0.1f, 100, std::move(lcBar));
      _hashMap[index] = std::make_shared<HashTableEntry>(lightCut);
      totalLightCutNodeSize += lightCut->Size();
      ++liveLightCutCount;
      ++lightCutCount;
    }
    lightCut = _hashMap[index]->lightCut->Clone();
  }

  Float avgLcBar = 0.0f;
  for (int i = 0; i < lightCut->Size(); ++i) {
    int lightTreeIndex = lightCut->_cut[i].lightTreeNodeIndex;
    const LinearLightTreeNode* treeNode = _lightTree->getNodeByIndex(lightTreeIndex);
    avgLcBar += lcBar(*_lightTree->getNodeByIndex(lightTreeIndex)) *
      (LinearLightTreeNodeCentroid(treeNode) - its.p).LengthSquared() / lightCut->Size();
  }
  for (int i = 0; i < lightCut->Size(); ++i) {
    int lightTreeIndex = lightCut->_cut[i].lightTreeNodeIndex;
    const LinearLightTreeNode* treeNode = _lightTree->getNodeByIndex(lightTreeIndex);
    Float no = _hashMap[index]->no[i];
    Float nv = _hashMap[index]->nv[i];
    Float s1x = _hashMap[index]->s1x[i];
    Float s2x = _hashMap[index]->s2x[i];
    Float d = (LinearLightTreeNodeCentroid(treeNode) - its.p).Length();
    Float mu0 = 0.5f * (avgLcBar + lcBar(*treeNode)) * d * d;

    Float p0 = (-1 + _nOBar + no) / (-2 + _nOBar + _nVBar + no + nv);
    Float k = s1x * (nv / (_nBar + nv)) + mu0 * (_nBar / (_nBar + nv));
    Float hBase = (2 * _nAlphaBar + nv - 1) * (_nBar + nv);
    Float h = -2 * mu0 * s1x * (_nBar * nv / hBase) - s1x * s1x * (nv * nv / hBase) +
      mu0 * mu0 * _nBar * nv / hBase + s2x * ((_nBar + nv) / hBase * nv) + 
      2 * _beta * ((_nBar + nv) / hBase);

    Float p = std::sqrt((1 - p0) * (p0 * k * k + h) + (1 - p0) * (1 - p0) * k * k) / (d * d);

    lightCut->Update(i, lightTreeIndex, [&](LightCutNode* lightCutNode) {
        lightCutNode->weight = p;
    });
  }
  lightCut->UpdateDistribution();

  auto callback = [&](uint32_t, const LightTreeSamplingResult&){};

  Spectrum l(0.0f);
  if (_useMIS && !_isVpl) {
    Float pBRDF = 0.5f;
    Float uBRDF = sampler.Get1D();
    uint32_t nBRDFSamples = 0;
    uint32_t nBORASSamples = 0;
    {
      std::vector<LightSamplingRecord> records;
      for (int i = 0; i < nSamples; ++i) {
        // Naive BRDF Sampling
        {
          int lightCutNodeIndex;
          int lightTreeNodeIndex;
          Float lightCutPdf;
          lightCut->UniformSample(sampler.Get1D(), lightCutNodeIndex, lightTreeNodeIndex, lightCutPdf);
          const LinearLightTreeNode* lightTreeNode = _lightTree->getNodeByIndex(lightTreeNodeIndex);

          Float value = 0.0f;
          Float lightPdf = 0.0f;
          Float scatteringPdf = 0.0f;
          Point2f uLight = sampler.Get2D();
          auto estimationFunction = [&](const std::shared_ptr<Light>& light) {
            BxDFType bsdfFlags = BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
            Spectrum Ld(0.f);
            // Sample light source with multiple importance sampling
            Vector3f wi;
            VisibilityTester visibility;
            Spectrum Li = light->Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
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
                        value = 0.0f;
                      } else
                        VLOG(2) << "  shadow ray unoccluded";
                    }

                    // Add light's contribution to reflected radiance
                    if (!Li.IsBlack()) {
                      Ld += f * Li;
                      value = Spectrum(f * Li).y();
                    }
                }
            }

            return Ld;
          };

          LightTreeSamplingResult result = _lightTree->Sample(wi, its, lightTreeNodeIndex, 
              sampler.Get1D(), std::move(weightingFunction), 
              std::move(estimationFunction), std::move(callback));

          ++nBORASSamples;
          if (lightCutPdf > 0.0f && result.second > 0.0f && lightPdf > 0.0f) {
            records.push_back(LightSamplingRecord(lightCutNodeIndex, result.first, 
                  lightPdf * result.second * lightCutPdf, scatteringPdf, value / (result.second * lightPdf),
                  SamplingMethod::BORAS));
            if (result.first.IsBlack()) {
              _hashMap[index]->ReportOccluded(lightCutNodeIndex);
            }
          }
        }

        if (uBRDF < pBRDF) {
          ++nBRDFSamples;
          uBRDF = uBRDF / pBRDF;
          Float brdfScatteringPdf = 0.0f;
          Float brdfLightPdf = 0.0f;
          auto weightingFunction = [](const Vector3f& wi, const Interaction& its,
              const LinearLightTreeNode& n) {
              return n.power;
          };
          auto lightTreePdfFunc = [&](const Interaction& isect, const Vector3f& wi, const Light* light) {
            Float lightCutPdf = 1.0f;
            Float lightTreeSamplingPdf = 1.0f;
            uint32_t lightcutIndex;
            auto it = _hashMap[index]->lightSamplingCache.find(light->index);
            if (it != _hashMap[index]->lightSamplingCache.end()) {
              lightcutIndex = _hashMap[index]->lightSamplingCache[light->index];
            } else {
              for (uint32_t idx = 0; idx < lightCut->Size(); ++idx) {
                int lightTreeNodeIndex = lightCut->_cut[idx].lightTreeNodeIndex;
                const LinearLightTreeNode* lightTreeNode = _lightTree->getNodeByIndex(lightTreeNodeIndex);
                if (light->index >= lightTreeNode->lightsOffset &&
                    light->index < lightTreeNode->lightsOffset + lightTreeNode->nLight) {
                  lightcutIndex = idx;
                  _hashMap[index]->lightSamplingCache[light->index] = idx;
                  break;
                }
              }
            }
            int lightTreeNodeIndex = lightCut->_cut[lightcutIndex].lightTreeNodeIndex;
            const LinearLightTreeNode* lightTreeNode = _lightTree->getNodeByIndex(lightTreeNodeIndex);
            lightCutPdf = lightCut->Pdf(lightcutIndex);
            lightTreeSamplingPdf = light->Power().y() / lightTreeNode->power;
            return lightCutPdf * lightTreeSamplingPdf;
          };
          Spectrum lBRDF = NaiveBRDFSampling(its, sampler.Get2D(), scene, sampler, brdfScatteringPdf, 
              brdfLightPdf, std::move(lightTreePdfFunc), false, false);
          if (!lBRDF.IsBlack() && brdfScatteringPdf > 0.0f && brdfLightPdf > 0.0f) {
            records.push_back(LightSamplingRecord(0, lBRDF, brdfLightPdf, 
                  brdfScatteringPdf, 0.0f, SamplingMethod::BRDF));
            pBRDF = std::min(pBRDF + (4.0f / nSamples), 0.99f);
          } else {
            pBRDF = std::max(pBRDF - (2.0f / nSamples), 0.01f);
          }
        } else {
          uBRDF = (uBRDF - pBRDF) / (1.0f - pBRDF);
        }
      }

      for (const auto& record : records) {
        Spectrum ld;
        Float lightPdf = record.lightPdf;
        Float scatteringPdf = record.scatteringPdf;
        if (record.method == SamplingMethod::BRDF) {
          Float weight = PowerHeuristic(nBRDFSamples, scatteringPdf, nBORASSamples, lightPdf);
          ld = record.l * weight / record.scatteringPdf;
          l += ld / nBRDFSamples;
        } else {
          Float weight = PowerHeuristic(nBORASSamples, lightPdf, nBRDFSamples, scatteringPdf);
          ld = record.l * weight / lightPdf;
          l += ld / nBORASSamples;
          _hashMap[index]->ReportVisible(record.lightCutIndex, record.reward * weight);
        }
      }
    }
  } else {
    for (int i = 0; i < nSamples; ++i) {
      int lightCutNodeIndex;
      int lightTreeNodeIndex;
      Float lightCutPdf;
      lightCut->UniformSample(sampler.Get1D(), lightCutNodeIndex, lightTreeNodeIndex, lightCutPdf);
      const LinearLightTreeNode* node = _lightTree->getNodeByIndex(lightTreeNodeIndex);

      Float value = 0.0f;
      Point2f uLight = sampler.Get2D();
      Point2f uScattering = sampler.Get2D();
      Float lightPdf = 0.0f;
      Float scatteringPdf = 0.0f;
      auto estimationFunction = [&](const std::shared_ptr<Light>& light) {
            BxDFType bsdfFlags = BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
            Spectrum Ld(0.f);
            // Sample light source with multiple importance sampling
            Vector3f wi;
            Float scatteringPdf = 0;
            VisibilityTester visibility;
            Spectrum Li = light->Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
            VLOG(2) << "EstimateDirect uLight:" << uLight << " -> Li: " << Li << ", wi: "
                    << wi << ", pdf: " << lightPdf;
            Float vLight = 0.0f;
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
                        vLight = 0.0f;
                      } else
                        VLOG(2) << "  shadow ray unoccluded";
                    }

                    // Add light's contribution to reflected radiance
                    if (!Li.IsBlack()) {
                        if (IsDeltaLight(light->flags)) {
                            Ld += f * Li / lightPdf;
                            vLight = Spectrum(f * Li).y() / lightPdf;
                        }
                        else {
                            Float weight =
                                PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                            Ld += f * Li * weight / lightPdf;
                            vLight = (Spectrum(f * Li).y() * weight / lightPdf);
                        }
                    }
                }
            }

            Float vBRDF = 0.0f;
            // Sample BSDF with multiple importance sampling
            if (!IsDeltaLight(light->flags)) {
                Spectrum f;
                bool sampledSpecular = false;
                if (it.IsSurfaceInteraction()) {
                    // Sample scattered direction for surface interactions
                    BxDFType sampledType;
                    const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
                    f = isect.bsdf->Sample_f(isect.wo, &wi, uScattering, &scatteringPdf,
                                             bsdfFlags, &sampledType);
                    f *= AbsDot(wi, isect.shading.n);
                    sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
                } else {
                    // Sample scattered direction for medium interactions
                    const MediumInteraction &mi = (const MediumInteraction &)it;
                    Float p = mi.phase->Sample_p(mi.wo, &wi, uScattering);
                    f = Spectrum(p);
                    scatteringPdf = p;
                }
                VLOG(2) << "  BSDF / phase sampling f: " << f << ", scatteringPdf: " <<
                    scatteringPdf;
                if (!f.IsBlack() && scatteringPdf > 0) {
                    // Account for light contributions along sampled direction _wi_
                    Float weight = 1;
                    if (!sampledSpecular) {
                        lightPdf = light->Pdf_Li(it, wi);
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
                        if (lightIsect.primitive->GetAreaLight() == light.get())
                            Li = lightIsect.Le(-wi);
                    } else
                        Li = light->Le(ray);
                    if (!Li.IsBlack()) {
                      vBRDF = Spectrum(f * Li * Tr).y() * weight * AbsDot(wi, its.shading.n) / scatteringPdf;
                      // vBRDF /= (lightIsect.p - its.p).LengthSquared();
                      Ld += f * Li * Tr * weight / scatteringPdf;
                    }
                }
            }
            value = vLight + vBRDF;
            return Ld;
      };

      // Float dBar2 = (LinearLightTreeNodeCentroid(_lightTree->getNodeByIndex(lightTreeNodeIndex)) - its.p).
      //   LengthSquared();
      LightTreeSamplingResult result = _lightTree->Sample(wi, its, lightTreeNodeIndex,
          sampler.Get1D(), std::move(weightingFunction), std::move(estimationFunction), std::move(callback));

      if (lightCutPdf > 0.0f && result.second > 0.0f) {
        if (result.first.IsBlack()) {
          _hashMap[index]->ReportOccluded(lightCutNodeIndex);
        } else {
          _hashMap[index]->ReportVisible(lightCutNodeIndex, value / result.second);
        }

        l += result.first / (lightCutPdf * result.second * nSamples);
      }
    }
  }

  return l;
}

LightSampler* CreateVarianceAwareBayesianOnlineRegressionLightSampler(const ParamSet& params) {
  uint32_t shadingPointClusters = params.FindOneInt("shadingpointclusters", 32768);
  uint32_t directionalGridNumber = params.FindOneInt("directionalgridnumber", 8);
  Float noBar = params.FindOneFloat("nobar", 2.0f);
  Float nvBar = params.FindOneFloat("nvbar", 2.0f);
  Float nBar = params.FindOneFloat("nbar", 1.0f);
  Float nAlphaBar = params.FindOneFloat("nalphabar", 1.0f);
  Float beta = params.FindOneFloat("beta", 1e-6);
  Float useMIS = params.FindOneBool("usemis", false);

  return new VarianceAwareBayesianOnlineRegressionLightSampler(shadingPointClusters, directionalGridNumber, 
      noBar, nvBar, nBar, nAlphaBar, beta, useMIS);
}

}
