// Copyright @TwoCookingMice 2020

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTSAMPLERS_VARLMIS_H
#define PBRT_LIGHTSAMPLERS_VARLMIS_H

#include "lightsampler.h"
#include "accelerators/lighttree.h"
#include "paramset.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pbrt {

class VARLMISLightSampler : public LightTreeSampler {
  public:
    VARLMISLightSampler(uint32_t direction, Float lr, Float gamma, 
        uint32_t initLightCutSize, uint32_t maxLightCutSize, 
        bool adaptive = true, uint32_t shadingPointClusters = 32768, bool biased = false):
      _directionDivideNumber(direction), 
      _learningRate(lr), 
      _gamma(gamma), 
      _initLightCutSize(initLightCutSize),
      _maxLightCutSize(maxLightCutSize),
      _isAdaptive(adaptive),
      _shadingPointClusters(shadingPointClusters),
      _biased(biased) {}

    virtual void Preprocess(const Scene& scene, 
        const std::vector<std::shared_ptr<Light>>& lights, bool isVpl = false) override;
    virtual Spectrum Sample(const Vector3f& wi, const Interaction& it, const Scene& scene,
        MemoryArena& arena, Sampler& sampler,
        bool handleMedia = false, uint32_t nSamples = 1) override;

  public:
    enum SamplingMethod { BRDF, VARL };

    struct LightSamplingResult {
      uint32_t lightCutIndex;
      uint32_t lightTreeIndex;
      Spectrum l;
      Float lightPdf;
      Float scatteringPdf;
      Float reward;
      SamplingMethod method;

      LightSamplingResult(uint32_t _lightCutIndex, 
          const Spectrum& _l, 
          Float _lightPdf, 
          Float _scatteringPdf, 
          Float _reward,
          SamplingMethod _method):
        lightCutIndex(_lightCutIndex),
        l(_l), 
        lightPdf(_lightPdf), 
        scatteringPdf(_scatteringPdf), 
        reward(_reward),
        method(_method) {}
    };

    struct HashTableEntry {
      std::shared_ptr<LightCut> lightCut;
      std::shared_ptr<LightTreeSamplingResults> samplingResults;
      uint32_t iteration;
      uint32_t curMaxQValueIndex;
      Float noChangeIterations;

      HashTableEntry(const std::shared_ptr<LightCut> lightCut, uint32_t maxLightCutSize):
        lightCut(lightCut->Clone()), 
        iteration(1),
        noChangeIterations(0),
        samplingResults(new LightTreeSamplingResults(maxLightCutSize)) {}

      uint32_t MemoryCost() const {
        return lightCut->MemoryCost() + samplingResults->MemoryCost() + 2 * sizeof(uint32_t);
      }
    };

    struct SamplingRecord {
      uint32_t lightCutIndex;
      uint32_t lightTreeIndex;
      LightTreeSamplingResult result;
      Float reward;

      SamplingRecord() = delete;
      
      SamplingRecord(uint32_t _lightCutIndex, uint32_t _lightTreeIndex, 
          const LightTreeSamplingResult& _result, Float _reward): lightCutIndex(_lightCutIndex),
          lightTreeIndex(_lightTreeIndex), result(_result), reward(_reward) {}
    };

  private:
    uint32_t SampleAllocation(const LightCut*, uint32_t) const;
    inline Float ShadingPointEstimation(const LinearLightTreeNode* lightTreeNode) const;
    Spectrum EstimateSingleLight(const Interaction &it, const Light &light, 
        const Point2f &uLight, const Scene &scene, Sampler &sampler, 
        MemoryArena &arena, Float& reward, Float& lightPdf, Float& scatteringPdf,
        bool handleMedia = false, bool specular = false) const;

  private:
    uint32_t _directionDivideNumber;
    Float _learningRate;
    Float _gamma;
    uint32_t _maxLightCutSize;
    uint32_t _initLightCutSize;
    bool _isAdaptive;
    uint32_t _shadingPointClusters;

    std::shared_ptr<LightCut> _globalLightCut;
    std::vector<std::shared_ptr<HashTableEntry> > _hashMap;
    std::mutex _hashMapMutex[4];
    Float _spaceGridSize[3];
    Float _directionalGridSize[2];
    Bounds3f _worldBound;
    bool _biased;
};

LightSampler* CreateVARLMISLightSampler(const ParamSet& params);

} // namespace pbrt

#endif
