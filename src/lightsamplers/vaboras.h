/*************************************************************************
    > File Name: src/lightsamplers/boras.h
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Sunday, February 21, 2021 PM04:27:05 HKT
 ************************************************************************/

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTSAMPLERS_VABORAS_H
#define PBRT_LIGHTSAMPLERS_VABORAS_H

#include "lightsampler.h"
#include "accelerators/lighttree.h"
#include "paramset.h"

#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace pbrt {

// Implementation of paper "Bayesian online regression for adaptive direct illumination sampling"
// from SIGGRAPH 2018 and Variance-Aware Path Guiding in SIGGRAPH 2020.
class VarianceAwareBayesianOnlineRegressionLightSampler : public LightTreeSampler {
public:
  VarianceAwareBayesianOnlineRegressionLightSampler(uint32_t shadingPointClusters, uint32_t directionalGridNumber, 
      Float nOBar, Float nVBar, Float nBar, Float nAlphaBar, Float beta, bool useMIS):
    _shadingpointclusters(shadingPointClusters), _directionalGridNumber(directionalGridNumber), _nOBar(nOBar), 
    _nVBar(nVBar), _nBar(nBar), _nAlphaBar(nAlphaBar), _beta(beta), _useMIS(useMIS) {}
  virtual void Preprocess(const Scene& scene, 
      const std::vector<std::shared_ptr<Light>>& lights, bool isVpl = false) override;
  virtual Spectrum Sample(const Vector3f& wi, const Interaction& it, const Scene& scene,
      MemoryArena& arena, Sampler& sampler, bool handleMedia = false, uint32_t nSamples = 1) override;

public:

  enum SamplingMethod { BRDF, BORAS };

  struct LightSamplingRecord {
    uint32_t lightCutIndex;
    Spectrum l;
    Float lightPdf;
    Float scatteringPdf;
    Float reward;
    SamplingMethod method;

    LightSamplingRecord(uint32_t _lightCutIndex, 
        const Spectrum& _l, 
        Float _lightPdf, 
        Float _scatteringPdf, 
        Float _reward,
        SamplingMethod _method):
          lightCutIndex(_lightCutIndex) ,
          l(_l), 
          lightPdf(_lightPdf), 
          scatteringPdf(_scatteringPdf), 
          reward(_reward),
          method(_method) {}
  };

  struct HashTableEntry {
    std::shared_ptr<LightCut> lightCut;
    std::vector<int> nv;
    std::vector<int> no;
    std::vector<Float> s1x;
    std::vector<Float> s2x;
    std::map<uint32_t, uint32_t> lightSamplingCache;
    std::mutex mutex;

    HashTableEntry(std::shared_ptr<LightCut> _lightCut): 
      lightCut(_lightCut) {
        nv = std::vector<int>(_lightCut->Size(), 0);
        no = std::vector<int>(_lightCut->Size(), 0);
        s1x = std::vector<Float>(_lightCut->Size(), 0.0f);
        s2x = std::vector<Float>(_lightCut->Size(), 0.0f);
    }

    void ReportOccluded(int lightCutIndex) {
      CHECK_LT(lightCutIndex, lightCut->Size());
      std::lock_guard<std::mutex> _(mutex);
      ++no[lightCutIndex];
    }

    void ReportVisible(int lightCutIndex, Float value) {
      CHECK_LT(lightCutIndex, lightCut->Size());
      std::lock_guard<std::mutex> _(mutex);
      Float value2 = value * value;
      ++nv[lightCutIndex];
      s1x[lightCutIndex] = value / nv[lightCutIndex] + 
        s1x[lightCutIndex] * (nv[lightCutIndex] - 1) / nv[lightCutIndex];
      s2x[lightCutIndex] = value2 / nv[lightCutIndex] + 
        s2x[lightCutIndex] * (nv[lightCutIndex] - 1) / nv[lightCutIndex];
    }
  };

private:
  uint32_t _shadingpointclusters;
  uint32_t _directionalGridNumber;
  Float _nOBar;
  Float _nVBar;
  Float _nBar;
  Float _nAlphaBar;
  Float _beta;
  std::vector<std::shared_ptr<HashTableEntry> > _hashMap;

  std::mutex _hashMapMutex[4];
  Float _directionalGridSize[2];
  Bounds3f _worldBound;
  bool _useMIS;
};

LightSampler* CreateVarianceAwareBayesianOnlineRegressionLightSampler(const ParamSet& params);

}

#endif
