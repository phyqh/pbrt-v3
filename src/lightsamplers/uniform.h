// Copyright @TwoCookingMice 2020

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTSAMPLERS_UNIFORM_H
#define PBRT_LIGHTSAMPLERS_UNIFORM_H

#include "lightsampler.h"
#include "lightdistrib.h"
#include "paramset.h"

#include <memory>
#include <string>

namespace pbrt {

class UniformLightSampler : public LightSampler {
  public:
    UniformLightSampler(const std::string sampleStrategy):
      _sampleStrategy(sampleStrategy) {}
    void Preprocess(const Scene& scene, const std::vector<std::shared_ptr<Light>>& lights, 
        bool isVpl = false) override;
    virtual Spectrum Sample(const Vector3f& wi, const Interaction& it, const Scene& scene,
        MemoryArena& arena, Sampler& sample,
        bool handleMedia = false, uint32_t nSamples = 1) override;

  private:
    std::string _sampleStrategy;
    std::unique_ptr<LightDistribution> _lightDistribution;
    std::vector<std::shared_ptr<Light>> _lights;
};

LightSampler *CreateUniformLightSampler(const ParamSet& params);

} // namespace pbrt

#endif
