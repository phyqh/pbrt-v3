// Copyright @TwoCookingMice 2020

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTSAMPLERS_SLC_H
#define PBRT_LIGHTSAMPLERS_SLC_H

#include "lightsampler.h"
#include "accelerators/lighttree.h"
#include "paramset.h"
#include "scene.h"

#include <functional>
#include <memory>

namespace pbrt {

// We implement Prof. Cem Yuksel's Stochastic Lightcut
// in TVCG 2019.
class SLCLightSampler : public LightTreeSampler {
  public:
    SLCLightSampler() {}
    virtual Spectrum Sample(const Vector3f& wi, const Interaction& it, 
        const Scene& scene, MemoryArena& arena, Sampler& sampler,
        bool handleMedia = false, uint32_t nSample = 1) override;
};

LightSampler* CreateSLCLightSampler(const ParamSet& params);

} // namespace pbrt

#endif
