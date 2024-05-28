// Copyright @TwoCookingMice 2020

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_LIGHTSAMPLER_H
#define PBRT_CORE_LIGHTSAMPLER_H

#include "accelerators/lighttree.h"
#include "pbrt.h"
#include "primitive.h"
#include "sampler.h"
#include "scene.h"

namespace pbrt {

// LightSampler Declarations
class LightSampler {
public:
  virtual ~LightSampler();
  virtual void Preprocess(const Scene& scene, const std::vector<std::shared_ptr<Light>>& lights, 
      bool isVpl = false) {
    _isVpl = isVpl;
  }
  virtual Spectrum Sample(const Vector3f& wi, const Interaction& it, 
      const Scene& scene, MemoryArena& arena, Sampler& sampler,
      bool handleMedia = false, uint32_t nSamples = 1) = 0;

protected:
  bool _isVpl;

protected:
  Spectrum EstimateDirect(const Interaction &it, const Point2f &uShading,
                          const Light &light, const Point2f &uLight,
                          const Scene &scene, Sampler &sampler,
                          MemoryArena &arena, Float& lightPdf,
                          bool handleMedia = false, bool specular = false) const;

  Spectrum EstimateDirectIllumination(const Interaction &it, const Point2f &uShading,
                          const Light &light, const Point2f &uLight,
                          const Scene &scene, Sampler &sampler,
                          MemoryArena &arena, Float& lightPdf,
                          bool handleMedia = false, bool specular = false) const;

  Spectrum NaiveBRDFSampling(const Interaction& it, const Point2f& uScattering, const Scene& scene,
      Sampler& sampler, Float& scatteringPdf, Float& lightPdf, 
      std::function<Float(const Interaction&, const Vector3f&, const Light*)>&&,
      bool handleMedia = false, bool specular = false) const;
};

class LightTreeSampler : public LightSampler {
public:
  LightTreeSampler(): _lightTree(nullptr) {}
  virtual void Preprocess(const Scene& scene, const std::vector<std::shared_ptr<Light>>& lights, 
      bool isVpl = false);

protected:
  std::shared_ptr<LightTree> _lightTree;
};

}

#endif
