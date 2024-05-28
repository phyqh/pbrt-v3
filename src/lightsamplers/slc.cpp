// Copyright @TwoCookingMice 2020

// lightsamplers/slc.cpp*
#include "lightsamplers/slc.h"
#include "stats.h"

namespace pbrt {

Spectrum SLCLightSampler::Sample(const Vector3f& wi, const Interaction& it, 
    const Scene& scene, MemoryArena& arena, Sampler& sampler,
    bool handleMedia, uint32_t nSamples) {
    ProfilePhase p(Prof::DirectLighting);

    std::vector<Float> weights(_lightTree->TotalNodes(), -1.0f);
    Spectrum L(0.0f);

    auto weightingFunction = [&](const Vector3f& wi, 
        const Interaction& its, 
        const LinearLightTreeNode& n) {
      if (!its.IsSurfaceInteraction()) {
        return n.power;
      } else {
        const SurfaceInteraction& iit = (const SurfaceInteraction&) its;
        Vector3f wo = Normalize(LinearLightTreeNodeCentroid(&n) - its.p);
        Vector3f p(iit.p[0], iit.p[1], iit.p[2]);
        Vector3f normal(iit.shading.n[0], iit.shading.n[1], iit.shading.n[2]);
        Float G = LinearLightTreeNodeGeoTermBound(&n, its.p, Vector3f(normal));

        return n.power * G;
      }
    };

    for (uint32_t i = 0; i < nSamples; ++i) {
      int index;
      Float lightPdf;
      _lightTree->sample(wi, it, 0, sampler.Get1D(), index, lightPdf, weights, weightingFunction);

      if (lightPdf <= 0.0f) {
        continue;
      
      }
      std::shared_ptr<Light> light = _lightTree->getLightByIndex(index);
      Point2f uLight = sampler.Get2D();
      Point2f uScattering = sampler.Get2D();
      Float lightSamplingPdf;

      L += EstimateDirect(it, uScattering, *light, uLight,
                        scene, sampler, arena, lightSamplingPdf, handleMedia) / lightPdf;
    }

    return L / nSamples;
}

LightSampler* CreateSLCLightSampler(const ParamSet& params) {
  return new SLCLightSampler();
}

} // namespace pbrt
