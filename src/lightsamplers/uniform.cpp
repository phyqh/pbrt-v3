// Copyright @TwoCookingMice 2020

// lightsamplers/uniform.cpp*
#include "lightsamplers/uniform.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"

namespace pbrt {

void UniformLightSampler::Preprocess(const Scene& scene, 
    const std::vector<std::shared_ptr<Light>>& lights, bool isVpl) {
  LightSampler::Preprocess(scene, lights, isVpl);
  _lightDistribution = CreateLightSampleDistribution(_sampleStrategy, scene, lights);
  _lights.assign(lights.begin(), lights.end());
}

Spectrum UniformLightSampler::Sample(const Vector3f& wi, const Interaction& it, 
    const Scene& scene, MemoryArena& arena, Sampler& sampler,
    bool handleMedia, uint32_t nSamples) {
  ProfilePhase p(Prof::DirectLighting);
  const Distribution1D* lightDistrib = _lightDistribution->Lookup(it.p);

  Spectrum L(0.0f);

  int nLights = int(_lights.size());
  if (nLights == 0) return Spectrum(0.f);
  for (uint32_t i = 0; i < nSamples; ++i) {
    // Randomly choose a single light to sample, _light_
    int lightNum;
    Float lightPdf;
    if (lightDistrib) {
        lightNum = lightDistrib->SampleDiscrete(sampler.Get1D(), &lightPdf);
        if (lightPdf == 0) return Spectrum(0.f);
    } else {
        lightNum = std::min((int)(sampler.Get1D() * nLights), nLights - 1);
        lightPdf = Float(1) / nLights;
    }

    const std::shared_ptr<Light> &light = _lights[lightNum];
    Point2f uLight = sampler.Get2D();
    Point2f uScattering = sampler.Get2D();
    Float lightSamplingPdf;
    CHECK_GE(lightPdf, 0.0f);
    L += EstimateDirect(it, uScattering, *light, uLight,
        scene, sampler, arena, lightSamplingPdf, handleMedia) / lightPdf;
  }

  return L / nSamples;
}

LightSampler* CreateUniformLightSampler(const ParamSet& params) {
  std::string lightSampleStrategy = params.FindOneString("lightsamplestrategy", "spatial");
  return new UniformLightSampler(lightSampleStrategy);
}

} // namespace pbrt
