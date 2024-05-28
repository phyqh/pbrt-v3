/*************************************************************************
    > File Name: src/integrators/vpl.h
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Sat Feb  6 18:08:11 2021
 ************************************************************************/

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_VPL_H
#define PBRT_INTEGRATORS_VPL_H

// integrators/vpl.h*
#include "pbrt.h"
#include "integrator.h"
#include "lightdistrib.h"

namespace pbrt {

// VPLIntegrator Delarations
class VPLIntegrator : public SamplerIntegrator {
public:
  VPLIntegrator(int nLightPaths, Float rrThreshold, 
      int maxDepth, int maxSpecularDepth, int lightSamples, 
      std::shared_ptr<const Camera> camera,
      std::shared_ptr<Sampler> sampler, 
      const Bounds2i& pixelBounds): _nLightPaths(nLightPaths),
  _rrThreshold(rrThreshold), _maxDepth(maxDepth), _maxSpecularDepth(maxSpecularDepth),
  _lightSamples(lightSamples), SamplerIntegrator(camera, sampler, pixelBounds) {}

  void Preprocess(const Scene& scene, Sampler& sampler);
  Spectrum Li(const RayDifferential& ray, const Scene& scene,
      Sampler& sampler, MemoryArena& arena, int depth) const;

private:
  const int _nLightPaths;
  const Float _rrThreshold;
  const int _maxDepth;
  const int _maxSpecularDepth;
  const int _lightSamples;
  std::vector<int> _nLightSamples;
};

VPLIntegrator* CreateVPLIntegrator(const ParamSet& params, 
    std::shared_ptr<Sampler> sampler, std::shared_ptr<const Camera> camera);

} // namespace pbrt

#endif
