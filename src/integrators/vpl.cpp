/*************************************************************************
    > File Name: src/integrators/vpl.cpp
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Sat Feb  6 18:23:23 2021
 ************************************************************************/

// integrators/path.cpp*
#include "integrators/vpl.h"

#include "camera.h"
#include "lights/virtualpoint.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"
#include <fstream>

namespace pbrt {

void VPLIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    std::vector<std::shared_ptr<Light>> virtualPointLights;
    virtualPointLights.reserve(_nLightPaths * _maxDepth);

    const int nSamples = 256;
    MemoryArena arena(1024 * 1024);

    std::unique_ptr<Distribution1D> lightSamplingDistribution = ComputeLightPowerDistribution(scene);

    // Open a file to write VPL definitions
    std::ofstream vplFile("vpl_definitions.txt");
    if (!vplFile.is_open()) {
        Error("Failed to open VPL output file.");
        return;
    }

    for (uint32_t i = 0; i < _nLightPaths; ++i) {
        RNG rng(i);
        // Sample a light as the start of light tracing.
        float lightPdf;
        int ln = lightSamplingDistribution->SampleDiscrete(rng.UniformFloat(), &lightPdf);
        const Light* light = scene.lights[ln].get();

        Normal3f nl;
        RayDifferential ray;
        Float pdfDir, pdfPos;
        Spectrum alpha = light->Sample_Le(rng.Uniform2D(), rng.Uniform2D(), 
            camera->shutterOpen, &ray, &nl, &pdfPos, &pdfDir);
        if (pdfPos * pdfDir * lightPdf <= 0.0f || alpha.IsBlack()) continue;
        alpha *= AbsDot(ray.d, nl) / (pdfDir * pdfPos * lightPdf);
        SurfaceInteraction its;
        Float etaScale = 1.0f;
        int bounce = 0;
        uint32_t start = virtualPointLights.size();
        while(!alpha.IsBlack()) {
            bool foundIntersection = scene.Intersect(ray, &its);
            if (!foundIntersection || bounce >= _maxDepth) {
                break;
            }

            Vector3f wo = -ray.d;
            its.ComputeScatteringFunctions(ray, arena, true);
            if (its.bsdf == nullptr) {
                ray = its.SpawnRay(ray.d);
                continue;
            }
            const BSDF* bsdf = its.bsdf;

            Point2f* samples = AllocAligned<Point2f>(nSamples);
            StratifiedSample2D(samples, 16, 16, rng);
            Spectrum contrib = alpha * bsdf->rho(wo, nSamples, samples) * etaScale / M_PI;
            contrib /= _nLightPaths;

            // Write VPL definition to file
            vplFile << "AttributeBegin\n";
            vplFile << "LightSource \"virtualpoint\"\n";
            vplFile << "    \"rgb I\" [ " << contrib[0] << " " << contrib[1] << " " << contrib[2] << " ]\n";
            vplFile << "    \"point3 from\" [ " << its.p.x << " " << its.p.y << " " << its.p.z << " ]\n";
            vplFile << "    \"normal normal\" [ " << its.shading.n.x << " " << its.shading.n.y << " " << its.shading.n.z << " ]\n";
            vplFile << "    \"float radius\" [ 0.001 ]\n";
            vplFile << "    \"float scale\" [ 1 ]\n";
            vplFile << "AttributeEnd\n";

            virtualPointLights.push_back(std::make_shared<VirtualPointLight>(its.p, its.shading.n, 1e-4, 
                its.mediumInterface, contrib / _nLightPaths));

            // Sample a new direction
            Vector3f wi;
            float brdfPdf;
            BxDFType flags;
            Spectrum fr = bsdf->Sample_f(wo, &wi, rng.Uniform2D(), &brdfPdf, BSDF_ALL, &flags);
            if (fr.IsBlack() || brdfPdf <= 0.0f) break;

            if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
                Float eta = bsdf->eta;
                etaScale *= (Dot(wo, its.n) > 0) ? (eta * eta) : 1.0f / (eta * eta);
            }

            Spectrum contribScale = fr * AbsDot(wi, its.shading.n) / brdfPdf;

            // Control the light tracing with Russian roulette.
            Float rrProb = std::min(0.95f, contribScale.y());
            if (rng.UniformFloat() > rrProb) {
                break;
            }
            alpha *= contribScale / rrProb;
            ray = its.SpawnRay(wi);
            ++bounce;
        }

        uint32_t end = virtualPointLights.size();
        for (uint32_t j = start; j < end; ++j) {
            VirtualPointLight* vpl = static_cast<VirtualPointLight*>(virtualPointLights[j].get());
            vpl->I /= (end - start);
        }
    }

    vplFile.close(); // Close the file after writing all VPLs

  // for (const auto& light : scene.lights) {
  //   virtualPointLights.push_back(light);
  // }

  _nLightSamples.reserve(scene.lights.size());
  for (const auto& light : scene.lights) {
    _nLightSamples.push_back(sampler.RoundCount(light->nSamples));
  }

  for (int i = 0; i < _maxDepth; ++i) {
    for (size_t j = 0; j < scene.lights.size(); ++j) {
      sampler.Request2DArray(_nLightSamples[j]);
      sampler.Request2DArray(_nLightSamples[j]);
    }
  }

  CHECK_GT(virtualPointLights.size(), 0);
  scene.PreprocessWithVPL(virtualPointLights);
}

Spectrum VPLIntegrator::Li(const RayDifferential& r, const Scene& scene,
    Sampler& sampler, MemoryArena& arena, int depth) const {
  ProfilePhase _(Prof::SamplerIntegratorLi);

  Spectrum L(0.0f);
  SurfaceInteraction its;
  if (!scene.Intersect(r, &its)) {
    for (const auto& light : scene.lights) {
      L += light->Le(r);
    }
    return L;
  }

  its.ComputeScatteringFunctions(r, arena);
  if (!its.bsdf) {
    return Li(its.SpawnRay(r.d), scene, sampler, arena, depth);
  }
  Vector3f wo = its.wo;
  L += its.Le(its.wo);
  L += UniformSampleAllLights(its, scene, arena, sampler, _nLightSamples);
  L += scene.SampleLights(r.d, its, arena, sampler, false, _lightSamples);

  // // Specular Computation
  // if (depth < 2) {
  //   int nSamples = (depth == 0) ? 8 : 1;
  //   for (int j = 0; j < nSamples; ++j) {
  //     Vector3f wi;
  //     Float brdfPdf;
  //     Spectrum f = its.bsdf->Sample_f(its.wo, &wi, sampler.Get2D(), &brdfPdf, 
  //         BxDFType(BSDF_ALL & ~BSDF_SPECULAR));
  //     if (!f.IsBlack() && brdfPdf > 0.0f) {
  //       Float maxDist = std::sqrt(AbsDot(wi, its.shading.n) / 16.0f);
  //       RayDifferential gatherRay(its.p, wi, maxDist);
  //       Spectrum li = this->Li(gatherRay, scene, sampler, arena, depth + 1);

  //       if (li.IsBlack()) continue;

  //       L += f * li * AbsDot(wi, its.shading.n) / (nSamples * brdfPdf);
  //     }
  //   }
  // }

  if (depth + 1 < _maxSpecularDepth) {
    L += SpecularReflect(r, its, scene, sampler, arena, depth + 1);
    L += SpecularTransmit(r, its, scene, sampler, arena, depth + 1);
  }

  return L;
}

VPLIntegrator* CreateVPLIntegrator(const ParamSet& params, 
    std::shared_ptr<Sampler> sampler, 
    std::shared_ptr<const Camera> camera) {
  int nLightPaths = params.FindOneInt("nlightpaths", 10000);
  Float rrThreshold = params.FindOneFloat("rrThreshold", 0.01);
  int maxDepth = params.FindOneInt("maxdepth", 8);
  int maxSpecularDepth = params.FindOneInt("maxspeculardepth", 8);
  int lightSamples = params.FindOneInt("lightsamples", 1);
  int np;
  const int *pb = params.FindInt("pixelbounds", &np);
  Bounds2i pixelBounds = camera->film->GetSampleBounds();
  if (pb) {
    if (np != 4)
      Error("Expected four values for \"pixelbounds\" parameter. Got %d.", np);
    else {
      pixelBounds = Intersect(pixelBounds,
          Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
      if (pixelBounds.Area() == 0)
        Error("Degenerate \"pixelbounds\" specified.");
    }
  }

  return new VPLIntegrator(nLightPaths, rrThreshold,
      maxDepth, maxSpecularDepth, lightSamples, camera, sampler, pixelBounds);
}

} // namespace vpl
