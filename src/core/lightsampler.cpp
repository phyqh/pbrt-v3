// Copyright @TwoCookingMice 2020

#include "lightsampler.h"
#include "light.h"
#include "reflection.h"
#include "spectrum.h"
#include "stats.h"

namespace pbrt {

STAT_COUNTER("Lightsampler/Total Light Tree Nodes", totalLightTreeNodes);

LightSampler::~LightSampler() {}

void LightTreeSampler::Preprocess(const Scene &scene,
    const std::vector<std::shared_ptr<Light>>& lights, bool isVpl) {
  LightSampler::Preprocess(scene, lights, isVpl);
  if (!isVpl) {
    _lightTree = std::make_shared<LightTree>(lights, 1, LightTree::SplitMethod::SAH);
  } else {
    _lightTree = std::make_shared<LightTree>(lights, 16, LightTree::SplitMethod::SAH);
  }
  totalLightTreeNodes = _lightTree->TotalNodes();
}

Spectrum LightSampler::EstimateDirect(const Interaction &it, const Point2f &uScattering,
                        const Light &light, const Point2f &uLight,
                        const Scene &scene, Sampler &sampler,
                        MemoryArena &arena, Float& lightPdf,
                        bool handleMedia, bool specular) const {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    Float scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
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
              } else
                VLOG(2) << "  shadow ray unoccluded";
            }

            // Add light's contribution to reflected radiance
            if (!Li.IsBlack()) {
                if (IsDeltaLight(light.flags))
                    Ld += f * Li / lightPdf;
                else {
                    Float weight =
                        PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                    Ld += f * Li * weight / lightPdf;
                }
            }
        }
    }

    // Sample BSDF with multiple importance sampling
    if (!IsDeltaLight(light.flags)) {
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
                lightPdf = light.Pdf_Li(it, wi);
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
                if (lightIsect.primitive->GetAreaLight() == &light)
                    Li = lightIsect.Le(-wi);
            } else
                Li = light.Le(ray);
            if (!Li.IsBlack()) Ld += f * Li * Tr * weight / scatteringPdf;
        }
    }
    return Ld;
}

Spectrum LightSampler::EstimateDirectIllumination(const Interaction& it, const Point2f& uShading,
    const Light& light, const Point2f& uLight,
    const Scene& scene, Sampler& sampler,
    MemoryArena& arena, Float& lightPdf,
    bool handleMedia, bool specular) const {
    BxDFType bsdfFlags =
        specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    // Sample light source with multiple importance sampling
    Vector3f wi;
    lightPdf = 0.0f;
    Float scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light.Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
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
              } else
                VLOG(2) << "  shadow ray unoccluded";
            }

            // Add light's contribution to reflected radiance
            if (!Li.IsBlack()) {
              Ld += f * Li / lightPdf;
            }
        }
    }

    return Ld;
}

Spectrum LightSampler::NaiveBRDFSampling(const Interaction& it, const Point2f& uScattering, const Scene& scene,
    Sampler& sampler, Float& scatteringPdf, Float& lightPdf, 
    std::function<Float(const Interaction&, const Vector3f&, const Light*)>&& lightSelectionPdf,
    bool handleMedia, bool specular) const {
  BxDFType bsdfFlags = BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
  Spectrum l(0.0f);

  Vector3f wi;
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
      // Find intersection and compute transmittance
      SurfaceInteraction lightIsect;
      Ray ray = it.SpawnRay(wi);
      Spectrum Tr(1.f);
      bool foundSurfaceInteraction = handleMedia ? scene.IntersectTr(ray, sampler, &lightIsect, &Tr) :
        scene.Intersect(ray, &lightIsect);

      if (foundSurfaceInteraction) {
        const Light* light = lightIsect.primitive->GetAreaLight();
        if (light != nullptr) {
          l += lightIsect.Le(-wi) * f * Tr;
          lightPdf = lightSelectionPdf(it, wi, light) * light->Pdf_Li(it, wi);
        }
      }
  }

  return l;
}

} // namespace pbrt
