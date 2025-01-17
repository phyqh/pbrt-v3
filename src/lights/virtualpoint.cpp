/*************************************************************************
    > File Name: src/lights/virtualpoint.cpp
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Thursday, February 04, 2021 PM04:57:26 HKT
 ************************************************************************/

#include "lights/virtualpoint.h"

#include "sampling.h"
#include "stats.h"
#include "paramset.h"
#include "scene.h"

namespace pbrt {
  
const Float gLimit = 32.0f;

Spectrum VirtualPointLight::Sample_Li(const Interaction& ref, const Point2f &u, 
    Vector3f *wi, Float *pdf, VisibilityTester *vis) const {
  ProfilePhase _(Prof::LightSample);

  if (!ref.IsSurfaceInteraction()) 
    return Spectrum(0.0f);
  const SurfaceInteraction& its = static_cast<const SurfaceInteraction&>(ref);

  *wi = Normalize(_p - ref.p);
  *pdf = 1.0f;
  *vis = VisibilityTester(ref, Interaction(_p, ref.time,
        mediumInterface));

  Float gGather = AbsDot(-(*wi), _n) / DistanceSquared(its.p, _p);
  gGather = std::min(gGather, gLimit);

  return I * gGather;
}

Spectrum VirtualPointLight::Power() const {
  return 4.0f * Pi * I;
}

Float VirtualPointLight::Pdf_Li(const Interaction& ref, const Vector3f &wi) const {
  ProfilePhase _(Prof::LightPdf);
  return 0.0f;
}

Spectrum VirtualPointLight::Sample_Le(const Point2f &u1, const Point2f &u2, Float time, 
    Ray *ray, Normal3f *nLight, Float *pdfPos, Float *pdfDir) const {
  ProfilePhase _(Prof::LightSample);
  
  Vector3f outDir = CosineSampleHemisphere(u1);
  *pdfPos = 1.0f;
  *pdfDir = CosineHemispherePdf(std::abs(outDir.z));
  Vector3f v1, v2, n;
  CoordinateSystem(n, &v1, &v2);
  outDir = outDir.x * v1 + outDir.y * v2 + outDir.z * n;
  *nLight = _n;
  *ray = Ray(_p, outDir, Infinity, time, mediumInterface.inside);

  return I;
}

void VirtualPointLight::Pdf_Le(const Ray &ray, const Normal3f &n, Float *pdfPos, Float *pdfDir) const {
  ProfilePhase _(Prof::LightPdf);
  *pdfPos = 0.0f;
  *pdfDir = CosineHemispherePdf(Dot(ray.d, _n));
}

Bounds3f VirtualPointLight::WorldBound() const {
  return Bounds3f(_p, _p);
}

bool VirtualPointLight::GetOrientationAttributes(Vector3f &axis, Float &thetaO, Float &thetaE) const {
  axis = Vector3f(_n);
  thetaO = Pi;
  thetaE = PiOver2;
  return true;
}

std::shared_ptr<VirtualPointLight> CreateVirtualPointLight(
    const Transform &light2world,
    const Medium *medium,
    const ParamSet &paramSet) {
    // Get the intensity parameter (I)
    Spectrum I = paramSet.FindOneSpectrum("I", Spectrum(1.0));
    
    // Get the scaling factor
    Spectrum sc = paramSet.FindOneSpectrum("scale", Spectrum(1.0));
    I *= sc;
    
    // Get the position and transform it
    Point3f P = paramSet.FindOnePoint3f("from", Point3f(0, 0, 0));
    Point3f pLight = light2world(P);
    
    // Get the normal direction (defaulting to up vector if not specified)
    Normal3f N = paramSet.FindOneNormal3f("normal", Normal3f(0, 1, 0));
    Normal3f nLight = Normalize(light2world(N));
    
    // Get the radius epsilon parameter (defaulting to 0)
    Float reps = paramSet.FindOneFloat("radius", 0.0f);
    
    // Create and return the VirtualPointLight with the correct constructor signature
    return std::make_shared<VirtualPointLight>(pLight, nLight, reps,
        MediumInterface(medium), I);
}

} // namespace pbrt
