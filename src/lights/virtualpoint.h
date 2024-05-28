/*************************************************************************
    > File Name: src/lights/virtualpoint.h
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Tuesday, February 02, 2021 PM03:50:47 HKT
 ************************************************************************/

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTS_VIRTUAL_POINT_H
#define PBRT_LIGHTS_VIRTUAL_POINT_H

#include "pbrt.h"
#include "light.h"
#include "shape.h"

namespace pbrt {

// Virtual Point light declarations, please not
// expose it to the users.
class VirtualPointLight : public Light {
public:
  VirtualPointLight(const Point3f& p, const Normal3f& n, Float reps,
      const MediumInterface& mediumInterface, const Spectrum& I)
    : Light((int)LightFlags::DeltaPosition, Transform(), mediumInterface),
    _p(p), _n(n), _reps(reps), I(I) {}

  virtual Spectrum Sample_Li(const Interaction& ref, const Point2f& u, Vector3f* wi,
      Float *pdf, VisibilityTester* vis) const override;
  virtual Spectrum Power() const override;
  virtual Float Pdf_Li(const Interaction& ref, const Vector3f& wi) const override;
  virtual Spectrum Sample_Le(const Point2f& u1, const Point2f& u2, Float time,
      Ray* ray, Normal3f* nLight, Float* pdfPos, Float* pdfDir) const override;
  void Pdf_Le(const Ray& ray, const Normal3f& n, Float* pdfPos, Float* pdfDir) const override;

  virtual Bounds3f WorldBound() const override;
  virtual bool GetOrientationAttributes(Vector3f& axis, Float& thetaO, Float& thetaE) const override;

public:
  Spectrum I;

private:
  const Point3f _p;
  const Normal3f _n;
  const Float _reps;
};

} // namespace pbrt

#endif // PBRT_LIGHTS_VIRTUAL_POINT_H
