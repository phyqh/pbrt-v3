/*************************************************************************
    > File Name: src/accelerators/lighttree.h
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Wed May 20 15:14:36 2020
 ************************************************************************/

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_ACCELERATORS_LIGHTTREE_H_
#define PBRT_ACCELERATORS_LIGHTTREE_H_

#include "light.h"
#include "pbrt.h"
#include "sampling.h"

#include <functional>
#include <memory>
#include <mutex>
#include <utility>

namespace pbrt {

struct LightInfo;
struct LightTreeBuildNode;
class LightCone;

struct LightCone {
  LightCone();
  LightCone(const Vector3f&, Float, Float);

  Float measure() const;

  Vector3f axis() const {
    return _axis;
  }

  Float thetaO() const {
    return _thetaO;
  }

  Float thetaE() const {
    return _thetaE;
  }

  Float thetaW() const {
    return std::min(_thetaO + _thetaE,  Pi);
  }

  static LightCone Union(const LightCone& lc1, const LightCone& lc2);

  Vector3f _axis;
  Float _thetaO;
  Float _thetaE;
};

struct LinearLightTreeNode {
  int splitAxis;
  Float power;
  Float power2;
  int parent;
  int lightsOffset; // For leaf nodes
  int secondChildOffset; // For internal nodes
  int nLight;
  Bounds3f bound;
  Vector3f axis;
  Float thetaO;
  Float thetaE;
  uint32_t nodeID;
};

Float LinearLightTreeNodeGeoTermBound(const LinearLightTreeNode* node, const Point3f& p,
    const Vector3f& normal);

inline Float LinearLightTreeNodeSurfaceArea(const LinearLightTreeNode* node) {
  CHECK_NOTNULL(node);
  return node->bound.SurfaceArea();
}

inline bool LinearLightTreeNodeIsLeaf(const LinearLightTreeNode* node) {
  CHECK_NOTNULL(node);
  return node->splitAxis < 0;
}

inline Float LinearLightTreeNodePowerVariance(const LinearLightTreeNode* node) {
  CHECK_NOTNULL(node);
  return node->power2 / node->nLight - powf(node->power / node->nLight, 2.0f);
}

inline Point3f LinearLightTreeNodeCentroid(const LinearLightTreeNode* node) {
  CHECK_NOTNULL(node);
  return 0.5f * (node->bound.pMin + node->bound.pMax);
}

inline Vector3f LinearLightTreeNodeDiagonal(const LinearLightTreeNode* node) {
  CHECK_NOTNULL(node);
  return node->bound.Diagonal();
}

struct LightCutNode {
  int lightTreeNodeIndex;
  Float weight;
  Float bsdfVal;
  Float geometryTerm;

  LightCutNode(int _lightTreeNodeIndex, Float _weight):
    lightTreeNodeIndex(_lightTreeNodeIndex),
    weight(_weight) {}
};

// Definition of LightCut
struct LightCut {
  std::vector<LightCutNode> _cut;
  std::unique_ptr<Distribution1D> _distribution;
  bool _sizeChanged;
  std::mutex _mutex;

  LightCut():
    _distribution(nullptr),
    _sizeChanged(false) {}

  LightCut(uint32_t size):
    _distribution(nullptr),
    _sizeChanged(false) {
      _cut.reserve(size);
    }

  void Append(const LightCutNode& node) {
    std::lock_guard<std::mutex> _(_mutex);
    _cut.push_back(node);
    _sizeChanged = true;
  }

  void Erase(uint32_t index) {
    CHECK_GT(_cut.size(), index);
    std::lock_guard<std::mutex> _(_mutex);
    _cut.erase(_cut.begin() + index);
    _sizeChanged = true;
  }

  Float MaxWeight() const {
    Float res = -1.0;
    for (const auto& node : _cut) {
      res = std::max(res, node.weight);
    }

    return res;
  }

  Float MeanWeight() const {
    Float res = 0.0f;
    for (uint32_t i = 0; i < Size(); ++i) {
      res += _cut[i].weight / Size();
    }

    return res;
  }

  void Update(uint32_t lightCutIndex, uint32_t lightTreeIndex, 
      std::function<void(LightCutNode*)>&& F) {
    std::lock_guard<std::mutex> _(_mutex);
    if (lightCutIndex < Size() || 
        lightTreeIndex == _cut[lightCutIndex].lightTreeNodeIndex) {
      F(&_cut[lightCutIndex]);
    }
  }

  std::shared_ptr<LightCut> Clone() const {
    uint32_t lightCutSize = Size();
    LightCut* clonedCut = new LightCut(lightCutSize);
    for (uint32_t i = 0; i < lightCutSize; ++i) {
      clonedCut->Append(LightCutNode(_cut[i].lightTreeNodeIndex, _cut[i].weight));
    }
    clonedCut->UpdateDistribution();

    return std::shared_ptr<LightCut>(clonedCut);
  }

  bool Verify(uint32_t lightCutIndex, uint32_t lightTreeIndex) const {
    return _cut[lightCutIndex].lightTreeNodeIndex == lightTreeIndex;
  }

  uint32_t Size() const {
    return _cut.size();
  }

  uint32_t MemoryCost() const {
    return _cut.size() * sizeof(LightCutNode);
  }

  void UpdateDistribution() {
    std::lock_guard<std::mutex> _(_mutex);
    if (_sizeChanged) {
      uint32_t lightCutSize = Size();
      std::vector<Float> weights(lightCutSize);
      for (uint32_t i = 0; i < lightCutSize; ++i) {
        weights[i] = _cut[i].weight;
      }
      _distribution = std::make_unique<Distribution1D>(&weights[0], _cut.size());
      _sizeChanged = false;
    } else {
      uint32_t lightCutSize = _cut.size();
      for (uint32_t i = 0; i < lightCutSize; ++i) {
        _distribution->Update(i, _cut[i].weight);
      }
    }
    _distribution->Normalize();
  }

  // Uniform sample a node in light tree from
  // light cut.
  void UniformSample(Float sample, int& lightNodeIndex, Float& pdf, 
      Float* remapped = nullptr) const {
    int index = _distribution->SampleDiscrete(sample, &pdf, remapped);
    lightNodeIndex = _cut[index].lightTreeNodeIndex;
  }

  void UniformSample(Float sample, int& cutNodeIndex, int& lightNodeIndex, Float& pdf,
      Float* remapped = nullptr) const {
    int index = _distribution->SampleDiscrete(sample, &pdf, remapped);
    cutNodeIndex = index;
    lightNodeIndex = _cut[index].lightTreeNodeIndex;
  }

  Float Pdf(int lightCutNodeIndex) const {
    return _distribution->DiscretePDF(lightCutNodeIndex);
  }
};

using LightTreeSamplingResult = std::pair<Spectrum, Float>;
struct LightTreeSamplingResults {
  std::vector<Float> E;
  std::vector<Float> E2;
  std::vector<Float> m2c;
  std::vector<uint32_t> sampleCount;
  std::mutex mutex;

  LightTreeSamplingResults(uint32_t size):
    E(size, 0.0f),
    E2(size, 0.0f),
    m2c(size, 0.0f),
    sampleCount(size, 0) {}

  void Append(uint32_t index, Float val, Float val2) {
    std::lock_guard<std::mutex> _(mutex);
    CHECK_GT(E.size(), index);
    uint32_t t = sampleCount[index];
    Float delta = val - E[index];
    E[index] += delta / (t + 1);
    Float delta2 = val - E[index];
    m2c[index] += delta * delta2;
    E2[index] = t / (t + 1) * (E2[index] - val2) + val2; 
    ++sampleCount[index];
  }

  void Append(uint32_t index, const LightTreeSamplingResult& result) {
    std::lock_guard<std::mutex> _(mutex);
    CHECK_GT(E.size(), index);
    uint32_t t = sampleCount[index];
    Float val = result.first.y() / result.second;
    Float delta = val - E[index];
    Float val2 = val * result.first.y();
    E[index] += delta / (t + 1);
    Float delta2 = val - E[index];
    m2c[index] += delta * delta2;
    E2[index] = t / (t + 1) * (E2[index] - val2) + val2; 
    ++sampleCount[index];
  }

  Float Exp(uint32_t index) const {
    return E[index];
  }

  Float Exp2(uint32_t index) const {
    return E2[index];
  }

  uint32_t MemoryCost() const {
    uint32_t size = E.size();
    return size * (2 * sizeof(Float) + sizeof(uint32_t));
  }

  Float Var(uint32_t index) const {
    return m2c[index] / (sampleCount[index] + 1);
  }

  uint32_t SampleCount(uint32_t index) const {
    return sampleCount[index];
  }

  void Reset(uint32_t index) {
    std::lock_guard<std::mutex> _(mutex);
    E[index] = 0.0f;
    E2[index] = 0.0f;
    m2c[index] = 0.0f;
    sampleCount[index] = 0;
  }
};

class LightTree {
  public:
    enum class SplitMethod {
      Middle,
      EqualCounts,
      SAH
    };

  public:
    LightTree(const std::vector<std::shared_ptr<Light>>& lights,
        int maxLightsPerNode = 1,
        SplitMethod splitMethod = SplitMethod::SAH);
    ~LightTree();

    void sample(const Vector3f& wi, const SurfaceInteraction& its, 
        int curOffset, Float u, int& lightIndex, Float& pdf,
        std::vector<Float>& weights) const;
    void sample(const Vector3f& wi, const Interaction& its,
        int curOffset, Float u, int& lightIndex, Float& pdf,
        std::vector<Float>& weights, 
        std::function<Float(const Vector3f&, const Interaction&, const LinearLightTreeNode&)> F) const;

    LightTreeSamplingResult Sample(const Vector3f& wi, const Interaction& its,
        int curOffset, Float u,
        std::function<Float(const Vector3f&, const Interaction&, const LinearLightTreeNode&)>&& weightFunction,
        std::function<Spectrum(const std::shared_ptr<Light>&)>&& estimationFunction,
        std::function<void(uint32_t, const LightTreeSamplingResult&)>&& callback) const;

    Float Pdf(uint32_t startIndex, const Vector3f& wi, const Interaction& its, const Light* light,
        std::function<Float(
          const Vector3f&, const Interaction&, const LinearLightTreeNode&)>&& weightFunction) const;

    std::shared_ptr<Light> getLightByIndex(int index) const;
    const LinearLightTreeNode* getNodeByIndex(int index) const;

    std::shared_ptr<LightCut> generateLightcut(Float threshold, int maxSize, 
        std::function<Float(const LinearLightTreeNode&)>&& weightFunction) const;

    int TotalNodes() const {
      return _totalNodes;
    }

  private:
    LightTreeBuildNode* recursiveBuild(MemoryArena& arena, std::vector<LightInfo>&,
        int start, int end, int* totalNodes, 
        std::vector<std::shared_ptr<Light>>&);

    int flattenTree(LightTreeBuildNode* node, int& offset, 
        std::vector<std::shared_ptr<Light>>&, int parentID);

    int sampleOneLight(const LinearLightTreeNode* node, Float u, Float* pdf = nullptr) const;

  private:
    std::vector<std::shared_ptr<Light>> _lights;
    std::vector<Float> _lightDistrib;
    const int _maxLightsPerNode;
    const SplitMethod _splitMethod;
    LinearLightTreeNode* _nodes;
    int _totalNodes;
};

std::shared_ptr<LightTree> CreateLightTree(
    std::vector<std::shared_ptr<Light>> lights,
    const ParamSet& params);

} // namespace pbrt

#endif // PBRT_ACCELERATORS_LIGHTTREE_H_
