/*************************************************************************
    > File Name: src/accelerators/lighttree.cpp
    > Author: Yuchen Wang
    > Mail: wyc8094@gmail.com 
    > Created Time: Wed May 20 15:32:36 2020
 ************************************************************************/

// accelerators/lighttree.cpp*
#include "accelerators/lighttree.h"

#include "materials/ltc.h"
#include "paramset.h"
#include "reflection.h"
#include "sampler.h"
#include "sampling.h"
#include "stats.h"

#include <cmath>
#include <memory>
#include <queue>

namespace pbrt {

LightCone::LightCone():
  _axis(0.0f, 0.0f, 0.0f),
  _thetaO(0.0f),
  _thetaE(0.0f) {}

LightCone::LightCone(const Vector3f& axis, Float thetaO, Float thetaE):
  _axis(axis),
  _thetaO(thetaO),
  _thetaE(thetaE) {}

Float LightCone::measure() const {
  Float constant = 2.0f * Pi * (1.0f - std::cos(_thetaO));

  Float thetaW = std::min(Pi, _thetaO + _thetaE);
  Float integrate = 0.5f * Pi * (2 * thetaW * std::sin(_thetaO) - 
      std::cos(_thetaO-2*thetaW) - 2*_thetaO*std::sin(_thetaO) +
      std::cos(_thetaO));
  return constant + integrate;
}

LightCone LightCone::Union(const LightCone& lc1, 
    const LightCone& lc2) {

  const LightCone* minLc;
  const LightCone* maxLc;
  if (lc1.thetaO() < lc2.thetaO()) {
    minLc = &lc1;
    maxLc = &lc2;
  } else {
    minLc = &lc2;
    maxLc = &lc1;
  }

  Float thetaD = std::acos(Dot(minLc->axis(), maxLc->axis()));
  Float thetaE = std::max(minLc->thetaE(), maxLc->thetaE());

  if (std::min(thetaD+minLc->thetaO(), Pi) < maxLc->thetaO()) {
    return LightCone(maxLc->axis(), maxLc->thetaO(), thetaE);
  } else {
    Float thetaO = (minLc->thetaO()+maxLc->thetaO()+thetaD) / 2.0f;
    if (thetaO >= Pi) {
      return LightCone(maxLc->axis(), Pi, thetaE);
    }

    Float thetaR = thetaO - maxLc->thetaO();
    Vector3f pivot = Cross(minLc->axis(), maxLc->axis());
    if (pivot.LengthSquared() > 1e-6) {
      Vector3f newAxis = Rotate(Degrees(thetaO - maxLc->thetaO()), pivot)(maxLc->axis());
      return LightCone(newAxis, thetaO, thetaE);
    } else {
      if (thetaD < PiOver2) {
        return LightCone(maxLc->axis(), maxLc->thetaO(), thetaE);
      } else {
        return LightCone(maxLc->axis(), Pi, thetaE);
      }
    }
  }
}

Float LinearLightTreeNodeGeoTermBound(const LinearLightTreeNode* node, const Point3f& p,
    const Vector3f& n) {
  CHECK_NOTNULL(node);
  Float r = node->bound.Diagonal().Length();

  auto cosSubClamped = [](Float cos_a, Float sin_a, Float cos_b, Float sin_b) {
    if (cos_a > cos_b) {
      return 1.0f;
    } else {
      return cos_a * cos_b + sin_a * sin_b;
    }
  };

  auto sinSubClamped = [](Float cos_a, Float sin_a, Float cos_b, Float sin_b) {
    if (cos_a > cos_b) {
      return 0.0f;
    } else {
      return sin_a * cos_b - sin_b * cos_a;
    }
  };

  auto safeSqrt = [](Float a) {
    return std::sqrt(std::max(a, 0.0f));
  };

  Float thetaO = node->thetaO;
  Float thetaE = node->thetaE;
  Point3f nodeCenter = LinearLightTreeNodeCentroid(node);
  Vector3f wi = Normalize(nodeCenter - p);
  Float d = (p - nodeCenter).Length();
  d = std::max(d, node->bound.Diagonal().Length() / 2.0f);

  Float sinThetaU = r / d;
  Float cosThetaU = safeSqrt(1 - sinThetaU * sinThetaU);
  Float cosThetaI = AbsDot(wi, n);
  Float sinThetaI = safeSqrt(1 - cosThetaI * cosThetaI);
  Float cosThetaIPrime = cosSubClamped(cosThetaI, sinThetaI, cosThetaU, sinThetaU);
  Float sinThetaIPrime = sinSubClamped(cosThetaI, sinThetaI, cosThetaU, sinThetaU);

  Float cosTheta = AbsDot(-wi, node->axis);
  Float sinTheta = safeSqrt(1 - cosTheta * cosTheta);
  Float cosThetaO = std::cos(thetaO);
  Float sinThetaO = safeSqrt(1 - cosThetaO * cosThetaO);

  Float cosThetaMinusThetaO = cosSubClamped(cosTheta, sinTheta, cosThetaO, sinThetaO);
  Float sinThetaMinusThetaO = sinSubClamped(cosTheta, sinTheta, cosThetaO, sinThetaO);
  Float cosThetaPrime = cosSubClamped(cosThetaMinusThetaO, sinThetaMinusThetaO, cosThetaU, sinThetaU);

  if (cosThetaI < std::cos(thetaE)) {
    return 0.0f;
  }

  Float res = cosThetaPrime / (d * d);
  if (n != Vector3f(0, 0, 0)) {
    res *= cosThetaIPrime;
  }

  return std::max(res, 0.0f);
}

struct LightInfo {
  LightInfo(size_t _lightIndex, Bounds3f _bound, Float _power, 
      const LightCone& _cone):
    lightIndex(_lightIndex),
    bound(_bound),
    power(_power),
    centroid(0.5f*_bound.pMin + 0.5f*_bound.pMax),
    cone(_cone) {}

  LightInfo():
    power(0.0f) {}

  size_t lightIndex;
  Bounds3f bound;
  Float power;
  Point3f centroid;
  LightCone cone;
};

struct LightTreeBuildNode {
  Bounds3f bound;
  LightTreeBuildNode* children[2];
  size_t firstLightOffset;
  int splitAxis;
  int nLights;
  Float power;
  Float power2;
  LightCone lightCone;

  void InitLeaf(size_t _offset, int _nLights, 
      Bounds3f _bound, Float _power, Float _power2, LightCone& cone) {
    firstLightOffset = _offset;
    nLights = _nLights;
    bound = _bound;
    splitAxis = -1;
    children[0] = children[1] = nullptr;
    power = _power;
    power2 = _power2;
    lightCone = cone;
  }

  void InitInternal(int _splitAxis, const LightCone& cone, LightTreeBuildNode* l, 
      LightTreeBuildNode* r) {
    children[0] = l;
    children[1] = r;
    splitAxis = _splitAxis;
    bound = Union(children[0]->bound, children[1]->bound);
    nLights = children[0]->nLights + children[1]->nLights;
    power = children[0]->power + children[1]->power;
    lightCone = cone;
    firstLightOffset = std::min(children[0]->firstLightOffset, children[1]->firstLightOffset);
  }

  bool isLeaf() const {
    return splitAxis < 0;
  }
};

LightTree::LightTree(const std::vector<std::shared_ptr<Light>>& lights,
    int maxLightsPerNode, 
    SplitMethod splitMethod):_maxLightsPerNode(maxLightsPerNode),
                             _splitMethod(splitMethod) {
    ProfilePhase _(Prof::LightTreeConstruction);

    _lights.assign(lights.begin(), lights.end());

    // Check if there are ligths;
    if (_lights.empty()) {
      Warning("LightTree::LightTree(): No light found.");
      return;
    }

    // Initializa light infos
    std::vector<LightInfo> lightsInfo;
    for (size_t i = 0; i < _lights.size(); ++i) {
      Vector3f axis;
      Float theatO;
      Float thetaE;
      _lights[i]->GetOrientationAttributes(axis, theatO, thetaE);
      lightsInfo.push_back(LightInfo(i, 
            _lights[i]->WorldBound(), 
            _lights[i]->Power().y(),
            LightCone(axis, theatO, thetaE)));
    }

    // Build light tree
    MemoryArena memory(1024 * 1024);
    int totalNodes = 0;
    _lightDistrib.resize(_lights.size());
    std::vector<std::shared_ptr<Light>> orderedLights;
    orderedLights.reserve(_lights.size());

    LightTreeBuildNode* root = recursiveBuild(memory, lightsInfo, 0, lightsInfo.size(), 
        &totalNodes, orderedLights);

    _nodes = AllocAligned<LinearLightTreeNode>(totalNodes);
    int off = 0;
    flattenTree(root, off, orderedLights, -1);
    _lights = orderedLights;
    _totalNodes = totalNodes;
}

LightTree::~LightTree() {
  FreeAligned(_nodes);
}

LightTreeBuildNode* LightTree::recursiveBuild(MemoryArena& arena, std::vector<LightInfo>& lightsInfo,
    int start, int end, int* totalNodes,
    std::vector<std::shared_ptr<Light>>& orderedLights) {
  CHECK_NE(start, end);
  LightTreeBuildNode* node = arena.Alloc<LightTreeBuildNode>();
  (*totalNodes)++;
  
  // Compute Bounds
  Bounds3f bound;
  LightCone cone;
  Float power = 0.0f;
  Float power2 = 0.0f;
  for (int i = start; i < end; ++i) {
    bound = Union(bound, lightsInfo[i].bound);
    cone = LightCone::Union(cone, lightsInfo[i].cone);
    power += lightsInfo[i].power;
    power2 += lightsInfo[i].power * lightsInfo[i].power;
  }

  int numLights = end - start;
  if (numLights <= _maxLightsPerNode) {
    // Init a leaf when there is only one leaf.
    size_t offset = orderedLights.size();
    for (int i = start; i < end; ++i) {
      _lights[lightsInfo[i].lightIndex]->index = orderedLights.size();
      orderedLights.push_back(_lights[lightsInfo[i].lightIndex]);
    }
    node->InitLeaf(offset, numLights, bound, power, power2, cone);
    return node;
  } else {
    Bounds3f centroidBound;
    for (int i = start; i < end; ++i) {
      centroidBound = Union(centroidBound, lightsInfo[i].centroid);
    }
    int dim = centroidBound.MaximumExtent();
    int mid;

    switch (_splitMethod) {
      case SplitMethod::Middle:
        {
          Float lmid = (centroidBound.pMin[dim] + centroidBound.pMax[dim]) / 2.0f;
          LightInfo* midLight = std::partition(&lightsInfo[start], &lightsInfo[end-1]+1, 
              [dim, lmid](const LightInfo& i) {
                return i.centroid[dim] < lmid;
              });
          mid = midLight - &lightsInfo[0];
          // If multiple bounding box overlaps each other, then use equal counts
          if (mid != start && mid != end)
            break;
        }
      case SplitMethod::EqualCounts:
        {
          mid = (start + end) / 2;
          std::nth_element(&lightsInfo[start], &lightsInfo[mid], &lightsInfo[end-1]+1,
              [dim](const LightInfo& a, const LightInfo& b) {
                return a.centroid[dim] < b.centroid[dim];
              });
          break;
        }
      case SplitMethod::SAH:
      default:
        {
          if (numLights <= 2) {
            mid = (start + end) / 2;
            std::nth_element(&lightsInfo[start], &lightsInfo[mid], &lightsInfo[end-1]+1,
                [dim](const LightInfo& a, const LightInfo& b) {
                  return a.centroid[dim] < b.centroid[dim];
                });
          } else {
            PBRT_CONSTEXPR int nBuckets = 12;
            Float minCost = std::numeric_limits<Float>::max();
            int minCostSplitBucket = -1;

            Float maxLength = bound.Diagonal()[bound.MaximumExtent()];
            for (int dim_t = 0; dim_t < 3; ++dim_t) {
              LightInfo buckets[nBuckets];
              for (int i = start; i < end; ++i) {
                int b = nBuckets * centroidBound.Offset(lightsInfo[i].centroid)[dim_t];
                if (b == nBuckets) b--;
                CHECK_GE(b, 0);
                CHECK_LT(b, nBuckets);
                buckets[b].power += lightsInfo[i].power;
                buckets[b].cone = LightCone::Union(buckets[b].cone, lightsInfo[i].cone);
                buckets[b].bound = Union(buckets[b].bound, lightsInfo[i].bound);
              }

              Float k = maxLength / bound.Diagonal()[dim_t];
              for (int i = 0; i < nBuckets-1; ++i) {
                Bounds3f b0, b1;
                LightCone c0, c1;
                Float power0 = 0.0f, power1 = 0.0f;
                for (int j = 0; j <=i; ++j) {
                  b0 = Union(b0, buckets[j].bound);
                  c0 = LightCone::Union(c0, buckets[j].cone);
                  power0 += buckets[j].power;
                }
                for (int j = i+1; j < nBuckets; ++j) {
                  b1 = Union(b1, buckets[j].bound);
                  c1 = LightCone::Union(c1, buckets[j].cone);
                  power1 += buckets[j].power;
                }

                Float cost = k * (power0 * b0.SurfaceArea() * c0.measure() + 
                    power1 * b1.SurfaceArea() * c1.measure()) / (bound.SurfaceArea() * cone.measure() + 1.0f);

                if (cost <  minCost) {
                  minCost = cost;
                  dim = dim_t;
                  minCostSplitBucket = i;
                }
              }
            }

            Float leafCost = power;
            if (numLights > _maxLightsPerNode || minCost < leafCost) {
              LightInfo* lmid = std::partition(&lightsInfo[start], &lightsInfo[end-1]+1,
                  [=](const LightInfo& a) {
                    int b = nBuckets * centroidBound.Offset(a.centroid)[dim];
                    if (b == nBuckets) b--;
                    CHECK_GE(b, 0);
                    CHECK_LT(b, nBuckets);
                    return b <= minCostSplitBucket;
                  });
              mid = lmid - &lightsInfo[0];
            } else {
              int firstLightOffset = orderedLights.size();
              for (int i = start; i < end; ++i) {
                int lightIndex = lightsInfo[i].lightIndex;
                _lights[lightIndex]->index = orderedLights.size();
                orderedLights.push_back(_lights[lightIndex]);
              }
              node->InitLeaf(firstLightOffset, numLights, bound, power, power2, cone);
              return node; 
            }
            break;
          }
        }
    }
    LightTreeBuildNode* l = recursiveBuild(arena, lightsInfo, 
        start, mid, totalNodes, orderedLights);
    LightTreeBuildNode* r = recursiveBuild(arena, lightsInfo, 
        mid, end, totalNodes, orderedLights);
    node->InitInternal(dim, cone, l, r);
  }

  return node;
}

int LightTree::flattenTree(LightTreeBuildNode* node, 
    int& off, std::vector<std::shared_ptr<Light> >& orderedLights, int parentID) {
  LinearLightTreeNode* linearNode = &_nodes[off];
  linearNode->power = node->power;
  linearNode->power2 = node->power2;
  linearNode->bound = node->bound;
  linearNode->parent = parentID;
  linearNode->axis = node->lightCone.axis();
  linearNode->thetaE = node->lightCone.thetaE();
  linearNode->thetaO = node->lightCone.thetaO();
  linearNode->nodeID = off;
  int curOffset = off++;
  if (node->isLeaf()) {
    linearNode->lightsOffset = node->firstLightOffset;
    linearNode->splitAxis = -1;
    linearNode->nLight = node->nLights;

    Float curPower = 0.0f;
    for (int iLight = linearNode->lightsOffset; 
        iLight < linearNode->lightsOffset + linearNode->nLight; ++iLight) {
      _lightDistrib[iLight] = orderedLights[iLight]->Power().y();
      curPower += _lightDistrib[iLight];
      _lightDistrib[iLight] = linearNode->power <= 0.0f ? (iLight - linearNode->lightsOffset + 1) / (linearNode->nLight) 
        : curPower / linearNode->power;
    }
    _lightDistrib[linearNode->lightsOffset + linearNode->nLight - 1] = 1.0f;
  } else {
    linearNode->splitAxis = node->splitAxis;
    linearNode->nLight = node->nLights;
    linearNode->lightsOffset = node->firstLightOffset;
    flattenTree(node->children[0], off, orderedLights, curOffset);
    linearNode->secondChildOffset = flattenTree(node->children[1], off, orderedLights, curOffset);
  }

  return curOffset;
}

int LightTree::sampleOneLight(const LinearLightTreeNode* node, Float u, Float* pdf) const {
  CHECK_NOTNULL(node);
  DCHECK(LinearLightTreeNodeIsLeaf(node));

  if (node->nLight == 1) {
    *pdf = 1.0f;
    return 0;
  }

  for (int iIndex = 0; iIndex < node->nLight; ++iIndex) {
    Float last = iIndex > 0 ? _lightDistrib[node->lightsOffset + iIndex - 1] : 0.0f;
    Float cur = _lightDistrib[node->lightsOffset + iIndex];
    if (cur >= u) {
      *pdf = cur - last;
      return iIndex;
    }
  }

  *pdf = 1.0f - _lightDistrib[node->lightsOffset + node->nLight - 2];
  return node->nLight - 1;
}

Float LightTree::Pdf(uint32_t startIndex, const Vector3f &wi, const Interaction &its, 
    const Light* light, std::function<Float (
      const Vector3f &, const Interaction &, const LinearLightTreeNode &)> &&weightFunction) const {
  CHECK_NOTNULL(light);
  uint32_t curOffset = startIndex;
  Float pdf = 1.0f;
  while(!LinearLightTreeNodeIsLeaf(&_nodes[curOffset])) {
    const LinearLightTreeNode& lightTreeNode = _nodes[curOffset];
    Float w1 = weightFunction(wi, its, _nodes[curOffset + 1]);
    Float w2 = weightFunction(wi, its, _nodes[lightTreeNode.secondChildOffset]);

    if (w1 <= 0.0f && w2 <= 0.0f) {
      return 0.0f;
    }

    const LinearLightTreeNode& lchild = _nodes[curOffset + 1];
    const LinearLightTreeNode& rchild = _nodes[lightTreeNode.secondChildOffset];
    if (light->index >= lchild.lightsOffset && 
        light->index < lchild.lightsOffset + lchild.nLight) {
      pdf *= (w1 / (w1 + w2));
      ++curOffset;
    } else if (light->index >= rchild.lightsOffset &&
        light->index < rchild.lightsOffset + rchild.nLight) {
      pdf *= (w2 / (w1 + w2));
      curOffset = lightTreeNode.secondChildOffset;
    } else {
      return 0.0f;
    }
  }

  if (LinearLightTreeNodeIsLeaf(&_nodes[curOffset])) {
    pdf *= (light->Power().y() / _nodes[curOffset].power);
  } else {
    return 0.0f;
  }

  return pdf;
}

LightTreeSamplingResult LightTree::Sample(const Vector3f& wi, const Interaction& its,
    int curOffset, Float u,
    std::function<Float(const Vector3f&, const Interaction&, const LinearLightTreeNode&)>&& weightFunction,
    std::function<Spectrum(const std::shared_ptr<Light>&)>&& estimationFunction,
    std::function<void(uint32_t, const LightTreeSamplingResult&)>&& callback) const {
   const LinearLightTreeNode& curNode = _nodes[curOffset];
   LightTreeSamplingResult result;
 
   if (LinearLightTreeNodeIsLeaf(&curNode)) {
     Float lightPdf;
     int offset = sampleOneLight(&curNode, u, &lightPdf);
     int selectedIndex = curNode.lightsOffset + offset;
 
     const std::shared_ptr<Light>& light = this->getLightByIndex(selectedIndex);
     Spectrum l = estimationFunction(light);
     result = { l, lightPdf };
   } else {
     Float w1 = weightFunction(wi, its, _nodes[curOffset + 1]);
     Float w2 = weightFunction(wi, its, _nodes[curNode.secondChildOffset]);
 
     if (w1 <= 0.0f && w2 <= 0.0f) {
       return { Spectrum(0.0f), 0.0f };
     }
 
     Float p = w1 / (w1 + w2);
     if (u < p) {
       result = Sample(wi, its, curOffset + 1, u / p, std::move(weightFunction), 
                                                      std::move(estimationFunction), 
                                                      std::move(callback));
       result.second *= p;
     } else {
       result = Sample(wi, its, curNode.secondChildOffset, (u - p) / (1.0f - p),
           std::move(weightFunction), 
           std::move(estimationFunction), 
           std::move(callback));
       result.second *= (1.0f - p);
     }
   }

   callback(curOffset, result);

   return result;
}

void LightTree::sample(const Vector3f& wi, const Interaction& its,
    int curOffset, Float u, int& lightOffset, Float& pdf,
    std::vector<Float>& weights,
    std::function<Float(const Vector3f&, const Interaction&, const LinearLightTreeNode&)> F) const {
  LinearLightTreeNode curNode = _nodes[curOffset];

  if (LinearLightTreeNodeIsLeaf(&curNode)) {
    Float lightPdf;
    int offset = sampleOneLight(&curNode, u, &lightPdf);
    int selectedIndex = curNode.lightsOffset + offset;
    lightOffset = selectedIndex;
    pdf = lightPdf;
    return;
  } else {
    Float w1, w2;
    if (weights[curOffset] >= 0.0f && weights[curNode.secondChildOffset] >= 0.0f) {
      w1 = weights[curOffset+1];
      w2 = weights[curNode.secondChildOffset];
    } else {
      weights[curOffset+1] = w1 = F(wi, its, _nodes[curOffset+1]);
      weights[curNode.secondChildOffset] = w2 = F(wi, its, _nodes[curNode.secondChildOffset]);
    }

    if (w1 <= 0.0f && w2 <= 0.0f) {
      pdf = 0.0f;
      return;
    }

    Float p = w1 / (w1 + w2);
    if (u < p) {
      sample(wi, its, curOffset+1, u/p, lightOffset, pdf, weights, F);
      pdf = p * pdf;
    } else {
      sample(wi, its, curNode.secondChildOffset, (u - p) / (1.0f - p), lightOffset, pdf, weights, F);
      pdf = (1.0f - p) * pdf;
    }
    if (pdf > 1.0) {
      std::cout << p << " " << pdf << std::endl;
    }
  }
}

void LightTree::sample(const Vector3f& wi, const SurfaceInteraction& its,
    int curOffset, Float u, int& lightOffset, Float& pdf,
    std::vector<Float>& weights) const {
  LinearLightTreeNode curNode = _nodes[curOffset];

  BRDFRecord record;
  bool haveLTC = LTC::GetBRDFRecord(its, record);

  if (LinearLightTreeNodeIsLeaf(&curNode)) {
    Float lightPdf;
    int offset = sampleOneLight(&curNode, u, &lightPdf);
    int selectedIndex = curNode.lightsOffset + offset;
    lightOffset = selectedIndex;
    pdf = lightPdf;
    return;
  } else {
    Float w1, w2;
    if (weights[curOffset] >= 0.0f && weights[curNode.secondChildOffset] >= 0.0f) {
      w1 = weights[curOffset+1];
      w2 = weights[curNode.secondChildOffset];
    } else {
      Float d1 = SafeMinDistance(_nodes[curOffset+1].bound, its.p);
      Float d2 = SafeMinDistance(_nodes[curNode.secondChildOffset].bound, its.p);
  
      Vector3f wo1 = Normalize(LinearLightTreeNodeCentroid(&_nodes[curOffset+1]) - its.p);
      Vector3f wo2 = Normalize(LinearLightTreeNodeCentroid(&_nodes[curNode.secondChildOffset]) - its.p);
  
      Float bsdfVal1;
      Float bsdfVal2;
      bsdfVal1 = bsdfVal2 = 1.0f;

      weights[curOffset+1] = bsdfVal1 * _nodes[curOffset+1].power / std::pow(d1, 2.0f);
      weights[curNode.secondChildOffset] = bsdfVal2 * _nodes[curNode.secondChildOffset].power / std::pow(d2, 2.0f);

      w1 = weights[curOffset+1];
      w2 = weights[curNode.secondChildOffset];
    }

    if (w1 <= 0.0f && w2 <= 0.0f) {
      pdf = 0.0f;
      return;
    }

    Float p = w1 / (w1 + w2);
    if (u < p) {
      sample(wi, its, curOffset+1, u/p, lightOffset, pdf, weights);
      pdf = p * pdf;
    } else {
      sample(wi, its, curNode.secondChildOffset, (1.0f-u)/(1.0f-p), lightOffset, pdf, weights);
      pdf = (1.0f - p) * pdf;
    }
    if (pdf > 1.0) {
      std::cout << p << " " << pdf << std::endl;
    }
  }
}

std::shared_ptr<Light> LightTree::getLightByIndex(int index) const {
  CHECK_NE(index, _lights.size());
  return _lights[index];
}

std::shared_ptr<LightCut> LightTree::generateLightcut(Float thresholdP, int maxSize, 
    std::function<Float(const LinearLightTreeNode&)>&& weightingFunction) const {
  LightCut* cut = new LightCut();
  cut->_cut.reserve(maxSize);

  std::vector<int> indexList;
  indexList.reserve(maxSize);

  auto compare = [&, this](int a, int b) {
    return weightingFunction(_nodes[a]) < weightingFunction(_nodes[b]);
  };

  std::priority_queue<int, 
    std::vector<int>, 
    decltype(compare)> Q(compare);
  
  Float weightSum = weightingFunction(_nodes[0]);
  Q.push(0);
  int currentSize = 1;
  while(!Q.empty() && currentSize < maxSize) {
    int curNodeIndex = Q.top();
    Q.pop();
    if (LinearLightTreeNodeIsLeaf(&_nodes[curNodeIndex]) || 
        weightingFunction(_nodes[curNodeIndex]) <= thresholdP * weightSum) {
      indexList.push_back(curNodeIndex);
    } else {
      ++currentSize;
      uint32_t secondChildOffset = _nodes[curNodeIndex].secondChildOffset;
      Q.push(curNodeIndex + 1);
      Q.push(secondChildOffset);
      weightSum = (weightSum - weightingFunction(_nodes[curNodeIndex]) + 
          weightingFunction(_nodes[curNodeIndex + 1]) + weightingFunction(_nodes[secondChildOffset]));
    }
  }

  while(!Q.empty()) {
    indexList.push_back(Q.top());
    Q.pop();
  }

  int cutSize = indexList.size();

  std::sort(indexList.begin(), indexList.end());
  for (int i = 0; i < cutSize; ++i) {
    Float weight = weightingFunction(_nodes[indexList[i]]);
    cut->Append(LightCutNode(indexList[i], weight));
  }
  cut->UpdateDistribution();

  return std::shared_ptr<LightCut>(cut);
}

const LinearLightTreeNode* LightTree::getNodeByIndex(int index) const {
  if (_nodes != nullptr) {
    return &_nodes[index];
  }
  return nullptr;
}

std::shared_ptr<LightTree> CreateLightTree(
    std::vector<std::shared_ptr<Light>> lights,
    const ParamSet& paramset) {
  LightTree::SplitMethod splitMethod;
  std::string splitMethodName = paramset.FindOneString("splitmethod", "sah");
  if (splitMethodName == "sah") {
    splitMethod = LightTree::SplitMethod::SAH;
  } else if (splitMethodName == "middle") {
    splitMethod = LightTree::SplitMethod::Middle;
  } else if (splitMethodName == "equal") {
    splitMethod = LightTree::SplitMethod::EqualCounts;
  } else {
    Warning("LightTree split method unknown. Using sah instead.");
    splitMethod = LightTree::SplitMethod::SAH;
  }

  int maxLightsPerNode = paramset.FindOneInt("maxlightspernode", 4);
  return std::make_shared<LightTree>(std::move(lights), maxLightsPerNode, splitMethod);
}

} // namespace pbrt
