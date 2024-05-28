// Copyright @TwoCookingMice 2020

#include "lightsamplers/nrl.h"
#include "geometry.h"
#include "stats.h"

#include <vector>

namespace pbrt {

STAT_COUNTER("Lightsampler/Total Hash Table Iterations", totalIterations);
STAT_MEMORY_COUNTER("Memory/Hash Table Size", totalHashTableSize);
STAT_PERCENT("Lightsampler/Hash Table Sparsity", liveLightCutCount, totalLightCutCount);
STAT_COUNTER("Lightsampler/Merge Collapse Count", updateCount);

void NaiveRLLightSampler::Preprocess(const Scene& scene, 
    const std::vector<std::shared_ptr<Light>>& lights, bool isVpl) {
  LightTreeSampler::Preprocess(scene, lights, isVpl);
  scene.ComputeShadingPointClusters(_shadingPointClusters);

  if (_maxLightCutSize == 0) {
    _maxLightCutSize = (_lightTree->TotalNodes()) / 10 + 9;
  }
  _globalLightCut = _lightTree->generateLightcut(0.001, _maxLightCutSize, 
  [](const LinearLightTreeNode& node) {
    return node.power;
  });

  _worldBound = scene.WorldBound();

  // Our Light Field is encoded as ((Phi, Theta), (x, y, z))
  _directionalGridSize[0] = Pi / _directionDivideNumber;
  _directionalGridSize[1] = 2.0f * Pi / _directionDivideNumber;

  uint32_t hashTableSize = _shadingPointClusters * _directionDivideNumber * _directionDivideNumber;
  _hashMap = std::vector<std::shared_ptr<HashTableEntry> >(hashTableSize, std::shared_ptr<HashTableEntry>(nullptr));
  // totalHashTableSize += _hashMap.size() * sizeof(HashTableEntry*);

  totalLightCutCount = hashTableSize;
}

Spectrum NaiveRLLightSampler::Sample(const Vector3f& wi, const Interaction& it, const Scene& scene,
    MemoryArena& arena, Sampler& sampler,
    bool handleMedia, uint32_t nSamples) {
  if (!it.IsSurfaceInteraction()) {
    return Spectrum(0.0f);
  }
  SurfaceInteraction its = static_cast<const SurfaceInteraction&>(it);
  CHECK_GE(its.shadingPointCluster, 0);

  Vector3f position(its.p);
  Vector3f normal(its.shading.n);

  // Calculate hash table coordinates
  Float lightFieldEncoding[2] = { SphericalTheta(wi), SphericalPhi(wi) };
  uint32_t index = 0;
  uint32_t factor = 1;
  for (uint32_t idx = 0; idx < 2; ++idx) {
    Float gridMin;
    Float gridSize;
    uint32_t divideNumber;
    if (idx < 2) {
      gridMin = 0.0f;
      gridSize = _directionalGridSize[idx];
      divideNumber = _directionDivideNumber;
    }

    uint32_t hashTableCoordinate = std::floor((lightFieldEncoding[idx] - gridMin) / gridSize);
    hashTableCoordinate = std::min(hashTableCoordinate, divideNumber - 1);
    index += (factor * hashTableCoordinate);
    factor *= divideNumber;
  }
  index += (factor * its.shadingPointCluster);

  {
    std::lock_guard<std::mutex> _(_hashMapMutex[index&3]);
    if (_hashMap[index] == nullptr) {
      _hashMap[index] = std::make_shared<HashTableEntry>(_globalLightCut, _lightTree->TotalNodes());
      ++liveLightCutCount;
      totalHashTableSize += _globalLightCut->Size();
    }
  }
  auto lightCut = _hashMap[index]->lightCut->Clone();
  lightCut->UpdateDistribution();

  auto weightingFunction = [&](const Vector3f& wi, 
      const Interaction& its, 
      const LinearLightTreeNode& n) {
    return n.power;
  };

  auto callback = [&](uint32_t lightTreeIndex, const LightTreeSamplingResult& result) {};

  auto estimateDirectFunction = [&](const std::shared_ptr<Light>& light) {
    Point2f uLight = sampler.Get2D();
    Point2f uScattering = sampler.Get2D();
    Float lightSamplingPdf = 0.0f;
    Spectrum ld = EstimateDirect(it, uScattering, *light, uLight, scene, 
        sampler, arena, lightSamplingPdf);

    return ld;
  };

  Float lr = _learningRate;
  if (lr <= 0.0f) {
    lr = 1.0f / std::pow(4.0f * (_hashMap[index]->iteration), 0.857f);
  }

  Spectrum l(0.0f);
  Float uLightCut = sampler.Get1D();
  for (uint32_t i = 0; i < nSamples; ++i) {
    int initTreeNode;
    int cutNode;
    Float cutPdf;
    Float uLightTree = sampler.Get1D();
    lightCut->UniformSample(uLightCut, cutNode, initTreeNode, cutPdf, &uLightCut);

    LightTreeSamplingResult result = _lightTree->Sample(wi, it, initTreeNode, uLightTree, std::move(weightingFunction), 
        std::move(estimateDirectFunction), std::move(callback));

    if (result.second > 0.0f && cutPdf > 0.0f) {
      Spectrum ld = result.first / (cutPdf * result.second);
      l += ld / nSamples;

      Float wt = _hashMap[index]->lightCut->_cut[cutNode].weight;
      wt = (1.0f - lr) * wt + lr * ld.y();
      _hashMap[index]->lightCut->Update(cutNode, initTreeNode, [&](LightCutNode* node) {
          node->weight = wt;
      });

      if (_biased) {
        lightCut->Update(cutNode, initTreeNode, [&](LightCutNode* node) {
            node->weight = wt;
        });
        lightCut->UpdateDistribution();
      }
    }
  }

  if (_hashMap[index]->lightCut->Size() > 2) {
    std::lock_guard<std::mutex> _(_hashMapMutex[index&3]);
    uint32_t lightCutSize = _hashMap[index]->lightCut->Size();

    uint32_t maxWeightLightCutIndex = -1;
    Float maxWeight = -1.0;
    for (uint32_t idx = 0; idx < lightCutSize; ++idx) {
      int lightTreeNodeIndex = _hashMap[index]->lightCut->_cut[idx].lightTreeNodeIndex;
      Float curWeight = _hashMap[index]->lightCut->_cut[idx].weight;
      if (!LinearLightTreeNodeIsLeaf(_lightTree->getNodeByIndex(lightTreeNodeIndex)) && 
          curWeight > maxWeight) {
        maxWeight = curWeight;
        maxWeightLightCutIndex = idx;
      }
    } 

    if (maxWeightLightCutIndex >= 0) {
      uint32_t minWeightLightTreeNodeIndex = -1;
      Float minWeight = 1e+9;
      for (uint32_t idx = 0; idx < lightCutSize; ++idx) {
        int lightTreeNodeIndex = _hashMap[index]->lightCut->_cut[idx].lightTreeNodeIndex;
        const LinearLightTreeNode* node = _lightTree->getNodeByIndex(lightTreeNodeIndex);
        const LinearLightTreeNode* parentNode = _lightTree->getNodeByIndex(node->parent);

        float lWeight = _hashMap[index]->Weight(node->parent+1);
        float rWeight = _hashMap[index]->Weight(parentNode->secondChildOffset);

        if (lWeight >= 0.0f && rWeight >= 0.0f && lWeight + rWeight < minWeight) {
          minWeight = lWeight + rWeight;
          minWeightLightTreeNodeIndex = lightTreeNodeIndex;
        }
      }

      if (minWeightLightTreeNodeIndex >= 0 && minWeight < maxWeight * 0.01) {
        const LinearLightTreeNode* mergeNode = _lightTree->getNodeByIndex(minWeightLightTreeNodeIndex);
        const LinearLightTreeNode* mergeParentNode = _lightTree->getNodeByIndex(mergeNode->parent);
        int mergeNodeCutIndex = _hashMap[index]->treeNodeToCutNode[mergeNode->parent+1];
        int mergeNodeCutIndex2 = _hashMap[index]->treeNodeToCutNode[mergeParentNode->secondChildOffset];
        int splitNodeTreeIndex = _hashMap[index]->lightCut->_cut[maxWeightLightCutIndex].lightTreeNodeIndex;
        const LinearLightTreeNode* splitNode = _lightTree->getNodeByIndex(splitNodeTreeIndex);
        int splitNodeSecondChild = splitNode->secondChildOffset;

        _hashMap[index]->lightCut->Update(mergeNodeCutIndex, mergeNode->parent+1, 
            [&](LightCutNode* cutNode) {
            cutNode->lightTreeNodeIndex = mergeNode->parent;
            cutNode->weight = minWeight;
            _hashMap[index]->treeNodeToCutNode[mergeNode->parent+1] = -1;
            _hashMap[index]->treeNodeToCutNode[mergeNode->parent] = mergeNodeCutIndex;
            });
        _hashMap[index]->lightCut->Update(maxWeightLightCutIndex, splitNodeTreeIndex, 
            [&](LightCutNode* cutNode) {
            cutNode->lightTreeNodeIndex += 1;
            cutNode->weight = _lightTree->getNodeByIndex(cutNode->lightTreeNodeIndex)->power;
            _hashMap[index]->treeNodeToCutNode[splitNodeTreeIndex+1] = maxWeightLightCutIndex;
            _hashMap[index]->treeNodeToCutNode[splitNodeTreeIndex] = -1;
            });
        _hashMap[index]->lightCut->Update(mergeNodeCutIndex2, mergeParentNode->secondChildOffset, 
            [&](LightCutNode* cutNode) {
            cutNode->lightTreeNodeIndex = splitNodeSecondChild;
            cutNode->weight = _lightTree->getNodeByIndex(cutNode->lightTreeNodeIndex)->power;
            _hashMap[index]->treeNodeToCutNode[mergeParentNode->secondChildOffset] = -1;
            _hashMap[index]->treeNodeToCutNode[splitNodeSecondChild] = mergeNodeCutIndex2;
            });
        ++updateCount;
      }
    }

  }

  _hashMap[index]->lightCut->UpdateDistribution();
  ++_hashMap[index]->iteration;
  ++totalIterations;

  return l;
}

LightSampler* CreateNaiveRLLightSampler(const ParamSet& params) {
  uint32_t shadingPointClusters = params.FindOneInt("shadingpointclusters", 32768);
  uint32_t directionalGridNumber = params.FindOneInt("directionalgridnumber", 1);
  // Defaultly we apply 1/T for learning rate.
  Float learningRate = params.FindOneFloat("learningrate", -1.0);
  uint32_t maxLightCutSize = params.FindOneInt("maxlightcutsize", 1);
  bool biased = params.FindOneBool("biased", false);
  return new NaiveRLLightSampler(shadingPointClusters, directionalGridNumber, learningRate, maxLightCutSize, biased);
}

} // namespace pbrt
