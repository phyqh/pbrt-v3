// Copyright @TwoCookingMice 2020

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_LIGHTSAMPLERS_NRL_H
#define PBRT_LIGHTSAMPLERS_NRL_H

#include "lightsampler.h"
#include "accelerators/lighttree.h"
#include "paramset.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pbrt {

// Implementation of paper "Importance Sampling of Many Lights with Reinforcement Lightcuts Learning"
// from Prof. Jacopo Pantaleoni
class NaiveRLLightSampler : public LightTreeSampler {
  public:
    NaiveRLLightSampler(uint32_t shadingPointClusters, uint32_t direction, Float lr, uint32_t maxLightCutSize, bool biased):
      _shadingPointClusters(shadingPointClusters), 
      _directionDivideNumber(direction), 
      _learningRate(lr), 
      _maxLightCutSize(maxLightCutSize),
      _biased(biased) {}
    virtual void Preprocess(const Scene& scene, 
        const std::vector<std::shared_ptr<Light>>& lights, bool isVpl = false) override;
    virtual Spectrum Sample(const Vector3f& wi, const Interaction& it, const Scene& scene,
        MemoryArena& arena, Sampler& sampler,
        bool handleMedia = false, uint32_t nSamples = 1) override;

  public:
    struct HashTableEntry {
      std::shared_ptr<LightCut> lightCut;
      std::vector<int> treeNodeToCutNode;
      uint32_t iteration;

      HashTableEntry(const std::shared_ptr<LightCut> lightCut, int maxLightCutSize):
        lightCut(lightCut->Clone()), treeNodeToCutNode(maxLightCutSize, -1), iteration(1) {
          treeNodeToCutNode.reserve(maxLightCutSize);
          for (int i = 0; i < lightCut->Size(); ++i) {
            int lightTreeNodeIndex = lightCut->_cut[i].lightTreeNodeIndex;
            treeNodeToCutNode[lightTreeNodeIndex] = i;
          }
      }

      Float Weight(uint32_t lightTreeNodeIndex) const {
        if (treeNodeToCutNode[lightTreeNodeIndex] < 0) {
          return -1.0;
        }
        uint32_t lightCutNodeIndex = treeNodeToCutNode[lightTreeNodeIndex];
        return lightCut->_cut[lightCutNodeIndex].weight;
      }
    };

  private:
    uint32_t _shadingPointClusters;
    uint32_t _directionDivideNumber;
    Float _learningRate;
    uint32_t _maxLightCutSize;
    std::shared_ptr<LightCut> _globalLightCut;
    std::vector<std::shared_ptr<HashTableEntry> > _hashMap;
    std::mutex _hashMapMutex[4];
    Float _directionalGridSize[2];
    Bounds3f _worldBound;
    bool _biased;
};

LightSampler* CreateNaiveRLLightSampler(const ParamSet& params);

} // namespace pbrt

#endif
