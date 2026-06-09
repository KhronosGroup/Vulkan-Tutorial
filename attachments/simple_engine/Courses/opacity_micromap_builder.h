/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

// =============================================================================
// OpacityMicromapBuilder — Engine-integrated header
// =============================================================================
//
// This class is part of the "Opacity Micromaps" course module.
// See: en/Building_a_Simple_Engine/Courses/Opacity_Micromaps/
//
// WHAT THIS DOES
// --------------
// Builds a VkAccelerationStructureKHR (type=eOpacityMicromap) for every
// alpha-masked mesh submitted to it, then attaches the result into the engine's
// BLAS build via VkAccelerationStructureTrianglesOpacityMicromapKHR in the
// pNext chain.
//
// After the BLAS is rebuilt the GPU hardware traversal unit resolves most
// shadow-ray hits against alpha-tested geometry without running any shader
// code. Only micro-triangles at the very edge of the alpha boundary still
// invoke the any-hit shader.
//
// This module requires VK_KHR_opacity_micromap and VK_KHR_device_address_commands:
//   - Build sizes via vkGetAccelerationStructureBuildSizesKHR
//   - Micromap AS via vkCreateAccelerationStructure2KHR (type=eOpacityMicromap)
//   - Build recorded via vkCmdBuildAccelerationStructuresKHR
//   - Attached to BLAS via VkAccelerationStructureTrianglesOpacityMicromapKHR
//   - Ray query shaders must declare the OpacityMicromapKHR SPIR-V execution
//     mode (via SPV_KHR_opacity_micromap) for the hardware optimisation to
//     activate. Without it the traversal unit ignores all micromap data.
//
// INTEGRATION SEQUENCE
// --------------------
// 1. OpacityMicromapBuilder::init(renderer)
//       — call once after Renderer::Initialize().
// 2. OpacityMicromapBuilder::buildForMesh(mesh, pixels, w, h, ch, config)
//       — call for every alpha-masked mesh just after texture data is
//         available, before buildAccelerationStructures().
// 3. During BLAS construction, look up getInfo(meshComp)->pNextChain and
//       attach it to the geometry struct (done in omm_integration.cpp).
// 4. OpacityMicromapBuilder::reset()
//       — call on scene change / engine shutdown.
//
// COURSE NOTE
// -----------
// Read alongside:
//   en/Building_a_Simple_Engine/Courses/Opacity_Micromaps/05_implementation_overview.adoc
// =============================================================================

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_raii.hpp>
#include "../vulkan_compatibility.h"

// Engine types
#include "../mesh_component.h"
#include "../model_loader.h"
#include "../memory_pool.h"

// Forward declarations
class Renderer;

// ---------------------------------------------------------------------------
// OmmConfig — tuning knobs exposed to students and the ImGui panel
// ---------------------------------------------------------------------------
struct OmmConfig {
  /// Subdivision level 0–4.  Level 2 → 16 micro-triangles per triangle.
  /// Capped by VkPhysicalDeviceOpacityMicromapPropertiesKHR::maxOpacity4StateSubdivisionLevel.
  /// With lossyBuild, up to maxOpacityLossy4StateSubdivisionLevel may be used.
  uint32_t subdivisionLevel = 2;

  /// Alpha values below this threshold → TRANSPARENT (hardware passes ray, no shader).
  float transparentThreshold = 0.05f;

  /// Alpha values at or above this threshold → OPAQUE (hardware blocks ray, no shader).
  float opaqueThreshold = 0.95f;

  /// Samples taken per micro-triangle during classification.
  uint32_t samplesPerMicroTriangle = 4;

  /// When true, boundary micro-triangles are classified UNKNOWN — the any-hit
  /// shader fires for them. When false, they are forced OPAQUE.
  bool allowUnknownState = true;

  /// Request VK_BUILD_ACCELERATION_STRUCTURE_MICROMAP_LOSSY_BIT_KHR.
  /// Allows the driver to apply lossy compression, potentially supporting
  /// higher subdivision levels at the cost of occasional Unknown substitution.
  bool lossyBuild = false;
};

// ---------------------------------------------------------------------------
// OmmMeshInfo — per-mesh result returned by buildForMesh()
// ---------------------------------------------------------------------------
struct OmmMeshInfo {
  /// True when a micromap was successfully built for this mesh.
  bool built = false;

  /// Pointer to a VkAccelerationStructureTrianglesOpacityMicromapKHR that
  /// must be placed in the pNext chain of VkAccelerationStructureGeometryTrianglesDataKHR.
  /// Owned by OpacityMicromapBuilder; valid until reset() is called.
  void* pNextChain = nullptr;

  // Diagnostics shown in the ImGui panel.
  float    pctOpaque       = 0.f;
  float    pctTransparent  = 0.f;
  float    pctUnknown      = 0.f;
  uint32_t totalMicroTris  = 0;
  uint64_t gpuBytes        = 0;
};

// ---------------------------------------------------------------------------
// OpacityMicromapBuilder
// ---------------------------------------------------------------------------
class OpacityMicromapBuilder {
public:
  OpacityMicromapBuilder()  = default;
  ~OpacityMicromapBuilder() { reset(); }

  OpacityMicromapBuilder(const OpacityMicromapBuilder&)            = delete;
  OpacityMicromapBuilder& operator=(const OpacityMicromapBuilder&) = delete;
  OpacityMicromapBuilder(OpacityMicromapBuilder&&)                 = delete;
  OpacityMicromapBuilder& operator=(OpacityMicromapBuilder&&)      = delete;

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  void swap(OpacityMicromapBuilder& other) noexcept;

  /// Bind engine resources. Safe to call even when the extension is absent.
  void init(Renderer& renderer);

  /// Release all GPU resources. Must be called before the Renderer is destroyed.
  void reset();

  /// Returns true if VK_KHR_opacity_micromap is available and was enabled.
  [[nodiscard]] bool isSupported() const { return m_supported; }

  // ── Build ──────────────────────────────────────────────────────────────────

  /// Classify micro-triangles and build the micromap acceleration structure
  /// for one alpha-masked mesh. Call once per mesh after CPU texture data is
  /// available, before the BLAS build.
  [[nodiscard]] OmmMeshInfo buildForMesh(const MeshComponent* mesh,
                                         const uint8_t*       texPixels,
                                         uint32_t             texW,
                                         uint32_t             texH,
                                         uint32_t             texChannels,
                                         const OmmConfig&     config = {});

  /// Look up the OmmMeshInfo for a given mesh. Returns nullptr if none exists.
  [[nodiscard]] const OmmMeshInfo* getInfo(const MeshComponent* mesh) const;

  // ── Statistics ─────────────────────────────────────────────────────────────

  [[nodiscard]] uint32_t micromapCount() const;
  [[nodiscard]] uint64_t totalGpuBytes() const;

private:
  // ── Internal phase implementations ────────────────────────────────────────

  // Phase 1: does this mesh have alpha variation worth encoding?
  [[nodiscard]] bool analyseVariation(const std::vector<Vertex>&   verts,
                                      const std::vector<uint32_t>& indices,
                                      const uint8_t* pixels,
                                      uint32_t w, uint32_t h, uint32_t ch,
                                      const OmmConfig& cfg) const;

  // Phase 2: CPU classification — assign a 2-bit state per micro-triangle.
  void classify(const std::vector<Vertex>&   verts,
                const std::vector<uint32_t>& indices,
                const uint8_t* pixels,
                uint32_t w, uint32_t h, uint32_t ch,
                const OmmConfig& cfg,
                std::vector<uint8_t>& outStates,
                OmmMeshInfo& outInfo) const;

  // Phase 3: GPU construction — upload packed states, create and build the
  // VkAccelerationStructureKHR micromap, fill the pNext attachment chain.
  [[nodiscard]] void* buildOnGpu(const MeshComponent* mesh,
                                  const std::vector<uint8_t>& unpackedStates,
                                  uint32_t triangleCount,
                                  uint32_t subdivisionLevel,
                                  bool lossyBuild,
                                  OmmMeshInfo& outInfo);

  // ── Helpers ────────────────────────────────────────────────────────────────

  [[nodiscard]] float sampleAlpha(const uint8_t* pixels,
                                   uint32_t w, uint32_t h, uint32_t ch,
                                   float u, float v) const;

  [[nodiscard]] vk::raii::CommandBuffer beginOneShot(vk::raii::CommandPool& pool) const;
  void                                  submitOneShot(vk::raii::CommandBuffer& cb) const;

  // ── Per-micromap GPU resource ownership ───────────────────────────────────
  //
  // Micromaps are VkAccelerationStructureKHR objects (type=eOpacityMicromap)
  // built via VK_KHR_device_address_commands + VK_KHR_opacity_micromap.

  struct GpuEntry {
    // State data and triangle-array buffers (build inputs)
    vk::raii::Buffer       dataBuf   {nullptr};
    vk::raii::DeviceMemory dataMem   {nullptr};
    vk::raii::Buffer       triBuf    {nullptr};
    vk::raii::DeviceMemory triMem    {nullptr};

    // Backing storage buffer for the micromap AS
    vk::raii::Buffer       mmStoreBuf {nullptr};
    vk::raii::DeviceMemory mmStoreMem {nullptr};

    // The micromap acceleration structure (VK_KHR_opacity_micromap)
    vk::raii::AccelerationStructureKHR micromap {nullptr};

    // VkAccelerationStructureTrianglesOpacityMicromapKHR + usage entry.
    // Heap-allocated and never moved — the BLAS build holds a raw pointer.
    struct PNextStorage {
      VkMicromapUsageKHR usageEntry{};
      VkAccelerationStructureTrianglesOpacityMicromapKHR chain{};
    };
    std::unique_ptr<PNextStorage> pNextOwner;
  };

  // ── State ──────────────────────────────────────────────────────────────────

  mutable std::mutex              m_mutex;

  bool                            m_initialised = false;
  bool                            m_supported   = false;
  Renderer*                       m_renderer    = nullptr;
  const vk::raii::Device*         m_device      = nullptr;
  const vk::raii::PhysicalDevice* m_physDev     = nullptr;
  uint32_t                        m_gfxFamily   = 0;

  std::unordered_map<const MeshComponent*, size_t> m_meshToEntry;
  std::vector<GpuEntry>    m_entries;
  std::vector<OmmMeshInfo> m_infos;

  uint64_t m_totalGpuBytes = 0;
};
