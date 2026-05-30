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
// Builds a VkMicromapEXT for every alpha-masked mesh submitted to it, then
// attaches the result into the engine's BLAS build via the standard Vulkan
// pNext chain.  After the BLAS is rebuilt the GPU hardware traversal unit
// resolves most shadow-ray hits against alpha-tested geometry without running
// any shader code.  Only micro-triangles at the very edge of the alpha
// boundary still invoke the any-hit shader.
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
// The shadow shader (shaders/ray_query.slang) does NOT need any changes;
// the speed-up is entirely driven by micromap data embedded in the BLAS.
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

// Engine types used by the builder.
#include "../mesh_component.h"   // MeshComponent, Vertex
#include "../model_loader.h"     // Material, ModelLoader
#include "../memory_pool.h"      // MemoryPool (forward-only, we don't call it)

// Forward declarations
class Renderer;

// ---------------------------------------------------------------------------
// OmmConfig — tuning knobs exposed to students / the UI panel
// ---------------------------------------------------------------------------
struct OmmConfig {
  /// Subdivision level 0–4.  Level 2 → 16 micro-triangles per triangle.
  /// Higher = better accuracy at alpha edges, more GPU memory.
  uint32_t subdivisionLevel = 2;

  /// Alpha values below this threshold → TRANSPARENT state (hardware passes ray).
  float transparentThreshold = 0.05f;

  /// Alpha values at or above this threshold → OPAQUE state (hardware blocks ray).
  float opaqueThreshold = 0.95f;

  /// Samples taken per micro-triangle centroid during classification.
  /// More samples → fewer mis-classifications at gradient edges; slower build.
  uint32_t samplesPerMicroTriangle = 4;

  /// When true, micro-triangles whose average alpha falls between the two
  /// thresholds are classified UNKNOWN — the any-hit shader fires for them.
  /// When false, they are forced OPAQUE (zero shader cost, minor visual bias).
  bool allowUnknownState = true;
};

// ---------------------------------------------------------------------------
// OmmMeshInfo — per-mesh result returned by buildForMesh()
// ---------------------------------------------------------------------------
struct OmmMeshInfo {
  /// True when a VkMicromapEXT was successfully built for this mesh.
  bool built = false;

  /// Pointer to a VkAccelerationStructureTrianglesOpacityMicromapEXT that
  /// should be placed in the pNext chain of the geometry triangles struct.
  /// This memory is owned by OpacityMicromapBuilder and stays valid until
  /// reset() is called.
  void* pNextChain = nullptr;

  // Diagnostics — shown in the ImGui panel.
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

  // Non-copyable; non-movable (due to mutex).
  OpacityMicromapBuilder(const OpacityMicromapBuilder&)            = delete;
  OpacityMicromapBuilder& operator=(const OpacityMicromapBuilder&) = delete;
  OpacityMicromapBuilder(OpacityMicromapBuilder&&)                 = delete;
  OpacityMicromapBuilder& operator=(OpacityMicromapBuilder&&)      = delete;

  // ── Lifecycle ─────────────────────────────────────────────────────────────
  void swap(OpacityMicromapBuilder& other) noexcept;

  /// Bind engine resources.  Safe to call even when the extension is not
  /// supported — buildForMesh() will return false in that case.
  void init(Renderer& renderer);

  /// Release all GPU resources.  Must be called before the Renderer is
  /// destroyed.  Safe to call multiple times.
  void reset();

  /// Returns true if VK_EXT_opacity_micromap is available and was enabled.
  [[nodiscard]] bool isSupported() const { return m_supported; }

  // ── Build ─────────────────────────────────────────────────────────────────

  /// Classify micro-triangles for one alpha-masked mesh and build its
  /// VkMicromapEXT.  Call once per mesh after CPU texture data is available.
  ///
  /// @param mesh       The MeshComponent whose BLAS will receive the micromap.
  /// @param texPixels  CPU-accessible RGBA/RGB/R pixel data (stb_image layout).
  /// @param texW       Texture width in texels.
  /// @param texH       Texture height in texels.
  /// @param texChannels Bytes per texel (1, 2, 3, or 4; alpha taken from last channel when ≥2).
  /// @param config     Quality/memory trade-off settings.
  /// @returns OmmMeshInfo describing the result.
  [[nodiscard]] OmmMeshInfo buildForMesh(const MeshComponent* mesh,
                                         const uint8_t*       texPixels,
                                         uint32_t             texW,
                                         uint32_t             texH,
                                         uint32_t             texChannels,
                                         const OmmConfig&     config = {});

  /// Look up the OmmMeshInfo built for a given mesh.
  /// Returns nullptr when no micromap exists for that mesh.
  [[nodiscard]] const OmmMeshInfo* getInfo(const MeshComponent* mesh) const;

  // ── Statistics ────────────────────────────────────────────────────────────

  [[nodiscard]] uint32_t micromapCount() const;
  [[nodiscard]] uint64_t totalGpuBytes() const;

private:
  // ── Internal phase implementations ───────────────────────────────────────

  [[nodiscard]] bool analyseVariation(const std::vector<Vertex>&   verts,
                                      const std::vector<uint32_t>& indices,
                                      const uint8_t* pixels,
                                      uint32_t w, uint32_t h, uint32_t ch,
                                      const OmmConfig& cfg) const;

  void classify(const std::vector<Vertex>&   verts,
                const std::vector<uint32_t>& indices,
                const uint8_t* pixels,
                uint32_t w, uint32_t h, uint32_t ch,
                const OmmConfig& cfg,
                std::vector<uint8_t>& outStates,
                OmmMeshInfo& outInfo) const;

  [[nodiscard]] void* buildOnGpu(const MeshComponent* mesh,
                                  const std::vector<uint8_t>& unpackedStates,
                                  uint32_t triangleCount,
                                  uint32_t subdivisionLevel,
                                  OmmMeshInfo& outInfo);

  // ── Helpers ───────────────────────────────────────────────────────────────

  [[nodiscard]] float sampleAlpha(const uint8_t* pixels,
                                   uint32_t w, uint32_t h, uint32_t ch,
                                   float u, float v) const;

  [[nodiscard]] vk::raii::CommandBuffer beginOneShot(vk::raii::CommandPool& pool) const;
  void                                  submitOneShot(vk::raii::CommandBuffer& cb) const;

  // ── Per-micromap GPU resource ownership ──────────────────────────────────

  // The pNext chain that the BLAS build reads must stay alive for the BLAS's
  // lifetime.  We keep it here alongside the GPU buffers it refers to.
  struct GpuEntry {
    vk::raii::Buffer       dataBuf  {nullptr}; // 2-bit state data
    vk::raii::DeviceMemory dataMem  {nullptr};
    vk::raii::Buffer       mmBuf    {nullptr}; // VkMicromapEXT storage
    vk::raii::DeviceMemory mmMem    {nullptr};
    vk::raii::Buffer       triBuf   {nullptr}; // Array of VkMicromapTriangleEXT
    vk::raii::DeviceMemory triMem   {nullptr};
    vk::raii::MicromapEXT  micromap {nullptr};

    // Owns the pNext chain struct lifetime — never relocated after construction.
    struct PNextStorage {
      vk::AccelerationStructureTrianglesOpacityMicromapEXT chain{};
      vk::MicromapUsageEXT usageEntry{};
    };
    std::unique_ptr<PNextStorage> pNextOwner;
  };

  // ── State ─────────────────────────────────────────────────────────────────
  mutable std::mutex           m_mutex;

  bool                         m_initialised = false;
  bool                         m_supported   = false;
  Renderer*                    m_renderer    = nullptr;
  const vk::raii::Device*      m_device      = nullptr;
  const vk::raii::PhysicalDevice* m_physDev  = nullptr;
  const vk::raii::CommandPool* m_commandPool = nullptr;
  uint32_t                     m_gfxFamily   = 0;

  // Map from MeshComponent* → index into m_entries / m_infos.
  std::unordered_map<const MeshComponent*, size_t> m_meshToEntry;
  std::vector<GpuEntry>   m_entries;
  std::vector<OmmMeshInfo> m_infos;

  uint64_t m_totalGpuBytes = 0;
};
