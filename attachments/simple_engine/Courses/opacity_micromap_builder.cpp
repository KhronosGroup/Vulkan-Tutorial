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

// =============================================================================
// OpacityMicromapBuilder — Implementation
// =============================================================================
//
// Read alongside the course chapter:
//   en/Building_a_Simple_Engine/Courses/Opacity_Micromaps/05_implementation_overview.adoc
//
// THREE PHASES (mirroring the course narrative)
// ----------------------------------------------
//   Phase 1  analyseVariation()  Does this mesh have mixed opaque/transparent
//                                regions worth encoding?  If not, skip.
//   Phase 2  classify()          Sample each micro-triangle centroid in texture
//                                space → assign a 2-bit opacity state.
//   Phase 3  buildOnGpu()        Pack states, upload to GPU, call
//                                vkBuildMicromapsEXT, fill pNext chain.
//
// STATE ENCODING (VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT, 2 bits/entry)
// -----------------------------------------------------------------------
//   0b00  STATE_TRANSPARENT    Hardware passes the ray — no shader runs.
//   0b01  STATE_OPAQUE         Hardware blocks the ray — no shader runs.
//   0b11  STATE_UNKNOWN_OPAQUE Hardware falls back to the any-hit shader.
//                              (Used only for edge micro-triangles when
//                               OmmConfig::allowUnknownState == true.)
//
// IMPORTANT: The builder is keyed by MeshComponent*.  omm_integration.cpp
// calls buildForMesh() for each alpha-masked mesh it finds, then wires the
// returned pNextChain into the BLAS geometry struct before the AS build.
// =============================================================================

#include "opacity_micromap_builder.h"
#include "../renderer.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// Anonymous namespace — internal constants & geometry helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {

constexpr uint8_t STATE_TRANSPARENT    = 0b00;
constexpr uint8_t STATE_OPAQUE         = 0b01;
constexpr uint8_t STATE_UNKNOWN_OPAQUE = 0b11;

// Number of micro-triangles at subdivision level N  =  4^N.
constexpr uint32_t microTriCount(uint32_t level) {
  uint32_t n = 1;
  for (uint32_t i = 0; i < level; ++i) n *= 4;
  return n;
}

// ─────────────────────────────────────────────────────────────────────────────
// generateCentroids
//
// Returns the barycentric (bU, bV) centroid for every micro-triangle at the
// given subdivision level, in the row-major "bird-curve" traversal order
// required by VK_EXT_opacity_micromap.
//
// At level L the grid has 2^L rows.  Row r contains:
//   (2^L − r)       upward-pointing triangles
//   (2^L − r − 1)   downward-pointing triangles (interleaved with upward)
// ─────────────────────────────────────────────────────────────────────────────
void generateRecursive(uint32_t level, uint32_t currentLevel,
                       glm::vec2 v0, glm::vec2 v1, glm::vec2 v2,
                       std::vector<std::array<float,2>>& out) {
  if (currentLevel == level) {
    const glm::vec2 cen = (v0 + v1 + v2) / 3.0f;
    out.push_back({ cen.x, cen.y });
    return;
  }
  const glm::vec2 m01 = (v0 + v1) * 0.5f;
  const glm::vec2 m12 = (v1 + v2) * 0.5f;
  const glm::vec2 m20 = (v2 + v0) * 0.5f;

  // Bird curve recursive traversal (standard Vulkan OMM layout for triangles):
  // 0: Sub-triangle at vertex 0
  // 1: Middle sub-triangle (flipped)
  // 2: Sub-triangle at vertex 1
  // 3: Sub-triangle at vertex 2
  generateRecursive(level, currentLevel + 1, v0,  m01, m20, out);
  generateRecursive(level, currentLevel + 1, m12, m20, m01, out);
  generateRecursive(level, currentLevel + 1, m01, v1,  m12, out);
  generateRecursive(level, currentLevel + 1, m20, m12, v2,  out);
}

// Returns the barycentric (bU, bV) centroid for every micro-triangle at the
// given subdivision level, in the recursive Bird curve traversal order
// required by VK_EXT_opacity_micromap.
std::vector<std::array<float,2>> generateCentroids(uint32_t level) {
  std::vector<std::array<float,2>> out;
  out.reserve(microTriCount(level));
  generateRecursive(level, 0, {0,0}, {1,0}, {0,1}, out);
  assert(out.size() == static_cast<size_t>(microTriCount(level)));
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// packStates
//
// Converts unpacked states (one uint8_t per micro-triangle, values 0/1/3) into
// the 2-bits-per-entry format that vkBuildMicromapsEXT expects.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<uint8_t> packStates(const std::vector<uint8_t>& unpacked, uint32_t triangleCount, uint32_t subdivisionLevel) {
  const uint32_t microTrisPerTri = 1 << (2 * subdivisionLevel);
  const uint32_t bitsPerTri      = microTrisPerTri * 2; // 4-state format
  const uint32_t bytesPerTri     = (bitsPerTri + 7) / 8; // Each triangle's data is padded to the next byte if needed.

  std::vector<uint8_t> packed(triangleCount * bytesPerTri, 0);
  for (uint32_t t = 0; t < triangleCount; ++t) {
    for (uint32_t m = 0; m < microTrisPerTri; ++m) {
      const uint8_t  s         = unpacked[t * microTrisPerTri + m] & 0x3u;
      const uint32_t bitOffset = m * 2;
      packed[t * bytesPerTri + (bitOffset / 8)] |= static_cast<uint8_t>(s << (bitOffset % 8));
    }
  }
  return packed;
}

// ─────────────────────────────────────────────────────────────────────────────
// findMemType — local helper so we don't depend on MemoryPool internals
// ─────────────────────────────────────────────────────────────────────────────
uint32_t findMemType(const vk::raii::PhysicalDevice& physDev,
                     uint32_t filter,
                     vk::MemoryPropertyFlags flags) {
  const auto props = physDev.getMemoryProperties();
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    if ((filter & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  throw std::runtime_error("[OMM] No suitable Vulkan memory type found");
}

} // namespace


// =============================================================================
// Lifecycle
// =============================================================================

void OpacityMicromapBuilder::swap(OpacityMicromapBuilder& other) noexcept {
  if (this == &other) return;
  std::lock_guard<std::mutex> lock1(m_mutex);
  std::lock_guard<std::mutex> lock2(other.m_mutex);

  std::swap(m_initialised, other.m_initialised);
  std::swap(m_supported, other.m_supported);
  std::swap(m_renderer, other.m_renderer);
  std::swap(m_device, other.m_device);
  std::swap(m_physDev, other.m_physDev);
  std::swap(m_gfxFamily, other.m_gfxFamily);
  std::swap(m_entries, other.m_entries);
  std::swap(m_meshToEntry, other.m_meshToEntry);
  std::swap(m_infos, other.m_infos);
  std::swap(m_totalGpuBytes, other.m_totalGpuBytes);
}

void OpacityMicromapBuilder::init(Renderer& renderer) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_renderer   = &renderer;
  m_device     = &renderer.GetRaiiDevice();
  m_physDev    = &renderer.GetPhysicalDevice();
  m_commandPool = &renderer.GetCommandPool();
  m_gfxFamily  = renderer.GetGraphicsQueueFamilyIndex();
  m_supported  = renderer.GetOpacityMicromapEnabled();
  m_initialised = true;

  if (m_supported)
    std::cout << "[OMM] Initialised — VK_EXT_opacity_micromap is enabled.\n";
  else
    std::cout << "[OMM] VK_EXT_opacity_micromap not supported on this device; "
                 "alpha-tested shadows will use the any-hit shader path.\n";
}

void OpacityMicromapBuilder::reset() {
  std::lock_guard<std::mutex> lock(m_mutex);
  // RAII destructors in GpuEntry clean up all Vulkan objects automatically.
  m_entries.clear();
  m_infos.clear();
  m_meshToEntry.clear();
  m_totalGpuBytes = 0;
  m_initialised   = false;
}


// =============================================================================
// buildForMesh — the public entry point; runs all three phases
// =============================================================================
OmmMeshInfo OpacityMicromapBuilder::buildForMesh(const MeshComponent* mesh,
                                                  const uint8_t*       texPixels,
                                                  uint32_t             texW,
                                                  uint32_t             texH,
                                                  uint32_t             texChannels,
                                                  const OmmConfig&     config) {
  OmmMeshInfo result{};
  if (!m_initialised || !m_supported)            return result;
  if (!mesh || !texPixels || texW == 0 || texH == 0) return result;

  const auto& verts   = mesh->GetVertices();
  const auto& indices = mesh->GetIndices();
  if (verts.empty() || indices.empty() || indices.size() % 3 != 0) return result;

  // ── Phase 1: Does this mesh actually have alpha variation? ─────────────────
  if (!analyseVariation(verts, indices, texPixels, texW, texH, texChannels, config)) {
    std::cout << "[OMM] Skipping mesh — no meaningful alpha variation detected.\n";
    return result;
  }

  // ── Phase 2: Classify every micro-triangle ─────────────────────────────────
  std::vector<uint8_t> unpacked;
  classify(verts, indices, texPixels, texW, texH, texChannels, config, unpacked, result);

  // ── Phase 3: Build the GPU micromap ─────────────────────────────────────────
  const uint32_t triCount = static_cast<uint32_t>(indices.size() / 3);
  result.pNextChain = buildOnGpu(mesh, unpacked, triCount, config.subdivisionLevel, result);
  result.built      = (result.pNextChain != nullptr);

  if (result.built) {
    std::lock_guard<std::mutex> lock(m_mutex);
    // buildOnGpu already registered the mesh and updated m_entries while holding the lock.
    m_infos.push_back(result);
    m_totalGpuBytes += result.gpuBytes;

    std::cout << "[OMM] Micromap OK — triangles=" << triCount
              << "  opaque="      << static_cast<int>(result.pctOpaque      * 100.f) << "%"
              << "  transparent=" << static_cast<int>(result.pctTransparent * 100.f) << "%"
              << "  unknown="     << static_cast<int>(result.pctUnknown     * 100.f) << "%"
              << "  GPU=" << result.gpuBytes / 1024 << " KiB\n";
  }

  return result;
}

const OmmMeshInfo* OpacityMicromapBuilder::getInfo(const MeshComponent* mesh) const {
  std::lock_guard<std::mutex> lock(m_mutex);
  auto it = m_meshToEntry.find(mesh);
  if (it == m_meshToEntry.end()) return nullptr;
  return &m_infos[it->second];
}

uint32_t OpacityMicromapBuilder::micromapCount() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return static_cast<uint32_t>(m_entries.size());
}

uint64_t OpacityMicromapBuilder::totalGpuBytes() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_totalGpuBytes;
}


// =============================================================================
// Phase 1 — analyseVariation
//
// Samples the alpha channel at the centroid of ~10% of the triangles (up to
// 512).  Returns false if every sample was either fully opaque or fully
// transparent — in that case there is nothing useful to bake into a micromap.
// =============================================================================
bool OpacityMicromapBuilder::analyseVariation(
    const std::vector<Vertex>&   verts,
    const std::vector<uint32_t>& indices,
    const uint8_t*               pixels,
    uint32_t w, uint32_t h, uint32_t ch,
    const OmmConfig&             cfg) const {

  const uint32_t triCount   = static_cast<uint32_t>(indices.size() / 3);
  const uint32_t stride     = std::max(1u, triCount / std::min(triCount, 512u));
  bool foundOpaque = false, foundTransparent = false;

  for (uint32_t t = 0; t < triCount; t += stride) {
    const uint32_t i0 = indices[t*3+0], i1 = indices[t*3+1], i2 = indices[t*3+2];
    if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;

    const glm::vec2 cen = (verts[i0].texCoord + verts[i1].texCoord + verts[i2].texCoord) / 3.f;
    const float     a   = sampleAlpha(pixels, w, h, ch, cen.x, cen.y);

    if (a <  cfg.transparentThreshold) foundTransparent = true;
    if (a >= cfg.opaqueThreshold)      foundOpaque      = true;
    if (foundOpaque && foundTransparent) return true;
  }
  return false;
}


// =============================================================================
// Phase 2 — classify
//
// For every micro-triangle in every source triangle, sample the alpha texture
// at the centroid (and a few nearby jitter points) and assign one of the three
// 2-bit opacity states.  Results go into `outStates` (one uint8_t per entry,
// unpacked), ready for packStates().
// =============================================================================
void OpacityMicromapBuilder::classify(
    const std::vector<Vertex>&   verts,
    const std::vector<uint32_t>& indices,
    const uint8_t*               pixels,
    uint32_t w, uint32_t h, uint32_t ch,
    const OmmConfig&             cfg,
    std::vector<uint8_t>&        outStates,
    OmmMeshInfo&                 outInfo) const {

  const uint32_t triCount   = static_cast<uint32_t>(indices.size() / 3);
  const uint32_t uPerTri    = microTriCount(cfg.subdivisionLevel);
  outStates.assign(static_cast<size_t>(triCount) * uPerTri, STATE_UNKNOWN_OPAQUE);

  const auto centroids = generateCentroids(cfg.subdivisionLevel);
  assert(centroids.size() == static_cast<size_t>(uPerTri));

  std::atomic<uint32_t> nOpaque{0}, nTrans{0}, nUnknown{0};

  // ── Parallel Classification ────────────────────────────────────────────────
  // We use the renderer's thread pool to process chunks of triangles.
  // This prevents the engine from hanging during a rebuild of a large scene.
  const uint32_t numThreads = std::thread::hardware_concurrency();
  const uint32_t chunkSize  = std::max(1u, triCount / (numThreads * 4));

  std::vector<std::future<void>> futures;

  auto processTriangles = [&](uint32_t startTri, uint32_t endTri) {
    uint32_t localOpaque = 0, localTrans = 0, localUnknown = 0;

    for (uint32_t t = startTri; t < endTri; ++t) {
      const uint32_t i0 = indices[t*3+0], i1 = indices[t*3+1], i2 = indices[t*3+2];

      if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) {
        localUnknown += uPerTri;
        continue; // states already initialized to UNKNOWN
      }

      const glm::vec2 uv0 = verts[i0].texCoord;
      const glm::vec2 uv1 = verts[i1].texCoord;
      const glm::vec2 uv2 = verts[i2].texCoord;

      for (uint32_t m = 0; m < uPerTri; ++m) {
        const auto& cen = centroids[m];
        const float bU = cen[0], bV = cen[1], bW = 1.f - bU - bV;

        float alphaSum = 0.f;
        for (uint32_t s = 0; s < cfg.samplesPerMicroTriangle; ++s) {
          const float jU = bU + (s == 1 ? 0.04f : 0.f) - (s == 2 ? 0.02f : 0.f) - (s == 3 ? 0.02f : 0.f);
          const float jV = bV + (s == 2 ? 0.04f : 0.f) - (s == 1 ? 0.02f : 0.f) - (s == 3 ? 0.02f : 0.f);
          const float jW = std::max(0.f, 1.f - jU - jV);
          const glm::vec2 uv = jW * uv0 + jU * uv1 + jV * uv2;
          alphaSum += sampleAlpha(pixels, w, h, ch, uv.x, uv.y);
        }
        const float avg = alphaSum / static_cast<float>(cfg.samplesPerMicroTriangle);

        uint8_t state;
        if      (avg <  cfg.transparentThreshold) { state = STATE_TRANSPARENT;   ++localTrans;   }
        else if (avg >= cfg.opaqueThreshold)      { state = STATE_OPAQUE;        ++localOpaque;  }
        else if (cfg.allowUnknownState)           { state = STATE_UNKNOWN_OPAQUE; ++localUnknown; }
        else                                      { state = STATE_OPAQUE;        ++localOpaque;  }

        outStates[static_cast<size_t>(t) * uPerTri + m] = state;
      }
    }
    nOpaque   += localOpaque;
    nTrans    += localTrans;
    nUnknown  += localUnknown;
  };

  if (m_renderer && m_renderer->GetThreadPool()) {
    for (uint32_t t = 0; t < triCount; t += chunkSize) {
      uint32_t end = std::min(t + chunkSize, triCount);
      futures.push_back(m_renderer->GetThreadPool()->enqueue(processTriangles, t, end));
    }

    // Wait and keep the watchdog alive
    for (auto& f : futures) {
      while (f.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
        if (m_renderer) {
          m_renderer->KickWatchdog("OMM classify (parallel)");
        }
      }
    }
  } else {
    processTriangles(0, triCount);
  }

  const uint32_t total  = triCount * uPerTri;
  outInfo.totalMicroTris = total;
  if (total > 0) {
    const float inv         = 1.f / static_cast<float>(total);
    outInfo.pctOpaque       = static_cast<float>(nOpaque.load())  * inv;
    outInfo.pctTransparent  = static_cast<float>(nTrans.load())   * inv;
    outInfo.pctUnknown      = static_cast<float>(nUnknown.load()) * inv;
  }
}


// =============================================================================
// Phase 3 — buildOnGpu
//
// Uploads the packed state data to a device-local buffer, creates the
// VkMicromapEXT, runs vkBuildMicromapsEXT, then populates the pNext chain
// struct that will be handed to the BLAS build.
//
// All GPU resources are stored in a GpuEntry and kept alive until reset().
// =============================================================================
void* OpacityMicromapBuilder::buildOnGpu(
    const MeshComponent* mesh,
    const std::vector<uint8_t>& unpackedStates,
    uint32_t triangleCount,
    uint32_t subdivisionLevel,
    OmmMeshInfo& outInfo) {

  const auto& dev = *m_device;

  // ── Pack 2 bits per state ─────────────────────────────────────────────────
  const std::vector<uint8_t> packed   = packStates(unpackedStates, triangleCount, subdivisionLevel);
  const vk::DeviceSize       dataSize = static_cast<vk::DeviceSize>(packed.size());

  // ── Upload state data to device-local buffer (one-shot CB) ───────────────
  // We create a dedicated transient command pool for this build to ensure thread-safety,
  // as this may be called from a background thread while the main thread uses the global pool.
  vk::raii::CommandPool localPool(dev, vk::CommandPoolCreateInfo{
    .flags = vk::CommandPoolCreateFlagBits::eTransient,
    .queueFamilyIndex = m_gfxFamily
  });
  auto cb = beginOneShot(localPool);

  // Staging buffer (host-visible, coherent)
  vk::BufferCreateInfo stagCI{
    .size = dataSize,
    .usage = vk::BufferUsageFlagBits::eTransferSrc
  };
  vk::raii::Buffer     stagBuf(dev, stagCI);
  auto stagReqs = stagBuf.getMemoryRequirements();
  vk::MemoryAllocateInfo stagAlloc{
    .allocationSize  = stagReqs.size,
    .memoryTypeIndex = findMemType(*m_physDev, stagReqs.memoryTypeBits,
                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent)
  };
  vk::raii::DeviceMemory stagMem(dev, stagAlloc);
  stagBuf.bindMemory(*stagMem, 0);
  {
    void* p = stagMem.mapMemory(0, dataSize);
    std::memcpy(p, packed.data(), static_cast<size_t>(dataSize));
    stagMem.unmapMemory();
  }

  // Device-local data buffer
  vk::BufferCreateInfo dataCI{
    .size  = dataSize,
    .usage = vk::BufferUsageFlagBits::eTransferDst
           | vk::BufferUsageFlagBits::eShaderDeviceAddress
           | vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT
  };
  vk::raii::Buffer     dataBuf(dev, dataCI);
  auto dataReqs = dataBuf.getMemoryRequirements();
  vk::MemoryAllocateFlagsInfo dataFlags{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
  vk::MemoryAllocateInfo dataAlloc{
    .pNext           = &dataFlags,
    .allocationSize  = dataReqs.size,
    .memoryTypeIndex = findMemType(*m_physDev, dataReqs.memoryTypeBits,
                                   vk::MemoryPropertyFlagBits::eDeviceLocal)
  };
  vk::raii::DeviceMemory dataMem(dev, dataAlloc);
  dataBuf.bindMemory(*dataMem, 0);

  // Copy staging → device
  cb.copyBuffer(*stagBuf, *dataBuf, vk::BufferCopy{ .size = dataSize });

  // Barrier: transfer write → micromap build read
  vk::BufferMemoryBarrier2 dataBarr{
    .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
    .dstStageMask  = vk::PipelineStageFlagBits2::eMicromapBuildEXT,
    .dstAccessMask = vk::AccessFlagBits2::eMicromapReadEXT,
    .buffer = *dataBuf, .offset = 0, .size = dataSize
  };
  cb.pipelineBarrier2(vk::DependencyInfo{
    .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &dataBarr
  });

  // ── Usage entry describes triangleCount micro-triangle sets ───────────────
  // We use 4-state format (2 bits per micro-triangle) to support Unknown states.
  const auto mmFormat = vk::OpacityMicromapFormatEXT::e4State;

  vk::MicromapUsageEXT usage{
    .count            = triangleCount,
    .subdivisionLevel = subdivisionLevel,
    .format           = static_cast<uint32_t>(mmFormat)
  };

  // ── Query build sizes ─────────────────────────────────────────────────────
  vk::MicromapBuildInfoEXT buildInfo{
    .type             = vk::MicromapTypeEXT::eOpacityMicromap,
    .flags            = {},
    .mode             = vk::BuildMicromapModeEXT::eBuild,
    .usageCountsCount = 1,
    .pUsageCounts     = &usage
  };
  const auto sizes = dev.getMicromapBuildSizesEXT(
      vk::AccelerationStructureBuildTypeKHR::eDevice, buildInfo);

  // ── Micromap storage buffer ───────────────────────────────────────────────
  vk::BufferCreateInfo mmBufCI{
    .size  = sizes.micromapSize,
    .usage = vk::BufferUsageFlagBits::eMicromapStorageEXT
           | vk::BufferUsageFlagBits::eShaderDeviceAddress
  };
  vk::raii::Buffer     mmBuf(dev, mmBufCI);
  auto mmReqs = mmBuf.getMemoryRequirements();
  vk::MemoryAllocateFlagsInfo mmFlags{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
  vk::MemoryAllocateInfo mmAlloc{
    .pNext           = &mmFlags,
    .allocationSize  = mmReqs.size,
    .memoryTypeIndex = findMemType(*m_physDev, mmReqs.memoryTypeBits,
                                   vk::MemoryPropertyFlagBits::eDeviceLocal)
  };
  vk::raii::DeviceMemory mmMem(dev, mmAlloc);
  mmBuf.bindMemory(*mmMem, 0);

  // ── Create VkMicromapEXT ──────────────────────────────────────────────────
  vk::MicromapCreateInfoEXT mmCI{
    .buffer = *mmBuf,
    .offset = 0,
    .size   = sizes.micromapSize,
    .type   = vk::MicromapTypeEXT::eOpacityMicromap
  };
  vk::raii::MicromapEXT micromap = dev.createMicromapEXT(mmCI);

  // ── Scratch buffer for the build ──────────────────────────────────────────
  vk::BufferCreateInfo scratchCI{
    .size  = sizes.buildScratchSize,
    .usage = vk::BufferUsageFlagBits::eStorageBuffer
           | vk::BufferUsageFlagBits::eShaderDeviceAddress
  };
  vk::raii::Buffer     scratchBuf(dev, scratchCI);
  auto scratchReqs = scratchBuf.getMemoryRequirements();
  vk::MemoryAllocateFlagsInfo scratchFlags{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
  vk::MemoryAllocateInfo scratchAlloc{
    .pNext           = &scratchFlags,
    .allocationSize  = scratchReqs.size,
    .memoryTypeIndex = findMemType(*m_physDev, scratchReqs.memoryTypeBits,
                                   vk::MemoryPropertyFlagBits::eDeviceLocal)
  };
  vk::raii::DeviceMemory scratchMem(dev, scratchAlloc);
  scratchBuf.bindMemory(*scratchMem, 0);

  const vk::DeviceAddress dataAddr    = dev.getBufferAddress({ .buffer = *dataBuf    });
  const vk::DeviceAddress scratchAddr = dev.getBufferAddress({ .buffer = *scratchBuf });

  // ── Prepare triangleArray (tells driver where each triangle's data is) ──
  // For 4-state format, each triangle uses (4^L * 2) bits.
  // For L >= 1, this is always a multiple of 8 bits (1 byte), so triangles are byte-aligned.
  const uint32_t microTrisPerTri = 1 << (2 * subdivisionLevel);
  const uint32_t bitsPerTri     = microTrisPerTri * 2; // 4-state format (2 bits per micro-triangle)
  const uint32_t bytesPerTri    = (bitsPerTri + 7) / 8; // round up to byte boundary

  std::vector<VkMicromapTriangleEXT> triMapping(triangleCount);
  for (uint32_t i = 0; i < triangleCount; ++i) {
    triMapping[i].dataOffset       = i * bytesPerTri;
    triMapping[i].subdivisionLevel = static_cast<uint16_t>(subdivisionLevel);
    triMapping[i].format           = static_cast<uint16_t>(mmFormat);
  }

  vk::DeviceSize triMapSize = triMapping.size() * sizeof(VkMicromapTriangleEXT);

  vk::BufferCreateInfo triStagCI{ .size = triMapSize, .usage = vk::BufferUsageFlagBits::eTransferSrc };
  vk::raii::Buffer     triStagBuf(dev, triStagCI);
  auto triStagReqs = triStagBuf.getMemoryRequirements();
  vk::MemoryAllocateInfo triStagAlloc{
    .allocationSize  = triStagReqs.size,
    .memoryTypeIndex = findMemType(*m_physDev, triStagReqs.memoryTypeBits,
                                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
  };
  vk::raii::DeviceMemory triStagMem(dev, triStagAlloc);
  triStagBuf.bindMemory(*triStagMem, 0);
  {
    void* p = triStagMem.mapMemory(0, triMapSize);
    std::memcpy(p, triMapping.data(), static_cast<size_t>(triMapSize));
    triStagMem.unmapMemory();
  }

  vk::BufferCreateInfo triCI{
    .size  = triMapSize,
    .usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eMicromapBuildInputReadOnlyEXT
  };
  vk::raii::Buffer       triBuf(dev, triCI);
  auto triReqs = triBuf.getMemoryRequirements();
  vk::MemoryAllocateFlagsInfo triFlags{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
  vk::MemoryAllocateInfo triAlloc{
    .pNext           = &triFlags,
    .allocationSize  = triReqs.size,
    .memoryTypeIndex = findMemType(*m_physDev, triReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
  };
  vk::raii::DeviceMemory triMem(dev, triAlloc);
  triBuf.bindMemory(*triMem, 0);

  cb.copyBuffer(*triStagBuf, *triBuf, vk::BufferCopy{ .size = triMapSize });

  vk::BufferMemoryBarrier2 triBarr{
    .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
    .dstStageMask  = vk::PipelineStageFlagBits2::eMicromapBuildEXT,
    .dstAccessMask = vk::AccessFlagBits2::eMicromapReadEXT,
    .buffer = *triBuf, .offset = 0, .size = triMapSize
  };
  cb.pipelineBarrier2(vk::DependencyInfo{ .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &triBarr });

  const vk::DeviceAddress triAddr = dev.getBufferAddress({ .buffer = *triBuf });

  // ── Record the vkBuildMicromapsEXT command ────────────────────────────────
  buildInfo.dstMicromap                 = *micromap;
  buildInfo.data.deviceAddress          = dataAddr;
  buildInfo.triangleArray.deviceAddress = triAddr;
  buildInfo.triangleArrayStride         = sizeof(VkMicromapTriangleEXT);
  buildInfo.scratchData.deviceAddress   = scratchAddr;

  cb.buildMicromapsEXT(buildInfo);

  // Submit, wait for completion, then discard scratch & staging buffers.
  submitOneShot(cb);

  // ── Build the pNext chain struct ──────────────────────────────────────────
  // This struct is referenced by the BLAS build and MUST outlive the BLAS.
  // We heap-allocate it inside GpuEntry::PNextStorage and never move it.
  auto pNextOwner = std::make_unique<GpuEntry::PNextStorage>();
  pNextOwner->usageEntry = usage;

  auto& chain = pNextOwner->chain;
  chain.sType = vk::StructureType::eAccelerationStructureTrianglesOpacityMicromapEXT;
  chain.pNext = nullptr;

  chain.usageCountsCount = 1;
  chain.pUsageCounts     = &pNextOwner->usageEntry;
  chain.micromap         = *micromap;

  void* pNextPtr = &chain;

  outInfo.gpuBytes = static_cast<uint64_t>(sizes.micromapSize) + dataSize + triMapSize;

  // ── Store all GPU resources in a GpuEntry ────────────────────────────────
  GpuEntry ge;
  ge.dataBuf    = std::move(dataBuf);
  ge.dataMem    = std::move(dataMem);
  ge.mmBuf      = std::move(mmBuf);
  ge.mmMem      = std::move(mmMem);
  ge.triBuf     = std::move(triBuf);
  ge.triMem     = std::move(triMem);
  ge.micromap   = std::move(micromap);
  ge.pNextOwner = std::move(pNextOwner);

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_meshToEntry[mesh] = m_entries.size();
    m_entries.push_back(std::move(ge));
  }

  return pNextPtr;
}


// =============================================================================
// Helpers
// =============================================================================

float OpacityMicromapBuilder::sampleAlpha(const uint8_t* pixels,
                                           uint32_t w, uint32_t h, uint32_t ch,
                                           float u, float v) const {
  // Repeat-wrap the UV coordinates.
  u -= std::floor(u);
  v -= std::floor(v);
  const uint32_t px  = std::min(static_cast<uint32_t>(u * static_cast<float>(w)), w - 1);
  const uint32_t py  = std::min(static_cast<uint32_t>(v * static_cast<float>(h)), h - 1);
  const size_t   off = (static_cast<size_t>(py) * w + px) * ch;

  if (ch == 1) return static_cast<float>(pixels[off])     / 255.f;
  if (ch == 2) return static_cast<float>(pixels[off + 1]) / 255.f;
  if (ch >= 4) return static_cast<float>(pixels[off + 3]) / 255.f;
  return 1.f; // RGB — treat as fully opaque (no alpha channel)
}

vk::raii::CommandBuffer OpacityMicromapBuilder::beginOneShot(vk::raii::CommandPool& pool) const {
  vk::CommandBufferAllocateInfo allocInfo{
    .commandPool        = *pool,
    .level              = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = 1
  };
  auto bufs = m_device->allocateCommandBuffers(allocInfo);
  auto cb   = std::move(bufs[0]);
  cb.begin(vk::CommandBufferBeginInfo{
    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
  });
  return cb;
}

void OpacityMicromapBuilder::submitOneShot(vk::raii::CommandBuffer& cb) const {
  cb.end();
  vk::raii::Fence fence(*m_device, vk::FenceCreateInfo{});
  m_renderer->SubmitToGraphicsQueue(*cb, *fence);
  auto _ = m_device->waitForFences(*fence, vk::True, UINT64_MAX);
  (void)_;
}
