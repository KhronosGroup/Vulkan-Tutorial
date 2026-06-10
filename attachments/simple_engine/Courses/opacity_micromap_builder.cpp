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
// Read alongside:
//   en/Building_a_Simple_Engine/Courses/Opacity_Micromaps/05_implementation_overview.adoc
//
// This module implements VK_KHR_opacity_micromap.
//
// THREE PHASES
// ------------
//   Phase 1  analyseVariation()   Does this mesh have mixed opaque/transparent
//                                 regions? If not, skip the build entirely.
//   Phase 2  classify()           Sample each micro-triangle centroid in
//                                 texture space → assign a 2-bit opacity state.
//   Phase 3  buildOnGpu()         Pack states, upload to device, create a
//                                 VkAccelerationStructureKHR (OPACITY_MICROMAP
//                                 type) via vkCreateAccelerationStructure2KHR,
//                                 build it with vkCmdBuildAccelerationStructuresKHR,
//                                 then fill the pNext chain.
//
// KHR DESIGN NOTES
// ----------------
// - Micromaps are VkAccelerationStructureKHR with type
//   VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR. They are created via
//   vkCreateAccelerationStructure2KHR (from VK_KHR_device_address_commands).
//   vkCreateAccelerationStructureKHR cannot be used for micromaps.
//
// - Build is device-side only via vkCmdBuildAccelerationStructuresKHR with
//   geometryType = VK_GEOMETRY_TYPE_MICROMAP_KHR. No host build path exists.
//
// - Size query is vkGetAccelerationStructureBuildSizesKHR with buildType =
//   VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR.
//   pMaxPrimitiveCounts[i] is the max triangle count for geometry[i].
//   The raw C dispatcher is used so we can directly control the call.
//
// - Input buffers require VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR.
//   The micromap backing buffer requires VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR.
//
// - Synchronisation uses VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR.
//
// - The BLAS attachment is VkAccelerationStructureTrianglesOpacityMicromapKHR
//   chained into pNext of VkAccelerationStructureGeometryTrianglesDataKHR.
//   Its indexBuffer is VkDeviceAddress (device-only, no host address variant).
//   Its micromap field is VkAccelerationStructureKHR.
//
// - The BLAS always holds a live reference to the micromap. There is no
//   "discardable" property (removed from KHR). Destroy BLASes before micromaps.
//
// - Ray query shaders must declare the OpacityMicromapKHR execution mode via
//   SPV_KHR_opacity_micromap for the hardware fast-path to activate.
//
// STATE ENCODING (VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR, 2 bits/entry)
// ----------------------------------------------------------------------
//   0b00  TRANSPARENT    Hardware passes the ray — no shader.
//   0b01  OPAQUE         Hardware blocks the ray — no shader.
//   0b11  UNKNOWN_OPAQUE Hardware falls back to the any-hit shader.
// =============================================================================

#include "opacity_micromap_builder.h"
#include "../renderer.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {

constexpr uint8_t STATE_TRANSPARENT    = 0b00;
constexpr uint8_t STATE_OPAQUE         = 0b01;
constexpr uint8_t STATE_UNKNOWN_OPAQUE = 0b11;

constexpr uint32_t microTriCount(uint32_t level) {
  uint32_t n = 1;
  for (uint32_t i = 0; i < level; ++i) n *= 4;
  return n;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bird-curve centroid generation
//
// Produces barycentric centroids in the traversal order required by the spec.
// Both KHR and EXT use the same recursive Bird-curve layout.
// ─────────────────────────────────────────────────────────────────────────────
void generateRecursive(uint32_t targetLevel, uint32_t currentLevel,
                       glm::vec2 v0, glm::vec2 v1, glm::vec2 v2,
                       std::vector<std::array<float, 2>>& out)
{
  if (currentLevel == targetLevel) {
    const glm::vec2 c = (v0 + v1 + v2) / 3.0f;
    out.push_back({c.x, c.y});
    return;
  }
  const glm::vec2 m01 = (v0 + v1) * 0.5f;
  const glm::vec2 m12 = (v1 + v2) * 0.5f;
  const glm::vec2 m20 = (v2 + v0) * 0.5f;
  generateRecursive(targetLevel, currentLevel + 1, v0,  m01, m20, out);
  generateRecursive(targetLevel, currentLevel + 1, m12, m20, m01, out);
  generateRecursive(targetLevel, currentLevel + 1, m01, v1,  m12, out);
  generateRecursive(targetLevel, currentLevel + 1, m20, m12, v2,  out);
}

std::vector<std::array<float, 2>> generateCentroids(uint32_t level) {
  std::vector<std::array<float, 2>> out;
  out.reserve(microTriCount(level));
  generateRecursive(level, 0, {0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, out);
  assert(out.size() == static_cast<size_t>(microTriCount(level)));
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// packStates
//
// Converts one uint8_t per micro-triangle into the 2-bits-per-entry layout
// consumed by vkCmdBuildAccelerationStructuresKHR.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<uint8_t> packStates(const std::vector<uint8_t>& unpacked,
                                uint32_t triangleCount,
                                uint32_t subdivisionLevel)
{
  const uint32_t uPerTri     = microTriCount(subdivisionLevel);
  const uint32_t bitsPerTri  = uPerTri * 2u;
  const uint32_t bytesPerTri = (bitsPerTri + 7u) / 8u;

  std::vector<uint8_t> packed(static_cast<size_t>(triangleCount) * bytesPerTri, 0u);
  for (uint32_t t = 0; t < triangleCount; ++t) {
    for (uint32_t m = 0; m < uPerTri; ++m) {
      const uint8_t  s   = unpacked[t * uPerTri + m] & 0x3u;
      const uint32_t bit = m * 2u;
      packed[static_cast<size_t>(t) * bytesPerTri + bit / 8u] |=
          static_cast<uint8_t>(s << (bit % 8u));
    }
  }
  return packed;
}

uint32_t findMemType(const vk::raii::PhysicalDevice& pd,
                     uint32_t                        filter,
                     vk::MemoryPropertyFlags         flags)
{
  const auto props = pd.getMemoryProperties();
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
    if ((filter & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  throw std::runtime_error("[OMM] No suitable memory type found");
}

// Allocate a device-local buffer with the given usage and return {buffer, memory}.
struct BufMem { vk::raii::Buffer buf; vk::raii::DeviceMemory mem; };

BufMem makeDeviceBuffer(const vk::raii::Device&    dev,
                        const vk::raii::PhysicalDevice& pd,
                        vk::DeviceSize             size,
                        vk::BufferUsageFlags       usage)
{
  vk::raii::Buffer buf(dev, vk::BufferCreateInfo{
    .size  = size,
    .usage = usage
  });
  auto reqs = buf.getMemoryRequirements();
  vk::MemoryAllocateFlagsInfo flagsInfo{ .flags = vk::MemoryAllocateFlagBits::eDeviceAddress };
  vk::raii::DeviceMemory mem(dev, vk::MemoryAllocateInfo{
    .pNext           = &flagsInfo,
    .allocationSize  = reqs.size,
    .memoryTypeIndex = findMemType(pd, reqs.memoryTypeBits,
                                   vk::MemoryPropertyFlagBits::eDeviceLocal)
  });
  buf.bindMemory(*mem, 0);
  return { std::move(buf), std::move(mem) };
}

BufMem makeStagingBuffer(const vk::raii::Device&    dev,
                         const vk::raii::PhysicalDevice& pd,
                         const void*                data,
                         vk::DeviceSize             size)
{
  vk::raii::Buffer buf(dev, vk::BufferCreateInfo{
    .size  = size,
    .usage = vk::BufferUsageFlagBits::eTransferSrc
  });
  auto reqs = buf.getMemoryRequirements();
  vk::raii::DeviceMemory mem(dev, vk::MemoryAllocateInfo{
    .allocationSize  = reqs.size,
    .memoryTypeIndex = findMemType(pd, reqs.memoryTypeBits,
                                   vk::MemoryPropertyFlagBits::eHostVisible |
                                   vk::MemoryPropertyFlagBits::eHostCoherent)
  });
  buf.bindMemory(*mem, 0);
  void* p = mem.mapMemory(0, size);
  std::memcpy(p, data, static_cast<size_t>(size));
  mem.unmapMemory();
  return { std::move(buf), std::move(mem) };
}

} // namespace


// =============================================================================
// Lifecycle
// =============================================================================

void OpacityMicromapBuilder::swap(OpacityMicromapBuilder& other) noexcept {
  if (this == &other) return;
  std::lock_guard<std::mutex> l1(m_mutex);
  std::lock_guard<std::mutex> l2(other.m_mutex);
  std::swap(m_initialised,  other.m_initialised);
  std::swap(m_supported,    other.m_supported);
  std::swap(m_renderer,     other.m_renderer);
  std::swap(m_device,       other.m_device);
  std::swap(m_physDev,      other.m_physDev);
  std::swap(m_gfxFamily,    other.m_gfxFamily);
  std::swap(m_entries,      other.m_entries);
  std::swap(m_meshToEntry,  other.m_meshToEntry);
  std::swap(m_infos,        other.m_infos);
  std::swap(m_totalGpuBytes, other.m_totalGpuBytes);
}

void OpacityMicromapBuilder::init(Renderer& renderer) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_renderer    = &renderer;
  m_device      = &renderer.GetRaiiDevice();
  m_physDev     = &renderer.GetPhysicalDevice();
  m_gfxFamily   = renderer.GetGraphicsQueueFamilyIndex();
  m_supported   = renderer.GetOpacityMicromapEnabled();
  m_initialised = true;

  if (m_supported)
    std::cout << "[OMM] Initialised — VK_KHR_opacity_micromap is enabled.\n";
  else
    std::cout << "[OMM] VK_KHR_opacity_micromap not supported on this device; "
                 "alpha-tested shadows will use the any-hit shader path.\n";
}

void OpacityMicromapBuilder::reset() {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_entries.clear();
  m_infos.clear();
  m_meshToEntry.clear();
  m_totalGpuBytes = 0;
  m_initialised   = false;
}


// =============================================================================
// buildForMesh — public entry point; orchestrates all three phases
// =============================================================================

OmmMeshInfo OpacityMicromapBuilder::buildForMesh(const MeshComponent* mesh,
                                                  const uint8_t*       texPixels,
                                                  uint32_t             texW,
                                                  uint32_t             texH,
                                                  uint32_t             texChannels,
                                                  const OmmConfig&     config)
{
  OmmMeshInfo result{};
  if (!m_initialised || !m_supported)                          return result;
  if (!mesh || !texPixels || texW == 0 || texH == 0)           return result;

  const auto& verts   = mesh->GetVertices();
  const auto& indices = mesh->GetIndices();
  if (verts.empty() || indices.empty() || indices.size() % 3 != 0) return result;

  // Phase 1
  if (!analyseVariation(verts, indices, texPixels, texW, texH, texChannels, config)) {
    std::cout << "[OMM] Skipping mesh — no meaningful alpha variation detected.\n";
    return result;
  }

  // Phase 2
  std::vector<uint8_t> unpacked;
  classify(verts, indices, texPixels, texW, texH, texChannels, config, unpacked, result);

  // Phase 3
  const uint32_t triCount = static_cast<uint32_t>(indices.size() / 3);
  result.pNextChain = buildOnGpu(mesh, unpacked, triCount,
                                  config.subdivisionLevel, config.lossyBuild, result);
  result.built = (result.pNextChain != nullptr);

  if (result.built) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_infos.push_back(result);
    m_totalGpuBytes += result.gpuBytes;

    std::cout << "[OMM] Built — tris=" << triCount
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
  return (it == m_meshToEntry.end()) ? nullptr : &m_infos[it->second];
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
// =============================================================================

bool OpacityMicromapBuilder::analyseVariation(
    const std::vector<Vertex>&   verts,
    const std::vector<uint32_t>& indices,
    const uint8_t* pixels, uint32_t w, uint32_t h, uint32_t ch,
    const OmmConfig& cfg) const
{
  const uint32_t triCount = static_cast<uint32_t>(indices.size() / 3);
  const uint32_t stride   = std::max(1u, triCount / std::min(triCount, 512u));
  bool foundOpaque = false, foundTransparent = false;

  for (uint32_t t = 0; t < triCount; t += stride) {
    const uint32_t i0 = indices[t*3+0], i1 = indices[t*3+1], i2 = indices[t*3+2];
    if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;
    const glm::vec2 cen = (verts[i0].texCoord + verts[i1].texCoord + verts[i2].texCoord) / 3.f;
    const float a = sampleAlpha(pixels, w, h, ch, cen.x, cen.y);
    if (a <  cfg.transparentThreshold) foundTransparent = true;
    if (a >= cfg.opaqueThreshold)      foundOpaque      = true;
    if (foundOpaque && foundTransparent) return true;
  }
  return false;
}


// =============================================================================
// Phase 2 — classify
// =============================================================================

void OpacityMicromapBuilder::classify(
    const std::vector<Vertex>&   verts,
    const std::vector<uint32_t>& indices,
    const uint8_t* pixels, uint32_t w, uint32_t h, uint32_t ch,
    const OmmConfig& cfg,
    std::vector<uint8_t>& outStates,
    OmmMeshInfo& outInfo) const
{
  const uint32_t triCount = static_cast<uint32_t>(indices.size() / 3);
  const uint32_t uPerTri  = microTriCount(cfg.subdivisionLevel);
  outStates.assign(static_cast<size_t>(triCount) * uPerTri, STATE_UNKNOWN_OPAQUE);

  const auto centroids = generateCentroids(cfg.subdivisionLevel);

  std::atomic<uint32_t> nOpaque{0}, nTrans{0}, nUnknown{0};

  const uint32_t numThreads = std::thread::hardware_concurrency();
  const uint32_t chunkSize  = std::max(1u, triCount / (numThreads * 4u));
  std::vector<std::future<void>> futures;

  auto processTriangles = [&](uint32_t startTri, uint32_t endTri) {
    uint32_t lO = 0, lT = 0, lU = 0;
    for (uint32_t t = startTri; t < endTri; ++t) {
      const uint32_t i0 = indices[t*3+0], i1 = indices[t*3+1], i2 = indices[t*3+2];
      if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) {
        lU += uPerTri; continue;
      }
      const glm::vec2 uv0 = verts[i0].texCoord;
      const glm::vec2 uv1 = verts[i1].texCoord;
      const glm::vec2 uv2 = verts[i2].texCoord;

      for (uint32_t m = 0; m < uPerTri; ++m) {
        const float bU = centroids[m][0], bV = centroids[m][1];
        float alphaSum = 0.f;
        for (uint32_t s = 0; s < cfg.samplesPerMicroTriangle; ++s) {
          const float jU = bU + (s==1 ? 0.04f:0.f) - (s==2 ? 0.02f:0.f) - (s==3 ? 0.02f:0.f);
          const float jV = bV + (s==2 ? 0.04f:0.f) - (s==1 ? 0.02f:0.f) - (s==3 ? 0.02f:0.f);
          const float jW = std::max(0.f, 1.f - jU - jV);
          const glm::vec2 uv = jW * uv0 + jU * uv1 + jV * uv2;
          alphaSum += sampleAlpha(pixels, w, h, ch, uv.x, uv.y);
        }
        const float avg = alphaSum / static_cast<float>(cfg.samplesPerMicroTriangle);

        uint8_t state;
        if      (avg <  cfg.transparentThreshold) { state = STATE_TRANSPARENT;    ++lT; }
        else if (avg >= cfg.opaqueThreshold)       { state = STATE_OPAQUE;         ++lO; }
        else if (cfg.allowUnknownState)            { state = STATE_UNKNOWN_OPAQUE; ++lU; }
        else                                       { state = STATE_OPAQUE;         ++lO; }

        outStates[static_cast<size_t>(t) * uPerTri + m] = state;
      }
    }
    nOpaque += lO; nTrans += lT; nUnknown += lU;
  };

  if (m_renderer && m_renderer->GetThreadPool()) {
    for (uint32_t t = 0; t < triCount; t += chunkSize) {
      futures.push_back(m_renderer->GetThreadPool()->enqueue(
          processTriangles, t, std::min(t + chunkSize, triCount)));
    }
    for (auto& f : futures) {
      while (f.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
        if (m_renderer) m_renderer->KickWatchdog("OMM classify");
    }
  } else {
    processTriangles(0, triCount);
  }

  const uint32_t total   = triCount * uPerTri;
  outInfo.totalMicroTris  = total;
  if (total > 0) {
    const float inv        = 1.f / static_cast<float>(total);
    outInfo.pctOpaque      = static_cast<float>(nOpaque.load())  * inv;
    outInfo.pctTransparent = static_cast<float>(nTrans.load())   * inv;
    outInfo.pctUnknown     = static_cast<float>(nUnknown.load()) * inv;
  }
}


// =============================================================================
// Phase 3 — buildOnGpu
//
// KHR micromap build path (VK_KHR_opacity_micromap):
//   1. Upload packed state data and VkMicromapTriangleKHR array to device-local
//      buffers via staging copies.
//   2. Fill VkAccelerationStructureGeometryMicromapDataKHR (chained via pNext of
//      VkAccelerationStructureGeometryKHR with geometryType=eMicromap).
//   3. Query build sizes via vkGetAccelerationStructureBuildSizesKHR.
//   4. Allocate storage buffer with eAccelerationStructureStorageKHR.
//   5. Create VkAccelerationStructureKHR (type=eOpacityMicromap) via
//      vkCreateAccelerationStructure2KHR (VK_KHR_device_address_commands).
//   6. Fill device addresses, record vkCmdBuildAccelerationStructuresKHR,
//      submit and wait.
//   7. Fill VkAccelerationStructureTrianglesOpacityMicromapKHR for the BLAS
//      pNext chain.
// =============================================================================

void* OpacityMicromapBuilder::buildOnGpu(
    const MeshComponent*        mesh,
    const std::vector<uint8_t>& unpackedStates,
    uint32_t                    triangleCount,
    uint32_t                    subdivisionLevel,
    bool                        lossyBuild,
    OmmMeshInfo&                outInfo)
{
#if defined(PLATFORM_ANDROID)
  return nullptr; // OMM build is not supported on Android yet due to missing KHR symbols
#else
  const auto& dev = *m_device;

  // ── Pack state data ────────────────────────────────────────────────────────
  const std::vector<uint8_t> packed = packStates(unpackedStates, triangleCount, subdivisionLevel);
  const vk::DeviceSize dataSize     = static_cast<vk::DeviceSize>(packed.size());

  // ── Build VkMicromapTriangleKHR array ─────────────────────────────────────
  const uint32_t uPerTri     = microTriCount(subdivisionLevel);
  const uint32_t bytesPerTri = (uPerTri * 2u + 7u) / 8u;

  std::vector<VkMicromapTriangleKHR> triArray(triangleCount);
  for (uint32_t i = 0; i < triangleCount; ++i) {
    triArray[i].dataOffset       = i * bytesPerTri;
    triArray[i].subdivisionLevel = static_cast<uint16_t>(subdivisionLevel);
    triArray[i].format           = static_cast<uint16_t>(VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR);
  }
  const vk::DeviceSize triArraySize = triangleCount * sizeof(VkMicromapTriangleKHR);

  // ── One-shot command buffer ────────────────────────────────────────────────
  vk::raii::CommandPool localPool(dev, vk::CommandPoolCreateInfo{
    .flags            = vk::CommandPoolCreateFlagBits::eTransient,
    .queueFamilyIndex = m_gfxFamily
  });
  auto cb = beginOneShot(localPool);

  // ── Stage + upload state data ──────────────────────────────────────────────
  auto [stagDataBuf, stagDataMem] = makeStagingBuffer(dev, *m_physDev, packed.data(), dataSize);
  auto [dataBuf, dataMem]         = makeDeviceBuffer(dev, *m_physDev, dataSize,
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR);

  cb.copyBuffer(*stagDataBuf, *dataBuf, vk::BufferCopy{ .size = dataSize });

  // ── Stage + upload triangle array ─────────────────────────────────────────
  auto [stagTriBuf, stagTriMem] = makeStagingBuffer(dev, *m_physDev, triArray.data(), triArraySize);
  auto [triBuf, triMem]         = makeDeviceBuffer(dev, *m_physDev, triArraySize,
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR);

  cb.copyBuffer(*stagTriBuf, *triBuf, vk::BufferCopy{ .size = triArraySize });

  // ── Barrier: transfer write → AS build read ───────────────────────────────
  const std::array<vk::BufferMemoryBarrier2, 2> barriers{{
    {
      .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
      .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .dstStageMask  = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
      .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR,
      .buffer = *dataBuf, .offset = 0, .size = dataSize
    },
    {
      .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
      .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .dstStageMask  = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
      .dstAccessMask = vk::AccessFlagBits2::eAccelerationStructureReadKHR,
      .buffer = *triBuf, .offset = 0, .size = triArraySize
    }
  }};
  cb.pipelineBarrier2(vk::DependencyInfo{
    .bufferMemoryBarrierCount = 2,
    .pBufferMemoryBarriers    = barriers.data()
  });

  // ── Usage entry ────────────────────────────────────────────────────────────
  VkMicromapUsageKHR usage{};
  usage.count            = triangleCount;
  usage.subdivisionLevel = subdivisionLevel;
  usage.format           = VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR;

  // ── Micromap geometry data (chained via pNext of the geometry struct) ──────
  VkAccelerationStructureGeometryMicromapDataKHR micromapData{};
  micromapData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR;
  micromapData.usageCountsCount = 1;
  micromapData.pUsageCounts = &usage;
  micromapData.triangleArrayStride = sizeof(VkMicromapTriangleKHR);
  // data and triangleArray device addresses filled after size query

  VkAccelerationStructureGeometryKHR geometry{};
  geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geometry.pNext = &micromapData;
  geometry.geometryType = VK_GEOMETRY_TYPE_MICROMAP_KHR;

  // ── Build info (size query phase) ─────────────────────────────────────────
  vk::BuildAccelerationStructureFlagsKHR buildFlags =
      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
  if (lossyBuild)
    buildFlags |= vk::BuildAccelerationStructureFlagBitsKHR::eMicromapLossy;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
  buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR;
  buildInfo.flags = static_cast<VkBuildAccelerationStructureFlagsKHR>(buildFlags);
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &geometry;

  // ── Pre-flight: verify VK_KHR_device_address_commands function is loaded ───
  if (!dev.getDispatcher()->vkCreateAccelerationStructure2KHR)
    throw std::runtime_error(
      "[OMM] vkCreateAccelerationStructure2KHR is null — "
      "VK_KHR_device_address_commands was not enabled at device creation.");

  // ── Size query ─────────────────────────────────────────────────────────────
  // pMaxPrimitiveCounts[i] is the maximum number of micromap triangles for
  // geometry[i] per the VK_KHR_opacity_micromap spec.
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
  sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  dev.getDispatcher()->vkGetAccelerationStructureBuildSizesKHR(
    *dev,
    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
    &buildInfo,
    &triangleCount,
    &sizeInfo);

  if (sizeInfo.accelerationStructureSize == 0)
    throw std::runtime_error(
      "[OMM] vkGetAccelerationStructureBuildSizesKHR returned zero size. "
      "Ensure VK_KHR_opacity_micromap and VK_KHR_device_address_commands are enabled.");

  // ── Storage buffer for the micromap AS ────────────────────────────────────
  auto [mmStoreBuf, mmStoreMem] = makeDeviceBuffer(dev, *m_physDev,
    sizeInfo.accelerationStructureSize,
    vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
    vk::BufferUsageFlagBits::eShaderDeviceAddress);

  // ── Scratch buffer ─────────────────────────────────────────────────────────
  auto [scratchBuf, scratchMem] = makeDeviceBuffer(dev, *m_physDev,
    std::max(sizeInfo.buildScratchSize, VkDeviceSize{4}),
    vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eShaderDeviceAddress);

  // ── Create VkAccelerationStructureKHR (type = opacity micromap) ───────────
  const vk::DeviceAddress storageAddr = dev.getBufferAddress({.buffer = *mmStoreBuf});
  auto micromap = dev.createAccelerationStructure2KHR(vk::AccelerationStructureCreateInfo2KHR{
    .addressRange = {storageAddr, sizeInfo.accelerationStructureSize},
    .type = vk::AccelerationStructureTypeKHR::eOpacityMicromap
  });

  // ── Fill device addresses and record build ────────────────────────────────
  micromapData.data = dev.getBufferAddress({.buffer = *dataBuf});
  micromapData.triangleArray = dev.getBufferAddress({.buffer = *triBuf});

  buildInfo.dstAccelerationStructure = *micromap;
  buildInfo.scratchData.deviceAddress = dev.getBufferAddress({ .buffer = *scratchBuf });

  const VkAccelerationStructureBuildRangeInfoKHR rangeInfo{
    .primitiveCount  = triangleCount,
    .primitiveOffset = 0,
    .firstVertex = 0,
    .transformOffset = 0
  };
  const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;
  dev.getDispatcher()->vkCmdBuildAccelerationStructuresKHR(*cb, 1, &buildInfo, &pRangeInfo);

  submitOneShot(cb);

  // ── Build the pNext attachment chain ──────────────────────────────────────
  // VkAccelerationStructureTrianglesOpacityMicromapKHR is chained into pNext
  // of VkAccelerationStructureGeometryTrianglesDataKHR for the BLAS build.
  auto pNextOwner = std::make_unique<GpuEntry::PNextStorage>();
  pNextOwner->usageEntry = usage;

  VkAccelerationStructureTrianglesOpacityMicromapKHR& chain = pNextOwner->chain;
  chain.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_KHR;
  chain.pNext = nullptr;
  chain.indexType = VK_INDEX_TYPE_NONE_KHR; // identity: tri N → entry N
  chain.indexBuffer = {};
  chain.indexStride = 0;
  chain.baseTriangle = 0;
  chain.micromap = *micromap;

  void* pNextPtr = &chain;

  outInfo.gpuBytes = sizeInfo.accelerationStructureSize + dataSize + triArraySize;

  // ── Store GPU resources ────────────────────────────────────────────────────
  GpuEntry ge;
  ge.dataBuf = std::move(dataBuf);
  ge.dataMem = std::move(dataMem);
  ge.triBuf      = std::move(triBuf);
  ge.triMem      = std::move(triMem);
  ge.mmStoreBuf  = std::move(mmStoreBuf);
  ge.mmStoreMem  = std::move(mmStoreMem);
  ge.micromap    = std::move(micromap);
  ge.pNextOwner  = std::move(pNextOwner);

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_meshToEntry[mesh] = m_entries.size();
    m_entries.push_back(std::move(ge));
  }

  return pNextPtr;
#endif // !defined(PLATFORM_ANDROID)
}

// =============================================================================
// Helpers
// =============================================================================

float OpacityMicromapBuilder::sampleAlpha(const uint8_t* pixels,
                                           uint32_t w, uint32_t h, uint32_t ch,
                                           float u, float v) const
{
  u -= std::floor(u);
  v -= std::floor(v);
  const uint32_t px  = std::min(static_cast<uint32_t>(u * static_cast<float>(w)), w - 1u);
  const uint32_t py  = std::min(static_cast<uint32_t>(v * static_cast<float>(h)), h - 1u);
  const size_t   off = (static_cast<size_t>(py) * w + px) * ch;
  if (ch == 1)  return static_cast<float>(pixels[off])     / 255.f;
  if (ch == 2)  return static_cast<float>(pixels[off + 1]) / 255.f;
  if (ch >= 4)  return static_cast<float>(pixels[off + 3]) / 255.f;
  return 1.f;
}

vk::raii::CommandBuffer OpacityMicromapBuilder::beginOneShot(vk::raii::CommandPool& pool) const {
  auto bufs = m_device->allocateCommandBuffers(vk::CommandBufferAllocateInfo{
    .commandPool        = *pool,
    .level              = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = 1
  });
  auto cb = std::move(bufs[0]);
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
