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
#include "imgui/imgui.h"
#include "imgui_system.h"
#include "mesh_component.h"
#include "model_loader.h"
#include "renderer.h"
#include "transform_component.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtx/norm.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <ranges>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>

// ===================== Culling helpers implementation =====================

Renderer::FrustumPlanes Renderer::extractFrustumPlanes(const glm::mat4& vp) {
  // Work in row-major form for standard plane extraction by transposing GLM's column-major matrix
  glm::mat4 m = glm::transpose(vp);
  FrustumPlanes fp{};
  // Left   : m[3] + m[0]
  fp.planes[0] = m[3] + m[0];
  // Right  : m[3] - m[0]
  fp.planes[1] = m[3] - m[0];
  // Bottom : m[3] + m[1]
  fp.planes[2] = m[3] + m[1];
  // Top    : m[3] - m[1]
  fp.planes[3] = m[3] - m[1];
  // Near   : m[2] (matches Vulkan [0, 1] clip range)
  fp.planes[4] = m[2];
  // Far    : m[3] - m[2]
  fp.planes[5] = m[3] - m[2];

  // Normalize planes
  for (auto& p : fp.planes) {
    glm::vec3 n(p.x, p.y, p.z);
    float len = glm::length(n);
    if (len > 0.0f) {
      p /= len;
    }
  }
  return fp;
}

void Renderer::transformAABB(const glm::mat4& M,
                             const glm::vec3& localMin,
                             const glm::vec3& localMax,
                             glm::vec3& outMin,
                             glm::vec3& outMax) {
  // OBB (from model) to world AABB using center/extents and absolute 3x3
  const glm::vec3 c = 0.5f * (localMin + localMax);
  const glm::vec3 e = 0.5f * (localMax - localMin);

  const glm::vec3 worldCenter = glm::vec3(M * glm::vec4(c, 1.0f));
  // Upper-left 3x3
  const glm::mat3 A = glm::mat3(M);
  const glm::mat3 AbsA = glm::mat3(glm::abs(A[0]), glm::abs(A[1]), glm::abs(A[2]));
  const glm::vec3 worldExtents = AbsA * e; // component-wise combination

  outMin = worldCenter - worldExtents;
  outMax = worldCenter + worldExtents;
}

bool Renderer::aabbIntersectsFrustum(const glm::vec3& worldMin,
                                     const glm::vec3& worldMax,
                                     const FrustumPlanes& frustum) {
  // Use the p-vertex test against each plane; if outside any plane → culled
  for (const auto& p : frustum.planes) {
    const glm::vec3 n(p.x, p.y, p.z);
    // Choose positive vertex (furthest in direction of normal)
    glm::vec3 v{
      n.x >= 0.0f ? worldMax.x : worldMin.x,
      n.y >= 0.0f ? worldMax.y : worldMin.y,
      n.z >= 0.0f ? worldMax.z : worldMin.z
    };

    // If the most positive vertex is still on the negative side of the plane,
    // then the entire box is on the negative side.
    // Use a small epsilon to avoid numerical issues.
    if (glm::dot(n, v) + p.w < -0.01f) {
      return false; // completely outside
    }
  }
  return true;
}

// This file contains rendering-related methods from the Renderer class

// Create swap chain
bool Renderer::createSwapChain() {
  try {
    // Query swap chain support
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    // Choose swap surface format, present mode, and extent
    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);

    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    // Choose image count
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    // Create swap chain info
    vk::SwapchainCreateInfoKHR createInfo{
      .surface = *surface,
      .minImageCount = imageCount,
      .imageFormat = surfaceFormat.format,
      .imageColorSpace = surfaceFormat.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
      .preTransform = swapChainSupport.capabilities.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = presentMode,
      .clipped = VK_TRUE,
      .oldSwapchain = nullptr
    };

    // Find queue families
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::array<uint32_t, 2> queueFamilyIndicesLoc = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    // Set sharing mode
    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
      createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndicesLoc.size());
      createInfo.pQueueFamilyIndices = queueFamilyIndicesLoc.data();
    } else {
      createInfo.imageSharingMode = vk::SharingMode::eExclusive;
      createInfo.queueFamilyIndexCount = 0;
      createInfo.pQueueFamilyIndices = nullptr;
    }

    // Create swap chain
    swapChain = vk::raii::SwapchainKHR(device, createInfo);

    // Get swap chain images
    swapChainImages = swapChain.getImages();

    // Swapchain images start in UNDEFINED layout; track per-image layout for correct barriers.
    swapChainImageLayouts.assign(swapChainImages.size(), vk::ImageLayout::eUndefined);

    // Store swap chain format and extent
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create swap chain: " << e.what() << std::endl;
    return false;
  }
}

// ===================== Planar reflections resources =====================
bool Renderer::createReflectionResources(uint32_t width, uint32_t height) {
  try {
    destroyReflectionResources();
    reflections.clear();
    reflections.resize(MAX_FRAMES_IN_FLIGHT);
    reflectionVPs.clear();
    reflectionVPs.resize(MAX_FRAMES_IN_FLIGHT, glm::mat4(1.0f));
    sampleReflectionVP = glm::mat4(1.0f);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
      auto& rt = reflections[i];
      rt.width = width;
      rt.height = height;

      // Color RT: use swapchain format to match existing PBR pipeline rendering formats
      vk::Format colorFmt = swapChainImageFormat;
      auto [colorImg, colorAlloc] = createImagePooled(
        width,
        height,
        colorFmt,
        vk::ImageTiling::eOptimal,
        // Allow sampling in glass and blitting to swapchain for diagnostics
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        /*mipLevels*/
        1,
        vk::SharingMode::eExclusive,
        {});
      rt.color = std::move(colorImg);
      rt.colorAlloc = std::move(colorAlloc);
      rt.colorView = createImageView(rt.color, colorFmt, vk::ImageAspectFlagBits::eColor, 1);
      // Simple sampler for sampling reflection texture (no mips)
      vk::SamplerCreateInfo sampInfo{.magFilter = vk::Filter::eLinear, .minFilter = vk::Filter::eLinear, .mipmapMode = vk::SamplerMipmapMode::eNearest, .addressModeU = vk::SamplerAddressMode::eClampToEdge, .addressModeV = vk::SamplerAddressMode::eClampToEdge, .addressModeW = vk::SamplerAddressMode::eClampToEdge, .minLod = 0.0f, .maxLod = 0.0f};
      rt.colorSampler = vk::raii::Sampler(device, sampInfo);

      // Depth RT
      vk::Format depthFmt = findDepthFormat();
      auto [depthImg, depthAlloc] = createImagePooled(
        width,
        height,
        depthFmt,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        /*mipLevels*/
        1,
        vk::SharingMode::eExclusive,
        {});
      rt.depth = std::move(depthImg);
      rt.depthAlloc = std::move(depthAlloc);
      rt.depthView = createImageView(rt.depth, depthFmt, vk::ImageAspectFlagBits::eDepth, 1);
    }

    // One-time initialization: transition all per-frame reflection color images
    // from UNDEFINED to SHADER_READ_ONLY_OPTIMAL so that the first frame can
    // legally sample the "previous" frame's image.
    if (!reflections.empty()) {
      vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
      };
      vk::raii::CommandPool tempPool(device, poolInfo);
      vk::CommandBufferAllocateInfo allocInfo{.commandPool = *tempPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
      vk::raii::CommandBuffers cbs(device, allocInfo);
      vk::raii::CommandBuffer& cb = cbs[0];
      cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

      std::vector<vk::ImageMemoryBarrier2> barriers;
      barriers.reserve(reflections.size());
      for (auto& rt : reflections) {
        if (!!*rt.color) {
          barriers.push_back(vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *rt.color,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
          });
        }
      }
      if (!barriers.empty()) {
        vk::DependencyInfo depInfo{.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()), .pImageMemoryBarriers = barriers.data()};
        cb.pipelineBarrier2(depInfo);
      }
      cb.end();
      vk::SubmitInfo submit{.commandBufferCount = 1, .pCommandBuffers = &*cb};
      vk::raii::Fence fence(device, vk::FenceCreateInfo{}); {
        std::lock_guard<std::mutex> lock(queueMutex);
        graphicsQueue.submit(submit, *fence);
      }
      vk::Result result = waitForFencesSafe(*fence, VK_TRUE);
      if (result != vk::Result::eSuccess) {
        std::cerr << "Error: Failed to wait for reflection resource fence: " << vk::to_string(result) << std::endl;
      }
    }

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create reflection resources: " << e.what() << std::endl;
    destroyReflectionResources();
    return false;
  }
}

void Renderer::destroyReflectionResources() {
  for (auto& rt : reflections) {
    rt.colorSampler = vk::raii::Sampler(nullptr);
    rt.colorView = vk::raii::ImageView(nullptr);
    rt.colorAlloc = nullptr;
    rt.color = vk::raii::Image(nullptr);
    rt.depthView = vk::raii::ImageView(nullptr);
    rt.depthAlloc = nullptr;
    rt.depth = vk::raii::Image(nullptr);
    rt.width = rt.height = 0;
  }
}

void Renderer::renderReflectionPass(vk::raii::CommandBuffer& cmd,
                                    const glm::vec4& planeWS,
                                    CameraComponent* camera,
                                    const std::vector<RenderJob>& jobs) {
  if (reflections.empty())
    return;
  auto& rt = reflections[currentFrame];
  if (rt.width == 0 || rt.height == 0 || !*rt.colorView || !*rt.depthView)
    return;

  // Transition reflection color to COLOR_ATTACHMENT_OPTIMAL (Sync2)
  vk::ImageMemoryBarrier2 toColor2{
    .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
    .srcAccessMask = {},
    .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
    .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .image = *rt.color,
    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
  };
  // Transition reflection depth to DEPTH_STENCIL_ATTACHMENT_OPTIMAL (Sync2)
  vk::ImageMemoryBarrier2 toDepth2{
    .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
    .srcAccessMask = {},
    .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
    .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eDepthStencilAttachmentRead,
    .oldLayout = vk::ImageLayout::eUndefined,
    .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .image = *rt.depth,
    .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
  };
  std::array<vk::ImageMemoryBarrier2, 2> preBarriers{toColor2, toDepth2};
  vk::DependencyInfo depInfoToColor{.imageMemoryBarrierCount = static_cast<uint32_t>(preBarriers.size()), .pImageMemoryBarriers = preBarriers.data()};
  cmd.pipelineBarrier2(depInfoToColor);

  vk::RenderingAttachmentInfo colorAtt{
    .imageView = *rt.colorView,
    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
    .loadOp = vk::AttachmentLoadOp::eClear,
    .storeOp = vk::AttachmentStoreOp::eStore,
    // Clear to black so scene content dominates reflections
    .clearValue = vk::ClearValue{vk::ClearColorValue{std::array < float, 4 >{0.0f, 0.0f, 0.0f, 1.0f}}}
  };
  vk::RenderingAttachmentInfo depthAtt{
    .imageView = *rt.depthView,
    .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
    .loadOp = vk::AttachmentLoadOp::eClear,
    .storeOp = vk::AttachmentStoreOp::eDontCare,
    .clearValue = vk::ClearValue{vk::ClearDepthStencilValue{1.0f, 0}}
  };
  vk::RenderingInfo rinfo{
    .renderArea = vk::Rect2D({0, 0}, {rt.width, rt.height}),
    .layerCount = 1,
    .colorAttachmentCount = 1,
    .pColorAttachments = &colorAtt,
    .pDepthAttachment = &depthAtt
  };
  cmd.beginRendering(rinfo);
  // Compute mirrored view matrix about planeWS (default Y=0 plane)
  glm::mat4 reflectM(1.0f);
  // For Y=0 plane, reflection is simply flip Y
  if (glm::length(glm::vec3(planeWS.x, planeWS.y, planeWS.z)) > 0.5f && fabsf(planeWS.y - 1.0f) < 1e-3f && fabsf(planeWS.x) < 1e-3f && fabsf(planeWS.z) < 1e-3f) {
    reflectM[1][1] = -1.0f;
  } else {
    // General plane reflection matrix R = I - 2*n*n^T for normalized plane; ignore translation for now
    glm::vec3 n = glm::normalize(glm::vec3(planeWS));
    glm::mat3 R = glm::mat3(1.0f) - 2.0f * glm::outerProduct(n, n);
    reflectM = glm::mat4(R);
  }

  glm::mat4 viewReflected = camera ? (camera->GetViewMatrix() * reflectM) : reflectM;
  glm::mat4 projReflected = camera ? camera->GetProjectionMatrix() : glm::mat4(1.0f);
  currentReflectionVP = projReflected * viewReflected;
  currentReflectionPlane = planeWS;
  if (currentFrame < reflectionVPs.size()) {
    reflectionVPs[currentFrame] = currentReflectionVP;
  }

  // Set viewport/scissor to reflection RT size
  vk::Viewport rv(0.0f, 0.0f, static_cast<float>(rt.width), static_cast<float>(rt.height), 0.0f, 1.0f);
  cmd.setViewport(0, rv);
  vk::Rect2D rs({0, 0}, {rt.width, rt.height});
  cmd.setScissor(0, rs);

  // Draw opaque entities with mirrored view
  // Use reflection-specific pipeline (cull none) to avoid mirrored winding issues.
  if (!!*pbrReflectionGraphicsPipeline) {
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrReflectionGraphicsPipeline);
  } else if (!!*pbrGraphicsPipeline) {
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrGraphicsPipeline);
  }

  // Prepare frustum for mirrored view to allow culling
  FrustumPlanes reflectFrustum = extractFrustumPlanes(currentReflectionVP);

  // Render all jobs (skip transparency)
  for (const auto& job : jobs) {
    Entity* entity = job.entity;
    MeshComponent* meshComponent = job.meshComp;
    EntityResources* entityRes = job.entityRes;
    MeshResources* meshRes = job.meshRes;

    if (entityRes->cachedIsBlended)
      continue;

    // Frustum culling for mirrored view
    if (meshComponent->HasLocalAABB()) {
      const glm::mat4 model = job.transformComp ? job.transformComp->GetModelMatrix() : glm::mat4(1.0f);
      glm::vec3 wmin, wmax;
      transformAABB(model, meshComponent->GetLocalAABBMin(), meshComponent->GetLocalAABBMax(), wmin, wmax);
      if (!aabbIntersectsFrustum(wmin, wmax, reflectFrustum)) {
        continue; // culled from reflection
      }
    }

    // Bind geometry
    std::array<vk::Buffer, 2> buffers = {*meshRes->vertexBuffer, *entityRes->instanceBuffer};
    std::array<vk::DeviceSize, 2> offsets = {0, 0};
    cmd.bindVertexBuffers(0, buffers, offsets);
    cmd.bindIndexBuffer(*meshRes->indexBuffer, 0, vk::IndexType::eUint32);

    // Populate UBO with mirrored view + clip plane and reflection flags
    UniformBufferObject ubo{};
    if (job.transformComp)
      ubo.model = job.transformComp->GetModelMatrix();
    else
      ubo.model = glm::mat4(1.0f);
    ubo.view = viewReflected;
    ubo.proj = projReflected;
    ubo.camPos = glm::vec4(camera ? camera->GetPosition() : glm::vec3(0), 1.0f);
    ubo.reflectionPass = 1;
    ubo.reflectionEnabled = 0;
    ubo.reflectionVP = currentReflectionVP;
    ubo.clipPlaneWS = planeWS;
    // Ray query shadows in reflection pass
    ubo.padding2 = enableRasterRayQueryShadows ? 1.0f : 0.0f;

    updateUniformBufferInternal(currentFrame, entity, entityRes, camera, ubo);

    // Bind descriptor set (PBR set 0)
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           *pbrPipelineLayout,
                           0,
                           *entityRes->pbrDescriptorSets[currentFrame],
                           nullptr);

    // Push material properties
    MaterialProperties mp = entityRes->cachedMaterialProps;
    // Transmission suppressed during reflection pass via UBO (reflectionPass=1)
    mp.transmissionFactor = 0.0f;
    pushMaterialProperties(*cmd, mp);

    // Issue draw
    uint32_t instanceCount = std::max(1u, static_cast<uint32_t>(meshComponent->GetInstanceCount()));
    cmd.drawIndexed(meshRes->indexCount, instanceCount, 0, 0, 0);
  }

  cmd.endRendering();

  // Transition reflection color to SHADER_READ_ONLY for sampling in main pass (Sync2)
  vk::ImageMemoryBarrier2 toSample2{
    .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
    .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
    .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
    .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
    .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .image = *rt.color,
    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
  };
  vk::DependencyInfo depInfoToSample{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toSample2};
  cmd.pipelineBarrier2(depInfoToSample);
}

// Create image views
bool Renderer::createImageViews() {
  try {
    opaqueSceneColorImages.clear();
    opaqueSceneColorImageAllocations.clear();
    opaqueSceneColorImageViews.clear();
    opaqueSceneColorImageLayouts.clear();
    opaqueSceneColorSampler.clear();
    // Resize image views vector
    swapChainImageViews.clear();
    swapChainImageViews.reserve(swapChainImages.size());

    // Create image view info template (image will be set per iteration)
    vk::ImageViewCreateInfo createInfo{
      .viewType = vk::ImageViewType::e2D,
      .format = swapChainImageFormat,
      .components = {
        .r = vk::ComponentSwizzle::eIdentity,
        .g = vk::ComponentSwizzle::eIdentity,
        .b = vk::ComponentSwizzle::eIdentity,
        .a = vk::ComponentSwizzle::eIdentity
      },
      .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}
    };

    // Create image view for each swap chain image
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      createInfo.image = swapChainImages[i];
      swapChainImageViews.emplace_back(device, createInfo);
    }

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create image views: " << e.what() << std::endl;
    return false;
  }
}

// Setup dynamic rendering
bool Renderer::setupDynamicRendering() {
  try {
    // Create color attachment
    colorAttachments = {
      vk::RenderingAttachmentInfo{
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(std::array < float, 4 >{0.0f, 0.0f, 0.0f, 1.0f})
      }
    };

    // Create depth attachment
    depthAttachment = vk::RenderingAttachmentInfo{
      .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = vk::ClearDepthStencilValue(1.0f, 0)
    };

    // Create rendering info
    renderingInfo = vk::RenderingInfo{
      .renderArea = vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent),
      .layerCount = 1,
      .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
      .pColorAttachments = colorAttachments.data(),
      .pDepthAttachment = &depthAttachment
    };

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to setup dynamic rendering: " << e.what() << std::endl;
    return false;
  }
}

// Create command pool
bool Renderer::createCommandPool() {
  try {
    // Find queue families
    QueueFamilyIndices queueFamilyIndicesLoc = findQueueFamilies(physicalDevice);

    // Create command pool info
    vk::CommandPoolCreateInfo poolInfo{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = queueFamilyIndicesLoc.graphicsFamily.value()
    };

    // Create command pool
    commandPool = vk::raii::CommandPool(device, poolInfo);

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create command pool: " << e.what() << std::endl;
    return false;
  }
}

// Create command buffers
bool Renderer::createCommandBuffers() {
  try {
    // Resize command buffers vector
    commandBuffers.clear();
    commandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);

    // Create command buffer allocation info
    vk::CommandBufferAllocateInfo allocInfo{
      .commandPool = *commandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
    };

    // Allocate command buffers
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    for (size_t i = 0; i < commandBuffers.size(); ++i) {
    }

    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create command buffers: " << e.what() << std::endl;
    return false;
  }
}

// Create sync objects
bool Renderer::createSyncObjects() {
  try {
    // Resize semaphores and fences vectors
    imageAvailableSemaphores.clear();
    renderFinishedSemaphores.clear();
    inFlightFences.clear();

    // Semaphores per swapchain image (indexed by imageIndex from acquireNextImage)
    // The presentation engine holds semaphores until the image is re-acquired, so we need
    // one semaphore per swapchain image to avoid reuse conflicts. See Vulkan spec:
    // https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
    const auto semaphoreCount = static_cast<uint32_t>(swapChainImages.size());
    imageAvailableSemaphores.reserve(semaphoreCount);
    renderFinishedSemaphores.reserve(semaphoreCount);

    // Fences per frame-in-flight for CPU-GPU synchronization (indexed by currentFrame)
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

    // Create semaphore info
    vk::SemaphoreCreateInfo semaphoreInfo{};

    // Create semaphores per swapchain image (indexed by imageIndex for presentation sync)
    for (uint32_t i = 0; i < semaphoreCount; i++) {
      imageAvailableSemaphores.emplace_back(device, semaphoreInfo);
      renderFinishedSemaphores.emplace_back(device, semaphoreInfo);
    }

    // Create fences per frame-in-flight (indexed by currentFrame for CPU-GPU pacing)
    vk::FenceCreateInfo fenceInfo{
      .flags = vk::FenceCreateFlagBits::eSignaled
    };
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      inFlightFences.emplace_back(device, fenceInfo);
    }

    // Ensure uploads timeline semaphore exists (created early in createLogicalDevice)
    // No action needed here unless reinitializing after swapchain recreation.
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create sync objects: " << e.what() << std::endl;
    return false;
  }
}

// Clean up swap chain
void Renderer::cleanupSwapChain() {
  // Clean up depth resources
  depthImageView = vk::raii::ImageView(nullptr);
  depthImage = vk::raii::Image(nullptr);
  depthImageAllocation = nullptr;

  // Clean up swap chain image views
  swapChainImageViews.clear();

  // Note: Keep descriptor pool alive here to ensure descriptor sets remain valid during swapchain recreation.
  // descriptorPool is preserved; it will be managed during full renderer teardown.

  // Destroy reflection render targets if present
  destroyReflectionResources();

  // Clean up pipelines
  graphicsPipeline = vk::raii::Pipeline(nullptr);
  pbrGraphicsPipeline = vk::raii::Pipeline(nullptr);
  lightingPipeline = vk::raii::Pipeline(nullptr);

  // Clean up pipeline layouts
  pipelineLayout = vk::raii::PipelineLayout(nullptr);
  pbrPipelineLayout = vk::raii::PipelineLayout(nullptr);
  lightingPipelineLayout = vk::raii::PipelineLayout(nullptr);

  // Clean up sync objects (they need to be recreated with new swap chain image count)
  imageAvailableSemaphores.clear();
  renderFinishedSemaphores.clear();
  inFlightFences.clear();

  // Clean up swap chain
  swapChain = vk::raii::SwapchainKHR(nullptr);
}

// Recreate swap chain
void Renderer::recreateSwapChain() {
  // Prevent background uploads worker from mutating descriptors while we rebuild
  StopUploadsWorker();

  // Block descriptor writes while we rebuild swapchain and descriptor pools
  descriptorSetsValid.store(false, std::memory_order_relaxed); {
    // Drop any deferred descriptor updates that target old descriptor sets
    std::lock_guard<std::mutex> lk(pendingDescMutex);
    pendingDescOps.clear();
    descriptorRefreshPending.store(false, std::memory_order_relaxed);
  }

  // Wait for all frames in flight to complete using the timeline
  if (*frameTimeline) {
    uint64_t waitValue = totalFrameCount.load();
    vk::SemaphoreWaitInfo waitInfo{
      .semaphoreCount = 1,
      .pSemaphores = &*frameTimeline,
      .pValues = &waitValue
    };
    if (device.waitSemaphores(waitInfo, UINT64_MAX) != vk::Result::eSuccess) {
      std::cerr << "Warning: Failed to wait for frameTimeline during swapchain recreation" << std::endl;
    }
  }

  // Wait for the device to be idle before recreating the swap chain
  // External synchronization required (VVL): serialize against queue submits/present.
  WaitIdle();

  // Clean up old swap chain resources
  cleanupSwapChain();

  // Recreate swap chain and related resources
  createSwapChain();
  createImageViews();
  setupDynamicRendering();
  createDepthResources();

  // (Re)create reflection resources if enabled
  if (enablePlanarReflections) {
    uint32_t rw = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * reflectionResolutionScale));
    uint32_t rh = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * reflectionResolutionScale));
    createReflectionResources(rw, rh);
  }

  // Recreate sync objects with correct sizing for new swap chain
  createSyncObjects();

  // Recreate off-screen opaque scene color and descriptor sets needed by transparent pass
  createOpaqueSceneColorResources();
  createTransparentDescriptorSets();
  createTransparentFallbackDescriptorSets();

  // Wait for all command buffers to complete before clearing resources
  for (const auto& fence : inFlightFences) {
    vk::Result result = waitForFencesSafe(*fence, VK_TRUE);
    if (result != vk::Result::eSuccess) {
      std::cerr << "Error: Failed to wait for fence before clearing resources: " << vk::to_string(result) << std::endl;
    }
  }

  // Clear all entity descriptor sets since they're now invalid (allocated from the old pool)
  {
    // Serialize descriptor frees against any other descriptor operations
    std::lock_guard<std::mutex> lk(descriptorMutex);
    for (auto& kv : entityResources) {
      auto& resources = kv.second;
      resources.basicDescriptorSets.clear();
      resources.pbrDescriptorSets.clear();
      // Descriptor initialization flags must be reset because new descriptor sets
      // will be allocated and only the current frame will be initialized at runtime.
      resources.pbrUboBindingWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
      resources.basicUboBindingWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
      resources.pbrImagesWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
      resources.basicImagesWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
      resources.pbrFixedBindingsWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
    }
  }

  // Clear ray query descriptor sets - they reference the old output image which will be destroyed
  // Must clear before recreating to avoid descriptor set corruption
  rayQueryDescriptorSets.clear();
  rayQueryDescriptorsWritten.clear();
  rayQueryDescriptorsDirtyMask.store(0u, std::memory_order_relaxed);

  // Destroy ray query output image resources - they're sized to old swapchain dimensions
  rayQueryOutputImageView = vk::raii::ImageView(nullptr);
  rayQueryOutputImage = vk::raii::Image(nullptr);
  rayQueryOutputImageAllocation = nullptr;

  createGraphicsPipeline();
  createPBRPipeline();
  createLightingPipeline();
  createCompositePipeline();

  // Recreate Forward+ specific pipelines/resources and resize tile buffers for new extent
  if (useForwardPlus) {
    createDepthPrepassPipeline();
    uint32_t tilesX = (swapChainExtent.width + forwardPlusTileSizeX - 1) / forwardPlusTileSizeX;
    uint32_t tilesY = (swapChainExtent.height + forwardPlusTileSizeY - 1) / forwardPlusTileSizeY;
    createOrResizeForwardPlusBuffers(tilesX, tilesY, forwardPlusSlicesZ);
  }

  // Re-create command buffers to ensure fresh recording against new swapchain state
  commandBuffers.clear();
  createCommandBuffers();
  currentFrame = 0;

  // Recreate ray query resources with new swapchain dimensions
  // This must happen after descriptor pool is valid but before marking descriptor sets valid
  if (rayQueryEnabled && accelerationStructureEnabled) {
    if (!createRayQueryResources()) {
      std::cerr << "Warning: Failed to recreate ray query resources after swapchain recreation\n";
    }
  }

  // Recreate descriptor sets for all entities after swapchain/pipeline rebuild
  for (const auto& kv : entityResources) {
    const auto& entity = kv.first;
    if (!entity)
      continue;
    auto meshComponent = entity->GetComponent<MeshComponent>();
    if (!meshComponent)
      continue;

    std::string texturePath = meshComponent->GetTexturePath();
    // Fallback for basic pipeline: use baseColor when legacy path is empty
    if (texturePath.empty()) {
      const std::string& baseColor = meshComponent->GetBaseColorTexturePath();
      if (!baseColor.empty()) {
        texturePath = baseColor;
      }
    }
    // Recreate basic descriptor sets (ignore failures here to avoid breaking resize)
    createDescriptorSets(entity, texturePath, false);
    // Recreate PBR descriptor sets
    createDescriptorSets(entity, texturePath, true);
  }

  // Descriptor sets are now valid again
  descriptorSetsValid.store(true, std::memory_order_relaxed);

  // Resume background uploads worker now that swapchain and descriptors are recreated
  StartUploadsWorker();
}

void Renderer::prepareFrameUboTemplate(CameraComponent* camera) {
  frameUboTemplate = UniformBufferObject{};
  if (!camera) return;

  frameUboTemplate.view = camera->GetViewMatrix();
  frameUboTemplate.proj = camera->GetProjectionMatrix();
  frameUboTemplate.proj[1][1] *= -1; // Flip Y for Vulkan
  frameUboTemplate.camPos = glm::vec4(camera->GetPosition(), 1.0f);

  frameUboTemplate.lightCount = static_cast<int>(lastFrameLightCount);
  frameUboTemplate.exposure = std::clamp(this->exposure, 0.2f, 4.0f);
  frameUboTemplate.gamma = this->gamma;
  frameUboTemplate.screenDimensions = glm::vec2(swapChainExtent.width, swapChainExtent.height);
  frameUboTemplate.nearZ = camera->GetNearPlane();
  frameUboTemplate.farZ = camera->GetFarPlane();
  frameUboTemplate.slicesZ = static_cast<float>(forwardPlusSlicesZ);

  int outputIsSRGB = (swapChainImageFormat == vk::Format::eR8G8B8A8Srgb ||
                       swapChainImageFormat == vk::Format::eB8G8R8A8Srgb) ? 1 : 0;
  frameUboTemplate.padding0 = outputIsSRGB;
  // Raster PBR shader uses padding1 as the Forward+ enable flag.
  // 0 = disabled (always use global light loop), non-zero = enabled (use culled tile lists).
  frameUboTemplate.padding1 = useForwardPlus ? 1.0f : 0.0f;
  frameUboTemplate.padding2 = enableRasterRayQueryShadows ? 1.0f : 0.0f;

  bool reflReady = false;
  if (enablePlanarReflections && !reflections.empty()) {
    const uint32_t count = static_cast<uint32_t>(reflections.size());
    const uint32_t prev = (currentFrame + count - 1u) % count;
    auto& rtPrev = reflections[prev];
    reflReady = (!!*rtPrev.colorView) && (!!*rtPrev.colorSampler);
  }
  frameUboTemplate.reflectionEnabled = reflReady ? 1 : 0;
  frameUboTemplate.reflectionVP = sampleReflectionVP;
  frameUboTemplate.clipPlaneWS = currentReflectionPlane;
  frameUboTemplate.reflectionIntensity = std::clamp(reflectionIntensity, 0.0f, 2.0f);
  frameUboTemplate.enableRayQueryReflections = enableRayQueryReflections ? 1 : 0;
  frameUboTemplate.enableRayQueryTransparency = enableRayQueryTransparency ? 1 : 0;

  // Ray-query shared buffers are also used by raster PBR when doing ray-query shadows.
  // Populate counts so shaders can bounds-check even when running in raster mode.
  frameUboTemplate.geometryInfoCount = static_cast<int>(geometryInfoCountCPU);
  frameUboTemplate.materialCount = static_cast<int>(materialCountCPU);
}

// Update uniform buffer
void Renderer::updateUniformBuffer(uint32_t currentImage, Entity* entity, EntityResources* entityRes, CameraComponent* camera, TransformComponent* tc) {
  if (!entityRes) {
    return;
  }

  // Get transform component
  auto transformComponent = tc ? tc : (entity ? entity->GetComponent<TransformComponent>() : nullptr);
  if (!transformComponent) {
    return;
  }

  // Create uniform buffer object
  UniformBufferObject ubo{};
  ubo.model = transformComponent->GetModelMatrix();
  ubo.view = camera->GetViewMatrix();
  ubo.proj = camera->GetProjectionMatrix();
  ubo.proj[1][1] *= -1; // Flip Y for Vulkan

  // Continue with the rest of the uniform buffer setup
  updateUniformBufferInternal(currentImage, entity, entityRes, camera, ubo);
}

// Overloaded version that accepts a custom transform matrix
void Renderer::updateUniformBuffer(uint32_t currentImage, Entity* entity, EntityResources* entityRes, CameraComponent* camera, const glm::mat4& customTransform) {
  if (!entityRes) return;
  // Create the uniform buffer object with custom transform
  UniformBufferObject ubo{};
  ubo.model = customTransform;
  ubo.view = camera->GetViewMatrix();
  ubo.proj = camera->GetProjectionMatrix();
  ubo.proj[1][1] *= -1; // Flip Y for Vulkan

  // Continue with the rest of the uniform buffer setup
  updateUniformBufferInternal(currentImage, entity, entityRes, camera, ubo);
}

// Internal helper function to complete uniform buffer setup
void Renderer::updateUniformBufferInternal(uint32_t currentImage, Entity* entity, EntityResources* entityRes, CameraComponent* camera, UniformBufferObject& ubo) {
  if (!entityRes) {
    return;
  }

  // Use frame template for most fields
  UniformBufferObject finalUbo = frameUboTemplate;
  finalUbo.model = ubo.model;

  // For reflection pass, we must override view/proj/reflection flags
  if (ubo.reflectionPass == 1) {
    finalUbo.view = ubo.view;
    finalUbo.proj = ubo.proj;
    finalUbo.reflectionPass = 1;
    finalUbo.reflectionEnabled = 0;
    finalUbo.reflectionVP = ubo.reflectionVP;
    finalUbo.clipPlaneWS = ubo.clipPlaneWS;
    finalUbo.padding2 = ubo.padding2;
  }

  // Copy to uniform buffer (guard against null mapped pointer)
  void* dst = entityRes->uniformBuffersMapped[currentImage];
  if (!dst) {
    std::cerr << "Warning: UBO mapped ptr null for entity '" << (entity ? entity->GetName() : "unknown") << "' frame " << currentImage << std::endl;
    return;
  }
  std::memcpy(dst, &finalUbo, sizeof(UniformBufferObject));
}

void Renderer::ensureEntityMaterialCache(Entity* entity, EntityResources& res) {
  if (!entity)
    return;

  if (res.materialCacheValid)
    return;

  res.materialCacheValid = true;
  res.cachedMaterial = nullptr;
  res.cachedIsBlended = false;
  res.cachedIsGlass = false;
  res.cachedIsLiquid = false;

  // Defaults represent the common case (no explicit material); textures come from descriptor bindings.
  MaterialProperties mp{};
  // Sensible defaults for entities without explicit material
  mp.baseColorFactor = glm::vec4(1.0f);
  mp.metallicFactor = 0.0f;
  mp.roughnessFactor = 1.0f;
  mp.baseColorTextureSet = 0;
  mp.physicalDescriptorTextureSet = 0;
  mp.normalTextureSet = -1;
  mp.occlusionTextureSet = -1;
  mp.emissiveTextureSet = -1;
  mp.alphaMask = 0.0f;
  mp.alphaMaskCutoff = 0.5f;
  mp.emissiveFactor = glm::vec3(0.0f);
  mp.emissiveStrength = 1.0f;
  mp.transmissionFactor = 0.0f;
  mp.useSpecGlossWorkflow = 0;
  mp.glossinessFactor = 0.0f;
  mp.specularFactor = glm::vec3(1.0f);
  mp.ior = 1.5f;
  mp.hasEmissiveStrengthExtension = 0;

  if (modelLoader) {
    const std::string& entityName = entity->GetName();
    const size_t tagPos = entityName.find("_Material_");
    if (tagPos != std::string::npos) {
      const size_t afterTag = tagPos + std::string("_Material_").size();
      if (afterTag < entityName.length()) {
        // Entity name format: "modelName_Material_<index>_<materialName>"
        const std::string remainder = entityName.substr(afterTag);
        const size_t nextUnderscore = remainder.find('_');
        if (nextUnderscore != std::string::npos && nextUnderscore + 1 < remainder.length()) {
          const std::string materialName = remainder.substr(nextUnderscore + 1);
          if (const Material* material = modelLoader->GetMaterial(materialName)) {
            res.cachedMaterial = material;
            res.cachedIsGlass = material->isGlass;
            res.cachedIsLiquid = material->isLiquid;

            // Base factors
            mp.baseColorFactor = glm::vec4(material->albedo, material->alpha);
            mp.metallicFactor = material->metallic;
            mp.roughnessFactor = material->roughness;

            // Texture set flags (-1 = no texture)
            mp.baseColorTextureSet = material->albedoTexturePath.empty() ? -1 : 0;
            // physical descriptor: MR or SpecGloss
            if (material->useSpecularGlossiness) {
              mp.useSpecGlossWorkflow = 1;
              mp.physicalDescriptorTextureSet = material->specGlossTexturePath.empty() ? -1 : 0;
              mp.glossinessFactor = material->glossinessFactor;
              mp.specularFactor = material->specularFactor;
            } else {
              mp.useSpecGlossWorkflow = 0;
              mp.physicalDescriptorTextureSet = material->metallicRoughnessTexturePath.empty() ? -1 : 0;
            }
            mp.normalTextureSet = material->normalTexturePath.empty() ? -1 : 0;
            mp.occlusionTextureSet = material->occlusionTexturePath.empty() ? -1 : 0;
            mp.emissiveTextureSet = material->emissiveTexturePath.empty() ? -1 : 0;

            // Emissive and transmission/IOR
            mp.emissiveFactor = material->emissive;
            mp.emissiveStrength = material->emissiveStrength;
            // Heuristic: consider emissive strength extension present when strength != 1.0
            mp.hasEmissiveStrengthExtension = (std::abs(material->emissiveStrength - 1.0f) > 1e-6f) ? 1 : 0;
            mp.transmissionFactor = material->transmissionFactor;
            mp.ior = material->ior;

            // Alpha mask handling
            mp.alphaMask = (material->alphaMode == "MASK") ? 1.0f : 0.0f;
            mp.alphaMaskCutoff = material->alphaCutoff;

            // Blended classification (opaque materials stay in the opaque pass)
            const bool alphaBlend = (material->alphaMode == "BLEND");
            const bool highTransmission = (material->transmissionFactor > 0.2f);
            res.cachedIsBlended = alphaBlend || highTransmission || res.cachedIsGlass || res.cachedIsLiquid;
          }
        }
      }
    }
  }

  res.cachedMaterialProps = mp;
}

// Render the scene (unique_ptr container overload)
// Convert to a raw-pointer snapshot so callers can safely release their container locks.
void Renderer::Render(const std::vector<std::unique_ptr<Entity>>& entities, CameraComponent* camera, ImGuiSystem* imguiSystem) {
  std::vector<Entity *> snapshot;
  snapshot.reserve(entities.size());
  for (const auto& uptr : entities) {
    snapshot.push_back(uptr.get());
  }
  Render(snapshot, camera, imguiSystem);
}

// Render the scene (raw pointer snapshot overload)
void Renderer::Render(const std::vector<Entity *>& entities, CameraComponent* camera, ImGuiSystem* imguiSystem) {
  auto startRender = std::chrono::steady_clock::now();
  static uint64_t renderCallCount = 0;
  // 1. Initial Load State Machine Management
  // Keep the fullscreen overlay until geometry preallocation is done.
  const InternalLoadingState currentLoadState = currentInternalLoadingState.load(std::memory_order_relaxed);
  const bool isParsing = (currentLoadState == InternalLoadingState::Parsing);
  const bool isPreallocating = (currentLoadState == InternalLoadingState::Preallocating);
  const bool isPhysicsInit = (currentLoadState == InternalLoadingState::PhysicsInit);
  const bool isPlaying = (currentLoadState == InternalLoadingState::Play);

  bool loadDone = isPlaying;

  if (isPlaying) {
    initialLoadComplete.store(true, std::memory_order_relaxed);
  } else {
    // Determine loading UI text based on phase
    // Only call SetLoadingPhase if it's NOT already correct, to avoid progress reset.
    // SetLoadingPhase handles the check itself now.
    if (isParsing) SetLoadingPhase(LoadingPhase::Scene);
    else if (isPreallocating) SetLoadingPhase(LoadingPhase::Scene); // Still geometry
    else if (isPhysicsInit) SetLoadingPhase(LoadingPhase::Physics);
  }

  // 1. Determine next frame value and wait for the previous frame slot to be ready using our frame timeline
  // This replaces inFlightFences with a single monotonic counter and ensures proper CPU-GPU pacing.
  const uint64_t nextFrameCount = totalFrameCount.load() + 1;
  const uint64_t waitValue = (nextFrameCount > MAX_FRAMES_IN_FLIGHT) ?
    ((nextFrameCount - MAX_FRAMES_IN_FLIGHT) * 10 + TimelineMilestones::eGpuWorkFinished) : 0;

  if (waitValue > 0) {
    auto waitStart = std::chrono::steady_clock::now();
    watchdogProgressLabel.store("Render: wait frameTimeline", std::memory_order_relaxed);
    vk::SemaphoreWaitInfo waitInfo{
      .semaphoreCount = 1,
      .pSemaphores = &*frameTimeline,
      .pValues = &waitValue
    };
    // Always use a bounded timeout so the render loop never blocks forever.
    uint64_t timeoutNs = 1'000'000'000; // 1 second
    auto waitResult = device.waitSemaphores(waitInfo, timeoutNs);

    if (waitResult == vk::Result::eTimeout) {
      // GPU is too busy; skip this frame to keep the UI/engine loop responsive.
      // IMPORTANT: Do NOT advance or signal the frame timeline here. Only render UI and try again next frame.
      if (renderCallCount % 10 == 0) {
      }
      if (imguiSystem) {
        imguiSystem->EndFrameWithoutRendering();
      }
      return;
    }

    if (waitResult != vk::Result::eSuccess) {
      std::cerr << "Error: Failed to wait for frameTimeline! Result: " << vk::to_string(waitResult) << std::endl;
      if (imguiSystem) {
        imguiSystem->EndFrameWithoutRendering();
      }
      return;
    }
  }

  // Officially move to the next frame once synchronization is confirmed.
  totalFrameCount++;
  renderCallCount++;
  // Ensure currentFrame slot index is perfectly in sync with totalFrameCount
  currentFrame = (totalFrameCount.load() - 1) % MAX_FRAMES_IN_FLIGHT;
  currentTimelineValue = totalFrameCount.load() * 10;

  if (renderCallCount % 10 == 1) {
    uint32_t texSched = textureTasksScheduled.load(std::memory_order_relaxed);
    uint32_t texDone = textureTasksCompleted.load(std::memory_order_relaxed);

    // Update UI progress for background texture loading
    if (texSched > 0 && texDone < texSched) {
      // Automatically transition to Textures phase if we have jobs and aren't finishing up
      if (GetLoadingPhase() == LoadingPhase::Scene || GetLoadingPhase() == LoadingPhase::Physics) {
        SetLoadingPhase(LoadingPhase::Textures);
      }

      if (GetLoadingPhase() == LoadingPhase::Textures) {
        float progress = static_cast<float>(texDone) / static_cast<float>(texSched);
        loadingPhaseProgress.store(progress, std::memory_order_relaxed);
      }
    }
  }

  static uint64_t postLoadFrameCount = 0;
  bool isPostLoad = false;
  if (loadDone) {
    postLoadFrameCount++;
    isPostLoad = true;
  }

  // 3. Update watchdog timestamp to prove frame is progressing
  KickWatchdog();
  watchdogProgressLabel.store("Render: frame begin", std::memory_order_relaxed);

  // Suppress watchdog during heavy loading or while draining the preallocation queue
  const bool stillPreallocating = pendingEntityPreallocQueued.load(std::memory_order_relaxed);
  if (IsLoading() || stillPreallocating) {
    watchdogSuppressed.store(true, std::memory_order_relaxed);
  } else if (!asBuildRequested.load(std::memory_order_relaxed)) {
    // Only unsuppress if no background heavy tasks are pending
    watchdogSuppressed.store(false, std::memory_order_relaxed);
  }

  // Execute any pending GPU uploads
  watchdogProgressLabel.store("Render: ProcessPendingMeshUploads", std::memory_order_relaxed);
  auto pmuStart = std::chrono::steady_clock::now();
  ProcessPendingMeshUploads();

  // Drain some pending texture jobs every frame to guarantee forward progress
  watchdogProgressLabel.store("Render: ProcessPendingTextureJobs", std::memory_order_relaxed);
  ProcessPendingTextureJobs(/*maxJobs=*/16, /*includeCritical=*/true, /*includeNonCritical=*/true);

  // Execute pending entity preallocations with a time budget.
  // Chunked preallocation: at most 1 entity per frame to keep UI responsive.
  if (pendingEntityPreallocQueued.load(std::memory_order_relaxed)) {
    watchdogProgressLabel.store("Render: ProcessPendingEntityPreallocations", std::memory_order_relaxed);
    auto budgetStart = std::chrono::steady_clock::now();

    // Chunked preallocation: 1 entity per frame to keep UI responsive
    ProcessPendingEntityPreallocations();
    KickWatchdog();
  }

  // Check if we just finished the initial geometry preallocation AND all data is on GPU.
  // This must be outside the 'if (pendingEntityPreallocQueued)' block to ensure the transition
  // is evaluated correctly even if the queue becomes empty and uploads are still in-flight.
  if (currentInternalLoadingState.load(std::memory_order_relaxed) == InternalLoadingState::Preallocating &&
      !pendingEntityPreallocQueued.load(std::memory_order_relaxed) &&
      !IsSceneLoaderActive() && !HasPendingMeshUploads()) {
    currentInternalLoadingState.store(InternalLoadingState::PhysicsInit, std::memory_order_release);
    SetLoadingPhase(LoadingPhase::Physics);
    // Trigger the first AS build now that all geometry is ready
    asDevOverrideAllowRebuild = true;
    RequestAccelerationStructureBuild("Initial geometry preallocation complete");
  }

  // Transition from PhysicsInit to Play once physics/base textures are mostly ready
  if (currentInternalLoadingState.load() == InternalLoadingState::PhysicsInit) {
    static int physicsInitFrames = 0;
    if (++physicsInitFrames > 10) {
       // Wait for AS build if requested, before moving to Play
       if (!asBuildRequested.load(std::memory_order_acquire)) {
          MarkInitialLoadComplete();
          SetLoading(false);
          currentInternalLoadingState.store(InternalLoadingState::Play, std::memory_order_release);
       } else {
          // If AS build is pending, show the AccelerationStructures phase
          if (GetLoadingPhase() != LoadingPhase::AccelerationStructures) {
             SetLoadingPhase(LoadingPhase::AccelerationStructures);
          }
       }
    }
  }

  // Lock shared resources for the remainder of the render call
  // (After preallocation and uploads are processed to avoid self-deadlocks)
  std::shared_lock<std::shared_mutex> entityLock(entityResourcesMutex);
  std::shared_lock<std::shared_mutex> meshLock(meshResourcesMutex);

  if (memoryPool)
    memoryPool->setRenderingActive(true);
  struct RenderingStateGuard {
    MemoryPool* pool;
    explicit RenderingStateGuard(MemoryPool* p) : pool(p) {
    }
    ~RenderingStateGuard() {
      if (pool)
        pool->setRenderingActive(false);
    }
  } guard(memoryPool.get());

  // Track if ray query rendered successfully this frame to skip rasterization code path
  bool rayQueryRenderedThisFrame = false;

  // --- Extract lights for the frame ---
  // Build a single light list once per frame (emissive lights only for this scene)
  std::vector<ExtractedLight> lightsSubset;
  if (loadDone && camera && !staticLights.empty()) {
    lightsSubset.reserve(std::min(staticLights.size(), static_cast<size_t>(MAX_ACTIVE_LIGHTS)));
    for (const auto& L : staticLights) {
      // Include all lights (Directional, Point, Emissive) up to the limit
      lightsSubset.push_back(L);
      if (lightsSubset.size() >= MAX_ACTIVE_LIGHTS)
        break;
    }
  }
  lastFrameLightCount = static_cast<uint32_t>(lightsSubset.size());
  if (loadDone && camera && !lightsSubset.empty()) {
    updateLightStorageBuffer(currentFrame, lightsSubset, camera);
  }

  // Pre-calculate frame-constant UBO data
  if (loadDone && camera) {
    prepareFrameUboTemplate(camera);
  }

  // 2. Improved Garbage Collection using Timeline Semaphore
  // Instead of counting frames, we check if the GPU has reached the timeline value
  // from when the resource was last used.
  {
    uint64_t gpuCompletedValue = frameTimeline.getCounterValue();
    auto it = pendingASDeletions.begin();
    while (it != pendingASDeletions.end()) {
      // Check if the GPU has finished using this resource slot
      if (it->timelineValue <= gpuCompletedValue) {
        // Safe to delete
        it = pendingASDeletions.erase(it);
      } else {
        ++it;
      }
    }
  }
  watchdogProgressLabel.store("Render: after pendingASDeletions", std::memory_order_relaxed);

  // Opportunistically request AS rebuild when more meshes become ready than in the last built AS.
  // This makes the TLAS grow as streaming/allocations complete, then settle (no rebuild spam).
  // NOTE: This scan can be relatively heavy and is not needed for the default startup path.
  // Only run it when opportunistic rebuilds are enabled.
  // While loading, allow opportunistic AS rebuild scanning even if the user-facing toggle is off.
  // This prevents nondeterministic “missing outdoor props” across app restarts when the first TLAS
  // build happens before all entities exist.
  if (rayQueryEnabled && accelerationStructureEnabled && (asOpportunisticRebuildEnabled || IsLoading())) {
    // Only scan readiness periodically or during loading to avoid high CPU overhead
    static auto lastScanTime = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    const auto currentLoadState = currentInternalLoadingState.load(std::memory_order_relaxed);
    bool shouldScan = false;
    if (currentLoadState == InternalLoadingState::PhysicsInit) {
      // Allow scan in PhysicsInit with 1s interval
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastScanTime).count() > 1000) {
        shouldScan = true;
      }
    } else if (currentLoadState == InternalLoadingState::Play) {
      // 5s cooldown in Play state per specification
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastScanTime).count() >= 5000) {
        shouldScan = true;
      }
    }

    // Disable entirely during Parsing/Preallocating
    if (currentLoadState == InternalLoadingState::Parsing || currentLoadState == InternalLoadingState::Preallocating) {
      shouldScan = false;
    }

    // Skip expensive scan while heavy background preallocation is in progress
    const bool stillPreallocating = pendingEntityPreallocQueued.load(std::memory_order_relaxed);
    if (shouldScan && !stillPreallocating) {
      lastScanTime = now;
      watchdogProgressLabel.store("Render: AS readiness scan", std::memory_order_relaxed);
      size_t readyRenderableCount = 0;
      size_t readyUniqueMeshCount = 0; {
        auto lastKick = std::chrono::steady_clock::now();
        auto kickWatchdog = [&]() {
          auto now = std::chrono::steady_clock::now();
          if (now - lastKick > std::chrono::milliseconds(200)) {
            lastFrameUpdateTime.store(now, std::memory_order_relaxed);
            lastKick = now;
          }
        };

        uint32_t processedInScan = 0;
        std::unordered_map<MeshComponent *, uint32_t> meshToBLASProbe;
        for (Entity* e : entities) {
          if (++processedInScan % 100 == 0) kickWatchdog();
          if (!e || !e->IsActive())
            continue;
          // In Ray Query static-only mode, ignore dynamic/animated entities for readiness
          if (IsRayQueryStaticOnly()) {
            const std::string& nm = e->GetName();
            if (nm.find("_AnimNode_") != std::string::npos)
              continue;
            if (!nm.empty() && nm.rfind("Ball_", 0) == 0)
              continue;
          }
          auto meshComp = e->GetComponent<MeshComponent>();
          if (!meshComp)
            continue;
          try {
            auto it = meshResources.find(meshComp);
            if (it == meshResources.end())
              continue;
            const auto& res = it->second;
            // STRICT readiness: uploads must be finished (staging sizes zero)
            if (res.vertexBufferSizeBytes != 0 || res.indexBufferSizeBytes != 0)
              continue;
            if (!*res.vertexBuffer || !*res.indexBuffer)
              continue;
            if (res.indexCount == 0)
              continue;
          } catch (...) {
            continue;
          }
          readyRenderableCount++;
          if (meshToBLASProbe.find(meshComp) == meshToBLASProbe.end()) {
            meshToBLASProbe[meshComp] = static_cast<uint32_t>(meshToBLASProbe.size());
          }
        }
        readyUniqueMeshCount = meshToBLASProbe.size();
      }

      // Gate rebuilds with a readiness delta in Play state.
      // During PhysicsInit, use a very low threshold (1) to ensure the scene "fills in" before gameplay starts.
      const size_t deltaThreshold = (currentLoadState == InternalLoadingState::Play) ? 20 : 1;

      if ((!asFrozen || IsLoading()) && (readyUniqueMeshCount >= lastASBuiltBLASCount + deltaThreshold) && !asBuildRequested.load(std::memory_order_relaxed)) {
        std::cout << "AS rebuild requested: counts increased (built instances=" << lastASBuiltInstanceCount
            << ", ready instances=" << readyRenderableCount
            << ", built meshes=" << lastASBuiltBLASCount
            << ", ready meshes=" << readyUniqueMeshCount
            << ", threshold=" << deltaThreshold << ")\n";
        RequestAccelerationStructureBuild("counts increased");
      }

      // Post-load full scene repair
      if (currentLoadState == InternalLoadingState::Play && !asBuildRequested.load(std::memory_order_relaxed)) {
        const size_t targetInstances = readyRenderableCount;
        if (targetInstances > 0 && lastASBuiltInstanceCount < static_cast<size_t>(static_cast<double>(targetInstances) * 0.95)) {
          asDevOverrideAllowRebuild = true;
          std::cout << "AS rebuild requested: post-load full build repair\n";
          RequestAccelerationStructureBuild("post-load full build");
        }
      }
    }
  }

  // If in Ray Query static-only mode and TLAS not yet built post-load, request a one-time build now.
  // (Does not require a readiness scan.)
  if (rayQueryEnabled&& accelerationStructureEnabled && currentRenderMode
  ==
  RenderMode::RayQuery&& IsRayQueryStaticOnly() &&
  !IsLoading() &&
      !*tlasStructure.handle && !asBuildRequested.load(std::memory_order_relaxed)
  ) {
    RequestAccelerationStructureBuild("static-only initial build");
  }

  // Check if acceleration structure build was requested (e.g., after scene loading or counts grew)
  // Build at this safe frame point to avoid threading issues
  // Defer building for a few frames after loading to allow initial descriptor/UBO updates to settle
  bool requested = asBuildRequested.load(std::memory_order_acquire) && (!loadDone || postLoadFrameCount > 5);
  watchdogProgressLabel.store("Render: AS build request check", std::memory_order_relaxed);
  if (renderCallCount % 100 == 1 && requested) {
  }
  if (requested) {
    static bool firstLog = true;
    if (firstLog) {
      firstLog = false;
    }
    watchdogProgressLabel.store("Render: AS build request handling", std::memory_order_relaxed);

    // Defer TLAS/BLAS build while the scene loader is still in Parsing/Preallocating state
    // to avoid partial builds. We allow builds to proceed once in PhysicsInit so the
    // initial TLAS can be built before moving to Play.
    const auto currentLoadState = currentInternalLoadingState.load(std::memory_order_relaxed);
    if (currentLoadState == InternalLoadingState::Parsing ||
        currentLoadState == InternalLoadingState::Preallocating) {
      // Defer
      if (renderCallCount % 100 == 1) {
    }
    } else if (asFrozen && !asDevOverrideAllowRebuild && !IsLoading()) {
      // Ignore
      std::cout << "AS rebuild request ignored (frozen). Reason: " << lastASBuildRequestReason << "\n";
      asBuildRequested.store(false, std::memory_order_release);
      asBuildRequestStartNs.store(0, std::memory_order_relaxed);
      watchdogSuppressed.store(false, std::memory_order_relaxed);
    } else {
      // Gate initial build until readiness is high enough to represent the full scene
      size_t totalRenderableEntities = 0;
      size_t readyRenderableCount = 0;
      size_t readyUniqueMeshCount = 0;

      // OPTIMIZATION: Only do the full O(N) scan every 30 frames or if explicitly requested post-load
      static uint64_t lastScanFrame = 0;
      static size_t cachedTotal = 0;
      static size_t cachedReady = 0;
      static size_t cachedMeshes = 0;
      bool forceScan = (lastScanFrame == 0);
      bool isInitialPostLoad = (!lastASBuildRequestReason.empty() &&
                                (lastASBuildRequestReason.find("Scene loading complete") != std::string::npos ||
                                 lastASBuildRequestReason.find("Initial geometry preallocation complete") != std::string::npos));

      if (forceScan || (totalFrameCount % 30 == 0) || isInitialPostLoad) {
        std::shared_lock<std::shared_mutex> meshLock(meshResourcesMutex);
        size_t missingMeshResources = 0;
        size_t pendingUploadsCount = 0;
        size_t nullBuffersCount = 0;
        size_t zeroIndicesCount = 0; {
          auto lastKick = std::chrono::steady_clock::now();
          auto kickWatchdog = [&]() {
            auto now = std::chrono::steady_clock::now();
            if (now - lastKick > std::chrono::milliseconds(200)) {
              lastFrameUpdateTime.store(now, std::memory_order_relaxed);
              lastKick = now;
            }
          };
          std::map<MeshComponent *, uint32_t> meshToBLASProbe;
          for (Entity* e : entities) {
            kickWatchdog();
            if (!e || !e->IsActive())
              continue;
            // In Ray Query static-only mode, ignore dynamic/animated entities for totals/readiness
            if (IsRayQueryStaticOnly()) {
              const std::string& nm = e->GetName();
              if (nm.find("_AnimNode_") != std::string::npos)
                continue;
              if (!nm.empty() && nm.rfind("Ball_", 0) == 0)
                continue;
            }
            auto meshComp = e->GetComponent<MeshComponent>();
            if (!meshComp)
              continue;
            totalRenderableEntities++;
            try {
              auto it = meshResources.find(meshComp);
              if (it == meshResources.end()) {
                missingMeshResources++;
                continue;
              }
              const auto& res = it->second;
              // STRICT readiness here too: uploads finished
              if (res.vertexBufferSizeBytes != 0 || res.indexBufferSizeBytes != 0) {
                pendingUploadsCount++;
                continue;
              }
              if (!*res.vertexBuffer || !*res.indexBuffer) {
                nullBuffersCount++;
                continue;
              }
              if (res.indexCount == 0) {
                zeroIndicesCount++;
                continue;
              }
            } catch (...) {
              continue;
            }
            readyRenderableCount++;
            if (meshToBLASProbe.find(meshComp) == meshToBLASProbe.end()) {
              meshToBLASProbe[meshComp] = static_cast<uint32_t>(meshToBLASProbe.size());
            }
          }
          readyUniqueMeshCount = meshToBLASProbe.size();
        }
        cachedTotal = totalRenderableEntities;
        cachedReady = readyRenderableCount;
        cachedMeshes = readyUniqueMeshCount;
        lastScanFrame = totalFrameCount;
      } else {
        totalRenderableEntities = cachedTotal;
        readyRenderableCount = cachedReady;
        readyUniqueMeshCount = cachedMeshes;
      }

      const double readiness = (totalRenderableEntities > 0) ? static_cast<double>(readyRenderableCount) / static_cast<double>(totalRenderableEntities) : 0.0;
      double buildThreshold = 0.95; // prefer building when ~full scene is ready
      // If the build was explicitly requested after scene loading, lower the bar to avoid deadlock
      // on large scenes where uploads may still be finishing.
      if (isInitialPostLoad) {
        buildThreshold = 0.0; // Force build immediately after loading is done
        asDevOverrideAllowRebuild = true;
      } else if (!lastASBuildRequestReason.empty() && lastASBuildRequestReason.find("Scene loading complete") != std::string::npos) {
        buildThreshold = 0.10; // build with whatever is ready; we will rebuild/refit as more arrives
      }

      // Bounded deferral: avoid getting stuck forever waiting for perfect readiness.
      // After a short timeout from the original request, build with the best available data.
      const uint64_t reqNs = asBuildRequestStartNs.load(std::memory_order_relaxed);
      const uint64_t nowNs = std::chrono::steady_clock::now().time_since_epoch().count();
      const double maxDeferralSeconds = 5.0; // tighten to kick off first build faster on large scenes
      const bool deferralTimedOut = (reqNs != 0) && (nowNs > reqNs) &&
          (static_cast<double>(nowNs - reqNs) / 1'000'000'000.0) >= maxDeferralSeconds;

      // Rate limit AS rebuilds to avoid CPU/GPU starvation.
      // Use both time-based cooldown and readiness-based thresholds.
      auto currentTime = std::chrono::steady_clock::now();
      const double minRebuildInterval = IsLoading() ? 5.0 : 2.0; // conservative while loading
      const bool intervalPassed = std::chrono::duration<double>(currentTime - lastASBuildTime).count() >= minRebuildInterval;

      // Delta-based gate: only rebuild if a significant number of new meshes are ready
      const size_t readyDeltaThreshold = IsLoading() ? 20 : 5;
      const bool significantDelta = (readyUniqueMeshCount >= lastBuiltUniqueMeshCount + readyDeltaThreshold);

      // Never rebuild while heavy preallocation is active to avoid frame-time spikes.
      // Full geometry preallocation must finish before we start building AS.
      const auto currentLoadState = currentInternalLoadingState.load(std::memory_order_relaxed);
      const bool preallocActive = (currentLoadState == InternalLoadingState::Parsing ||
                                   currentLoadState == InternalLoadingState::Preallocating ||
                                   pendingEntityPreallocQueued.load(std::memory_order_relaxed));

      if (readiness < buildThreshold && !asDevOverrideAllowRebuild && !deferralTimedOut) {
        // ... defer logic ...
      } else if ((!intervalPassed && !significantDelta && !asDevOverrideAllowRebuild && !isInitialPostLoad) || preallocActive) {
        // Skip build this frame to maintain frame rate or wait for preallocation to finish.
      } else if (readyRenderableCount > 0 || (totalRenderableEntities == 0 && !*tlasStructure.handle)) {
        if (deferralTimedOut && readiness < buildThreshold && !asDevOverrideAllowRebuild) {
          std::cout << "AS build forced after " << maxDeferralSeconds
              << "s deferral (readiness " << readyRenderableCount << "/" << totalRenderableEntities
              << ", uniqueMeshesReady=" << readyUniqueMeshCount << ")\n";
        }
        struct WatchdogSuppressGuard {
          std::atomic<bool>& flag;
          explicit WatchdogSuppressGuard(std::atomic<bool>& f) : flag(f) {
            flag.store(true, std::memory_order_relaxed);
          }
          ~WatchdogSuppressGuard() {
            flag.store(false, std::memory_order_relaxed);
          }
        } watchdogGuard(watchdogSuppressed);

        // Ensure previous GPU work is complete BEFORE building AS.
        //
        // Wait for all *other* frame-in-flight fences to signal using a finite timeout loop
        // and kick the watchdog while we wait.
        // We already wait for the frameTimeline at the start of Render(),
        // which ensures the GPU has finished the previous frame's work.
        // Redundant inFlightFences wait removed to avoid deadlock with timeline-only sync.
        {
          // No-op
        }

        watchdogProgressLabel.store("Render: buildAccelerationStructures", std::memory_order_relaxed);
        if (IsLoading()) {
          SetLoadingPhase(LoadingPhase::AccelerationStructures);
        }
        if (buildAccelerationStructures(entities)) {
          watchdogProgressLabel.store("Render: after buildAccelerationStructures", std::memory_order_relaxed);
          asBuildRequested.store(false, std::memory_order_release);
          asBuildRequestStartNs.store(0, std::memory_order_relaxed);
          // AS build request resolved; restore normal watchdog sensitivity.
          watchdogSuppressed.store(false, std::memory_order_relaxed);
          // Transition the loading UI to a finalizing phase (descriptor cold-init, etc.).
          if (IsLoading()) {
            SetLoadingPhase(LoadingPhase::Finalizing);
            SetLoadingPhaseProgress(0.0f);
          }

          // The TLAS handle can transition from null -> valid (or change on rebuild).
          // Ensure raster PBR descriptor sets (set 0, binding 11 `tlas`) are rewritten after an AS build
          // so subsequent Raster draws never see an unwritten/stale acceleration-structure descriptor.
          for (auto& kv : entityResources) {
            kv.second.pbrFixedBindingsWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
          }
          for (Entity* e : entities) {
            MarkEntityDescriptorsDirty(e);
          }

          // Freeze only when the built AS covers EVERY renderable entity.
          // This ensures that subsequent streaming (if any) or late-arriving meshes can still trigger a rebuild
          // until the scene is truly 100% complete.
          if (asFreezeAfterFullBuild) {
            if (totalRenderableEntities > 0 && lastASBuiltInstanceCount >= totalRenderableEntities) {
              asFrozen = true;
            }
          }

          // One concise TLAS summary with consistent units.
          if (!!*tlasStructure.handle) {
            if (IsRayQueryStaticOnly()) {
              std::cout << "TLAS ready (static-only): tlasInstances=" << lastASBuiltTlasInstanceCount
                  << ", entities=" << lastASBuiltInstanceCount
                  << ", BLAS=" << lastASBuiltBLASCount
                  << ", addr=0x" << std::hex << tlasStructure.deviceAddress << std::dec << std::endl;
            } else {
              std::cout << "TLAS ready: tlasInstances=" << lastASBuiltTlasInstanceCount
                  << ", entities=" << lastASBuiltInstanceCount
                  << ", BLAS=" << lastASBuiltBLASCount
                  << ", addr=0x" << std::hex << tlasStructure.deviceAddress << std::dec << std::endl;
            }
          }
        } else {
          if (!accelerationStructureEnabled || !rayQueryEnabled) {
            // Permanent failure due to lack of support; do not retry.
            asBuildRequested.store(false, std::memory_order_release);
            asBuildRequestStartNs.store(0, std::memory_order_relaxed);
            watchdogSuppressed.store(false, std::memory_order_relaxed);
          } else {
            // If nothing is ready yet (e.g., mesh uploads still pending), don't spam logs.
            if (readyRenderableCount > 0 || readyUniqueMeshCount > 0) {
              std::cout << "Failed to build acceleration structures, will retry next frame" << std::endl;
            }
          }
        }
        // Reset dev override after one use
        asDevOverrideAllowRebuild = false;
      }
    }
  }

  // Safe point: the previous work referencing this frame's descriptor sets is complete.
  // Apply any deferred descriptor set updates for entities whose textures finished streaming.
  watchdogProgressLabel.store("Render: ProcessDirtyDescriptorsForFrame", std::memory_order_relaxed);
  ProcessDirtyDescriptorsForFrame(currentFrame);
  watchdogProgressLabel.store("Render: after ProcessDirtyDescriptorsForFrame", std::memory_order_relaxed);

  // --- 1. PREPARATION PASS ---
  // Gather active entities with mesh resources, perform per-frame descriptor initialization,
  // and execute culling. This single pass replaces multiple redundant scans and reduces map lookups.
  std::vector<RenderJob> opaqueJobs;
  std::vector<RenderJob> transparentJobs;
  opaqueJobs.reserve(entities.size());

  // Optimization: skip scene rendering while initial scene loading is active or no camera exists.
  // The loading overlay (rendered via ImGui at the end) is sufficient.
  if (camera && loadDone) {
    watchdogProgressLabel.store("Render: preparation pass", std::memory_order_relaxed);

    // Prepare frustum once per frame for culling
    FrustumPlanes frustum{};
    const bool doCulling = enableFrustumCulling && camera;
    if (doCulling && camera) {
      glm::mat4 proj = camera->GetProjectionMatrix();
      proj[1][1] *= -1.0f;
      const glm::mat4 vp = proj * camera->GetViewMatrix();
      frustum = extractFrustumPlanes(vp);
    }
    lastCullingVisibleCount = 0;
    lastCullingCulledCount = 0;

    uint32_t entityProcessCount = 0;
    std::vector<Entity*> activeEntities;
    activeEntities.reserve(entities.size());
    for (Entity* entity : entities) {
      if (entity && entity->IsActive()) activeEntities.push_back(entity);
    }

    uint32_t coldInitBurst = 0;
    uint32_t processedInPass = 0;
    // STAGGERED ACTIVATION: Only process a subset of entities for the first few frames
    // to avoid a massive CPU spike on the first game frame in Debug mode.
    uint64_t maxToProcess = entities.size();
    if (postLoadFrameCount < 100) {
      maxToProcess = std::min((uint64_t)entities.size(), 100 * postLoadFrameCount + 500);
    }

    for (Entity* entity : activeEntities) {
      if (++processedInPass > maxToProcess) break;

      // Kick watchdog periodically during heavy preparation pass
      if (processedInPass % 100 == 0) {
        KickWatchdog();
      }

      auto meshComponent = entity->GetComponent<MeshComponent>();
      if (!meshComponent)
        continue;

      EntityResources* pEntityRes = nullptr;
      MeshResources* pMeshRes = nullptr;
      {
        std::shared_lock<std::shared_mutex> entityLock(entityResourcesMutex);
        auto entityIt = entityResources.find(entity);
        if (entityIt != entityResources.end()) pEntityRes = &entityIt->second;
      }
      {
        std::shared_lock<std::shared_mutex> meshLock(meshResourcesMutex);
        auto meshIt = meshResources.find(meshComponent);
        if (meshIt != meshResources.end()) pMeshRes = &meshIt->second;
      }

      if (!pEntityRes || !pMeshRes)
        continue;

      EntityResources& entityRes = *pEntityRes;
      MeshResources& meshRes = *pMeshRes;

      // Ensure material cache is valid once per frame
      ensureEntityMaterialCache(entity, entityRes);

      // --- Per-frame Descriptor Cold-Init (Integrated) ---
      // OPTIMIZATION: Stagger initial creation/updates for huge scenes to avoid main-thread hangs
      // During post-load initialization, increase the burst size so pink fallback clears faster.
      const uint32_t maxColdInitPerFrame = (isPostLoad && postLoadFrameCount < 200) ? 1000 : 50;
      if (entityRes.basicDescriptorSets.empty() || entityRes.pbrDescriptorSets.empty()) {
        if (++coldInitBurst > maxColdInitPerFrame) continue;
        std::string texPath = meshComponent->GetBaseColorTexturePath();
        if (texPath.empty()) texPath = meshComponent->GetTexturePath();
        if (entityRes.basicDescriptorSets.empty()) createDescriptorSets(entity, entityRes, texPath, false);
        if (entityRes.pbrDescriptorSets.empty()) createDescriptorSets(entity, entityRes, texPath, true);
      }

      // Initialize binding 0 (UBO) for the current frame slot if not already done.
      if (!entityRes.pbrUboBindingWritten[currentFrame] || !entityRes.basicUboBindingWritten[currentFrame]) {
        if (++coldInitBurst > maxColdInitPerFrame) continue;
        std::string texPath = meshComponent->GetBaseColorTexturePath();
        if (texPath.empty()) texPath = meshComponent->GetTexturePath();
        if (!entityRes.pbrUboBindingWritten[currentFrame]) {
          updateDescriptorSetsForFrame(entity, entityRes, texPath, true, currentFrame, false, true);
        }
        if (!entityRes.basicUboBindingWritten[currentFrame]) {
          updateDescriptorSetsForFrame(entity, entityRes, texPath, false, currentFrame, false, true);
        }
      }

      // Initialize images for the current frame slot if not already done.
      if (!entityRes.pbrImagesWritten[currentFrame] || !entityRes.basicImagesWritten[currentFrame]) {
        if (++coldInitBurst > maxColdInitPerFrame) continue;
        std::string texPath = meshComponent->GetBaseColorTexturePath();
        if (texPath.empty()) texPath = meshComponent->GetTexturePath();
        if (!entityRes.pbrImagesWritten[currentFrame]) {
          updateDescriptorSetsForFrame(entity, entityRes, texPath, true, currentFrame, true, false);
          entityRes.pbrImagesWritten[currentFrame] = true;
        }
        if (!entityRes.basicImagesWritten[currentFrame]) {
          updateDescriptorSetsForFrame(entity, entityRes, texPath, false, currentFrame, true, false);
          entityRes.basicImagesWritten[currentFrame] = true;
        }
      }

      // --- Culling & Classification ---
      auto* tc = entity->GetComponent<TransformComponent>();
      bool useBlended = entityRes.cachedIsBlended;

      if (meshComponent->HasLocalAABB()) {
        const glm::mat4 model = tc ? tc->GetModelMatrix() : glm::mat4(1.0f);
        glm::vec3 wmin, wmax;
        transformAABB(model, meshComponent->GetLocalAABBMin(), meshComponent->GetLocalAABBMax(), wmin, wmax);

        // 1. Frustum Culling
        if (doCulling && !aabbIntersectsFrustum(wmin, wmax, frustum)) {
          lastCullingCulledCount++;
          continue;
        }

        // 2. Distance-based LOD
        if (enableDistanceLOD && camera) {
          glm::vec3 camPos = camera->GetPosition();
          bool cameraInside = (camPos.x >= wmin.x && camPos.x <= wmax.x &&
                               camPos.y >= wmin.y && camPos.y <= wmax.y &&
                               camPos.z >= wmin.z && camPos.z <= wmax.z);
          if (!cameraInside) {
            float dx = std::max({0.0f, wmin.x - camPos.x, camPos.x - wmax.x});
            float dy = std::max({0.0f, wmin.y - camPos.y, camPos.y - wmax.y});
            float dz = std::max({0.0f, wmin.z - camPos.z, camPos.z - wmax.z});
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            float z_eff = std::max(0.1f, dist);
            float fov = glm::radians(camera->GetFieldOfView());
            float radius = glm::length(0.5f * (wmax - wmin));
            float pixelDiameter = (radius * 2.0f * static_cast<float>(swapChainExtent.height)) / (z_eff * 2.0f * std::tan(fov * 0.5f));
            float threshold = useBlended ? lodPixelThresholdTransparent : lodPixelThresholdOpaque;
            if (pixelDiameter < threshold) {
              lastCullingCulledCount++;
              continue;
            }
          }
        }
      }

      lastCullingVisibleCount++;
      bool isAlphaMasked = false;
      if (entityRes.materialCacheValid) {
        isAlphaMasked = (entityRes.cachedMaterialProps.alphaMask > 0.5f);
      }

      // Update UBO for visible entity once per frame (shared across all main passes)
      updateUniformBuffer(currentFrame, entity, &entityRes, camera, tc);

      RenderJob job{entity, &entityRes, &meshRes, meshComponent, tc, isAlphaMasked};
      if (useBlended) {
        transparentJobs.push_back(job);
      } else {
        opaqueJobs.push_back(job);
      }

      // Update watchdog periodically
      if (++entityProcessCount % 100 == 0) {
        lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
      }
    }
    watchdogProgressLabel.store("Render: after preparation pass", std::memory_order_relaxed);
  }

  // Safe point: flush any descriptor updates that were deferred while a command buffer
  // was recording in a prior frame. Only apply ops for the current frame to avoid
  // update-after-bind on pending frames.
  if (descriptorRefreshPending.load(std::memory_order_relaxed)) {
    watchdogProgressLabel.store("Render: flush deferred descriptor ops", std::memory_order_relaxed);
    std::vector<PendingDescOp> ops; {
      std::lock_guard<std::mutex> lk(pendingDescMutex);
      ops.swap(pendingDescOps);
      descriptorRefreshPending.store(false, std::memory_order_relaxed);
    }
    uint32_t opCount = 0;
    for (auto& op : ops) {
      // Kick watchdog periodically during potentially heavy descriptor update bursts
      if ((++opCount % 50u) == 0u) {
        lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
      }

      if (op.frameIndex == currentFrame) {
        // Now not recording; safe to apply updates for this frame
        updateDescriptorSetsForFrame(op.entity, op.texPath, op.usePBR, op.frameIndex, op.imagesOnly);
      } else {
        // Keep other frame ops queued for next frame’s safe point
        std::lock_guard<std::mutex> lk(pendingDescMutex);
        pendingDescOps.push_back(op);
        descriptorRefreshPending.store(true, std::memory_order_relaxed);
      }
    }
    watchdogProgressLabel.store("Render: after deferred descriptor ops", std::memory_order_relaxed);
  }

  // Safe point: handle any pending reflection resource (re)creation and per-frame descriptor refreshes
  if (reflectionResourcesDirty) {
    if (enablePlanarReflections) {
      uint32_t rw = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.width) * reflectionResolutionScale));
      uint32_t rh = std::max(1u, static_cast<uint32_t>(static_cast<float>(swapChainExtent.height) * reflectionResolutionScale));
      createReflectionResources(rw, rh);
    } else {
      destroyReflectionResources();
    }
    reflectionResourcesDirty = false;
  }

  // Reflection descriptor binding refresh is handled elsewhere; avoid redundant per-frame mass updates here.
  // Pick the VP associated with the previous frame's reflection texture for sampling in the main pass
  if (enablePlanarReflections && !reflectionVPs.empty()) {
    uint32_t prev = (currentFrame > 0) ? (currentFrame - 1) : (static_cast<uint32_t>(reflectionVPs.size()) - 1);
    sampleReflectionVP = reflectionVPs[prev];
  }

  // This function updates bindings 6/7/8 (storage buffers) which don't have UPDATE_AFTER_BIND.
  // Updating these every frame causes "updated without UPDATE_AFTER_BIND" errors with MAX_FRAMES_IN_FLIGHT > 1.
  // These bindings are already initialized in createDescriptorSets and updated when buffers change.
  // Binding 10 (reflection map) has UPDATE_AFTER_BIND and can be updated separately if needed.
  // refreshPBRForwardPlusBindingsForFrame(currentFrame);

  // Acquire next swapchain image
  // acquireNextImage returns imageIndex (which swapchain image is available).
  // Use currentFrame to select an imageAvailableSemaphore for acquire.
  // Use imageIndex to select renderFinishedSemaphore for present (ties semaphore to the specific image).
  const uint32_t acquireSemaphoreIndex = currentFrame % static_cast<uint32_t>(imageAvailableSemaphores.size());

  uint32_t imageIndex;
  vk::Result acquireResultCode = vk::Result::eSuccess;
  // Helper overloads to normalize acquireNextImage return across Vulkan-Hpp versions
  auto extractAcquire = [](auto const& ret, vk::Result& code, uint32_t& idx) {
    using RetT = std::decay_t<decltype(ret)>;
    if constexpr (std::is_same_v<RetT, vk::ResultValue<uint32_t>>) {
      code = ret.result;
      idx = ret.value;
    } else {
      // Assume older std::pair<vk::Result, uint32_t>
      code = ret.first;
      idx = ret.second;
    }
  };
  try {
    watchdogProgressLabel.store("Render: acquireNextImage", std::memory_order_relaxed);
    // Use a 100ms timeout to avoid infinite hangs in headless/CI environments.
    // If acquire fails after 100ms, we skip the frame and return.
    auto acquireRet = swapChain.acquireNextImage(100'000'000, *imageAvailableSemaphores[acquireSemaphoreIndex]);
    // Vulkan-Hpp changed the return type of acquireNextImage for RAII swapchain across versions.
    // Support both vk::ResultValue<uint32_t> (newer) and std::pair<vk::Result, uint32_t> (older).
    extractAcquire(acquireRet, acquireResultCode, imageIndex);
  } catch (const vk::OutOfDateKHRError&) {
    watchdogProgressLabel.store("Render: acquireNextImage out-of-date", std::memory_order_relaxed);
    if (imguiSystem)
      ImGui::EndFrame();
    recreateSwapChain();
    device.signalSemaphore({.semaphore = *frameTimeline, .value = currentTimelineValue + TimelineMilestones::eGpuWorkFinished});
    return;
  }

  // imageIndex already populated above
  watchdogProgressLabel.store("Render: acquired swapchain image", std::memory_order_relaxed);

  if (acquireResultCode == vk::Result::eTimeout) {
    // Expected in headless/CI environments where the window may not be visible.
    // Return early without error so the engine loop can continue.
    if (imguiSystem)
      ImGui::EndFrame();

    // Signal the timeline even on timeout to avoid deadlocking subsequent frames
    // that wait for this frame's completion.
    {
      vk::SubmitInfo2 emptySubmit2{};
      vk::SemaphoreSubmitInfo signalInfo{
        .semaphore = *frameTimeline,
        .value = currentTimelineValue + TimelineMilestones::eGpuWorkFinished,
        .stageMask = vk::PipelineStageFlagBits2::eTopOfPipe
      };
      emptySubmit2.signalSemaphoreInfoCount = 1;
      emptySubmit2.pSignalSemaphoreInfos = &signalInfo;
      Submit2(*graphicsQueue, emptySubmit2, nullptr);
    }
    return;
  }
  if (framebufferResized.load(std::memory_order_relaxed)) {
    framebufferResized.store(false, std::memory_order_relaxed);
    if (imguiSystem)
      ImGui::EndFrame();
    recreateSwapChain();
    device.signalSemaphore({.semaphore = *frameTimeline, .value = currentTimelineValue + TimelineMilestones::eGpuWorkFinished});
    return;
  }
  if (acquireResultCode == vk::Result::eSuboptimalKHR) {
    acquireResultCode = vk::Result::eSuccess;
  }
  if (acquireResultCode != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to acquire swap chain image");
  }


  // Perform any descriptor updates that must not happen during command buffer recording
  if (useForwardPlus) {
    uint32_t tilesX_pre = (swapChainExtent.width + forwardPlusTileSizeX - 1) / forwardPlusTileSizeX;
    uint32_t tilesY_pre = (swapChainExtent.height + forwardPlusTileSizeY - 1) / forwardPlusTileSizeY;
    // Only update current frame's descriptors to avoid touching in-flight frames
    createOrResizeForwardPlusBuffers(tilesX_pre, tilesY_pre, forwardPlusSlicesZ, /*updateOnlyCurrentFrame=*/true);
    // After (re)creating Forward+ buffers, bindings 7/8 will be refreshed as needed.
  }

  // Ensure light buffers are sufficiently large before recording to avoid resizing while in use
  {
    // Reserve capacity based on emissive lights only (punctual lights disabled for now)
    size_t desiredLightCapacity = 0;
    if (!staticLights.empty()) {
      size_t emissiveCount = 0;
      for (const auto& L : staticLights) {
        if (L.type == ExtractedLight::Type::Emissive) {
          ++emissiveCount;
          if (emissiveCount >= MAX_ACTIVE_LIGHTS)
            break;
        }
      }
      desiredLightCapacity = emissiveCount;
    }
    if (desiredLightCapacity > 0) {
      createOrResizeLightStorageBuffers(desiredLightCapacity);
      // Ensure compute (binding 0) sees the current frame's lights buffer
      refreshForwardPlusComputeLightsBindingForFrame(currentFrame);
      // Bindings 6/7/8 for PBR are refreshed only when buffers change (handled in resize path).
    }
  }

  // Safe point: Update ray query descriptor sets if ray query mode is active
  // This MUST happen before command buffer recording starts to avoid "descriptor updated without UPDATE_AFTER_BIND" errors
  if (currentRenderMode == RenderMode::RayQuery && rayQueryEnabled && accelerationStructureEnabled) {
    if (!!*tlasStructure.handle) {
      watchdogProgressLabel.store("Render: updateRayQueryDescriptorSets", std::memory_order_relaxed);
      updateRayQueryDescriptorSets(currentFrame, entities);
      watchdogProgressLabel.store("Render: after updateRayQueryDescriptorSets", std::memory_order_relaxed);
    }
  }

  // Refit TLAS if needed (either for Ray Query mode or for Raster shadows)
  // Skip during initial 20 post-load frames to ensure smooth verification.
  const bool needTLAS = (currentRenderMode == RenderMode::RayQuery || enableRasterRayQueryShadows) && accelerationStructureEnabled;
  if (needTLAS && !!*tlasStructure.handle && postLoadFrameCount > 20) {
    if (!IsRayQueryStaticOnly()) {
      watchdogProgressLabel.store("Render: refitTopLevelAS", std::memory_order_relaxed);
      refitTopLevelAS(entities, camera);
    }
  }

  commandBuffers[currentFrame].reset();
  // Begin command buffer recording for this frame
  commandBuffers[currentFrame].begin(vk::CommandBufferBeginInfo());
  isRecordingCmd.store(true, std::memory_order_relaxed);


  // Ray query rendering mode dispatch
  if (currentRenderMode == RenderMode::RayQuery && rayQueryEnabled && accelerationStructureEnabled) {
    // Check if TLAS handle is valid (dereference RAII handle)
    if (!*tlasStructure.handle) {
      // TLAS not built yet.
      // During loading, allow the raster path (and the progress overlay) to render normally
      // instead of presenting a diagnostic magenta frame.
      if (!IsLoading()) {
        // If we are in Ray Query mode but AS is not built yet, don't just show magenta.
        // Fall back to Rasterization so the user sees something while the background build proceeds.
        static auto lastPinkLog = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastPinkLog).count() >= 5) {
          std::cout << "Ray Query active but TLAS not ready; falling back to Rasterization for this frame." << std::endl;
          lastPinkLog = now;
        }
        rayQueryRenderedThisFrame = false; // Proceed to raster path
      }
    } else {
      // TLAS is valid and descriptor sets were already updated at safe point
      // Proceed with ray query rendering
      // Bind ray query compute pipeline
      commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, *rayQueryPipeline);

      // Bind descriptor set
      commandBuffers[currentFrame].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        *rayQueryPipelineLayout,
        0,
        *rayQueryDescriptorSets[currentFrame],
        nullptr);

      // This dedicated UBO is separate from entity UBOs and uses a Ray Query-specific layout.
      if (rayQueryUniformBuffersMapped.size() > currentFrame && rayQueryUniformBuffersMapped[currentFrame]) {
        RayQueryUniformBufferObject ubo{};
        ubo.model = glm::mat4(1.0f); // Identity - not used for ray query

        // Force view matrix update to reflect current camera position
        // (the dirty flag isn't automatically set when camera position changes)
        camera->ForceViewMatrixUpdate();

        // Get camera matrices
        glm::mat4 camView = camera->GetViewMatrix();
        ubo.view = camView;
        ubo.proj = camera->GetProjectionMatrix();
        ubo.proj[1][1] *= -1; // Flip Y for Vulkan
        ubo.camPos = glm::vec4(camera->GetPosition(), 1.0f);
        // Clamp to sane ranges to avoid black output (exposure=0 → 1-exp(0)=0)
        ubo.exposure = std::clamp(exposure, 0.2f, 4.0f);
        ubo.gamma = std::clamp(gamma, 1.6f, 2.6f);
        // Match raster convention: ambient scale factor for simple IBL/ambient term.
        // (Raster defaults to ~1.0 in the main pass; keep Ray Query consistent.)
        ubo.scaleIBLAmbient = 1.0f;
        // Provide the per-frame light count so the ray query shader can iterate lights.
        ubo.lightCount = static_cast<int>(lastFrameLightCount);
        ubo.screenDimensions = glm::vec2(swapChainExtent.width, swapChainExtent.height);
        ubo.enableRayQueryReflections = enableRayQueryReflections ? 1 : 0;
        ubo.enableRayQueryTransparency = enableRayQueryTransparency ? 1 : 0;
        // Max secondary bounces (reflection/refraction). Stored in the padding slot to avoid UBO layout churn.
        // Shader clamps this value.
        ubo._pad0 = rayQueryMaxBounces;
        // Thick-glass toggles and tuning
        ubo.enableThickGlass = enableThickGlass ? 1 : 0;
        ubo.thicknessClamp = thickGlassThicknessClamp;
        ubo.absorptionScale = thickGlassAbsorptionScale;
        // Ray Query hard shadows (see `shaders/ray_query.slang`)
        ubo._pad1 = enableRayQueryShadows ? 1 : 0;
        ubo.shadowSampleCount = std::clamp(rayQueryShadowSampleCount, 1, 32);
        ubo.shadowSoftness = std::clamp(rayQueryShadowSoftness, 0.0f, 1.0f);
        ubo.reflectionIntensity = reflectionIntensity;
        // Provide geometry info count for shader-side bounds checking (per-instance)
        ubo.geometryInfoCount = static_cast<int>(tlasInstanceCount);
        // Provide material buffer count for shader-side bounds checking
        ubo.materialCount = static_cast<int>(materialCountCPU);

        // Copy to mapped memory
        std::memcpy(rayQueryUniformBuffersMapped[currentFrame], &ubo, sizeof(RayQueryUniformBufferObject));
      } else {
        // Keep concise error for visibility
        std::cerr << "Ray Query UBO not mapped for frame " << currentFrame << "\n";
      }

      // Dispatch compute shader (8x8 workgroups as defined in shader)
      uint32_t workgroupsX = (swapChainExtent.width + 7) / 8;
      uint32_t workgroupsY = (swapChainExtent.height + 7) / 8;
      commandBuffers[currentFrame].dispatch(workgroupsX, workgroupsY, 1);

      // Barrier: wait for compute shader to finish writing to output image,
      // then make it readable by fragment shader for sampling in composite pass
      vk::ImageMemoryBarrier2 rqToSample{};
      rqToSample.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
      rqToSample.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
      rqToSample.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
      rqToSample.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
      rqToSample.oldLayout = vk::ImageLayout::eGeneral;
      rqToSample.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      rqToSample.image = *rayQueryOutputImage;
      rqToSample.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
      rqToSample.subresourceRange.levelCount = 1;
      rqToSample.subresourceRange.layerCount = 1;

      vk::DependencyInfo depRQToSample{};
      depRQToSample.imageMemoryBarrierCount = 1;
      depRQToSample.pImageMemoryBarriers = &rqToSample;
      commandBuffers[currentFrame].pipelineBarrier2(depRQToSample);

      // Composite fullscreen: sample rayQueryOutputImage to the swapchain using the composite pipeline
      // Transition swapchain image to COLOR_ATTACHMENT_OPTIMAL
      vk::ImageMemoryBarrier2 swapchainToColor{};
      swapchainToColor.srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
      swapchainToColor.srcAccessMask = vk::AccessFlagBits2::eNone;
      swapchainToColor.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
      swapchainToColor.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead;
      swapchainToColor.oldLayout = (imageIndex < swapChainImageLayouts.size()) ? swapChainImageLayouts[imageIndex] : vk::ImageLayout::eUndefined;
      swapchainToColor.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
      swapchainToColor.image = swapChainImages[imageIndex];
      swapchainToColor.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
      swapchainToColor.subresourceRange.levelCount = 1;
      swapchainToColor.subresourceRange.layerCount = 1;
      vk::DependencyInfo depSwapToColor{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &swapchainToColor};
      commandBuffers[currentFrame].pipelineBarrier2(depSwapToColor);
      if (imageIndex < swapChainImageLayouts.size())
        swapChainImageLayouts[imageIndex] = swapchainToColor.newLayout;

      // Begin dynamic rendering for composite (no depth)
      colorAttachments[0].imageView = *swapChainImageViews[imageIndex];
      colorAttachments[0].loadOp = vk::AttachmentLoadOp::eClear;
      depthAttachment.loadOp = vk::AttachmentLoadOp::eDontCare;
      renderingInfo.renderArea = vk::Rect2D({0, 0}, swapChainExtent);
      auto savedDepthPtr2 = renderingInfo.pDepthAttachment;
      renderingInfo.pDepthAttachment = nullptr;
      commandBuffers[currentFrame].beginRendering(renderingInfo);

      if (!!*compositePipeline) {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *compositePipeline);
      }
      vk::Viewport vp(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
      vk::Rect2D sc({0, 0}, swapChainExtent);
      commandBuffers[currentFrame].setViewport(0, vp);
      commandBuffers[currentFrame].setScissor(0, sc);

      // Bind the RQ composite descriptor set (samples rayQueryOutputImage)
      if (!rqCompositeDescriptorSets.empty()) {
        commandBuffers[currentFrame].bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          *compositePipelineLayout,
          0,
          {*rqCompositeDescriptorSets[currentFrame]},
          {});
      }

      // Push exposure/gamma and sRGB flag
      struct CompositePush {
        float exposure;
        float gamma;
        int outputIsSRGB;
        float _pad;
      } pc2{};
      pc2.exposure = std::clamp(this->exposure, 0.2f, 4.0f);
      pc2.gamma = this->gamma;
      pc2.outputIsSRGB = (swapChainImageFormat == vk::Format::eR8G8B8A8Srgb || swapChainImageFormat == vk::Format::eB8G8R8A8Srgb) ? 1 : 0;
      commandBuffers[currentFrame].pushConstants<CompositePush>(*compositePipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, pc2);

      commandBuffers[currentFrame].draw(3, 1, 0, 0);
      commandBuffers[currentFrame].endRendering();
      renderingInfo.pDepthAttachment = savedDepthPtr2;

      // Transition swapchain back to PRESENT and RQ image back to GENERAL for next frame
      vk::ImageMemoryBarrier2 swapchainToPresent{};
      swapchainToPresent.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
      swapchainToPresent.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
      swapchainToPresent.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
      swapchainToPresent.dstAccessMask = vk::AccessFlagBits2::eNone;
      swapchainToPresent.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
      swapchainToPresent.newLayout = vk::ImageLayout::ePresentSrcKHR;
      swapchainToPresent.image = swapChainImages[imageIndex];
      swapchainToPresent.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
      swapchainToPresent.subresourceRange.levelCount = 1;
      swapchainToPresent.subresourceRange.layerCount = 1;

      vk::ImageMemoryBarrier2 rqBackToGeneral{};
      rqBackToGeneral.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
      rqBackToGeneral.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
      rqBackToGeneral.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
      rqBackToGeneral.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;
      rqBackToGeneral.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      rqBackToGeneral.newLayout = vk::ImageLayout::eGeneral;
      rqBackToGeneral.image = *rayQueryOutputImage;
      rqBackToGeneral.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
      rqBackToGeneral.subresourceRange.levelCount = 1;
      rqBackToGeneral.subresourceRange.layerCount = 1;

      std::array<vk::ImageMemoryBarrier2, 2> barriers{swapchainToPresent, rqBackToGeneral};
      vk::DependencyInfo depEnd{.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()), .pImageMemoryBarriers = barriers.data()};
      commandBuffers[currentFrame].pipelineBarrier2(depEnd);
      if (imageIndex < swapChainImageLayouts.size())
        swapChainImageLayouts[imageIndex] = swapchainToPresent.newLayout;

      // Ray query rendering complete - set flag to skip rasterization code path
      rayQueryRenderedThisFrame = true;
    }
  }

  // Process texture streaming uploads (see Renderer::ProcessPendingTextureJobs)

  vk::raii::Pipeline* currentPipeline = nullptr;
  vk::raii::PipelineLayout* currentLayout = nullptr;

  // Incrementally process pending texture uploads on the main thread so that
  // all Vulkan submits happen from a single place while worker threads only
  // handle CPU-side decoding. While the loading screen is up, prioritize
  // critical textures so the first rendered frame looks mostly correct.
  if (IsLoading()) {
    // Larger budget while loading screen is visible so we don't stall
    // streaming of near-field baseColor textures.
    ProcessPendingTextureJobs(/*maxJobs=*/16, /*includeCritical=*/true, /*includeNonCritical=*/false);
  } else {
    // After loading screen disappears, we want the scene to remain
    // responsive (~20 fps) while textures stream in. Limit the number
    // of non-critical uploads per frame so we don't tank frame time.
    static uint32_t streamingFrameCounter = 0;
    streamingFrameCounter++;
    // Ray Query needs textures visible quickly; process more streaming work when in Ray Query mode.
    if (currentRenderMode == RenderMode::RayQuery) {
      // Aggressively drain both critical and non-critical queues each frame for faster bring-up.
      ProcessPendingTextureJobs(/*maxJobs=*/32, /*includeCritical=*/true, /*includeNonCritical=*/true);
    } else {
      // Raster path: keep previous throttling to avoid stalls.
      if ((streamingFrameCounter % 3) == 0) {
        ProcessPendingTextureJobs(/*maxJobs=*/1, /*includeCritical=*/false, /*includeNonCritical=*/true);
      }
    }
  }

  // Renderer UI - available for both ray query and rasterization modes.
  // Hide UI during loading; the progress overlay is handled by ImGuiSystem::NewFrame().
  if (imguiSystem && !imguiSystem->IsFrameRendered() && !IsLoading()) {
    if (ImGui::Begin("Renderer")) {
      // Declare variables that need to persist across conditional blocks
      bool prevFwdPlus = useForwardPlus;

      // === RENDERING MODE SELECTION (TOP) ===
      ImGui::Text("Rendering Mode:");
      if (rayQueryEnabled && accelerationStructureEnabled) {
        const char* modeNames[] = {"Rasterization", "Ray Query"};
        int currentMode = (currentRenderMode == RenderMode::RayQuery) ? 1 : 0;
        if (ImGui::Combo("Mode", &currentMode, modeNames, 2)) {
          RenderMode newMode = (currentMode == 1) ? RenderMode::RayQuery : RenderMode::Rasterization;
          if (newMode != currentRenderMode) {
            currentRenderMode = newMode;
            std::cout << "Switched to " << modeNames[currentMode] << " mode\n";

            // Request acceleration structure build when switching to ray query mode
            if (currentRenderMode == RenderMode::RayQuery) {
              std::cout << "Requesting acceleration structure build...\n";
              RequestAccelerationStructureBuild();
            }

            // Switching modes can change which pipelines are bound and whether ray-query-dependent
            // descriptor bindings (e.g., PBR binding 11 `tlas`) become statically used.
            // Mark entity descriptor sets dirty so the next safe point refreshes bindings for this frame.
            for (auto& kv : entityResources) {
              kv.second.pbrFixedBindingsWritten.assign(MAX_FRAMES_IN_FLIGHT, false);
            }
            for (Entity* e : entities) {
              MarkEntityDescriptorsDirty(e);
            }
          }
        }
      } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Rasterization only (ray query not supported)");
      }

      // === RASTERIZATION-SPECIFIC OPTIONS ===
      if (currentRenderMode == RenderMode::Rasterization) {
        ImGui::Separator();
        ImGui::Text("Rasterization Options:");

        // Lighting Controls - BRDF/PBR is now the default lighting model
        bool useBasicLighting = imguiSystem && !imguiSystem->IsPBREnabled();
        if (ImGui::Checkbox("Use Basic Lighting (Phong)", &useBasicLighting)) {
          imguiSystem->SetPBREnabled(!useBasicLighting);
          std::cout << "Lighting mode: " << (!useBasicLighting ? "BRDF/PBR (default)" : "Basic Phong") << std::endl;
        }

        if (!useBasicLighting) {
          ImGui::Text("Status: BRDF/PBR pipeline active (default)");
          ImGui::Text("All models rendered with physically-based lighting");
        } else {
          ImGui::Text("Status: Basic Phong pipeline active");
          ImGui::Text("All models rendered with basic Phong shading");
        }

        ImGui::Checkbox("Forward+ (tiled light culling)", &useForwardPlus);
        if (useForwardPlus && !prevFwdPlus) {
          // Lazily create Forward+ resources if enabled at runtime
          if (!*forwardPlusPipeline || !*forwardPlusDescriptorSetLayout || forwardPlusPerFrame.empty()) {
            createForwardPlusPipelinesAndResources();
          }
          if (!*depthPrepassPipeline) {
            createDepthPrepassPipeline();
          }
        }

        // Raster shadows via ray queries (experimental)
        if (rayQueryEnabled && accelerationStructureEnabled) {
          ImGui::Checkbox("RayQuery shadows (raster)", &enableRasterRayQueryShadows);
        } else {
          ImGui::TextDisabled("RayQuery shadows (raster) (requires ray query + AS)");
        }

        // Planar reflections controls
        ImGui::Spacing();
        /*
        if (ImGui::Checkbox("Planar reflections (experimental)", &enablePlanarReflections)) {
          // Defer actual (re)creation/destruction to the next safe point at frame start
          reflectionResourcesDirty = true;
        }
        */
        enablePlanarReflections = false;
        float scaleBefore = reflectionResolutionScale;
        if (ImGui::SliderFloat("Reflection resolution scale", &reflectionResolutionScale, 0.25f, 1.0f, "%.2f")) {
          reflectionResolutionScale = std::clamp(reflectionResolutionScale, 0.25f, 1.0f);
          if (enablePlanarReflections&& std::abs(scaleBefore - reflectionResolutionScale)
          >
          1e-3f
          ) {
            reflectionResourcesDirty = true;
          }
        }
        if (enablePlanarReflections && !reflections.empty()) {
          auto& rt = reflections[currentFrame];
          if (rt.width > 0) {
            ImGui::Text("Reflection RT: %ux%u", rt.width, rt.height);
          }
        }
      }

      // === RAY QUERY-SPECIFIC OPTIONS ===
      if (currentRenderMode == RenderMode::RayQuery && rayQueryEnabled && accelerationStructureEnabled) {
        ImGui::Separator();
        ImGui::Text("Ray Query Status:");

        // Show acceleration structure status
        if (!!*tlasStructure.handle) {
          ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Acceleration Structures: Built (%zu meshes)", blasStructures.size());
        } else {
          ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Acceleration Structures: Not built");
        }

        ImGui::Spacing();
        ImGui::Text("Ray Query Features:");
        ImGui::Checkbox("Enable Hard Shadows", &enableRayQueryShadows);
        if (enableRayQueryShadows) {
          ImGui::SliderInt("Shadow samples", &rayQueryShadowSampleCount, 1, 32);
          ImGui::SliderFloat("Shadow softness (fraction of range)", &rayQueryShadowSoftness, 0.0f, 0.2f, "%.3f");
        }
        ImGui::Checkbox("Enable Reflections", &enableRayQueryReflections);
        ImGui::Checkbox("Enable Transparency/Refraction", &enableRayQueryTransparency);
        ImGui::SliderInt("Max secondary bounces", &rayQueryMaxBounces, 0, 10);
        // Thick-glass realism controls
        ImGui::Separator();
        ImGui::Text("Thick Glass");
        ImGui::Checkbox("Enable Thick Glass", &enableThickGlass);
        ImGui::SliderFloat("Thickness Clamp (m)", &thickGlassThicknessClamp, 0.0f, 0.5f, "%.3f");
        ImGui::SliderFloat("Absorption Scale", &thickGlassAbsorptionScale, 0.0f, 4.0f, "%.2f");
      }

      // === SHARED OPTIONS (BOTH MODES) ===
      ImGui::Separator();
      ImGui::Text("Culling & LOD:");
      if (ImGui::Checkbox("Frustum culling", &enableFrustumCulling)) {
        // no-op, takes effect immediately
      }
      if (ImGui::Checkbox("Distance LOD (projected-size skip)", &enableDistanceLOD)) {
      }
      ImGui::SliderFloat("LOD threshold opaque (px)", &lodPixelThresholdOpaque, 0.5f, 8.0f, "%.1f");
      ImGui::SliderFloat("LOD threshold transparent (px)", &lodPixelThresholdTransparent, 0.5f, 12.0f, "%.1f");
      // Anisotropy control (recreate samplers on change)
      {
        float deviceMaxAniso = physicalDevice.getProperties().limits.maxSamplerAnisotropy;
        if (ImGui::SliderFloat("Sampler max anisotropy", &samplerMaxAnisotropy, 1.0f, deviceMaxAniso, "%.1f")) {
          // Recreate samplers for all textures to apply new anisotropy
          std::unique_lock<std::shared_mutex> texLock(textureResourcesMutex);
          for (auto& kv : textureResources) {
            createTextureSampler(kv.second);
          }
          // Default texture
          createTextureSampler(defaultTextureResources);
        }
      }
      if (lastCullingVisibleCount + lastCullingCulledCount > 0) {
        ImGui::Text("Culling: visible=%u, culled=%u", lastCullingVisibleCount, lastCullingCulledCount);
      }

      // Basic tone mapping controls
      ImGui::Separator();
      ImGui::Text("Tone Mapping & Tuning:");
      ImGui::SliderFloat("Reflection intensity", &reflectionIntensity, 0.0f, 2.0f, "%.2f");
      ImGui::SliderFloat("Exposure", &exposure, 0.1f, 4.0f, "%.2f");
      ImGui::SliderFloat("Gamma", &gamma, 1.6f, 2.6f, "%.2f");
    }
    ImGui::End();
  }

  // Rasterization rendering: only execute if ray query did not render this frame.
  if (!rayQueryRenderedThisFrame) {
    // Optional: render planar reflections first
    /*
    if (enablePlanarReflections) {
      glm::vec4 planeWS(0.0f, 1.0f, 0.0f, 0.0f);
      renderReflectionPass(commandBuffers[currentFrame], planeWS, camera, opaqueJobs);
    }
    */

    // Sort transparent entities back-to-front for correct blending of nested glass/liquids
    if (!transparentJobs.empty()) {
      glm::vec3 camPos = camera ? camera->GetPosition() : glm::vec3(0.0f);
      std::ranges::sort(transparentJobs,
                        [camPos](const RenderJob& a, const RenderJob& b) {
                          glm::vec3 pa = a.transformComp ? a.transformComp->GetPosition() : glm::vec3(0.0f);
                          glm::vec3 pb = b.transformComp ? b.transformComp->GetPosition() : glm::vec3(0.0f);
                          float da2 = glm::length2(pa - camPos);
                          float db2 = glm::length2(pb - camPos);
                          if (da2 != db2) return da2 > db2;
                          if (a.entityRes->cachedIsLiquid != b.entityRes->cachedIsLiquid) return a.entityRes->cachedIsLiquid;
                          return a.entity < b.entity;
                        });
    }


    // Track whether we executed a depth pre-pass this frame (used to choose depth load op and pipeline state)
    bool didOpaqueDepthPrepass = false;

    // Optional Forward+ depth pre-pass for opaque geometry
    if (useForwardPlus) {
      if (!opaqueJobs.empty()) {
        // Transition depth image for attachment write (Sync2)
        vk::ImageMemoryBarrier2 depthBarrier2{
          .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
          .srcAccessMask = {},
          .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
          .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          .oldLayout = vk::ImageLayout::eUndefined,
          .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = *depthImage,
          .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
        };
        vk::DependencyInfo depInfoDepth{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthBarrier2};
        commandBuffers[currentFrame].pipelineBarrier2(depInfoDepth);

        // Depth-only rendering
        vk::RenderingAttachmentInfo depthOnlyAttachment{.imageView = *depthImageView, .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal, .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore, .clearValue = vk::ClearDepthStencilValue{1.0f, 0}};
        vk::RenderingInfo depthOnlyInfo{.renderArea = vk::Rect2D({0, 0}, swapChainExtent), .layerCount = 1, .colorAttachmentCount = 0, .pColorAttachments = nullptr, .pDepthAttachment = &depthOnlyAttachment};
        commandBuffers[currentFrame].beginRendering(depthOnlyInfo);
        vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
        commandBuffers[currentFrame].setViewport(0, viewport);
        vk::Rect2D scissor({0, 0}, swapChainExtent);
        commandBuffers[currentFrame].setScissor(0, scissor);

        // Bind depth pre-pass pipeline
        if (!!*depthPrepassPipeline) {
          commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *depthPrepassPipeline);
        }

        for (const auto& job : opaqueJobs) {
          if (job.isAlphaMasked) continue;

          // Bind geometry
          std::array<vk::Buffer, 2> buffers = {*job.meshRes->vertexBuffer, *job.entityRes->instanceBuffer};
          std::array<vk::DeviceSize, 2> offsets = {0, 0};
          commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
          commandBuffers[currentFrame].bindIndexBuffer(*job.meshRes->indexBuffer, 0, vk::IndexType::eUint32);

          // Bind descriptor set (PBR set 0)
          commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                                           *pbrPipelineLayout,
                                                           0,
                                                           *job.entityRes->pbrDescriptorSets[currentFrame],
                                                           nullptr);

          // Issue draw
          uint32_t instanceCount = std::max(1u, static_cast<uint32_t>(job.meshComp->GetInstanceCount()));
          commandBuffers[currentFrame].drawIndexed(job.meshRes->indexCount, instanceCount, 0, 0, 0);
        }

        commandBuffers[currentFrame].endRendering();

        // Barrier to ensure depth is visible for subsequent passes (Sync2)
        vk::ImageMemoryBarrier2 depthToRead2{
          .srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests,
          .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
          .dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
          .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead,
          .oldLayout = vk::ImageLayout::eDepthAttachmentOptimal,
          .newLayout = vk::ImageLayout::eDepthAttachmentOptimal,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = *depthImage,
          .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
        };
        vk::DependencyInfo depInfoDepthToRead{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &depthToRead2};
        commandBuffers[currentFrame].pipelineBarrier2(depInfoDepthToRead);

        didOpaqueDepthPrepass = true;
      }

      // Forward+ compute culling based on current camera and screen tiles
      uint32_t tilesX = (swapChainExtent.width + forwardPlusTileSizeX - 1) / forwardPlusTileSizeX;
      uint32_t tilesY = (swapChainExtent.height + forwardPlusTileSizeY - 1) / forwardPlusTileSizeY;

      // Lights already extracted at frame start - use lastFrameLightCount for Forward+ params
      glm::mat4 view = camera->GetViewMatrix();
      glm::mat4 proj = camera->GetProjectionMatrix();
      proj[1][1] *= -1.0f;
      float nearZ = camera->GetNearPlane();
      float farZ = camera->GetFarPlane();
      updateForwardPlusParams(currentFrame, view, proj, lastFrameLightCount, tilesX, tilesY, forwardPlusSlicesZ, nearZ, farZ);
      // As a last guard before dispatch, make sure compute binding 0 is valid for this frame
      refreshForwardPlusComputeLightsBindingForFrame(currentFrame);

      dispatchForwardPlus(commandBuffers[currentFrame], tilesX, tilesY, forwardPlusSlicesZ);
    }

    // PASS 1: RENDER OPAQUE OBJECTS TO OFF-SCREEN TEXTURE
    // Transition off-screen color to attachment write (Sync2). On first use after creation or after switching
    // from a mode that never produced this image, the layout may still be UNDEFINED.
    vk::ImageLayout oscOldLayout = vk::ImageLayout::eUndefined;
    vk::PipelineStageFlags2 oscSrcStage = vk::PipelineStageFlagBits2::eTopOfPipe;
    vk::AccessFlags2 oscSrcAccess = vk::AccessFlagBits2::eNone;
    if (currentFrame < opaqueSceneColorImageLayouts.size()) {
      oscOldLayout = opaqueSceneColorImageLayouts[currentFrame];
      if (oscOldLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        oscSrcStage = vk::PipelineStageFlagBits2::eFragmentShader;
        oscSrcAccess = vk::AccessFlagBits2::eShaderRead;
      } else if (oscOldLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        oscSrcStage = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        oscSrcAccess = vk::AccessFlagBits2::eColorAttachmentWrite;
      } else {
        oscOldLayout = vk::ImageLayout::eUndefined;
        oscSrcStage = vk::PipelineStageFlagBits2::eTopOfPipe;
        oscSrcAccess = vk::AccessFlagBits2::eNone;
      }
    }
    vk::ImageMemoryBarrier2 oscToColor2{
      .srcStageMask = oscSrcStage,
      .srcAccessMask = oscSrcAccess,
      .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
      .oldLayout = oscOldLayout,
      .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = *opaqueSceneColorImages[currentFrame],
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    };
    vk::DependencyInfo depOscToColor{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &oscToColor2};
    commandBuffers[currentFrame].pipelineBarrier2(depOscToColor);
    if (currentFrame < opaqueSceneColorImageLayouts.size()) {
      opaqueSceneColorImageLayouts[currentFrame] = vk::ImageLayout::eColorAttachmentOptimal;
    }
    // PASS 1: OFF-SCREEN COLOR (Opaque)
    // Clear the off-screen target at the start of opaque rendering to a neutral black background
    vk::RenderingAttachmentInfo colorAttachment{.imageView = *opaqueSceneColorImageViews[currentFrame], .imageLayout = vk::ImageLayout::eColorAttachmentOptimal, .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore, .clearValue = vk::ClearColorValue(std::array < float, 4 >{0.0f, 0.0f, 0.0f, 1.0f})};
    depthAttachment.imageView = *depthImageView;
    depthAttachment.loadOp = (didOpaqueDepthPrepass) ? vk::AttachmentLoadOp::eLoad : vk::AttachmentLoadOp::eClear;
    vk::RenderingInfo passInfo{.renderArea = vk::Rect2D({0, 0}, swapChainExtent), .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment, .pDepthAttachment = &depthAttachment};
    commandBuffers[currentFrame].beginRendering(passInfo);
    vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
    commandBuffers[currentFrame].setViewport(0, viewport);
    vk::Rect2D scissor({0, 0}, swapChainExtent);
    commandBuffers[currentFrame].setScissor(0, scissor); {
      uint32_t opaqueDrawsThisPass = 0;
      for (const auto& job : opaqueJobs) {
        bool useBasic = (imguiSystem && !imguiSystem->IsPBREnabled());
        vk::raii::Pipeline* selectedPipeline = nullptr;
        vk::raii::PipelineLayout* selectedLayout = nullptr;
        if (useBasic) {
          selectedPipeline = &graphicsPipeline;
          selectedLayout = &pipelineLayout;
        } else {
          // If masked, we need depth writes with alpha test; otherwise, after-prepass read-only is fine.
          if (job.isAlphaMasked) {
            selectedPipeline = &pbrGraphicsPipeline; // writes depth, compare Less
          } else {
            selectedPipeline = didOpaqueDepthPrepass && !!*pbrPrepassGraphicsPipeline ? &pbrPrepassGraphicsPipeline : &pbrGraphicsPipeline;
          }
          selectedLayout = &pbrPipelineLayout;
        }
        if (currentPipeline != selectedPipeline) {
          commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, **selectedPipeline);
          currentPipeline = selectedPipeline;
          currentLayout = selectedLayout;
        }

        std::array<vk::Buffer, 2> buffers = {*job.meshRes->vertexBuffer, *job.entityRes->instanceBuffer};
        std::array<vk::DeviceSize, 2> offsets = {0, 0};
        commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
        commandBuffers[currentFrame].bindIndexBuffer(*job.meshRes->indexBuffer, 0, vk::IndexType::eUint32);

        auto* descSetsPtr = useBasic ? &job.entityRes->basicDescriptorSets : &job.entityRes->pbrDescriptorSets;
        if (descSetsPtr->empty() || currentFrame >= descSetsPtr->size()) {
          continue;
        }

        if (useBasic) {
          commandBuffers[currentFrame].bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            **selectedLayout,
            0,
            {*(*descSetsPtr)[currentFrame]},
            {});
        } else {
          vk::DescriptorSet set1Opaque = (transparentDescriptorSets.empty() || IsLoading())
                                           ? *transparentFallbackDescriptorSets[currentFrame]
                                           : *transparentDescriptorSets[currentFrame];
          commandBuffers[currentFrame].bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            **selectedLayout,
            0,
            {*(*descSetsPtr)[currentFrame], set1Opaque},
            {});

          commandBuffers[currentFrame].pushConstants<MaterialProperties>(**selectedLayout, vk::ShaderStageFlagBits::eFragment, 0, {job.entityRes->cachedMaterialProps});
        }
        uint32_t instanceCount = std::max(1u, static_cast<uint32_t>(job.meshComp->GetInstanceCount()));
        commandBuffers[currentFrame].drawIndexed(job.meshRes->indexCount, instanceCount, 0, 0, 0);
        ++opaqueDrawsThisPass;
      }
    }
    commandBuffers[currentFrame].endRendering();
    // PASS 1b: PRESENT – composite path
    {
      // Transition off-screen to SHADER_READ for sampling (Sync2)
      vk::ImageMemoryBarrier2 opaqueToSample2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *opaqueSceneColorImages[currentFrame],
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
      };
      vk::DependencyInfo depOpaqueToSample{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &opaqueToSample2};
      commandBuffers[currentFrame].pipelineBarrier2(depOpaqueToSample);
      if (currentFrame < opaqueSceneColorImageLayouts.size()) {
        opaqueSceneColorImageLayouts[currentFrame] = vk::ImageLayout::eShaderReadOnlyOptimal;
      }

      // Make the swapchain image ready for color attachment output and clear it (Sync2)
      vk::ImageMemoryBarrier2 swapchainToColor2{
        .srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapChainImages[imageIndex],
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
      };
      vk::DependencyInfo depSwapchainToColor{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &swapchainToColor2};
      commandBuffers[currentFrame].pipelineBarrier2(depSwapchainToColor);

      // Begin rendering to swapchain for composite
      colorAttachments[0].imageView = *swapChainImageViews[imageIndex];
      colorAttachments[0].loadOp = vk::AttachmentLoadOp::eClear; // clear before composing base layer (full-screen composite overwrites all pixels)
      depthAttachment.loadOp = vk::AttachmentLoadOp::eDontCare; // no depth for composite
      renderingInfo.renderArea = vk::Rect2D({0, 0}, swapChainExtent);
      // IMPORTANT: Composite pass does not use a depth attachment. Avoid binding it to satisfy dynamic rendering VUIDs.
      auto savedDepthPtr = renderingInfo.pDepthAttachment; // save to restore later
      renderingInfo.pDepthAttachment = nullptr;
      commandBuffers[currentFrame].beginRendering(renderingInfo);

      // Bind composite pipeline
      if (!!*compositePipeline) {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *compositePipeline);
      }
      vk::Viewport vp(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
      commandBuffers[currentFrame].setViewport(0, vp);
      vk::Rect2D sc({0, 0}, swapChainExtent);
      commandBuffers[currentFrame].setScissor(0, sc);

      // Bind descriptor set 0 for the composite. During loading, force fallback to avoid sampling uninitialized off-screen color.
      vk::DescriptorSet setComposite = (transparentDescriptorSets.empty() || IsLoading())
                                         ? *transparentFallbackDescriptorSets[currentFrame]
                                         : *transparentDescriptorSets[currentFrame];
      commandBuffers[currentFrame].bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        *compositePipelineLayout,
        0,
        {setComposite},
        {});

      // Push exposure/gamma and sRGB flag
      struct CompositePush {
        float exposure;
        float gamma;
        int outputIsSRGB;
        float _pad;
      } pc{};
      pc.exposure = std::clamp(this->exposure, 0.2f, 4.0f);
      pc.gamma = this->gamma;
      pc.outputIsSRGB = (swapChainImageFormat == vk::Format::eR8G8B8A8Srgb || swapChainImageFormat == vk::Format::eB8G8R8A8Srgb) ? 1 : 0;
      commandBuffers[currentFrame].pushConstants<CompositePush>(*compositePipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, pc);

      // Draw fullscreen triangle
      commandBuffers[currentFrame].draw(3, 1, 0, 0);

      commandBuffers[currentFrame].endRendering();
      // Restore depth attachment pointer for subsequent passes
      renderingInfo.pDepthAttachment = savedDepthPtr;
    }
    // PASS 2: RENDER TRANSPARENT OBJECTS TO THE SWAPCHAIN
    {
      // Ensure depth attachment is bound again for the transparent pass
      renderingInfo.pDepthAttachment = &depthAttachment;
      colorAttachments[0].imageView = *swapChainImageViews[imageIndex];
      colorAttachments[0].loadOp = vk::AttachmentLoadOp::eLoad;
      depthAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
      renderingInfo.renderArea = vk::Rect2D({0, 0}, swapChainExtent);
      commandBuffers[currentFrame].beginRendering(renderingInfo);
      commandBuffers[currentFrame].setViewport(0, viewport);
      commandBuffers[currentFrame].setScissor(0, scissor);

      if (!transparentJobs.empty()) {
        currentLayout = &pbrTransparentPipelineLayout;
        vk::raii::Pipeline* activeTransparentPipeline = nullptr;

        for (const auto& job : transparentJobs) {
          vk::raii::Pipeline* desiredPipeline = job.entityRes->cachedIsGlass ? &glassGraphicsPipeline : &pbrBlendGraphicsPipeline;
          if (desiredPipeline != activeTransparentPipeline) {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, **desiredPipeline);
            activeTransparentPipeline = desiredPipeline;
          }

          std::array<vk::Buffer, 2> buffers = {*job.meshRes->vertexBuffer, *job.entityRes->instanceBuffer};
          std::array<vk::DeviceSize, 2> offsets = {0, 0};
          commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
          commandBuffers[currentFrame].bindIndexBuffer(*job.meshRes->indexBuffer, 0, vk::IndexType::eUint32);

          vk::DescriptorSet set1 = (transparentDescriptorSets.empty() || IsLoading())
                                     ? *transparentFallbackDescriptorSets[currentFrame]
                                     : *transparentDescriptorSets[currentFrame];
          commandBuffers[currentFrame].bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            **currentLayout,
            0,
            {*job.entityRes->pbrDescriptorSets[currentFrame], set1},
            {});

          MaterialProperties pushConstants = job.entityRes->cachedMaterialProps;
          if (job.entityRes->cachedIsLiquid) {
            pushConstants.transmissionFactor = 0.0f;
          }
          commandBuffers[currentFrame].pushConstants < MaterialProperties > (**currentLayout, vk::ShaderStageFlagBits::eFragment, 0,  {
            pushConstants
          }
          )
          ;
          uint32_t instanceCountT = std::max(1u, static_cast<uint32_t>(job.meshComp->GetInstanceCount()));
          commandBuffers[currentFrame].drawIndexed(job.meshRes->indexCount, instanceCountT, 0, 0, 0);
        }
      }
      // End transparent rendering pass before any layout transitions (even if no transparent draws)
      commandBuffers[currentFrame].endRendering();
    } {
      // Screenshot and final present transition are handled in rasterization path only
      // Ray query path handles these separately

      // Final layout transition for present (rasterization path only)
      {
        vk::ImageMemoryBarrier2 presentBarrier2{
          .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
          .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
          .dstStageMask = vk::PipelineStageFlagBits2::eNone,
          .dstAccessMask = {},
          .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
          .newLayout = vk::ImageLayout::ePresentSrcKHR,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .image = swapChainImages[imageIndex],
          .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
        };
        vk::DependencyInfo depToPresentFinal{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &presentBarrier2};
        commandBuffers[currentFrame].pipelineBarrier2(depToPresentFinal);
        if (imageIndex < swapChainImageLayouts.size())
          swapChainImageLayouts[imageIndex] = presentBarrier2.newLayout;
      }
    }
  } // skip rasterization when ray query has rendered

  // Render ImGui UI overlay AFTER rasterization/ray query (must always execute regardless of render mode)
  // ImGui expects Render() to be called every frame after NewFrame() - skipping it causes hangs
  if (imguiSystem && !imguiSystem->IsFrameRendered()) {
    // When ray query renders, swapchain is in PRESENT layout with valid content.
    // When rasterization renders, swapchain is also in PRESENT layout with valid content.
    // Transition to COLOR_ATTACHMENT with loadOp=eLoad to preserve existing pixels for ImGui overlay.
    vk::ImageMemoryBarrier2 presentToColor{
      .srcStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
      .srcAccessMask = vk::AccessFlagBits2::eNone,
      .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead,
      .oldLayout = (imageIndex < swapChainImageLayouts.size()) ? swapChainImageLayouts[imageIndex] : vk::ImageLayout::eUndefined,
      .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = swapChainImages[imageIndex],
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    };
    vk::DependencyInfo depInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &presentToColor};
    commandBuffers[currentFrame].pipelineBarrier2(depInfo);
    if (imageIndex < swapChainImageLayouts.size())
      swapChainImageLayouts[imageIndex] = presentToColor.newLayout;

    // Begin a dedicated render pass for ImGui (UI overlay)
    vk::RenderingAttachmentInfo imguiColorAttachment{
      .imageView = *swapChainImageViews[imageIndex],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eLoad, // Load existing content
      .storeOp = vk::AttachmentStoreOp::eStore
    };
    vk::RenderingInfo imguiRenderingInfo{
      .renderArea = vk::Rect2D({0, 0}, swapChainExtent),
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &imguiColorAttachment,
      .pDepthAttachment = nullptr
    };
    commandBuffers[currentFrame].beginRendering(imguiRenderingInfo);

    imguiSystem->Render(commandBuffers[currentFrame], currentFrame);

    commandBuffers[currentFrame].endRendering();

    // Transition swapchain back to PRESENT layout after ImGui renders
    vk::ImageMemoryBarrier2 colorToPresent{
      .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
      .dstAccessMask = vk::AccessFlagBits2::eNone,
      .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .newLayout = vk::ImageLayout::ePresentSrcKHR,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = swapChainImages[imageIndex],
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    };
    vk::DependencyInfo depInfoBack{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &colorToPresent};
    commandBuffers[currentFrame].pipelineBarrier2(depInfoBack);
    if (imageIndex < swapChainImageLayouts.size())
      swapChainImageLayouts[imageIndex] = colorToPresent.newLayout;
  }

  commandBuffers[currentFrame].end();
  isRecordingCmd.store(false, std::memory_order_relaxed);

  // Submit and present (Synchronization 2)
  uint64_t uploadsValueToWait = 0;
  {
    std::lock_guard<std::mutex> lock(queueMutex);
    uint64_t nextUp = nextUploadTimelineValue.load(std::memory_order_relaxed);
    uploadsValueToWait = (nextUp > 0) ? (nextUp - 1) : 0;
  }

  // Use acquireSemaphoreIndex for imageAvailable semaphore (same as we used in acquireNextImage)
  // Use imageIndex for renderFinished semaphore (matches the image being presented)

  std::vector<vk::SemaphoreSubmitInfo> waitInfos = {
    vk::SemaphoreSubmitInfo{
      .semaphore = *imageAvailableSemaphores[acquireSemaphoreIndex],
      .value = 0,
      .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      .deviceIndex = 0
    },
    vk::SemaphoreSubmitInfo{
      .semaphore = *uploadsTimeline,
      .value = uploadsValueToWait,
      .stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
      .deviceIndex = 0
    },
    // Wait-Before-Signal: Graphics waits for Physics simulation to complete
    vk::SemaphoreSubmitInfo{
      .semaphore = *frameTimeline,
      .value = currentTimelineValue + TimelineMilestones::ePhysicsFinished,
      .stageMask = vk::PipelineStageFlagBits2::eVertexShader,
      .deviceIndex = 0
    }
  };

  vk::CommandBufferSubmitInfo cmdInfo{.commandBuffer = *commandBuffers[currentFrame], .deviceMask = 0};

  std::array<vk::SemaphoreSubmitInfo, 2> signalInfos = {
    vk::SemaphoreSubmitInfo{
      .semaphore = *renderFinishedSemaphores[imageIndex],
      .value = 0,
      .stageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
      .deviceIndex = 0
    },
    vk::SemaphoreSubmitInfo{
      .semaphore = *frameTimeline,
      .value = currentTimelineValue + TimelineMilestones::eGpuWorkFinished,
      .stageMask = vk::PipelineStageFlagBits2::eBottomOfPipe,
      .deviceIndex = 0
    }
  };

  vk::SubmitInfo2 submit2{
    .waitSemaphoreInfoCount = static_cast<uint32_t>(waitInfos.size()),
    .pWaitSemaphoreInfos = waitInfos.data(),
    .commandBufferInfoCount = 1,
    .pCommandBufferInfos = &cmdInfo,
    .signalSemaphoreInfoCount = static_cast<uint32_t>(signalInfos.size()),
    .pSignalSemaphoreInfos = signalInfos.data()
  };


  // Update watchdog BEFORE queue submit because submit can block waiting for GPU
  // This proves frame CPU work is complete even if GPU queue is busy
  lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);

  // Submit work with monotonic timeline value (guarded by Submit2)
  Submit2(*graphicsQueue, submit2, nullptr);

  vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex], .swapchainCount = 1, .pSwapchains = &*swapChain, .pImageIndices = &imageIndex};
  vk::Result presentResult = vk::Result::eSuccess;
  try {
    std::lock_guard<std::mutex> lock(queueMutex);
    presentResult = presentQueue.presentKHR(presentInfo);
  } catch (const vk::OutOfDateKHRError&) {
    framebufferResized.store(true, std::memory_order_relaxed);
  }
  if (presentResult == vk::Result::eSuboptimalKHR || framebufferResized.load(std::memory_order_relaxed)) {
    framebufferResized.store(false, std::memory_order_relaxed);
    recreateSwapChain();
  } else if (presentResult != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to present swap chain image");
  }

  currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// Public toggle APIs for planar reflections (keyboard/UI)
void Renderer::SetPlanarReflectionsEnabled(bool enabled) {
  // Flip mode and mark resources dirty so RTs are created/destroyed at the next safe point
  enablePlanarReflections = enabled;
  reflectionResourcesDirty = true;
}

void Renderer::TogglePlanarReflections() {
  SetPlanarReflectionsEnabled(!enablePlanarReflections);
}
