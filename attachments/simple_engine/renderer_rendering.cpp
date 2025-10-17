#include "renderer.h"
#include "imgui_system.h"
#include "imgui/imgui.h"
#include "model_loader.h"
#include <fstream>
#include <stdexcept>
#include <array>
#include <cstring>
#include <iostream>
#include <ranges>
#include <cmath>
#include <ctime>

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
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        // Set sharing mode
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }

        // Create swap chain
        swapChain = vk::raii::SwapchainKHR(device, createInfo);

        // Get swap chain images
        swapChainImages = swapChain.getImages();

        // Store swap chain format and extent
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create swap chain: " << e.what() << std::endl;
        return false;
    }
}

// Create image views
bool Renderer::createImageViews() {
    try {
        opaqueSceneColorImage.clear();
        opaqueSceneColorImageView.clear();
        opaqueSceneColorSampler.clear();
        // Resize image views vector
        swapChainImageViews.clear();
        swapChainImageViews.reserve(swapChainImages.size());

        // Create image view for each swap chain image
        for (const auto& image : swapChainImages) {
            // Create image view info
            vk::ImageViewCreateInfo createInfo{
                .image = image,
                .viewType = vk::ImageViewType::e2D,
                .format = swapChainImageFormat,
                .components = {
                    .r = vk::ComponentSwizzle::eIdentity,
                    .g = vk::ComponentSwizzle::eIdentity,
                    .b = vk::ComponentSwizzle::eIdentity,
                    .a = vk::ComponentSwizzle::eIdentity
                },
                .subresourceRange = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };

            // Create image view
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
                .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})
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
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        // Create command pool info
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
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

        // Create semaphores per swapchain image to avoid reuse issues
        size_t swapchainImageCount = swapChainImages.size();
        imageAvailableSemaphores.reserve(swapchainImageCount);
        renderFinishedSemaphores.reserve(swapchainImageCount);

        // Keep fences per frame in flight for CPU-GPU synchronization
        inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

        // Create semaphore and fence info
        vk::SemaphoreCreateInfo semaphoreInfo{};
        vk::FenceCreateInfo fenceInfo{
            .flags = vk::FenceCreateFlagBits::eSignaled
        };

        // Create semaphores for each swapchain image
        for (size_t i = 0; i < swapchainImageCount; i++) {
            imageAvailableSemaphores.emplace_back(device, semaphoreInfo);
            renderFinishedSemaphores.emplace_back(device, semaphoreInfo);
        }

        // Create fences for each frame in flight
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
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
    depthImageView = nullptr;
    depthImage = nullptr;
    depthImageAllocation = nullptr;

    // Clean up swap chain image views
    swapChainImageViews.clear();

    // Note: Keep descriptor pool alive here to ensure descriptor sets remain valid during swapchain recreation.
    // descriptorPool is preserved; it will be managed during full renderer teardown.

    // Clean up pipelines
    graphicsPipeline = nullptr;
    pbrGraphicsPipeline = nullptr;
    lightingPipeline = nullptr;

    // Clean up pipeline layouts
    pipelineLayout = nullptr;
    pbrPipelineLayout = nullptr;
    lightingPipelineLayout = nullptr;

    // Clean up sync objects (they need to be recreated with new swap chain image count)
    imageAvailableSemaphores.clear();
    renderFinishedSemaphores.clear();

    // Clean up swap chain
    swapChain = nullptr;
}

// Recreate swap chain
void Renderer::recreateSwapChain() {
    // Wait for all frames in flight to complete before recreating the swap chain
    std::vector<vk::Fence> allFences;
    allFences.reserve(inFlightFences.size());
    for (const auto& fence : inFlightFences) {
        allFences.push_back(*fence);
    }
    if (!allFences.empty()) {
        if (device.waitForFences(allFences, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {}
    }

    // Wait for the device to be idle before recreating the swap chain
    device.waitIdle();

    // Clean up old swap chain resources
    cleanupSwapChain();

    // Recreate swap chain and related resources
    createSwapChain();
    createImageViews();
    setupDynamicRendering();
    createDepthResources();

    // Recreate sync objects with correct sizing for new swap chain
    createSyncObjects();

    // Recreate descriptor pool and pipelines
    createDescriptorPool();

    // Recreate off-screen opaque scene color and descriptor sets needed by transparent pass
    createOpaqueSceneColorResources();
    createTransparentDescriptorSets();
    createTransparentFallbackDescriptorSets();

    // Wait for all command buffers to complete before clearing resources
    for (const auto& fence : inFlightFences) {
        if (device.waitForFences(*fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {}
    }

    // Clear all entity descriptor sets since they're now invalid (allocated from the old pool)
    for (auto& resources : entityResources | std::views::values) {
        resources.basicDescriptorSets.clear();
        resources.pbrDescriptorSets.clear();
    }

    createGraphicsPipeline();
    createPBRPipeline();
    createLightingPipeline();

    // Re-create command buffers to ensure fresh recording against new swapchain state
    commandBuffers.clear();
    createCommandBuffers();
    currentFrame = 0;

    // Recreate descriptor sets for all entities after swapchain/pipeline rebuild
    for (auto& [entity, resources] : entityResources) {
        if (!entity) continue;
        auto meshComponent = entity->GetComponent<MeshComponent>();
        if (!meshComponent) continue;

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
}

// Update uniform buffer
void Renderer::updateUniformBuffer(uint32_t currentImage, Entity* entity, CameraComponent* camera) {
    // Get entity resources
    auto entityIt = entityResources.find(entity);
    if (entityIt == entityResources.end()) {
        return;
    }

    // Get transform component
    auto transformComponent = entity->GetComponent<TransformComponent>();
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
    updateUniformBufferInternal(currentImage, entity, camera, ubo);
}

// Overloaded version that accepts a custom transform matrix
void Renderer::updateUniformBuffer(uint32_t currentImage, Entity* entity, CameraComponent* camera, const glm::mat4& customTransform) {
    // Create the uniform buffer object with custom transform
    UniformBufferObject ubo{};
    ubo.model = customTransform;
    ubo.view = camera->GetViewMatrix();
    ubo.proj = camera->GetProjectionMatrix();
    ubo.proj[1][1] *= -1; // Flip Y for Vulkan

    // Continue with the rest of the uniform buffer setup
    updateUniformBufferInternal(currentImage, entity, camera, ubo);
}

// Internal helper function to complete uniform buffer setup
void Renderer::updateUniformBufferInternal(uint32_t currentImage, Entity* entity, CameraComponent* camera, UniformBufferObject& ubo) {
    // Get entity resources
    auto entityIt = entityResources.find(entity);
    if (entityIt == entityResources.end()) {
        return;
    }

    // Use static lights loaded during model initialization
    const std::vector<ExtractedLight>& extractedLights = staticLights;

    if (!extractedLights.empty()) {
        // Limit the number of active lights for performance
        size_t numLights = std::min(extractedLights.size(), size_t(MAX_ACTIVE_LIGHTS));

        // Create a subset of lights to upload this frame
        std::vector<ExtractedLight> lightsSubset;
        lightsSubset.reserve(numLights);
        for (size_t i = 0; i < numLights; ++i) {
            lightsSubset.push_back(extractedLights[i]);
        }

        // Apply UI-driven sun control to the first directional light with a Paris-based solar path
        {
            // Find first directional light to treat as the Sun
            size_t sunIdx = SIZE_MAX;
            for (size_t i = 0; i < lightsSubset.size(); ++i) {
                if (lightsSubset[i].type == ExtractedLight::Type::Directional) { sunIdx = i; break; }
            }
            if (sunIdx != SIZE_MAX) {
                auto &sun = lightsSubset[sunIdx];
                float s = std::clamp(sunPosition, 0.0f, 1.0f);

                // Paris latitude (degrees)
                const float latDeg = 48.8566f;
                const float lat = latDeg * 0.01745329251994329577f; // radians

                // Get current day-of-year (0..365) for declination
                int yday = 172; // Default to around June solstice if time is unavailable
                std::time_t now = std::time(nullptr);
                if (now != (std::time_t)(-1)) {
                    std::tm localTm{};
                    #ifdef _WIN32
                        localtime_s(&localTm, &now);
                    #else
                        std::tm* ptm = std::localtime(&now);
                        if (ptm) localTm = *ptm;
                    #endif
                    if (localTm.tm_yday >= 0) yday = localTm.tm_yday; // 0-based
                }

                // Solar declination (degrees) using Cooper's approximation
                // δ = 23.45° * sin(360° * (284 + n) / 365)
                float declDeg = 23.45f * std::sin((6.283185307179586f) * (284.0f + (float)(yday + 1)) / 365.0f);
                float decl = declDeg * 0.01745329251994329577f; // radians

                // Map slider to local solar time (0..24h), hour angle H in radians (0 at noon)
                float hours = s * 24.0f;
                float Hdeg = (hours - 12.0f) * 15.0f; // degrees per hour
                float H = Hdeg * 0.01745329251994329577f; // radians

                // Solar altitude (elevation) from spherical astronomy
                // sin(alt) = sin φ sin δ + cos φ cos δ cos H
                float sinAlt = std::sin(lat) * std::sin(decl) + std::cos(lat) * std::cos(decl) * std::cos(H);
                sinAlt = std::clamp(sinAlt, -1.0f, 1.0f);
                float alt = std::asin(sinAlt); // radians

                // Build horizontal azimuth basis from original sun direction (treat original as local solar noon azimuth)
                glm::vec3 origDir = sun.direction;
                glm::vec2 baseHoriz2 = glm::normalize(glm::vec2(origDir.x, origDir.z));
                if (!std::isfinite(baseHoriz2.x)) { baseHoriz2 = glm::vec2(0.0f, -1.0f); }

                // Rotate base horizontal around Y by hour angle H (east-west movement). Positive H -> afternoon (west)
                float cosH = std::cos(H);
                float sinH = std::sin(H);
                glm::vec2 horizRot2 = glm::normalize(glm::vec2(
                    baseHoriz2.x * cosH - baseHoriz2.y * sinH,
                    baseHoriz2.x * sinH + baseHoriz2.y * cosH));

                // Compose final direction from altitude and rotated horizontal
                float cosAlt = std::cos(alt);
                float sinAltClamped = std::sin(alt);
                glm::vec3 newDir = glm::normalize(glm::vec3(horizRot2.x * cosAlt, -sinAltClamped, horizRot2.y * cosAlt));
                sun.direction = newDir;

                // Intensity scales with daylight (altitude); zero when below horizon
                float dayFactor = std::max(0.0f, sinAltClamped); // 0..1 roughly

                // Warm tint increases near horizon when sun is above horizon
                float horizonFactor = 0.0f;
                if (sinAltClamped > 0.0f) {
                    // More warmth for low altitude, fade to zero near high noon
                    float normAlt = std::clamp(alt / (1.57079632679f), 0.0f, 1.0f); // 0 at horizon, 1 at zenith
                    horizonFactor = 1.0f - normAlt; // 1 near horizon, 0 at zenith
                }
                glm::vec3 warm(1.0f, 0.75f, 0.55f);
                float tintAmount = 0.7f * horizonFactor;
                sun.color = glm::mix(sun.color, warm, tintAmount);

                // Apply intensity scaling (preserve original magnitude shape)
                sun.intensity *= dayFactor;
            }
        }

        // Update the light storage buffer with the subset of light data
        updateLightStorageBuffer(currentImage, lightsSubset);

        ubo.lightCount = static_cast<int>(numLights);
        // Shadows removed: no shadow maps
    } else {
        ubo.lightCount = 0;
    }

    // Shadows removed: no shadow bias

    // Set camera position for PBR calculations
    ubo.camPos = glm::vec4(camera->GetPosition(), 1.0f);

    // Set PBR parameters (use member variables for UI control)
    // Clamp exposure to a sane range to avoid washout
    ubo.exposure = std::clamp(this->exposure, 0.2f, 2.0f);
    ubo.gamma = this->gamma;
    ubo.prefilteredCubeMipLevels = 0.0f;
    ubo.scaleIBLAmbient = 0.25f;
    ubo.screenDimensions = glm::vec2(swapChainExtent.width, swapChainExtent.height);

    // Signal to the shader whether swapchain is sRGB (1) or not (0) using padding0
    int outputIsSRGB = (swapChainImageFormat == vk::Format::eR8G8B8A8Srgb ||
                        swapChainImageFormat == vk::Format::eB8G8R8A8Srgb) ? 1 : 0;
    ubo.padding0 = outputIsSRGB;

    // Copy to uniform buffer
    std::memcpy(entityIt->second.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

// Render the scene
void Renderer::Render(const std::vector<std::unique_ptr<Entity>>& entities, CameraComponent* camera, ImGuiSystem* imguiSystem) {
    if (memoryPool) memoryPool->setRenderingActive(true);
    struct RenderingStateGuard { MemoryPool* pool; explicit RenderingStateGuard(MemoryPool* p) : pool(p) {} ~RenderingStateGuard() { if (pool) pool->setRenderingActive(false); } } guard(memoryPool.get());

    if (device.waitForFences(*inFlightFences[currentFrame], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {}

    uint32_t imageIndex;
    auto result = swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphores[currentFrame]);
    imageIndex = result.second;

    if (result.first == vk::Result::eErrorOutOfDateKHR || result.first == vk::Result::eSuboptimalKHR || framebufferResized.load(std::memory_order_relaxed)) {
        framebufferResized.store(false, std::memory_order_relaxed);
        if (imguiSystem) ImGui::EndFrame();
        recreateSwapChain();
        return;
    }
    if (result.first != vk::Result::eSuccess) { throw std::runtime_error("Failed to acquire swap chain image"); }

    device.resetFences(*inFlightFences[currentFrame]);
    if (framebufferResized.load(std::memory_order_relaxed)) { recreateSwapChain(); return; }

    commandBuffers[currentFrame].reset();
    commandBuffers[currentFrame].begin(vk::CommandBufferBeginInfo());
    if (framebufferResized.load(std::memory_order_relaxed)) { commandBuffers[currentFrame].end(); recreateSwapChain(); return; }

    vk::raii::Pipeline* currentPipeline = nullptr;
    vk::raii::PipelineLayout* currentLayout = nullptr;
    std::vector<Entity*> blendedQueue;
    std::unordered_set<Entity*> blendedSet;

    bool blockScene = false;
    if (imguiSystem) {
        blockScene = IsLoading() || (GetTextureTasksScheduled() > 0 && GetTextureTasksCompleted() < GetTextureTasksScheduled());
    }

    if (!blockScene) {
        for (const auto& uptr : entities) {
            Entity* entity = uptr.get();
            if (!entity || !entity->IsActive()) continue;
            auto meshComponent = entity->GetComponent<MeshComponent>();
            if (!meshComponent) continue;
            bool useBlended = false;
            if (modelLoader && entity->GetName().find("_Material_") != std::string::npos) {
                std::string entityName = entity->GetName();
                size_t tagPos = entityName.find("_Material_");
                if (tagPos != std::string::npos) {
                    size_t afterTag = tagPos + std::string("_Material_").size();
                    if (afterTag < entityName.length()) {
                        // Entity name format: "modelName_Material_<index>_<materialName>"
                        // Find the next underscore after the material index to get the actual material name
                        std::string remainder = entityName.substr(afterTag);
                        size_t nextUnderscore = remainder.find('_');
                        if (nextUnderscore != std::string::npos && nextUnderscore + 1 < remainder.length()) {
                            std::string materialName = remainder.substr(nextUnderscore + 1);
                            Material* material = modelLoader->GetMaterial(materialName);
                            if (material && (material->alphaMode == "BLEND" || material->transmissionFactor > 0.001f)) {
                                useBlended = true;
                            }
                        }
                    }
                }
            }
            if (useBlended) {
                blendedQueue.push_back(entity);
                blendedSet.insert(entity);
            }
        }
    }

    // Sort transparent entities back-to-front for correct blending of nested glass/liquids
    if (!blendedQueue.empty()) {
        // Sort by view-space depth using the camera's view matrix for robust back-to-front order
        glm::mat4 V = camera ? camera->GetViewMatrix() : glm::mat4(1.0f);
        std::sort(blendedQueue.begin(), blendedQueue.end(), [V](Entity* a, Entity* b) {
            auto* ta = (a ? a->GetComponent<TransformComponent>() : nullptr);
            auto* tb = (b ? b->GetComponent<TransformComponent>() : nullptr);
            glm::vec3 pa = ta ? ta->GetPosition() : glm::vec3(0.0f);
            glm::vec3 pb = tb ? tb->GetPosition() : glm::vec3(0.0f);
            float za = (V * glm::vec4(pa, 1.0f)).z;
            float zb = (V * glm::vec4(pb, 1.0f)).z;
            if (za != zb) {
                // In view space (looking down -Z), farther objects have more negative z. Sort ascending: farther first.
                return za < zb;
            }
            // Fallback to stable ordering
            return a < b;
        });
    }

    // PASS 1: RENDER OPAQUE OBJECTS TO OFF-SCREEN TEXTURE
    {
        vk::ImageMemoryBarrier barrier{ .srcAccessMask = vk::AccessFlagBits::eNone, .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eColorAttachmentOptimal, .image = *opaqueSceneColorImage, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} };
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {}, barrier);
        vk::RenderingAttachmentInfo colorAttachment{ .imageView = *opaqueSceneColorImageView, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal, .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore, .clearValue = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}) };
        depthAttachment.imageView = *depthImageView;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        vk::RenderingInfo passInfo{ .renderArea = vk::Rect2D({0, 0}, swapChainExtent), .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorAttachment, .pDepthAttachment = &depthAttachment };
        commandBuffers[currentFrame].beginRendering(passInfo);
        vk::Viewport viewport(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
        commandBuffers[currentFrame].setViewport(0, viewport);
        vk::Rect2D scissor({0, 0}, swapChainExtent);
        commandBuffers[currentFrame].setScissor(0, scissor);
        if (!blockScene) {
            for (const auto& uptr : entities) {
                Entity* entity = uptr.get();
                if (!entity || !entity->IsActive() || blendedSet.count(entity)) continue;
                auto meshComponent = entity->GetComponent<MeshComponent>();
                if (!meshComponent) continue;
                bool useBasic = imguiSystem && !imguiSystem->IsPBREnabled();
                vk::raii::Pipeline* selectedPipeline = useBasic ? &graphicsPipeline : &pbrGraphicsPipeline;
                vk::raii::PipelineLayout* selectedLayout = useBasic ? &pipelineLayout : &pbrPipelineLayout;
                if (currentPipeline != selectedPipeline) {
                    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, **selectedPipeline);
                    currentPipeline = selectedPipeline;
                    currentLayout = selectedLayout;
                }
                auto meshIt = meshResources.find(meshComponent);
                auto entityIt = entityResources.find(entity);
                if (meshIt == meshResources.end() || entityIt == entityResources.end()) continue;
                std::array<vk::Buffer, 2> buffers = {*meshIt->second.vertexBuffer, *entityIt->second.instanceBuffer};
                std::array<vk::DeviceSize, 2> offsets = {0, 0};
                commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
                commandBuffers[currentFrame].bindIndexBuffer(*meshIt->second.indexBuffer, 0, vk::IndexType::eUint32);
                updateUniformBuffer(currentFrame, entity, camera);
                auto& descSets = useBasic ? entityIt->second.basicDescriptorSets : entityIt->second.pbrDescriptorSets;
                if (descSets.empty() || currentFrame >= descSets.size()) continue;
                if (useBasic) {
                    // Basic pipeline expects only set 0
                    commandBuffers[currentFrame].bindDescriptorSets(
                        vk::PipelineBindPoint::eGraphics,
                        **currentLayout,
                        0,
                        { *descSets[currentFrame] },
                        {}
                    );
                } else {
                    // Opaque PBR pipeline: bind set 0 (PBR) and a valid set 1 (fallback scene color)
                    vk::DescriptorSet set1Opaque = *transparentFallbackDescriptorSets[currentFrame];
                    commandBuffers[currentFrame].bindDescriptorSets(
                        vk::PipelineBindPoint::eGraphics,
                        **currentLayout,
                        0,
                        { *descSets[currentFrame], set1Opaque },
                        {}
                    );
                }
                if (!useBasic) {
                    MaterialProperties pushConstants{};
                    // Sensible defaults for entities without explicit material
                    pushConstants.baseColorFactor = glm::vec4(1.0f);
                    pushConstants.metallicFactor = 0.0f;
                    pushConstants.roughnessFactor = 1.0f;
                    pushConstants.baseColorTextureSet = 0; // sample bound baseColor (falls back to shared default if none)
                    pushConstants.physicalDescriptorTextureSet = 0; // default to sampling metallic-roughness on binding 2
                    pushConstants.normalTextureSet = -1;
                    pushConstants.occlusionTextureSet = -1;
                    pushConstants.emissiveTextureSet = -1;
                    pushConstants.alphaMask = 0.0f;
                    pushConstants.alphaMaskCutoff = 0.5f;
                    pushConstants.emissiveFactor = glm::vec3(0.0f);
                    pushConstants.emissiveStrength = 1.0f;
                    pushConstants.hasEmissiveStrengthExtension = false; // Default entities don't have emissive strength extension
                    pushConstants.transmissionFactor = 0.0f;
                    pushConstants.useSpecGlossWorkflow = 0;
                    pushConstants.glossinessFactor = 0.0f;
                    pushConstants.specularFactor = glm::vec3(1.0f);
                    // pushConstants.ior already 1.5f default
                    if (modelLoader && entity->GetName().find("_Material_") != std::string::npos) {
                        std::string entityName = entity->GetName();
                        size_t tagPos = entityName.find("_Material_");
                        if (tagPos != std::string::npos) {
                            size_t afterTag = tagPos + std::string("_Material_").size();
                            if (afterTag < entityName.length()) {
                                // Entity name format: "modelName_Material_<index>_<materialName>"
                                // Find the next underscore after the material index to get the actual material name
                                std::string remainder = entityName.substr(afterTag);
                                size_t nextUnderscore = remainder.find('_');
                                if (nextUnderscore != std::string::npos && nextUnderscore + 1 < remainder.length()) {
                                    std::string materialName = remainder.substr(nextUnderscore + 1);
                                    Material* material = modelLoader->GetMaterial(materialName);
                                    if (material) {
                                        // Base factors
                                        pushConstants.baseColorFactor = glm::vec4(material->albedo, material->alpha);
                                        pushConstants.metallicFactor = material->metallic;
                                        pushConstants.roughnessFactor = material->roughness;

                                        // Texture set flags (-1 = no texture)
                                        pushConstants.baseColorTextureSet = material->albedoTexturePath.empty() ? -1 : 0;
                                        // physical descriptor: MR or SpecGloss
                                        if (material->useSpecularGlossiness) {
                                            pushConstants.useSpecGlossWorkflow = 1;
                                            pushConstants.physicalDescriptorTextureSet = material->specGlossTexturePath.empty() ? -1 : 0;
                                            pushConstants.glossinessFactor = material->glossinessFactor;
                                            pushConstants.specularFactor = material->specularFactor;
                                        } else {
                                            pushConstants.useSpecGlossWorkflow = 0;
                                            pushConstants.physicalDescriptorTextureSet = material->metallicRoughnessTexturePath.empty() ? -1 : 0;
                                        }
                                        pushConstants.normalTextureSet = material->normalTexturePath.empty() ? -1 : 0;
                                        pushConstants.occlusionTextureSet = material->occlusionTexturePath.empty() ? -1 : 0;
                                        pushConstants.emissiveTextureSet = material->emissiveTexturePath.empty() ? -1 : 0;

                                        // Emissive and transmission/IOR
                                        pushConstants.emissiveFactor = material->emissive;
                                        pushConstants.emissiveStrength = material->emissiveStrength;
                                        pushConstants.hasEmissiveStrengthExtension = false; // Material has emissive strength data
                                        pushConstants.transmissionFactor = material->transmissionFactor;
                                        pushConstants.ior = material->ior;

                                        // Alpha mask handling
                                        pushConstants.alphaMask = (material->alphaMode == "MASK") ? 1.0f : 0.0f;
                                        pushConstants.alphaMaskCutoff = material->alphaCutoff;
                                    }
                                }
                            }
                        }
                    }
                    // If no explicit MASK from a material, infer it from the baseColor texture's alpha usage
                    if (pushConstants.alphaMask < 0.5f) {
                        std::string baseColorPath;
                        if (meshComponent) {
                            if (!meshComponent->GetBaseColorTexturePath().empty()) {
                                baseColorPath = meshComponent->GetBaseColorTexturePath();
                            } else if (!meshComponent->GetTexturePath().empty()) {
                                baseColorPath = meshComponent->GetTexturePath();
                            } else {
                                baseColorPath = SHARED_DEFAULT_ALBEDO_ID;
                            }
                        } else {
                            baseColorPath = SHARED_DEFAULT_ALBEDO_ID;
                        }
                        // Avoid inferring MASK from the shared default albedo (semi-transparent placeholder)
                        if (baseColorPath != SHARED_DEFAULT_ALBEDO_ID) {
                            const std::string resolvedBase = ResolveTextureId(baseColorPath);
                            std::shared_lock<std::shared_mutex> texLock(textureResourcesMutex);
                            auto itTex = textureResources.find(resolvedBase);
                            if (itTex != textureResources.end() && itTex->second.alphaMaskedHint) {
                                pushConstants.alphaMask = 1.0f;
                                pushConstants.alphaMaskCutoff = 0.5f;
                            }
                        }
                    }
                    commandBuffers[currentFrame].pushConstants<MaterialProperties>(**currentLayout, vk::ShaderStageFlagBits::eFragment, 0, { pushConstants });
                }
                uint32_t instanceCount = std::max(1u, static_cast<uint32_t>(meshComponent->GetInstanceCount()));
                commandBuffers[currentFrame].drawIndexed(meshIt->second.indexCount, instanceCount, 0, 0, 0);
            }
        }
        commandBuffers[currentFrame].endRendering();
    }
    // BARRIER AND COPY
    {
        vk::ImageMemoryBarrier opaqueSrcBarrier{ .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .dstAccessMask = vk::AccessFlagBits::eTransferRead, .oldLayout = vk::ImageLayout::eColorAttachmentOptimal, .newLayout = vk::ImageLayout::eTransferSrcOptimal, .image = *opaqueSceneColorImage, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} };
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, opaqueSrcBarrier);
        vk::ImageMemoryBarrier swapchainDstBarrier{ .srcAccessMask = vk::AccessFlagBits::eNone, .dstAccessMask = vk::AccessFlagBits::eTransferWrite, .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eTransferDstOptimal, .image = swapChainImages[imageIndex], .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} };
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, swapchainDstBarrier);
        vk::ImageCopy copyRegion{ .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, .extent = {swapChainExtent.width, swapChainExtent.height, 1} };
        commandBuffers[currentFrame].copyImage(*opaqueSceneColorImage, vk::ImageLayout::eTransferSrcOptimal, swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal, copyRegion);
        vk::ImageMemoryBarrier opaqueShaderBarrier{ .srcAccessMask = vk::AccessFlagBits::eTransferRead, .dstAccessMask = vk::AccessFlagBits::eShaderRead, .oldLayout = vk::ImageLayout::eTransferSrcOptimal, .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal, .image = *opaqueSceneColorImage, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} };
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, opaqueShaderBarrier);
        vk::ImageMemoryBarrier swapchainTargetBarrier{ .srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .oldLayout = vk::ImageLayout::eTransferDstOptimal, .newLayout = vk::ImageLayout::eColorAttachmentOptimal, .image = swapChainImages[imageIndex], .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} };
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {}, swapchainTargetBarrier);
    }
    // PASS 2: RENDER TRANSPARENT OBJECTS TO THE SWAPCHAIN
    {
        colorAttachments[0].imageView = *swapChainImageViews[imageIndex];
        colorAttachments[0].loadOp = vk::AttachmentLoadOp::eLoad;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
        renderingInfo.renderArea = vk::Rect2D({0, 0}, swapChainExtent);
        commandBuffers[currentFrame].beginRendering(renderingInfo);
        vk::Viewport viewport(0.0f, 0.0f, (float)swapChainExtent.width, (float)swapChainExtent.height, 0.0f, 1.0f);
        commandBuffers[currentFrame].setViewport(0, viewport);
        vk::Rect2D scissor({0, 0}, swapChainExtent);
        commandBuffers[currentFrame].setScissor(0, scissor);

        if (!blendedQueue.empty()) {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrBlendGraphicsPipeline);
            currentLayout = &pbrTransparentPipelineLayout;

            for (Entity* entity : blendedQueue) {
                auto meshComponent = entity->GetComponent<MeshComponent>();
                auto entityIt = entityResources.find(entity);
                auto meshIt = meshResources.find(meshComponent);
                if (!meshComponent || entityIt == entityResources.end() || meshIt == meshResources.end()) continue;

                std::array<vk::Buffer, 2> buffers = {*meshIt->second.vertexBuffer, *entityIt->second.instanceBuffer};
                std::array<vk::DeviceSize, 2> offsets = {0, 0};
                commandBuffers[currentFrame].bindVertexBuffers(0, buffers, offsets);
                commandBuffers[currentFrame].bindIndexBuffer(*meshIt->second.indexBuffer, 0, vk::IndexType::eUint32);
                updateUniformBuffer(currentFrame, entity, camera);

                auto& pbrDescSets = entityIt->second.pbrDescriptorSets;
                if (pbrDescSets.empty() || currentFrame >= pbrDescSets.size()) continue;

                // Bind PBR (set 0) and scene color (set 1). If primary set 1 is unavailable, use fallback.
                vk::DescriptorSet set1 = transparentDescriptorSets.empty()
                    ? *transparentFallbackDescriptorSets[currentFrame]
                    : *transparentDescriptorSets[currentFrame];
                commandBuffers[currentFrame].bindDescriptorSets(
                    vk::PipelineBindPoint::eGraphics,
                    **currentLayout,
                    0,
                    { *pbrDescSets[currentFrame], set1 },
                    {}
                );

                MaterialProperties pushConstants{};
                // Sensible defaults for entities without explicit material
                pushConstants.baseColorFactor = glm::vec4(1.0f);
                pushConstants.metallicFactor = 0.0f;
                pushConstants.roughnessFactor = 1.0f;
                pushConstants.baseColorTextureSet = 0; // sample bound baseColor (falls back to shared default if none)
                pushConstants.physicalDescriptorTextureSet = 0; // default to sampling metallic-roughness on binding 2
                pushConstants.normalTextureSet = -1;
                pushConstants.occlusionTextureSet = -1;
                pushConstants.emissiveTextureSet = -1;
                pushConstants.alphaMask = 0.0f;
                pushConstants.alphaMaskCutoff = 0.5f;
                pushConstants.emissiveFactor = glm::vec3(0.0f);
                pushConstants.emissiveStrength = 1.0f;
                pushConstants.hasEmissiveStrengthExtension = false;
                pushConstants.transmissionFactor = 0.0f;
                pushConstants.useSpecGlossWorkflow = 0;
                pushConstants.glossinessFactor = 0.0f;
                pushConstants.specularFactor = glm::vec3(1.0f);
                // pushConstants.ior already 1.5f default
                if (modelLoader && entity->GetName().find("_Material_") != std::string::npos) {
                    std::string entityName = entity->GetName();
                    size_t tagPos = entityName.find("_Material_");
                    if (tagPos != std::string::npos) {
                        size_t afterTag = tagPos + std::string("_Material_").size();
                        if (afterTag < entityName.length()) {
                            // Entity name format: "modelName_Material_<index>_<materialName>"
                            // Find the next underscore after the material index to get the actual material name
                            std::string remainder = entityName.substr(afterTag);
                            size_t nextUnderscore = remainder.find('_');
                            if (nextUnderscore != std::string::npos && nextUnderscore + 1 < remainder.length()) {
                                std::string materialName = remainder.substr(nextUnderscore + 1);
                                Material* material = modelLoader->GetMaterial(materialName);
                                if (material) {
                                    // Base factors
                                    pushConstants.baseColorFactor = glm::vec4(material->albedo, material->alpha);
                                    pushConstants.metallicFactor = material->metallic;
                                    pushConstants.roughnessFactor = material->roughness;

                                    // Texture set flags (-1 = no texture)
                                    pushConstants.baseColorTextureSet = material->albedoTexturePath.empty() ? -1 : 0;
                                    if (material->useSpecularGlossiness) {
                                        pushConstants.useSpecGlossWorkflow = 1;
                                        pushConstants.physicalDescriptorTextureSet = material->specGlossTexturePath.empty() ? -1 : 0;
                                        pushConstants.glossinessFactor = material->glossinessFactor;
                                        pushConstants.specularFactor = material->specularFactor;
                                    } else {
                                        pushConstants.useSpecGlossWorkflow = 0;
                                        pushConstants.physicalDescriptorTextureSet = material->metallicRoughnessTexturePath.empty() ? -1 : 0;
                                    }
                                    pushConstants.normalTextureSet = material->normalTexturePath.empty() ? -1 : 0;
                                    pushConstants.occlusionTextureSet = material->occlusionTexturePath.empty() ? -1 : 0;
                                    pushConstants.emissiveTextureSet = material->emissiveTexturePath.empty() ? -1 : 0;

                                    // Emissive and transmission/IOR
                                    pushConstants.emissiveFactor = material->emissive;
                                    pushConstants.emissiveStrength = material->emissiveStrength;
                                    pushConstants.hasEmissiveStrengthExtension = false; // Material has emissive strength data
                                    pushConstants.transmissionFactor = material->transmissionFactor;
                                    pushConstants.ior = material->ior;

                                    // Alpha mask handling
                                    pushConstants.alphaMask = (material->alphaMode == "MASK") ? 1.0f : 0.0f;
                                    pushConstants.alphaMaskCutoff = material->alphaCutoff;
                                }
                            }
                        }
                    }
                }
                commandBuffers[currentFrame].pushConstants<MaterialProperties>(**currentLayout, vk::ShaderStageFlagBits::eFragment, 0, { pushConstants });
                uint32_t instanceCountT = std::max(1u, static_cast<uint32_t>(meshComponent->GetInstanceCount()));
                commandBuffers[currentFrame].drawIndexed(meshIt->second.indexCount, instanceCountT, 0, 0, 0);
            }
        }

        if (imguiSystem) {
            imguiSystem->Render(commandBuffers[currentFrame], currentFrame);
        }
        commandBuffers[currentFrame].endRendering();
    }

    // ... (final barrier, submit, and present logic is the same) ...
    vk::ImageMemoryBarrier presentBarrier{ .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .dstAccessMask = vk::AccessFlagBits::eNone, .oldLayout = vk::ImageLayout::eColorAttachmentOptimal, .newLayout = vk::ImageLayout::ePresentSrcKHR, .image = swapChainImages[imageIndex], .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} };
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, presentBarrier);
    commandBuffers[currentFrame].end();
    std::array<vk::Semaphore, 2> waitSems = { *imageAvailableSemaphores[currentFrame], *uploadsTimeline };
    std::array<vk::PipelineStageFlags, 2> waitStages = { vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader };
    uint64_t uploadsValueToWait = uploadTimelineLastSubmitted.load(std::memory_order_relaxed);
    std::array<uint64_t, 2> waitValues = { 0ull, uploadsValueToWait };
    vk::TimelineSemaphoreSubmitInfo timelineWaitInfo{ .waitSemaphoreValueCount = (uint32_t)waitValues.size(), .pWaitSemaphoreValues = waitValues.data() };
    vk::SubmitInfo submitInfo{ .pNext = &timelineWaitInfo, .waitSemaphoreCount = (uint32_t)waitSems.size(), .pWaitSemaphores = waitSems.data(), .pWaitDstStageMask = waitStages.data(), .commandBufferCount = 1, .pCommandBuffers = &*commandBuffers[currentFrame], .signalSemaphoreCount = 1, .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex] };
    if (framebufferResized.load(std::memory_order_relaxed)) {
        vk::SubmitInfo emptySubmit{};
        { std::lock_guard<std::mutex> lock(queueMutex); graphicsQueue.submit(emptySubmit, *inFlightFences[currentFrame]); }
        recreateSwapChain();
        return;
    }
    { std::lock_guard<std::mutex> lock(queueMutex); graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]); }
    vk::PresentInfoKHR presentInfo{ .waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex], .swapchainCount = 1, .pSwapchains = &*swapChain, .pImageIndices = &imageIndex };
    try {
        std::lock_guard<std::mutex> lock(queueMutex);
        result.first = presentQueue.presentKHR(presentInfo);
    } catch (const vk::OutOfDateKHRError&) {
        framebufferResized.store(true, std::memory_order_relaxed);
    }
    if (result.first == vk::Result::eErrorOutOfDateKHR || result.first == vk::Result::eSuboptimalKHR || framebufferResized.load(std::memory_order_relaxed)) {
        framebufferResized.store(false, std::memory_order_relaxed);
        recreateSwapChain();
    } else if (result.first != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to present swap chain image");
    }
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}
