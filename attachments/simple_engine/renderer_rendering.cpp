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
#include <glm/gtx/norm.hpp>

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
    for (const auto& entity : entityResources | std::views::keys) {
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

    // Use static lights loaded during model initialization. For the
    // current tutorial we render a fixed "night" scene lit only by
    // emissive-derived lights from the GLTF; any punctual
    // directional/point/spot lights are ignored.
    const std::vector<ExtractedLight>& extractedLights = staticLights;

    if (!extractedLights.empty()) {
        std::vector<ExtractedLight> lightsSubset;
        lightsSubset.reserve(std::min(extractedLights.size(), static_cast<size_t>(MAX_ACTIVE_LIGHTS)));

        for (const auto& L : extractedLights) {
            if (L.type != ExtractedLight::Type::Emissive) {
                continue; // skip directional/point/spot lights
            }
            lightsSubset.push_back(L);
            if (lightsSubset.size() >= MAX_ACTIVE_LIGHTS) {
                break;
            }
        }

        if (!lightsSubset.empty()) {
            // Update the light storage buffer with emissive lights only
            updateLightStorageBuffer(currentImage, lightsSubset);
            ubo.lightCount = static_cast<int>(lightsSubset.size());
        } else {
            ubo.lightCount = 0;
        }
    } else {
        ubo.lightCount = 0;
    }

    // Shadows removed: no shadow bias

    // Set camera position for PBR calculations
    ubo.camPos = glm::vec4(camera->GetPosition(), 1.0f);

    // Set PBR parameters (use member variables for UI control)
    // Clamp exposure to a sane range to avoid washout
    ubo.exposure = std::clamp(this->exposure, 0.2f, 4.0f);
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
    vk::ResultValue<uint32_t> result{{},0};
    try {
        result = swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphores[currentFrame]);
    } catch (const vk::OutOfDateKHRError&) {
        // Swapchain is out of date (e.g., window resized) before we could
        // query the result. Trigger recreation and exit this frame cleanly.
        framebufferResized.store(true, std::memory_order_relaxed);
        if (imguiSystem) ImGui::EndFrame();
        recreateSwapChain();
        return;
    }

    imageIndex = result.value;

    if (result.result == vk::Result::eErrorOutOfDateKHR || result.result == vk::Result::eSuboptimalKHR || framebufferResized.load(std::memory_order_relaxed)) {
        framebufferResized.store(false, std::memory_order_relaxed);
        if (imguiSystem) ImGui::EndFrame();
        recreateSwapChain();
        return;
    }
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to acquire swap chain image");
    }

    device.resetFences(*inFlightFences[currentFrame]);
    if (framebufferResized.load(std::memory_order_relaxed)) { recreateSwapChain(); return; }

    commandBuffers[currentFrame].reset();
    commandBuffers[currentFrame].begin(vk::CommandBufferBeginInfo());
    if (framebufferResized.load(std::memory_order_relaxed)) { commandBuffers[currentFrame].end(); recreateSwapChain(); return; }

    // Process texture streaming uploads (see Renderer::ProcessPendingTextureJobs)

    vk::raii::Pipeline* currentPipeline = nullptr;
    vk::raii::PipelineLayout* currentLayout = nullptr;
    std::vector<Entity*> blendedQueue;
    std::unordered_set<Entity*> blendedSet;

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
        // Only perform a small amount of streaming work every few frames.
        if ((streamingFrameCounter % 3) == 0) {
            ProcessPendingTextureJobs(/*maxJobs=*/1, /*includeCritical=*/false, /*includeNonCritical=*/true);
        }
    }

    bool blockScene = false;
    if (imguiSystem) {
        blockScene = IsLoading();
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
        // Sort by squared distance from the camera in world space.
        // Farther objects must be rendered first so that nearer glass correctly
        // appears in front (standard back-to-front transparency ordering).
        glm::vec3 camPos = camera ? camera->GetPosition() : glm::vec3(0.0f);
        std::ranges::sort(blendedQueue, [this, camPos](Entity* a, Entity* b) {
            auto* ta = (a ? a->GetComponent<TransformComponent>() : nullptr);
            auto* tb = (b ? b->GetComponent<TransformComponent>() : nullptr);
            glm::vec3 pa = ta ? ta->GetPosition() : glm::vec3(0.0f);
            glm::vec3 pb = tb ? tb->GetPosition() : glm::vec3(0.0f);
            float da2 = glm::length2(pa - camPos);
            float db2 = glm::length2(pb - camPos);

            // Primary key: distance (farther first)
            if (da2 != db2) {
                return da2 > db2;
            }

            // Secondary key: for entities at nearly the same distance, prefer
            // rendering liquid volumes before glass shells so bar glasses look
            // correctly filled. This is a heuristic based on material flags.
            auto classify = [this](Entity* e) {
                bool hasGlass = false;
                bool hasLiquid = false;
                if (!e || !modelLoader) return std::pair<bool,bool>{false, false};

                std::string name = e->GetName();
                size_t tagPos = name.find("_Material_");
                if (tagPos != std::string::npos) {
                    size_t afterTag = tagPos + std::string("_Material_").size();
                    if (afterTag < name.length()) {
                        std::string remainder = name.substr(afterTag);
                        size_t nextUnderscore = remainder.find('_');
                        if (nextUnderscore != std::string::npos && nextUnderscore + 1 < remainder.length()) {
                            std::string materialName = remainder.substr(nextUnderscore + 1);
                            if (Material* m = modelLoader->GetMaterial(materialName)) {
                                hasGlass  = m->isGlass;
                                hasLiquid = m->isLiquid;
                            }
                        }
                    }
                }
                return std::pair<bool,bool>{hasGlass, hasLiquid};
            };

            auto [aIsGlass, aIsLiquid] = classify(a);
            auto [bIsGlass, bIsLiquid] = classify(b);

            // If one is liquid and the other is glass at the same distance,
            // render the liquid first (i.e., treat it as slightly farther).
            if (aIsLiquid && bIsGlass && !bIsLiquid) {
                return true;  // a (liquid) comes before b (glass)
            }
            if (bIsLiquid && aIsGlass && !aIsLiquid) {
                return false; // b (liquid) comes before a (glass)
            }

            // Fallback to stable ordering when distances and classifications are equal.
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
        vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
        commandBuffers[currentFrame].setViewport(0, viewport);
        vk::Rect2D scissor({0, 0}, swapChainExtent);
        commandBuffers[currentFrame].setScissor(0, scissor);
        if (!blockScene) {
            for (const auto& uptr : entities) {
                Entity* entity = uptr.get();
                if (!entity || !entity->IsActive() || blendedSet.contains(entity)) continue;
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
        vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f);
        commandBuffers[currentFrame].setViewport(0, viewport);
        vk::Rect2D scissor({0, 0}, swapChainExtent);
        commandBuffers[currentFrame].setScissor(0, scissor);

        if (!blendedQueue.empty()) {
            currentLayout = &pbrTransparentPipelineLayout;

            // Track currently bound pipeline so we only rebind when needed
            vk::raii::Pipeline* activeTransparentPipeline = nullptr;

            for (Entity* entity : blendedQueue) {
                auto meshComponent = entity->GetComponent<MeshComponent>();
                auto entityIt = entityResources.find(entity);
                auto meshIt = meshResources.find(meshComponent);
                if (!meshComponent || entityIt == entityResources.end() || meshIt == meshResources.end()) continue;

                // Resolve material for this entity (if any)
                Material* material = nullptr;
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
                                material = modelLoader->GetMaterial(materialName);
                            }
                        }
                    }
                }

                // Choose pipeline: specialized glass pipeline for architectural glass,
                // otherwise the generic blended PBR pipeline.
                bool useGlassPipeline = material && material->isGlass;
                vk::raii::Pipeline* desiredPipeline = useGlassPipeline ? &glassGraphicsPipeline : &pbrBlendGraphicsPipeline;
                if (desiredPipeline != activeTransparentPipeline) {
                    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, **desiredPipeline);
                    activeTransparentPipeline = desiredPipeline;
                }

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

                    // For bar liquids and similar volumes, we want the fill to be
                    // clearly visible rather than fully transmissive. For these
                    // materials, disable the transmission branch in the PBR shader
                    // and treat them as regular alpha-blended PBR surfaces.
                    if (material->isLiquid) {
                        pushConstants.transmissionFactor = 0.0f;
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

    // Final layout transition and present
    vk::ImageMemoryBarrier presentBarrier{ .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .dstAccessMask = vk::AccessFlagBits::eNone, .oldLayout = vk::ImageLayout::eColorAttachmentOptimal, .newLayout = vk::ImageLayout::ePresentSrcKHR, .image = swapChainImages[imageIndex], .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} };
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, presentBarrier);
    commandBuffers[currentFrame].end();
    std::array<vk::Semaphore, 2> waitSems = { *imageAvailableSemaphores[currentFrame], *uploadsTimeline };
    std::array<vk::PipelineStageFlags, 2> waitStages = { vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader };
    uint64_t uploadsValueToWait = uploadTimelineLastSubmitted.load(std::memory_order_relaxed);
    std::array<uint64_t, 2> waitValues = { 0ull, uploadsValueToWait };
    vk::TimelineSemaphoreSubmitInfo timelineWaitInfo{ .waitSemaphoreValueCount = static_cast<uint32_t>(waitValues.size()), .pWaitSemaphoreValues = waitValues.data() };
    vk::SubmitInfo submitInfo{ .pNext = &timelineWaitInfo, .waitSemaphoreCount = static_cast<uint32_t>(waitSems.size()), .pWaitSemaphores = waitSems.data(), .pWaitDstStageMask = waitStages.data(), .commandBufferCount = 1, .pCommandBuffers = &*commandBuffers[currentFrame], .signalSemaphoreCount = 1, .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex] };
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
        result.result = presentQueue.presentKHR(presentInfo);
    } catch (const vk::OutOfDateKHRError&) {
        framebufferResized.store(true, std::memory_order_relaxed);
    }
    if (result.result == vk::Result::eErrorOutOfDateKHR || result.result == vk::Result::eSuboptimalKHR || framebufferResized.load(std::memory_order_relaxed)) {
        framebufferResized.store(false, std::memory_order_relaxed);
        recreateSwapChain();
    } else if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to present swap chain image");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}
