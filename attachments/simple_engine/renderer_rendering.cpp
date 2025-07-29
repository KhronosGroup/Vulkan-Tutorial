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
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
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

    // Clean up descriptor pool (this will automatically clean up descriptor sets)
    descriptorPool = nullptr;

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
    createDepthResources();

    // Recreate sync objects with correct sizing for new swap chain
    createSyncObjects();

    // Recreate descriptor pool and pipelines
    createDescriptorPool();

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

    // Use static lights loaded during model initialization
    const std::vector<ExtractedLight>& extractedLights = staticLights;

    if (!extractedLights.empty()) {
        // Use all available lights (no hardcoded limit)
        size_t numLights = extractedLights.size();

        // Update the light storage buffer with all light data
        updateLightStorageBuffer(currentImage, extractedLights);

        ubo.lightCount = static_cast<int>(numLights);
        // Set shadow map count based on shadowsEnabled flag
        if (shadowsEnabled) {
            ubo.shadowMapCount = static_cast<int>(std::min(numLights, size_t(16))); // Limit shadow maps to 16 for performance (indices 0-15)
        } else {
            ubo.shadowMapCount = 0; // Disable shadows when shadowsEnabled is false
        }
    } else {
        ubo.lightCount = 0;
        ubo.shadowMapCount = 0;
    }

    // Set shadow mapping parameters
    ubo.shadowBias = 0.005f; // Adjust to prevent shadow acne

    // Set camera position for PBR calculations
    ubo.camPos = glm::vec4(camera->GetPosition(), 1.0f);

    // Set PBR parameters (use member variables for UI control)
    ubo.exposure = this->exposure;
    ubo.gamma = this->gamma;
    ubo.prefilteredCubeMipLevels = 0.0f;
    ubo.scaleIBLAmbient = 1.0f;

    // Copy to uniform buffer
    std::memcpy(entityIt->second.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

// Render the scene
void Renderer::Render(const std::vector<Entity*>& entities, CameraComponent* camera, ImGuiSystem* imguiSystem) {
    // Set rendering active to prevent memory pool growth during rendering
    if (memoryPool) {
        memoryPool->setRenderingActive(true);
    }

    // Use RAII to ensure rendering state is always reset, even if an exception occurs
    struct RenderingStateGuard {
        MemoryPool* pool;
        explicit RenderingStateGuard(MemoryPool* p) : pool(p) {}
        ~RenderingStateGuard() {
            if (pool) {
                pool->setRenderingActive(false);
            }
        }
    } guard(memoryPool.get());

    // Wait for the previous frame to finish
    if (device.waitForFences(*inFlightFences[currentFrame], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess) {}

    // Acquire the next image from the swap chain
    uint32_t imageIndex;
    // Use currentFrame for semaphore indexing to ensure consistency
    uint32_t semaphoreIndex = currentFrame % imageAvailableSemaphores.size();
    auto result = swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphores[semaphoreIndex]);
    imageIndex = result.second;

    // Check if the swap chain needs to be recreated
    if (result.first == vk::Result::eErrorOutOfDateKHR || result.first == vk::Result::eSuboptimalKHR || framebufferResized) {
        framebufferResized = false;

        // If ImGui has started a frame, we need to end it properly before returning
        if (imguiSystem) {
            ImGui::EndFrame();
        }

        recreateSwapChain();
        return;
    }
    if (result.first != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to acquire swap chain image");
    }

    // Reset the fence for the current frame
    device.resetFences(*inFlightFences[currentFrame]);

    // Reset the command buffer
    commandBuffers[currentFrame].reset();

    // Record the command buffer
    commandBuffers[currentFrame].begin(vk::CommandBufferBeginInfo());

    // Update dynamic rendering attachments
    colorAttachments[0].setImageView(*swapChainImageViews[imageIndex]);
    depthAttachment.setImageView(*depthImageView);

    // Update rendering area
    renderingInfo.setRenderArea(vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

    // Transition swapchain image layout for rendering
    vk::ImageMemoryBarrier renderBarrier{
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapChainImages[imageIndex],
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    commandBuffers[currentFrame].pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::DependencyFlags{},
        {},
        {},
        renderBarrier
    );

    // Begin dynamic rendering with vk::raii
    commandBuffers[currentFrame].beginRendering(renderingInfo);

    // Set the viewport
    vk::Viewport viewport(0.0f, 0.0f,
                         static_cast<float>(swapChainExtent.width),
                         static_cast<float>(swapChainExtent.height),
                         0.0f, 1.0f);
    commandBuffers[currentFrame].setViewport(0, viewport);

    // Set the scissor
    vk::Rect2D scissor(vk::Offset2D(0, 0), swapChainExtent);
    commandBuffers[currentFrame].setScissor(0, scissor);

    // Track current pipeline to avoid unnecessary bindings
    vk::raii::Pipeline* currentPipeline = nullptr;
    vk::raii::PipelineLayout* currentLayout = nullptr;

    // Render each entity
    for (auto entity : entities) {
        // Get the mesh component
        auto meshComponent = entity->GetComponent<MeshComponent>();
        if (!meshComponent) {
            continue;
        }

        // Get the transform component
        auto transformComponent = entity->GetComponent<TransformComponent>();
        if (!transformComponent) {
            continue;
        }

        // Determine which pipeline to use - now defaults to BRDF/PBR instead of Phong
        // Use basic pipeline only when PBR is explicitly disabled via ImGui
        bool useBasic = imguiSystem && !imguiSystem->IsPBREnabled();
        bool usePBR = !useBasic; // BRDF/PBR is now the default lighting model
        vk::raii::Pipeline* selectedPipeline = usePBR ? &pbrGraphicsPipeline : &graphicsPipeline;
        vk::raii::PipelineLayout* selectedLayout = usePBR ? &pbrPipelineLayout : &pipelineLayout;

        // Get the mesh resources - they should already exist from pre-allocation
        auto meshIt = meshResources.find(meshComponent);
        if (meshIt == meshResources.end()) {
            std::cerr << "ERROR: Mesh resources not found for entity " << entity->GetName()
                      << " - resources should have been pre-allocated during scene loading!" << std::endl;
            continue;
        }

        // Get the entity resources - they should already exist from pre-allocation
        auto entityIt = entityResources.find(entity);
        if (entityIt == entityResources.end()) {
            std::cerr << "ERROR: Entity resources not found for entity " << entity->GetName()
                      << " - resources should have been pre-allocated during scene loading!" << std::endl;
            continue;
        }

        // Bind pipeline if it changed
        if (currentPipeline != selectedPipeline) {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, **selectedPipeline);
            currentPipeline = selectedPipeline;
            currentLayout = selectedLayout;
        }

        // Update the uniform buffer
        updateUniformBuffer(currentFrame, entity, camera);

        // Bind the vertex buffer
        std::array<vk::Buffer, 1> vertexBuffers = {*meshIt->second.vertexBuffer};
        std::array<vk::DeviceSize, 1> offsets = {0};
        commandBuffers[currentFrame].bindVertexBuffers(0, vertexBuffers, offsets);

        // Bind the index buffer
        commandBuffers[currentFrame].bindIndexBuffer(*meshIt->second.indexBuffer, 0, vk::IndexType::eUint32);

        // Bind the descriptor set using the appropriate pipeline layout
        auto& selectedDescriptorSets = usePBR ? entityIt->second.pbrDescriptorSets : entityIt->second.basicDescriptorSets;

        // Check if descriptor sets exist for the current pipeline type
        if (selectedDescriptorSets.empty()) {
            std::cerr << "Error: No descriptor sets available for entity " << entity->GetName()
                      << " (pipeline: " << (usePBR ? "PBR" : "basic") << ")" << std::endl;
            continue; // Skip this entity
        }

        if (currentFrame >= selectedDescriptorSets.size()) {
            std::cerr << "Error: Invalid frame index " << currentFrame
                      << " for entity " << entity->GetName()
                      << " (descriptor sets size: " << selectedDescriptorSets.size() << ")" << std::endl;
            continue; // Skip this entity
        }

        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **currentLayout, 0, {*selectedDescriptorSets[currentFrame]}, {});

        // Define PBR push constants structure matching the shader
        struct PushConstants {
            glm::vec4 baseColorFactor;
            float metallicFactor;
            float roughnessFactor;
            int baseColorTextureSet;
            int physicalDescriptorTextureSet;
            int normalTextureSet;
            int occlusionTextureSet;
            int emissiveTextureSet;
            float alphaMask;
            float alphaMaskCutoff;
            glm::vec3 emissiveFactor;  // Add emissive factor for HDR emissive sources
            float emissiveStrength;    // KHR_materials_emissive_strength extension
        };

        // Set PBR material properties using push constants
        if (usePBR) {
            PushConstants pushConstants{};

            // Try to get material properties for this specific entity
            if (modelLoader && entity->GetName().find("_Material_") != std::string::npos) {
                // Extract material name from entity name for any GLTF model entities
                std::string entityName = entity->GetName();
                size_t materialStart = entityName.find("_Material_");
                if (materialStart != std::string::npos) {
                    // Try to extract material name from entity name
                    size_t nameStart = entityName.find_last_of("_");
                    if (nameStart != std::string::npos && nameStart < entityName.length() - 1) {
                        std::string materialName = entityName.substr(nameStart + 1);
                        Material* material = modelLoader->GetMaterial(materialName);
                        if (material) {
                            // Use actual PBR properties from the GLTF material
                            pushConstants.baseColorFactor = glm::vec4(material->albedo, 1.0f);
                            pushConstants.metallicFactor = material->metallic;
                            pushConstants.roughnessFactor = material->roughness;
                            pushConstants.emissiveFactor = material->emissive;  // Set emissive factor for HDR emissive sources
                            pushConstants.emissiveStrength = material->emissiveStrength;  // Set emissive strength from KHR_materials_emissive_strength extension
                        } else {
                            // Fallback: Use entity-specific variation
                            size_t hash = std::hash<std::string>{}(entityName);
                            float variation = (hash % 100) / 100.0f;
                            pushConstants.baseColorFactor = glm::vec4(0.7f + variation * 0.3f, 0.6f + variation * 0.4f, 0.5f + variation * 0.5f, 1.0f);
                            pushConstants.metallicFactor = variation * 0.8f;
                            pushConstants.roughnessFactor = 0.2f + variation * 0.6f;
                            pushConstants.emissiveFactor = glm::vec3(0.0f);  // No emissive for fallback materials
                            pushConstants.emissiveStrength = 1.0f;  // Default emissive strength
                        }
                    }
                }
            } else {
                // Default PBR material properties for non-Bistro entities
                pushConstants.baseColorFactor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
                pushConstants.metallicFactor = 0.1f;
                pushConstants.roughnessFactor = 0.7f;
                pushConstants.emissiveFactor = glm::vec3(0.0f);  // No emissive for default materials
                pushConstants.emissiveStrength = 1.0f;  // Default emissive strength
            }

            // Apply highlighting effect if this entity is highlighted
            if (highlightedEntity == entity) {
                // Add a bright yellow tint to highlight the entity
                pushConstants.baseColorFactor.r = std::min(1.0f, pushConstants.baseColorFactor.r + 0.3f);
                pushConstants.baseColorFactor.g = std::min(1.0f, pushConstants.baseColorFactor.g + 0.3f);
                pushConstants.baseColorFactor.b = std::min(1.0f, pushConstants.baseColorFactor.b * 0.7f); // Reduce blue for yellow tint
                // Also add some emissive glow for extra visibility
                pushConstants.emissiveFactor = glm::vec3(0.2f, 0.2f, 0.0f);
                pushConstants.emissiveStrength = 2.0f;
            }

            // Set texture binding indices
            pushConstants.baseColorTextureSet = 0;
            pushConstants.physicalDescriptorTextureSet = 0;
            pushConstants.normalTextureSet = 0;
            pushConstants.occlusionTextureSet = 0;
            pushConstants.emissiveTextureSet = 0;
            pushConstants.alphaMask = 0.0f;
            pushConstants.alphaMaskCutoff = 0.5f;

            // Push constants to the shader
            commandBuffers[currentFrame].pushConstants(
                **currentLayout,
                vk::ShaderStageFlagBits::eFragment,
                0,
                vk::ArrayProxy<const uint8_t>(sizeof(PushConstants), reinterpret_cast<const uint8_t*>(&pushConstants))
            );
        }

        // Draw the entity once (no redundant draw calls)
        commandBuffers[currentFrame].drawIndexed(meshIt->second.indexCount, 1, 0, 0, 0);
    }

    // Render ImGui if provided
    if (imguiSystem) {
        imguiSystem->Render(commandBuffers[currentFrame]);
    }

    // End dynamic rendering
    commandBuffers[currentFrame].endRendering();

    // Transition swapchain image layout for presentation
    vk::ImageMemoryBarrier imageBarrier{
        .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .dstAccessMask = vk::AccessFlagBits::eNone,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::ePresentSrcKHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = swapChainImages[imageIndex],
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    commandBuffers[currentFrame].pipelineBarrier(
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eBottomOfPipe,
        vk::DependencyFlags{},
        {},
        {},
        imageBarrier
    );

    // End command buffer
    commandBuffers[currentFrame].end();

    // Submit command buffer
    vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
    vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*imageAvailableSemaphores[semaphoreIndex],
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[semaphoreIndex]
    };

    // Use mutex to ensure thread-safe access to graphics queue
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);
    }

    // Present the image
    vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[semaphoreIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex
    };

    // Use mutex to ensure thread-safe access to present queue
    try {
        std::lock_guard<std::mutex> lock(queueMutex);
        result.first = presentQueue.presentKHR(presentInfo);
    } catch (const vk::OutOfDateKHRError&) {
        framebufferResized = true;
    }

    if (result.first == vk::Result::eErrorOutOfDateKHR || result.first == vk::Result::eSuboptimalKHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    } else if (result.first != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to present swap chain image");
    }

    // Advance to the next frame
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}
