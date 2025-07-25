#include "renderer.h"
#include "imgui_system.h"
#include "imgui/imgui.h"
#include "model_loader.h"
#include <fstream>
#include <stdexcept>
#include <array>
#include <cstring>
#include <iostream>

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
    depthImageMemory = nullptr;

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
    for (const auto& fence : inFlightFences) {
        allFences.push_back(*fence);
    }
    if (!allFences.empty()) {
        device.waitForFences(allFences, VK_TRUE, UINT64_MAX);
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
        device.waitForFences(*fence, VK_TRUE, UINT64_MAX);
    }

    // Clear all entity descriptor sets since they're now invalid (allocated from old pool)
    for (auto& [entity, resources] : entityResources) {
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

    // Set up lighting from extracted GLTF lights or fallback to default
    std::vector<ExtractedLight> extractedLights;
    if (modelLoader) {
        extractedLights = modelLoader->GetExtractedLights("../Assets/Bistro.glb");
    }

    if (!extractedLights.empty()) {
        // Fill up to 4 lights from extracted lights
        for (size_t i = 0; i < 4 && i < extractedLights.size(); ++i) {
            const auto& light = extractedLights[i];
            ubo.lightPositions[i] = glm::vec4(light.position, 1.0f);
            ubo.lightColors[i] = glm::vec4(light.color * light.intensity, 1.0f);
        }

        // Fill remaining slots with dim lights if we have fewer than 4
        for (size_t i = extractedLights.size(); i < 4; ++i) {
            ubo.lightPositions[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            ubo.lightColors[i] = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f); // Very dim fallback
        }
    } else {
        // Fallback to default hardcoded lights if no extracted lights available
        std::cout << "No extracted lights found, using fallback lighting" << std::endl;

        // Light 0: Main key light (white, positioned above and to the right)
        ubo.lightPositions[0] = glm::vec4(10.0f, 10.0f, 10.0f, 1.0f);
        ubo.lightColors[0] = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

        // Light 1: Fill light (warm, positioned to the left)
        ubo.lightPositions[1] = glm::vec4(-8.0f, 5.0f, 8.0f, 1.0f);
        ubo.lightColors[1] = glm::vec4(0.8f, 0.7f, 0.6f, 1.0f);

        // Light 2: Rim light (cool, positioned behind)
        ubo.lightPositions[2] = glm::vec4(0.0f, 8.0f, -12.0f, 1.0f);
        ubo.lightColors[2] = glm::vec4(0.6f, 0.7f, 1.0f, 1.0f);

        // Light 3: Ambient fill (dim, positioned below)
        ubo.lightPositions[3] = glm::vec4(0.0f, -5.0f, 5.0f, 1.0f);
        ubo.lightColors[3] = glm::vec4(0.3f, 0.3f, 0.4f, 1.0f);
    }

    // Set camera position for PBR calculations
    ubo.camPos = glm::vec4(camera->GetPosition(), 1.0f);

    // Set PBR parameters
    ubo.exposure = 1.0f;
    ubo.gamma = 2.2f;
    ubo.prefilteredCubeMipLevels = 0.0f;
    ubo.scaleIBLAmbient = 1.0f;

    // Copy to uniform buffer
    std::memcpy(entityIt->second.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

// Render the scene
void Renderer::Render(const std::vector<Entity*>& entities, CameraComponent* camera, ImGuiSystem* imguiSystem) {

    // Wait for the previous frame to finish
    device.waitForFences(*inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

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
    } else if (result.first != vk::Result::eSuccess) {
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

        // Determine which pipeline to use based on ImGui PBR toggle and entity type
        bool usePBR = imguiSystem && imguiSystem->IsPBREnabled() && (entity->GetName().find("Bistro") == 0);
        vk::raii::Pipeline* selectedPipeline = usePBR ? &pbrGraphicsPipeline : &graphicsPipeline;
        vk::raii::PipelineLayout* selectedLayout = usePBR ? &pbrPipelineLayout : &pipelineLayout;

        // Get the mesh resources
        auto meshIt = meshResources.find(meshComponent);
        if (meshIt == meshResources.end()) {
            // Create mesh resources if they don't exist
            if (!createMeshResources(meshComponent)) {
                continue;
            }
            meshIt = meshResources.find(meshComponent);
        }

        // Get the entity resources
        auto entityIt = entityResources.find(entity);
        if (entityIt == entityResources.end()) {
            // Create entity resources if they don't exist
            if (!createUniformBuffers(entity)) {
                continue;
            }

            // Create descriptor sets with correct pipeline type
            if (!createDescriptorSets(entity, meshComponent->GetTexturePath(), usePBR)) {
                continue;
            }

            entityIt = entityResources.find(entity);
        } else {
            // Entity exists, but check if we need to create descriptor sets for the current pipeline type
            auto& targetDescriptorSets = usePBR ? entityIt->second.pbrDescriptorSets : entityIt->second.basicDescriptorSets;
            if (targetDescriptorSets.empty()) {
                // Create missing descriptor sets for the current pipeline type
                if (!createDescriptorSets(entity, meshComponent->GetTexturePath(), usePBR)) {
                    continue;
                }
            }
        }

        // Bind pipeline if it changed
        if (currentPipeline != selectedPipeline) {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, **selectedPipeline);
            currentPipeline = selectedPipeline;
            currentLayout = selectedLayout;

            if (usePBR) {
                std::cout << "Using PBR pipeline for entity: " << entity->GetName() << std::endl;
            }
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

        // Push constants for PBR pipeline
        if (usePBR) {
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
            };

            // Set default PBR material properties
            PushConstants pushConstants{};
            pushConstants.baseColorFactor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // White base color
            pushConstants.metallicFactor = 0.0f;   // Non-metallic
            pushConstants.roughnessFactor = 1.0f;  // Fully rough
            pushConstants.baseColorTextureSet = 0;      // Use texture set 0
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

        // Draw the mesh
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
