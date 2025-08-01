#include "renderer.h"
#include "model_loader.h"
#include <iostream>
#include <array>

// This file contains shadow mapping implementation for the Renderer class

bool Renderer::createShadowMaps() {
    try {
        std::cout << "Creating shadow maps..." << std::endl;

        // Initialize shadow maps vector - limit to 16 for performance/memory
        const uint32_t ACTUAL_SHADOW_MAPS = 16;
        shadowMaps.resize(ACTUAL_SHADOW_MAPS);

        for (uint32_t i = 0; i < ACTUAL_SHADOW_MAPS; ++i) {
            auto& shadowMap = shadowMaps[i];

            // Create shadow map image using memory pool
            auto [shadowImg, shadowImgAllocation] = createImagePooled(
                DEFAULT_SHADOW_MAP_SIZE,
                DEFAULT_SHADOW_MAP_SIZE,
                vk::Format::eD32Sfloat, // 32-bit depth format for high precision
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal
            );

            shadowMap.shadowMapImage = std::move(shadowImg);
            shadowMap.shadowMapImageAllocation = std::move(shadowImgAllocation);
            shadowMap.shadowMapSize = DEFAULT_SHADOW_MAP_SIZE;

            // Create shadow map image view
            shadowMap.shadowMapImageView = createImageView(
                shadowMap.shadowMapImage,
                vk::Format::eD32Sfloat,
                vk::ImageAspectFlagBits::eDepth
            );

            // Create shadow map sampler
            vk::SamplerCreateInfo samplerInfo{
                .magFilter = vk::Filter::eLinear,
                .minFilter = vk::Filter::eLinear,
                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                .addressModeU = vk::SamplerAddressMode::eClampToBorder,
                .addressModeV = vk::SamplerAddressMode::eClampToBorder,
                .addressModeW = vk::SamplerAddressMode::eClampToBorder,
                .mipLodBias = 0.0f,
                .anisotropyEnable = VK_FALSE,
                .maxAnisotropy = 1.0f,
                .compareEnable = VK_TRUE, // Enable depth comparison for shadow mapping
                .compareOp = vk::CompareOp::eLessOrEqual,
                .minLod = 0.0f,
                .maxLod = 1.0f,
                .borderColor = vk::BorderColor::eFloatOpaqueWhite, // White border = no shadow
                .unnormalizedCoordinates = VK_FALSE
            };

            shadowMap.shadowMapSampler = vk::raii::Sampler(device, samplerInfo);

            // Transition shadow map to read-only layout for shader sampling
            // Shadow maps will be transitioned to attachment layout when rendering, then back to read-only
            transitionImageLayout(
                *shadowMap.shadowMapImage,
                vk::Format::eD32Sfloat,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eDepthStencilReadOnlyOptimal
            );

            std::cout << "  Created shadow map " << i << " (" << DEFAULT_SHADOW_MAP_SIZE << "x" << DEFAULT_SHADOW_MAP_SIZE << ")" << std::endl;
        }

        std::cout << "Successfully created " << ACTUAL_SHADOW_MAPS << " shadow maps" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to create shadow maps: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createShadowMapRenderPass() {
    try {
        std::cout << "Creating shadow map render pass..." << std::endl;

        // We'll use dynamic rendering instead of traditional render passes
        // This is more flexible and matches our existing rendering approach

        std::cout << "Shadow map render pass created (using dynamic rendering)" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to create shadow map render pass: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createShadowMapFramebuffers() {
    try {
        std::cout << "Creating shadow map framebuffers..." << std::endl;

        // With dynamic rendering, we don't need traditional framebuffers
        // The shadow map images will be used directly in dynamic rendering

        std::cout << "Shadow map framebuffers created (using dynamic rendering)" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to create shadow map framebuffers: " << e.what() << std::endl;
        return false;
    }
}

bool Renderer::createShadowMapDescriptorSetLayout() {
    try {
        std::cout << "Creating shadow map descriptor set layout..." << std::endl;

        // We need to update the existing PBR descriptor set layout to include shadow maps
        // This will be done by modifying the createPBRDescriptorSetLayout method

        std::cout << "Shadow map descriptor set layout will be integrated with PBR layout" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to create shadow map descriptor set layout: " << e.what() << std::endl;
        return false;
    }
}

void Renderer::renderShadowMaps(const std::vector<Entity*>& entities, const std::vector<ExtractedLight>& lights) {
    // This method will render the scene from each light's perspective to generate shadow maps
    // Implementation will be added after basic shadow map creation is working
    std::cout << "Shadow map rendering not yet implemented" << std::endl;
}

void Renderer::updateShadowMapUniforms(uint32_t lightIndex, const ExtractedLight& light) {
    // This method will calculate and update the light space matrix for a specific light
    // Implementation will be added after basic shadow map creation is working
    std::cout << "Shadow map uniform updates not yet implemented" << std::endl;
}
