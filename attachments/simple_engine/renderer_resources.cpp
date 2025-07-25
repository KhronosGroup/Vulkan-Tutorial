#include "renderer.h"
#include <fstream>
#include <stdexcept>
#include <array>
#include <iostream>
#include <cstring>

// Define STB_IMAGE_IMPLEMENTATION before including stb_image.h to provide the implementation
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// This file contains resource-related methods from the Renderer class

// Create depth resources
bool Renderer::createDepthResources() {
    try {
        // Find depth format
        vk::Format depthFormat = findDepthFormat();

        // Create depth image
        auto [depthImg, depthImgMem] = createImage(
            swapChainExtent.width,
            swapChainExtent.height,
            depthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        depthImage = std::move(depthImg);
        depthImageMemory = std::move(depthImgMem);

        // Create depth image view
        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);

        // Transition depth image layout
        transitionImageLayout(
            *depthImage,
            depthFormat,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal
        );

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create depth resources: " << e.what() << std::endl;
        return false;
    }
}

// Create texture image
bool Renderer::createTextureImage(const std::string& texturePath, TextureResources& resources) {
    try {
        // Check if texture already exists
        auto it = textureResources.find(texturePath);
        if (it != textureResources.end()) {
            resources = std::move(it->second);
            return true;
        }

        // Load texture image
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(texturePath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            std::cerr << "Failed to load texture image: " << texturePath << std::endl;
            return false;
        }

        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        // Create staging buffer
        auto [stagingBuffer, stagingBufferMemory] = createBuffer(
            imageSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        // Copy pixel data to staging buffer
        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        stagingBufferMemory.unmapMemory();

        // Free pixel data
        stbi_image_free(pixels);

        // Create texture image
        auto [textureImg, textureImgMem] = createImage(
            texWidth,
            texHeight,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        resources.textureImage = std::move(textureImg);
        resources.textureImageMemory = std::move(textureImgMem);

        // Transition image layout for copy
        transitionImageLayout(
            *resources.textureImage,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        // Copy buffer to image
        copyBufferToImage(
            *stagingBuffer,
            *resources.textureImage,
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight)
        );

        // Transition image layout for shader access
        transitionImageLayout(
            *resources.textureImage,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        // Create texture image view
        if (!createTextureImageView(resources)) {
            return false;
        }

        // Create texture sampler
        if (!createTextureSampler(resources)) {
            return false;
        }

        // Add to texture resources map
        textureResources[texturePath] = std::move(resources);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create texture image: " << e.what() << std::endl;
        return false;
    }
}

// Create texture image view
bool Renderer::createTextureImageView(TextureResources& resources) {
    try {
        resources.textureImageView = createImageView(
            resources.textureImage,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageAspectFlagBits::eColor
        );
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create texture image view: " << e.what() << std::endl;
        return false;
    }
}

// Create default texture resources (1x1 white texture)
bool Renderer::createDefaultTextureResources() {
    try {
        // Create a 1x1 white texture
        const uint32_t width = 1;
        const uint32_t height = 1;
        const uint32_t pixelSize = 4; // RGBA
        const std::vector<uint8_t> pixels = {255, 255, 255, 255}; // White pixel (RGBA)

        // Create staging buffer
        vk::DeviceSize imageSize = width * height * pixelSize;
        auto [stagingBuffer, stagingBufferMemory] = createBuffer(
            imageSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        // Copy pixel data to staging buffer
        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels.data(), static_cast<size_t>(imageSize));
        stagingBufferMemory.unmapMemory();

        // Create texture image
        auto [textureImg, textureImgMem] = createImage(
            width,
            height,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        defaultTextureResources.textureImage = std::move(textureImg);
        defaultTextureResources.textureImageMemory = std::move(textureImgMem);

        // Transition image layout for copy
        transitionImageLayout(
            *defaultTextureResources.textureImage,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        // Copy buffer to image
        copyBufferToImage(
            *stagingBuffer,
            *defaultTextureResources.textureImage,
            width,
            height
        );

        // Transition image layout for shader access
        transitionImageLayout(
            *defaultTextureResources.textureImage,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        // Create texture image view
        defaultTextureResources.textureImageView = createImageView(
            defaultTextureResources.textureImage,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageAspectFlagBits::eColor
        );

        // Create texture sampler
        return createTextureSampler(defaultTextureResources);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create default texture resources: " << e.what() << std::endl;
        return false;
    }
}

// Create texture sampler
bool Renderer::createTextureSampler(TextureResources& resources) {
    try {
        // Get physical device properties
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

        // Create sampler
        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = VK_FALSE,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = VK_FALSE
        };

        resources.textureSampler = vk::raii::Sampler(device, samplerInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create texture sampler: " << e.what() << std::endl;
        return false;
    }
}

// Create mesh resources
bool Renderer::createMeshResources(MeshComponent* meshComponent) {
    try {
        // Check if mesh resources already exist
        auto it = meshResources.find(meshComponent);
        if (it != meshResources.end()) {
            return true;
        }

        // Get mesh data
        const auto& vertices = meshComponent->GetVertices();
        const auto& indices = meshComponent->GetIndices();

        if (vertices.empty() || indices.empty()) {
            std::cerr << "Mesh has no vertices or indices" << std::endl;
            return false;
        }

        // Create vertex buffer
        vk::DeviceSize vertexBufferSize = sizeof(vertices[0]) * vertices.size();
        auto [stagingVertexBuffer, stagingVertexBufferMemory] = createBuffer(
            vertexBufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        // Copy vertex data to staging buffer
        void* vertexData = stagingVertexBufferMemory.mapMemory(0, vertexBufferSize);
        memcpy(vertexData, vertices.data(), static_cast<size_t>(vertexBufferSize));
        stagingVertexBufferMemory.unmapMemory();

        // Create vertex buffer on device
        auto [vertexBuffer, vertexBufferMemory] = createBuffer(
            vertexBufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        // Copy from staging buffer to device buffer
        copyBuffer(stagingVertexBuffer, vertexBuffer, vertexBufferSize);

        // Create index buffer
        vk::DeviceSize indexBufferSize = sizeof(indices[0]) * indices.size();
        auto [stagingIndexBuffer, stagingIndexBufferMemory] = createBuffer(
            indexBufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        // Copy index data to staging buffer
        void* indexData = stagingIndexBufferMemory.mapMemory(0, indexBufferSize);
        memcpy(indexData, indices.data(), static_cast<size_t>(indexBufferSize));
        stagingIndexBufferMemory.unmapMemory();

        // Create index buffer on device
        auto [indexBuffer, indexBufferMemory] = createBuffer(
            indexBufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        // Copy from staging buffer to device buffer
        copyBuffer(stagingIndexBuffer, indexBuffer, indexBufferSize);

        // Create mesh resources
        MeshResources resources;
        resources.vertexBuffer = std::move(vertexBuffer);
        resources.vertexBufferMemory = std::move(vertexBufferMemory);
        resources.indexBuffer = std::move(indexBuffer);
        resources.indexBufferMemory = std::move(indexBufferMemory);
        resources.indexCount = static_cast<uint32_t>(indices.size());

        // Add to mesh resources map
        meshResources[meshComponent] = std::move(resources);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create mesh resources: " << e.what() << std::endl;
        return false;
    }
}

// Create uniform buffers
bool Renderer::createUniformBuffers(Entity* entity) {
    try {
        // Check if entity resources already exist
        auto it = entityResources.find(entity);
        if (it != entityResources.end()) {
            return true;
        }

        // Create entity resources
        EntityResources resources;

        // Create uniform buffers
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            auto [buffer, bufferMemory] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            void* mappedMemory = bufferMemory.mapMemory(0, bufferSize);

            resources.uniformBuffers.emplace_back(std::move(buffer));
            resources.uniformBuffersMemory.emplace_back(std::move(bufferMemory));
            resources.uniformBuffersMapped.emplace_back(mappedMemory);
        }

        // Add to entity resources map
        entityResources[entity] = std::move(resources);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create uniform buffers: " << e.what() << std::endl;
        return false;
    }
}

// Create descriptor pool
bool Renderer::createDescriptorPool() {
    try {
        // Create descriptor pool sizes
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 100)
            },
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 100)
            }
        };

        // Create descriptor pool
        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * 100),
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create descriptor pool: " << e.what() << std::endl;
        return false;
    }
}

// Create descriptor sets
bool Renderer::createDescriptorSets(Entity* entity, const std::string& texturePath) {
    try {
        // Get entity resources
        auto entityIt = entityResources.find(entity);
        if (entityIt == entityResources.end()) {
            std::cerr << "Entity resources not found" << std::endl;
            return false;
        }

        // Get texture resources
        TextureResources textureRes;
        if (!texturePath.empty()) {
            auto textureIt = textureResources.find(texturePath);
            if (textureIt == textureResources.end()) {
                // Create texture resources if they don't exist
                if (!createTextureImage(texturePath, textureRes)) {
                    return false;
                }
            } else {
                textureRes = std::move(textureIt->second);
            }
        }

        // Create descriptor sets using RAII
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts = layouts.data()
        };

        // Allocate descriptor sets using RAII wrapper
        vk::raii::DescriptorSets raiiDescriptorSets(device, allocInfo);

        // Convert to vector of individual RAII descriptor sets
        entityIt->second.descriptorSets.clear();
        entityIt->second.descriptorSets.reserve(MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            entityIt->second.descriptorSets.emplace_back(std::move(raiiDescriptorSets[i]));
            std::cout << "Created descriptor set " << i << " with handle: " << *entityIt->second.descriptorSets[i] << std::endl;
        }

        // Update descriptor sets
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // Uniform buffer descriptor
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = *entityIt->second.uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };

            // Always update both descriptors
            std::array<vk::WriteDescriptorSet, 2> descriptorWrites;

            // Uniform buffer descriptor write
            descriptorWrites[0] = vk::WriteDescriptorSet{
                .dstSet = entityIt->second.descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo
            };

            // Check if texture resources are valid
            bool hasValidTexture = !texturePath.empty() &&
                                  *textureRes.textureSampler &&
                                  *textureRes.textureImageView;

            // Texture sampler descriptor
            vk::DescriptorImageInfo imageInfo;
            if (hasValidTexture) {
                // Use provided texture resources
                imageInfo = vk::DescriptorImageInfo{
                    .sampler = *textureRes.textureSampler,
                    .imageView = *textureRes.textureImageView,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                };
            } else {
                // Use default texture resources
                imageInfo = vk::DescriptorImageInfo{
                    .sampler = *defaultTextureResources.textureSampler,
                    .imageView = *defaultTextureResources.textureImageView,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                };
            }

            // Texture sampler descriptor write
            descriptorWrites[1] = vk::WriteDescriptorSet{
                .dstSet = entityIt->second.descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo
            };

            // Update descriptor sets with both descriptors
            device.updateDescriptorSets(descriptorWrites, {});
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create descriptor sets: " << e.what() << std::endl;
        return false;
    }
}

// Create buffer
std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> Renderer::createBuffer(
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties) {
    try {
        // Create buffer
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        vk::raii::Buffer buffer(device, bufferInfo);

        // Allocate memory
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        vk::raii::DeviceMemory bufferMemory(device, allocInfo);

        // Bind memory to buffer
        buffer.bindMemory(*bufferMemory, 0);

        return {std::move(buffer), std::move(bufferMemory)};
    } catch (const std::exception& e) {
        std::cerr << "Failed to create buffer: " << e.what() << std::endl;
        throw;
    }
}

// Copy buffer
void Renderer::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
    try {
        // Create command buffer using RAII
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffers commandBuffers(device, allocInfo);
        vk::raii::CommandBuffer& commandBuffer = commandBuffers[0];

        // Begin command buffer
        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        commandBuffer.begin(beginInfo);

        // Copy buffer
        vk::BufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size
        };

        commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

        // End command buffer
        commandBuffer.end();

        // Submit command buffer
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };

        // Use mutex to ensure thread-safe access to graphics queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            graphicsQueue.submit(submitInfo, nullptr);
            graphicsQueue.waitIdle();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to copy buffer: " << e.what() << std::endl;
        throw;
    }
}

// Create image
std::pair<vk::raii::Image, vk::raii::DeviceMemory> Renderer::createImage(
    uint32_t width,
    uint32_t height,
    vk::Format format,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlags properties) {
    try {
        // Create image
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        vk::raii::Image image(device, imageInfo);

        // Allocate memory
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        vk::raii::DeviceMemory imageMemory(device, allocInfo);

        // Bind memory to image
        image.bindMemory(*imageMemory, 0);

        return {std::move(image), std::move(imageMemory)};
    } catch (const std::exception& e) {
        std::cerr << "Failed to create image: " << e.what() << std::endl;
        throw;
    }
}

// Create image view
vk::raii::ImageView Renderer::createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
    try {
        // Create image view
        vk::ImageViewCreateInfo viewInfo{
            .image = *image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspectFlags,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        return vk::raii::ImageView(device, viewInfo);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create image view: " << e.what() << std::endl;
        throw;
    }
}

// Transition image layout
void Renderer::transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    try {
        // Create command buffer using RAII
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffers commandBuffers(device, allocInfo);
        vk::raii::CommandBuffer& commandBuffer = commandBuffers[0];

        // Begin command buffer
        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        commandBuffer.begin(beginInfo);

        // Create image barrier
        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = format == vk::Format::eD32Sfloat || format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint
                    ? vk::ImageAspectFlagBits::eDepth
                    : vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        // Set access masks and pipeline stages based on layouts
        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eNone;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eNone;
            barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else {
            throw std::invalid_argument("Unsupported layout transition!");
        }

        // Add barrier to command buffer
        commandBuffer.pipelineBarrier(
            sourceStage, destinationStage,
            vk::DependencyFlagBits::eByRegion,
            nullptr,
            nullptr,
            barrier
        );

        // End command buffer
        commandBuffer.end();

        // Submit command buffer
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };

        // Use mutex to ensure thread-safe access to graphics queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            graphicsQueue.submit(submitInfo, nullptr);
            graphicsQueue.waitIdle();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to transition image layout: " << e.what() << std::endl;
        throw;
    }
}

// Copy buffer to image
void Renderer::copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
    try {
        // Create command buffer using RAII
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffers commandBuffers(device, allocInfo);
        vk::raii::CommandBuffer& commandBuffer = commandBuffers[0];

        // Begin command buffer
        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        commandBuffer.begin(beginInfo);

        // Create buffer image copy region
        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1}
        };

        // Copy buffer to image
        commandBuffer.copyBufferToImage(
            buffer,
            image,
            vk::ImageLayout::eTransferDstOptimal,
            region
        );

        // End command buffer
        commandBuffer.end();

        // Submit command buffer
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };

        // Use mutex to ensure thread-safe access to graphics queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            graphicsQueue.submit(submitInfo, nullptr);
            graphicsQueue.waitIdle();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to copy buffer to image: " << e.what() << std::endl;
        throw;
    }
}
