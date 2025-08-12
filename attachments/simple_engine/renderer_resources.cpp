#include "renderer.h"
#include "model_loader.h"
#include "mesh_component.h"
#include "transform_component.h"
#include <fstream>
#include <stdexcept>
#include <array>
#include <iostream>
#include <filesystem>
#include <cstring>

// stb_image dependency removed; all GLTF textures are uploaded via memory path from ModelLoader.

// KTX2 support
#include <ktx.h>
#include <ktxvulkan.h>

// This file contains resource-related methods from the Renderer class

// Define shared default PBR texture identifiers (static constants)
const std::string Renderer::SHARED_DEFAULT_ALBEDO_ID = "__shared_default_albedo__";
const std::string Renderer::SHARED_DEFAULT_NORMAL_ID = "__shared_default_normal__";
const std::string Renderer::SHARED_DEFAULT_METALLIC_ROUGHNESS_ID = "__shared_default_metallic_roughness__";
const std::string Renderer::SHARED_DEFAULT_OCCLUSION_ID = "__shared_default_occlusion__";
const std::string Renderer::SHARED_DEFAULT_EMISSIVE_ID = "__shared_default_emissive__";
const std::string Renderer::SHARED_BRIGHT_RED_ID = "__shared_bright_red__";

// Create depth resources
bool Renderer::createDepthResources() {
    try {
        // Find depth format
        vk::Format depthFormat = findDepthFormat();

        // Create depth image using memory pool
        auto [depthImg, depthImgAllocation] = createImagePooled(
            swapChainExtent.width,
            swapChainExtent.height,
            depthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        depthImage = std::move(depthImg);
        depthImageAllocation = std::move(depthImgAllocation);

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
bool Renderer::createTextureImage(const std::string& texturePath_, TextureResources& resources) {
    try {
        auto texturePath = const_cast<std::string&>(texturePath_);
        // Check if texture already exists
        auto it = textureResources.find(texturePath);
        if (it != textureResources.end()) {
            // Texture already loaded and cached; leave cache intact and return success
            return true;
        }

        // Check if this is a KTX2 file
        bool isKtx2 = texturePath.find(".ktx2") != std::string::npos;

        // If it's a KTX2 texture but the path doesn't exist, try common fallback filename variants
        if (isKtx2) {
            std::filesystem::path origPath(texturePath);
            if (!std::filesystem::exists(origPath)) {
                std::string fname = origPath.filename().string();
                std::string dir = origPath.parent_path().string();
                auto tryCandidate = [&](const std::string& candidateName) -> bool {
                    std::filesystem::path cand = std::filesystem::path(dir) / candidateName;
                    if (std::filesystem::exists(cand)) {
                        std::cout << "Resolved missing texture '" << texturePath << "' to existing file '" << cand.string() << "'" << std::endl;
                        texturePath = cand.string();
                        return true;
                    }
                    return false;
                };
                // Known suffix variants near the end of filename before extension
                // Examples: *_c.ktx2, *_d.ktx2, *_cm.ktx2, *_diffuse.ktx2, *_basecolor.ktx2, *_albedo.ktx2
                std::vector<std::string> suffixes = {"_c", "_d", "_cm", "_diffuse", "_basecolor", "_albedo"};
                // If filename matches one known suffix, try others
                for (const auto& s : suffixes) {
                    std::string key = s + ".ktx2";
                    if (fname.size() > key.size() && fname.rfind(key) == fname.size() - key.size()) {
                        std::string prefix = fname.substr(0, fname.size() - key.size());
                        for (const auto& alt : suffixes) {
                            if (alt == s) continue;
                            std::string candName = prefix + alt + ".ktx2";
                            if (tryCandidate(candName)) { isKtx2 = true; break; }
                        }
                        break; // Only replace last suffix occurrence
                    }
                }
            }
        }

        int texWidth, texHeight, texChannels;
        unsigned char* pixels = nullptr;
        ktxTexture2* ktxTex = nullptr;
        vk::DeviceSize imageSize;

        // Track KTX2 transcoding state and original format across the function scope
        bool wasTranscoded = false;
        VkFormat ktxHeaderVkFormat = VK_FORMAT_UNDEFINED;

        uint32_t mipLevels = 1;
        std::vector<vk::BufferImageCopy> copyRegions;

        if (isKtx2) {
            // Load KTX2 file
            KTX_error_code result = ktxTexture2_CreateFromNamedFile(texturePath.c_str(),
                                                                   KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
                                                                   &ktxTex);
            if (result != KTX_SUCCESS) {
                // Retry with sibling suffix variants if file exists but cannot be parsed/opened
                std::filesystem::path origPath(texturePath);
                std::string fname = origPath.filename().string();
                std::string dir = origPath.parent_path().string();
                auto tryLoad = [&](const std::string& candidateName) -> bool {
                    std::filesystem::path cand = std::filesystem::path(dir) / candidateName;
                    if (std::filesystem::exists(cand)) {
                        std::string candStr = cand.string();
                        std::cout << "Retrying KTX2 load with sibling candidate '" << candStr << "' for original '" << texturePath << "'" << std::endl;
                        result = ktxTexture2_CreateFromNamedFile(candStr.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTex);
                        if (result == KTX_SUCCESS) {
                            texturePath = candStr; // Use the successfully opened candidate
                            return true;
                        }
                    }
                    return false;
                };
                // Known suffix variants near the end of filename before extension
                std::vector<std::string> suffixes = {"_c", "_d", "_cm", "_diffuse", "_basecolor", "_albedo"};
                for (const auto& s : suffixes) {
                    std::string key = s + ".ktx2";
                    if (fname.size() > key.size() && fname.rfind(key) == fname.size() - key.size()) {
                        std::string prefix = fname.substr(0, fname.size() - key.size());
                        bool loaded = false;
                        for (const auto& alt : suffixes) {
                            if (alt == s) continue;
                            std::string candName = prefix + alt + ".ktx2";
                            if (tryLoad(candName)) { loaded = true; break; }
                        }
                        if (loaded) break;
                    }
                }
                assert (result != KTX_SUCCESS);
            }

            // Cache header-provided VkFormat
            ktxHeaderVkFormat = static_cast<VkFormat>(ktxTex->vkFormat);

            // Check if texture needs transcoding (Basis Universal compressed)
            wasTranscoded = ktxTexture2_NeedsTranscoding(ktxTex);
            if (wasTranscoded) {
                // Transcode to RGBA8 uncompressed format for Vulkan compatibility
                ktx_transcode_fmt_e transcodeFormat = KTX_TTF_RGBA32;

                result = ktxTexture2_TranscodeBasis(ktxTex, transcodeFormat, 0);
                if (result != KTX_SUCCESS) {
                    std::cerr << "Failed to transcode KTX2 texture: " << texturePath << " (error: " << result << ")" << std::endl;
                    ktxTexture_Destroy((ktxTexture*)ktxTex);
                    return false;
                }
            }

            texWidth = ktxTex->baseWidth;
            texHeight = ktxTex->baseHeight;
            texChannels = 4; // KTX2 textures are typically RGBA
            // Disable mipmapping for now - memory pool only supports single mip level
            // TODO: Implement proper mipmap support in memory pool
            mipLevels = 1;

            // Calculate size for base level only
            if (wasTranscoded) {
                imageSize = texWidth * texHeight * 4; // RGBA = 4 bytes per pixel
            } else {
                imageSize = ktxTexture_GetImageSize((ktxTexture*)ktxTex, 0); // Only level 0
            }

            // Create single copy region for base level only
            copyRegions.push_back({
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
                .imageExtent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1}
            });
        } else {
            // Non-KTX texture loading via file path is disabled to simplify pipeline.
            std::cerr << "Unsupported non-KTX2 texture path: " << texturePath << std::endl;
            return false;
        }

        // Create staging buffer
        auto [stagingBuffer, stagingBufferMemory] = createBuffer(
            imageSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        // Copy pixel data to staging buffer
        void* data = stagingBufferMemory.mapMemory(0, imageSize);

        if (isKtx2) {
            // Copy KTX2 texture data for base level only (level 0)
            size_t levelSize;
            const void* levelData;

            if (ktxTexture2_NeedsTranscoding(ktxTex)) {
                // For transcoded textures, get data from the transcoded buffer
                levelSize = texWidth * texHeight * 4; // RGBA = 4 bytes per pixel
                ktx_size_t offset;
                ktxTexture_GetImageOffset((ktxTexture*)ktxTex, 0, 0, 0, &offset);
                levelData = ktxTexture_GetData((ktxTexture*)ktxTex) + offset;
            } else {
                // For non-transcoded textures, get data directly
                levelSize = ktxTexture_GetImageSize((ktxTexture*)ktxTex, 0);
                ktx_size_t offset;
                ktxTexture_GetImageOffset((ktxTexture*)ktxTex, 0, 0, 0, &offset);
                levelData = ktxTexture_GetData((ktxTexture*)ktxTex) + offset;
            }

            memcpy(data, levelData, levelSize);
        } else {
            // Copy regular image data
            memcpy(data, pixels, static_cast<size_t>(imageSize));
        }

        stagingBufferMemory.unmapMemory();

        // Free pixel data
        if (isKtx2) {
            ktxTexture_Destroy((ktxTexture*)ktxTex);
        } else {
            // no-op: non-KTX path disabled
        }

        // Determine appropriate texture format based on texture type and KTX2 metadata
        vk::Format textureFormat;
        if (isKtx2) {
            if (wasTranscoded) {
                // For transcoded Basis to RGBA32, choose by heuristic (sRGB for baseColor/albedo/diffuse)
                textureFormat = Renderer::determineTextureFormat(texturePath);
            } else {
                // Use the VkFormat provided by the KTX2 container if available (from header)
                VkFormat vkfmt = ktxHeaderVkFormat;
                if (vkfmt == VK_FORMAT_UNDEFINED) {
                    textureFormat = Renderer::determineTextureFormat(texturePath);
                } else {
                    textureFormat = static_cast<vk::Format>(vkfmt);
                }
            }
        } else {
            textureFormat = Renderer::determineTextureFormat(texturePath);
        }

        // Create texture image using memory pool
        auto [textureImg, textureImgAllocation] = createImagePooled(
            texWidth,
            texHeight,
            textureFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        resources.textureImage = std::move(textureImg);
        resources.textureImageAllocation = std::move(textureImgAllocation);

        // Transition image layout for copy
        transitionImageLayout(
            *resources.textureImage,
            textureFormat,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            mipLevels
        );

        // Copy buffer to image
        copyBufferToImage(
            *stagingBuffer,
            *resources.textureImage,
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight),
            copyRegions
        );

        // Transition image layout for shader access
        transitionImageLayout(
            *resources.textureImage,
            textureFormat,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            mipLevels
        );

        // Store the format and mipLevels for createTextureImageView
        resources.format = textureFormat;
        resources.mipLevels = mipLevels;

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
            resources.format, // Use the stored format instead of hardcoded sRGB
            vk::ImageAspectFlagBits::eColor,
            resources.mipLevels // Use the stored mipLevels
        );
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create texture image view: " << e.what() << std::endl;
        return false;
    }
}

// Create shared default PBR textures (to avoid creating hundreds of identical textures)
bool Renderer::createSharedDefaultPBRTextures() {
    try {
        unsigned char translucentPixel[4] = {128, 128, 128, 125}; // 50% alpha
        if (!LoadTextureFromMemory(SHARED_DEFAULT_ALBEDO_ID, translucentPixel, 1, 1, 4)) {
            std::cerr << "Failed to create shared default albedo texture" << std::endl;
            return false;
        }

        // Create shared default normal texture (flat normal)
        unsigned char normalPixel[4] = {128, 128, 255, 255}; // (0.5, 0.5, 1.0, 1.0) in 0-255 range
        if (!LoadTextureFromMemory(SHARED_DEFAULT_NORMAL_ID, normalPixel, 1, 1, 4)) {
            std::cerr << "Failed to create shared default normal texture" << std::endl;
            return false;
        }

        // Create shared default metallic-roughness texture (non-metallic, fully rough)
        unsigned char metallicRoughnessPixel[4] = {0, 255, 0, 255}; // (unused, roughness=1.0, metallic=0.0, alpha=1.0)
        if (!LoadTextureFromMemory(SHARED_DEFAULT_METALLIC_ROUGHNESS_ID, metallicRoughnessPixel, 1, 1, 4)) {
            std::cerr << "Failed to create shared default metallic-roughness texture" << std::endl;
            return false;
        }

        // Create shared default occlusion texture (white - no occlusion)
        unsigned char occlusionPixel[4] = {255, 255, 255, 255};
        if (!LoadTextureFromMemory(SHARED_DEFAULT_OCCLUSION_ID, occlusionPixel, 1, 1, 4)) {
            std::cerr << "Failed to create shared default occlusion texture" << std::endl;
            return false;
        }

        // Create shared default emissive texture (black - no emission)
        unsigned char emissivePixel[4] = {0, 0, 0, 255};
        if (!LoadTextureFromMemory(SHARED_DEFAULT_EMISSIVE_ID, emissivePixel, 1, 1, 4)) {
            std::cerr << "Failed to create shared default emissive texture" << std::endl;
            return false;
        }

        // Create shared bright red texture for ball visibility
        unsigned char brightRedPixel[4] = {255, 0, 0, 255}; // Bright red (R=255, G=0, B=0, A=255)
        if (!LoadTextureFromMemory(SHARED_BRIGHT_RED_ID, brightRedPixel, 1, 1, 4)) {
            std::cerr << "Failed to create shared bright red texture" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create shared default PBR textures: " << e.what() << std::endl;
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

        // Create texture image using memory pool
        auto [textureImg, textureImgAllocation] = createImagePooled(
            width,
            height,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        defaultTextureResources.textureImage = std::move(textureImg);
        defaultTextureResources.textureImageAllocation = std::move(textureImgAllocation);

        // Transition image layout for copy
        transitionImageLayout(
            *defaultTextureResources.textureImage,
            vk::Format::eR8G8B8A8Srgb,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        // Copy buffer to image
        std::vector<vk::BufferImageCopy> regions = {{
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
        }};
        copyBufferToImage(
            *stagingBuffer,
            *defaultTextureResources.textureImage,
            width,
            height,
            regions
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

        // Create sampler (mipmapping disabled)
        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eNearest, // Disable mipmap filtering
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = std::min(properties.limits.maxSamplerAnisotropy, 8.0f),
            .compareEnable = VK_FALSE,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f, // Force single mip level (no mipmapping)
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

// Load texture from file (public wrapper for createTextureImage)
bool Renderer::LoadTexture(const std::string& texturePath) {
    if (texturePath.empty()) {
        std::cerr << "LoadTexture: Empty texture path provided" << std::endl;
        return false;
    }

    // Check if texture is already loaded
    auto it = textureResources.find(texturePath);
    if (it != textureResources.end()) {
        // Texture already loaded
        return true;
    }

    // Create temporary texture resources (unused output; cache will be populated internally)
    TextureResources tempResources;

    // Use existing createTextureImage method (it inserts into textureResources on success)
    bool success = createTextureImage(texturePath, tempResources);

    if (!success) {
        std::cerr << "Failed to load texture: " << texturePath << std::endl;
    }

    return success;
}

// Determine appropriate texture format based on texture type
vk::Format Renderer::determineTextureFormat(const std::string& textureId) {
    // Determine sRGB vs Linear in a case-insensitive way
    std::string idLower = textureId;
    std::transform(idLower.begin(), idLower.end(), idLower.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    // BaseColor/Albedo/Diffuse textures should be in sRGB space for proper gamma correction
    if (idLower.find("basecolor") != std::string::npos ||
        idLower.find("base_color") != std::string::npos ||
        idLower.find("albedo") != std::string::npos ||
        idLower.find("diffuse") != std::string::npos ||
        textureId == Renderer::SHARED_DEFAULT_ALBEDO_ID) {
        return vk::Format::eR8G8B8A8Srgb;
    }

    // All other PBR textures (normal, metallic-roughness, occlusion, emissive) should be linear
    // because they contain non-color data that shouldn't be gamma corrected
    return vk::Format::eR8G8B8A8Unorm;
}

// Load texture from raw image data in memory
bool Renderer::LoadTextureFromMemory(const std::string& textureId, const unsigned char* imageData,
                                    int width, int height, int channels) {
    if (textureId.empty() || !imageData || width <= 0 || height <= 0 || channels <= 0) {
        std::cerr << "LoadTextureFromMemory: Invalid parameters" << std::endl;
        return false;
    }

    // Check if texture is already loaded
    auto it = textureResources.find(textureId);
    if (it != textureResources.end()) {
        // Texture already loaded
        return true;
    }

    try {
        TextureResources resources;

        // Calculate image size (ensure 4 channels for RGBA)
        int targetChannels = 4; // Always use RGBA for consistency
        vk::DeviceSize imageSize = width * height * targetChannels;

        // Create a staging buffer
        auto [stagingBuffer, stagingBufferMemory] = createBuffer(
            imageSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        // Copy and convert pixel data to staging buffer
        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        auto* stagingData = static_cast<unsigned char*>(data);

        if (channels == 4) {
            // Already RGBA, direct copy
            memcpy(stagingData, imageData, imageSize);
        } else if (channels == 3) {
            // RGB to RGBA conversion
            for (int i = 0; i < width * height; ++i) {
                stagingData[i * 4 + 0] = imageData[i * 3 + 0]; // R
                stagingData[i * 4 + 1] = imageData[i * 3 + 1]; // G
                stagingData[i * 4 + 2] = imageData[i * 3 + 2]; // B
                stagingData[i * 4 + 3] = 255; // A
            }
        } else if (channels == 2) {
            // Grayscale + Alpha to RGBA conversion
            for (int i = 0; i < width * height; ++i) {
                stagingData[i * 4 + 0] = imageData[i * 2 + 0]; // R (grayscale)
                stagingData[i * 4 + 1] = imageData[i * 2 + 0]; // G (grayscale)
                stagingData[i * 4 + 2] = imageData[i * 2 + 0]; // B (grayscale)
                stagingData[i * 4 + 3] = imageData[i * 2 + 1]; // A (alpha)
            }
        } else if (channels == 1) {
            // Grayscale to RGBA conversion
            for (int i = 0; i < width * height; ++i) {
                stagingData[i * 4 + 0] = imageData[i]; // R
                stagingData[i * 4 + 1] = imageData[i]; // G
                stagingData[i * 4 + 2] = imageData[i]; // B
                stagingData[i * 4 + 3] = 255; // A
            }
        } else {
            std::cerr << "LoadTextureFromMemory: Unsupported channel count: " << channels << std::endl;
            stagingBufferMemory.unmapMemory();
            return false;
        }

        stagingBufferMemory.unmapMemory();

        // Determine the appropriate texture format based on the texture type
        vk::Format textureFormat = determineTextureFormat(textureId);
        // Create texture image using memory pool
        auto [textureImg, textureImgAllocation] = createImagePooled(
            width,
            height,
            textureFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        resources.textureImage = std::move(textureImg);
        resources.textureImageAllocation = std::move(textureImgAllocation);

        // Transition image layout for copy
        transitionImageLayout(
            *resources.textureImage,
            textureFormat,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        // Copy buffer to image
        std::vector<vk::BufferImageCopy> regions = {{
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
            .imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1}
        }};
        copyBufferToImage(
            *stagingBuffer,
            *resources.textureImage,
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            regions
        );

        // Transition image layout for shader access
        transitionImageLayout(
            *resources.textureImage,
            textureFormat,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        // Store the format for createTextureImageView
        resources.format = textureFormat;

        // Create texture image view
        resources.textureImageView = createImageView(
            resources.textureImage,
            textureFormat,
            vk::ImageAspectFlagBits::eColor
        );

        // Create texture sampler
        if (!createTextureSampler(resources)) {
            return false;
        }

        // Add to texture resources map
        textureResources[textureId] = std::move(resources);

        std::cout << "Successfully loaded texture from memory: " << textureId
                  << " (" << width << "x" << height << ", " << channels << " channels)" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load texture from memory: " << e.what() << std::endl;
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

        // Create vertex buffer on device using memory pool
        auto [vertexBuffer, vertexBufferAllocation] = createBufferPooled(
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

        // Create index buffer on device using memory pool
        auto [indexBuffer, indexBufferAllocation] = createBufferPooled(
            indexBufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        // Copy from staging buffer to device buffer
        copyBuffer(stagingIndexBuffer, indexBuffer, indexBufferSize);

        // Create mesh resources
        MeshResources resources;
        resources.vertexBuffer = std::move(vertexBuffer);
        resources.vertexBufferAllocation = std::move(vertexBufferAllocation);
        resources.indexBuffer = std::move(indexBuffer);
        resources.indexBufferAllocation = std::move(indexBufferAllocation);
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

        // Create uniform buffers using memory pool
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            auto [buffer, bufferAllocation] = createBufferPooled(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            // Use the memory pool's mapped pointer if available
            void* mappedMemory = bufferAllocation->mappedPtr;
            if (!mappedMemory) {
                std::cerr << "Warning: Uniform buffer allocation is not mapped" << std::endl;
            }

            resources.uniformBuffers.emplace_back(std::move(buffer));
            resources.uniformBufferAllocations.emplace_back(std::move(bufferAllocation));
            resources.uniformBuffersMapped.emplace_back(mappedMemory);
        }

        // Create instance buffer for all entities (shaders always expect instance data)
        auto* meshComponent = entity->GetComponent<MeshComponent>();
        if (meshComponent) {
            std::vector<InstanceData> instanceData;

            // CRITICAL FIX: Check if entity has any instance data first
            if (meshComponent->GetInstanceCount() > 0) {
                // Use existing instance data from GLTF loading (whether 1 or many instances)
                instanceData = meshComponent->GetInstances();
            } else {
                // Create single instance data using IDENTITY matrix to avoid double-transform with UBO.model
                InstanceData singleInstance;
                singleInstance.setModelMatrix(glm::mat4(1.0f));
                instanceData = {singleInstance};
            }

            vk::DeviceSize instanceBufferSize = sizeof(InstanceData) * instanceData.size();

            auto [instanceBuffer, instanceBufferAllocation] = createBufferPooled(
                instanceBufferSize,
                vk::BufferUsageFlagBits::eVertexBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            // Copy instance data to buffer
            void* instanceMappedMemory = instanceBufferAllocation->mappedPtr;
            if (instanceMappedMemory) {
                std::memcpy(instanceMappedMemory, instanceData.data(), instanceBufferSize);
            } else {
                std::cerr << "Warning: Instance buffer allocation is not mapped" << std::endl;
            }

            resources.instanceBuffer = std::move(instanceBuffer);
            resources.instanceBufferAllocation = std::move(instanceBufferAllocation);
            resources.instanceBufferMapped = instanceMappedMemory;
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
        // Calculate pool sizes for all Bistro materials plus additional entities
        // The Bistro model creates many more entities than initially expected
        // Each entity needs descriptor sets for both basic and PBR pipelines
        // PBR pipeline needs 7 descriptors per set (1 UBO + 5 PBR textures + 1 shadow map array with 16 shadow maps)
        // Basic pipeline needs 2 descriptors per set (1 UBO + 1 texture)
        const uint32_t maxEntities = 20000; // Increased to 20k entities to handle large scenes like Bistro reliably
        const uint32_t maxDescriptorSets = MAX_FRAMES_IN_FLIGHT * maxEntities * 2; // 2 pipeline types per entity

        // Calculate descriptor counts
        // UBO descriptors: 1 per descriptor set
        const uint32_t uboDescriptors = maxDescriptorSets;
        // Texture descriptors: Basic pipeline uses 1, PBR uses 21 (5 PBR textures + 16 shadow maps)
        // Allocate for worst case: all entities using PBR (21 texture descriptors each)
        const uint32_t textureDescriptors = MAX_FRAMES_IN_FLIGHT * maxEntities * 21;
        // Storage buffer descriptors: PBR pipeline uses 1 light storage buffer per descriptor set
        // Only PBR entities need storage buffers, so allocate for all entities using PBR
        const uint32_t storageBufferDescriptors = MAX_FRAMES_IN_FLIGHT * maxEntities;

        std::array<vk::DescriptorPoolSize, 3> poolSizes = {
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = uboDescriptors
            },
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = textureDescriptors
            },
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = storageBufferDescriptors
            }
        };

        // Create descriptor pool
        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = maxDescriptorSets,
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
bool Renderer::createDescriptorSets(Entity* entity, const std::string& texturePath, bool usePBR) {
    try {
        // Get entity resources
        auto entityIt = entityResources.find(entity);
        if (entityIt == entityResources.end()) {
            std::cerr << "Entity resources not found" << std::endl;
            return false;
        }

        // Get texture resources - use default texture as fallback if specific texture fails
        TextureResources* textureRes = nullptr;
        if (!texturePath.empty()) {
            auto textureIt = textureResources.find(texturePath);
            if (textureIt == textureResources.end()) {
                // If this is a GLTF embedded texture ID, don't try to load from disk
                if (texturePath.rfind("gltf_", 0) == 0) {
                    // Handle both gltf_baseColor_{i} and gltf_basecolor_{i}
                    const std::string prefixUpper = "gltf_baseColor_";
                    const std::string prefixLower = "gltf_basecolor_";
                    if (texturePath.rfind(prefixUpper, 0) == 0 || texturePath.rfind(prefixLower, 0) == 0) {
                        const bool isUpper = texturePath.rfind(prefixUpper, 0) == 0;
                        std::string index = texturePath.substr((isUpper ? prefixUpper.size() : prefixLower.size()));
                        // Try direct baseColor id first
                        std::string baseColorId = "gltf_baseColor_" + index;
                        auto bcIt = textureResources.find(baseColorId);
                        if (bcIt != textureResources.end()) {
                            textureRes = &bcIt->second;
                        } else {
                            // Try alias to generic gltf_texture_{index}
                            std::string alias = "gltf_texture_" + index;
                            auto aliasIt = textureResources.find(alias);
                            if (aliasIt != textureResources.end()) {
                                textureRes = &aliasIt->second;
                            } else {
                                std::cerr << "Warning: Embedded texture not found: " << texturePath
                                          << " (also missing alias: " << alias << ") using default." << std::endl;
                                textureRes = &defaultTextureResources;
                            }
                        }
                    } else {
                        std::cerr << "Warning: Embedded texture not found: " << texturePath << ", using default." << std::endl;
                        textureRes = &defaultTextureResources;
                    }
                } else {
                    std::cerr << "Warning: On-demand texture loading disabled for " << texturePath
                              << "; using default texture instead" << std::endl;
                    textureRes = &defaultTextureResources;
                }
            } else {
                textureRes = &textureIt->second;
            }
        } else {
            // No texture path specified, use default texture
            textureRes = &defaultTextureResources;
        }

        // Create descriptor sets using RAII - choose layout based on pipeline type
        vk::DescriptorSetLayout selectedLayout = usePBR ? *pbrDescriptorSetLayout : *descriptorSetLayout;
        std::vector layouts(MAX_FRAMES_IN_FLIGHT, selectedLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts = layouts.data()
        };

        // Choose the appropriate descriptor set vector based on pipeline type
        auto& targetDescriptorSets = usePBR ? entityIt->second.pbrDescriptorSets : entityIt->second.basicDescriptorSets;

        // Only create descriptor sets if they don't already exist for this pipeline type
        if (targetDescriptorSets.empty()) {
            try {
                // Allocate descriptor sets using RAII wrapper
                vk::raii::DescriptorSets raiiDescriptorSets(device, allocInfo);

                // Convert to vector of individual RAII descriptor sets
                targetDescriptorSets.clear();
                targetDescriptorSets.reserve(MAX_FRAMES_IN_FLIGHT);
                for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                    targetDescriptorSets.emplace_back(std::move(raiiDescriptorSets[i]));
                }
            } catch (const std::exception& e) {
                std::cerr << "Failed to allocate descriptor sets for entity " << entity->GetName()
                          << " (pipeline: " << (usePBR ? "PBR" : "basic") << "): " << e.what() << std::endl;
                return false;
            }
        }

        // Validate descriptor sets before using them
        if (targetDescriptorSets.size() != MAX_FRAMES_IN_FLIGHT) {
            std::cerr << "Invalid descriptor set count for entity " << entity->GetName()
                      << " (expected: " << MAX_FRAMES_IN_FLIGHT << ", got: " << targetDescriptorSets.size() << ")" << std::endl;
            return false;
        }

        // Validate default texture resources before using them
        if (!*defaultTextureResources.textureSampler || !*defaultTextureResources.textureImageView) {
            std::cerr << "Invalid default texture resources for entity " << entity->GetName() << std::endl;
            return false;
        }

        // Update descriptor sets
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // Validate descriptor set handle before using it
            if (!*targetDescriptorSets[i]) {
                std::cerr << "Invalid descriptor set handle for entity " << entity->GetName()
                          << " at frame " << i << " (pipeline: " << (usePBR ? "PBR" : "basic") << ")" << std::endl;
                return false;
            }

            // Validate uniform buffer before creating descriptor
            if (i >= entityIt->second.uniformBuffers.size() ||
                !*entityIt->second.uniformBuffers[i]) {
                std::cerr << "Invalid uniform buffer for entity " << entity->GetName()
                          << " at frame " << i << std::endl;
                return false;
            }

            // Uniform buffer descriptor
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = *entityIt->second.uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };

            if (usePBR) {
                // PBR pipeline: Create 7 descriptor writes (UBO + 5 textures + light storage buffer)
                std::array<vk::WriteDescriptorSet, 7> descriptorWrites;
                std::array<vk::DescriptorImageInfo, 5> imageInfos;

                // Uniform buffer descriptor writes (binding 0)
                descriptorWrites[0] = vk::WriteDescriptorSet{
                    .dstSet = targetDescriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo
                };

                // Get all PBR texture paths from the entity's MeshComponent
                auto meshComponent = entity->GetComponent<MeshComponent>();
                // Resolve baseColor path with multiple fallbacks: GLTF baseColor -> legacy texturePath -> material DB -> shared default
                std::string resolvedBaseColor;
                if (meshComponent && !meshComponent->GetBaseColorTexturePath().empty()) {
                    resolvedBaseColor = meshComponent->GetBaseColorTexturePath();
                } else if (meshComponent && !meshComponent->GetTexturePath().empty()) {
                    resolvedBaseColor = meshComponent->GetTexturePath();
                } else {
                    // Try to use material name from entity name to query ModelLoader
                    std::string entityName = entity->GetName();
                    size_t tagPos = entityName.find("_Material_");
                    if (tagPos != std::string::npos) {
                        size_t afterTag = tagPos + std::string("_Material_").size();
                        // Expect format: <model>_Material_<index>_<materialName>
                        size_t sep = entityName.find('_', afterTag);
                        if (sep != std::string::npos && sep + 1 < entityName.length()) {
                            std::string materialName = entityName.substr(sep + 1);
                            if (modelLoader) {
                                Material* mat = modelLoader->GetMaterial(materialName);
                                if (mat && !mat->albedoTexturePath.empty()) {
                                    resolvedBaseColor = mat->albedoTexturePath;
                                }
                            }
                        }
                    }
                    if (resolvedBaseColor.empty()) {
                        resolvedBaseColor = SHARED_DEFAULT_ALBEDO_ID;
                    }
                }

                // Heuristic: if still default and we have an external normal map like *_ddna.ktx2, try to guess base color sibling
                if (resolvedBaseColor == SHARED_DEFAULT_ALBEDO_ID && meshComponent) {
                    std::string normalPath = meshComponent->GetNormalTexturePath();
                    if (!normalPath.empty() && normalPath.rfind("gltf_", 0) != 0) {
                        // Make a lowercase copy for pattern checks
                        std::string normalLower = normalPath;
                        std::transform(normalLower.begin(), normalLower.end(), normalLower.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
                        if (normalLower.find("_ddna") != std::string::npos) {
                            // Try replacing _ddna with common diffuse/basecolor suffixes
                            std::vector<std::string> suffixes = {"_d", "_c", "_cm", "_diffuse", "_basecolor"};
                            for (const auto& suf : suffixes) {
                                std::string candidate = normalPath;
                                // Replace only the first occurrence of _ddna
                                size_t pos = normalLower.find("_ddna");
                                if (pos != std::string::npos) {
                                    candidate.replace(pos, 5, suf);
                                    // Attempt to load; if successful, use this as resolved base color
                                    // On-demand loading disabled; skip attempting to load candidate
                                    (void)candidate; // suppress unused
                                    break;
                                }
                            }
                        }
                    }
                }

                std::vector pbrTexturePaths = {
                    // Binding 1: baseColor
                    resolvedBaseColor,
                    // Binding 2: metallicRoughness - use GLTF texture or fallback to shared default
                    (meshComponent && !meshComponent->GetMetallicRoughnessTexturePath().empty()) ?
                        meshComponent->GetMetallicRoughnessTexturePath() : SHARED_DEFAULT_METALLIC_ROUGHNESS_ID,
                    // Binding 3: normal - use GLTF texture or fallback to shared default
                    (meshComponent && !meshComponent->GetNormalTexturePath().empty()) ?
                        meshComponent->GetNormalTexturePath() : SHARED_DEFAULT_NORMAL_ID,
                    // Binding 4: occlusion - use GLTF texture or fallback to shared default
                    (meshComponent && !meshComponent->GetOcclusionTexturePath().empty()) ?
                        meshComponent->GetOcclusionTexturePath() : SHARED_DEFAULT_OCCLUSION_ID,
                    // Binding 5: emissive - use GLTF texture or fallback to shared default
                    (meshComponent && !meshComponent->GetEmissiveTexturePath().empty()) ?
                        meshComponent->GetEmissiveTexturePath() : SHARED_DEFAULT_EMISSIVE_ID
                };

                // Create image infos for each PBR texture binding
                for (int j = 0; j < 5; j++) {
                    const std::string& currentTexturePath = pbrTexturePaths[j];

                    // Find the texture resources for this binding
                    auto textureIt = textureResources.find(currentTexturePath);
                    if (textureIt != textureResources.end()) {
                        // Use the specific texture for this binding
                        const auto& texRes = textureIt->second;

                        // Validate texture resources before using them (check if RAII objects are valid)
                        if (*texRes.textureSampler == VK_NULL_HANDLE || *texRes.textureImageView == VK_NULL_HANDLE) {
                            std::cerr << "Invalid texture resources for " << currentTexturePath
                                      << " in entity " << entity->GetName() << ", using default texture" << std::endl;
                            // Fall back to default texture
                            imageInfos[j] = vk::DescriptorImageInfo{
                                .sampler = *defaultTextureResources.textureSampler,
                                .imageView = *defaultTextureResources.textureImageView,
                                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                            };
                        } else {
                            imageInfos[j] = vk::DescriptorImageInfo{
                                .sampler = *texRes.textureSampler,
                                .imageView = *texRes.textureImageView,
                                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                            };
                        }
                    } else {
                        // On-demand texture loading disabled; use alias or defaults below
                        // Try alias for embedded baseColor textures: gltf_baseColor_{i} -> gltf_texture_{i}
                        if (currentTexturePath.rfind("gltf_baseColor_", 0) == 0 ||
                            currentTexturePath.rfind("gltf_basecolor_", 0) == 0) {
                            std::string prefix = (currentTexturePath.rfind("gltf_baseColor_", 0) == 0)
                                ? std::string("gltf_baseColor_")
                                : std::string("gltf_basecolor_");
                            std::string index = currentTexturePath.substr(prefix.size());
                            std::string alias = "gltf_texture_" + index;
                            auto aliasIt = textureResources.find(alias);
                            if (aliasIt != textureResources.end()) {
                                const auto& texRes = aliasIt->second;
                                imageInfos[j] = vk::DescriptorImageInfo{
                                    .sampler = *texRes.textureSampler,
                                    .imageView = *texRes.textureImageView,
                                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                                };
                            } else {
                                // Fall back to default white texture if the specific texture is not found
                                imageInfos[j] = vk::DescriptorImageInfo{
                                    .sampler = *defaultTextureResources.textureSampler,
                                    .imageView = *defaultTextureResources.textureImageView,
                                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                                };
                            }
                        } else {
                            // Fall back to default white texture if the specific texture is not found
                            imageInfos[j] = vk::DescriptorImageInfo{
                                .sampler = *defaultTextureResources.textureSampler,
                                .imageView = *defaultTextureResources.textureImageView,
                                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                            };
                        }
                    }
                    descriptor_path_resolved: ;
                }

                // Create descriptor writes for all 5 texture bindings
                for (int binding = 1; binding <= 5; binding++) {
                    descriptorWrites[binding] = vk::WriteDescriptorSet{
                        .dstSet = targetDescriptorSets[i],
                        .dstBinding = static_cast<uint32_t>(binding),
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                        .pImageInfo = &imageInfos[binding - 1]
                    };
                }

                // No shadow maps: binding 6 is now the light storage buffer

                // Create descriptor write for light storage buffer (binding 6)
                // Check if light storage buffers are initialized
                if (i < lightStorageBuffers.size() && *lightStorageBuffers[i].buffer) {
                    vk::DescriptorBufferInfo lightBufferInfo{
                        .buffer = *lightStorageBuffers[i].buffer,
                        .offset = 0,
                        .range = VK_WHOLE_SIZE
                    };

                    descriptorWrites[6] = vk::WriteDescriptorSet{
                        .dstSet = targetDescriptorSets[i],
                        .dstBinding = 6,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &lightBufferInfo
                    };
                } else {
                    // Ensure light storage buffers are initialized before creating descriptor sets
                    // Initialize with at least 1 light to create the buffers
                    if (!createOrResizeLightStorageBuffers(1)) {
                        std::cerr << "Failed to initialize light storage buffers for descriptor set creation" << std::endl;
                        return false;
                    }

                    // Now use the properly initialized light storage buffer
                    vk::DescriptorBufferInfo lightBufferInfo{
                        .buffer = *lightStorageBuffers[i].buffer,
                        .offset = 0,
                        .range = VK_WHOLE_SIZE
                    };

                    descriptorWrites[6] = vk::WriteDescriptorSet{
                        .dstSet = targetDescriptorSets[i],
                        .dstBinding = 6,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .pBufferInfo = &lightBufferInfo
                    };
                }

                // Update descriptor sets with all 7 descriptors
                device.updateDescriptorSets(descriptorWrites, {});
            } else {
                // Basic pipeline: Create 2 descriptor writes (UBO + 1 texture)
                std::array<vk::WriteDescriptorSet, 2> descriptorWrites;

                // Uniform buffer descriptor write
                descriptorWrites[0] = vk::WriteDescriptorSet{
                    .dstSet = targetDescriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo
                };

                // Check if texture resources are valid
                bool hasValidTexture = !texturePath.empty() && textureRes &&
                                      *textureRes->textureSampler &&
                                      *textureRes->textureImageView;

                // Texture sampler descriptor
                vk::DescriptorImageInfo imageInfo;
                if (hasValidTexture) {
                    // Use provided texture resources
                    imageInfo = vk::DescriptorImageInfo{
                        .sampler = *textureRes->textureSampler,
                        .imageView = *textureRes->textureImageView,
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
                    .dstSet = targetDescriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &imageInfo
                };

                // Update descriptor sets with both descriptors
                device.updateDescriptorSets(descriptorWrites, {});
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create descriptor sets: " << e.what() << std::endl;
        return false;
    }
}

// Pre-allocate all Vulkan resources for an entity during scene loading
bool Renderer::preAllocateEntityResources(Entity* entity) {
    try {
        // Get the mesh component
        auto meshComponent = entity->GetComponent<MeshComponent>();
        if (!meshComponent) {
            std::cerr << "Entity " << entity->GetName() << " has no mesh component" << std::endl;
            return false;
        }

        // 1. Create mesh resources (vertex/index buffers)
        if (!createMeshResources(meshComponent)) {
            std::cerr << "Failed to create mesh resources for entity: " << entity->GetName() << std::endl;
            return false;
        }

        // 2. Create uniform buffers
        if (!createUniformBuffers(entity)) {
            std::cerr << "Failed to create uniform buffers for entity: " << entity->GetName() << std::endl;
            return false;
        }

        // 3. Pre-allocate BOTH basic and PBR descriptor sets
        std::string texturePath = meshComponent->GetTexturePath();
        // Fallback: if legacy texturePath is empty, use PBR baseColor texture
        if (texturePath.empty()) {
            const std::string& baseColor = meshComponent->GetBaseColorTexturePath();
            if (!baseColor.empty()) {
                texturePath = baseColor;
            }
        }

        // Create basic descriptor sets
        if (!createDescriptorSets(entity, texturePath, false)) {
            std::cerr << "Failed to create basic descriptor sets for entity: " << entity->GetName() << std::endl;
            return false;
        }

        // Create PBR descriptor sets
        if (!createDescriptorSets(entity, texturePath, true)) {
            std::cerr << "Failed to create PBR descriptor sets for entity: " << entity->GetName() << std::endl;
            return false;
        }
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to pre-allocate resources for entity " << entity->GetName() << ": " << e.what() << std::endl;
        return false;
    }
}

// Create buffer using memory pool for efficient allocation
std::pair<vk::raii::Buffer, std::unique_ptr<MemoryPool::Allocation>> Renderer::createBufferPooled(
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties) {
    try {
        if (!memoryPool) {
            throw std::runtime_error("Memory pool not initialized");
        }

        // Use memory pool for allocation
        auto [buffer, allocation] = memoryPool->createBuffer(size, usage, properties);

        return {std::move(buffer), std::move(allocation)};

    } catch (const std::exception& e) {
        std::cerr << "Failed to create buffer with memory pool: " << e.what() << std::endl;
        throw;
    }
}

// Legacy createBuffer function - now strictly enforces memory pool usage
std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> Renderer::createBuffer(
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties) {

    // This function should only be used for temporary staging buffers during resource creation
    // All persistent resources should use createBufferPooled directly

    if (!memoryPool) {
        throw std::runtime_error("Memory pool not available - cannot create buffer");
    }

    // Check if we're trying to allocate during rendering
    if (memoryPool->isRenderingActive()) {
        std::cerr << "ERROR: Attempted to create buffer during rendering! Size: " << size << " bytes" << std::endl;
        std::cerr << "This violates the constraint that no new memory should be allocated during rendering." << std::endl;
        throw std::runtime_error("Buffer creation attempted during rendering - this is not allowed");
    }

    // Only allow direct allocation for staging buffers (temporary, host-visible)
    if (!(properties & vk::MemoryPropertyFlagBits::eHostVisible)) {
        std::cerr << "ERROR: Legacy createBuffer should only be used for staging buffers!" << std::endl;
        throw std::runtime_error("Legacy createBuffer used for non-staging buffer");
    }

    try {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        vk::raii::Buffer buffer(device, bufferInfo);

        // Allocate memory directly for staging buffers only
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

        // Align allocation size to nonCoherentAtomSize (64 bytes) to prevent validation errors
        // VUID-VkMappedMemoryRange-size-01390 requires memory flush sizes to be multiples of nonCoherentAtomSize
        const vk::DeviceSize nonCoherentAtomSize = 64; // Typical value, should query from device properties
        vk::DeviceSize alignedSize = ((memRequirements.size + nonCoherentAtomSize - 1) / nonCoherentAtomSize) * nonCoherentAtomSize;

        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = alignedSize,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        vk::raii::DeviceMemory bufferMemory(device, allocInfo);

        // Bind memory to buffer
        buffer.bindMemory(*bufferMemory, 0);

        return {std::move(buffer), std::move(bufferMemory)};

    } catch (const std::exception& e) {
        std::cerr << "Failed to create staging buffer: " << e.what() << std::endl;
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

// Create image using memory pool for efficient allocation
std::pair<vk::raii::Image, std::unique_ptr<MemoryPool::Allocation>> Renderer::createImagePooled(
    uint32_t width,
    uint32_t height,
    vk::Format format,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    uint32_t mipLevels) {
    try {
        if (!memoryPool) {
            throw std::runtime_error("Memory pool not initialized");
        }

        // Use memory pool for allocation (mipmap support limited by memory pool API)
        auto [image, allocation] = memoryPool->createImage(width, height, format, tiling, usage, properties);
        std::cout << "Created image using memory pool: " << width << "x" << height << " format=" << static_cast<int>(format) << " mipLevels=" << mipLevels << std::endl;

        return {std::move(image), std::move(allocation)};

    } catch (const std::exception& e) {
        std::cerr << "Failed to create image with memory pool: " << e.what() << std::endl;
        throw;
    }
}

// Create an image view
vk::raii::ImageView Renderer::createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
    try {
        // Create image view
        vk::ImageViewCreateInfo viewInfo{
            .image = *image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspectFlags,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        return { device, viewInfo };
    } catch (const std::exception& e) {
        std::cerr << "Failed to create image view: " << e.what() << std::endl;
        throw;
    }
}

// Transition image layout
void Renderer::transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels) {
    try {
        // Create a command buffer using RAII
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

        // Create an image barrier
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
                .levelCount = mipLevels,
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
        } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilReadOnlyOptimal) {
            // Support for shadow map creation: transition from undefined to read-only depth layout
            barrier.srcAccessMask = vk::AccessFlagBits::eNone;
            barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else {
            throw std::invalid_argument("Unsupported layout transition!");
        }

        // Add a barrier to command buffer
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

        // Use mutex to ensure thread-safe access to the graphics queue
        {
            vk::SubmitInfo submitInfo{
                .commandBufferCount = 1,
                .pCommandBuffers = &*commandBuffer
            };
            std::lock_guard lock(queueMutex);
            graphicsQueue.submit(submitInfo, nullptr);
            graphicsQueue.waitIdle();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to transition image layout: " << e.what() << std::endl;
        throw;
    }
}

// Copy buffer to image
void Renderer::copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height, const std::vector<vk::BufferImageCopy>& regions) const {
    try {
        // Create a command buffer using RAII
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

        // Copy buffer to image using provided regions
        commandBuffer.copyBufferToImage(
            buffer,
            image,
            vk::ImageLayout::eTransferDstOptimal,
            regions
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

// Create or resize light storage buffers to accommodate the given number of lights
bool Renderer::createOrResizeLightStorageBuffers(size_t lightCount) {
    try {
        // Ensure we have storage buffers for each frame in flight
        if (lightStorageBuffers.size() != MAX_FRAMES_IN_FLIGHT) {
            lightStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        }

        // Check if we need to resize buffers
        bool needsResize = false;
        for (auto& buffer : lightStorageBuffers) {
            if (buffer.capacity < lightCount) {
                needsResize = true;
                break;
            }
        }

        if (!needsResize) {
            return true; // Buffers are already large enough
        }

        // Calculate new capacity (with some headroom for growth)
        size_t newCapacity = std::max(lightCount * 2, size_t(64));
        vk::DeviceSize bufferSize = sizeof(LightData) * newCapacity;

        // Wait for device to be idle before destroying old buffers to prevent validation errors
        // This ensures no GPU operations are using the old buffers
        device.waitIdle();

        // Create new buffers for each frame
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            auto& buffer = lightStorageBuffers[i];

            // Clean up old buffer if it exists (now safe after waitIdle)
            if (buffer.allocation) {
                buffer.buffer = nullptr;
                buffer.allocation.reset();
                buffer.mapped = nullptr;
            }

            // Create new storage buffer
            auto [newBuffer, newAllocation] = createBufferPooled(
                bufferSize,
                vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            // Get the mapped pointer from the allocation
            void* mapped = newAllocation->mappedPtr;

            // Store the new buffer
            buffer.buffer = std::move(newBuffer);
            buffer.allocation = std::move(newAllocation);
            buffer.mapped = mapped;
            buffer.capacity = newCapacity;
            buffer.size = 0;
        }

        // Update all existing descriptor sets to reference the new light storage buffers
        updateAllDescriptorSetsWithNewLightBuffers();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create or resize light storage buffers: " << e.what() << std::endl;
        return false;
    }
}

// Update all existing descriptor sets with new light storage buffer references
void Renderer::updateAllDescriptorSetsWithNewLightBuffers() {
    try {
        // Iterate through all entity resources and update their PBR descriptor sets
        for (auto& [entity, resources] : entityResources) {
            // Only update PBR descriptor sets (they have light buffer bindings)
            if (!resources.pbrDescriptorSets.empty()) {
                for (size_t i = 0; i < resources.pbrDescriptorSets.size() && i < lightStorageBuffers.size(); ++i) {
                    if (i < lightStorageBuffers.size() && *lightStorageBuffers[i].buffer) {
                        // Create descriptor write for light storage buffer (binding 7)
                        vk::DescriptorBufferInfo lightBufferInfo{
                            .buffer = *lightStorageBuffers[i].buffer,
                            .offset = 0,
                            .range = VK_WHOLE_SIZE
                        };

                        vk::WriteDescriptorSet descriptorWrite{
                            .dstSet = *resources.pbrDescriptorSets[i],
                            .dstBinding = 6,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = vk::DescriptorType::eStorageBuffer,
                            .pBufferInfo = &lightBufferInfo
                        };

                        // Update the descriptor set
                        device.updateDescriptorSets(descriptorWrite, {});
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to update descriptor sets with new light buffers: " << e.what() << std::endl;
    }
}

// Update the light storage buffer with current light data
bool Renderer::updateLightStorageBuffer(uint32_t frameIndex, const std::vector<ExtractedLight>& lights) {
    try {
        // Ensure buffers are large enough and properly initialized
        if (!createOrResizeLightStorageBuffers(lights.size())) {
            return false;
        }

        // Now check frame index after buffers are properly initialized
        if (frameIndex >= lightStorageBuffers.size()) {
            std::cerr << "Invalid frame index for light storage buffer update: " << frameIndex
                      << " >= " << lightStorageBuffers.size() << std::endl;
            return false;
        }

        auto& buffer = lightStorageBuffers[frameIndex];
        if (!buffer.mapped) {
            std::cerr << "Light storage buffer not mapped" << std::endl;
            return false;
        }

        // Convert ExtractedLight data to LightData format
        auto* lightData = static_cast<LightData*>(buffer.mapped);
        for (size_t i = 0; i < lights.size(); ++i) {
            const auto& light = lights[i];

            // For directional lights, store direction in position field (they don't need position)
            // For other lights, store position
            if (light.type == ExtractedLight::Type::Directional) {
                lightData[i].position = glm::vec4(light.direction, 0.0f); // w=0 indicates direction
            } else {
                lightData[i].position = glm::vec4(light.position, 1.0f); // w=1 indicates position
            }

            lightData[i].color = glm::vec4(light.color * light.intensity, 1.0f);

            // Calculate light space matrix for shadow mapping
            glm::mat4 lightProjection, lightView;
            if (light.type == ExtractedLight::Type::Directional) {
                float orthoSize = 50.0f;
                lightProjection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 0.1f, 100.0f);
                lightView = glm::lookAt(light.position, light.position + light.direction, glm::vec3(0.0f, 1.0f, 0.0f));
            } else {
                lightProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, light.range);
                lightView = glm::lookAt(light.position, light.position + light.direction, glm::vec3(0.0f, 1.0f, 0.0f));
            }
            lightData[i].lightSpaceMatrix = lightProjection * lightView;

            // Set light type
            switch (light.type) {
                case ExtractedLight::Type::Point:
                    lightData[i].lightType = 0;
                    break;
                case ExtractedLight::Type::Directional:
                    lightData[i].lightType = 1;
                    break;
                case ExtractedLight::Type::Spot:
                    lightData[i].lightType = 2;
                    break;
                case ExtractedLight::Type::Emissive:
                    lightData[i].lightType = 3;
                    break;
            }

            // Set other light properties
            lightData[i].range = light.range;
            lightData[i].innerConeAngle = light.innerConeAngle;
            lightData[i].outerConeAngle = light.outerConeAngle;
        }

        // Update buffer size
        buffer.size = lights.size();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to update light storage buffer: " << e.what() << std::endl;
        return false;
    }
}
