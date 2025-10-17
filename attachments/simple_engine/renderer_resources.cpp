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
#include <functional>

// stb_image dependency removed; all GLTF textures are uploaded via memory path from ModelLoader.

// KTX2 support
#include <ktx.h>

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

// Helper: coerce an sRGB/UNORM variant of a given VkFormat while preserving block type where possible
static vk::Format CoerceFormatSRGB(vk::Format fmt, bool wantSRGB) {
    switch (fmt) {
        case vk::Format::eR8G8B8A8Unorm: return wantSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
        case vk::Format::eR8G8B8A8Srgb:  return wantSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;

        case vk::Format::eBc1RgbUnormBlock:  return wantSRGB ? vk::Format::eBc1RgbSrgbBlock  : vk::Format::eBc1RgbUnormBlock;
        case vk::Format::eBc1RgbSrgbBlock:   return wantSRGB ? vk::Format::eBc1RgbSrgbBlock  : vk::Format::eBc1RgbUnormBlock;
        case vk::Format::eBc1RgbaUnormBlock: return wantSRGB ? vk::Format::eBc1RgbaSrgbBlock : vk::Format::eBc1RgbaUnormBlock;
        case vk::Format::eBc1RgbaSrgbBlock:  return wantSRGB ? vk::Format::eBc1RgbaSrgbBlock : vk::Format::eBc1RgbaUnormBlock;

        case vk::Format::eBc2UnormBlock: return wantSRGB ? vk::Format::eBc2SrgbBlock : vk::Format::eBc2UnormBlock;
        case vk::Format::eBc2SrgbBlock:  return wantSRGB ? vk::Format::eBc2SrgbBlock : vk::Format::eBc2UnormBlock;

        case vk::Format::eBc3UnormBlock: return wantSRGB ? vk::Format::eBc3SrgbBlock : vk::Format::eBc3UnormBlock;
        case vk::Format::eBc3SrgbBlock:  return wantSRGB ? vk::Format::eBc3SrgbBlock : vk::Format::eBc3UnormBlock;

        case vk::Format::eBc7UnormBlock: return wantSRGB ? vk::Format::eBc7SrgbBlock : vk::Format::eBc7UnormBlock;
        case vk::Format::eBc7SrgbBlock:  return wantSRGB ? vk::Format::eBc7SrgbBlock : vk::Format::eBc7UnormBlock;

        default: return fmt;
    }
}

// Create texture image
bool Renderer::createTextureImage(const std::string& texturePath_, TextureResources& resources) {
    try {
        ensureThreadLocalVulkanInit();
        const std::string textureId = ResolveTextureId(texturePath_);
        // Check if texture already exists
        {
            std::shared_lock<std::shared_mutex> texLock(textureResourcesMutex);
            auto it = textureResources.find(textureId);
            if (it != textureResources.end()) {
                // Texture already loaded and cached; leave cache intact and return success
                return true;
            }
        }

        // Resolve on-disk path (may differ from logical ID)
        std::string resolvedPath = textureId;

        // Ensure command pool is initialized before any GPU work
        if (!*commandPool) {
            std::cerr << "createTextureImage: commandPool not initialized yet for '" << textureId << "'" << std::endl;
            return false;
        }

        // Per-texture de-duplication (serialize loads of the same texture ID only)
        {
            std::unique_lock<std::mutex> lk(textureLoadStateMutex);
            while (texturesLoading.find(textureId) != texturesLoading.end()) {
                textureLoadStateCv.wait(lk);
            }
        }
        // Double-check cache after the wait
        {
            std::shared_lock<std::shared_mutex> texLock(textureResourcesMutex);
            auto it2 = textureResources.find(textureId);
            if (it2 != textureResources.end()) {
                return true;
            }
        }
        // Mark as loading and ensure we notify on all exit paths
        {
            std::lock_guard<std::mutex> lk(textureLoadStateMutex);
            texturesLoading.insert(textureId);
        }
        auto _loadingGuard = std::unique_ptr<void, std::function<void(void*)>>((void*)1, [this, textureId](void*){
            std::lock_guard<std::mutex> lk(textureLoadStateMutex);
            texturesLoading.erase(textureId);
            textureLoadStateCv.notify_all();
        });

        // Check if this is a KTX2 file
        bool isKtx2 = resolvedPath.find(".ktx2") != std::string::npos;

        // If it's a KTX2 texture but the path doesn't exist, try common fallback filename variants
        if (isKtx2) {
            std::filesystem::path origPath(resolvedPath);
            if (!std::filesystem::exists(origPath)) {
                std::string fname = origPath.filename().string();
                std::string dir = origPath.parent_path().string();
                auto tryCandidate = [&](const std::string& candidateName) -> bool {
                    std::filesystem::path cand = std::filesystem::path(dir) / candidateName;
                    if (std::filesystem::exists(cand)) {
                        std::cout << "Resolved missing texture '" << resolvedPath << "' to existing file '" << cand.string() << "'" << std::endl;
                        resolvedPath = cand.string();
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

        // Track KTX2 transcoding state across the function scope (BasisU only)
        bool wasTranscoded = false;
        // Track KTX2 header-provided VkFormat (0 == VK_FORMAT_UNDEFINED)
        uint32_t headerVkFormatRaw = 0;

        uint32_t mipLevels = 1;
        std::vector<vk::BufferImageCopy> copyRegions;

        if (isKtx2) {
            // Load KTX2 file
            KTX_error_code result = ktxTexture2_CreateFromNamedFile(resolvedPath.c_str(),
                                                                   KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
                                                                   &ktxTex);
            if (result != KTX_SUCCESS) {
                // Retry with sibling suffix variants if file exists but cannot be parsed/opened
                std::filesystem::path origPath(resolvedPath);
                std::string fname = origPath.filename().string();
                std::string dir = origPath.parent_path().string();
                auto tryLoad = [&](const std::string& candidateName) -> bool {
                    std::filesystem::path cand = std::filesystem::path(dir) / candidateName;
                    if (std::filesystem::exists(cand)) {
                        std::string candStr = cand.string();
                        std::cout << "Retrying KTX2 load with sibling candidate '" << candStr << "' for original '" << resolvedPath << "'" << std::endl;
                        result = ktxTexture2_CreateFromNamedFile(candStr.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTex);
                        if (result == KTX_SUCCESS) {
                            resolvedPath = candStr; // Use the successfully opened candidate
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
            }

            // Bail out if we still failed to load
            if (result != KTX_SUCCESS || ktxTex == nullptr) {
                std::cerr << "Failed to load KTX2 texture: " << resolvedPath << " (error: " << result << ")" << std::endl;
                return false;
            }

            // Read header-provided vkFormat (if already GPU-compressed/transcoded offline)
            headerVkFormatRaw = static_cast<uint32_t>(ktxTex->vkFormat);

            // Check if the texture needs BasisU transcoding; if so, transcode to RGBA32
            wasTranscoded = ktxTexture2_NeedsTranscoding(ktxTex);
            if (wasTranscoded) {
                result = ktxTexture2_TranscodeBasis(ktxTex, KTX_TTF_RGBA32, 0);
                if (result != KTX_SUCCESS) {
                    std::cerr << "Failed to transcode KTX2 BasisU texture to RGBA32: " << resolvedPath << " (error: " << result << ")" << std::endl;
                    ktxTexture_Destroy((ktxTexture*)ktxTex);
                    return false;
                }
            }

            texWidth = ktxTex->baseWidth;
            texHeight = ktxTex->baseHeight;
            texChannels = 4; // logical channels; compressed size handled below
            // Disable mipmapping for now - memory pool only supports single mip level
            // TODO: Implement proper mipmap support in memory pool
            mipLevels = 1;

            // Calculate size for base level only (use libktx for correct size incl. compression)
            imageSize = ktxTexture_GetImageSize((ktxTexture*)ktxTex, 0);

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
            std::cerr << "Unsupported non-KTX2 texture path: " << textureId << std::endl;
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
            // Copy KTX2 texture data for base level only (level 0), regardless of transcode target
            ktx_size_t offset = 0;
            ktxTexture_GetImageOffset((ktxTexture*)ktxTex, 0, 0, 0, &offset);
            const void* levelData = ktxTexture_GetData((ktxTexture*)ktxTex) + offset;
            size_t levelSize = ktxTexture_GetImageSize((ktxTexture*)ktxTex, 0);
            memcpy(data, levelData, levelSize);
        } else {
            // Copy regular image data
            memcpy(data, pixels, static_cast<size_t>(imageSize));
        }

        stagingBufferMemory.unmapMemory();


        // Determine appropriate texture format
        vk::Format textureFormat;
        const bool wantSRGB = (Renderer::determineTextureFormat(textureId) == vk::Format::eR8G8B8A8Srgb);
        bool alphaMaskedHint = false;
        if (isKtx2) {
            // If the KTX2 provided a valid VkFormat and we did NOT transcode, respect its block type
            // but coerce the sRGB/UNORM variant based on texture usage (baseColor vs data maps)
            if (!wasTranscoded) {
                VkFormat headerFmt = static_cast<VkFormat>(headerVkFormatRaw);
                if (headerFmt != VK_FORMAT_UNDEFINED) {
                    textureFormat = CoerceFormatSRGB(static_cast<vk::Format>(headerFmt), wantSRGB);
                } else {
                    textureFormat = wantSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
                }
                // Can't easily scan alpha in compressed formats here; leave hint at default false
            } else {
                // Transcoded to RGBA32; choose SRGB/UNORM by heuristic
                textureFormat = wantSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
                // We have CPU-visible RGBA data in 'levelData' above; scan alpha for masking hint
                if (ktxTex) {
                    ktx_size_t offsetScan = 0;
                    ktxTexture_GetImageOffset((ktxTexture*)ktxTex, 0, 0, 0, &offsetScan);
                    const uint8_t* rgba = ktxTexture_GetData((ktxTexture*)ktxTex) + offsetScan;
                    size_t pixelCount = static_cast<size_t>(texWidth) * static_cast<size_t>(texHeight);
                    for (size_t i = 0; i < pixelCount; ++i) {
                        if (rgba[i * 4 + 3] < 250) { alphaMaskedHint = true; break; }
                    }
                }
            }
        } else {
            textureFormat = wantSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
        }

        // Now that we're done reading libktx data, destroy the KTX texture to avoid leaks
        if (isKtx2 && ktxTex) {
            ktxTexture_Destroy((ktxTexture*)ktxTex);
            ktxTex = nullptr;
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

        // GPU upload for this texture
        uploadImageFromStaging(*stagingBuffer, *resources.textureImage, textureFormat, copyRegions, mipLevels);

        // Store the format and mipLevels for createTextureImageView
        resources.format = textureFormat;
        resources.mipLevels = mipLevels;
        resources.alphaMaskedHint = alphaMaskedHint;

        // Create texture image view
        if (!createTextureImageView(resources)) {
            return false;
        }

        // Create texture sampler
        if (!createTextureSampler(resources)) {
            return false;
        }

        // Add to texture resources map (guarded)
        {
            std::unique_lock<std::shared_mutex> texLock(textureResourcesMutex);
            textureResources[textureId] = std::move(resources);
        }

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
        ensureThreadLocalVulkanInit();
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
    ensureThreadLocalVulkanInit();
    if (texturePath.empty()) {
        std::cerr << "LoadTexture: Empty texture path provided" << std::endl;
        return false;
    }

    // Resolve aliases (canonical ID -> actual key)
    const std::string resolvedId = ResolveTextureId(texturePath);

    // Check if texture is already loaded
    {
        std::shared_lock<std::shared_mutex> texLock(textureResourcesMutex);
        auto it = textureResources.find(resolvedId);
        if (it != textureResources.end()) {
            // Texture already loaded
            return true;
        }
    }

    // Create temporary texture resources (unused output; cache will be populated internally)
    TextureResources tempResources;

    // Use existing createTextureImage method (it inserts into textureResources on success)
    bool success = createTextureImage(resolvedId, tempResources);

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

    // BaseColor/Albedo/Diffuse & SpecGloss RGB should be sRGB for proper gamma correction
    if (idLower.find("basecolor") != std::string::npos ||
        idLower.find("base_color") != std::string::npos ||
        idLower.find("albedo") != std::string::npos ||
        idLower.find("diffuse") != std::string::npos ||
        idLower.find("specgloss") != std::string::npos ||
        idLower.find("specularglossiness") != std::string::npos ||
        textureId == Renderer::SHARED_DEFAULT_ALBEDO_ID) {
        return vk::Format::eR8G8B8A8Srgb;
    }

    // Emissive is color data and should be sampled in sRGB
    if (idLower.find("emissive") != std::string::npos ||
        textureId == Renderer::SHARED_DEFAULT_EMISSIVE_ID) {
        return vk::Format::eR8G8B8A8Srgb;
    }

    // Shared bright red (ball) is a color texture; ensure sRGB for vivid appearance
    if (textureId == Renderer::SHARED_BRIGHT_RED_ID) {
        return vk::Format::eR8G8B8A8Srgb;
    }

    // All other PBR textures (normal, metallic-roughness, occlusion) should be linear
    // because they contain non-color data that shouldn't be gamma corrected
    return vk::Format::eR8G8B8A8Unorm;
}

// Load texture from raw image data in memory
bool Renderer::LoadTextureFromMemory(const std::string& textureId, const unsigned char* imageData,
                                    int width, int height, int channels) {
    ensureThreadLocalVulkanInit();
    const std::string resolvedId = ResolveTextureId(textureId);
    std::cout << "[LoadTextureFromMemory] start id=" << textureId << " -> resolved=" << resolvedId << " size=" << width << "x" << height << " ch=" << channels << std::endl;
    if (resolvedId.empty() || !imageData || width <= 0 || height <= 0 || channels <= 0) {
        std::cerr << "LoadTextureFromMemory: Invalid parameters" << std::endl;
        return false;
    }

    // Check if texture is already loaded
    {
        std::shared_lock<std::shared_mutex> texLock(textureResourcesMutex);
        auto it = textureResources.find(resolvedId);
        if (it != textureResources.end()) {
            // Texture already loaded
            return true;
        }
    }

    // Per-texture de-duplication (serialize loads of the same texture ID only)
    {
        std::unique_lock<std::mutex> lk(textureLoadStateMutex);
        while (texturesLoading.find(resolvedId) != texturesLoading.end()) {
            textureLoadStateCv.wait(lk);
        }
    }
    // Double-check cache after the wait
    {
        std::shared_lock<std::shared_mutex> texLock(textureResourcesMutex);
        auto it2 = textureResources.find(resolvedId);
        if (it2 != textureResources.end()) {
            return true;
        }
    }
    // Mark as loading and ensure we notify on all exit paths
    {
        std::lock_guard<std::mutex> lk(textureLoadStateMutex);
        texturesLoading.insert(resolvedId);
    }
    auto _loadingGuard = std::unique_ptr<void, std::function<void(void*)>>((void*)1, [this, resolvedId](void*){
        std::lock_guard<std::mutex> lk(textureLoadStateMutex);
        texturesLoading.erase(resolvedId);
        textureLoadStateCv.notify_all();
    });

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

        // Analyze alpha to set alphaMaskedHint (treat as masked if any pixel alpha < ~1.0)
        bool alphaMaskedHint = false;
        for (int i = 0, n = width * height; i < n; ++i) {
            if (stagingData[i * 4 + 3] < 250) { alphaMaskedHint = true; break; }
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

        // GPU upload. Copy buffer to image in a single submit.
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
        uploadImageFromStaging(*stagingBuffer, *resources.textureImage, textureFormat, regions, 1);

        // Store the format for createTextureImageView
        resources.format = textureFormat;
        resources.alphaMaskedHint = alphaMaskedHint;

        // Use resolvedId as the cache key to avoid duplicates
        const std::string& cacheId = resolvedId;

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

        // Add to texture resources map (guarded)
        {
            std::unique_lock<std::shared_mutex> texLock(textureResourcesMutex);
            textureResources[cacheId] = std::move(resources);
        }

        std::cout << "Successfully loaded texture from memory: " << cacheId
                  << " (" << width << "x" << height << ", " << channels << " channels)" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load texture from memory: " << e.what() << std::endl;
        return false;
    }
}

// Create mesh resources
bool Renderer::createMeshResources(MeshComponent* meshComponent) {
    ensureThreadLocalVulkanInit();
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
    ensureThreadLocalVulkanInit();
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
    // Resolve alias before taking the shared lock to avoid nested shared_lock on the same mutex
    const std::string resolvedTexturePath = ResolveTextureId(texturePath);
    std::shared_lock<std::shared_mutex> texLock(textureResourcesMutex);
    try {
        auto entityIt = entityResources.find(entity);
        if (entityIt == entityResources.end()) return false;

        vk::DescriptorSetLayout selectedLayout = usePBR ? *pbrDescriptorSetLayout : *descriptorSetLayout;
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, selectedLayout);
        vk::DescriptorSetAllocateInfo allocInfo{ .descriptorPool = *descriptorPool, .descriptorSetCount = MAX_FRAMES_IN_FLIGHT, .pSetLayouts = layouts.data() };

        auto& targetDescriptorSets = usePBR ? entityIt->second.pbrDescriptorSets : entityIt->second.basicDescriptorSets;
        if (targetDescriptorSets.empty()) {
            targetDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo bufferInfo{ .buffer = *entityIt->second.uniformBuffers[i], .range = sizeof(UniformBufferObject) };

            if (usePBR) {
                // PBR sets now only have 7 bindings (0-6)
                std::array<vk::WriteDescriptorSet, 7> descriptorWrites;
                std::array<vk::DescriptorImageInfo, 5> imageInfos;

                descriptorWrites[0] = { .dstSet = *targetDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo };

                auto meshComponent = entity->GetComponent<MeshComponent>();
                std::vector<std::string> pbrTexturePaths = { /* ... same as before ... */ };
                // ... (logic to get texture paths is the same)
                {
                    const std::string legacyPath = (meshComponent ? meshComponent->GetTexturePath() : std::string());
                    const std::string baseColorPath = (meshComponent && !meshComponent->GetBaseColorTexturePath().empty())
                                                      ? meshComponent->GetBaseColorTexturePath()
                                                      : (!legacyPath.empty() ? legacyPath : SHARED_DEFAULT_ALBEDO_ID);
                    const std::string mrPath = (meshComponent && !meshComponent->GetMetallicRoughnessTexturePath().empty())
                                               ? meshComponent->GetMetallicRoughnessTexturePath()
                                               : SHARED_DEFAULT_METALLIC_ROUGHNESS_ID;
                    const std::string normalPath = (meshComponent && !meshComponent->GetNormalTexturePath().empty())
                                                   ? meshComponent->GetNormalTexturePath()
                                                   : SHARED_DEFAULT_NORMAL_ID;
                    const std::string occlusionPath = (meshComponent && !meshComponent->GetOcclusionTexturePath().empty())
                                                      ? meshComponent->GetOcclusionTexturePath()
                                                      : SHARED_DEFAULT_OCCLUSION_ID;
                    const std::string emissivePath = (meshComponent && !meshComponent->GetEmissiveTexturePath().empty())
                                                    ? meshComponent->GetEmissiveTexturePath()
                                                    : SHARED_DEFAULT_EMISSIVE_ID;

                    pbrTexturePaths = { baseColorPath, mrPath, normalPath, occlusionPath, emissivePath };
                }


                for (int j = 0; j < 5; j++) {
                    const auto resolvedBindingPath = ResolveTextureId(pbrTexturePaths[j]);
                    auto textureIt = textureResources.find(resolvedBindingPath);
                    TextureResources* texRes = (textureIt != textureResources.end()) ? &textureIt->second : &defaultTextureResources;
                    imageInfos[j] = { .sampler = *texRes->textureSampler, .imageView = *texRes->textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
                    descriptorWrites[j + 1] = { .dstSet = *targetDescriptorSets[i], .dstBinding = static_cast<uint32_t>(j + 1), .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &imageInfos[j] };
                }

                vk::DescriptorBufferInfo lightBufferInfo{ .buffer = *lightStorageBuffers[i].buffer, .range = VK_WHOLE_SIZE };
                descriptorWrites[6] = { .dstSet = *targetDescriptorSets[i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &lightBufferInfo };

                device.updateDescriptorSets(descriptorWrites, {});
            } else { // Basic Pipeline
                // ... (this part remains the same)
                 auto textureIt = textureResources.find(resolvedTexturePath);
                TextureResources* texRes = (textureIt != textureResources.end()) ? &textureIt->second : &defaultTextureResources;
                vk::DescriptorImageInfo imageInfo{ .sampler = *texRes->textureSampler, .imageView = *texRes->textureImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
                std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {
                    vk::WriteDescriptorSet{ .dstSet = *targetDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo },
                    vk::WriteDescriptorSet{ .dstSet = *targetDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &imageInfo }
                };
                device.updateDescriptorSets(descriptorWrites, {});
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create descriptor sets for " << entity->GetName() << ": " << e.what() << std::endl;
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

void Renderer::createTransparentDescriptorSets() {
    // We need one descriptor set per frame in flight for this resource
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *transparentDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .pSetLayouts = layouts.data()
    };

    transparentDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    // Update each descriptor set to point to our single off-screen opaque color image
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorImageInfo imageInfo{
            .sampler = *opaqueSceneColorSampler,
            .imageView = *opaqueSceneColorImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };

        vk::WriteDescriptorSet descriptorWrite{
            .dstSet = *transparentDescriptorSets[i],
            .dstBinding = 0, // Binding 0 in Set 1
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &imageInfo
        };

        device.updateDescriptorSets(descriptorWrite, nullptr);
    }
}

void Renderer::createTransparentFallbackDescriptorSets() {
    // Allocate one descriptor set per frame in flight using the same layout (single combined image sampler at binding 0)
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *transparentDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .pSetLayouts = layouts.data()
    };

    transparentFallbackDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);

    // Point each set to the default texture, which is guaranteed to be in SHADER_READ_ONLY_OPTIMAL when used in the opaque pass
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorImageInfo imageInfo{
            .sampler = *defaultTextureResources.textureSampler,
            .imageView = *defaultTextureResources.textureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };

        vk::WriteDescriptorSet descriptorWrite{
            .dstSet = *transparentFallbackDescriptorSets[i],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &imageInfo
        };

        device.updateDescriptorSets(descriptorWrite, nullptr);
    }
}

bool Renderer::createOpaqueSceneColorResources() {
    try {
        // Create the image
        auto [image, allocation] = createImagePooled(
            swapChainExtent.width,
            swapChainExtent.height,
            swapChainImageFormat, // Use the same format as the swapchain
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc, // <-- Note the new usage flags
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        opaqueSceneColorImage = std::move(image);
        // We don't need a member for the allocation, it's managed by the unique_ptr

        // Create the image view
        opaqueSceneColorImageView = createImageView(opaqueSceneColorImage, swapChainImageFormat, vk::ImageAspectFlagBits::eColor);

        // Create the sampler
        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        };
        opaqueSceneColorSampler = vk::raii::Sampler(device, samplerInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create opaque scene color resources: " << e.what() << std::endl;
        return false;
    }
}

// Copy buffer
void Renderer::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
    ensureThreadLocalVulkanInit();
    try {
        // Create a temporary transient command pool and command buffer to isolate per-thread usage (transfer family)
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.transferFamily.value()
        };
        vk::raii::CommandPool tempPool(device, poolInfo);
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *tempPool,
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

        // Use mutex to ensure thread-safe access to transfer queue
        vk::raii::Fence fence(device, vk::FenceCreateInfo{});
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            transferQueue.submit(submitInfo, *fence);
        }
        device.waitForFences({*fence}, VK_TRUE, UINT64_MAX);
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

        return {std::move(image), std::move(allocation)};

    } catch (const std::exception& e) {
        std::cerr << "Failed to create image with memory pool: " << e.what() << std::endl;
        throw;
    }
}

// Create an image view
vk::raii::ImageView Renderer::createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) {
    try {
        ensureThreadLocalVulkanInit();
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
    ensureThreadLocalVulkanInit();
    try {
        // Create a temporary transient command pool and command buffer to isolate per-thread usage
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
        };
        vk::raii::CommandPool tempPool(device, poolInfo);
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *tempPool,
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
        std::cout << "[transitionImageLayout] recorded barrier image=" << (void*)image << " old=" << (int)oldLayout << " new=" << (int)newLayout << std::endl;

        // End command buffer
        commandBuffer.end();

        // Submit command buffer

        // Submit transition; protect submit with mutex but wait outside
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };
        vk::raii::Fence fence(device, vk::FenceCreateInfo{});
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            graphicsQueue.submit(submitInfo, *fence);
        }
        device.waitForFences({*fence}, VK_TRUE, UINT64_MAX);
    } catch (const std::exception& e) {
        std::cerr << "Failed to transition image layout: " << e.what() << std::endl;
        throw;
    }
}

// Copy buffer to image
void Renderer::copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height, const std::vector<vk::BufferImageCopy>& regions) const {
    ensureThreadLocalVulkanInit();
    try {
        // Create a temporary transient command pool for the GRAPHICS queue to avoid cross-queue races
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
        };
        vk::raii::CommandPool tempPool(device, poolInfo);
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *tempPool,
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
        std::cout << "[copyBufferToImage] recorded copy img=" << (void*)image << std::endl;

        // End command buffer
        commandBuffer.end();

        // Submit command buffer
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };

        // Protect submit with queue mutex, wait outside
        vk::raii::Fence fence(device, vk::FenceCreateInfo{});
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            graphicsQueue.submit(submitInfo, *fence);
        }
        device.waitForFences({*fence}, VK_TRUE, UINT64_MAX);
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


// Asynchronous texture loading implementations using ThreadPool
std::future<bool> Renderer::LoadTextureAsync(const std::string& texturePath) {
    if (texturePath.empty()) {
        return std::async(std::launch::deferred, [] { return false; });
    }
    // Schedule the load without dropping tasks; throttling is handled during GPU upload
    textureTasksScheduled.fetch_add(1, std::memory_order_relaxed);
    auto task = [this, texturePath]() {
        bool ok = this->LoadTexture(texturePath);
        textureTasksCompleted.fetch_add(1, std::memory_order_relaxed);
        return ok;
    };
    
    // Use shared_lock to prevent race condition with threadPool destruction
    std::shared_lock<std::shared_mutex> lock(threadPoolMutex);
    if (!threadPool) {
        return std::async(std::launch::async, task);
    }
    return threadPool->enqueue(task);
}

std::future<bool> Renderer::LoadTextureFromMemoryAsync(const std::string& textureId, const unsigned char* imageData,
                              int width, int height, int channels) {
    if (!imageData || textureId.empty() || width <= 0 || height <= 0 || channels <= 0) {
        return std::async(std::launch::deferred, [] { return false; });
    }
    // Copy the source bytes so the caller can free/modify their buffer immediately
    size_t srcSize = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);
    std::vector<unsigned char> dataCopy(srcSize);
    std::memcpy(dataCopy.data(), imageData, srcSize);

    textureTasksScheduled.fetch_add(1, std::memory_order_relaxed);
    auto task = [this, textureId, data = std::move(dataCopy), width, height, channels]() mutable {
        bool ok = this->LoadTextureFromMemory(textureId, data.data(), width, height, channels);
        textureTasksCompleted.fetch_add(1, std::memory_order_relaxed);
        return ok;
    };
    
    // Use shared_lock to prevent race condition with threadPool destruction
    std::shared_lock<std::shared_mutex> lock(threadPoolMutex);
    if (!threadPool) {
        return std::async(std::launch::async, std::move(task));
    }
    return threadPool->enqueue(std::move(task));
}


// Record both layout transitions and the copy in a single submission with a fence
void Renderer::uploadImageFromStaging(vk::Buffer staging,
                                      vk::Image image,
                                      vk::Format format,
                                      const std::vector<vk::BufferImageCopy>& regions,
                                      uint32_t mipLevels) {
    ensureThreadLocalVulkanInit();
    try {
        // Use a temporary transient command pool for the GRAPHICS queue family to avoid cross-queue races
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
        };
        vk::raii::CommandPool tempPool(device, poolInfo);
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *tempPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        vk::raii::CommandBuffers cbs(device, allocInfo);
        vk::raii::CommandBuffer& cb = cbs[0];

        vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
        cb.begin(beginInfo);

        // Barrier: Undefined -> TransferDstOptimal
        vk::ImageMemoryBarrier toTransfer{
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = (format == vk::Format::eD32Sfloat || format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint)
                               ? vk::ImageAspectFlagBits::eDepth
                               : vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        toTransfer.srcAccessMask = vk::AccessFlagBits::eNone;
        toTransfer.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                           vk::PipelineStageFlagBits::eTransfer,
                           vk::DependencyFlagBits::eByRegion,
                           nullptr, nullptr, toTransfer);

        // Copy
        cb.copyBufferToImage(staging, image, vk::ImageLayout::eTransferDstOptimal, regions);

        // Barrier: TransferDstOptimal -> ShaderReadOnlyOptimal
        vk::ImageMemoryBarrier toShader{
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = (format == vk::Format::eD32Sfloat || format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint)
                               ? vk::ImageAspectFlagBits::eDepth
                               : vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        toShader.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        // Keep dstAccessMask empty; visibility is ensured via submission ordering and timeline wait
        cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                           vk::PipelineStageFlagBits::eTransfer,
                           vk::DependencyFlagBits::eByRegion,
                           nullptr, nullptr, toShader);

        cb.end();

        // Submit once on the GRAPHICS queue; signal uploads timeline if available
        vk::raii::Fence fence(device, vk::FenceCreateInfo{});
        bool canSignalTimeline = uploadsTimeline != nullptr;
        uint64_t signalValue = 0;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            vk::SubmitInfo submit{};
            if (canSignalTimeline) {
                signalValue = uploadTimelineLastSubmitted.fetch_add(1, std::memory_order_relaxed) + 1;
                vk::TimelineSemaphoreSubmitInfo timelineInfo{
                    .signalSemaphoreValueCount = 1,
                    .pSignalSemaphoreValues = &signalValue
                };
                submit.pNext = &timelineInfo;
                submit.signalSemaphoreCount = 1;
                submit.pSignalSemaphores = &*uploadsTimeline;
            }
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &*cb;

            graphicsQueue.submit(submit, *fence);
        }
        device.waitForFences({*fence}, VK_TRUE, UINT64_MAX);
    } catch (const std::exception& e) {
        std::cerr << "uploadImageFromStaging failed: " << e.what() << std::endl;
        throw;
    }
}
