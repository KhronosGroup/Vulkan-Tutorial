/* Copyright (c) 2026, Khronos Group and contributors
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include <VkBootstrap.h>
#include <vulkan/vulkan.hpp>

#include "util.h"

struct GLFWwindow;

namespace siggraph {

struct ShaderFilePair {
    std::string_view vertexName;
    std::string_view fragmentName;
};

// Helper with all scene paths. This makes it easy to change the scene for testing.
namespace SceneData {

constexpr std::array<std::string_view, 3> sceneFileNames = {"external/sponza/Models/Sponza/glTF/Sponza.gltf",
                                                            "SponzaExtras/VulkanLogo/VulkanLogo.gltf",
                                                            "SponzaExtras/camera.gltf"};

// We have three shader variants: albedo with normal map, albedo only, and solid color.
constexpr ShaderFilePair albedoAndNormalShaderFilePair{.vertexName = "basic.vert.spv",
                                                       .fragmentName = "basic.frag.spv"};

constexpr ShaderFilePair albedoShaderFilePair{.vertexName = "basic.vert.spv", .fragmentName = "albedo.frag.spv"};

// The solid-color shader is only used for objects like the Vulkan logo that lack textures in the glTF.
constexpr ShaderFilePair solidColorShaderFilePair{.vertexName = "basic.vert.spv",
                                                  .fragmentName = "solid_color.frag.spv"};

} // namespace SceneData

// Owns the window, Vulkan bootstrap objects, and the render loop for the tutorial.
// Keeping ownership in one class makes destruction order explicit and easy to follow.
class Application {
public:
    Application() = default;

    // The destructor calls cleanup(), so destroying the Application object releases Vulkan and GLFW resources.
    ~Application();

    // Vulkan objects are unique OS/GPU handles. Copying the Application would duplicate
    // ownership incorrectly, so copying is disabled.
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    // Entry point for the tutorial: create the window, initialize Vulkan, render until close.
    void run();

    void setFrameLimit(std::uint32_t frameLimit);

private:
    // Helper to initialize image subresource ranges with common defaults.
    static constexpr vk::ImageSubresourceRange
    CreateImageSubresourceRange(const std::uint32_t baseMipLevel = 0,
                                const vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor,
                                const std::uint32_t levelCount = 1)
    {
        return vk::ImageSubresourceRange{
            .aspectMask = aspectMask,
            .baseMipLevel = baseMipLevel,
            .levelCount = levelCount,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
    }

    // Helper to keep track of the state of an image to issue barriers.
    struct ImageState {
        vk::ImageLayout m_layout = vk::ImageLayout::eUndefined;
        vk::PipelineStageFlags2 m_stageMask = vk::PipelineStageFlagBits2::eNone;
        vk::AccessFlags2 m_accessMask = vk::AccessFlagBits2::eNone;
        vk::ImageAspectFlags m_aspectMask = vk::ImageAspectFlagBits::eColor;

        bool operator==(const ImageState&) const = default;
    };

    // Wrapper around a GPU buffer and its related resources.
    struct GpuBuffer {
        // Memory is declared before the buffer because members are destroyed in reverse order.
        // That lets the Vulkan-Hpp unique buffer release the VkBuffer before the memory is freed.
        vk::UniqueDeviceMemory m_memory{};
        vk::UniqueBuffer m_buffer{};
        vk::DeviceAddress m_addressGPU{};
        vk::DeviceSize m_size = 0;
        void* m_addressCPU = nullptr;

        void destroy(vk::Device device);
    };

    // Wrapper around a GPU image and its related resources.
    struct GpuImage {
        // Image memory must outlive the image bound to it.
        vk::UniqueDeviceMemory m_memory{};
        vk::UniqueImage m_image{};
        std::uint32_t m_mipLevels = 1;
        std::vector<ImageState> m_states{};

        void destroy();
        void transition(vk::CommandBuffer commandBuffer, const ImageState& newState);
        void transition(vk::CommandBuffer commandBuffer, const ImageState& newState, std::uint32_t baseMipLevel,
                        std::uint32_t mipLevelCount);
    };

    struct GpuViewImage {
        // Image views must be destroyed before the image.
        GpuImage m_image{};
        vk::UniqueImageView m_imageView{};

        void destroy();
    };

    // Resources used during rendering.
    // We can render multiple frames in flight, so we have to duplicate some resources.
    // This allows the CPU to start preparing frame N+1 before the GPU finishes frame N.
    struct FrameInFlightResources {
        vk::CommandBuffer m_commandBuffer{};
        GpuBuffer m_camera{};

        vk::UniqueSemaphore m_imageAvailableSemaphore{};
        vk::UniqueFence m_inFlightFence{};

        GpuViewImage m_depthImage{};
    };

    // Resources that belong to each swapchain image.
    // Swapchain images are owned by Vulkan; we own views and per-image synchronization around them.
    struct SwapchainImageResources {
        // The image provided by the swapchain. We render directly into it.
        // Owned by the swapchain, so we don't destroy it.
        // This is a non-owning copy; we only keep track of it for synchronization.
        vk::Image m_image{};

        vk::UniqueImageView m_imageView{};

        // Presentation can keep waiting on this binary semaphore.
        // Keep one per swapchain image so we only reuse it after that image is acquired again,
        // and we wait for the acquire semaphore.
        vk::UniqueSemaphore m_renderFinishedSemaphore{};

        // Track the swapchain image layout between frames so the next render pass knows the
        // correct source state for its first barrier.
        ImageState m_state{};
    };

    struct HelperCommandBuffer {
        vk::CommandBuffer m_commandBuffer{};
        bool m_active = false;
    };

    struct CameraData {
        glm::mat4 viewProjection;
        glm::vec4 cameraPosition;
    };

    // Object data lives in a storage buffer. The vertex shader reads one model matrix per instance.
    struct alignas(16) ObjectData {
        glm::mat4 model;
    };
    static_assert(sizeof(ObjectData) == 64);

    struct alignas(16) LightData {
        glm::vec3 pointPosition;
        float pointIntensity;
        glm::vec3 ambientColor;
        float ambientIntensity;
    };
    static_assert(offsetof(LightData, pointIntensity) == 12);
    static_assert(offsetof(LightData, ambientColor) == 16);
    static_assert(offsetof(LightData, ambientIntensity) == 28);
    static_assert(sizeof(LightData) == 32);

    struct alignas(16) SolidColorData {
        glm::vec4 color;
    };
    static_assert(sizeof(SolidColorData) == 16);

    // For simplicity, the tutorial hardcodes three shaders.
    enum class ShaderVariant : std::uint32_t {
        AlbedoAndNormal,
        Albedo,
        SolidColor,
    };

    // We store the data used to draw a renderable object.
    // We store our data in vectors and store indices into those vectors.
    // This simplifies reallocation and requires less memory, as an index is smaller than a pointer.
    struct SceneDraw {
        std::uint32_t m_meshId = 0;
        std::uint32_t m_objectIndex = 0;
        ShaderVariant m_shaderVariant = ShaderVariant::AlbedoAndNormal;
        std::uint32_t m_albedoTextureIndex = util::gltf::invalidGltfId;
        std::uint32_t m_normalTextureIndex = util::gltf::invalidGltfId;
        std::string m_debugName;
    };

    struct GpuMesh {
        GpuBuffer m_vertices{};
        GpuBuffer m_indices{};
        std::uint32_t m_indexCount = 0;
    };

    struct SceneResources {
        // Geometry, object transforms, and textures used by scene draws.
        GpuBuffer m_objects{};
        GpuBuffer m_pointLight{};
        GpuBuffer m_solidColor{};
        std::vector<GpuMesh> m_meshes{};
        std::vector<GpuImage> m_textures{};
        std::vector<SceneDraw> m_drawData{};
    };

    // Note: To save memory some applications might consider storing alignments, offsets and sizes as uint32_t
    struct DescriptorHeapResources {

        // Wrapper to handle the data of a descriptor heap buffer.
        // Helps reuse code across the sampler and descriptor heaps.
        struct DescriptorHeapData {
            GpuBuffer m_buffer;

            // Bound heap ranges must start at device-aligned GPU addresses. This offset
            // moves from the buffer's base address to the aligned heap base used by Vulkan.
            VkDeviceSize m_bindOffset = 0;

            // Size of the heap.
            VkDeviceSize m_rangeSize = 0;

            // Obtained from device properties during device creation.
            // These values are only needed for allocating and binding heap ranges.
            // We keep them as part of the heap data for simplicity.
            // Note: the resource and sampler heaps can have different requirements.
            //
            // GPU-address alignment requirement for the bound heap range.
            VkDeviceSize m_alignment = 0;
            // Minimum implementation-reserved range size for heap operations.
            VkDeviceSize m_minReservedRange = 0;
        };

        // Host-visible buffers that contain raw descriptor bytes for EXT_descriptor_heap.
        DescriptorHeapData m_resourceHeap{};
        DescriptorHeapData m_samplerHeap{};

        // Descriptor sizes and heap alignments are device properties, so they are queried at startup.
        VkDeviceSize m_sampledImageSize = 0;
        VkDeviceSize m_samplerSize = 0;
        VkDeviceSize m_uniformBufferSize = 0;
        VkDeviceSize m_storageBufferSize = 0;
        VkDeviceSize m_imageDescriptorAlignment = 0;
        VkDeviceSize m_bufferDescriptorAlignment = 0;
        VkDeviceSize m_samplerDescriptorAlignment = 0;
        VkDeviceSize m_maxPushDataSize = 0;
        std::uint32_t m_uniformBufferStride = 0;
        std::uint32_t m_sampledImageStride = 0;

        // Byte offsets of each descriptor inside the heap buffers.
        std::uint32_t m_cameraOffset = 0;
        std::uint32_t m_objectsOffset = 0;
        std::uint32_t m_pointLightOffset = 0;
        std::uint32_t m_solidColorOffset = 0;
        std::uint32_t m_pushTextureOffset = 0;
        std::uint32_t m_linearSamplerOffset = 0;
        std::uint32_t m_nearestSamplerOffset = 0;
    };

    struct ShaderGroup {
        vk::UniqueShaderEXT m_vertex{};
        vk::UniqueShaderEXT m_fragment{};
    };

    //****************
    // Vulkan initialization
    //****************

    void initGLFWWindow();

    void initVulkanVKB();

    void initCommandPool();

    void initFramesInFlightResources();

    void initSwapchainImageSyncObjects();

    void initDepthImages();

    //****************
    // Scene initialization
    //****************

    void initSceneResources();

    void initSceneMeshesAndDrawData(const util::gltf::ParsedData& gltfData, std::vector<ObjectData>& objects);

    void initSceneCamera(const util::gltf::ParsedData& gltfData);

    void initObjectBuffer(std::span<const ObjectData> objects);

    void initPointLight(const util::gltf::ParsedData& gltfData);

    void initSolidColor();

    void initSceneTextures(const util::gltf::ParsedData& gltfData);

    //****************
    // Shader and descriptor setup
    //****************

    void initDescriptorHeaps();

    [[nodiscard]] std::vector<vk::DescriptorSetAndBindingMappingEXT> buildShaderDescriptorMappings(
        const std::unordered_map<std::string, util::slang::ShaderResourceBinding>& shaderResourceBindings) const;

    void calculateVertexInputs();

    void initShaderObjects();

    ShaderGroup createShaderGroup(const ShaderFilePair& shaderFilePair);

    //****************
    // Runtime loop
    //****************

    void mainLoop();

    void updateCamera(float deltaSeconds);

    void renderFrame();

    void waitForFrameResources(FrameInFlightResources& frame);

    std::uint32_t acquireSwapchainImage(FrameInFlightResources& frame);

    void uploadCameraData(FrameInFlightResources& frame);

    void startRecordingCommandBuffer(FrameInFlightResources& frame);

    void recordRenderingCommandBuffer(FrameInFlightResources& frame, std::uint32_t frameIndex,
                                      std::uint32_t swapchainImageIndex);
    void finishAndSubmitMainCommandBuffer(FrameInFlightResources& frame, std::uint32_t swapchainImageIndex);

    //****************
    // Helper command buffer utils
    //****************

    [[nodiscard]] vk::CommandBuffer getHelperCommandBuffer() const;

    void beginHelperCommands();

    void endHelperCommandsAndFlushUploads();

    //****************
    // Destruction and cleanup
    //****************

    void cleanup();

    //****************
    // Resource utils
    //****************

    [[nodiscard]] std::uint32_t findMemoryType(std::uint32_t typeBits, vk::MemoryPropertyFlags properties) const;
    [[nodiscard]] GpuBuffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                         vk::MemoryPropertyFlags memoryProperties, std::string_view debugName) const;
    [[nodiscard]] GpuBuffer uploadToNewStagingBuffer(std::span<const std::byte> data, std::string_view debugName) const;
    [[nodiscard]] GpuBuffer uploadToNewGpuBuffer(std::span<const std::byte> data, vk::BufferUsageFlags finalUsage,
                                                 std::string_view debugName);
    void uploadBuffer(const GpuBuffer& buffer, std::span<const std::byte> data) const;
    void allocateGpuImage(GpuImage& image) const;
    [[nodiscard]] GpuImage createTexture(const util::ImageRgba8& image, std::string_view debugName);

    template <typename Handle>
    void setDebugName(Handle handle, vk::ObjectType objectType, const std::string& name) const;
    static void beginDebugLabel(vk::CommandBuffer commandBuffer, const std::string& name,
                                const std::array<float, 4>& color);
    static void endDebugLabel(vk::CommandBuffer commandBuffer);

    static void transitionImage(vk::CommandBuffer commandBuffer, vk::Image image, ImageState& currentState,
                                const ImageState& newState, vk::ImageSubresourceRange subresourceRange);
    //****************
    // Data
    //****************

    // GLFW owns the platform window; Vulkan owns the surface created from it.
    GLFWwindow* m_window = nullptr;

    // vk-bootstrap metadata needed for cleanup during destruction.
    struct {
        vkb::Instance m_instance{};
        vkb::Device m_device{};
        vkb::Swapchain m_swapchain{};
    } m_vkbData;

    vk::SurfaceKHR m_surface{};
    vk::Device m_logicalDevice{};
    std::vector<vk::MemoryPropertyFlags> m_memoryTypeFlags{};

    vk::Queue m_graphicsQueue{};
    vk::Queue m_presentQueue{};
    std::uint32_t m_graphicsQueueFamily = 0;

    vk::Format m_swapchainFormat{};
    vk::Extent2D m_swapchainExtent{};
    std::vector<SwapchainImageResources> m_swapchainImages;

    // Owns the command buffers used by the frame resources.
    // UniqueCommandPool releases it automatically.
    vk::UniqueCommandPool m_commandPool{};
    HelperCommandBuffer m_helperCommandBuffer{};
    std::vector<GpuBuffer> m_pendingUploadStagingBuffers{};

    // Two frames in flight provide useful overlap without much extra memory usage.
    static constexpr std::uint32_t maxFramesInFlight = 2;
    std::array<FrameInFlightResources, maxFramesInFlight> m_framesInFlight{};

    std::uint32_t m_currentFrameInFlight = 0;

    // Number of frames before the application will close itself. Useful for automated testing.
    // 0 for unlimited frames.
    std::uint32_t m_remainingFrameLimit = 0;

    struct VertexInput {
        std::vector<vk::VertexInputBindingDescription2EXT> m_vertexBindings{};
        std::vector<vk::VertexInputAttributeDescription2EXT> m_vertexAttributes{};
    };
    VertexInput m_vertexInput;

    struct ShaderObjects {
        ShaderGroup m_albedoAndNormal{};
        ShaderGroup m_albedo{};
        ShaderGroup m_solidColor{};
    };
    ShaderObjects m_shaderObjects;

    struct ShaderBinaryCacheProperties {
        std::array<std::uint8_t, VK_UUID_SIZE> m_shaderBinaryUUID{};
        std::uint32_t m_shaderBinaryVersion = 0;
    };
    ShaderBinaryCacheProperties m_shaderBinaryCacheProperties{};

    SceneResources m_scene{};
    DescriptorHeapResources m_descriptorHeaps{};

    glm::vec3 m_cameraPos{};
    float m_cameraYaw = 0.0F;
    float m_cameraPitch = 0.0F;

    //*******
    // Constant values used to configure the application.
    // *******
    static constexpr float cameraPitchLimit = glm::radians(89.0F);

    // The tutorial has a fixed window size and does not handle resizing.
    // Resize is not implemented because it requires recreating the swapchain and all related resources.
    static constexpr int windowWidth = 1280;
    static constexpr int windowHeight = 720;

    static struct DebugData {
        // This enables adding debug names and labels, gates validation layers, and enables other debug features.
        static constexpr bool enableGpuDebug = true;

        // These colors are used for debug markers.
        // This allows us to visually identify which part of the frame corresponds to each call.
        static constexpr std::array<float, 4> frameColor{0.18F, 0.34F, 0.78F, 1.0F};
        static constexpr std::array<float, 4> setupColor{0.62F, 0.42F, 0.16F, 1.0F};
        static constexpr std::array<float, 4> transferColor{0.78F, 0.42F, 0.18F, 1.0F};
        static constexpr std::array<float, 4> renderColor{0.18F, 0.55F, 0.35F, 1.0F};
        static constexpr std::array<float, 4> drawColor{0.62F, 0.24F, 0.72F, 1.0F};
        static constexpr std::array<float, 4> barrierColor{0.48F, 0.48F, 0.48F, 1.0F};

    } debugData;

    // Store depth/color formats and image layouts for easy reuse.
    static constexpr vk::Format depthFormat = vk::Format::eD32Sfloat;
    static constexpr vk::Format mainTextureFormat = vk::Format::eR8G8B8A8Unorm;

    static constexpr vk::ImageLayout colorAttachmentLayout = vk::ImageLayout::eAttachmentOptimal;
    static constexpr vk::ImageLayout depthAttachmentLayout = vk::ImageLayout::eAttachmentOptimal;

    // We render directly to the swapchain images.
    static constexpr vk::PipelineStageFlags2 swapchainAcquireWaitStage =
        vk::PipelineStageFlagBits2::eColorAttachmentOutput;

    // Helper to initialize component mapping with identity swizzles.
    static constexpr vk::ComponentMapping identityComponentMapping{
        .r = vk::ComponentSwizzle::eIdentity,
        .g = vk::ComponentSwizzle::eIdentity,
        .b = vk::ComponentSwizzle::eIdentity,
        .a = vk::ComponentSwizzle::eIdentity,
    };

}; // class Application

// Note: a normal application might generate shader-specific structs
// like Vertex or DescriptorHeapDrawPushIndicesAlbedoAndNormal from shader reflection data,
// usually with a build-time script that reads the Slang JSON and emits C++ code.
using Vertex = util::PackedVertex;

// Push data is small per-draw data sent directly into the command buffer.
// Here it selects the frame camera descriptor and the per-material combined sampled images.
// Texture fields use the combined image-sampler index layout expected by VK_EXT_descriptor_heap.
// There are multiple ways to send data using descriptor heaps.
// We send indices to select descriptors, but we could also send raw data directly.
struct DescriptorHeapDrawPushIndicesAlbedoAndNormal {
    std::uint32_t cameraIndex;
    std::uint32_t albedoTextureIndex;
    std::uint32_t normalTextureIndex;
};

struct DescriptorHeapDrawPushIndicesAlbedo {
    std::uint32_t cameraIndex;
    std::uint32_t albedoTextureIndex;
};

struct DescriptorHeapDrawPushIndicesSolidColor {
    std::uint32_t cameraIndex;
};

// We use the offsets to link mappings with the push data.
static_assert(offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, cameraIndex) == 0);
static_assert(offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, albedoTextureIndex) == sizeof(std::uint32_t));
static_assert(offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, normalTextureIndex) == sizeof(std::uint32_t) * 2U);

static_assert(offsetof(DescriptorHeapDrawPushIndicesAlbedo, cameraIndex) == 0);
static_assert(offsetof(DescriptorHeapDrawPushIndicesAlbedo, albedoTextureIndex) == sizeof(std::uint32_t));
static_assert(offsetof(DescriptorHeapDrawPushIndicesSolidColor, cameraIndex) == 0);
} // namespace siggraph
