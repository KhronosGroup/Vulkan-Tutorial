#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_hpp_macros.hpp>
#include <vulkan/vk_platform.h>
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <algorithm>
#include <memory>
#include <future>
#include <unordered_set>
#include <condition_variable>
#include <atomic>

#include "platform.h"
#include "entity.h"
#include "mesh_component.h"
#include "camera_component.h"
#include "memory_pool.h"
#include "model_loader.h"
#include "thread_pool.h"

// Forward declarations
class ImGuiSystem;

/**
 * @brief Structure for Vulkan queue family indices.
 */
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> transferFamily; // optional dedicated transfer queue family

    [[nodiscard]] bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
    }
};

/**
 * @brief Structure for swap chain support details.
 */
struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

/**
 * @brief Structure for individual light data in the storage buffer.
 */
struct LightData {
    alignas(16) glm::vec4 position;      // Light position (w component used for direction vs position)
    alignas(16) glm::vec4 color;         // Light color and intensity
    alignas(16) glm::mat4 lightSpaceMatrix; // Light space matrix for shadow mapping
    alignas(4) int lightType;            // 0=Point, 1=Directional, 2=Spot, 3=Emissive
    alignas(4) float range;              // Light range
    alignas(4) float innerConeAngle;     // For spotlights
    alignas(4) float outerConeAngle;     // For spotlights
};

/**
 * @brief Structure for the uniform buffer object (now without fixed light arrays).
 */
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 camPos;
    alignas(4) float exposure;
    alignas(4) float gamma;
    alignas(4) float prefilteredCubeMipLevels;
    alignas(4) float scaleIBLAmbient;
    alignas(4) int lightCount;
    alignas(4) int padding0;    // match shader UBO layout
    alignas(4) float padding1;  // match shader UBO layout
    alignas(4) float padding2;  // match shader UBO layout
    alignas(8) glm::vec2 screenDimensions;
};


/**
 * @brief Structure for PBR material properties.
 * This structure must match the PushConstants structure in the PBR shader.
 */
struct MaterialProperties {
    alignas(16) glm::vec4 baseColorFactor;
    alignas(4) float metallicFactor;
    alignas(4) float roughnessFactor;
    alignas(4) int baseColorTextureSet;
    alignas(4) int physicalDescriptorTextureSet;
    alignas(4) int normalTextureSet;
    alignas(4) int occlusionTextureSet;
    alignas(4) int emissiveTextureSet;
    alignas(4) float alphaMask;
    alignas(4) float alphaMaskCutoff;
    alignas(16) glm::vec3 emissiveFactor;  // Emissive factor for HDR emissive sources
    alignas(4) float emissiveStrength;     // KHR_materials_emissive_strength extension
    alignas(4) float transmissionFactor;   // KHR_materials_transmission
    alignas(4) int useSpecGlossWorkflow;   // 1 if using KHR_materials_pbrSpecularGlossiness
    alignas(4) float glossinessFactor;     // SpecGloss glossiness scalar
    alignas(16) glm::vec3 specularFactor;  // SpecGloss specular color factor
    alignas(4) float ior = 1.5f; // index of refraction
    alignas(4) bool hasEmissiveStrengthExtension;
};

/**
 * @brief Class for managing Vulkan rendering.
 *
 * This class implements the rendering pipeline as described in the Engine_Architecture chapter:
 * @see en/Building_a_Simple_Engine/Engine_Architecture/05_rendering_pipeline.adoc
 */
class Renderer {
public:
    /**
     * @brief Constructor with a platform.
     * @param platform The platform to use for rendering.
     */
    explicit Renderer(Platform* platform);

    /**
     * @brief Destructor for proper cleanup.
     */
    ~Renderer();

    /**
     * @brief Initialize the renderer.
     * @param appName The name of the application.
     * @param enableValidationLayers Whether to enable validation layers.
     * @return True if initialization was successful, false otherwise.
     */
    bool Initialize(const std::string& appName, bool enableValidationLayers = true);

    /**
     * @brief Clean up renderer resources.
     */
    void Cleanup();

    /**
     * @brief Render the scene.
     * @param entities The entities to render.
     * @param camera The camera to use for rendering.
     * @param imguiSystem The ImGui system for UI rendering (optional).
     */
    void Render(const std::vector<std::unique_ptr<Entity>>& entities, CameraComponent* camera, ImGuiSystem* imguiSystem = nullptr);

    /**
     * @brief Wait for the device to be idle.
     */
    void WaitIdle();

    /**
     * @brief Dispatch a compute shader.
     * @param groupCountX The number of local workgroups to dispatch in the X dimension.
     * @param groupCountY The number of local workgroups to dispatch in the Y dimension.
     * @param groupCountZ The number of local workgroups to dispatch in the Z dimension.
     * @param inputBuffer The input buffer.
     * @param outputBuffer The output buffer.
     * @param hrtfBuffer The HRTF data buffer.
     * @param paramsBuffer The parameters buffer.
     * @return A fence that can be used to synchronize with the compute operation.
     */
    vk::raii::Fence DispatchCompute(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
                                   vk::Buffer inputBuffer, vk::Buffer outputBuffer,
                                   vk::Buffer hrtfBuffer, vk::Buffer paramsBuffer);

    /**
     * @brief Check if the renderer is initialized.
     * @return True if the renderer is initialized, false otherwise.
     */
    bool IsInitialized() const { return initialized; }



    /**
     * @brief Get the Vulkan device.
     * @return The Vulkan device.
     */
    vk::Device GetDevice() const { return *device; }

    // Expose max frames in flight for per-frame resource duplication
    uint32_t GetMaxFramesInFlight() const { return MAX_FRAMES_IN_FLIGHT; }

    /**
     * @brief Get the Vulkan RAII device.
     * @return The Vulkan RAII device.
     */
    const vk::raii::Device& GetRaiiDevice() const { return device; }

    // Expose uploads timeline semaphore and last value for external waits
    vk::Semaphore GetUploadsTimelineSemaphore() const { return *uploadsTimeline; }
    uint64_t GetUploadsTimelineValue() const { return uploadTimelineLastSubmitted.load(std::memory_order_relaxed); }

    /**
     * @brief Get the compute queue.
     * @return The compute queue.
     */
    vk::Queue GetComputeQueue() const {
        std::lock_guard<std::mutex> lock(queueMutex);
        return *computeQueue;
    }

    /**
     * @brief Find a suitable memory type.
     * @param typeFilter The type filter.
     * @param properties The memory properties.
     * @return The memory type index.
     */
    uint32_t FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        return findMemoryType(typeFilter, properties);
    }

    /**
     * @brief Get the compute queue family index.
     * @return The compute queue family index.
     */
    uint32_t GetComputeQueueFamilyIndex() const {
        if (queueFamilyIndices.computeFamily.has_value()) {
            return queueFamilyIndices.computeFamily.value();
        }
        // Fallback to graphics family to avoid crashes on devices without a separate compute queue
        return queueFamilyIndices.graphicsFamily.value();
    }

    /**
     * @brief Submit a command buffer to the compute queue with proper dispatch loader preservation.
     * @param commandBuffer The command buffer to submit.
     * @param fence The fence to signal when the operation completes.
     */
    void SubmitToComputeQueue(vk::CommandBuffer commandBuffer, vk::Fence fence) const {
        // Use mutex to ensure thread-safe access to queues
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer
        };
        std::lock_guard<std::mutex> lock(queueMutex);
        // Prefer compute queue when available; otherwise, fall back to graphics queue to avoid crashes
        if (*computeQueue) {
            computeQueue.submit(submitInfo, fence);
        } else {
            graphicsQueue.submit(submitInfo, fence);
        }
    }

    /**
     * @brief Create a shader module from SPIR-V code.
     * @param code The SPIR-V code.
     * @return The shader module.
     */
    vk::raii::ShaderModule CreateShaderModule(const std::vector<char>& code) {
        return createShaderModule(code);
    }

    /**
     * @brief Create a shader module from a file.
     * @param filename The filename.
     * @return The shader module.
     */
    vk::raii::ShaderModule CreateShaderModule(const std::string& filename) {
        auto code = readFile(filename);
        return createShaderModule(code);
    }

    /**
     * @brief Load a texture from a file.
     * @param texturePath The path to the texture file.
     * @return True if the texture was loaded successfully, false otherwise.
     */
    bool LoadTexture(const std::string& texturePath);

    // Asynchronous texture loading APIs (thread-pool backed).
    // The 'critical' flag is used to front-load important textures (e.g.,
    // baseColor/albedo) so the scene looks mostly correct before the loading
    // screen disappears. Non-critical textures (normals, MR, AO, emissive)
    // can stream in after geometry is visible.
    std::future<bool> LoadTextureAsync(const std::string& texturePath, bool critical = false);

    /**
     * @brief Load a texture from raw image data in memory.
     * @param textureId The identifier for the texture.
     * @param imageData The raw image data.
     * @param width The width of the image.
     * @param height The height of the image.
     * @param channels The number of channels in the image.
     * @return True if the texture was loaded successfully, false otherwise.
     */
    bool LoadTextureFromMemory(const std::string& textureId, const unsigned char* imageData,
                              int width, int height, int channels);

    // Asynchronous upload from memory (RGBA/RGB/other). Safe for concurrent calls.
    std::future<bool> LoadTextureFromMemoryAsync(const std::string& textureId, const unsigned char* imageData,
                              int width, int height, int channels, bool critical = false);

    // Progress query for UI
    uint32_t GetTextureTasksScheduled() const { return textureTasksScheduled.load(); }
    uint32_t GetTextureTasksCompleted() const { return textureTasksCompleted.load(); }

    // GPU upload progress (per-texture jobs processed on the main thread).
    uint32_t GetUploadJobsTotal() const { return uploadJobsTotal.load(); }
    uint32_t GetUploadJobsCompleted() const { return uploadJobsCompleted.load(); }

    // Block until all currently-scheduled texture tasks have completed.
    // Intended for use during initial scene loading so that descriptor
    // creation sees the final textureResources instead of fallbacks.
    void WaitForAllTextureTasks();

    // Process pending texture GPU uploads on the calling thread.
    // This should be invoked from the main/render thread so that all
    // Vulkan work happens from a single thread while worker threads
    // perform only CPU-side decoding.
    //
    // Parameters allow us to:
    //  - limit the number of jobs processed per call (for streaming), and
    //  - choose whether to include critical and/or non-critical jobs.
    void ProcessPendingTextureJobs(uint32_t maxJobs = UINT32_MAX,
                                   bool includeCritical = true,
                                   bool includeNonCritical = true);

    // Track which entities use a given texture ID so that descriptor sets
    // can be refreshed when textures finish streaming in.
    void RegisterTextureUser(const std::string& textureId, Entity* entity);
    void OnTextureUploaded(const std::string& textureId);

    // Global loading state (model/scene). Consider the scene "loading" while
    // either the model is being parsed/instantiated OR there are still
    // outstanding critical texture uploads (e.g., baseColor/albedo).
    bool IsLoading() const { return loadingFlag.load() || criticalJobsOutstanding.load() > 0; }
    void SetLoading(bool v) { loadingFlag.store(v); }

    // Texture aliasing: map canonical IDs to actual loaded keys (e.g., file paths) to avoid duplicates
    inline void RegisterTextureAlias(const std::string& aliasId, const std::string& targetId) {
        std::unique_lock<std::shared_mutex> lock(textureResourcesMutex);
        if (aliasId.empty() || targetId.empty()) return;
        // Resolve targetId without re-locking by walking the alias map directly
        std::string resolved = targetId;
        for (int i = 0; i < 8; ++i) {
            auto it = textureAliases.find(resolved);
            if (it == textureAliases.end()) break;
            if (it->second == resolved) break;
            resolved = it->second;
        }
        if (aliasId == resolved) {
            textureAliases.erase(aliasId);
        } else {
            textureAliases[aliasId] = resolved;
        }
    }
    inline std::string ResolveTextureId(const std::string& id) const {
        std::shared_lock<std::shared_mutex> lock(textureResourcesMutex);
        std::string cur = id;
        for (int i = 0; i < 8; ++i) { // prevent pathological cycles
            auto it = textureAliases.find(cur);
            if (it == textureAliases.end()) break;
            if (it->second == cur) break; // self-alias guard
            cur = it->second;
        }
        return cur;
    }

    /**
     * @brief Transition an image layout.
     * @param image The image.
     * @param format The image format.
     * @param oldLayout The old layout.
     * @param newLayout The new layout.
     */
    void TransitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        transitionImageLayout(image, format, oldLayout, newLayout);
    }

    /**
     * @brief Copy a buffer to an image.
     * @param buffer The buffer.
     * @param image The image.
     * @param width The image width.
     * @param height The image height.
     */
    void CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) const {
        // Create a default single region for backward compatibility
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
        copyBufferToImage(buffer, image, width, height, regions);
    }

    /**
     * @brief Get the current command buffer.
     * @return The current command buffer.
     */
    vk::raii::CommandBuffer& GetCurrentCommandBuffer() {
        return commandBuffers[currentFrame];
    }

    /**
     * @brief Get the swap chain image format.
     * @return The swap chain image format.
     */
    vk::Format GetSwapChainImageFormat() const {
        return swapChainImageFormat;
    }

    /**
     * @brief Set the framebuffer resized flag.
     * This should be called when the window is resized to trigger swap chain recreation.
     */
    void SetFramebufferResized() {
        framebufferResized.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Set the model loader reference for accessing extracted lights.
     * @param _modelLoader Pointer to the model loader.
     */
    void SetModelLoader(ModelLoader* _modelLoader) {
        modelLoader = _modelLoader;
    }

    /**
     * @brief Set static lights loaded during model initialization.
     * @param lights The lights to store statically.
     */
    void SetStaticLights(const std::vector<ExtractedLight>& lights) { staticLights = lights; }

    /**
     * @brief Set the gamma correction value for PBR rendering.
     * @param _gamma The gamma correction value (typically 2.2).
     */
    void SetGamma(float _gamma) {
        gamma = _gamma;
    }

    /**
     * @brief Set the exposure value for HDR tone mapping.
     * @param _exposure The exposure value (1.0 = no adjustment).
     */
    void SetExposure(float _exposure) {
        exposure = _exposure;
    }

    /**
     * @brief Create or resize light storage buffers to accommodate the given number of lights.
     * @param lightCount The number of lights to accommodate.
     * @return True if successful, false otherwise.
     */
    bool createOrResizeLightStorageBuffers(size_t lightCount);

    /**
     * @brief Update the light storage buffer with current light data.
     * @param frameIndex The current frame index.
     * @param lights The light data to upload.
     * @return True if successful, false otherwise.
     */
    bool updateLightStorageBuffer(uint32_t frameIndex, const std::vector<ExtractedLight>& lights);

    /**
     * @brief Update all existing descriptor sets with new light storage buffer references.
     * Called when light storage buffers are recreated to ensure descriptor sets reference valid buffers.
     */
    void updateAllDescriptorSetsWithNewLightBuffers();

    // Upload helper: record both layout transitions and the copy in a single submit with a fence
    void uploadImageFromStaging(vk::Buffer staging,
                                vk::Image image,
                                vk::Format format,
                                const std::vector<vk::BufferImageCopy>& regions,
                                uint32_t mipLevels = 1);

    vk::Format findDepthFormat();

    /**
     * @brief Pre-allocate all Vulkan resources for an entity during scene loading.
     * @param entity The entity to pre-allocate resources for.
     * @return True if pre-allocation was successful, false otherwise.
     */
    bool preAllocateEntityResources(Entity* entity);

    /**
     * @brief Pre-allocate Vulkan resources for a batch of entities, batching mesh uploads.
     *
     * This variant is optimized for large scene loads (e.g., GLTF Bistro). It will:
     *  - Create per-mesh GPU buffers as usual, but record all buffer copy commands
     *    into a single command buffer and submit them in one batch.
     *  - Then create uniform buffers and descriptor sets per entity.
     *
     * Callers that load many geometry entities at once (like GLTF scene loading)
     * should prefer this over repeated preAllocateEntityResources() calls.
     */
    bool preAllocateEntityResourcesBatch(const std::vector<Entity*>& entities);

    // Shared default PBR texture identifiers (to avoid creating hundreds of identical textures)
    static const std::string SHARED_DEFAULT_ALBEDO_ID;
    static const std::string SHARED_DEFAULT_NORMAL_ID;
    static const std::string SHARED_DEFAULT_METALLIC_ROUGHNESS_ID;
    static const std::string SHARED_DEFAULT_OCCLUSION_ID;
    static const std::string SHARED_DEFAULT_EMISSIVE_ID;
    static const std::string SHARED_BRIGHT_RED_ID;

    /**
     * @brief Determine the appropriate texture format based on the texture type.
     * @param textureId The texture identifier to analyze.
     * @return The appropriate Vulkan format (sRGB for baseColor, linear for others).
     */
    static vk::Format determineTextureFormat(const std::string& textureId);

private:
    // Platform
    Platform* platform = nullptr;

    // Model loader reference for accessing extracted lights
    class ModelLoader* modelLoader = nullptr;

    // PBR rendering parameters
    float gamma = 2.2f;     // Gamma correction value
    float exposure = 1.2f;  // HDR exposure value (default tuned to avoid washout)

    // Vulkan RAII context
    vk::raii::Context context;

    // Vulkan instance and debug messenger
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    // Vulkan device
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    // Memory pool for efficient memory management
    std::unique_ptr<MemoryPool> memoryPool;

    // Vulkan queues
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;
    vk::raii::Queue computeQueue = nullptr;

    // Vulkan surface
    vk::raii::SurfaceKHR surface = nullptr;

    // Swap chain
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent = {0, 0};
    std::vector<vk::raii::ImageView> swapChainImageViews;

    // Dynamic rendering info
    vk::RenderingInfo renderingInfo;
    std::vector<vk::RenderingAttachmentInfo> colorAttachments;
    vk::RenderingAttachmentInfo depthAttachment;

    // Pipelines
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::PipelineLayout pbrPipelineLayout = nullptr;
    vk::raii::Pipeline pbrGraphicsPipeline = nullptr;
    vk::raii::Pipeline pbrBlendGraphicsPipeline = nullptr;
    // Specialized pipeline for architectural glass (windows, lamp glass, etc.).
    // Shares descriptor layouts and vertex input with the PBR pipelines but uses
    // a dedicated fragment shader entry point for more stable glass shading.
    vk::raii::Pipeline glassGraphicsPipeline = nullptr;
    vk::raii::PipelineLayout lightingPipelineLayout = nullptr;
    vk::raii::Pipeline lightingPipeline = nullptr;

    // Pipeline rendering create info structures (for proper lifetime management)
    vk::PipelineRenderingCreateInfo mainPipelineRenderingCreateInfo;
    vk::PipelineRenderingCreateInfo pbrPipelineRenderingCreateInfo;
    vk::PipelineRenderingCreateInfo lightingPipelineRenderingCreateInfo;

    // Compute pipeline
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;
    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::DescriptorPool computeDescriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;
    vk::raii::CommandPool computeCommandPool = nullptr;

    // Thread safety for queue access - unified mutex since queues may share the same underlying VkQueue
    mutable std::mutex queueMutex;

    // Command pool and buffers
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    // Protect usage of shared commandPool for transient command buffers
    mutable std::mutex commandMutex;

    // Dedicated transfer queue (falls back to graphics if unavailable)
    vk::raii::Queue transferQueue = nullptr;

    // Synchronization objects
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    // Upload timeline semaphore for transfer -> graphics handoff (signaled per upload)
    vk::raii::Semaphore uploadsTimeline = nullptr;
    // Tracks last timeline value that has been submitted for signaling on uploadsTimeline
    std::atomic<uint64_t> uploadTimelineLastSubmitted{0};

    // Depth buffer
    vk::raii::Image depthImage = nullptr;
    std::unique_ptr<MemoryPool::Allocation> depthImageAllocation = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    // Descriptor set layouts (declared before pools and sets)
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout pbrDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout transparentDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pbrTransparentPipelineLayout = nullptr;

    // The texture that will hold a snapshot of the opaque scene
    vk::raii::Image opaqueSceneColorImage{nullptr};
    vk::raii::DeviceMemory opaqueSceneColorImageMemory{nullptr}; // <-- Standard Vulkan memory
    vk::raii::ImageView opaqueSceneColorImageView{nullptr};
    vk::raii::Sampler opaqueSceneColorSampler{nullptr};

    // A descriptor set for the opaque scene color texture. We will have one for each frame in flight
    // to match the swapchain images.
    std::vector<vk::raii::DescriptorSet> transparentDescriptorSets;
    // Fallback descriptor sets for opaque pass (binds a default SHADER_READ_ONLY texture as Set 1)
    std::vector<vk::raii::DescriptorSet> transparentFallbackDescriptorSets;

    // Mesh resources
    struct MeshResources {
        // Device-local vertex/index buffers used for rendering
        vk::raii::Buffer vertexBuffer = nullptr;
        std::unique_ptr<MemoryPool::Allocation> vertexBufferAllocation = nullptr;
        vk::raii::Buffer indexBuffer = nullptr;
        std::unique_ptr<MemoryPool::Allocation> indexBufferAllocation = nullptr;
        uint32_t indexCount = 0;

        // Optional per-mesh staging buffers used when uploads are batched.
        // These are populated when createMeshResources(..., deferUpload=true) is used
        // and are consumed and cleared by preAllocateEntityResourcesBatch().
        vk::raii::Buffer stagingVertexBuffer = nullptr;
        vk::raii::DeviceMemory stagingVertexBufferMemory = nullptr;
        vk::DeviceSize vertexBufferSizeBytes = 0;

        vk::raii::Buffer stagingIndexBuffer = nullptr;
        vk::raii::DeviceMemory stagingIndexBufferMemory = nullptr;
        vk::DeviceSize indexBufferSizeBytes = 0;
    };
    std::unordered_map<MeshComponent*, MeshResources> meshResources;

    // Texture resources
    struct TextureResources {
        vk::raii::Image textureImage = nullptr;
        std::unique_ptr<MemoryPool::Allocation> textureImageAllocation = nullptr;
        vk::raii::ImageView textureImageView = nullptr;
        vk::raii::Sampler textureSampler = nullptr;
        vk::Format format = vk::Format::eR8G8B8A8Srgb; // Store texture format for proper color space handling
        uint32_t mipLevels = 1; // Store number of mipmap levels
        // Hint: true if source texture appears to use alpha masking (any alpha < ~1.0)
        bool alphaMaskedHint = false;
    };
    std::unordered_map<std::string, TextureResources> textureResources;

    // Pending texture jobs that require GPU-side work. Worker threads
    // enqueue these jobs; the main thread drains them and performs the
    // actual LoadTexture/LoadTextureFromMemory calls.
    struct PendingTextureJob {
        enum class Type { FromFile, FromMemory } type;
        enum class Priority { Critical, NonCritical } priority;
        std::string idOrPath;
        std::vector<unsigned char> data; // only used for FromMemory
        int width = 0;
        int height = 0;
        int channels = 0;
    };

    std::mutex pendingTextureJobsMutex;
    std::vector<PendingTextureJob> pendingTextureJobs;
    // Track outstanding critical texture jobs (for IsLoading)
    std::atomic<uint32_t> criticalJobsOutstanding{0};

    // Track how many texture upload jobs have been scheduled vs completed
    // on the GPU side. Used only for UI feedback during streaming.
    std::atomic<uint32_t> uploadJobsTotal{0};
    std::atomic<uint32_t> uploadJobsCompleted{0};

    // Reverse mapping from texture ID to entities that reference it. Used to
    // update descriptor sets when a streamed texture finishes uploading.
    std::mutex textureUsersMutex;
    std::unordered_map<std::string, std::vector<Entity*>> textureToEntities;

    // Protect concurrent access to textureResources
    mutable std::shared_mutex textureResourcesMutex;

    // Texture aliasing: maps alias (canonical) IDs to actual loaded keys
    std::unordered_map<std::string, std::string> textureAliases;

    // Per-texture load de-duplication (serialize loads of the same texture ID only)
    mutable std::mutex textureLoadStateMutex;
    std::condition_variable textureLoadStateCv;
    std::unordered_set<std::string> texturesLoading;

    // Serialize GPU-side texture upload (image/buffer creation, transitions) to avoid driver/memory pool races
    mutable std::mutex textureUploadMutex;

    // Thread pool for background background tasks (textures, etc.)
    std::unique_ptr<ThreadPool> threadPool;
    // Mutex to protect threadPool access during initialization/cleanup
    mutable std::shared_mutex threadPoolMutex;

    // Texture loading progress (for UI)
    std::atomic<uint32_t> textureTasksScheduled{0};
    std::atomic<uint32_t> textureTasksCompleted{0};
    std::atomic<bool> loadingFlag{false};

    // Default texture resources (used when no texture is provided)
    TextureResources defaultTextureResources;

    // Performance clamps (to reduce per-frame cost)
    static constexpr uint32_t MAX_ACTIVE_LIGHTS = 1024;      // Limit the number of lights processed per frame

    // Static lights loaded during model initialization
    std::vector<ExtractedLight> staticLights;


    // Dynamic lighting system using storage buffers
    struct LightStorageBuffer {
        vk::raii::Buffer buffer = nullptr;
        std::unique_ptr<MemoryPool::Allocation> allocation = nullptr;
        void* mapped = nullptr;
        size_t capacity = 0;  // Current capacity in number of lights
        size_t size = 0;      // Current number of lights
    };
    std::vector<LightStorageBuffer> lightStorageBuffers; // One per frame in flight

    // Entity resources (contains descriptor sets - must be declared before descriptor pool)
    struct EntityResources {
        std::vector<vk::raii::Buffer> uniformBuffers;
        std::vector<std::unique_ptr<MemoryPool::Allocation>> uniformBufferAllocations;
        std::vector<void*> uniformBuffersMapped;
        std::vector<vk::raii::DescriptorSet> basicDescriptorSets;  // For basic pipeline
        std::vector<vk::raii::DescriptorSet> pbrDescriptorSets;    // For PBR pipeline

        // Instance buffer for instanced rendering
        vk::raii::Buffer instanceBuffer = nullptr;
        std::unique_ptr<MemoryPool::Allocation> instanceBufferAllocation = nullptr;
        void* instanceBufferMapped = nullptr;
    };
    std::unordered_map<Entity*, EntityResources> entityResources;

    // Descriptor pool (declared after entity resources to ensure proper destruction order)
    vk::raii::DescriptorPool descriptorPool = nullptr;

    // Current frame index
    uint32_t currentFrame = 0;

    // Queue family indices
    QueueFamilyIndices queueFamilyIndices;

    // Validation layers
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    // Required device extensions
    const std::vector<const char*> requiredDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    // Optional device extensions
    const std::vector<const char*> optionalDeviceExtensions = {
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
        VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME
    };

    // All device extensions (required + optional)
    std::vector<const char*> deviceExtensions;

    // Initialization flag
    bool initialized = false;

    // Framebuffer resized flag (atomic to handle platform callback vs. render thread)
    std::atomic<bool> framebufferResized{false};

    // Maximum number of frames in flight
    const uint32_t MAX_FRAMES_IN_FLIGHT = 2u;

    // Private methods
    bool createInstance(const std::string& appName, bool enableValidationLayers);
    bool setupDebugMessenger(bool enableValidationLayers);
    bool createSurface();
    bool checkValidationLayerSupport() const;
    bool pickPhysicalDevice();
    void addSupportedOptionalExtensions();
    bool createLogicalDevice(bool enableValidationLayers);
    bool createSwapChain();
    bool createImageViews();
    bool setupDynamicRendering();
    bool createDescriptorSetLayout();
    bool createPBRDescriptorSetLayout();
    bool createGraphicsPipeline();

    bool createPBRPipeline();
    bool createLightingPipeline();
    bool createComputePipeline();
    void pushMaterialProperties(vk::CommandBuffer commandBuffer, const MaterialProperties& material) const;
    bool createCommandPool();

    // Shadow mapping methods
    bool createComputeCommandPool();
    bool createDepthResources();
    bool createTextureImage(const std::string& texturePath, TextureResources& resources);
    bool createTextureImageView(TextureResources& resources);
    bool createTextureSampler(TextureResources& resources);
    bool createDefaultTextureResources();
    bool createSharedDefaultPBRTextures();
    bool createMeshResources(MeshComponent* meshComponent, bool deferUpload = false);
    bool createUniformBuffers(Entity* entity);
    bool createDescriptorPool();
    bool createDescriptorSets(Entity* entity, const std::string& texturePath, bool usePBR = false);
    bool createCommandBuffers();
    bool createSyncObjects();

    void cleanupSwapChain();

    // Ensure Vulkan-Hpp dispatcher is initialized for the current thread when using RAII objects on worker threads
    void ensureThreadLocalVulkanInit() const;
    void recreateSwapChain();

    void updateUniformBuffer(uint32_t currentImage, Entity* entity, CameraComponent* camera);
    void updateUniformBuffer(uint32_t currentImage, Entity* entity, CameraComponent* camera, const glm::mat4& customTransform);
    void updateUniformBufferInternal(uint32_t currentImage, Entity* entity, CameraComponent* camera, UniformBufferObject& ubo);

    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code);

    QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device);
    SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device);
    bool isDeviceSuitable(vk::raii::PhysicalDevice& device);
    bool checkDeviceExtensionSupport(vk::raii::PhysicalDevice& device);

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const;

    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);
    bool createOpaqueSceneColorResources();
    void createTransparentDescriptorSets();
    void createTransparentFallbackDescriptorSets();
    std::pair<vk::raii::Buffer, std::unique_ptr<MemoryPool::Allocation>> createBufferPooled(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);
    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);

    std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties);
    std::pair<vk::raii::Image, std::unique_ptr<MemoryPool::Allocation>> createImagePooled(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, uint32_t mipLevels = 1);
    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels = 1);
    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height, const std::vector<vk::BufferImageCopy>& regions) const;

    vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels = 1);
    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    bool hasStencilComponent(vk::Format format);

    std::vector<char> readFile(const std::string& filename);
};
