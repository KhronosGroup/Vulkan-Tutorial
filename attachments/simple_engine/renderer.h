#pragma once

#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <vulkan/vulkan_hpp_macros.hpp>
#include <vulkan/vk_platform.h>
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>

#include "platform.h"
#include "entity.h"
#include "mesh_component.h"
#include "camera_component.h"

// Forward declarations
class ImGuiSystem;

/**
 * @brief Structure for Vulkan queue family indices.
 */
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;

    bool isComplete() const {
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
 * @brief Structure for uniform buffer object.
 */
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 lightPos;
    alignas(16) glm::vec4 lightColor;
    alignas(16) glm::vec4 viewPos;
};

/**
 * @brief Structure for material properties.
 */
struct MaterialProperties {
    alignas(16) glm::vec4 ambientColor;
    alignas(16) glm::vec4 diffuseColor;
    alignas(16) glm::vec4 specularColor;
    alignas(4) float shininess;
    alignas(4) float padding[3]; // Padding to ensure alignment
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
    void Render(const std::vector<Entity*>& entities, CameraComponent* camera, ImGuiSystem* imguiSystem = nullptr);

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
     */
    void DispatchCompute(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
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

    /**
     * @brief Get the Vulkan RAII device.
     * @return The Vulkan RAII device.
     */
    const vk::raii::Device& GetRaiiDevice() const { return device; }

    /**
     * @brief Get the compute queue.
     * @return The compute queue.
     */
    vk::Queue GetComputeQueue() const { return *computeQueue; }

    /**
     * @brief Find a suitable memory type.
     * @param typeFilter The type filter.
     * @param properties The memory properties.
     * @return The memory type index.
     */
    uint32_t FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        return findMemoryType(typeFilter, properties);
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
    void CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        copyBufferToImage(buffer, image, width, height);
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

private:
    // Platform
    Platform* platform = nullptr;

    // Vulkan RAII context
    vk::raii::Context context;

    // Vulkan instance and debug messenger
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    // Vulkan device
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

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

    // Command pool and buffers
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    // Synchronization objects
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    // Depth buffer
    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    // Descriptor set layouts (declared before pools and sets)
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout pbrDescriptorSetLayout = nullptr;

    // Mesh resources
    struct MeshResources {
        vk::raii::Buffer vertexBuffer = nullptr;
        vk::raii::DeviceMemory vertexBufferMemory = nullptr;
        vk::raii::Buffer indexBuffer = nullptr;
        vk::raii::DeviceMemory indexBufferMemory = nullptr;
        uint32_t indexCount = 0;
    };
    std::unordered_map<MeshComponent*, MeshResources> meshResources;

    // Texture resources
    struct TextureResources {
        vk::raii::Image textureImage = nullptr;
        vk::raii::DeviceMemory textureImageMemory = nullptr;
        vk::raii::ImageView textureImageView = nullptr;
        vk::raii::Sampler textureSampler = nullptr;
    };
    std::unordered_map<std::string, TextureResources> textureResources;

    // Default texture resources (used when no texture is provided)
    TextureResources defaultTextureResources;

    // Entity resources (contains descriptor sets - must be declared before descriptor pool)
    struct EntityResources {
        std::vector<vk::raii::Buffer> uniformBuffers;
        std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
        std::vector<void*> uniformBuffersMapped;
        std::vector<vk::raii::DescriptorSet> descriptorSets;
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
        VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME
    };

    // All device extensions (required + optional)
    std::vector<const char*> deviceExtensions;

    // Initialization flag
    bool initialized = false;

    // Framebuffer resized flag
    bool framebufferResized = false;

    // Maximum number of frames in flight
    const uint32_t MAX_FRAMES_IN_FLIGHT = 2u;

    // Private methods
    bool createInstance(const std::string& appName, bool enableValidationLayers);
    bool setupDebugMessenger(bool enableValidationLayers);
    bool createSurface();
    bool checkValidationLayerSupport();
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
    void pushMaterialProperties(vk::CommandBuffer commandBuffer, const MaterialProperties& material);
    bool createCommandPool();
    bool createDepthResources();
    bool createTextureImage(const std::string& texturePath, TextureResources& resources);
    bool createTextureImageView(TextureResources& resources);
    bool createTextureSampler(TextureResources& resources);
    bool createDefaultTextureResources();
    bool createMeshResources(MeshComponent* meshComponent);
    bool createUniformBuffers(Entity* entity);
    bool createDescriptorPool();
    bool createDescriptorSets(Entity* entity, const std::string& texturePath);
    bool createCommandBuffers();
    bool createSyncObjects();

    void cleanupSwapChain();
    void recreateSwapChain();

    void updateUniformBuffer(uint32_t currentImage, Entity* entity, CameraComponent* camera);

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
    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);

    std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties);
    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);

    vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags);
    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    vk::Format findDepthFormat();
    bool hasStencilComponent(vk::Format format);

    std::vector<char> readFile(const std::string& filename);
};
