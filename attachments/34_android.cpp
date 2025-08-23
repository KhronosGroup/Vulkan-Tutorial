#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>
#include <optional>
#include <assert.h>

#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <vulkan/vk_platform.h>
#if defined(__ANDROID__)
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_android.h>
#endif
#include <vulkan/vulkan_profiles.hpp>

// Platform detection
#if defined(__ANDROID__)
    #define PLATFORM_ANDROID 1
#else
    #define PLATFORM_DESKTOP 1
#endif


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

// Platform-specific includes
#if PLATFORM_ANDROID
    // Android-specific includes
    #include <android/log.h>
    #include <game-activity/native_app_glue/android_native_app_glue.h>
    #include <android/asset_manager.h>
    #include <android/asset_manager_jni.h>

    // Declare and implement app_dummy function from native_app_glue
    extern "C" void app_dummy() {
        // This is a dummy function that does nothing
        // It's used to prevent the linker from stripping out the native_app_glue code
    }

    // Define AAssetManager type for Android
    typedef AAssetManager AssetManagerType;

    // Define logging macros for Android
    #define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "VulkanTutorial", __VA_ARGS__))
    #define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "VulkanTutorial", __VA_ARGS__))
    #define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "VulkanTutorial", __VA_ARGS__))
    #define LOG_INFO(msg) LOGI("%s", msg)
    #define LOG_ERROR(msg) LOGE("%s", msg)
#else
    // Define AAssetManager type for non-Android platforms
    typedef void AssetManagerType;
    // Desktop-specific includes
    #define GLFW_INCLUDE_VULKAN
    #include <GLFW/glfw3.h>

    // Define logging macros for Desktop
    #define LOGI(...) printf(__VA_ARGS__); printf("\n")
    #define LOGW(...) printf(__VA_ARGS__); printf("\n")
    #define LOGE(...) fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")
    #define LOG_INFO(msg) std::cout << msg << std::endl
    #define LOG_ERROR(msg) std::cerr << msg << std::endl
#endif

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CXX11
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint64_t FenceTimeout = 100000000;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

#if PLATFORM_ANDROID
// Define VpProfileProperties structure if not already defined
#ifndef VP_PROFILE_PROPERTIES_DEFINED
#define VP_PROFILE_PROPERTIES_DEFINED
struct VpProfileProperties {
    char name[256];
    uint32_t specVersion;
};
#endif

// Define Vulkan Profile constants
#ifndef VP_KHR_ROADMAP_2022_NAME
#define VP_KHR_ROADMAP_2022_NAME "VP_KHR_roadmap_2022"
#endif

#ifndef VP_KHR_ROADMAP_2022_SPEC_VERSION
#define VP_KHR_ROADMAP_2022_SPEC_VERSION 1
#endif
#endif

// Application info structure to store profile support flags
struct AppInfo {
    bool profileSupported = false;
    VpProfileProperties profile;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
            vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
            vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) )
        };
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

template<> struct std::hash<Vertex> {
    size_t operator()(Vertex const& vertex) const noexcept {
        return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
    }
};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

// Cross-platform file reading function
std::vector<char> readFile(const std::string& filename, std::optional<AssetManagerType*> assetManager = std::nullopt) {
#if PLATFORM_ANDROID
    // On Android, use asset manager if provided
    if (assetManager.has_value() && *assetManager != nullptr) {
        // Open the asset
        AAsset* asset = AAssetManager_open(*assetManager, filename.c_str(), AASSET_MODE_BUFFER);
        if (!asset) {
            LOGE("Failed to open asset: %s", filename.c_str());
            throw std::runtime_error("Failed to open file: " + filename);
        }

        // Get the file size
        off_t fileSize = AAsset_getLength(asset);
        std::vector<char> buffer(fileSize);

        // Read the file data
        AAsset_read(asset, buffer.data(), fileSize);

        // Close the asset
        AAsset_close(asset);

        return buffer;
    }
#endif

    // Desktop version or Android fallback to filesystem
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

// Cross-platform application class
class HelloTriangleApplication {
public:
#if PLATFORM_DESKTOP
    // Desktop constructor
    HelloTriangleApplication() {
        // No Android-specific initialization needed
    }
#else
    // Android constructor
    HelloTriangleApplication(android_app* app) : androidApp(app) {
        androidApp->userData = this;
        androidApp->onAppCmd = handleAppCommand;
        // Note: onInputEvent is no longer a member of android_app in the current NDK version
        // Input events are now handled differently

        // Get the asset manager
        assetManager = androidApp->activity->assetManager;
    }
#endif

    void run() {
#if PLATFORM_DESKTOP
        // Desktop main loop
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
#else
        // Android main loop is handled by Android
        while (!initialized) {
            // Wait for app to initialize
            int events;
            android_poll_source* source;
            if (ALooper_pollOnce(0, nullptr, &events, (void**)&source) >= 0) {
                if (source != nullptr) {
                    source->process(androidApp, source);
                }
            }
        }
#endif
    }

#if PLATFORM_DESKTOP
    // Initialize window (Desktop only)
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Cross-Platform", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

        LOG_INFO("Desktop window created");
    }

    // Desktop main loop
    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    // Desktop framebuffer resize callback
    static void framebufferResizeCallback(GLFWwindow* window, int, int) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
#endif

    void cleanup() {
        if (initialized) {
            // Wait for device to finish operations
            if (*device) {
                device.waitIdle();
            }

            // Cleanup resources
            cleanupSwapChain();

            initialized = false;
        }
    }

private:
#if PLATFORM_ANDROID
    // Android-specific members
    android_app* androidApp = nullptr;
    AssetManagerType* assetManager = nullptr;
#else
    // Desktop-specific members
    GLFWwindow* window = nullptr;
#endif
    bool initialized = false;
    bool framebufferResized = false;

    // Vulkan objects
    vk::raii::Context                context;
    vk::raii::Instance               instance       = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR             surface        = nullptr;
    vk::raii::PhysicalDevice         physicalDevice = nullptr;
    vk::raii::Device                 device         = nullptr;
    uint32_t                         queueIndex     = ~0;
    vk::raii::Queue                  queue          = nullptr;
    vk::raii::SwapchainKHR           swapChain      = nullptr;
    std::vector<vk::Image>           swapChainImages;
    vk::SurfaceFormatKHR             swapChainSurfaceFormat;
    vk::Extent2D                     swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::RenderPass renderPass = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers;
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;
    vk::raii::Image textureImage = nullptr;
    vk::raii::DeviceMemory textureImageMemory = nullptr;
    vk::raii::ImageView textureImageView = nullptr;
    vk::raii::Sampler textureSampler = nullptr;
    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t currentFrame = 0;

    // Application info
    AppInfo appInfo;

    // Model data
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Swap chain support details
    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    // Required device extensions
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    // Initialize Vulkan
    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        checkFeatureSupport();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();

        initialized = true;
    }

    // Create Vulkan instance
    void createInstance() {
        // Application info
        vk::ApplicationInfo appInfo{
            .pApplicationName = "Vulkan Android",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3
        };

        // Get required extensions
        std::vector<const char*> extensions = getRequiredExtensions();

        // Create instance
        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()
        };

        instance = vk::raii::Instance(context, createInfo);
        LOGI("Vulkan instance created");
    }

    // Create platform-specific surface
    void createSurface() {
        VkSurfaceKHR _surface;

#if PLATFORM_ANDROID
        // Create Android surface
        VkAndroidSurfaceCreateInfoKHR createInfo = {
            .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = 0,
            .window = androidApp->window
        };

        VkResult result = vkCreateAndroidSurfaceKHR(
            *instance,
            &createInfo,
            nullptr,
            &_surface
        );

        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create Android surface");
        }

        LOG_INFO("Android surface created");
#else
        // Create desktop surface using GLFW
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("Failed to create window surface");
        }

        LOG_INFO("Desktop surface created");
#endif

        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    // Pick physical device
    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto devIter = std::ranges::find_if(
            devices,
            [&](auto const& device) {
                // Check if any of the queue families support graphics operations
                auto queueFamilies = device.getQueueFamilyProperties();
                bool supportsGraphics =
                    std::ranges::any_of(queueFamilies, [](auto const& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

                // Check if all required device extensions are available
                auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
                bool supportsAllRequiredExtensions =
                    std::ranges::all_of(deviceExtensions,
                        [&availableDeviceExtensions](auto const& requiredDeviceExtension) {
                            return std::ranges::any_of(availableDeviceExtensions,
                                [requiredDeviceExtension](auto const& availableDeviceExtension) {
                                    return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
                                });
                        });

                return supportsGraphics && supportsAllRequiredExtensions;
            });

        if (devIter != devices.end()) {
            physicalDevice = *devIter;

            // Print device information
            vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
            LOGI("Selected GPU: %s", deviceProperties.deviceName.data());
        } else {
            throw std::runtime_error("Failed to find a suitable GPU");
        }
    }

    // Check feature support
    void checkFeatureSupport() {
        // Define the KHR roadmap 2022 profile
        appInfo.profile = {
            VP_KHR_ROADMAP_2022_NAME,
            VP_KHR_ROADMAP_2022_SPEC_VERSION
        };

        // Check if the profile is supported
        VkBool32 supported = VK_FALSE;

#ifdef PLATFORM_ANDROID
        // Create a vp::ProfileDesc from our VpProfileProperties
        vp::ProfileDesc profileDesc = {
            appInfo.profile.name,
            appInfo.profile.specVersion
        };

        // Use vp::GetProfileSupport instead of vpGetPhysicalDeviceProfileSupport
        bool result = vp::GetProfileSupport(
            *physicalDevice,  // Pass the physical device directly
            &profileDesc,     // Pass the profile description
            &supported        // Output parameter for support status
        );
#else
        VkResult vk_result = vpGetPhysicalDeviceProfileSupport(
            *instance,
            *physicalDevice,
            &appInfo.profile,
            &supported
            );
        bool result = vk_result == VK_SUCCESS;
#endif

        if (result && supported == VK_TRUE) {
            appInfo.profileSupported = true;
            LOGI("Using KHR roadmap 2022 profile");
        } else {
            appInfo.profileSupported = false;
            LOGI("Falling back to traditional rendering (profile not supported)");
        }
    }

    // Create logical device
    void createLogicalDevice() {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports both graphics and present
        for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
        {
            if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
                physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
            {
                // found a queue family that supports both graphics and present
                queueIndex = qfpIndex;
                break;
            }
        }
        if (queueIndex == ~0)
        {
            throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
        }

        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority };

        if (appInfo.profileSupported) {
            // Enable required features
            vk::PhysicalDeviceFeatures2 features2;
            vk::PhysicalDeviceFeatures deviceFeatures{};
            deviceFeatures.samplerAnisotropy = VK_TRUE;
            deviceFeatures.sampleRateShading = VK_TRUE;
            features2.features = deviceFeatures;

            // Enable dynamic rendering
            vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures;
            dynamicRenderingFeatures.dynamicRendering = VK_TRUE;
            features2.pNext = &dynamicRenderingFeatures;

            // Create a vk::DeviceCreateInfo with the required features
            vk::DeviceCreateInfo vkDeviceCreateInfo{
                .pNext = &features2,
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &deviceQueueCreateInfo,
                .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
                .ppEnabledExtensionNames = deviceExtensions.data()
            };

            // Create the device with the vk::DeviceCreateInfo
            device = vk::raii::Device(physicalDevice, vkDeviceCreateInfo);
        } else {
            // Fallback to manual device creation
            vk::PhysicalDeviceFeatures deviceFeatures{};
            deviceFeatures.samplerAnisotropy = VK_TRUE;
            deviceFeatures.sampleRateShading = VK_TRUE;

            vk::DeviceCreateInfo createInfo{
                .queueCreateInfoCount = 1,
                .pQueueCreateInfos = &deviceQueueCreateInfo,
                .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
                .ppEnabledExtensionNames = deviceExtensions.data(),
                .pEnabledFeatures = &deviceFeatures
            };

            device = vk::raii::Device(physicalDevice, createInfo);
        }

        queue = device.getQueue(queueIndex, 0);
    }

    // Create swap chain
    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        swapChainExtent                          = chooseSwapExtent( swapChainSupport.capabilities );
        swapChainSurfaceFormat                   = chooseSwapSurfaceFormat( swapChainSupport.formats );
        vk::SwapchainCreateInfoKHR swapChainCreateInfo{ .surface          = *surface,
                                                        .minImageCount    = chooseSwapMinImageCount( swapChainSupport.capabilities ),
                                                        .imageFormat      = swapChainSurfaceFormat.format,
                                                        .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
                                                        .imageExtent      = swapChainExtent,
                                                        .imageArrayLayers = 1,
                                                        .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
                                                        .imageSharingMode = vk::SharingMode::eExclusive,
                                                        .preTransform     = swapChainSupport.capabilities.currentTransform,
                                                        .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                        .presentMode      = chooseSwapPresentMode( swapChainSupport.presentModes ),
                                                        .clipped          = true };

        swapChain = device.createSwapchainKHR(swapChainCreateInfo);
        swapChainImages = swapChain.getImages();
    }

    // Create image views
    void createImageViews() {
        swapChainImageViews.reserve(swapChainImages.size());

        for (const auto& image : swapChainImages) {
            vk::ImageViewCreateInfo createInfo{
                .image = image,
                .viewType = vk::ImageViewType::e2D,
                .format = swapChainSurfaceFormat.format,
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

            swapChainImageViews.push_back(device.createImageView(createInfo));
        }
    }

    // Create render pass
    void createRenderPass() {
        vk::AttachmentDescription colorAttachment{
            .format = swapChainSurfaceFormat.format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::ePresentSrcKHR
        };

        vk::AttachmentReference colorAttachmentRef{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal
        };

        vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef
        };

        vk::SubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
        };

        vk::RenderPassCreateInfo renderPassInfo{
            .attachmentCount = 1,
            .pAttachments = &colorAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency
        };

        renderPass = device.createRenderPass(renderPassInfo);
    }

    // Create descriptor set layout
    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex
        };

        vk::DescriptorSetLayoutBinding samplerLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment
        };

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };

        descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
    }

    // Create graphics pipeline
    void createGraphicsPipeline() {
        // Load shader code from asset files
        LOGI("Loading shaders from assets");

        // Load shader files using cross-platform function
#if PLATFORM_ANDROID
        std::optional<AssetManagerType*> optionalAssetManager = assetManager;
#else
        std::optional<void*> optionalAssetManager = std::nullopt;
#endif
        std::vector<char> vertShaderCode = readFile("shaders/vert.spv", optionalAssetManager);
        std::vector<char> fragShaderCode = readFile("shaders/frag.spv", optionalAssetManager);

        LOGI("Shaders loaded successfully");

        // Create shader modules
        vk::ShaderModuleCreateInfo vertShaderModuleInfo{
            .codeSize = vertShaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(vertShaderCode.data())
        };
        vk::raii::ShaderModule vertShaderModule = device.createShaderModule(vertShaderModuleInfo);

        vk::ShaderModuleCreateInfo fragShaderModuleInfo{
            .codeSize = fragShaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(fragShaderCode.data())
        };
        vk::raii::ShaderModule fragShaderModule = device.createShaderModule(fragShaderModuleInfo);

        // Create shader stages
        vk::PipelineShaderStageCreateInfo shaderStages[] = {
            {
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = *vertShaderModule,
                .pName = "main"
            },
            {
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = *fragShaderModule,
                .pName = "main"
            }
        };

        // Vertex input
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()
        };

        // Input assembly
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE
        };

        // Viewport and scissor
        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };

        // Rasterization
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f
        };

        // Multisampling
        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = VK_FALSE
        };

        // Color blending
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_FALSE,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = VK_FALSE,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        // Dynamic states
        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        // Pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*descriptorSetLayout
        };

        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        // Create the graphics pipeline
        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = nullptr,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = *pipelineLayout,
            .renderPass = *renderPass,
            .subpass = 0
        };

        // Create the pipeline
        graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);
    }

    // Create framebuffers
    void createFramebuffers() {
        swapChainFramebuffers.reserve(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            vk::ImageView attachments[] = {
                *swapChainImageViews[i]
            };

            vk::FramebufferCreateInfo framebufferInfo{
                .renderPass = *renderPass,
                .attachmentCount = 1,
                .pAttachments = attachments,
                .width = swapChainExtent.width,
                .height = swapChainExtent.height,
                .layers = 1
            };

            swapChainFramebuffers.push_back(device.createFramebuffer(framebufferInfo));
        }
    }

    // Create command pool
    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueIndex
        };

        commandPool = device.createCommandPool(poolInfo);
    }

    // Create texture image
    void createTextureImage() {
        // Load texture image
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = nullptr;

#if PLATFORM_ANDROID
        // Load image from Android assets
        std::optional<AssetManagerType*> optionalAssetManager = assetManager;
        std::vector<char> imageData = readFile(TEXTURE_PATH, optionalAssetManager);
        pixels = stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(imageData.data()),
            static_cast<int>(imageData.size()),
            &texWidth, &texHeight, &texChannels, STBI_rgb_alpha
        );
#else
        // Load image from filesystem
        pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
#endif

        if (!pixels) {
            throw std::runtime_error("Failed to load texture image: " + TEXTURE_PATH);
        }

        LOG_INFO("Texture loaded successfully");

        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        // Create staging buffer
        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        // Copy pixel data to staging buffer
        void* data;
        data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        stagingBufferMemory.unmapMemory();

        // Free the pixel data
        if (pixels != nullptr) {
            stbi_image_free(pixels);
        }

        // Create image
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Srgb,
            .extent = {
                .width = static_cast<uint32_t>(texWidth),
                .height = static_cast<uint32_t>(texHeight),
                .depth = 1
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined
        };

        textureImage = device.createImage(imageInfo);

        // Allocate memory for the image
        vk::MemoryRequirements memRequirements = textureImage.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
        };

        textureImageMemory = device.allocateMemory(allocInfo);
        textureImage.bindMemory(*textureImageMemory, 0);

        // Transition image layout and copy buffer to image
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    // Create texture image view
    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
    }

    // Create texture sampler
    void createTextureSampler() {
        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16.0f,
            .compareEnable = VK_FALSE,
            .compareOp = vk::CompareOp::eAlways,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = VK_FALSE
        };

        textureSampler = device.createSampler(samplerInfo);
    }

    // Load model
    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

#if PLATFORM_ANDROID
        // Load OBJ file from Android assets
        std::optional<AssetManagerType*> optionalAssetManager = assetManager;
        std::vector<char> objData = readFile(MODEL_PATH, optionalAssetManager);
        std::string objString(objData.begin(), objData.end());
        std::istringstream objStream(objString);

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, &objStream)) {
            throw std::runtime_error("Failed to load model: " + MODEL_PATH + " - " + warn + err);
        }
#else
        // Load OBJ file from filesystem
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error("Failed to load model: " + MODEL_PATH + " - " + warn + err);
        }
#endif

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = {1.0f, 1.0f, 1.0f};

                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }

        LOG_INFO("Model loaded successfully");
    }

    // Create vertex buffer
    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data;
        data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    }

    // Create index buffer
    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data;
        data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), (size_t) bufferSize);
        stagingBufferMemory.unmapMemory();

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    }

    // Create uniform buffers
    void createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.clear();
        uniformBuffersMemory.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            uniformBuffers.push_back(nullptr);
            uniformBuffersMemory.push_back(nullptr);
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffers[i], uniformBuffersMemory[i]);
        }
    }

    // Create descriptor pool
    void createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
            },
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
            }
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        descriptorPool = device.createDescriptorPool(poolInfo);
    }

    // Create descriptor sets
    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts = layouts.data()
        };

        descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = *uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };

            vk::DescriptorImageInfo imageInfo{
                .sampler = *textureSampler,
                .imageView = *textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &bufferInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &imageInfo
                }
            };

            device.updateDescriptorSets(descriptorWrites, nullptr);
        }
    }

    // Create command buffers
    void createCommandBuffers() {
        commandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);

        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
        };

        commandBuffers = device.allocateCommandBuffers(allocInfo);
    }

    // Create synchronization objects
    void createSyncObjects() {
        imageAvailableSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

        vk::SemaphoreCreateInfo semaphoreInfo{};
        vk::FenceCreateInfo fenceInfo{
            .flags = vk::FenceCreateFlagBits::eSignaled
        };

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores.push_back(device.createSemaphore(semaphoreInfo));
            renderFinishedSemaphores.push_back(device.createSemaphore(semaphoreInfo));
            inFlightFences.push_back(device.createFence(fenceInfo));
        }
    }

    // Clean up swap chain
    void cleanupSwapChain() {
        for (auto& framebuffer : swapChainFramebuffers) {
            framebuffer = nullptr;
        }

        for (auto& imageView : swapChainImageViews) {
            imageView = nullptr;
        }

        swapChain = nullptr;
    }

    // Record command buffer
    void recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex) {
        vk::CommandBufferBeginInfo beginInfo{};
        commandBuffer.begin(beginInfo);

        vk::RenderPassBeginInfo renderPassInfo{
            .renderPass = *renderPass,
            .framebuffer = *swapChainFramebuffers[imageIndex],
            .renderArea = {
                .offset = {0, 0},
                .extent = swapChainExtent
            }
        };

        vk::ClearValue clearColor;
        clearColor.color.float32 = std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

        vk::Viewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(swapChainExtent.width),
            .height = static_cast<float>(swapChainExtent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        };
        commandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = swapChainExtent
        };
        commandBuffer.setScissor(0, scissor);

        commandBuffer.bindVertexBuffers(0, {*vertexBuffer}, {0});
        commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, {*descriptorSets[currentFrame]}, nullptr);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        commandBuffer.endRenderPass();
        commandBuffer.end();
    }

    // Draw frame
    void drawFrame() {
        static_cast<void>(device.waitForFences({*inFlightFences[currentFrame]}, VK_TRUE, FenceTimeout));

        uint32_t imageIndex;
        try {
            auto [result, idx] = swapChain.acquireNextImage(FenceTimeout, *imageAvailableSemaphores[currentFrame]);
            imageIndex = idx;
        } catch (vk::OutOfDateKHRError&) {
            recreateSwapChain();
            return;
        }

        // Update uniform buffer with current transformation
        updateUniformBuffer(currentFrame);

        device.resetFences({*inFlightFences[currentFrame]});

        commandBuffers[currentFrame].reset();
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*imageAvailableSemaphores[currentFrame],
            .pWaitDstStageMask = &waitDestinationStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffers[currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[currentFrame]
        };
        queue.submit(submitInfo, *inFlightFences[currentFrame]);

        const vk::PresentInfoKHR presentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*renderFinishedSemaphores[currentFrame],
            .swapchainCount = 1,
            .pSwapchains = &*swapChain,
            .pImageIndices = &imageIndex
        };

        vk::Result result;
        try {
            result = queue.presentKHR(presentInfoKHR);
        } catch (vk::OutOfDateKHRError&) {
            result = vk::Result::eErrorOutOfDateKHR;
        }

        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present swap chain image");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Recreate swap chain
    void recreateSwapChain() {
        // Wait for device to finish operations
        device.waitIdle();

        // Clean up old swap chain
        cleanupSwapChain();

        // Create new swap chain
        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    // Get required extensions
    std::vector<const char*> getRequiredExtensions() {
#if PLATFORM_ANDROID
        // Android requires these extensions
        std::vector<const char*> extensions = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_ANDROID_SURFACE_EXTENSION_NAME
        };
#else
        // Get the required extensions from GLFW
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
#endif

        // Check if the debug utils extension is available
        std::vector<vk::ExtensionProperties> props = context.enumerateInstanceExtensionProperties();
        bool debugUtilsAvailable = std::ranges::any_of(props,
            [](vk::ExtensionProperties const & ep) {
                return strcmp(ep.extensionName, vk::EXTDebugUtilsExtensionName) == 0;
            });

        // Always include the debug utils extension if available
        if (debugUtilsAvailable) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
#if PLATFORM_DESKTOP
        } else {
            LOG_INFO("VK_EXT_debug_utils extension not available. Validation layers may not work.");
#endif
        }

        return extensions;
    }

    static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const & surfaceCapabilities) {
        auto minImageCount = std::max( 3u, surfaceCapabilities.minImageCount );
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
            minImageCount = surfaceCapabilities.maxImageCount;
        }
        return minImageCount;
    }

    // Choose swap surface format
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        assert(!availableFormats.empty());
        const auto formatIt = std::ranges::find_if(
            availableFormats,
            []( const auto & format ) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; } );
        return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
    }

    // Choose swap present mode
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        assert(std::ranges::any_of(availablePresentModes, [](auto presentMode){ return presentMode == vk::PresentModeKHR::eFifo; }));
        return std::ranges::any_of(availablePresentModes,
            [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; } ) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    // Choose swap extent
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != 0xFFFFFFFF) {
            return capabilities.currentExtent;
        } else {
#if PLATFORM_ANDROID
            // Get the window size from Android
            int32_t width = ANativeWindow_getWidth(androidApp->window);
            int32_t height = ANativeWindow_getHeight(androidApp->window);
#else
            // Get the window size from GLFW
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
#endif

            vk::Extent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width,
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    // Query swap chain support
    SwapChainSupportDetails querySwapChainSupport(vk::raii::PhysicalDevice device) {
        SwapChainSupportDetails details;
        details.capabilities = device.getSurfaceCapabilitiesKHR(*surface);
        details.formats = device.getSurfaceFormatsKHR(*surface);
        details.presentModes = device.getSurfacePresentModesKHR(*surface);
        return details;
    }

    // Create buffer
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        buffer = device.createBuffer(bufferInfo);

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        bufferMemory = device.allocateMemory(allocInfo);
        buffer.bindMemory(*bufferMemory, 0);
    }

    // Copy buffer
    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffer commandBuffer = std::move(device.allocateCommandBuffers(allocInfo)[0]);

        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        commandBuffer.begin(beginInfo);

        vk::BufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = size
        };
        commandBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);

        commandBuffer.end();

        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };

        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    // Find memory type
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    // Create image view
    vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format) {
        vk::ImageViewCreateInfo viewInfo{
            .image = *image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        return device.createImageView(viewInfo);
    }

    // Transition image layout
    void transitionImageLayout(vk::raii::Image& image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffer commandBuffer = std::move(device.allocateCommandBuffers(allocInfo)[0]);

        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        commandBuffer.begin(beginInfo);

        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

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
        } else {
            throw std::invalid_argument("Unsupported layout transition");
        }

        commandBuffer.pipelineBarrier(
            sourceStage, destinationStage,
            vk::DependencyFlagBits::eByRegion,
            nullptr,
            nullptr,
            barrier
        );

        commandBuffer.end();

        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };

        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    // Copy buffer to image
    void copyBufferToImage(vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffer commandBuffer = std::move(device.allocateCommandBuffers(allocInfo)[0]);

        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        commandBuffer.begin(beginInfo);

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

        commandBuffer.copyBufferToImage(
            *buffer,
            *image,
            vk::ImageLayout::eTransferDstOptimal,
            region
        );

        commandBuffer.end();

        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };

        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    // Update uniform buffer
    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        void* data;
        data = uniformBuffersMemory[currentImage].mapMemory(0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        uniformBuffersMemory[currentImage].unmapMemory();
    }

#if PLATFORM_ANDROID
    // Handle app commands
    static void handleAppCommand(android_app* app, int32_t cmd) {
        auto* vulkanApp = static_cast<HelloTriangleApplication*>(app->userData);
        switch (cmd) {
            case APP_CMD_INIT_WINDOW:
                // Window created, initialize Vulkan
                if (app->window != nullptr) {
                    vulkanApp->initVulkan();
                }
                break;
            case APP_CMD_TERM_WINDOW:
                // Window destroyed, clean up Vulkan
                vulkanApp->cleanup();
                break;
            default:
                break;
        }
    }

    // Handle input events
    static int32_t handleInputEvent(android_app* app, AInputEvent* event) {
        auto* vulkanApp = static_cast<HelloTriangleApplication*>(app->userData);
        if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
            // Handle touch events
            float x = AMotionEvent_getX(event, 0);
            float y = AMotionEvent_getY(event, 0);

            // Process touch coordinates
            LOGI("Touch at: %f, %f", x, y);

            return 1;
        }
        return 0;
    }
#endif
};

// Platform-specific entry point
#if PLATFORM_ANDROID
// Android main entry point
void android_main(android_app* app) {
    // Make sure glue isn't stripped
    app_dummy();

    try {
        // Create and run the Vulkan application
        HelloTriangleApplication vulkanApp(app);
        vulkanApp.run();
    } catch (const std::exception& e) {
        LOGE("Exception caught: %s", e.what());
    }
}
#else
// Desktop main entry point
int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
#endif
