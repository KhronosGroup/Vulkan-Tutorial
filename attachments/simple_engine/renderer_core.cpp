#include "renderer.h"
#include <fstream>
#include <iostream>
#include <set>
#include <cstring>
#include <ranges>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE; // In a .cpp file

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_platform.h>

// Debug callback for vk::raii
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallbackVkRaii(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    // Check if this is a shader debug printf message
    if (messageType & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) {
        std::string message(pCallbackData->pMessage);
        if (message.find("DEBUG-PRINTF") != std::string::npos) {
            // This is a shader debug printf message - always show it
            std::cout << "FINDME =====   SHADER DEBUG: " << pCallbackData->pMessage << std::endl;
            return VK_FALSE;
        }
    }

    if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        // Print a message to the console
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

// This implementation corresponds to the Engine_Architecture chapter in the tutorial:
// @see en/Building_a_Simple_Engine/Engine_Architecture/05_rendering_pipeline.adoc

// Constructor
Renderer::Renderer(Platform* platform)
    : platform(platform) {
    // Initialize deviceExtensions with required extensions only
    // Optional extensions will be added later after checking device support
    deviceExtensions = requiredDeviceExtensions;
}

// Destructor
Renderer::~Renderer() {
    Cleanup();
}

// Initialize the renderer
bool Renderer::Initialize(const std::string& appName, bool enableValidationLayers) {
    vk::detail::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    // Create a Vulkan instance
    if (!createInstance(appName, enableValidationLayers)) {
        return false;
    }

    // Setup debug messenger
    if (!setupDebugMessenger(enableValidationLayers)) {
        return false;
    }

    // Create surface
    if (!createSurface()) {
        return false;
    }

    // Pick physical device
    if (!pickPhysicalDevice()) {
        return false;
    }

    // Create logical device
    if (!createLogicalDevice(enableValidationLayers)) {
        return false;
    }

    // Initialize memory pool for efficient memory management
    try {
        memoryPool = std::make_unique<MemoryPool>(device, physicalDevice);
        if (!memoryPool->initialize()) {
            std::cerr << "Failed to initialize memory pool" << std::endl;
            return false;
        }

        // Pre-allocate memory pools to prevent allocation during rendering
        if (!memoryPool->preAllocatePools()) {
            std::cerr << "Failed to pre-allocate memory pools" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to create memory pool: " << e.what() << std::endl;
        return false;
    }

    // Create swap chain
    if (!createSwapChain()) {
        return false;
    }

    // Create image views
    if (!createImageViews()) {
        return false;
    }

    // Setup dynamic rendering
    if (!setupDynamicRendering()) {
        return false;
    }

    // Create descriptor set layout
    if (!createDescriptorSetLayout()) {
        return false;
    }

    // Create graphics pipeline
    if (!createGraphicsPipeline()) {
        return false;
    }

    // Create PBR pipeline
    if (!createPBRPipeline()) {
        return false;
    }

    // Create lighting pipeline
    if (!createLightingPipeline()) {
        std::cerr << "Failed to create lighting pipeline" << std::endl;
        return false;
    }

    // Create compute pipeline
    if (!createComputePipeline()) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        return false;
    }

    // Create command pool
    if (!createCommandPool()) {
        return false;
    }

    // Create depth resources
    if (!createDepthResources()) {
        return false;
    }

    // Create descriptor pool
    if (!createDescriptorPool()) {
        return false;
    }

    // Create default texture resources
    if (!createDefaultTextureResources()) {
        std::cerr << "Failed to create default texture resources" << std::endl;
        return false;
    }

    // Create shared default PBR textures (to avoid creating hundreds of identical textures)
    if (!createSharedDefaultPBRTextures()) {
        std::cerr << "Failed to create shared default PBR textures" << std::endl;
        return false;
    }

    // Create shadow maps for shadow mapping
    if (!createShadowMaps()) {
        std::cerr << "Failed to create shadow maps" << std::endl;
        return false;
    }

    // Create a shadow map render pass
    if (!createShadowMapRenderPass()) {
        std::cerr << "Failed to create shadow map render pass" << std::endl;
        return false;
    }

    // Create shadow map framebuffers
    if (!createShadowMapFramebuffers()) {
        std::cerr << "Failed to create shadow map framebuffers" << std::endl;
        return false;
    }

    // Create a shadow map descriptor set layout
    if (!createShadowMapDescriptorSetLayout()) {
        std::cerr << "Failed to create shadow map descriptor set layout" << std::endl;
        return false;
    }

    // Create command buffers
    if (!createCommandBuffers()) {
        return false;
    }

    // Create sync objects
    if (!createSyncObjects()) {
        return false;
    }

    initialized = true;
    return true;
}

// Clean up renderer resources
void Renderer::Cleanup() {
    if (initialized) {
        std::cout << "Starting renderer cleanup..." << std::endl;

        // Wait for the device to be idle before cleaning up
        device.waitIdle();
        for (auto& resources : entityResources | std::views::values) {
            // Memory pool handles unmapping automatically, no need to manually unmap
            resources.basicDescriptorSets.clear();
            resources.pbrDescriptorSets.clear();
            resources.uniformBuffers.clear();
            resources.uniformBufferAllocations.clear();
            resources.uniformBuffersMapped.clear();
        }
        std::cout << "Renderer cleanup completed." << std::endl;
        initialized = false;
    }
}

// Create instance
bool Renderer::createInstance(const std::string& appName, bool enableValidationLayers) {
    try {
        // Create application info
        vk::ApplicationInfo appInfo{
            .pApplicationName = appName.c_str(),
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "Simple Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3
        };

        // Get required extensions
        std::vector<const char*> extensions;

        // Add required extensions for GLFW
#if PLATFORM_DESKTOP
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        extensions.insert(extensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);
#endif

        // Add debug extension if validation layers are enabled
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        // Create instance info
        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()
        };

        // Enable validation layers if requested
        vk::ValidationFeaturesEXT validationFeatures{};
        std::vector<vk::ValidationFeatureEnableEXT> enabledValidationFeatures;

        if (enableValidationLayers) {
            if (!checkValidationLayerSupport()) {
                std::cerr << "Validation layers requested, but not available" << std::endl;
                return false;
            }

            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // Enable debug printf functionality for shader debugging
            enabledValidationFeatures.push_back(vk::ValidationFeatureEnableEXT::eDebugPrintf);

            validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(enabledValidationFeatures.size());
            validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures.data();

            createInfo.pNext = &validationFeatures;
        }

        // Create instance
        instance = vk::raii::Instance(context, createInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create instance: " << e.what() << std::endl;
        return false;
    }
}

// Setup debug messenger
bool Renderer::setupDebugMessenger(bool enableValidationLayers) {
    if (!enableValidationLayers) {
        return true;
    }

    try {
        // Create debug messenger info
        vk::DebugUtilsMessengerCreateInfoEXT createInfo{
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                              vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                              vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                              vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                          vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            .pfnUserCallback = debugCallbackVkRaii
        };

        // Create debug messenger
        debugMessenger = vk::raii::DebugUtilsMessengerEXT(instance, createInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to set up debug messenger: " << e.what() << std::endl;
        return false;
    }
}

// Create surface
bool Renderer::createSurface() {
    try {
        // Create surface
        VkSurfaceKHR _surface;
        if (!platform->CreateVulkanSurface(*instance, &_surface)) {
            std::cerr << "Failed to create window surface" << std::endl;
            return false;
        }

        surface = vk::raii::SurfaceKHR(instance, _surface);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create surface: " << e.what() << std::endl;
        return false;
    }
}

// Pick a physical device
bool Renderer::pickPhysicalDevice() {
    try {
        // Get available physical devices
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

        if (devices.empty()) {
            std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
            return false;
        }

        // Find a suitable device using modern C++ ranges
        const auto devIter = std::ranges::find_if(
            devices,
            [&](auto& device) {
                // Print device properties for debugging
                vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
                std::cout << "Checking device: " << deviceProperties.deviceName << std::endl;

                // Check if device supports Vulkan 1.3
                bool supportsVulkan1_3 = deviceProperties.apiVersion >= VK_API_VERSION_1_3;
                if (!supportsVulkan1_3) {
                    std::cout << "  - Does not support Vulkan 1.3" << std::endl;
                }

                // Check queue families
                QueueFamilyIndices indices = findQueueFamilies(device);
                bool supportsGraphics = indices.isComplete();
                if (!supportsGraphics) {
                    std::cout << "  - Missing required queue families" << std::endl;
                }

                // Check device extensions
                bool supportsAllRequiredExtensions = checkDeviceExtensionSupport(device);
                if (!supportsAllRequiredExtensions) {
                    std::cout << "  - Missing required extensions" << std::endl;
                }

                // Check swap chain support
                bool swapChainAdequate = false;
                if (supportsAllRequiredExtensions) {
                    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
                    swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
                    if (!swapChainAdequate) {
                        std::cout << "  - Inadequate swap chain support" << std::endl;
                    }
                }

                // Check for required features
                auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
                bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering;
                if (!supportsRequiredFeatures) {
                    std::cout << "  - Does not support required features (dynamicRendering)" << std::endl;
                }

                return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && swapChainAdequate && supportsRequiredFeatures;
            });

        if (devIter != devices.end()) {
            physicalDevice = *devIter;
            vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
            std::cout << "Selected device: " << deviceProperties.deviceName << std::endl;

            // Store queue family indices for the selected device
            queueFamilyIndices = findQueueFamilies(physicalDevice);

            // Add supported optional extensions
            addSupportedOptionalExtensions();

            return true;
        }
        std::cerr << "Failed to find a suitable GPU. Make sure your GPU supports Vulkan and has the required extensions." << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Failed to pick physical device: " << e.what() << std::endl;
        return false;
    }
}

// Add supported optional extensions
void Renderer::addSupportedOptionalExtensions() {
    try {
        // Get available extensions
        auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

        // Check which optional extensions are supported and add them to deviceExtensions
        for (const auto& optionalExt : optionalDeviceExtensions) {
            for (const auto& availableExt : availableExtensions) {
                if (strcmp(availableExt.extensionName, optionalExt) == 0) {
                    deviceExtensions.push_back(optionalExt);
                    std::cout << "Adding optional extension: " << optionalExt << std::endl;
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to add optional extensions: " << e.what() << std::endl;
    }
}

// Create logical device
bool Renderer::createLogicalDevice(bool enableValidationLayers) {
    try {
        // Create queue create info for each unique queue family
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set uniqueQueueFamilies = {
            queueFamilyIndices.graphicsFamily.value(),
            queueFamilyIndices.presentFamily.value(),
            queueFamilyIndices.computeFamily.value()
        };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo{
                .queueFamilyIndex = queueFamily,
                .queueCount = 1,
                .pQueuePriorities = &queuePriority
            };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Enable required features
        auto features = physicalDevice.getFeatures2();
        features.features.samplerAnisotropy = vk::True;

        // Explicitly configure device features to prevent validation layer warnings
        // These features are required by extensions or other features, so we enable them explicitly

        // Timeline semaphore features (required for synchronization2)
        vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures;
        timelineSemaphoreFeatures.timelineSemaphore = vk::True;

        // Vulkan memory model features (required for some shader operations)
        vk::PhysicalDeviceVulkanMemoryModelFeatures memoryModelFeatures;
        memoryModelFeatures.vulkanMemoryModel = vk::True;
        memoryModelFeatures.vulkanMemoryModelDeviceScope = vk::True;

        // Buffer device address features (required for some buffer operations)
        vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures;
        bufferDeviceAddressFeatures.bufferDeviceAddress = vk::True;

        // 8-bit storage features (required for some shader storage operations)
        vk::PhysicalDevice8BitStorageFeatures storage8BitFeatures;
        storage8BitFeatures.storageBuffer8BitAccess = vk::True;

        // Enable Vulkan 1.3 features
        vk::PhysicalDeviceVulkan13Features vulkan13Features;
        vulkan13Features.dynamicRendering = vk::True;
        vulkan13Features.synchronization2 = vk::True;

        // Chain the feature structures together
        timelineSemaphoreFeatures.pNext = &memoryModelFeatures;
        memoryModelFeatures.pNext = &bufferDeviceAddressFeatures;
        bufferDeviceAddressFeatures.pNext = &storage8BitFeatures;
        storage8BitFeatures.pNext = &vulkan13Features;
        features.pNext = &timelineSemaphoreFeatures;

        // Create a device
        vk::DeviceCreateInfo createInfo{
            .pNext = &features,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = nullptr // Using pNext for features
        };

        // Enable validation layers if requested
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        // Create the logical device
        device = vk::raii::Device(physicalDevice, createInfo);

        // Get queue handles
        graphicsQueue = vk::raii::Queue(device, queueFamilyIndices.graphicsFamily.value(), 0);
        presentQueue = vk::raii::Queue(device, queueFamilyIndices.presentFamily.value(), 0);
        computeQueue = vk::raii::Queue(device, queueFamilyIndices.computeFamily.value(), 0);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create logical device: " << e.what() << std::endl;
        return false;
    }
}

// Check validation layer support
bool Renderer::checkValidationLayerSupport() const {
    // Get available layers
    std::vector<vk::LayerProperties> availableLayers = context.enumerateInstanceLayerProperties();

    // Check if all requested layers are available
    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}
