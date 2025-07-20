#include "renderer.h"
#include <fstream>
#include <iostream>
#include <set>
#include <cstring>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE; // In a .cpp file

#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <vulkan/vk_platform.h>

// Debug callback for vk::raii
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallbackVkRaii(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        // Print message to console
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

// Debug callback
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        // Print message to console
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
    // Create Vulkan instance
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
        // Wait for the device to be idle before cleaning up
        device.waitIdle();

        // Clean up swap chain
        cleanupSwapChain();

        // Clear resources - RAII will handle destruction
        imageAvailableSemaphores.clear();
        renderFinishedSemaphores.clear();
        inFlightFences.clear();
        commandBuffers.clear();
        commandPool = nullptr;
        descriptorPool = nullptr;
        pbrGraphicsPipeline = nullptr;
        pbrPipelineLayout = nullptr;
        lightingPipeline = nullptr;
        lightingPipelineLayout = nullptr;
        graphicsPipeline = nullptr;
        pipelineLayout = nullptr;
        computePipeline = nullptr;
        computePipelineLayout = nullptr;
        computeDescriptorSetLayout = nullptr;
        computeDescriptorPool = nullptr;
        descriptorSetLayout = nullptr;

        // Clear mesh resources - RAII will handle destruction
        meshResources.clear();

        // Clear texture resources - RAII will handle destruction
        textureResources.clear();

        // Clear entity resources - RAII will handle destruction
        entityResources.clear();

        // Clear device, surface, debug messenger, and instance - RAII will handle destruction
        device = nullptr;
        surface = nullptr;
        debugMessenger = nullptr;
        instance = nullptr;

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
        if (enableValidationLayers) {
            if (!checkValidationLayerSupport()) {
                std::cerr << "Validation layers requested, but not available" << std::endl;
                return false;
            }

            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
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

// Pick physical device
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
        } else {
            std::cerr << "Failed to find a suitable GPU. Make sure your GPU supports Vulkan and has the required extensions." << std::endl;
            return false;
        }
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
            bool supported = false;
            for (const auto& availableExt : availableExtensions) {
                if (strcmp(availableExt.extensionName, optionalExt) == 0) {
                    supported = true;
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
        // Create queue create infos for each unique queue family
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
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

        // Enable Vulkan 1.3 features
        vk::PhysicalDeviceVulkan13Features vulkan13Features;
        vulkan13Features.dynamicRendering = vk::True;
        vulkan13Features.synchronization2 = vk::True;
        features.pNext = &vulkan13Features;

        // Create device
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
bool Renderer::checkValidationLayerSupport() {
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
