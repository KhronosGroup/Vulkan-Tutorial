#include "renderer.h"
#include <fstream>
#include <iostream>
#include <set>
#include <map>
#include <cstring>
#include <ranges>
#include <thread>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE; // In a .cpp file

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_platform.h>

// Debug callback for vk::raii
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallbackVkRaii(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageType,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        // Print a message to the console
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
    } else {
        // Print a message to the console
        std::cout << "Validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

// Renderer core implementation for the "Rendering Pipeline" chapter of the tutorial.
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

    // Pick the physical device
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

        // Optionally pre-allocate initial memory blocks for pools
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

    // Create the descriptor set layout
    if (!createDescriptorSetLayout()) {
        return false;
    }

    // Create the graphics pipeline
    if (!createGraphicsPipeline()) {
        return false;
    }

    // Create PBR pipeline
    if (!createPBRPipeline()) {
        return false;
    }

    // Create the lighting pipeline
    if (!createLightingPipeline()) {
        std::cerr << "Failed to create lighting pipeline" << std::endl;
        return false;
    }

    // Create compute pipeline
    if (!createComputePipeline()) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        return false;
    }

    // Create the command pool
    if (!createCommandPool()) {
        return false;
    }

    // Create depth resources
    if (!createDepthResources()) {
        return false;
    }

    // Create the descriptor pool
    if (!createDescriptorPool()) {
        return false;
    }

    if (!createOrResizeLightStorageBuffers(1)) {
        std::cerr << "Failed to create initial light storage buffers" << std::endl;
        return false;
    }

    if (!createOpaqueSceneColorResources()) {
        return false;
    }

    createTransparentDescriptorSets();

    // Create default texture resources
    if (!createDefaultTextureResources()) {
        std::cerr << "Failed to create default texture resources" << std::endl;
        return false;
    }

    // Create fallback transparent descriptor sets (must occur after default textures exist)
    createTransparentFallbackDescriptorSets();

    // Create shared default PBR textures (to avoid creating hundreds of identical textures)
    if (!createSharedDefaultPBRTextures()) {
        std::cerr << "Failed to create shared default PBR textures" << std::endl;
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

    // Initialize background thread pool for async tasks (textures, etc.) AFTER all Vulkan resources are ready
    try {
        // Size the thread pool based on hardware concurrency, clamped to a sensible range
        unsigned int hw = std::max(2u, std::min(8u, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4u));
        threadPool = std::make_unique<ThreadPool>(hw);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create thread pool: " << e.what() << std::endl;
        return false;
    }

    initialized = true;
    return true;
}

void Renderer::ensureThreadLocalVulkanInit() const {
    // Initialize Vulkan-Hpp dispatcher per-thread; required for multi-threaded RAII usage
    static thread_local bool s_tlsInitialized = false;
    if (s_tlsInitialized) return;
    try {
        vk::detail::DynamicLoader dl;
        auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        if (vkGetInstanceProcAddr) {
            VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
        }
        if (*instance) {
            VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
        }
        if (*device) {
            VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
        }
        s_tlsInitialized = true;
    } catch (...) {
        // best-effort
    }
}

// Clean up renderer resources
void Renderer::Cleanup() {
    // Ensure background workers are stopped before tearing down Vulkan resources
    {
        std::unique_lock<std::shared_mutex> lock(threadPoolMutex);
        if (threadPool) {
            threadPool.reset();
        }
    }
    if (initialized) {
        std::cout << "Starting renderer cleanup..." << std::endl;

        // Wait for the device to be idle before cleaning up
        device.waitIdle();

        // Clean up swapchain-bound resources (depth, offscreen color, pipelines, views, etc.)
        cleanupSwapChain();

        // Release per-entity resources and return pooled memory to the MemoryPool
        for (auto& resources : entityResources | std::views::values) {
            // Descriptor sets are RAII and freed with descriptorPool; just clear holders
            resources.basicDescriptorSets.clear();
            resources.pbrDescriptorSets.clear();

            // Destroy UBO buffers and return pooled allocations
            resources.uniformBuffers.clear();
            for (auto& alloc : resources.uniformBufferAllocations) {
                if (alloc) {
                    try { memoryPool->deallocate(std::move(alloc)); }
                    catch (const std::exception& e) {
                        std::cerr << "Warning: failed to deallocate UBO allocation during Cleanup: " << e.what() << std::endl;
                    }
                }
            }
            resources.uniformBufferAllocations.clear();
            resources.uniformBuffersMapped.clear();

            // Destroy instance buffer and return pooled allocation
            resources.instanceBuffer = nullptr;
            if (resources.instanceBufferAllocation) {
                try { memoryPool->deallocate(std::move(resources.instanceBufferAllocation)); }
                catch (const std::exception& e) {
                    std::cerr << "Warning: failed to deallocate instance buffer allocation during Cleanup: " << e.what() << std::endl;
                }
            }
            resources.instanceBufferMapped = nullptr;
        }
        entityResources.clear();

        // Release light storage buffers
        for (auto& lsb : lightStorageBuffers) {
            lsb.buffer = nullptr;
            if (lsb.allocation) {
                try { memoryPool->deallocate(std::move(lsb.allocation)); }
                catch (const std::exception& e) {
                    std::cerr << "Warning: failed to deallocate light storage buffer during Cleanup: " << e.what() << std::endl;
                }
            }
            lsb.mapped = nullptr;
            lsb.capacity = 0;
            lsb.size = 0;
        }
        lightStorageBuffers.clear();

        // Release mesh device-local buffers and return pooled allocations
        for (auto& [mesh, mres] : meshResources) {
            mres.vertexBuffer = nullptr;
            if (mres.vertexBufferAllocation) {
                try { memoryPool->deallocate(std::move(mres.vertexBufferAllocation)); }
                catch (const std::exception& e) {
                    std::cerr << "Warning: failed to deallocate vertex buffer allocation during Cleanup: " << e.what() << std::endl;
                }
            }
            mres.indexBuffer = nullptr;
            if (mres.indexBufferAllocation) {
                try { memoryPool->deallocate(std::move(mres.indexBufferAllocation)); }
                catch (const std::exception& e) {
                    std::cerr << "Warning: failed to deallocate index buffer allocation during Cleanup: " << e.what() << std::endl;
                }
            }
            // Staging buffers are RAII and not pooled
            mres.stagingVertexBuffer = nullptr;
            mres.stagingVertexBufferMemory = nullptr;
            mres.stagingIndexBuffer = nullptr;
            mres.stagingIndexBufferMemory = nullptr;
            mres.vertexBufferSizeBytes = 0;
            mres.indexBufferSizeBytes = 0;
            mres.indexCount = 0;
        }
        meshResources.clear();

        // Release textures and return pooled allocations
        {
            std::unique_lock<std::shared_mutex> texLock(textureResourcesMutex);
            for (auto& [key, tres] : textureResources) {
                tres.textureSampler = nullptr;
                tres.textureImageView = nullptr;
                tres.textureImage = nullptr;
                if (tres.textureImageAllocation) {
                    try { memoryPool->deallocate(std::move(tres.textureImageAllocation)); }
                    catch (const std::exception& e) {
                        std::cerr << "Warning: failed to deallocate texture image allocation during Cleanup: " << e.what() << std::endl;
                    }
                }
            }
            textureResources.clear();
            textureAliases.clear();
            textureToEntities.clear();
        }

        // Release default texture resources if allocated
        defaultTextureResources.textureSampler = nullptr;
        defaultTextureResources.textureImageView = nullptr;
        defaultTextureResources.textureImage = nullptr;
        if (defaultTextureResources.textureImageAllocation) {
            try { memoryPool->deallocate(std::move(defaultTextureResources.textureImageAllocation)); }
            catch (const std::exception& e) {
                std::cerr << "Warning: failed to deallocate default texture image allocation during Cleanup: " << e.what() << std::endl;
            }
        }
        // Also clear global descriptor sets that are allocated from descriptorPool, so they are
        // destroyed while the pool is still valid (avoid vkFreeDescriptorSets invalid pool errors)
        transparentDescriptorSets.clear();
        transparentFallbackDescriptorSets.clear();
        computeDescriptorSets.clear();
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
#if defined(PLATFORM_DESKTOP)
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

        // Prioritize discrete GPUs (like NVIDIA RTX 2080) over integrated GPUs (like Intel UHD Graphics)
        // First, collect all suitable devices with their suitability scores
        std::multimap<int, vk::raii::PhysicalDevice> suitableDevices;

        for (auto& _device : devices) {
            // Print device properties for debugging
            vk::PhysicalDeviceProperties deviceProperties = _device.getProperties();
            std::cout << "Checking device: " << deviceProperties.deviceName
                      << " (Type: " << vk::to_string(deviceProperties.deviceType) << ")" << std::endl;

            // Check if the device supports Vulkan 1.3
            bool supportsVulkan1_3 = deviceProperties.apiVersion >= VK_API_VERSION_1_3;
            if (!supportsVulkan1_3) {
                std::cout << "  - Does not support Vulkan 1.3" << std::endl;
                continue;
            }

            // Check queue families
            QueueFamilyIndices indices = findQueueFamilies(_device);
            bool supportsGraphics = indices.isComplete();
            if (!supportsGraphics) {
                std::cout << "  - Missing required queue families" << std::endl;
                continue;
            }

            // Check device extensions
            bool supportsAllRequiredExtensions = checkDeviceExtensionSupport(_device);
            if (!supportsAllRequiredExtensions) {
                std::cout << "  - Missing required extensions" << std::endl;
                continue;
            }

            // Check swap chain support
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_device);
            bool swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            if (!swapChainAdequate) {
                std::cout << "  - Inadequate swap chain support" << std::endl;
                continue;
            }

            // Check for required features
            auto features = _device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
            bool supportsRequiredFeatures = features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering;
            if (!supportsRequiredFeatures) {
                std::cout << "  - Does not support required features (dynamicRendering)" << std::endl;
                continue;
            }

            // Calculate suitability score - prioritize discrete GPUs
            int score = 0;

            // Discrete GPUs get the highest priority (NVIDIA RTX 2080, AMD, etc.)
            if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                score += 1000;
                std::cout << "  - Discrete GPU: +1000 points" << std::endl;
            }
            // Integrated GPUs get lower priority (Intel UHD Graphics, etc.)
            else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
                score += 100;
                std::cout << "  - Integrated GPU: +100 points" << std::endl;
            }

            // Add points for memory size (more VRAM is better)
            vk::PhysicalDeviceMemoryProperties memProperties = _device.getMemoryProperties();
            for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
                if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                    // Add 1 point per GB of VRAM
                    score += static_cast<int>(memProperties.memoryHeaps[i].size / (1024 * 1024 * 1024));
                    break;
                }
            }

            std::cout << "  - Device is suitable with score: " << score << std::endl;
            suitableDevices.emplace(score, _device);
        }

        if (!suitableDevices.empty()) {
            // Select the device with the highest score (discrete GPU with most VRAM)
            physicalDevice = suitableDevices.rbegin()->second;
            vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
            std::cout << "Selected device: " << deviceProperties.deviceName
                      << " (Type: " << vk::to_string(deviceProperties.deviceType)
                      << ", Score: " << suitableDevices.rbegin()->first << ")" << std::endl;

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

        // Build a set of available extension names for quick lookup
        std::set<std::string> avail;
        for (const auto& e : availableExtensions) { avail.insert(e.extensionName); }

        // First, handle dependency: VK_EXT_attachment_feedback_loop_dynamic_state requires VK_EXT_attachment_feedback_loop_layout
        const char* dynState = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME;
        const char* layoutReq = "VK_EXT_attachment_feedback_loop_layout";
        bool dynSupported = avail.contains(dynState);
        bool layoutSupported = avail.contains(layoutReq);
        for (const auto& optionalExt : optionalDeviceExtensions) {
            if (std::strcmp(optionalExt, dynState) == 0) {
                if (dynSupported && layoutSupported) {
                    deviceExtensions.push_back(dynState);
                    deviceExtensions.push_back(layoutReq);
                    std::cout << "Adding optional extension: " << dynState << std::endl;
                    std::cout << "Adding required-by-optional extension: " << layoutReq << std::endl;
                } else if (dynSupported && !layoutSupported) {
                    std::cout << "Skipping extension due to missing dependency: " << dynState << " requires " << layoutReq << std::endl;
                }
                continue; // handled
            }
            if (avail.contains(optionalExt)) {
                deviceExtensions.push_back(optionalExt);
                std::cout << "Adding optional extension: " << optionalExt << std::endl;
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
            queueFamilyIndices.computeFamily.value(),
            queueFamilyIndices.transferFamily.value()
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
        features.features.depthBiasClamp = vk::True;

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

        // Create a device. Device layers are deprecated and ignored, so we
        // only configure extensions and features here; validation is enabled
        // via instance layers.
        vk::DeviceCreateInfo createInfo{
            .pNext = &features,
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
            .ppEnabledExtensionNames = deviceExtensions.data(),
            .pEnabledFeatures = nullptr // Using pNext for features
        };

        // Create the logical device
        device = vk::raii::Device(physicalDevice, createInfo);

        // Get queue handles
        graphicsQueue = vk::raii::Queue(device, queueFamilyIndices.graphicsFamily.value(), 0);
        presentQueue = vk::raii::Queue(device, queueFamilyIndices.presentFamily.value(), 0);
        computeQueue = vk::raii::Queue(device, queueFamilyIndices.computeFamily.value(), 0);
        transferQueue = vk::raii::Queue(device, queueFamilyIndices.transferFamily.value(), 0);

        // Create global timeline semaphore for uploads early (needed before default texture creation)
        vk::SemaphoreTypeCreateInfo typeInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0
        };
        vk::SemaphoreCreateInfo timelineCreateInfo{ .pNext = &typeInfo };
        uploadsTimeline = vk::raii::Semaphore(device, timelineCreateInfo);
        uploadTimelineLastSubmitted.store(0, std::memory_order_relaxed);

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
