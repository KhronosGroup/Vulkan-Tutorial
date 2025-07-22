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
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>

#ifdef __INTELLISENSE__
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <vulkan/vk_platform.h>

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint64_t FenceTimeout = 100000000;
constexpr uint32_t PARTICLE_COUNT = 8192;

constexpr int MAX_FRAMES_IN_FLIGHT = 2;


struct UniformBufferObject {
    float deltaTime = 1.0f;
};

struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Particle), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position) ),
            vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color) ),
        };
    }
};

// Simple logging function
template<typename... Args>
void log(Args&&... args) {
    // Only log in debug builds
#ifdef _DEBUG
    (std::cout << ... << std::forward<Args>(args)) << std::endl;
#endif
}

class ThreadSafeResourceManager {
private:
    std::mutex resourceMutex;
    std::vector<vk::raii::CommandPool> commandPools;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

public:
    void createThreadCommandPools(vk::raii::Device& device, uint32_t queueFamilyIndex, uint32_t threadCount) {
        std::lock_guard<std::mutex> lock(resourceMutex);

        commandBuffers.clear();
        commandPools.clear();

        for (uint32_t i = 0; i < threadCount; i++) {
            vk::CommandPoolCreateInfo poolInfo{
                .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                .queueFamilyIndex = queueFamilyIndex
            };
            try {
                commandPools.emplace_back(device, poolInfo);
            } catch (const std::exception&) {
                throw; // Re-throw the exception to be caught by the caller
            }
        }
    }

    vk::raii::CommandPool& getCommandPool(uint32_t threadIndex) {
        std::lock_guard lock(resourceMutex);
        return commandPools[threadIndex];
    }

    void allocateCommandBuffers(vk::raii::Device& device, uint32_t threadCount, uint32_t buffersPerThread) {
        std::lock_guard lock(resourceMutex);

        commandBuffers.clear();

        if (commandPools.size() < threadCount) {
            throw std::runtime_error("Not enough command pools for thread count");
        }

        for (uint32_t i = 0; i < threadCount; i++) {
            vk::CommandBufferAllocateInfo allocInfo{
                .commandPool = *commandPools[i],
                .level = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = buffersPerThread
            };
            try {
                auto threadBuffers = device.allocateCommandBuffers(allocInfo);
                for (auto& buffer : threadBuffers) {
                    commandBuffers.emplace_back(std::move(buffer));
                }
            } catch (const std::exception&) {
                throw; // Re-throw the exception to be caught by the caller
            }
        }
    }

    vk::raii::CommandBuffer& getCommandBuffer(uint32_t index) {
        // No need for mutex here as each thread accesses its own command buffer
        if (index >= commandBuffers.size()) {
            throw std::runtime_error("Command buffer index out of range: " + std::to_string(index) +
                                    " (available: " + std::to_string(commandBuffers.size()) + ")");
        }
        return commandBuffers[index];
    }
};

class MultithreadedApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        initThreads();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *             window = nullptr;
    vk::raii::Context        context;
    vk::raii::Instance       instance       = nullptr;
    vk::raii::SurfaceKHR     surface        = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device         device         = nullptr;
    uint32_t                 queueIndex     = ~0;
    vk::raii::Queue          queue          = nullptr;

    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::SurfaceFormatKHR swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;

    std::vector<vk::raii::Buffer> shaderStorageBuffers;
    std::vector<vk::raii::DeviceMemory> shaderStorageBuffersMemory;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> computeDescriptorSets;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> graphicsCommandBuffers;

    vk::raii::Semaphore timelineSemaphore = nullptr;
    uint64_t timelineValue = 0;
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    uint32_t currentFrame = 0;

    double lastFrameTime = 0.0;

    // Removed resize-related variables and FSM state management as per simplification request

    double lastTime = 0.0f;

    uint32_t threadCount = 0;
    std::vector<std::thread> workerThreads;
    std::atomic<bool> shouldExit{false};
    std::vector<std::atomic<bool>> threadWorkReady;
    std::vector<std::atomic<bool>> threadWorkDone;

    std::mutex queueSubmitMutex;
    std::mutex workCompleteMutex;
    std::condition_variable workCompleteCv;

    ThreadSafeResourceManager resourceManager;
    struct ParticleGroup {
        uint32_t startIndex;
        uint32_t count;
    };
    std::vector<ParticleGroup> particleGroups;

    std::vector<const char*> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };

    // Helper functions
    [[nodiscard]] static std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        return extensions;
    }
    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }
    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }
    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        return {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }
    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
        vk::raii::ShaderModule shaderModule{ device, createInfo };

        return shaderModule;
    }
    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();

        return buffer;
    }

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Multithreading", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);

        lastTime = glfwGetTime();
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createComputeDescriptorSetLayout();
        createGraphicsPipeline();
        createComputePipeline();
        createCommandPool();
        createShaderStorageBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createComputeDescriptorSets();
        createGraphicsCommandBuffers();
        createSyncObjects();
    }

    void initThreads() {
        // Increase thread count for better parallelism
        threadCount = 8u;
        log("Initializing ", threadCount, " threads for sequential execution");

        threadWorkReady = std::vector<std::atomic<bool>>(threadCount);
        threadWorkDone = std::vector<std::atomic<bool>>(threadCount);

        for (uint32_t i = 0; i < threadCount; i++) {
            threadWorkReady[i] = false;
            threadWorkDone[i] = true;
        }

        initThreadResources();

        const uint32_t particlesPerThread = PARTICLE_COUNT / threadCount;
        particleGroups.resize(threadCount);

        for (uint32_t i = 0; i < threadCount; i++) {
            particleGroups[i].startIndex = i * particlesPerThread;
            particleGroups[i].count = (i == threadCount - 1) ?
                (PARTICLE_COUNT - i * particlesPerThread) : particlesPerThread;
            log("Thread ", i, " will process particles ",
                particleGroups[i].startIndex, " to ",
                (particleGroups[i].startIndex + particleGroups[i].count - 1),
                " (count: ", particleGroups[i].count, ")");
        }

        for (uint32_t i = 0; i < threadCount; i++) {
            workerThreads.emplace_back(&MultithreadedApplication::workerThreadFunc, this, i);
            log("Started worker thread ", i);
        }
    }

    void workerThreadFunc(uint32_t threadIndex) {
        while (!shouldExit) {
            // Wait for work using condition variable
            {
                std::unique_lock<std::mutex> lock(workCompleteMutex);
                workCompleteCv.wait(lock, [this, threadIndex]() {
                    return shouldExit || threadWorkReady[threadIndex].load(std::memory_order_acquire);
                });

                if (shouldExit) {
                    break;
                }

                if (!threadWorkReady[threadIndex].load(std::memory_order_acquire)) {
                    continue;
                }
            }

            const ParticleGroup& group = particleGroups[threadIndex];
            bool workCompleted = false;

            try {
                // Get command buffer and record commands
                vk::raii::CommandBuffer* cmdBuffer = &resourceManager.getCommandBuffer(threadIndex);
                recordComputeCommandBuffer(*cmdBuffer, group.startIndex, group.count);
                workCompleted = true;
            } catch (const std::exception&) {
                workCompleted = false;
            }

            // Mark work as done
            threadWorkDone[threadIndex].store(true, std::memory_order_release);
            threadWorkReady[threadIndex].store(false, std::memory_order_release);

            // If this is not the last thread, signal the next thread to start
            if (threadIndex < threadCount - 1) {
                threadWorkReady[threadIndex + 1].store(true, std::memory_order_release);
            }

            // Notify main thread and other threads
            {
                std::lock_guard<std::mutex> lock(workCompleteMutex);
                workCompleteCv.notify_all();
            }
        }
    }

    void mainLoop() {
        const double targetFrameTime = 1.0 / 60.0;

        while (!glfwWindowShouldClose(window)) {
            double frameStartTime = glfwGetTime();

            glfwPollEvents();
            drawFrame();

            double currentTime = glfwGetTime();
            lastFrameTime = (currentTime - lastTime) * 1000.0;
            lastTime = currentTime;

            double frameTime = currentTime - frameStartTime;

            if (frameTime < targetFrameTime) {
                double sleepTime = targetFrameTime - frameTime;
                std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
            }
        }

        device.waitIdle();
    }

    void cleanupSwapChain() {
        swapChainImageViews.clear();
        graphicsPipeline = nullptr;
        pipelineLayout = nullptr;
        computePipeline = nullptr;
        computePipelineLayout = nullptr;
        computeDescriptorSets.clear();
        computeDescriptorSetLayout = nullptr;
        descriptorPool = nullptr;

        // Unmap and clean up uniform buffers
        for (size_t i = 0; i < uniformBuffersMapped.size(); i++) {
            uniformBuffersMemory[i].unmapMemory();
        }
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        // Clean up shader storage buffers
        shaderStorageBuffers.clear();
        shaderStorageBuffersMemory.clear();

        swapChain = nullptr;
    }

    void stopThreads() {
        shouldExit.store(true, std::memory_order_release);

        for (uint32_t i = 0; i < threadCount; i++) {
            threadWorkDone[i].store(true, std::memory_order_release);
            threadWorkReady[i].store(false, std::memory_order_release);
        }

        // Notify all threads in case they're waiting on the condition variable
        {
            std::lock_guard<std::mutex> lock(workCompleteMutex);
            workCompleteCv.notify_all();
        }

        for (auto& thread : workerThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        workerThreads.clear();
    }

    void initThreadResources() {
        resourceManager.createThreadCommandPools(device, queueIndex, threadCount);
        resourceManager.allocateCommandBuffers(device, threadCount, 1);
    }

    void cleanup() {
        stopThreads();

        glfwDestroyWindow(window);
        glfwTerminate();
    }


    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{ .pApplicationName   = "Vulkan Multithreading",
                .applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
                .pEngineName        = "No Engine",
                .engineVersion      = VK_MAKE_VERSION( 1, 0, 0 ),
                .apiVersion         = vk::ApiVersion14 };
        auto extensions = getRequiredExtensions();
        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = 0,
            .ppEnabledLayerNames     = nullptr,
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data() };
        instance = vk::raii::Instance(context, createInfo);
    }


    void createSurface() {
        VkSurfaceKHR       _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto devIter = std::ranges::find_if(
          devices,
          [&](auto const & device)
          {
            bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

            auto queueFamilies = device.getQueueFamilyProperties();
            bool supportsGraphics =
              std::ranges::any_of(queueFamilies, [](auto const & qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

            auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
            bool supportsAllRequiredExtensions =
              std::ranges::all_of(requiredDeviceExtension,
                                 [&availableDeviceExtensions](auto const & requiredDeviceExtension)
                      {
                                   return std::ranges::any_of(availableDeviceExtensions,
                                                             [requiredDeviceExtension](auto const & availableDeviceExtension)
                                                             { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
                      });

            auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
            bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                              features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

            return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
        });
        if (devIter != devices.end())
        {
            physicalDevice = *devIter;
        }
        else
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice() {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports both graphics and present
        for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
        {
            if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
			    (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
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

        auto features = physicalDevice.getFeatures2();
        features.features.samplerAnisotropy = vk::True;
        vk::PhysicalDeviceVulkan13Features vulkan13Features;
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures;
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR timelineSemaphoreFeatures;
        timelineSemaphoreFeatures.timelineSemaphore = vk::True;
        vulkan13Features.dynamicRendering = vk::True;
        vulkan13Features.synchronization2 = vk::True;
        extendedDynamicStateFeatures.extendedDynamicState = vk::True;
        extendedDynamicStateFeatures.pNext = &timelineSemaphoreFeatures;
        vulkan13Features.pNext = &extendedDynamicStateFeatures;
        features.pNext = &vulkan13Features;

        float queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
        vk::DeviceCreateInfo deviceCreateInfo{
            .pNext = &features,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &deviceQueueCreateInfo,
            .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
            .ppEnabledExtensionNames = requiredDeviceExtension.data()
        };

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        queue = vk::raii::Queue(device, queueIndex, 0);
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
        swapChainImageFormat = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        minImageCount = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;

        vk::raii::SwapchainKHR oldSwapChain = std::move(swapChain);

        vk::SwapchainCreateInfoKHR swapChainCreateInfo{
            .flags = vk::SwapchainCreateFlagsKHR(),
            .surface = surface, .minImageCount = minImageCount,
            .imageFormat = swapChainImageFormat.format, .imageColorSpace = swapChainImageFormat.colorSpace,
            .imageExtent = swapChainExtent, .imageArrayLayers =1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment, .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = surfaceCapabilities.currentTransform, .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)),
            .clipped = true,
            .oldSwapchain = *oldSwapChain ? *oldSwapChain : nullptr };

        swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
        oldSwapChain = nullptr;
        swapChainImages = swapChain.getImages();
    }

    void createImageViews() {
        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = swapChainImageFormat.format,
            .components = {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
            .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        };
        for ( auto image : swapChainImages )
        {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back( device, imageViewCreateInfo );
        }
    }

    void createComputeDescriptorSetLayout() {
        std::array layoutBindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr)
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(layoutBindings.size()), .pBindings = layoutBindings.data() };
        computeDescriptorSetLayout = vk::raii::DescriptorSetLayout( device, layoutInfo );
    }

    void createGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Particle::getBindingDescription();
        auto attributeDescriptions = Particle::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDescription, .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()), .pVertexAttributeDescriptions = attributeDescriptions.data() };
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::ePointList, .primitiveRestartEnable = vk::False };
        vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };
        vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayout = vk::raii::PipelineLayout( device, pipelineLayoutInfo );

        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{ .colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainImageFormat.format };
        vk::GraphicsPipelineCreateInfo pipelineInfo{ .pNext = &pipelineRenderingCreateInfo,
            .stageCount = 2,
            .pStages = shaderStages,
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = *pipelineLayout,
            .subpass = 0
        };

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createComputePipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        // Create push constant range for particle group information
        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(uint32_t) * 2  // startIndex and count
        };

        vk::PipelineShaderStageCreateInfo computeShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = "compMain" };
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*computeDescriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };
        computePipelineLayout = vk::raii::PipelineLayout( device, pipelineLayoutInfo );
        vk::ComputePipelineCreateInfo pipelineInfo{ .stage = computeShaderStageInfo, .layout = *computePipelineLayout };
        computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = queueIndex;
        commandPool = vk::raii::CommandPool(device, poolInfo);
    }

    void createShaderStorageBuffers() {
        std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
        std::uniform_real_distribution rndDist(0.0f, 1.0f);

        std::vector<Particle> particles(PARTICLE_COUNT);
        for (auto& particle : particles) {
            // Generate a random position for the particle
            float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;

            // Use square root of random value to ensure uniform distribution across the area
            // This prevents clustering near the center (which causes the donut effect)
            float r = sqrtf(rndDist(rndEngine)) * 0.25f;

            float x = r * cosf(theta) * HEIGHT / WIDTH;
            float y = r * sinf(theta);
            particle.position = glm::vec2(x, y);

            // Ensure a minimum velocity and scale based on distance from center
            float minVelocity = 0.001f;
            float velocityScale = 0.003f;
            float velocityMagnitude = std::max(minVelocity, r * velocityScale);
            particle.velocity = normalize(glm::vec2(x,y)) * velocityMagnitude;
            particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
        }

        vk::DeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, particles.data(), (size_t)bufferSize);
        stagingBufferMemory.unmapMemory();

        shaderStorageBuffers.clear();
        shaderStorageBuffersMemory.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::raii::Buffer shaderStorageBufferTemp({});
            vk::raii::DeviceMemory shaderStorageBufferTempMemory({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, shaderStorageBufferTemp, shaderStorageBufferTempMemory);
            copyBuffer(stagingBuffer, shaderStorageBufferTemp, bufferSize);
            shaderStorageBuffers.emplace_back(std::move(shaderStorageBufferTemp));
            shaderStorageBuffersMemory.emplace_back(std::move(shaderStorageBufferTempMemory));
        }
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back( uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createDescriptorPool() {
        std::array poolSize {
            vk::DescriptorPoolSize( vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize(  vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT * 2)
        };
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
        poolInfo.poolSizeCount = poolSize.size();
        poolInfo.pPoolSizes = poolSize.data();
        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
    }

    void createComputeDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = *descriptorPool;
        allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        allocInfo.pSetLayouts = layouts.data();
        computeDescriptorSets.clear();
        computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));

            vk::DescriptorBufferInfo storageBufferInfoLastFrame(shaderStorageBuffers[(i + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT], 0, sizeof(Particle) * PARTICLE_COUNT);
            vk::DescriptorBufferInfo storageBufferInfoCurrentFrame(shaderStorageBuffers[i], 0, sizeof(Particle) * PARTICLE_COUNT);
            std::array descriptorWrites{
                vk::WriteDescriptorSet{ .dstSet = *computeDescriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pImageInfo = nullptr, .pBufferInfo = &bufferInfo, .pTexelBufferView = nullptr },
                vk::WriteDescriptorSet{ .dstSet = *computeDescriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pImageInfo = nullptr, .pBufferInfo = &storageBufferInfoLastFrame, .pTexelBufferView = nullptr },
                vk::WriteDescriptorSet{ .dstSet = *computeDescriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pImageInfo = nullptr, .pBufferInfo = &storageBufferInfoCurrentFrame, .pTexelBufferView = nullptr },
            };
            device.updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) const {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        buffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(bufferMemory, 0);
    }

    [[nodiscard]] vk::raii::CommandBuffer beginSingleTimeCommands() const {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;
        vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers( device, allocInfo ).front());

        vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandBuffer;
        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    void copyBuffer(const vk::raii::Buffer & srcBuffer, const vk::raii::Buffer & dstBuffer, vk::DeviceSize size) const {
        vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
        commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
        endSingleTimeCommands(commandCopyBuffer);
    }

    [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createGraphicsCommandBuffers() {
        graphicsCommandBuffers.clear();
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        graphicsCommandBuffers = vk::raii::CommandBuffers(device, allocInfo);
    }

    void recordComputeCommandBuffer(vk::raii::CommandBuffer& cmdBuffer, uint32_t startIndex, uint32_t count) {
        cmdBuffer.reset();

        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        cmdBuffer.begin(beginInfo);

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, {*computeDescriptorSets[currentFrame]}, {});

        struct PushConstants {
            uint32_t startIndex;
            uint32_t count;
        } pushConstants{startIndex, count};

        cmdBuffer.pushConstants<PushConstants>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstants);

        uint32_t groupCount = (count + 255) / 256;
        cmdBuffer.dispatch(groupCount, 1, 1);

        cmdBuffer.end();
    }

    void recordGraphicsCommandBuffer(uint32_t imageIndex) {
        graphicsCommandBuffers[currentFrame].reset();

        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        graphicsCommandBuffers[currentFrame].begin(beginInfo);

        transition_image_layout(
            imageIndex,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        vk::RenderingAttachmentInfo attachmentInfo = {
            .imageView = swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor
        };
        vk::RenderingInfo renderingInfo = {
            .renderArea = { .offset = { 0, 0 }, .extent = swapChainExtent },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentInfo
        };

        graphicsCommandBuffers[currentFrame].beginRendering(renderingInfo);

        graphicsCommandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        graphicsCommandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        graphicsCommandBuffers[currentFrame].setScissor( 0, vk::Rect2D( vk::Offset2D( 0, 0 ), swapChainExtent ) );
        graphicsCommandBuffers[currentFrame].bindVertexBuffers(0, { shaderStorageBuffers[currentFrame] }, {0});
        graphicsCommandBuffers[currentFrame].draw( PARTICLE_COUNT, 1, 0, 0 );
        graphicsCommandBuffers[currentFrame].endRendering();

        transition_image_layout(
            imageIndex,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        graphicsCommandBuffers[currentFrame].end();
    }

    void transition_image_layout(
            uint32_t imageIndex,
            vk::ImageLayout old_layout,
            vk::ImageLayout new_layout,
            vk::AccessFlags2 src_access_mask,
            vk::AccessFlags2 dst_access_mask,
            vk::PipelineStageFlags2 src_stage_mask,
            vk::PipelineStageFlags2 dst_stage_mask
            ) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = src_stage_mask,
            .srcAccessMask = src_access_mask,
            .dstStageMask = dst_stage_mask,
            .dstAccessMask = dst_access_mask,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapChainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vk::DependencyInfo dependency_info = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        graphicsCommandBuffers[currentFrame].pipelineBarrier2(dependency_info);
    }

    void signalThreadsToWork() {
        // Mark all threads as not done
        for (uint32_t i = 0; i < threadCount; i++) {
            threadWorkDone[i].store(false, std::memory_order_release);
        }

        // Memory barrier to ensure all threads see the updated threadWorkDone values
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Only signal the first thread to start work
        threadWorkReady[0].store(true, std::memory_order_release);

        // Notify all threads in case they're waiting on the condition variable
        {
            std::lock_guard<std::mutex> lock(workCompleteMutex);
            workCompleteCv.notify_all();
        }
    }

    void waitForThreadsToComplete() {
        std::unique_lock<std::mutex> lock(workCompleteMutex);

        // Wait for the last thread to complete with a timeout
        auto waitResult = workCompleteCv.wait_for(lock, std::chrono::milliseconds(3000), [this]() {
            return threadWorkDone[threadCount - 1].load(std::memory_order_acquire);
        });

        // If we timed out, force completion
        if (!waitResult) {
            // Force all threads to complete
            for (uint32_t i = 0; i < threadCount; i++) {
                threadWorkDone[i].store(true, std::memory_order_release);
                threadWorkReady[i].store(false, std::memory_order_release);
            }

            // Notify all threads
            workCompleteCv.notify_all();
            lock.unlock();

            // Give threads a chance to respond to the forced completion
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.clear();
        inFlightFences.clear();

        vk::SemaphoreTypeCreateInfo semaphoreType{ .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0 };
        timelineSemaphore = vk::raii::Semaphore(device, {.pNext = &semaphoreType});
        timelineValue = 0;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());

            vk::FenceCreateInfo fenceInfo;
            fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
            inFlightFences.emplace_back(device, fenceInfo);
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        UniformBufferObject ubo{};
        ubo.deltaTime = static_cast<float>(lastFrameTime) * 2.0f;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        // Wait for the previous frame to finish
        while (vk::Result::eTimeout == device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
            ;
        device.resetFences(*inFlightFences[currentFrame]);

        // Acquire the next image
        auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *imageAvailableSemaphores[currentFrame], nullptr);

        // Update timeline values for synchronization
        uint64_t computeWaitValue = timelineValue;
        uint64_t computeSignalValue = ++timelineValue;
        uint64_t graphicsWaitValue = computeSignalValue;
        uint64_t graphicsSignalValue = ++timelineValue;

        // Update uniform buffer with the latest delta time
        updateUniformBuffer(currentFrame);

        // Signal worker threads to start processing particles
        signalThreadsToWork();

        // Record graphics command buffer while worker threads are busy
        recordGraphicsCommandBuffer(imageIndex);

        // Wait for all worker threads to complete
        waitForThreadsToComplete();

        // Collect command buffers from all threads
        std::vector<vk::CommandBuffer> computeCmdBuffers;
        computeCmdBuffers.reserve(threadCount);
        for (uint32_t i = 0; i < threadCount; i++) {
            try {
                computeCmdBuffers.push_back(*resourceManager.getCommandBuffer(i));
            } catch (const std::exception&) {
                // Skip this thread's command buffer if there was an error
            }
        }

        // Ensure we have at least one command buffer
        if (computeCmdBuffers.empty()) {
            return;
        }

        // Set up compute submission
        vk::TimelineSemaphoreSubmitInfo computeTimelineInfo{
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &computeWaitValue,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &computeSignalValue
        };

        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eComputeShader};

        vk::SubmitInfo computeSubmitInfo{
            .pNext = &computeTimelineInfo,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*timelineSemaphore,
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = static_cast<uint32_t>(computeCmdBuffers.size()),
            .pCommandBuffers = computeCmdBuffers.data(),
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*timelineSemaphore
        };

        // Submit compute work
        {
            std::lock_guard<std::mutex> lock(queueSubmitMutex);
            queue.submit(computeSubmitInfo, nullptr);
        }

        // Set up graphics submission
        vk::PipelineStageFlags graphicsWaitStages[] = {vk::PipelineStageFlagBits::eVertexInput, vk::PipelineStageFlagBits::eColorAttachmentOutput};

        std::array<vk::Semaphore, 2> waitSemaphores = {*timelineSemaphore, *imageAvailableSemaphores[currentFrame]};
        std::array<uint64_t, 2> waitSemaphoreValues = {graphicsWaitValue, 0};

        vk::TimelineSemaphoreSubmitInfo graphicsTimelineInfo{
            .waitSemaphoreValueCount = static_cast<uint32_t>(waitSemaphoreValues.size()),
            .pWaitSemaphoreValues = waitSemaphoreValues.data(),
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &graphicsSignalValue
        };

        vk::SubmitInfo graphicsSubmitInfo{
            .pNext = &graphicsTimelineInfo,
            .waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size()),
            .pWaitSemaphores = waitSemaphores.data(),
            .pWaitDstStageMask = graphicsWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &*graphicsCommandBuffers[currentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*timelineSemaphore
        };

        // Submit graphics work
        {
            std::lock_guard<std::mutex> lock(queueSubmitMutex);
            queue.submit(graphicsSubmitInfo, *inFlightFences[currentFrame]);
        }

        // Wait for graphics to complete before presenting
        vk::SemaphoreWaitInfo waitInfo{
            .semaphoreCount = 1,
            .pSemaphores = &*timelineSemaphore,
            .pValues = &graphicsSignalValue
        };

        auto waitResult = device.waitSemaphores(waitInfo, 5000000000);
        if (waitResult == vk::Result::eTimeout) {
            device.waitIdle();
            return;
        }

        // Present the image
        vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .swapchainCount = 1,
            .pSwapchains = &*swapChain,
            .pImageIndices = &imageIndex
        };

        result = queue.presentKHR(presentInfo);

        // Move to the next frame
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

};


int main() {
    try {
        MultithreadedApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
