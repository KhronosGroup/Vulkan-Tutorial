// Chapter 4 – Subgroup / Wave Operations: Hair Strands on a Sphere in Wind
//
// Demonstrates Vulkan subgroup (wave) operations in an interactive windowed app:
//   • WaveActiveSum  – per-strand wind force aggregation within a wave
//   • WavePrefixSum  – strand-segment chain propagation (prefix scan)
//   • WaveActiveBallot – cull strands in the wind shadow
//   • Atomic slot counter – safe wave-index assignment on Intel Arc ARL
//                           (variable SIMD8/16/32 within a single workgroup)
//
// Two dispatches per frame:
//   1. physicsMain (256,1,1) – one thread per strand, computes tip displacement
//   2. renderMain  (16,16,1) – one thread per pixel, ray-marches the scene
//
// Based on the windowed template from 02_compute_architecture.cpp.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#    include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#ifdef ANDROID_BUILD
#  include "glfw_android_shim.h"
#else
#  define GLFW_INCLUDE_VULKAN
#  include <GLFW/glfw3.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr uint32_t kWidth       = 1280;
constexpr uint32_t kHeight      = 720;
constexpr int      kMaxFrames   = 2;
constexpr int      kAcquireSemas = kMaxFrames + 1;

constexpr uint32_t kNumStrands  = 512;     // hair strands around the sphere
constexpr uint32_t kPhysicsGroupSize = 256;

const std::vector<char const *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// Push-constant layout – must match PushConst in the shader exactly
// ---------------------------------------------------------------------------
struct HairPush
{
    uint32_t width;
    uint32_t height;
    uint32_t numStrands;
    float    time;
    float    windStrength;
};
static_assert(sizeof(HairPush) == 20, "push constant size mismatch");

// ---------------------------------------------------------------------------
// HairApp
// ---------------------------------------------------------------------------
class HairApp
{
  public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    // -----------------------------------------------------------------------
    // Window state
    // -----------------------------------------------------------------------
    GLFWwindow *m_window  = nullptr;
    bool        m_resized = false;

    float m_windStrength = 1.0f;   // controlled by +/- keys

    // -----------------------------------------------------------------------
    // Core Vulkan handles
    // -----------------------------------------------------------------------
    vk::raii::Context                m_ctx;
    vk::raii::Instance               m_instance       = nullptr;
    vk::raii::DebugUtilsMessengerEXT m_debugMessenger = nullptr;
    vk::raii::SurfaceKHR             m_surface        = nullptr;
    vk::raii::PhysicalDevice         m_physDev        = nullptr;
    vk::raii::Device                 m_device         = nullptr;
    uint32_t                         m_queueFamily    = ~0u;
    vk::raii::Queue                  m_queue          = nullptr;

    // -----------------------------------------------------------------------
    // Swapchain
    // -----------------------------------------------------------------------
    vk::raii::SwapchainKHR m_swapchain  = nullptr;
    std::vector<vk::Image> m_swapImages;
    vk::SurfaceFormatKHR   m_swapFormat{};
    vk::Extent2D           m_swapExtent{};

    // -----------------------------------------------------------------------
    // Strand tip displacement buffer  (device-local storage)
    // -----------------------------------------------------------------------
    vk::raii::Buffer       m_strandBuf = nullptr;
    vk::raii::DeviceMemory m_strandMem = nullptr;

    // -----------------------------------------------------------------------
    // Descriptor set layout and pipeline layout (shared by both pipelines)
    // -----------------------------------------------------------------------
    vk::raii::DescriptorSetLayout m_dsLayout    = nullptr;
    vk::raii::PipelineLayout      m_pipeLayout  = nullptr;

    // Two compute pipelines – physics and render
    vk::raii::Pipeline m_physPipeline   = nullptr;
    vk::raii::Pipeline m_renderPipeline = nullptr;

    // Command pool (declared before m_frames so it outlives per-frame bufs)
    vk::raii::CommandPool m_cmdPool = nullptr;

    // -----------------------------------------------------------------------
    // Per-frame resources
    // -----------------------------------------------------------------------
    struct PerFrame
    {
        vk::raii::Image        storImg  = nullptr;
        vk::raii::DeviceMemory storMem  = nullptr;
        vk::raii::ImageView    storView = nullptr;

        vk::raii::DescriptorPool dsPool = nullptr;
        vk::DescriptorSet        dsSet  = nullptr;   // raw handle, pool-owned

        vk::raii::CommandBuffer cmdBuf = nullptr;
        vk::raii::Fence         fence  = nullptr;
    };
    std::array<PerFrame, kMaxFrames> m_frames;

    std::vector<vk::raii::Semaphore> m_imageAvail;
    int                              m_acquireIdx = 0;
    std::vector<vk::raii::Semaphore> m_renderDone;

    uint32_t m_frameIdx = 0;

    std::vector<const char *> m_devExts = {vk::KHRSwapchainExtensionName};

    // =======================================================================
    // Window
    // =======================================================================
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        m_window = glfwCreateWindow(kWidth, kHeight,
            "Hair Strands in Wind  |  Subgroup/Wave ops  |  +/- wind  ESC=quit",
            nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, cbResize);
        glfwSetKeyCallback(m_window, cbKey);
    }

    static void cbResize(GLFWwindow *w, int, int)
    {
        static_cast<HairApp *>(glfwGetWindowUserPointer(w))->m_resized = true;
    }

    static void cbKey(GLFWwindow *w, int key, int /*sc*/, int action, int /*mods*/)
    {
        if (action != GLFW_PRESS) return;
        auto *app = static_cast<HairApp *>(glfwGetWindowUserPointer(w));
        switch (key)
        {
            case GLFW_KEY_EQUAL:   app->m_windStrength = std::min(app->m_windStrength + 0.2f, 3.0f); break;
            case GLFW_KEY_MINUS:   app->m_windStrength = std::max(app->m_windStrength - 0.2f, 0.0f); break;
            case GLFW_KEY_R:       app->m_windStrength = 1.0f; break;
            case GLFW_KEY_ESCAPE:  glfwSetWindowShouldClose(w, GLFW_TRUE); break;
            default: break;
        }
    }

    // =======================================================================
    // Vulkan init
    // =======================================================================
    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        createStrandBuffer();
        createSwapchain();
        createDescriptorSetLayout();
        createPipelines();
        createPerFrameResources();
    }

    // =======================================================================
    // Main loop
    // =======================================================================
    void mainLoop()
    {
        auto startTime = std::chrono::steady_clock::now();
        while (!glfwWindowShouldClose(m_window))
        {
            glfwPollEvents();
            auto   now     = std::chrono::steady_clock::now();
            float  elapsed = std::chrono::duration<float>(now - startTime).count();
            drawFrame(elapsed);
        }
        m_device.waitIdle();
    }

    void cleanup()
    {
        m_renderDone.clear();
        m_imageAvail.clear();
        for (auto &f : m_frames)
        {
            f.fence    = nullptr;
            f.cmdBuf   = nullptr;
            f.dsPool   = nullptr;
            f.storView = nullptr;
            f.storMem  = nullptr;
            f.storImg  = nullptr;
        }
        m_cmdPool         = nullptr;
        m_renderPipeline  = nullptr;
        m_physPipeline    = nullptr;
        m_pipeLayout      = nullptr;
        m_dsLayout        = nullptr;
        m_strandBuf       = nullptr;
        m_strandMem       = nullptr;
        m_swapchain       = nullptr;
        m_queue           = nullptr;
        m_device          = nullptr;
        m_surface         = nullptr;
        m_debugMessenger  = nullptr;
        m_instance        = nullptr;

        glfwDestroyWindow(m_window);
        glfwTerminate();
        m_window = nullptr;
    }

    // =======================================================================
    // Instance
    // =======================================================================
    void createInstance()
    {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName   = "Hair Strands in Wind",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = vk::ApiVersion13};

        std::vector<char const *> layers;
        if (kEnableValidation)
            layers.assign(kValidationLayers.begin(), kValidationLayers.end());

        auto layerProps = m_ctx.enumerateInstanceLayerProperties();
        for (auto const *req : layers)
        {
            bool found = std::ranges::any_of(layerProps, [req](auto const &lp) {
                return strcmp(lp.layerName, req) == 0;
            });
            if (!found)
                throw std::runtime_error("Required layer not available: " + std::string(req));
        }

        auto exts     = getRequiredInstanceExtensions();
        auto extProps = m_ctx.enumerateInstanceExtensionProperties();
        for (auto const *req : exts)
        {
            bool found = std::ranges::any_of(extProps, [req](auto const &ep) {
                return strcmp(ep.extensionName, req) == 0;
            });
            if (!found)
                throw std::runtime_error("Required extension not available: " + std::string(req));
        }

        vk::InstanceCreateInfo ci{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames     = layers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(exts.size()),
            .ppEnabledExtensionNames = exts.data()};
        m_instance = vk::raii::Instance(m_ctx, ci);
    }

    void setupDebugMessenger()
    {
        if (!kEnableValidation) return;
        vk::DebugUtilsMessageSeverityFlagsEXT sev(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT type(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        vk::DebugUtilsMessengerCreateInfoEXT ci{
            .messageSeverity = sev,
            .messageType     = type,
            .pfnUserCallback = &debugCallback};
        m_debugMessenger = m_instance.createDebugUtilsMessengerEXT(ci);
    }

    void createSurface()
    {
        VkSurfaceKHR raw;
        if (glfwCreateWindowSurface(*m_instance, m_window, nullptr, &raw) != VK_SUCCESS)
            throw std::runtime_error("failed to create window surface!");
        m_surface = vk::raii::SurfaceKHR(m_instance, raw);
    }

    // =======================================================================
    // Physical device
    // =======================================================================
    void pickPhysicalDevice()
    {
        // Prefer discrete GPU > integrated GPU > virtual GPU > anything else.
        auto typeScore = [](vk::PhysicalDeviceType t) -> int {
            switch (t) {
                case vk::PhysicalDeviceType::eDiscreteGpu:   return 4;
                case vk::PhysicalDeviceType::eIntegratedGpu: return 3;
                case vk::PhysicalDeviceType::eVirtualGpu:    return 2;
                default:                                      return 1;
            }
        };
        int bestScore = 0;
        for (auto &pd : m_instance.enumeratePhysicalDevices())
        {
            auto qfps = pd.getQueueFamilyProperties();
            uint32_t qf = ~0u;
            for (uint32_t i = 0; i < static_cast<uint32_t>(qfps.size()); ++i)
            {
                bool hasCompute = !!(qfps[i].queueFlags & vk::QueueFlagBits::eCompute);
                bool hasPresent = pd.getSurfaceSupportKHR(i, *m_surface);
                if (hasCompute && hasPresent) { qf = i; break; }
            }
            if (qf == ~0u) continue;

            auto devExts    = pd.enumerateDeviceExtensionProperties();
            bool hasSwapchain = std::ranges::any_of(devExts, [](auto const &e) {
                return strcmp(e.extensionName, vk::KHRSwapchainExtensionName) == 0;
            });
            if (!hasSwapchain) continue;

            int score = typeScore(pd.getProperties().deviceType);
            if (score > bestScore) { bestScore = score; m_physDev = pd; m_queueFamily = qf; }
        }
        if (!*m_physDev)
            throw std::runtime_error("No suitable GPU found!");

        // Print subgroup properties — the educational content for this chapter
        vk::PhysicalDeviceSubgroupProperties sgProps{};
        vk::PhysicalDeviceProperties2 props2{.pNext = &sgProps};
        m_physDev.getProperties2(&props2);

        std::cout << "=== Subgroup / Wave Operations ===\n";
        std::cout << "  Device             : " << props2.properties.deviceName.data() << '\n';
        std::cout << "  Subgroup size      : " << sgProps.subgroupSize << '\n';
        std::cout << "  Supported stages   : " << vk::to_string(sgProps.supportedStages) << '\n';
        std::cout << "  Supported ops      : " << vk::to_string(sgProps.supportedOperations) << '\n';
        std::cout << "  (Intel Arc ARL may use SIMD8/16/32 variable subgroup sizes)\n";
        std::cout << "  (Shader uses atomic slot counter pattern for safe wave indexing)\n";
        std::cout << "==================================\n";
    }

    // =======================================================================
    // Logical device
    // =======================================================================
    void createLogicalDevice()
    {
        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features>
            featureChain = {
                {},
                {.scalarBlockLayout = true, .timelineSemaphore = true},
                {.synchronization2  = true, .dynamicRendering  = true}};

        float prio = 1.0f;
        vk::DeviceQueueCreateInfo qci{
            .queueFamilyIndex = m_queueFamily,
            .queueCount       = 1,
            .pQueuePriorities = &prio};
        vk::DeviceCreateInfo dci{
            .pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &qci,
            .enabledExtensionCount   = static_cast<uint32_t>(m_devExts.size()),
            .ppEnabledExtensionNames = m_devExts.data()};
        m_device = vk::raii::Device(m_physDev, dci);
        m_queue  = vk::raii::Queue(m_device, m_queueFamily, 0);
    }

    // =======================================================================
    // Command pool
    // =======================================================================
    void createCommandPool()
    {
        vk::CommandPoolCreateInfo ci{
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = m_queueFamily};
        m_cmdPool = vk::raii::CommandPool(m_device, ci);
    }

    // =======================================================================
    // Strand tip displacement buffer
    // One float per strand; physics dispatch writes here, render dispatch reads.
    // =======================================================================
    void createStrandBuffer()
    {
        vk::DeviceSize sz = kNumStrands * sizeof(float);
        vk::BufferCreateInfo bci{
            .size        = sz,
            .usage       = vk::BufferUsageFlagBits::eStorageBuffer,
            .sharingMode = vk::SharingMode::eExclusive};
        m_strandBuf = vk::raii::Buffer(m_device, bci);

        auto memReqs = m_strandBuf.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = memReqs.size,
            .memoryTypeIndex = findMemoryType(
                memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)};
        m_strandMem = vk::raii::DeviceMemory(m_device, mai);
        m_strandBuf.bindMemory(*m_strandMem, 0);
    }

    // =======================================================================
    // Swapchain
    // =======================================================================
    void createSwapchain(vk::SwapchainKHR oldSwapchain = nullptr)
    {
        auto caps        = m_physDev.getSurfaceCapabilitiesKHR(*m_surface);
        m_swapExtent     = chooseExtent(caps);
        auto fmts        = m_physDev.getSurfaceFormatsKHR(*m_surface);
        m_swapFormat     = chooseFormat(fmts, m_physDev);
        auto modes       = m_physDev.getSurfacePresentModesKHR(*m_surface);
        auto presentMode = chooseMode(modes);

        uint32_t imgCount = std::max(3u, caps.minImageCount);
        if (caps.maxImageCount > 0u)
            imgCount = std::min(imgCount, caps.maxImageCount);

        vk::SwapchainCreateInfoKHR sci{
            .surface          = *m_surface,
            .minImageCount    = imgCount,
            .imageFormat      = m_swapFormat.format,
            .imageColorSpace  = m_swapFormat.colorSpace,
            .imageExtent      = m_swapExtent,
            .imageArrayLayers = 1,
            .imageUsage       = vk::ImageUsageFlagBits::eTransferDst,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform     = caps.currentTransform,
            .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode      = presentMode,
            .clipped          = true,
            .oldSwapchain     = oldSwapchain};
        m_swapchain  = vk::raii::SwapchainKHR(m_device, sci);
        m_swapImages = m_swapchain.getImages();
    }

    // =======================================================================
    // Descriptor set layout
    //   binding 0 = storage image (render output)
    //   binding 1 = storage buffer (strand tip displacements)
    // =======================================================================
    void createDescriptorSetLayout()
    {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {
                .binding         = 0,
                .descriptorType  = vk::DescriptorType::eStorageImage,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute},
            {
                .binding         = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute}
        }};
        vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings    = bindings.data()};
        m_dsLayout = vk::raii::DescriptorSetLayout(m_device, ci);
    }

    // =======================================================================
    // Compute pipelines (physics + render)
    // =======================================================================
    void createPipelines()
    {
        auto code = readFile("shaders/slang.spv");
        vk::ShaderModuleCreateInfo smci{
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<uint32_t const *>(code.data())};
        vk::raii::ShaderModule shaderModule(m_device, smci);

        vk::PushConstantRange pcRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(HairPush)};
        vk::PipelineLayoutCreateInfo plci{
            .setLayoutCount         = 1,
            .pSetLayouts            = &*m_dsLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcRange};
        m_pipeLayout = vk::raii::PipelineLayout(m_device, plci);

        auto makePipeline = [&](const char *entry) {
            vk::PipelineShaderStageCreateInfo stage{
                .stage  = vk::ShaderStageFlagBits::eCompute,
                .module = *shaderModule,
                .pName  = entry};
            vk::ComputePipelineCreateInfo pci{.stage = stage, .layout = *m_pipeLayout};
            return vk::raii::Pipeline(m_device, nullptr, pci);
        };

        m_physPipeline   = makePipeline("physicsMain");
        m_renderPipeline = makePipeline("renderMain");
    }

    // =======================================================================
    // Per-frame resources
    // =======================================================================
    void createPerFrameResources()
    {
        vk::CommandBufferAllocateInfo cbai{
            .commandPool        = *m_cmdPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = kMaxFrames};
        auto cmdBufs = vk::raii::CommandBuffers(m_device, cbai);

        for (int i = 0; i < kMaxFrames; ++i)
        {
            auto &f = m_frames[i];
            createStorageImage(f);

            // Descriptor pool: 1 storage image + 1 storage buffer
            std::array<vk::DescriptorPoolSize, 2> poolSizes{{
                {.type = vk::DescriptorType::eStorageImage,  .descriptorCount = 1},
                {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1}
            }};
            vk::DescriptorPoolCreateInfo dpci{
                .maxSets       = 1,
                .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                .pPoolSizes    = poolSizes.data()};
            f.dsPool = vk::raii::DescriptorPool(m_device, dpci);

            // Descriptor set
            vk::DescriptorSetAllocateInfo dsai{
                .descriptorPool     = *f.dsPool,
                .descriptorSetCount = 1,
                .pSetLayouts        = &*m_dsLayout};
            f.dsSet = vk::raii::DescriptorSets(m_device, dsai)[0].release();

            // Bind storage image (binding 0) and strand buffer (binding 1)
            vk::DescriptorImageInfo imgInfo{
                .imageView   = *f.storView,
                .imageLayout = vk::ImageLayout::eGeneral};
            vk::DescriptorBufferInfo bufInfo{
                .buffer = *m_strandBuf,
                .offset = 0,
                .range  = kNumStrands * sizeof(float)};
            std::array<vk::WriteDescriptorSet, 2> writes{{
                {
                    .dstSet          = f.dsSet,
                    .dstBinding      = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eStorageImage,
                    .pImageInfo      = &imgInfo},
                {
                    .dstSet          = f.dsSet,
                    .dstBinding      = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo     = &bufInfo}
            }};
            m_device.updateDescriptorSets(writes, {});

            f.cmdBuf = std::move(cmdBufs[i]);
            f.fence  = vk::raii::Fence(m_device, vk::FenceCreateInfo{
                .flags = vk::FenceCreateFlagBits::eSignaled});
        }

        // Acquire semaphores: rotating pool of kAcquireSemas
        m_imageAvail.clear();
        for (int i = 0; i < kAcquireSemas; ++i)
            m_imageAvail.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        // renderDone semaphores: indexed by swapchain image index
        m_renderDone.clear();
        for (size_t i = 0; i < m_swapImages.size(); ++i)
            m_renderDone.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        transitionStorageImagesToGeneral();
    }

    void createStorageImage(PerFrame &f)
    {
        vk::ImageCreateInfo ici{
            .imageType     = vk::ImageType::e2D,
            .format        = vk::Format::eR8G8B8A8Unorm,
            .extent        = {m_swapExtent.width, m_swapExtent.height, 1},
            .mipLevels     = 1,
            .arrayLayers   = 1,
            .samples       = vk::SampleCountFlagBits::e1,
            .tiling        = vk::ImageTiling::eOptimal,
            .usage         = vk::ImageUsageFlagBits::eStorage |
                             vk::ImageUsageFlagBits::eTransferSrc,
            .sharingMode   = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined};
        f.storImg = vk::raii::Image(m_device, ici);

        auto memReqs = f.storImg.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = memReqs.size,
            .memoryTypeIndex = findMemoryType(
                memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)};
        f.storMem = vk::raii::DeviceMemory(m_device, mai);
        f.storImg.bindMemory(*f.storMem, 0);

        vk::ImageViewCreateInfo ivci{
            .image            = *f.storImg,
            .viewType         = vk::ImageViewType::e2D,
            .format           = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        f.storView = vk::raii::ImageView(m_device, ivci);
    }

    void transitionStorageImagesToGeneral()
    {
        vk::CommandBufferAllocateInfo cbai{
            .commandPool        = *m_cmdPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1};
        auto cb = std::move(vk::raii::CommandBuffers(m_device, cbai).front());
        cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        for (auto &f : m_frames)
        {
            vk::ImageMemoryBarrier2 barrier{
                .srcStageMask        = vk::PipelineStageFlagBits2::eNone,
                .srcAccessMask       = vk::AccessFlagBits2::eNone,
                .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask       = vk::AccessFlagBits2::eShaderWrite,
                .oldLayout           = vk::ImageLayout::eUndefined,
                .newLayout           = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image               = *f.storImg,
                .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
            cb.pipelineBarrier2(vk::DependencyInfo{
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers    = &barrier});
        }

        cb.end();
        vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
        m_queue.submit(si, nullptr);
        m_queue.waitIdle();
    }

    // =======================================================================
    // Draw frame
    // =======================================================================
    void drawFrame(float elapsed)
    {
        auto &f = m_frames[m_frameIdx];

        auto waitRes = m_device.waitForFences(*f.fence, vk::True, UINT64_MAX);
        if (waitRes != vk::Result::eSuccess)
            throw std::runtime_error("waitForFences failed");

        auto &acqSem = m_imageAvail[m_acquireIdx];
        m_acquireIdx = (m_acquireIdx + 1) % kAcquireSemas;

        uint32_t imageIndex;
        {
            auto [res, idx] = m_swapchain.acquireNextImage(UINT64_MAX, *acqSem, nullptr);
            if (res == vk::Result::eErrorOutOfDateKHR)
            {
                recreateSwapchain();
                return;
            }
            imageIndex = idx;
        }

        m_device.resetFences(*f.fence);
        recordCommands(f, imageIndex, elapsed);

        auto &rdSem = m_renderDone[imageIndex];

        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
        vk::SubmitInfo si{
            .waitSemaphoreCount   = 1,
            .pWaitSemaphores      = &*acqSem,
            .pWaitDstStageMask    = &waitStage,
            .commandBufferCount   = 1,
            .pCommandBuffers      = &*f.cmdBuf,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores    = &*rdSem};
        m_queue.submit(si, *f.fence);

        vk::PresentInfoKHR pi{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &*rdSem,
            .swapchainCount     = 1,
            .pSwapchains        = &*m_swapchain,
            .pImageIndices      = &imageIndex};
        auto pres = m_queue.presentKHR(pi);
        if (pres == vk::Result::eSuboptimalKHR ||
            pres == vk::Result::eErrorOutOfDateKHR ||
            m_resized)
        {
            m_resized = false;
            recreateSwapchain();
        }

        m_frameIdx = (m_frameIdx + 1) % kMaxFrames;
    }

    void recordCommands(PerFrame &f, uint32_t imageIndex, float elapsed)
    {
        auto &cb = f.cmdBuf;
        cb.reset();
        cb.begin({});

        HairPush push{
            .width        = m_swapExtent.width,
            .height       = m_swapExtent.height,
            .numStrands   = kNumStrands,
            .time         = elapsed,
            .windStrength = m_windStrength};

        // ---- Dispatch 1: physics (WaveActiveSum + WavePrefixSum) ----
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_physPipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_pipeLayout,
                              0, {f.dsSet}, {});
        cb.pushConstants<HairPush>(*m_pipeLayout,
                                   vk::ShaderStageFlagBits::eCompute, 0, push);

        // Dispatch enough groups to cover all strands (up to 256 per group)
        uint32_t physGroups = (kNumStrands + kPhysicsGroupSize - 1u) / kPhysicsGroupSize;
        cb.dispatch(physGroups, 1, 1);

        // Barrier: physics writes strandTip → render reads strandTip
        vk::MemoryBarrier2 bufBarrier{
            .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers    = &bufBarrier});

        // ---- Dispatch 2: render (WaveActiveBallot for culling) ----
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_renderPipeline);
        // descriptor set + push constants already bound; re-bind for completeness
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_pipeLayout,
                              0, {f.dsSet}, {});
        cb.pushConstants<HairPush>(*m_pipeLayout,
                                   vk::ShaderStageFlagBits::eCompute, 0, push);

        uint32_t gx = (m_swapExtent.width  + 15u) / 16u;
        uint32_t gy = (m_swapExtent.height + 15u) / 16u;
        cb.dispatch(gx, gy, 1);

        // ---- Barriers: storImg compute→transfer, swapchain →TRANSFER_DST ----
        vk::ImageMemoryBarrier2 storToTransfer{
            .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask       = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask        = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask       = vk::AccessFlagBits2::eTransferRead,
            .oldLayout           = vk::ImageLayout::eGeneral,
            .newLayout           = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = *f.storImg,
            .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        vk::ImageMemoryBarrier2 swapToTransfer{
            .srcStageMask        = vk::PipelineStageFlagBits2::eNone,
            .srcAccessMask       = vk::AccessFlagBits2::eNone,
            .dstStageMask        = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask       = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout           = vk::ImageLayout::eUndefined,
            .newLayout           = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_swapImages[imageIndex],
            .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

        std::array<vk::ImageMemoryBarrier2, 2> preBlitBarriers{storToTransfer, swapToTransfer};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<uint32_t>(preBlitBarriers.size()),
            .pImageMemoryBarriers    = preBlitBarriers.data()});

        // ---- Blit storage image → swapchain ----
        vk::ImageSubresourceLayers subres{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        vk::Offset3D zero{0, 0, 0};
        vk::Offset3D ext{
            static_cast<int32_t>(m_swapExtent.width),
            static_cast<int32_t>(m_swapExtent.height), 1};
        vk::ImageBlit2 region{
            .srcSubresource = subres,
            .srcOffsets     = std::array<vk::Offset3D, 2>{zero, ext},
            .dstSubresource = subres,
            .dstOffsets     = std::array<vk::Offset3D, 2>{zero, ext}};
        vk::BlitImageInfo2 blitInfo{
            .srcImage       = *f.storImg,
            .srcImageLayout = vk::ImageLayout::eGeneral,
            .dstImage       = m_swapImages[imageIndex],
            .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
            .regionCount    = 1,
            .pRegions       = &region,
            .filter         = vk::Filter::eNearest};
        cb.blitImage2(blitInfo);

        // ---- Post-blit barriers ----
        vk::ImageMemoryBarrier2 swapToPresent{
            .srcStageMask        = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask       = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask        = vk::PipelineStageFlagBits2::eBottomOfPipe,
            .dstAccessMask       = vk::AccessFlagBits2::eNone,
            .oldLayout           = vk::ImageLayout::eTransferDstOptimal,
            .newLayout           = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_swapImages[imageIndex],
            .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        vk::ImageMemoryBarrier2 storRelease{
            .srcStageMask        = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask       = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask       = vk::AccessFlagBits2::eShaderStorageWrite,
            .oldLayout           = vk::ImageLayout::eGeneral,
            .newLayout           = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = *f.storImg,
            .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

        std::array<vk::ImageMemoryBarrier2, 2> postBlitBarriers{swapToPresent, storRelease};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<uint32_t>(postBlitBarriers.size()),
            .pImageMemoryBarriers    = postBlitBarriers.data()});

        cb.end();
    }

    // =======================================================================
    // Swapchain recreation
    // =======================================================================
    void recreateSwapchain()
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(m_window, &w, &h);
        while (w == 0 || h == 0)
        {
            glfwGetFramebufferSize(m_window, &w, &h);
            glfwWaitEvents();
        }
        m_device.waitIdle();

        for (auto &f : m_frames)
        {
            f.storView = nullptr;
            f.storImg  = nullptr;
            f.storMem  = nullptr;
            f.dsPool   = nullptr;
            f.dsSet    = nullptr;
        }

        vk::SwapchainKHR oldHandle = *m_swapchain;
        createSwapchain(oldHandle);

        for (auto &f : m_frames)
            createStorageImage(f);

        for (auto &f : m_frames)
        {
            std::array<vk::DescriptorPoolSize, 2> poolSizes{{
                {.type = vk::DescriptorType::eStorageImage,  .descriptorCount = 1},
                {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1}
            }};
            vk::DescriptorPoolCreateInfo dpci{
                .maxSets       = 1,
                .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                .pPoolSizes    = poolSizes.data()};
            f.dsPool = vk::raii::DescriptorPool(m_device, dpci);

            vk::DescriptorSetAllocateInfo dsai{
                .descriptorPool     = *f.dsPool,
                .descriptorSetCount = 1,
                .pSetLayouts        = &*m_dsLayout};
            f.dsSet = vk::raii::DescriptorSets(m_device, dsai)[0].release();

            vk::DescriptorImageInfo imgInfo{
                .imageView   = *f.storView,
                .imageLayout = vk::ImageLayout::eGeneral};
            vk::DescriptorBufferInfo bufInfo{
                .buffer = *m_strandBuf,
                .offset = 0,
                .range  = kNumStrands * sizeof(float)};
            std::array<vk::WriteDescriptorSet, 2> writes{{
                {
                    .dstSet          = f.dsSet,
                    .dstBinding      = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eStorageImage,
                    .pImageInfo      = &imgInfo},
                {
                    .dstSet          = f.dsSet,
                    .dstBinding      = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo     = &bufInfo}
            }};
            m_device.updateDescriptorSets(writes, {});
        }

        m_renderDone.clear();
        for (size_t i = 0; i < m_swapImages.size(); ++i)
            m_renderDone.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        transitionStorageImagesToGeneral();
    }

    // =======================================================================
    // Helpers
    // =======================================================================
    [[nodiscard]] uint32_t findMemoryType(uint32_t filter, vk::MemoryPropertyFlags props) const
    {
        auto memProps = m_physDev.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        {
            if ((filter & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("no suitable memory type");
    }

    static vk::SurfaceFormatKHR chooseFormat(std::vector<vk::SurfaceFormatKHR> const &formats,
                                              vk::raii::PhysicalDevice const &physDev)
    {
        assert(!formats.empty());
        auto supportsBlitDst = [&](vk::Format fmt) {
            return !!(physDev.getFormatProperties(fmt).optimalTilingFeatures &
                      vk::FormatFeatureFlagBits::eBlitDst);
        };
        for (auto const &f : formats)
            if (f.format == vk::Format::eB8G8R8A8Unorm &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
                supportsBlitDst(f.format))
                return f;
        for (auto const &f : formats)
            if (f.format == vk::Format::eB8G8R8A8Unorm &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return f;
        for (auto const &f : formats)
            if (f.format == vk::Format::eB8G8R8A8Srgb &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return f;
        return formats[0];
    }

    static vk::PresentModeKHR chooseMode(std::vector<vk::PresentModeKHR> const &modes)
    {
        for (auto m : modes)
            if (m == vk::PresentModeKHR::eMailbox)
                return m;
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseExtent(vk::SurfaceCapabilitiesKHR const &caps)
    {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return caps.currentExtent;
        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);
        return {
            std::clamp<uint32_t>(w, caps.minImageExtent.width,  caps.maxImageExtent.width),
            std::clamp<uint32_t>(h, caps.minImageExtent.height, caps.maxImageExtent.height)};
    }

    [[nodiscard]] std::vector<char const *> getRequiredInstanceExtensions() const
    {
        uint32_t count = 0;
        auto raw = glfwGetRequiredInstanceExtensions(&count);
        std::vector<char const *> exts(raw, raw + count);
        if (kEnableValidation)
            exts.push_back(vk::EXTDebugUtilsExtensionName);
        return exts;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT      severity,
        vk::DebugUtilsMessageTypeFlagsEXT             type,
        vk::DebugUtilsMessengerCallbackDataEXT const *pData,
        void *)
    {
        if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
            std::cerr << "validation [" << to_string(type) << "]: " << pData->pMessage << '\n';
        return vk::False;
    }

    static std::vector<char> readFile(std::string const &path)
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("failed to open shader: " + path);
        std::vector<char> buf(file.tellg());
        file.seekg(0);
        file.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        return buf;
    }
};

// ---------------------------------------------------------------------------
#ifndef ANDROID_BUILD
int main()
{
    try
    {
        HairApp app;
        app.run();
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
#endif // ANDROID_BUILD

#ifdef ANDROID_BUILD
extern "C" void chapter04_run() {
    try { HairApp{}.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh04", "%s", e.what()); }
}
#endif
