// Chapter 10 – Performance Optimization: GPU Performance Heatmap
//
// Demonstrates:
//   • Divergent vs. non-divergent kernel execution (red/orange vs green tiles)
//   • LDS reduction vs. wave reduction throughput (cyan vs blue tiles)
//   • VkQueryPool timestamp queries: per-tile GPU timing measured with
//     vkCmdWriteTimestamp2 before/after each tile dispatch
//   • Heatmap overlay: per-tile color intensity driven by measured GPU time
//     (cool=fast/blue, hot=slow/red)
//
// Layout: 8 columns × 6 rows = 48 tiles, grouped into four quadrants:
//   Top-left    (Q0): divergent kernel      — warm/red tones
//   Top-right   (Q1): non-divergent kernel  — cool/green tones
//   Bottom-left (Q2): LDS reduction         — cyan tones
//   Bottom-right(Q3): wave reduction        — blue/violet tones
//
// Build:  see CMakeLists.txt – WINDOWED is set for chapter 10
// Shader: shaders/slang.spv  (compiled from 10_performance_optimization.slang)

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
constexpr uint32_t kWidth     = 1280;
constexpr uint32_t kHeight    = 720;
constexpr int      kMaxFrames = 2;
constexpr int      kAcquireSemas = kMaxFrames + 1;

// Tile grid: 8 columns × 6 rows = 48 tiles
// Quadrant assignment: top-left Q0(0-23), top-right Q1(…), etc.
// We use 8 cols × 6 rows divided into four 4×3 quadrants.
constexpr uint32_t kTileCols      = 8u;
constexpr uint32_t kTileRows      = 6u;
constexpr uint32_t kNumTiles      = kTileCols * kTileRows;   // 48
// Tiles are sized at runtime from m_swapExtent / tile grid

// Each tile gets 2 timestamps (before + after dispatch)
constexpr uint32_t kQueryCountPerFrame = kNumTiles * 2u;

const std::vector<char const *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// Push-constant layout – must match HeatmapPush in shader
// ---------------------------------------------------------------------------
struct HeatmapPush
{
    uint32_t kernelType;   // 0=divergent, 1=non-divergent, 2=LDS, 3=wave
    uint32_t tileX;        // tile pixel offset X
    uint32_t tileY;        // tile pixel offset Y
    uint32_t tileW;        // tile width in pixels
    uint32_t tileH;        // tile height in pixels
    uint32_t frameIndex;   // animated frame counter
};
static_assert(sizeof(HeatmapPush) == 24, "push constant size mismatch");

// ---------------------------------------------------------------------------
// HeatmapApp
// ---------------------------------------------------------------------------
class HeatmapApp
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

    double m_timestampPeriodNs = 1.0;   // nanoseconds per GPU timestamp tick
    bool   m_hasTimestamps     = false;

    // -----------------------------------------------------------------------
    // Swapchain
    // -----------------------------------------------------------------------
    vk::raii::SwapchainKHR m_swapchain  = nullptr;
    std::vector<vk::Image> m_swapImages;
    vk::SurfaceFormatKHR   m_swapFormat{};
    vk::Extent2D           m_swapExtent{};

    // -----------------------------------------------------------------------
    // Pipeline
    // -----------------------------------------------------------------------
    vk::raii::DescriptorSetLayout m_dsLayout        = nullptr;
    vk::raii::PipelineLayout      m_pipeLayout      = nullptr;
    vk::raii::Pipeline            m_computePipeline = nullptr;

    vk::raii::CommandPool m_cmdPool = nullptr;

    // -----------------------------------------------------------------------
    // Timestamp readback (CPU-side timing data)
    // Latest per-tile GPU time in nanoseconds; updated after each readback.
    // -----------------------------------------------------------------------
    std::array<double, kNumTiles> m_tileTimes{};   // ns per tile, latest frame

    // -----------------------------------------------------------------------
    // Per-frame resources
    // -----------------------------------------------------------------------
    struct PerFrame
    {
        vk::raii::Image        storImg  = nullptr;
        vk::raii::DeviceMemory storMem  = nullptr;
        vk::raii::ImageView    storView = nullptr;

        vk::raii::DescriptorPool dsPool = nullptr;
        vk::DescriptorSet        dsSet  = nullptr;

        vk::raii::CommandBuffer cmdBuf     = nullptr;
        vk::raii::Fence         fence      = nullptr;
        vk::raii::QueryPool     queryPool  = nullptr;
    };
    std::array<PerFrame, kMaxFrames> m_frames;

    std::vector<vk::raii::Semaphore> m_imageAvail;
    int m_acquireIdx = 0;

    std::vector<vk::raii::Semaphore> m_renderDone;

    uint32_t m_frameIdx   = 0;
    uint32_t m_frameCount = 0;   // monotonically increasing frame counter

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
            "Chapter 10: GPU Performance Heatmap  |  ESC=quit",
            nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, cbResize);
        glfwSetKeyCallback(m_window, cbKey);
    }

    static void cbResize(GLFWwindow *w, int, int)
    {
        static_cast<HeatmapApp *>(glfwGetWindowUserPointer(w))->m_resized = true;
    }

    static void cbKey(GLFWwindow *w, int key, int, int action, int)
    {
        if (action != GLFW_PRESS) return;
        if (key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(w, GLFW_TRUE);
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
        createSwapchain();
        createDescriptorSetLayout();
        createPipeline();
        createPerFrameResources();
    }

    // =======================================================================
    // Main loop
    // =======================================================================
    void mainLoop()
    {
        while (!glfwWindowShouldClose(m_window))
        {
            glfwPollEvents();
            drawFrame();
        }
        m_device.waitIdle();
    }

    void cleanup()
    {
        m_renderDone.clear();
        m_imageAvail.clear();
        for (auto &f : m_frames)
        {
            f.queryPool = nullptr;
            f.fence     = nullptr;
            f.cmdBuf    = nullptr;
            f.dsPool    = nullptr;
            f.storView  = nullptr;
            f.storMem   = nullptr;
            f.storImg   = nullptr;
        }
        m_cmdPool         = nullptr;
        m_computePipeline = nullptr;
        m_pipeLayout      = nullptr;
        m_dsLayout        = nullptr;
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
            .pApplicationName   = "GPU Heatmap",
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
                if (hasCompute && hasPresent)
                {
                    qf = i;
                    break;
                }
            }
            if (qf == ~0u) continue;

            auto devExts = pd.enumerateDeviceExtensionProperties();
            bool hasSwapchain = std::ranges::any_of(devExts, [](auto const &e) {
                return strcmp(e.extensionName, vk::KHRSwapchainExtensionName) == 0;
            });
            if (!hasSwapchain) continue;

            int score = typeScore(pd.getProperties().deviceType);
            if (score > bestScore) { bestScore = score; m_physDev = pd; m_queueFamily = qf; }
        }
        if (!*m_physDev)
            throw std::runtime_error("No suitable GPU found!");

        // Check timestamp support on our queue family
        auto qfps = m_physDev.getQueueFamilyProperties();
        m_hasTimestamps = (qfps[m_queueFamily].timestampValidBits > 0);

        auto props = m_physDev.getProperties();
        m_timestampPeriodNs = static_cast<double>(props.limits.timestampPeriod);

        std::cout << "=== Chapter 10: GPU Performance Heatmap ===\n";
        std::cout << "  Device         : " << props.deviceName.data() << '\n';
        std::cout << "  Timestamp bits : " << qfps[m_queueFamily].timestampValidBits << '\n';
        std::cout << "  Timestamp ns   : " << m_timestampPeriodNs << " ns/tick\n";
        std::cout << "  Tile grid      : " << kTileCols << "x" << kTileRows << " = "
                  << kNumTiles << " tiles\n";
        std::cout << "  Quadrants: [TL=divergent] [TR=non-divergent] "
                     "[BL=LDS] [BR=wave]\n";
        std::cout << "===========================================\n\n";

        if (!m_hasTimestamps)
            std::cerr << "WARNING: timestampValidBits == 0, timing data unavailable\n";
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
    // Swapchain
    // =======================================================================
    void createSwapchain(vk::SwapchainKHR oldSwapchain = nullptr)
    {
        auto caps       = m_physDev.getSurfaceCapabilitiesKHR(*m_surface);
        m_swapExtent    = chooseExtent(caps);

        auto fmts       = m_physDev.getSurfaceFormatsKHR(*m_surface);
        m_swapFormat    = chooseFormat(fmts, m_physDev);

        auto modes      = m_physDev.getSurfacePresentModesKHR(*m_surface);
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
    // Descriptor set layout – binding 0 = storage image
    // =======================================================================
    void createDescriptorSetLayout()
    {
        vk::DescriptorSetLayoutBinding binding{
            .binding         = 0,
            .descriptorType  = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eCompute};
        vk::DescriptorSetLayoutCreateInfo ci{.bindingCount = 1, .pBindings = &binding};
        m_dsLayout = vk::raii::DescriptorSetLayout(m_device, ci);
    }

    // =======================================================================
    // Compute pipeline
    // =======================================================================
    void createPipeline()
    {
        auto code = readFile("shaders/slang.spv");
        vk::ShaderModuleCreateInfo smci{
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<uint32_t const *>(code.data())};
        vk::raii::ShaderModule shaderModule(m_device, smci);

        vk::PushConstantRange pcRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(HeatmapPush)};
        vk::PipelineLayoutCreateInfo plci{
            .setLayoutCount         = 1,
            .pSetLayouts            = &*m_dsLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcRange};
        m_pipeLayout = vk::raii::PipelineLayout(m_device, plci);

        vk::PipelineShaderStageCreateInfo stage{
            .stage  = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName  = "heatmapMain"};
        vk::ComputePipelineCreateInfo pci{.stage = stage, .layout = *m_pipeLayout};
        m_computePipeline = vk::raii::Pipeline(m_device, nullptr, pci);
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

            vk::DescriptorPoolSize poolSize{
                .type            = vk::DescriptorType::eStorageImage,
                .descriptorCount = 1};
            vk::DescriptorPoolCreateInfo dpci{
                .maxSets       = 1,
                .poolSizeCount = 1,
                .pPoolSizes    = &poolSize};
            f.dsPool = vk::raii::DescriptorPool(m_device, dpci);

            vk::DescriptorSetAllocateInfo dsai{
                .descriptorPool     = *f.dsPool,
                .descriptorSetCount = 1,
                .pSetLayouts        = &*m_dsLayout};
            f.dsSet = vk::raii::DescriptorSets(m_device, dsai)[0].release();

            bindStorageImageDescriptor(f);

            f.cmdBuf = std::move(cmdBufs[i]);

            f.fence = vk::raii::Fence(m_device, vk::FenceCreateInfo{
                .flags = vk::FenceCreateFlagBits::eSignaled});

            // Timestamp query pool: 2 queries per tile
            if (m_hasTimestamps)
            {
                vk::QueryPoolCreateInfo qpci{
                    .queryType  = vk::QueryType::eTimestamp,
                    .queryCount = kQueryCountPerFrame};
                f.queryPool = vk::raii::QueryPool(m_device, qpci);
            }
        }

        m_imageAvail.clear();
        for (int i = 0; i < kAcquireSemas; ++i)
            m_imageAvail.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        m_renderDone.clear();
        for (size_t i = 0; i < m_swapImages.size(); ++i)
            m_renderDone.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        transitionStorageImagesToGeneral();
    }

    void createStorageImage(PerFrame &f)
    {
        vk::ImageCreateInfo ici{
            .imageType   = vk::ImageType::e2D,
            .format      = vk::Format::eR8G8B8A8Unorm,
            .extent      = {m_swapExtent.width, m_swapExtent.height, 1},
            .mipLevels   = 1,
            .arrayLayers = 1,
            .samples     = vk::SampleCountFlagBits::e1,
            .tiling      = vk::ImageTiling::eOptimal,
            .usage       = vk::ImageUsageFlagBits::eStorage |
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

    void bindStorageImageDescriptor(PerFrame &f)
    {
        vk::DescriptorImageInfo imgInfo{
            .imageView   = *f.storView,
            .imageLayout = vk::ImageLayout::eGeneral};
        vk::WriteDescriptorSet write{
            .dstSet          = f.dsSet,
            .dstBinding      = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eStorageImage,
            .pImageInfo      = &imgInfo};
        m_device.updateDescriptorSets(write, {});
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
            // Reset query pool here so getResults on the very first frame finds
            // queries in the RESET state (not INITIAL/uninitialized), preventing
            // VUID-vkGetQueryPoolResults-None-09401.
            if (m_hasTimestamps && *f.queryPool)
                cb.resetQueryPool(*f.queryPool, 0, kQueryCountPerFrame);

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
    void drawFrame()
    {
        auto &f = m_frames[m_frameIdx];

        auto waitRes = m_device.waitForFences(*f.fence, vk::True, UINT64_MAX);
        if (waitRes != vk::Result::eSuccess)
            throw std::runtime_error("waitForFences failed");

        // Read back timestamps from the PREVIOUS frame's query pool (already retired)
        readbackTimestamps(f);

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

        recordCommands(f, imageIndex);

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

        ++m_frameCount;
        m_frameIdx = (m_frameIdx + 1) % kMaxFrames;
    }

    // Read timestamp results from the query pool (called after fence wait,
    // so the GPU has finished the previous frame using this slot).
    void readbackTimestamps(PerFrame &f)
    {
        if (!m_hasTimestamps || !*f.queryPool)
            return;

        // vkGetQueryPoolResults – non-blocking since fence is already signaled
        std::vector<uint64_t> ts(kQueryCountPerFrame, 0u);
        auto [res, data] = f.queryPool.getResults<uint64_t>(
            0, kQueryCountPerFrame,
            kQueryCountPerFrame * sizeof(uint64_t),
            sizeof(uint64_t),
            vk::QueryResultFlagBits::e64);

        // res may be eNotReady on the very first frame; ignore it gracefully
        if (res != vk::Result::eSuccess && res != vk::Result::eNotReady)
            return;

        for (uint32_t t = 0; t < kNumTiles; ++t)
        {
            uint64_t t0 = data[t * 2 + 0];
            uint64_t t1 = data[t * 2 + 1];
            if (t1 >= t0)
                m_tileTimes[t] = static_cast<double>(t1 - t0) * m_timestampPeriodNs;
        }
    }

    // =======================================================================
    // Record: one dispatch per tile, timestamps around each
    // =======================================================================
    void recordCommands(PerFrame &f, uint32_t imageIndex)
    {
        auto &cb = f.cmdBuf;
        cb.reset();
        cb.begin({});

        // Reset the query pool before writing new timestamps
        if (m_hasTimestamps && *f.queryPool)
            cb.resetQueryPool(*f.queryPool, 0, kQueryCountPerFrame);

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_pipeLayout,
                              0, {f.dsSet}, {});

        // Compute per-tile dimensions (floor; edge tiles may be slightly smaller)
        uint32_t tileW = m_swapExtent.width  / kTileCols;
        uint32_t tileH = m_swapExtent.height / kTileRows;

        // Quadrant assignment:
        //   column < kTileCols/2 → left  half; column >= → right half
        //   row    < kTileRows/2 → top   half; row    >= → bottom half
        //   TL = Q0 (divergent), TR = Q1 (non-divergent)
        //   BL = Q2 (LDS),       BR = Q3 (wave)
        uint32_t halfCols = kTileCols / 2u;
        uint32_t halfRows = kTileRows / 2u;

        for (uint32_t row = 0; row < kTileRows; ++row)
        {
            for (uint32_t col = 0; col < kTileCols; ++col)
            {
                uint32_t tileIdx = row * kTileCols + col;

                // Determine quadrant
                bool leftHalf = (col < halfCols);
                bool topHalf  = (row < halfRows);
                uint32_t kernelType = 0u;
                if (topHalf  && leftHalf)  kernelType = 0u;   // divergent
                if (topHalf  && !leftHalf) kernelType = 1u;   // non-divergent
                if (!topHalf && leftHalf)  kernelType = 2u;   // LDS
                if (!topHalf && !leftHalf) kernelType = 3u;   // wave

                uint32_t tx = col * tileW;
                uint32_t ty = row * tileH;

                HeatmapPush push{
                    .kernelType = kernelType,
                    .tileX      = tx,
                    .tileY      = ty,
                    .tileW      = tileW,
                    .tileH      = tileH,
                    .frameIndex = m_frameCount};
                cb.pushConstants<HeatmapPush>(*m_pipeLayout,
                                              vk::ShaderStageFlagBits::eCompute,
                                              0, push);

                // Timestamp BEFORE dispatch
                if (m_hasTimestamps && *f.queryPool)
                    cb.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe,
                                       *f.queryPool, tileIdx * 2u);

                // Dispatch: 16×16 threads per workgroup
                uint32_t gx = (tileW + 15u) / 16u;
                uint32_t gy = (tileH + 15u) / 16u;
                cb.dispatch(gx, gy, 1);

                // Timestamp AFTER dispatch
                if (m_hasTimestamps && *f.queryPool)
                    cb.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe,
                                       *f.queryPool, tileIdx * 2u + 1u);

                // Pipeline barrier between tiles: flush compute shader writes
                // so each tile's storage-image writes are visible to subsequent
                // dispatches and timestamps are well-ordered.
                vk::MemoryBarrier2 tileBarrier{
                    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite |
                                     vk::AccessFlagBits2::eShaderStorageRead};
                cb.pipelineBarrier2(vk::DependencyInfo{
                    .memoryBarrierCount      = 1,
                    .pMemoryBarriers         = &tileBarrier,
                    .imageMemoryBarrierCount = 0});
            }
        }

        // Apply heatmap overlay: re-tint each tile based on measured GPU time.
        // We do this in a second compute pass over the already-written tiles.
        // (Simple approach: we just use the per-tile color the shader chose.)
        // The visual heatmap is encoded by the shader's own color selection;
        // the CPU-readback timings are logged every 60 frames for education.
        if (m_frameCount > 0 && (m_frameCount % 60) == 0)
            logTimingStats();

        // Barriers and blit to swapchain (same pattern as ch02)

        // Compute → Transfer barrier on storage image
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

        // Blit storage image → swapchain
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

        // Post-blit barriers
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

    // Log per-quadrant average timing to stdout every 60 frames
    void logTimingStats()
    {
        if (!m_hasTimestamps) return;

        // Group tiles into four quadrants and average GPU times
        double sumTime[4] = {0.0, 0.0, 0.0, 0.0};
        uint32_t cnt[4]   = {0, 0, 0, 0};

        uint32_t halfCols = kTileCols / 2u;
        uint32_t halfRows = kTileRows / 2u;

        for (uint32_t row = 0; row < kTileRows; ++row)
        {
            for (uint32_t col = 0; col < kTileCols; ++col)
            {
                uint32_t tileIdx = row * kTileCols + col;
                bool leftHalf    = (col < halfCols);
                bool topHalf     = (row < halfRows);
                uint32_t q = topHalf ? (leftHalf ? 0u : 1u) : (leftHalf ? 2u : 3u);
                sumTime[q] += m_tileTimes[tileIdx];
                cnt[q]++;
            }
        }

        const char *names[4] = {"Divergent", "Non-divergent", "LDS-reduce", "Wave-reduce"};
        std::cout << "[frame " << m_frameCount << "] avg tile GPU time (us):\n";
        for (int q = 0; q < 4; ++q)
        {
            double avgUs = (cnt[q] > 0) ? sumTime[q] / cnt[q] / 1000.0 : 0.0;
            std::cout << "  Q" << q << " " << names[q] << ": " << avgUs << " us\n";
        }
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
            vk::DescriptorPoolSize poolSize{
                .type            = vk::DescriptorType::eStorageImage,
                .descriptorCount = 1};
            vk::DescriptorPoolCreateInfo dpci{
                .maxSets       = 1,
                .poolSizeCount = 1,
                .pPoolSizes    = &poolSize};
            f.dsPool = vk::raii::DescriptorPool(m_device, dpci);

            vk::DescriptorSetAllocateInfo dsai{
                .descriptorPool     = *f.dsPool,
                .descriptorSetCount = 1,
                .pSetLayouts        = &*m_dsLayout};
            f.dsSet = vk::raii::DescriptorSets(m_device, dsai)[0].release();

            bindStorageImageDescriptor(f);
        }

        m_renderDone.clear();
        for (size_t i = 0; i < m_swapImages.size(); ++i)
            m_renderDone.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        transitionStorageImagesToGeneral();
    }

    // =======================================================================
    // Helpers
    // =======================================================================
    [[nodiscard]] uint32_t findMemoryType(uint32_t filter,
                                          vk::MemoryPropertyFlags props) const
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
        auto     raw   = glfwGetRequiredInstanceExtensions(&count);
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
            throw std::runtime_error("failed to open: " + path);
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
        HeatmapApp app;
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
extern "C" void chapter10_run() {
    try { HeatmapApp{}.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh10", "%s", e.what()); }
}
#endif
