// Chapter 9 – Cooperative Matrices & Specialized Math: FP16 Noise + Denoising
//
// Demonstrates:
//   • FP16 arithmetic via shaderFloat16 + storageBuffer16BitAccess (Vulkan 1.1/1.2 features)
//   • Two-pass compute pipeline:
//       Pass 1 (noiseMain)    — generate animated value-noise using FP16 math
//       Pass 2 (denoiseMain)  — tile-based FP16 Gaussian blur in groupshared memory
//                               (this is how cooperative matrices tile computation;
//                                VK_KHR_cooperative_matrix accelerates this pattern on
//                                supported hardware — we probe at startup and print a message)
//   • Graceful detection of VK_KHR_cooperative_matrix at startup
//   • Windowed app using the same blit-to-swapchain pattern as chapter 02
//
// Controls:
//   Scroll        — adjust blur radius (0..4)
//   +/-           — adjust noise frequency
//   R             — reset parameters
//   ESC           — quit
//
// Build:  see CMakeLists.txt – add_compute_chapter(09_specialized_math WINDOWED ...)
// Shader: shaders/slang.spv  (compiled from 09_specialized_math.slang)

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
constexpr uint32_t kWidth      = 1280;
constexpr uint32_t kHeight     = 720;
constexpr int      kMaxFrames  = 2;
// One more acquire semaphore than frames so we never reuse one the presentation
// engine still holds.
constexpr int      kAcquireSemas = kMaxFrames + 1;

const std::vector<char const *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// Push-constant layout — byte-identical to NoisePush in the shader
// ---------------------------------------------------------------------------
struct NoisePush
{
    float    time;          // animation clock (seconds)
    float    frequency;     // noise frequency multiplier
    uint32_t blurRadius;    // Gaussian blur half-width (0..4)
    uint32_t width;
    uint32_t height;
};
static_assert(sizeof(NoisePush) == 20, "push constant size mismatch");

// ---------------------------------------------------------------------------
// FP16NoiseApp
// ---------------------------------------------------------------------------
class FP16NoiseApp
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
    // Window + interactive state
    // -----------------------------------------------------------------------
    GLFWwindow *m_window  = nullptr;
    bool        m_resized = false;

    float    m_frequency  = 4.0f;
    uint32_t m_blurRadius = 2u;

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
    // Intermediate noise buffer (FP16 RG per pixel, packed as uint32)
    //   layout: width * height * sizeof(uint32_t) bytes
    //   binding 1 in the shaders — holds half2 packed values
    // -----------------------------------------------------------------------
    vk::raii::Buffer       m_noiseBuf = nullptr;
    vk::raii::DeviceMemory m_noiseMem = nullptr;

    // -----------------------------------------------------------------------
    // Pipelines / layouts
    // -----------------------------------------------------------------------
    // Pass 1: noise generation
    vk::raii::DescriptorSetLayout m_noisedsLayout   = nullptr;
    vk::raii::PipelineLayout      m_noisePipeLayout = nullptr;
    vk::raii::Pipeline            m_noisePipeline   = nullptr;

    // Pass 2: denoising / Gaussian blur
    vk::raii::DescriptorSetLayout m_blurDsLayout    = nullptr;
    vk::raii::PipelineLayout      m_blurPipeLayout  = nullptr;
    vk::raii::Pipeline            m_blurPipeline    = nullptr;

    vk::raii::CommandPool m_cmdPool = nullptr;

    // -----------------------------------------------------------------------
    // Per-frame resources
    // -----------------------------------------------------------------------
    struct PerFrame
    {
        // Storage image written by the blur shader, blitted to swapchain
        vk::raii::Image        storImg  = nullptr;
        vk::raii::DeviceMemory storMem  = nullptr;
        vk::raii::ImageView    storView = nullptr;

        // Descriptor pools — one for each pipeline pass
        vk::raii::DescriptorPool noiseDsPool = nullptr;
        vk::DescriptorSet        noiseDsSet  = nullptr;   // raw, owned by pool

        vk::raii::DescriptorPool blurDsPool  = nullptr;
        vk::DescriptorSet        blurDsSet   = nullptr;   // raw, owned by pool

        vk::raii::CommandBuffer cmdBuf = nullptr;
        vk::raii::Fence         fence  = nullptr;
    };
    std::array<PerFrame, kMaxFrames> m_frames;

    // Acquire semaphores: rotating pool of kAcquireSemas = kMaxFrames+1
    std::vector<vk::raii::Semaphore> m_imageAvail;
    int m_acquireIdx = 0;

    // renderDone indexed by swapchain IMAGE index (not frame slot)
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
            "FP16 Noise + Denoising  |  scroll=blur  +/-=frequency  R=reset  ESC=quit",
            nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, cbResize);
        glfwSetScrollCallback(m_window,           cbScroll);
        glfwSetKeyCallback(m_window,              cbKey);
    }

    // -----------------------------------------------------------------------
    // GLFW callbacks
    // -----------------------------------------------------------------------
    static void cbResize(GLFWwindow *w, int, int)
    {
        static_cast<FP16NoiseApp *>(glfwGetWindowUserPointer(w))->m_resized = true;
    }

    static void cbScroll(GLFWwindow *w, double /*dx*/, double dy)
    {
        auto *app = static_cast<FP16NoiseApp *>(glfwGetWindowUserPointer(w));
        if (dy > 0.0)
            app->m_blurRadius = std::min(app->m_blurRadius + 1u, 4u);
        else if (dy < 0.0 && app->m_blurRadius > 0u)
            --app->m_blurRadius;
        std::cout << "Blur radius: " << app->m_blurRadius << '\n';
    }

    static void cbKey(GLFWwindow *w, int key, int /*scan*/, int action, int /*mods*/)
    {
        if (action != GLFW_PRESS)
            return;
        auto *app = static_cast<FP16NoiseApp *>(glfwGetWindowUserPointer(w));
        switch (key)
        {
            case GLFW_KEY_R:
                app->m_frequency  = 4.0f;
                app->m_blurRadius = 2u;
                break;
            case GLFW_KEY_EQUAL:   // '+' / '='
                app->m_frequency = std::min(app->m_frequency * 1.25f, 64.0f);
                std::cout << "Frequency: " << app->m_frequency << '\n';
                break;
            case GLFW_KEY_MINUS:
                app->m_frequency = std::max(app->m_frequency / 1.25f, 0.5f);
                std::cout << "Frequency: " << app->m_frequency << '\n';
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(w, GLFW_TRUE);
                break;
            default: break;
        }
    }

    // =======================================================================
    // Vulkan init sequence
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
        createNoiseBuffer();
        createDescriptorSetLayouts();
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

            auto now     = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();

            drawFrame(elapsed);
        }
        m_device.waitIdle();
    }

    void cleanup()
    {
        // Destroy all RAII handles in dependency order BEFORE glfwTerminate(),
        // which dlclose()es libvulkan.so on Linux.
        m_renderDone.clear();
        m_imageAvail.clear();
        for (auto &f : m_frames)
        {
            f.fence       = nullptr;
            f.cmdBuf      = nullptr;
            f.blurDsPool  = nullptr;
            f.noiseDsPool = nullptr;
            f.storView    = nullptr;
            f.storMem     = nullptr;
            f.storImg     = nullptr;
        }
        m_cmdPool         = nullptr;
        m_blurPipeline    = nullptr;
        m_blurPipeLayout  = nullptr;
        m_blurDsLayout    = nullptr;
        m_noisePipeline   = nullptr;
        m_noisePipeLayout = nullptr;
        m_noisedsLayout   = nullptr;
        m_noiseBuf        = nullptr;
        m_noiseMem        = nullptr;
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
            .pApplicationName   = "FP16 Noise Demo",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = vk::ApiVersion13};

        std::vector<char const *> layers;
        if (kEnableValidation)
            layers.assign(kValidationLayers.begin(), kValidationLayers.end());

        auto exts = getRequiredInstanceExtensions();

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
        if (!kEnableValidation)
            return;
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
            if (qf == ~0u)
                continue;

            auto devExts = pd.enumerateDeviceExtensionProperties();
            bool hasSwapchain = std::ranges::any_of(devExts, [](auto const &e) {
                return strcmp(e.extensionName, vk::KHRSwapchainExtensionName) == 0;
            });
            if (!hasSwapchain)
                continue;

            int score = typeScore(pd.getProperties().deviceType);
            if (score > bestScore) { bestScore = score; m_physDev = pd; m_queueFamily = qf; }
        }
        if (!*m_physDev)
            throw std::runtime_error("No suitable GPU found!");

        // Print device info and probe cooperative matrix support
        vk::PhysicalDeviceProperties2 props2{};
        m_physDev.getProperties2(&props2);
        std::cout << "=== Chapter 9: FP16 Noise + Tile-Based Denoising ===\n";
        std::cout << "  Device : " << props2.properties.deviceName.data() << '\n';

        // Check for VK_KHR_cooperative_matrix support
        auto exts = m_physDev.enumerateDeviceExtensionProperties();
        bool hasCoopMat = std::ranges::any_of(exts, [](auto const &e) {
            return strcmp(e.extensionName, vk::KHRCooperativeMatrixExtensionName) == 0;
        });

        if (hasCoopMat)
        {
            std::cout << "  Cooperative matrix : SUPPORTED (VK_KHR_cooperative_matrix)\n";
            std::cout << "  --> On this hardware the denoise pass could use cooperative\n";
            std::cout << "      matrix MMA instructions for even faster tiled computation.\n";
        }
        else
        {
            std::cout << "  Cooperative matrix : NOT supported — running FP16 scalar path\n";
            std::cout << "  --> The tile-based groupshared blur below emulates the same\n";
            std::cout << "      data-sharing pattern that cooperative matrices use.\n";
        }
        std::cout << "====================================================\n";
    }

    // =======================================================================
    // Logical device
    // =======================================================================
    void createLogicalDevice()
    {
        // Feature promotion per Vulkan spec:
        //   storageBuffer16BitAccess → VkPhysicalDevice16BitStorageFeatures
        //                              promoted into Vulkan 1.1 (Vulkan11Features)
        //   shaderFloat16            → VkPhysicalDeviceShaderFloat16Int8Features
        //                              promoted into Vulkan 1.2 (Vulkan12Features)
        // Using separate extension structs alongside the promotion structs would
        // trigger VUID-02830 (duplicate pNext chain entries), so we set both
        // features inside the versioned structs only.
        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features>
            featureChain = {
                // Base features: shaderInt16 is required because the Slang-compiled
                // SPIR-V declares the Int16 capability (emitted for half<->int
                // conversions and half2 groupshared patterns).
                {.features = {.shaderInt16 = true}},
                // Vulkan 1.1: 16-bit storage in SSBOs — promoted from
                //   VK_KHR_16bit_storage / VkPhysicalDevice16BitStorageFeatures
                {.storageBuffer16BitAccess = true},
                // Vulkan 1.2: FP16 shader arithmetic — promoted from
                //   VK_KHR_shader_float16_int8 / VkPhysicalDeviceShaderFloat16Int8Features
                {.shaderFloat16 = true, .scalarBlockLayout = true, .timelineSemaphore = true},
                // Vulkan 1.3: synchronization2
                {.synchronization2 = true}};

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
        auto caps    = m_physDev.getSurfaceCapabilitiesKHR(*m_surface);
        m_swapExtent = chooseExtent(caps);
        auto fmts    = m_physDev.getSurfaceFormatsKHR(*m_surface);
        m_swapFormat = chooseFormat(fmts, m_physDev);
        auto modes   = m_physDev.getSurfacePresentModesKHR(*m_surface);
        auto mode    = chooseMode(modes);

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
            .presentMode      = mode,
            .clipped          = true,
            .oldSwapchain     = oldSwapchain};
        m_swapchain  = vk::raii::SwapchainKHR(m_device, sci);
        m_swapImages = m_swapchain.getImages();
    }

    // =======================================================================
    // Intermediate noise buffer
    //   width * height * 4 bytes — stores packed half2 (RG as uint32 per pixel)
    // =======================================================================
    void createNoiseBuffer()
    {
        destroyNoiseBuffer();   // safe to call when null

        vk::DeviceSize size = static_cast<vk::DeviceSize>(
            m_swapExtent.width) * m_swapExtent.height * sizeof(uint32_t);

        vk::BufferCreateInfo bci{
            .size        = size,
            .usage       = vk::BufferUsageFlagBits::eStorageBuffer,
            .sharingMode = vk::SharingMode::eExclusive};
        m_noiseBuf = vk::raii::Buffer(m_device, bci);

        auto memReqs = m_noiseBuf.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = memReqs.size,
            .memoryTypeIndex = findMemoryType(
                memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)};
        m_noiseMem = vk::raii::DeviceMemory(m_device, mai);
        m_noiseBuf.bindMemory(*m_noiseMem, 0);
    }

    void destroyNoiseBuffer()
    {
        m_noiseBuf = nullptr;
        m_noiseMem = nullptr;
    }

    // =======================================================================
    // Descriptor set layouts
    //   Pass 1 (noise): binding 0 = storage buffer (noise output, packed half2)
    //   Pass 2 (blur):  binding 0 = storage image (rgba8 output)
    //                   binding 1 = storage buffer (noise input, packed half2)
    // =======================================================================
    void createDescriptorSetLayouts()
    {
        // Pass 1 — noise generation: one SSBO
        {
            vk::DescriptorSetLayoutBinding b0{
                .binding         = 0,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags      = vk::ShaderStageFlagBits::eCompute};
            vk::DescriptorSetLayoutCreateInfo ci{.bindingCount = 1, .pBindings = &b0};
            m_noisedsLayout = vk::raii::DescriptorSetLayout(m_device, ci);
        }

        // Pass 2 — blur/denoise: storage image + SSBO
        {
            std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
                {.binding         = 0,
                 .descriptorType  = vk::DescriptorType::eStorageImage,
                 .descriptorCount = 1,
                 .stageFlags      = vk::ShaderStageFlagBits::eCompute},
                {.binding         = 1,
                 .descriptorType  = vk::DescriptorType::eStorageBuffer,
                 .descriptorCount = 1,
                 .stageFlags      = vk::ShaderStageFlagBits::eCompute}
            }};
            vk::DescriptorSetLayoutCreateInfo ci{
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pBindings    = bindings.data()};
            m_blurDsLayout = vk::raii::DescriptorSetLayout(m_device, ci);
        }
    }

    // =======================================================================
    // Pipelines
    // =======================================================================
    void createPipelines()
    {
        auto code = readFile("shaders/slang.spv");
        vk::ShaderModuleCreateInfo smci{
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<uint32_t const *>(code.data())};
        vk::raii::ShaderModule shader(m_device, smci);

        vk::PushConstantRange pcRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(NoisePush)};

        // Pass 1 pipeline
        {
            vk::PipelineLayoutCreateInfo plci{
                .setLayoutCount         = 1,
                .pSetLayouts            = &*m_noisedsLayout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges    = &pcRange};
            m_noisePipeLayout = vk::raii::PipelineLayout(m_device, plci);

            vk::PipelineShaderStageCreateInfo stage{
                .stage  = vk::ShaderStageFlagBits::eCompute,
                .module = *shader,
                .pName  = "noiseMain"};
            vk::ComputePipelineCreateInfo pci{.stage = stage, .layout = *m_noisePipeLayout};
            m_noisePipeline = vk::raii::Pipeline(m_device, nullptr, pci);
        }

        // Pass 2 pipeline
        {
            vk::PipelineLayoutCreateInfo plci{
                .setLayoutCount         = 1,
                .pSetLayouts            = &*m_blurDsLayout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges    = &pcRange};
            m_blurPipeLayout = vk::raii::PipelineLayout(m_device, plci);

            vk::PipelineShaderStageCreateInfo stage{
                .stage  = vk::ShaderStageFlagBits::eCompute,
                .module = *shader,
                .pName  = "denoiseMain"};
            vk::ComputePipelineCreateInfo pci{.stage = stage, .layout = *m_blurPipeLayout};
            m_blurPipeline = vk::raii::Pipeline(m_device, nullptr, pci);
        }
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
            createFrameDescriptors(f);

            f.cmdBuf = std::move(cmdBufs[i]);
            f.fence  = vk::raii::Fence(m_device, vk::FenceCreateInfo{
                .flags = vk::FenceCreateFlagBits::eSignaled});
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

    void createFrameDescriptors(PerFrame &f)
    {
        // --- Pass 1: noise (SSBO only) ---
        {
            vk::DescriptorPoolSize poolSize{
                .type            = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1};
            vk::DescriptorPoolCreateInfo dpci{
                .maxSets       = 1,
                .poolSizeCount = 1,
                .pPoolSizes    = &poolSize};
            f.noiseDsPool = vk::raii::DescriptorPool(m_device, dpci);

            vk::DescriptorSetAllocateInfo dsai{
                .descriptorPool     = *f.noiseDsPool,
                .descriptorSetCount = 1,
                .pSetLayouts        = &*m_noisedsLayout};
            f.noiseDsSet = vk::raii::DescriptorSets(m_device, dsai)[0].release();

            vk::DeviceSize noiseSize = static_cast<vk::DeviceSize>(
                m_swapExtent.width) * m_swapExtent.height * sizeof(uint32_t);
            vk::DescriptorBufferInfo bufInfo{
                .buffer = *m_noiseBuf,
                .offset = 0,
                .range  = noiseSize};
            vk::WriteDescriptorSet write{
                .dstSet          = f.noiseDsSet,
                .dstBinding      = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo     = &bufInfo};
            m_device.updateDescriptorSets(write, {});
        }

        // --- Pass 2: blur (storage image + SSBO) ---
        {
            std::array<vk::DescriptorPoolSize, 2> poolSizes{{
                {.type = vk::DescriptorType::eStorageImage,  .descriptorCount = 1},
                {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1}
            }};
            vk::DescriptorPoolCreateInfo dpci{
                .maxSets       = 1,
                .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                .pPoolSizes    = poolSizes.data()};
            f.blurDsPool = vk::raii::DescriptorPool(m_device, dpci);

            vk::DescriptorSetAllocateInfo dsai{
                .descriptorPool     = *f.blurDsPool,
                .descriptorSetCount = 1,
                .pSetLayouts        = &*m_blurDsLayout};
            f.blurDsSet = vk::raii::DescriptorSets(m_device, dsai)[0].release();

            vk::DescriptorImageInfo imgInfo{
                .imageView   = *f.storView,
                .imageLayout = vk::ImageLayout::eGeneral};
            vk::DeviceSize noiseSize = static_cast<vk::DeviceSize>(
                m_swapExtent.width) * m_swapExtent.height * sizeof(uint32_t);
            vk::DescriptorBufferInfo bufInfo{
                .buffer = *m_noiseBuf,
                .offset = 0,
                .range  = noiseSize};

            std::array<vk::WriteDescriptorSet, 2> writes{{
                {.dstSet          = f.blurDsSet,
                 .dstBinding      = 0,
                 .dstArrayElement = 0,
                 .descriptorCount = 1,
                 .descriptorType  = vk::DescriptorType::eStorageImage,
                 .pImageInfo      = &imgInfo},
                {.dstSet          = f.blurDsSet,
                 .dstBinding      = 1,
                 .dstArrayElement = 0,
                 .descriptorCount = 1,
                 .descriptorType  = vk::DescriptorType::eStorageBuffer,
                 .pBufferInfo     = &bufInfo}
            }};
            m_device.updateDescriptorSets(writes, {});
        }
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
    void drawFrame(float time)
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
        recordCommands(f, imageIndex, time);

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

    void recordCommands(PerFrame &f, uint32_t imageIndex, float time)
    {
        auto &cb = f.cmdBuf;
        cb.reset();
        cb.begin({});

        NoisePush push{
            .time       = time,
            .frequency  = m_frequency,
            .blurRadius = m_blurRadius,
            .width      = m_swapExtent.width,
            .height     = m_swapExtent.height};

        // ----------------------------------------------------------------
        // Pass 1: noise generation → intermediate SSBO
        // ----------------------------------------------------------------
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_noisePipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                              *m_noisePipeLayout, 0, {f.noiseDsSet}, {});
        cb.pushConstants<NoisePush>(*m_noisePipeLayout,
                                    vk::ShaderStageFlagBits::eCompute, 0, push);

        uint32_t gx = (m_swapExtent.width  + 15u) / 16u;
        uint32_t gy = (m_swapExtent.height + 15u) / 16u;
        cb.dispatch(gx, gy, 1);

        // Barrier: noise SSBO write → blur SSBO read
        vk::BufferMemoryBarrier2 noiseBufBarrier{
            .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask       = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask       = vk::AccessFlagBits2::eShaderStorageRead,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer              = *m_noiseBuf,
            .offset              = 0,
            .size                = VK_WHOLE_SIZE};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &noiseBufBarrier});

        // ----------------------------------------------------------------
        // Pass 2: tile-based FP16 Gaussian blur → storage image
        //
        // Educational note: This groupshared tile accumulation is exactly the
        // data-sharing pattern that VK_KHR_cooperative_matrix hardware-accelerates.
        // Each workgroup loads a 16×16 tile into LDS (groupshared memory), applies
        // a separable Gaussian kernel in FP16, and writes the blurred result.
        // On supported hardware, cooperative matrix MMA instructions would replace
        // the scalar FP16 accumulation loops with a single hardware instruction.
        // ----------------------------------------------------------------
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_blurPipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                              *m_blurPipeLayout, 0, {f.blurDsSet}, {});
        cb.pushConstants<NoisePush>(*m_blurPipeLayout,
                                    vk::ShaderStageFlagBits::eCompute, 0, push);
        cb.dispatch(gx, gy, 1);

        // ----------------------------------------------------------------
        // Barriers: storage image → blit source, swapchain → blit dest
        // ----------------------------------------------------------------
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
            f.blurDsPool  = nullptr;
            f.blurDsSet   = nullptr;
            f.noiseDsPool = nullptr;
            f.noiseDsSet  = nullptr;
            f.storView    = nullptr;
            f.storImg     = nullptr;
            f.storMem     = nullptr;
        }

        destroyNoiseBuffer();

        vk::SwapchainKHR oldHandle = *m_swapchain;
        createSwapchain(oldHandle);

        createNoiseBuffer();

        for (auto &f : m_frames)
        {
            createStorageImage(f);
            createFrameDescriptors(f);
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
        FP16NoiseApp app;
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
extern "C" void chapter09_run() {
    try { FP16NoiseApp{}.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh09", "%s", e.what()); }
}
#endif
