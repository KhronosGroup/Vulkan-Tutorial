// Chapter 2 – Compute Architecture: Mandelbrot Explorer
//
// Demonstrates:
//   • Querying and printing physical-device compute limits (subgroup size,
//     max workgroup invocations) – the "compute architecture" content
//   • A pure compute pipeline that renders directly into a storage image
//   • Blit from the storage image to the swapchain (no render pass needed)
//   • Double-buffered per-frame resources to avoid storage-image data races
//   • Interactive pan/zoom via GLFW scroll and drag callbacks
//   • Animated smooth coloring with the IQ cosine palette
//
// Build:  see CMakeLists.txt – add WINDOWED to add_compute_chapter()
// Shader: shaders/slang.spv  (compiled from 02_compute_architecture.slang)

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
#  define GLFW_INCLUDE_VULKAN   // required only for glfwCreateWindowSurface
#  include <GLFW/glfw3.h>
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr uint32_t kWidth      = 1280;
constexpr uint32_t kHeight     = 720;
constexpr int      kMaxFrames  = 2;
// Binary semaphores for image acquisition must not be reused while the
// presentation engine still holds a reference.  Having one more acquire
// semaphore than frames-in-flight guarantees the rotating pool is safe.
constexpr int      kAcquireSemas = kMaxFrames + 1;
// Minimum zoom (world-units/pixel).  With perturbation theory the per-pixel
// delta stays in full double-float precision regardless of depth, so the limit
// is now set by float underflow of the delta rather than the centre's ULP.
// ~1e-30 is comfortably reachable before deltas denormalise.
constexpr long double kMinZoom = 1e-30L;

const std::vector<char const *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// Push-constant layout – must be byte-identical to MandelbrotPush in the shader
//
// The centre is split into (hi, lo) float pairs using Veltkamp splitting so
// the shader can reconstruct the full double-precision centre in double-float
// (two-word float) form, enabling zoom depths around 1e-13.
// ---------------------------------------------------------------------------
// zoom is split into (hi, lo) float pairs using the same Veltkamp technique as
// the centre, so the per-pixel offset dx = pixel_offset * zoom is computed in
// double-float inside the shader.  This extends the practical zoom limit from
// ~1e-13 (single-float zoom) to ~1e-26 before adjacent pixels become identical.
//
// The iteration band [minIter, maxIter] is a moving window: minIter slides up
// with zoom depth so the shader only resolves detail in the current zoom layer.
// Pixels escaping before minIter belong to coarser layers that are no longer
// the focus; they are still drawn dimmed.  Both bounds are derived entirely
// from m_zoom — no manual iteration adjustment is needed.
struct MandelbrotPush
{
    float    zoomHi;     // float(m_zoom)
    float    zoomLo;     // float(m_zoom - double(zoomHi)) — sub-ULP residual
    uint32_t width;
    uint32_t height;
    uint32_t minIter;    // window floor (dimmed below this)
    uint32_t maxIter;    // window ceiling = auto-scaled total iterations
    uint32_t refLen;     // valid length of the reference orbit (perturbation)
    float    colorPhase;
};
static_assert(sizeof(MandelbrotPush) == 32, "push constant size mismatch");

// Reference-orbit capacity.  Must be ≥ the largest maxIter we ever push, which
// recordCommands() clamps to 131072.  Each entry is one float4 (two double-
// floats: real hi/lo, imag hi/lo) = 16 bytes → 2 MiB per buffer.
constexpr uint32_t kMaxRefIter = 131072u;

// ---------------------------------------------------------------------------
// MandelbrotApp
// ---------------------------------------------------------------------------
class MandelbrotApp
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
    // Window + view state
    // -----------------------------------------------------------------------
    GLFWwindow *m_window   = nullptr;
    bool        m_resized  = false;
    bool        m_dragging = false;
    double      m_lastMx   = 0.0, m_lastMy = 0.0;

    // View parameters stored as long double for precise navigation at deep zoom
    // levels; the per-pixel detail itself comes from the perturbation delta, so
    // these only need to locate the reference orbit accurately.
    long double m_cx      = -0.5L;
    long double m_cy      = 0.0L;
    long double m_zoom    = 3.5L / kWidth;  // fits the full set in the default window
    uint32_t    m_maxIter = 256u;
    float       m_phase   = 0.0f;

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
    vk::raii::SwapchainKHR           m_swapchain      = nullptr;
    std::vector<vk::Image>           m_swapImages;
    vk::SurfaceFormatKHR             m_swapFormat{};
    vk::Extent2D                     m_swapExtent{};

    // -----------------------------------------------------------------------
    // Pipelines / layouts
    // -----------------------------------------------------------------------
    vk::raii::DescriptorSetLayout m_dsLayout       = nullptr;
    vk::raii::PipelineLayout      m_pipeLayout     = nullptr;
    vk::raii::Pipeline            m_computePipeline = nullptr;

    // Command pool must be declared before m_frames so that it is destroyed
    // AFTER m_frames (reverse declaration order).  PerFrame::cmdBuf calls
    // vkFreeCommandBuffers on this pool from its destructor; the pool must
    // still be alive at that point.
    vk::raii::CommandPool m_cmdPool = nullptr;

    // -----------------------------------------------------------------------
    // Per-frame resources  (kMaxFrames = 2)
    // Each frame has its own storage image so the two in-flight frames never
    // race on the same image data.
    // -----------------------------------------------------------------------
    struct PerFrame
    {
        // Storage image written by the compute shader
        vk::raii::Image        storImg  = nullptr;
        vk::raii::DeviceMemory storMem  = nullptr;
        vk::raii::ImageView    storView = nullptr;

        // One descriptor pool + set per frame; the pool is reset on swapchain
        // recreate (no eFreeDescriptorSet needed).
        vk::raii::DescriptorPool dsPool = nullptr;
        vk::DescriptorSet        dsSet  = nullptr;   // raw handle, owned by pool

        // Reference orbit (perturbation theory) — host-visible, persistently
        // mapped, refilled by the CPU each frame at the current view centre.
        // Memory declared before buffer so RAII destroys the buffer first.
        vk::raii::DeviceMemory refMem    = nullptr;
        vk::raii::Buffer       refBuf    = nullptr;
        void                  *refMapped = nullptr;

        vk::raii::CommandBuffer cmdBuf = nullptr;
        vk::raii::Fence         fence  = nullptr;
    };
    std::array<PerFrame, kMaxFrames> m_frames;

    // Acquire semaphores: rotating pool of kAcquireSemas = kMaxFrames+1 entries
    // so we never re-signal a semaphore the presentation engine still holds.
    std::vector<vk::raii::Semaphore> m_imageAvail;
    int m_acquireIdx = 0;

    // renderDone semaphores are indexed by swapchain IMAGE INDEX (not frame slot).
    // This guarantees the semaphore has been consumed by the presentation engine
    // before it is re-signalled: image I can only be re-acquired after its
    // previous presentation completes, which consumes renderDone[I].
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
            "Mandelbrot  |  scroll=zoom  drag=pan  R=reset  =/- iterations",
            nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, cbResize);
        glfwSetScrollCallback(m_window,           cbScroll);
        glfwSetMouseButtonCallback(m_window,      cbMouseButton);
        glfwSetCursorPosCallback(m_window,        cbCursorPos);
        glfwSetKeyCallback(m_window,              cbKey);
    }

    // -----------------------------------------------------------------------
    // GLFW callbacks (static → member via glfwGetWindowUserPointer)
    // -----------------------------------------------------------------------
    static void cbResize(GLFWwindow *w, int, int)
    {
        static_cast<MandelbrotApp *>(glfwGetWindowUserPointer(w))->m_resized = true;
    }

    static void cbScroll(GLFWwindow *w, double /*dx*/, double dy)
    {
        auto *app = static_cast<MandelbrotApp *>(glfwGetWindowUserPointer(w));
        int   iw, ih;
        glfwGetWindowSize(w, &iw, &ih);
        double W = iw, H = ih;

        double mx, my;
        glfwGetCursorPos(w, &mx, &my);

        // Zoom toward the cursor: keep the world point under the cursor fixed.
        // World coordinates stay in long double so navigation does not lose
        // precision as the zoom deepens.
        long double factor = (dy > 0.0) ? 0.85L : (1.0L / 0.85L);
        long double wx = app->m_cx + (static_cast<long double>(mx) - W * 0.5L) * app->m_zoom;
        long double wy = app->m_cy + (static_cast<long double>(my) - H * 0.5L) * app->m_zoom;
        app->m_zoom *= factor;
        app->m_zoom  = std::max(app->m_zoom, kMinZoom);
        app->m_cx    = wx - (static_cast<long double>(mx) - W * 0.5L) * app->m_zoom;
        app->m_cy    = wy - (static_cast<long double>(my) - H * 0.5L) * app->m_zoom;
    }

    static void cbMouseButton(GLFWwindow *w, int button, int action, int /*mods*/)
    {
        auto *app = static_cast<MandelbrotApp *>(glfwGetWindowUserPointer(w));
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            app->m_dragging = (action == GLFW_PRESS);
            glfwGetCursorPos(w, &app->m_lastMx, &app->m_lastMy);
        }
    }

    static void cbCursorPos(GLFWwindow *w, double mx, double my)
    {
        auto *app = static_cast<MandelbrotApp *>(glfwGetWindowUserPointer(w));
        if (app->m_dragging)
        {
            double dx = mx - app->m_lastMx;
            double dy = my - app->m_lastMy;
            // Pan: move the centre in the opposite direction to the drag
            app->m_cx -= dx * app->m_zoom;
            app->m_cy -= dy * app->m_zoom;
        }
        app->m_lastMx = mx;
        app->m_lastMy = my;
    }

    static void cbKey(GLFWwindow *w, int key, int /*scancode*/, int action, int /*mods*/)
    {
        if (action != GLFW_PRESS)
            return;
        auto *app = static_cast<MandelbrotApp *>(glfwGetWindowUserPointer(w));
        switch (key)
        {
            case GLFW_KEY_R:
                app->m_cx      = -0.5L;
                app->m_cy      = 0.0L;
                app->m_zoom    = 3.5L / kWidth;
                app->m_maxIter = 256u;
                break;
            case GLFW_KEY_EQUAL:   // '+' / '=' on most keyboards
                app->m_maxIter = std::min(app->m_maxIter * 2u, 4096u);
                break;
            case GLFW_KEY_MINUS:
                app->m_maxIter = std::max(app->m_maxIter / 2u, 32u);
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
        createDescriptorSetLayout();
        createPipeline();
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

            // Animated color phase: one full cycle every ~20 seconds
            auto now     = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            m_phase = std::fmod(elapsed * 0.05f, 1.0f);

            drawFrame();
        }
        m_device.waitIdle();
    }

    void cleanup()
    {
        // Explicitly destroy every Vulkan RAII handle in dependency order
        // BEFORE calling glfwTerminate().  glfwTerminate() calls
        // _glfwTerminateVulkan() which dlclose()'s libvulkan.so; any RAII
        // destructor that fires after that point will call through an
        // unmapped function pointer and SIGSEGV.
        //
        // Assigning nullptr to a vk::raii handle immediately runs its
        // destructor and leaves the wrapper in a null state, so the
        // automatic member destructors that follow become no-ops.

        m_renderDone.clear();
        m_imageAvail.clear();
        for (auto &f : m_frames)
        {
            f.fence    = nullptr;
            f.cmdBuf   = nullptr;
            f.dsPool   = nullptr;   // raw f.dsSet freed by pool
            f.refBuf   = nullptr;   // unmaps + frees refMem on destruction
            f.refMem   = nullptr;
            f.storView = nullptr;
            f.storMem  = nullptr;
            f.storImg  = nullptr;
        }
        m_cmdPool         = nullptr;
        m_computePipeline = nullptr;
        m_pipeLayout      = nullptr;
        m_dsLayout        = nullptr;
        m_swapchain       = nullptr;
        m_queue           = nullptr;
        m_device          = nullptr;   // all device-owned objects already freed above
        m_surface         = nullptr;
        m_debugMessenger  = nullptr;
        m_instance        = nullptr;
        // m_ctx holds no Vulkan objects; let it destruct normally.

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
            .pApplicationName   = "Mandelbrot Explorer",
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

        auto exts      = getRequiredInstanceExtensions();
        auto extProps  = m_ctx.enumerateInstanceExtensionProperties();
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
            // Need: compute queue, swapchain extension, present support
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

        // Print compute architecture information – this is the educational content
        // for this chapter.
        vk::PhysicalDeviceSubgroupProperties subgroupProps{};
        vk::PhysicalDeviceProperties2 props2{.pNext = &subgroupProps};
        m_physDev.getProperties2(&props2);

        std::cout << "=== Compute Architecture ===\n";
        std::cout << "  Device                    : " << props2.properties.deviceName.data() << '\n';
        std::cout << "  Subgroup size             : " << subgroupProps.subgroupSize << '\n';
        std::cout << "  Max workgroup invocations : " << props2.properties.limits.maxComputeWorkGroupInvocations << '\n';
        std::cout << "============================\n";
    }

    // =======================================================================
    // Logical device
    // =======================================================================
    void createLogicalDevice()
    {
        // Vulkan 1.3: synchronization2 + dynamicRendering (required for blit path)
        // Vulkan 1.2: timelineSemaphore + scalarBlockLayout
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
    // Command pool  (reset-per-buffer so we can re-record each frame)
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
        auto caps   = m_physDev.getSurfaceCapabilitiesKHR(*m_surface);
        m_swapExtent = chooseExtent(caps);

        auto fmts   = m_physDev.getSurfaceFormatsKHR(*m_surface);
        m_swapFormat = chooseFormat(fmts, m_physDev);

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
            // eTransferDst: the swapchain images are blit destinations, not render targets
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
    //   binding 0 = output storage image
    //   binding 1 = reference-orbit storage buffer (perturbation)
    // =======================================================================
    void createDescriptorSetLayout()
    {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {.binding = 0, .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings    = bindings.data()};
        m_dsLayout = vk::raii::DescriptorSetLayout(m_device, ci);
    }

    // =======================================================================
    // Compute pipeline
    // =======================================================================
    void createPipeline()
    {
        auto code   = readFile("shaders/slang.spv");
        vk::ShaderModuleCreateInfo smci{
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<uint32_t const *>(code.data())};
        vk::raii::ShaderModule shaderModule(m_device, smci);

        // Push constant range: 44 bytes covering all members of MandelbrotPush
        vk::PushConstantRange pcRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(MandelbrotPush)};
        vk::PipelineLayoutCreateInfo plci{
            .setLayoutCount         = 1,
            .pSetLayouts            = &*m_dsLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcRange};
        m_pipeLayout = vk::raii::PipelineLayout(m_device, plci);

        vk::PipelineShaderStageCreateInfo stage{
            .stage  = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName  = "compMain"};
        vk::ComputePipelineCreateInfo pci{.stage = stage, .layout = *m_pipeLayout};
        m_computePipeline = vk::raii::Pipeline(m_device, nullptr, pci);
    }

    // =======================================================================
    // Per-frame resources
    // Creates storage image + view + descriptor pool/set + sync objects
    // for each of the kMaxFrames slots.
    // =======================================================================
    void createPerFrameResources()
    {
        // Allocate command buffers for all frames from the shared pool
        vk::CommandBufferAllocateInfo cbai{
            .commandPool        = *m_cmdPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = kMaxFrames};
        auto cmdBufs = vk::raii::CommandBuffers(m_device, cbai);

        for (int i = 0; i < kMaxFrames; ++i)
        {
            auto &f = m_frames[i];

            // -- Storage image --
            createStorageImage(f);

            // -- Reference-orbit buffer (created once; survives swapchain recreate) --
            createReferenceBuffer(f);

            // -- Descriptor pool (no eFreeDescriptorSet: reset the whole pool on recreate) --
            std::array<vk::DescriptorPoolSize, 2> poolSizes{{
                {.type = vk::DescriptorType::eStorageImage,  .descriptorCount = 1},
                {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1},
            }};
            vk::DescriptorPoolCreateInfo dpci{
                .maxSets       = 1,
                .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                .pPoolSizes    = poolSizes.data()};
            f.dsPool = vk::raii::DescriptorPool(m_device, dpci);

            // -- Descriptor set --
            vk::DescriptorSetAllocateInfo dsai{
                .descriptorPool     = *f.dsPool,
                .descriptorSetCount = 1,
                .pSetLayouts        = &*m_dsLayout};
            // Allocate via RAII then release the raw handle — the pool owns the lifetime.
            // Using release() avoids triggering vkFreeDescriptorSets on the temporary.
            f.dsSet = vk::raii::DescriptorSets(m_device, dsai)[0].release();

            writeFrameDescriptors(f);

            // -- Command buffer --
            f.cmdBuf = std::move(cmdBufs[i]);

            // Fence starts signalled so the first waitForFences on frame 0 returns immediately
            f.fence = vk::raii::Fence(m_device, vk::FenceCreateInfo{
                .flags = vk::FenceCreateFlagBits::eSignaled});
        }

        // Rotating pool of acquire semaphores (kAcquireSemas = kMaxFrames + 1)
        m_imageAvail.clear();
        for (int i = 0; i < kAcquireSemas; ++i)
            m_imageAvail.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        // renderDone semaphores – one per swapchain image (indexed by acquired image index)
        m_renderDone.clear();
        for (size_t i = 0; i < m_swapImages.size(); ++i)
            m_renderDone.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        // Transition all storage images to eGeneral before entering the render loop
        transitionStorageImagesToGeneral();
    }

    // Allocate a host-visible, persistently-mapped reference-orbit buffer for
    // one frame slot.  These do NOT depend on the swapchain size, so they are
    // created once and survive swapchain recreation.
    void createReferenceBuffer(PerFrame &f)
    {
        constexpr vk::DeviceSize sz = vk::DeviceSize(kMaxRefIter) * 4u * sizeof(float);
        vk::BufferCreateInfo bci{
            .size        = sz,
            .usage       = vk::BufferUsageFlagBits::eStorageBuffer,
            .sharingMode = vk::SharingMode::eExclusive};
        f.refBuf = vk::raii::Buffer(m_device, bci);

        auto memReqs = f.refBuf.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = memReqs.size,
            .memoryTypeIndex = findMemoryType(
                memReqs.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent)};
        f.refMem = vk::raii::DeviceMemory(m_device, mai);
        f.refBuf.bindMemory(*f.refMem, 0);
        f.refMapped = f.refMem.mapMemory(0, sz);
    }

    // Compute the reference orbit X_n at the current view centre in long-double
    // precision, packing each X_n as two double-floats (real hi/lo, imag hi/lo)
    // into f.refMapped.  Returns the number of valid entries (refLen): either
    // maxIter, or the iteration at which the reference escaped |X| > 2.
    [[nodiscard]] uint32_t fillReferenceOrbit(PerFrame &f, uint32_t maxIter) const
    {
        maxIter = std::min(maxIter, kMaxRefIter);
        const long double cr = m_cx;
        const long double ci = m_cy;
        long double zr = 0.0L, zi = 0.0L;
        auto *dst = static_cast<float *>(f.refMapped);

        uint32_t n = 0;
        while (n < maxIter)
        {
            // Store X_n as two double-floats.
            float zrHi = static_cast<float>(zr);
            float ziHi = static_cast<float>(zi);
            dst[n * 4 + 0] = zrHi;
            dst[n * 4 + 1] = static_cast<float>(zr - static_cast<long double>(zrHi));
            dst[n * 4 + 2] = ziHi;
            dst[n * 4 + 3] = static_cast<float>(zi - static_cast<long double>(ziHi));
            ++n;

            if (zr * zr + zi * zi > 4.0L)
                break;   // X_{n-1} escaped — reference is valid up to here

            long double nzr = zr * zr - zi * zi + cr;
            long double nzi = 2.0L * zr * zi + ci;
            zr = nzr;
            zi = nzi;
        }
        return n;
    }

    // Write both descriptor bindings for one frame slot:
    //   binding 0 = output storage image (GENERAL layout)
    //   binding 1 = reference-orbit storage buffer
    void writeFrameDescriptors(PerFrame &f)
    {
        vk::DescriptorImageInfo imgInfo{
            .imageView   = *f.storView,
            .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorBufferInfo refInfo{
            .buffer = *f.refBuf,
            .offset = 0,
            .range  = VK_WHOLE_SIZE};

        std::array<vk::WriteDescriptorSet, 2> writes{{
            {.dstSet = f.dsSet, .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,  .pImageInfo  = &imgInfo},
            {.dstSet = f.dsSet, .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &refInfo},
        }};
        m_device.updateDescriptorSets(writes, {});
    }

    // Allocate + bind a device-local storage image for one frame slot
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
            // eStorage: written by compute; eTransferSrc: blitted to swapchain
            .usage       = vk::ImageUsageFlagBits::eStorage |
                           vk::ImageUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive,
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

    // One-shot command to transition all storage images from UNDEFINED to GENERAL
    // before the render loop begins.  They stay in GENERAL permanently; the
    // compute→transfer barrier uses srcLayout=eGeneral / dstLayout=eGeneral.
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
    void drawFrame()
    {
        auto &f = m_frames[m_frameIdx];

        // Wait for this frame slot's previous work to complete
        auto waitRes = m_device.waitForFences(*f.fence, vk::True, UINT64_MAX);
        if (waitRes != vk::Result::eSuccess)
            throw std::runtime_error("waitForFences failed");

        // Pick the next acquire semaphore from the rotating pool.
        // This ensures we never reuse a semaphore the presentation engine still holds.
        auto& acqSem = m_imageAvail[m_acquireIdx];
        m_acquireIdx = (m_acquireIdx + 1) % kAcquireSemas;

        // Acquire swapchain image
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

        // Record command buffer
        recordCommands(f, imageIndex);

        // Use the renderDone semaphore indexed by the acquired image — not by frame slot.
        // This prevents re-signalling it before the presentation engine has consumed it.
        auto& rdSem = m_renderDone[imageIndex];

        // Submit
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

        // Present
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

    void recordCommands(PerFrame &f, uint32_t imageIndex)
    {
        auto &cb = f.cmdBuf;
        cb.reset();
        cb.begin({});

        // 1. Bind compute pipeline + this frame's descriptor set (storage image)
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_computePipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_pipeLayout,
                              0, {f.dsSet}, {});

        // 2. Push current view state.
        // Split the zoom into (hi, lo) float pairs so the shader computes the
        // per-pixel delta δc = pixelOffset · zoom in full double-float precision.
        auto splitDouble = [](long double v) -> std::pair<float, float> {
            float hi = static_cast<float>(v);
            float lo = static_cast<float>(v - static_cast<long double>(hi));
            return {hi, lo};
        };
        auto [zoomHi, zoomLo] = splitDouble(m_zoom);

        // Derive the iteration window entirely from zoom depth — no user input needed.
        // logDepth = 0 at the initial view, grows by 1 for each 2× zoom-in.
        // autoMax: total iterations required to resolve detail at this depth.
        // windowSize: colour-visible band (m_maxIter lets the user widen/narrow it).
        // minIter: window floor — pixels escaping below this are dimmed as coarser layers.
        constexpr long double kInitialZoom = 3.5L / kWidth;
        double   logDepth  = std::max(0.0, std::log2(static_cast<double>(kInitialZoom / m_zoom)));
        uint32_t autoMax   = static_cast<uint32_t>(
            std::clamp(256.0 * (1.0 + 0.5 * logDepth), 256.0, double(kMaxRefIter)));
        uint32_t windowSize = std::min(autoMax, m_maxIter);
        uint32_t minIter    = (autoMax > windowSize) ? autoMax - windowSize : 0u;

        // Compute the reference orbit for this frame at the view centre.  The
        // shader iterates only the per-pixel delta against this orbit, which is
        // what lets the zoom go far past the centre's own ULP without the image
        // degrading into flat rectangles.
        uint32_t refLen = fillReferenceOrbit(f, autoMax);

        MandelbrotPush push{
            .zoomHi     = zoomHi,
            .zoomLo     = zoomLo,
            .width      = m_swapExtent.width,
            .height     = m_swapExtent.height,
            .minIter    = minIter,
            .maxIter    = autoMax,
            .refLen     = refLen,
            .colorPhase = m_phase};
        cb.pushConstants<MandelbrotPush>(*m_pipeLayout,
                                         vk::ShaderStageFlagBits::eCompute, 0, push);

        // 3. Dispatch: one thread per pixel, workgroup tile 16×16
        uint32_t gx = (m_swapExtent.width  + 15u) / 16u;
        uint32_t gy = (m_swapExtent.height + 15u) / 16u;
        cb.dispatch(gx, gy, 1);

        // 4. Barrier: wait for compute writes to finish before the blit reads them.
        //    The storage image stays in eGeneral throughout – we only flip the
        //    access mask / pipeline stage.
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

        // 5. Barrier: swapchain image UNDEFINED → TRANSFER_DST_OPTIMAL
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

        // 6. Blit storage image → swapchain (NEAREST: pixel-exact copy, no filtering artefacts)
        vk::ImageSubresourceLayers subres{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        vk::Offset3D               zero{0, 0, 0};
        vk::Offset3D               ext{
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

        // 7. Post-blit barriers:
        //    a. Swapchain image: TRANSFER_DST → PRESENT_SRC
        //    b. Storage image:   release the transfer-read so next frame's
        //       compute shader can write again (GENERAL → GENERAL, flip access)
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
    // Swapchain recreation  (window resize or suboptimal)
    // =======================================================================
    void recreateSwapchain()
    {
        // Block while the window is minimized
        int w = 0, h = 0;
        glfwGetFramebufferSize(m_window, &w, &h);
        while (w == 0 || h == 0)
        {
            glfwGetFramebufferSize(m_window, &w, &h);
            glfwWaitEvents();
        }

        m_device.waitIdle();

        // Destroy per-frame storage images/views/pools (set to nullptr)
        for (auto &f : m_frames)
        {
            f.storView = nullptr;
            f.storImg  = nullptr;
            f.storMem  = nullptr;
            f.dsPool   = nullptr;
            f.dsSet    = nullptr;
        }

        // Recreate swapchain reusing the old handle for efficiency
        vk::SwapchainKHR oldHandle = *m_swapchain;
        createSwapchain(oldHandle);
        // The old swapchain RAII object is still live; replace it now
        // (createSwapchain already wrote m_swapchain = new handle)

        // Recreate per-frame images at the new resolution and re-bind descriptors
        for (auto &f : m_frames)
            createStorageImage(f);

        // Re-create descriptor pools and update bindings.  The reference-orbit
        // buffers are size-independent and persist across recreation; we just
        // re-point the new descriptor sets at them.
        for (auto &f : m_frames)
        {
            std::array<vk::DescriptorPoolSize, 2> poolSizes{{
                {.type = vk::DescriptorType::eStorageImage,  .descriptorCount = 1},
                {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1},
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
            // Allocate via RAII then release the raw handle — the pool owns the lifetime.
            // Using release() avoids triggering vkFreeDescriptorSets on the temporary.
            f.dsSet = vk::raii::DescriptorSets(m_device, dsai)[0].release();

            writeFrameDescriptors(f);
        }

        // Recreate renderDone semaphores to match the new swapchain image count
        m_renderDone.clear();
        for (size_t i = 0; i < m_swapImages.size(); ++i)
            m_renderDone.emplace_back(m_device, vk::SemaphoreCreateInfo{});

        // Transition new storage images to GENERAL before the render loop resumes
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
        // Prefer B8G8R8A8Unorm with blit-dst support (not guaranteed by spec)
        for (auto const &f : formats)
            if (f.format == vk::Format::eB8G8R8A8Unorm &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
                supportsBlitDst(f.format))
                return f;
        // Fall back to B8G8R8A8Unorm even without blit-dst guarantee
        for (auto const &f : formats)
            if (f.format == vk::Format::eB8G8R8A8Unorm &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return f;
        // Fall back to sRGB if Unorm is unavailable
        for (auto const &f : formats)
            if (f.format == vk::Format::eB8G8R8A8Srgb &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return f;
        return formats[0];
    }

    static vk::PresentModeKHR chooseMode(std::vector<vk::PresentModeKHR> const &modes)
    {
        // Mailbox drops old frames rather than queuing them – lowest latency for interactive use
        for (auto m : modes)
            if (m == vk::PresentModeKHR::eMailbox)
                return m;
        return vk::PresentModeKHR::eFifo;  // always available
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
        MandelbrotApp app;
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
extern "C" void chapter02_run() {
    try { MandelbrotApp{}.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh02", "%s", e.what()); }
}
#endif
