// Chapter 3 – Memory Models: Interactive 2-D Navier–Stokes Fluid
//
// An Eulerian "Stable Fluids" (Jos Stam) solver.  Drag the mouse to push the
// water and paint dye; the velocity field is advected and projected to be
// divergence-free every frame, so the dye swirls like ink in water.
//
// Demonstrates (the "Memory Models" teaching points):
//   • Memory barriers between compute dispatches — the seven solver passes
//     (splat → advect velocity → divergence → Jacobi×N → gradient →
//      advect dye → render) are each separated by a buffer barrier.
//   • The pressure Poisson solve is a JACOBI iteration: every dispatch reads the
//     previous dispatch's results, so a barrier between each iteration is what
//     makes the writes visible — the canonical inter-dispatch hazard.
//
// Controls:
//   Left drag       – push the fluid and inject dye
//   Scroll up/down  – stronger / weaker push
//   R               – reset (clear the water)
//   Escape          – quit
//
// Build:  see CMakeLists.txt – add WINDOWED to add_compute_chapter()
// Shader: shaders/slang.spv  (compiled from 03_memory_models.slang)

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
#include <random>
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

// Eulerian simulation grid (16:9 to match the window).  The solver runs on this
// fixed grid and the result is upscaled to the swapchain in the render pass.
constexpr uint32_t kGridX        = 320;
constexpr uint32_t kGridY        = 180;
constexpr uint32_t kNumCells     = kGridX * kGridY;
constexpr uint32_t kPressureIters = 40;     // Jacobi iterations per frame

constexpr float    kDt          = 1.0f;     // velocity is in cells/step
constexpr float    kDissipation = 0.997f;   // dye fade per step
constexpr float    kVelFade     = 0.999f;   // velocity damping per step
constexpr float    kSplatRadius = 8.0f;     // injection radius (cells)
constexpr float    kDyeAmount   = 0.45f;    // dye added under the mouse

const std::vector<char const *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// Push-constant layout – must be byte-identical to FluidPush in the shader
// ---------------------------------------------------------------------------
struct FluidPush
{
    uint32_t nx;
    uint32_t ny;
    float    dt;
    float    dissipation;
    float    velFade;
    float    mouseX;       // current mouse, grid coords
    float    mouseY;
    float    mousePx;      // previous mouse, grid coords
    float    mousePy;
    float    splatRadius;
    float    forceScale;
    float    dyeAmount;
    uint32_t mouseDown;
    uint32_t jacobiSrc;    // ping-pong selector / final-pressure selector
    uint32_t imgWidth;
    uint32_t imgHeight;
};
static_assert(sizeof(FluidPush) == 64, "push constant size mismatch");

// ---------------------------------------------------------------------------
// SPHApp
// ---------------------------------------------------------------------------
class SPHApp
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
    // Window + sim state
    // -----------------------------------------------------------------------
    GLFWwindow *m_window    = nullptr;
    bool        m_resized   = false;
    bool        m_dragging  = false;
    double      m_mouseWinX = 0.0, m_mouseWinY = 0.0;  // cursor in window pixels
    float       m_prevGridX = 0.0f, m_prevGridY = 0.0f; // last frame's mouse (grid)
    bool        m_haveMouse = false;                    // seed prev on first drag
    float       m_forceScale = 0.5f;                    // scroll adjusts push strength
    bool        m_resetField = false;                   // set by R to clear next frame
                                                        // (initial state comes from the seed)

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
    // GPU fluid field buffers (persistent across frames – single copy).
    // Memory declared before buffer so RAII destroys the buffer first.
    // -----------------------------------------------------------------------
    struct FieldBuffer { vk::raii::DeviceMemory mem = nullptr; vk::raii::Buffer buf = nullptr; };
    FieldBuffer m_velA, m_velB;     // velocity (float2) + scratch
    FieldBuffer m_pres0, m_pres1;   // pressure ping/pong (float)
    FieldBuffer m_div;              // divergence (float)
    FieldBuffer m_dyeA, m_dyeB;     // dye (float) + scratch

    // -----------------------------------------------------------------------
    // Pipelines / layouts — one per solver pass
    // -----------------------------------------------------------------------
    vk::raii::DescriptorSetLayout m_dsLayout       = nullptr;
    vk::raii::PipelineLayout      m_pipeLayout     = nullptr;
    vk::raii::Pipeline            m_splatPipe      = nullptr;
    vk::raii::Pipeline            m_advectVelPipe  = nullptr;
    vk::raii::Pipeline            m_divergencePipe = nullptr;
    vk::raii::Pipeline            m_jacobiPipe     = nullptr;
    vk::raii::Pipeline            m_gradientPipe   = nullptr;
    vk::raii::Pipeline            m_advectDyePipe  = nullptr;
    vk::raii::Pipeline            m_renderPipe     = nullptr;

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
        vk::DescriptorSet        dsSet  = nullptr;

        vk::raii::CommandBuffer cmdBuf = nullptr;
        vk::raii::Fence         fence  = nullptr;
    };
    std::array<PerFrame, kMaxFrames> m_frames;

    std::vector<vk::raii::Semaphore> m_imageAvail;
    int m_acquireIdx = 0;

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
            "Navier-Stokes Fluid  |  drag=push water  scroll=strength  R=reset  Esc=quit",
            nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, cbResize);
        glfwSetScrollCallback(m_window,           cbScroll);
        glfwSetMouseButtonCallback(m_window,      cbMouseButton);
        glfwSetCursorPosCallback(m_window,        cbCursorPos);
        glfwSetKeyCallback(m_window,              cbKey);
    }

    // -----------------------------------------------------------------------
    // GLFW callbacks
    // -----------------------------------------------------------------------
    static void cbResize(GLFWwindow *w, int, int)
    {
        static_cast<SPHApp *>(glfwGetWindowUserPointer(w))->m_resized = true;
    }

    static void cbScroll(GLFWwindow *w, double /*dx*/, double dy)
    {
        auto *app = static_cast<SPHApp *>(glfwGetWindowUserPointer(w));
        // Scroll adjusts how hard a drag pushes the water.
        app->m_forceScale = std::clamp(
            app->m_forceScale * (dy > 0.0 ? 1.25f : 0.8f), 0.05f, 5.0f);
    }

    static void cbMouseButton(GLFWwindow *w, int button, int action, int /*mods*/)
    {
        auto *app = static_cast<SPHApp *>(glfwGetWindowUserPointer(w));
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            app->m_dragging = (action == GLFW_PRESS);
            if (action == GLFW_PRESS)
            {
                glfwGetCursorPos(w, &app->m_mouseWinX, &app->m_mouseWinY);
                app->m_haveMouse = false;   // reseed prev so the first step has no jump
            }
        }
    }

    static void cbCursorPos(GLFWwindow *w, double mx, double my)
    {
        auto *app = static_cast<SPHApp *>(glfwGetWindowUserPointer(w));
        app->m_mouseWinX = mx;
        app->m_mouseWinY = my;
    }

    static void cbKey(GLFWwindow *w, int key, int /*scancode*/, int action, int /*mods*/)
    {
        if (action != GLFW_PRESS)
            return;
        auto *app = static_cast<SPHApp *>(glfwGetWindowUserPointer(w));
        switch (key)
        {
            case GLFW_KEY_R:
                app->m_resetField = true;   // cleared at the start of the next frame
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
        createFluidBuffers();
        createDescriptorSetLayout();
        createPipelines();
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
            f.fence    = nullptr;
            f.cmdBuf   = nullptr;
            f.dsPool   = nullptr;
            f.storView = nullptr;
            f.storMem  = nullptr;
            f.storImg  = nullptr;
        }
        m_cmdPool           = nullptr;
        m_renderPipe        = nullptr;
        m_advectDyePipe     = nullptr;
        m_gradientPipe      = nullptr;
        m_jacobiPipe        = nullptr;
        m_divergencePipe    = nullptr;
        m_advectVelPipe     = nullptr;
        m_splatPipe         = nullptr;
        m_pipeLayout        = nullptr;
        m_dsLayout          = nullptr;
        for (FieldBuffer *fb : {&m_velA, &m_velB, &m_pres0, &m_pres1,
                                &m_div, &m_dyeA, &m_dyeB})
        {
            fb->buf = nullptr;
            fb->mem = nullptr;
        }
        m_swapchain         = nullptr;
        m_queue             = nullptr;
        m_device            = nullptr;
        m_surface           = nullptr;
        m_debugMessenger    = nullptr;
        m_instance          = nullptr;

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
            .pApplicationName   = "SPH Fluid Simulation",
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
        auto caps    = m_physDev.getSurfaceCapabilitiesKHR(*m_surface);
        m_swapExtent = chooseExtent(caps);

        auto fmts    = m_physDev.getSurfaceFormatsKHR(*m_surface);
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
    // Fluid field GPU buffers — all device-local, cleared to zero via
    // vkCmdFillBuffer (no host staging needed for an all-zero initial state).
    // =======================================================================
    void createFluidBuffers()
    {
        auto deviceLocal     = vk::MemoryPropertyFlagBits::eDeviceLocal;
        // TransferSrc + TransferDst: fields are cleared/seeded via copies and the
        // dye is promoted (dyeB → dyeA) by a buffer copy each frame.
        auto storageTransfer = vk::BufferUsageFlagBits::eStorageBuffer |
                               vk::BufferUsageFlagBits::eTransferDst   |
                               vk::BufferUsageFlagBits::eTransferSrc;

        const vk::DeviceSize vecSz = vk::DeviceSize(kNumCells) * 2u * sizeof(float);
        const vk::DeviceSize sclSz = vk::DeviceSize(kNumCells) * sizeof(float);

        createBuffer(vecSz, storageTransfer, deviceLocal, m_velA.buf, m_velA.mem);
        createBuffer(vecSz, storageTransfer, deviceLocal, m_velB.buf, m_velB.mem);
        createBuffer(sclSz, storageTransfer, deviceLocal, m_pres0.buf, m_pres0.mem);
        createBuffer(sclSz, storageTransfer, deviceLocal, m_pres1.buf, m_pres1.mem);
        createBuffer(sclSz, storageTransfer, deviceLocal, m_div.buf,  m_div.mem);
        createBuffer(sclSz, storageTransfer, deviceLocal, m_dyeA.buf, m_dyeA.mem);
        createBuffer(sclSz, storageTransfer, deviceLocal, m_dyeB.buf, m_dyeB.mem);

        clearFields();
        seedInitialState();
    }

    // Seed a horizontal shear layer (top flows right, bottom flows left) with a
    // row of alternating vortices and a band of dye along the interface.  The
    // shear rolls the dye into a train of swirling billows (the Kelvin–Helmholtz
    // instability), so the demo shows recognisable, correctly-behaving fluid
    // motion the instant it opens — before the user touches the mouse.
    void seedInitialState()
    {
        std::vector<float> vel(kNumCells * 2, 0.0f);
        std::vector<float> dye(kNumCells, 0.0f);

        const float midY    = kGridY * 0.5f;
        const float layerH  = kGridY * 0.06f;   // shear-layer thickness
        const float shearV  = 2.2f;             // base horizontal flow speed

        struct Vortex { float cx, cy, strength; };
        std::vector<Vortex> vortices;
        const int   nv = 6;
        const float vR = kGridY * 0.10f;
        for (int i = 0; i < nv; ++i)
        {
            float cx = kGridX * (0.12f + 0.76f * (i + 0.5f) / nv);
            vortices.push_back({cx, midY, (i & 1) ? 2.4f : -2.4f});
        }

        for (uint32_t y = 0; y < kGridY; ++y)
        {
            for (uint32_t x = 0; x < kGridX; ++x)
            {
                uint32_t idx = y * kGridX + x;
                float px = x + 0.5f, py = y + 0.5f;

                // Shear layer: smooth tanh-like profile through the interface.
                float s = (py - midY) / layerH;
                float flow = shearV * std::tanh(s);
                vel[idx * 2 + 0] = flow;

                // Fill the whole domain with dye that varies in a soft large-scale
                // pattern, plus a bright band on the shear interface.  Because the
                // screen is never empty, the flow always has something to swirl —
                // the water stays full and lively instead of draining to one side.
                float base  = 0.45f + 0.30f * std::sin(px * 0.055f) * std::sin(py * 0.06f);
                float band  = 0.55f * std::exp(-(py - midY) * (py - midY) /
                                    (2.0f * (layerH * 1.8f) * (layerH * 1.8f)));
                dye[idx] = std::clamp(base + band, 0.0f, 1.0f);

                // Vortices kick off the rollup.
                for (auto &v : vortices)
                {
                    float dx = px - v.cx, dy = py - v.cy;
                    float r2 = dx * dx + dy * dy;
                    float fall = std::exp(-r2 / (2.0f * vR * vR));
                    vel[idx * 2 + 0] += -dy * v.strength * fall * 0.05f;
                    vel[idx * 2 + 1] +=  dx * v.strength * fall * 0.05f;
                }
            }
        }

        vk::DeviceSize velSz = vk::DeviceSize(kNumCells) * 2u * sizeof(float);
        vk::DeviceSize dyeSz = vk::DeviceSize(kNumCells) * sizeof(float);

        vk::raii::Buffer       stage   = nullptr;
        vk::raii::DeviceMemory stageMem = nullptr;
        createBuffer(velSz + dyeSz, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                     stage, stageMem);
        void *ptr = stageMem.mapMemory(0, velSz + dyeSz);
        std::memcpy(ptr, vel.data(), velSz);
        std::memcpy(static_cast<char *>(ptr) + velSz, dye.data(), dyeSz);
        stageMem.unmapMemory();

        auto cmdBufs = vk::raii::CommandBuffers(m_device,
            vk::CommandBufferAllocateInfo{
                .commandPool        = *m_cmdPool,
                .level              = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1});
        auto &cb = cmdBufs[0];
        cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cb.copyBuffer(*stage, *m_velA.buf, vk::BufferCopy{.srcOffset = 0,      .size = velSz});
        cb.copyBuffer(*stage, *m_dyeA.buf, vk::BufferCopy{.srcOffset = velSz,  .size = dyeSz});
        cb.end();
        vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
        m_queue.submit(si, nullptr);
        m_queue.waitIdle();
    }

    // Zero every field buffer (initial state and the R-key reset).
    void clearFields()
    {
        auto cmdBufs = vk::raii::CommandBuffers(m_device,
            vk::CommandBufferAllocateInfo{
                .commandPool        = *m_cmdPool,
                .level              = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1});
        auto &cb = cmdBufs[0];
        cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        for (FieldBuffer *fb : {&m_velA, &m_velB, &m_pres0, &m_pres1,
                                &m_div, &m_dyeA, &m_dyeB})
            cb.fillBuffer(*fb->buf, 0, VK_WHOLE_SIZE, 0u);
        cb.end();
        vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
        m_queue.submit(si, nullptr);
        m_queue.waitIdle();
    }

    // =======================================================================
    // Descriptor set layout
    //   binding 0 = velA (storage buffer, float2)   binding 1 = velB
    //   binding 2 = pres0 (float)                    binding 3 = pres1
    //   binding 4 = divergence (float)
    //   binding 5 = dyeA (float)                     binding 6 = dyeB
    //   binding 7 = output storage image
    // =======================================================================
    void createDescriptorSetLayout()
    {
        std::array<vk::DescriptorSetLayoutBinding, 8> bindings{};
        for (uint32_t i = 0; i < 7; ++i)
            bindings[i] = vk::DescriptorSetLayoutBinding{
                .binding = i, .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute};
        bindings[7] = vk::DescriptorSetLayoutBinding{
            .binding = 7, .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute};

        vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings    = bindings.data()};
        m_dsLayout = vk::raii::DescriptorSetLayout(m_device, ci);
    }

    // =======================================================================
    // Pipelines — one compute pipeline per solver pass (all share the layout)
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
            .size       = sizeof(FluidPush)};
        vk::PipelineLayoutCreateInfo plci{
            .setLayoutCount         = 1,
            .pSetLayouts            = &*m_dsLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcRange};
        m_pipeLayout = vk::raii::PipelineLayout(m_device, plci);

        auto makePipeline = [&](char const *entry) {
            vk::PipelineShaderStageCreateInfo stage{
                .stage  = vk::ShaderStageFlagBits::eCompute,
                .module = *shaderModule,
                .pName  = entry};
            return vk::raii::Pipeline(m_device, nullptr,
                vk::ComputePipelineCreateInfo{.stage = stage, .layout = *m_pipeLayout});
        };

        m_splatPipe      = makePipeline("splatPass");
        m_advectVelPipe  = makePipeline("advectVelPass");
        m_divergencePipe = makePipeline("divergencePass");
        m_jacobiPipe     = makePipeline("jacobiPass");
        m_gradientPipe   = makePipeline("gradientPass");
        m_advectDyePipe  = makePipeline("advectDyePass");
        m_renderPipe     = makePipeline("renderPass");
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
        // Pool sizes: 7 storage buffers + 1 storage image
        std::array<vk::DescriptorPoolSize, 2> poolSizes{{
            {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 7},
            {.type = vk::DescriptorType::eStorageImage,  .descriptorCount = 1},
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

        // The seven field buffers map to bindings 0..6 in declaration order.
        FieldBuffer *fields[7] = {&m_velA, &m_velB, &m_pres0, &m_pres1,
                                  &m_div, &m_dyeA, &m_dyeB};
        std::array<vk::DescriptorBufferInfo, 7> bufInfos{};
        for (uint32_t i = 0; i < 7; ++i)
            bufInfos[i] = vk::DescriptorBufferInfo{
                .buffer = *fields[i]->buf, .offset = 0, .range = VK_WHOLE_SIZE};

        vk::DescriptorImageInfo imgInfo{
            .imageView   = *f.storView,
            .imageLayout = vk::ImageLayout::eGeneral};

        std::array<vk::WriteDescriptorSet, 8> writes{};
        for (uint32_t i = 0; i < 7; ++i)
            writes[i] = vk::WriteDescriptorSet{
                .dstSet = f.dsSet, .dstBinding = i, .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &bufInfos[i]};
        writes[7] = vk::WriteDescriptorSet{
            .dstSet = f.dsSet, .dstBinding = 7, .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &imgInfo};
        m_device.updateDescriptorSets(writes, {});
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
    void drawFrame()
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

        m_frameIdx = (m_frameIdx + 1) % kMaxFrames;
    }

    // =======================================================================
    // Command recording – SPH simulation dispatches + render + blit
    // =======================================================================
    void recordCommands(PerFrame &f, uint32_t imageIndex)
    {
        auto &cb = f.cmdBuf;
        cb.reset();
        cb.begin({});

        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                              *m_pipeLayout, 0, {f.dsSet}, {});

        // Global compute↔compute barrier between solver passes.  Each pass reads
        // the field a previous pass wrote, so this barrier (the chapter's whole
        // point) makes those storage writes visible before the next dispatch.
        auto barrier = [&]() {
            vk::MemoryBarrier2 mb{
                .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                                 vk::AccessFlagBits2::eShaderStorageWrite};
            cb.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1, .pMemoryBarriers = &mb});
        };

        const uint32_t gGx = (kGridX + 15u) / 16u;
        const uint32_t gGy = (kGridY + 15u) / 16u;

        // R-key reset: clear every field to zero, then make it visible to compute.
        if (m_resetField)
        {
            m_resetField = false;
            for (FieldBuffer *fb : {&m_velA, &m_velB, &m_pres0, &m_pres1,
                                    &m_div, &m_dyeA, &m_dyeB})
                cb.fillBuffer(*fb->buf, 0, VK_WHOLE_SIZE, 0u);
            vk::MemoryBarrier2 mb{
                .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                                 vk::AccessFlagBits2::eShaderStorageWrite};
            cb.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1, .pMemoryBarriers = &mb});
        }

        // Mouse in grid coordinates (cursor Y is top-down, matching the image).
        float winW = static_cast<float>(m_swapExtent.width);
        float winH = static_cast<float>(m_swapExtent.height);
        float gxm  = static_cast<float>(m_mouseWinX / winW) * kGridX;
        float gym  = static_cast<float>(m_mouseWinY / winH) * kGridY;
        if (!m_haveMouse) { m_prevGridX = gxm; m_prevGridY = gym; m_haveMouse = true; }

        FluidPush push{
            .nx          = kGridX,
            .ny          = kGridY,
            .dt          = kDt,
            .dissipation = kDissipation,
            .velFade     = kVelFade,
            .mouseX      = gxm,
            .mouseY      = gym,
            .mousePx     = m_prevGridX,
            .mousePy     = m_prevGridY,
            .splatRadius = kSplatRadius,
            .forceScale  = m_forceScale,
            .dyeAmount   = kDyeAmount,
            .mouseDown   = m_dragging ? 1u : 0u,
            .jacobiSrc   = 0u,
            .imgWidth    = m_swapExtent.width,
            .imgHeight   = m_swapExtent.height};
        m_prevGridX = gxm;
        m_prevGridY = gym;

        auto setPush = [&]() {
            cb.pushConstants<FluidPush>(*m_pipeLayout,
                                        vk::ShaderStageFlagBits::eCompute, 0, push);
        };

        // The fields persist across frames (single copy, shared by both in-flight
        // frames).  Make the previous frame's velocity write (compute) and dye
        // promotion (transfer copy) visible before this frame's first pass reads
        // them.  On a single queue a barrier's source scope spans earlier submits.
        {
            vk::MemoryBarrier2 mb{
                .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader |
                                 vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite |
                                 vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                                 vk::AccessFlagBits2::eShaderStorageWrite};
            cb.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1, .pMemoryBarriers = &mb});
        }

        // PASS 1 – inject velocity + dye under the mouse (writes velA, dyeA).
        setPush();
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_splatPipe);
        cb.dispatch(gGx, gGy, 1);
        barrier();

        // PASS 2 – advect velocity (velA → velB).
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_advectVelPipe);
        cb.dispatch(gGx, gGy, 1);
        barrier();

        // PASS 3 – divergence of velB; clears the pressure field.
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_divergencePipe);
        cb.dispatch(gGx, gGy, 1);
        barrier();

        // PASS 4 – Jacobi pressure solve.  Ping-pong pres0/pres1 with a barrier
        // between every iteration so each reads the previous iteration's result.
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_jacobiPipe);
        for (uint32_t it = 0; it < kPressureIters; ++it)
        {
            push.jacobiSrc = it & 1u;   // 0: read pres0/write pres1, 1: swap
            setPush();
            cb.dispatch(gGx, gGy, 1);
            barrier();
        }
        // The final pressure lives in the buffer the last iteration wrote.
        // jacobiSrc s writes pres1 when s==0, pres0 when s==1.  For gradientPass,
        // jacobiSrc selects which buffer to READ, so point it at that final one.
        uint32_t lastS = (kPressureIters - 1u) & 1u;
        push.jacobiSrc = (lastS == 0u) ? 1u : 0u;   // final buffer holds the pressure
        setPush();

        // PASS 5 – subtract ∇pressure, enforce walls (velB → velA).
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_gradientPipe);
        cb.dispatch(gGx, gGy, 1);
        barrier();

        // PASS 6 – advect dye by the divergence-free velocity (dyeA → dyeB).
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_advectDyePipe);
        cb.dispatch(gGx, gGy, 1);

        // Make dyeB visible to both the render pass (compute) and the dye copy
        // (transfer) below.
        {
            vk::MemoryBarrier2 mb{
                .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader |
                                 vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                                 vk::AccessFlagBits2::eTransferRead};
            cb.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1, .pMemoryBarriers = &mb});
        }

        // PASS 7 – render the dye field (dyeB, velA) into the storage image.
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_renderPipe);
        uint32_t gx = (m_swapExtent.width  + 15u) / 16u;
        uint32_t gy = (m_swapExtent.height + 15u) / 16u;
        cb.dispatch(gx, gy, 1);

        // Promote dyeB → dyeA so the freshly-advected dye is the current field
        // for the next frame.  (velA already holds the projected velocity.)
        cb.copyBuffer(*m_dyeB.buf, *m_dyeA.buf,
                      vk::BufferCopy{.size = vk::DeviceSize(kNumCells) * sizeof(float)});

        // -----------------------------------------------------------------------
        // BARRIER 3: renderPass write → blit read (storage image)
        //            + swapchain UNDEFINED → TRANSFER_DST
        // -----------------------------------------------------------------------
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

        // Post-blit: swapchain → PRESENT_SRC, storage image release
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
    // Buffer helper
    // =======================================================================
    void createBuffer(vk::DeviceSize             size,
                      vk::BufferUsageFlags       usage,
                      vk::MemoryPropertyFlags    memProps,
                      vk::raii::Buffer          &outBuf,
                      vk::raii::DeviceMemory    &outMem)
    {
        vk::BufferCreateInfo bci{
            .size        = size,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive};
        outBuf = vk::raii::Buffer(m_device, bci);

        auto memReqs = outBuf.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = memReqs.size,
            .memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, memProps)};
        outMem = vk::raii::DeviceMemory(m_device, mai);
        outBuf.bindMemory(*outMem, 0);
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
        SPHApp app;
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
extern "C" void chapter03_run() {
    try { SPHApp{}.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh03", "%s", e.what()); }
}
#endif
