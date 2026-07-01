// Chapter 8 – Asynchronous Compute: Cloth Physics Simulation
//
// Demonstrates true asynchronous compute using two separate Vulkan queues:
//
//   - A DEDICATED ASYNC COMPUTE QUEUE runs cloth Verlet integration and spring-
//     constraint solving (8 Jacobi iterations per frame).  A compute-only queue
//     family is preferred; the code falls back to a second queue from the
//     graphics family, or to sharing queue 0 if only one queue is available.
//
//   - The GRAPHICS QUEUE renders a 32×32 cloth mesh as a lit triangle mesh
//     plus a collision sphere, reading the vertex positions written by the
//     compute pass of the *previous* frame.  This is the key overlap: while
//     the GPU renders frame N it simultaneously solves physics for frame N+1.
//
//   - A TIMELINE SEMAPHORE with monotonically increasing values coordinates
//     the two queues so that graphics never reads cloth positions while
//     compute is still writing them.
//
// Frame N timeline (true async overlap):
//   compute(N) : wait  lastGraphicsSignal(N-1) [0 for frame 0]
//                → Verlet integrate, writes clothPositionBuffers[(frameIndex+1)%2]
//                → signal computeSignal(N)
//   graphics(N): wait  computeTimeline(N-1)    [0 for frame 0]
//                → reads clothPositionBuffers[frameIndex] (written by compute(N-1))
//                → signal graphicsSignal(N)
// Because the two passes operate on different buffer slots they run concurrently
// on the GPU: graphics(N) renders frame N-1's cloth while compute(N) solves frame N.
//   CPU     : present  (inFlightFence guards resource reuse; renderDone sem gates present)
//
// The cloth:
//   • 32×32 grid of vertices (1024 total), pinned at top-left and top-right.
//   • Falls under gravity and drapes over a sphere at (0, -0.2, 0).
//   • Rendered as an indexed triangle list; indices are generated once on the CPU.
//   • Cloth positions live in two device-local SSBOs (positions + prevPositions).
//   • UV coordinates for the checker-board texture are in a separate vertex buffer.
//
// Build:  see CMakeLists.txt – add_compute_chapter(08_async_compute WINDOWED …)
// Shader: shaders/slang.spv  (compiled from 08_async_compute.slang)

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
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#ifdef ANDROID_BUILD
#  include "glfw_android_shim.h"
#else
#  define GLFW_INCLUDE_VULKAN
#  include <GLFW/glfw3.h>
#endif

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr uint32_t WIDTH           = 1280;
constexpr uint32_t HEIGHT          = 720;
// Cloth is 32×32 = 1024 vertices so the entire grid fits in ONE compute
// workgroup, letting the constraint solver use a real workgroup-wide barrier
// between relaxation steps.  maxComputeWorkGroupInvocations minimum is 128
// (Vulkan 1.0 core) and 256 (Roadmap 2022 / Vulkan 1.4); CLOTH_N=1024 is
// not guaranteed – isDeviceSuitable enforces it at runtime.
constexpr uint32_t CLOTH_W         = 32;
constexpr uint32_t CLOTH_H         = 32;
constexpr uint32_t CLOTH_N         = CLOTH_W * CLOTH_H;          // 1024
constexpr uint32_t CLOTH_TRIS      = (CLOTH_W - 1) * (CLOTH_H - 1) * 2;
constexpr uint32_t CLOTH_INDICES   = CLOTH_TRIS * 3;
constexpr uint32_t CONSTRAINT_ITER = 24;
constexpr int      MAX_FRAMES      = 2;

// Sphere rendered via a separate indexed draw (icosphere approximation is
// replaced here with a simple pre-tessellated sphere built on the CPU).
constexpr uint32_t SPHERE_RINGS    = 24;
constexpr uint32_t SPHERE_SEGS     = 32;

// Sphere collision/animation: the sphere mesh is baked centred at y = SPHERE_BASE_Y
// and translated each frame to sphereCentreY(t).  The same curve drives both the
// physics (compute) and the rendered sphere so they stay locked together.
constexpr float SPHERE_BASE_Y   = 0.3f;
constexpr float SPHERE_RADIUS   = 0.55f;
inline float sphereCentreY(float t) { return 0.3f + 0.6f * std::sin(t * 0.6f); }

const std::vector<char const *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// CPU-side data structures
// ---------------------------------------------------------------------------

// Matches ClothUBO in the shader (std140: 8 floats = 32 bytes)
struct ClothUBO
{
    float    deltaTime    = 0.016f;
    uint32_t iterCount    = CONSTRAINT_ITER;
    float    time         = 0.0f;
    float    pad          = 0.0f;
    glm::vec3 sphereCenter = {0.0f, 0.3f, 0.0f};
    float    sphereRadius = 0.55f;
};

// Matches ClothPush in the shader (two float4x4 + one float4)
struct ClothPush
{
    glm::mat4 mvp;
    glm::mat4 normalMatrix;
    glm::vec4 lightDir;   // xyz = light direction, w = renderSphere flag
};

// Vertex for cloth: position SSBO is used directly; UV is a separate vertex
// buffer so the layout matches [[vk::location(0)]] float4 and [[vk::location(1)]] float2.
struct ClothUV
{
    glm::vec2 uv;
};

// Sphere vertex (position only – normals computed in fragment via derivatives)
struct SphereVertex
{
    glm::vec3 position;
};

// ---------------------------------------------------------------------------
// Utility: build sphere geometry
// ---------------------------------------------------------------------------
static void buildSphere(float radius, glm::vec3 centre,
                        std::vector<SphereVertex> &verts,
                        std::vector<uint32_t>      &inds)
{
    verts.clear();
    inds.clear();

    for (uint32_t r = 0; r <= SPHERE_RINGS; ++r)
    {
        float phi = glm::pi<float>() * float(r) / float(SPHERE_RINGS);
        for (uint32_t s = 0; s <= SPHERE_SEGS; ++s)
        {
            float theta = 2.0f * glm::pi<float>() * float(s) / float(SPHERE_SEGS);
            glm::vec3 p{
                radius * std::sin(phi) * std::cos(theta),
                radius * std::cos(phi),
                radius * std::sin(phi) * std::sin(theta)};
            verts.push_back({centre + p});
        }
    }

    for (uint32_t r = 0; r < SPHERE_RINGS; ++r)
    {
        for (uint32_t s = 0; s < SPHERE_SEGS; ++s)
        {
            uint32_t a = r * (SPHERE_SEGS + 1) + s;
            uint32_t b = a + 1;
            uint32_t c = a + (SPHERE_SEGS + 1);
            uint32_t d = c + 1;
            inds.push_back(a); inds.push_back(c); inds.push_back(b);
            inds.push_back(b); inds.push_back(c); inds.push_back(d);
        }
    }
}

// ===========================================================================
// Application class
// ===========================================================================
class AsyncComputeApplication
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
    // GLFW / Vulkan core objects
    // -----------------------------------------------------------------------
    GLFWwindow                      *window         = nullptr;
    vk::raii::Context                context;
    vk::raii::Instance               instance       = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR             surface        = nullptr;
    vk::raii::PhysicalDevice         physicalDevice = nullptr;
    vk::raii::Device                 device         = nullptr;

    // -----------------------------------------------------------------------
    // Two queues: graphics+present and async compute.
    // asyncComputeQueueFamily may equal graphicsQueueFamily when the device
    // has no dedicated compute-only family.
    // -----------------------------------------------------------------------
    uint32_t        graphicsQueueFamily     = ~0u;
    uint32_t        asyncComputeQueueFamily = ~0u;
    vk::raii::Queue graphicsQueue           = nullptr;
    vk::raii::Queue asyncComputeQueue       = nullptr;

    // -----------------------------------------------------------------------
    // Swapchain
    // -----------------------------------------------------------------------
    vk::raii::SwapchainKHR           swapChain      = nullptr;
    std::vector<vk::Image>           swapChainImages;
    vk::SurfaceFormatKHR             swapChainSurfaceFormat;
    vk::Extent2D                     swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    // -----------------------------------------------------------------------
    // Depth buffer (memory first so RAII destroys buffer/image before freeing memory)
    // -----------------------------------------------------------------------
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::Image        depthImage       = nullptr;
    vk::raii::ImageView    depthImageView   = nullptr;
    vk::Format             depthFormat      = vk::Format::eD32Sfloat;

    // -----------------------------------------------------------------------
    // Pipelines
    // -----------------------------------------------------------------------
    // Compute pipeline (cloth physics)
    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout      computePipelineLayout      = nullptr;
    vk::raii::Pipeline            computePipeline            = nullptr;

    // Graphics pipeline (cloth + sphere, push-constant driven)
    vk::raii::PipelineLayout graphicsPipelineLayout = nullptr;
    vk::raii::Pipeline       clothPipeline          = nullptr;
    vk::raii::Pipeline       spherePipeline         = nullptr;

    // -----------------------------------------------------------------------
    // Cloth GPU buffers (per frame – double-buffered)
    // Memory declared before buffer so RAII destroys buffer before freeing memory.
    // -----------------------------------------------------------------------
    std::vector<vk::raii::DeviceMemory> clothPositionMemory;
    std::vector<vk::raii::Buffer>       clothPositionBuffers;
    std::vector<vk::raii::DeviceMemory> clothPrevMemory;
    std::vector<vk::raii::Buffer>       clothPrevBuffers;
    // Smooth per-vertex normals (written by compute, read as a vertex attribute)
    std::vector<vk::raii::DeviceMemory> clothNormalMemory;
    std::vector<vk::raii::Buffer>       clothNormalBuffers;

    // UV coordinates (static, uploaded once)
    vk::raii::DeviceMemory clothUVMemory  = nullptr;
    vk::raii::Buffer       clothUVBuffer  = nullptr;

    // Index buffer for cloth triangle list (static)
    vk::raii::DeviceMemory clothIndexMemory  = nullptr;
    vk::raii::Buffer       clothIndexBuffer  = nullptr;

    // -----------------------------------------------------------------------
    // Sphere GPU buffers (static)
    // -----------------------------------------------------------------------
    vk::raii::DeviceMemory sphereVertexMemory = nullptr;
    vk::raii::Buffer       sphereVertexBuffer = nullptr;
    vk::raii::DeviceMemory sphereIndexMemory  = nullptr;
    vk::raii::Buffer       sphereIndexBuffer  = nullptr;
    vk::raii::DeviceMemory sphereNormalMemory = nullptr;
    vk::raii::Buffer       sphereNormalBuffer = nullptr;
    uint32_t               sphereIndexCount   = 0;

    // -----------------------------------------------------------------------
    // Uniform buffers (compute UBO, per frame)
    // -----------------------------------------------------------------------
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<vk::raii::Buffer>       uniformBuffers;
    std::vector<void *>                 uniformBuffersMapped;

    // -----------------------------------------------------------------------
    // Descriptor sets (compute, per frame)
    // -----------------------------------------------------------------------
    vk::raii::DescriptorPool             descriptorPool     = nullptr;
    std::vector<vk::raii::DescriptorSet> computeDescSets;

    // -----------------------------------------------------------------------
    // Command pools and buffers
    // -----------------------------------------------------------------------
    vk::raii::CommandPool                graphicsCommandPool     = nullptr;
    vk::raii::CommandPool                asyncComputeCommandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::raii::CommandBuffer> computeCommandBuffers;

    // -----------------------------------------------------------------------
    // Synchronisation
    //
    // Two timeline semaphores — one per queue — so each is only ever signalled by
    // its own queue.  A single shared timeline would require signals from both
    // queues to arrive in strictly increasing value order, which is not guaranteed
    // when both start with no GPU-side wait (frame 0).
    //
    //   computeTimeline : only the async-compute queue signals this
    //                     compute[N] signals N+1
    //   graphicsTimeline: only the graphics queue signals this
    //                     graphics[N] signals N+1
    //
    // Cross-queue waits:
    //   compute[N]  waits graphicsTimeline >= N   (graphics[N-1] released the write slot)
    //   graphics[N] waits computeTimeline  >= N   (compute[N-1]  finished the read  slot)
    // Both start at 0, so frame 0 has no GPU-side wait on either side.
    //
    // Binary semaphores for image acquisition and per-image render-done use.
    // -----------------------------------------------------------------------
    vk::raii::Semaphore computeTimeline  = nullptr;  // signalled only by async-compute queue
    vk::raii::Semaphore graphicsTimeline = nullptr;  // signalled only by graphics queue
    uint64_t            frameCount       = 0;        // monotonic; drives both timeline values
    std::vector<vk::raii::Semaphore> acquireSemaphores;      // MAX_FRAMES+1 rotating pool
    std::vector<vk::raii::Semaphore> renderDoneSems;       // one per swapchain image
    std::vector<vk::raii::Fence>     inFlightFences;       // guard graphics command-buffer reuse
    std::vector<vk::raii::Fence>     computeInFlightFences; // guard compute command-buffer reuse

    uint32_t acquireSemIdx = 0;  // rotating acquire semaphore index
    uint32_t frameIndex    = 0;

    // -----------------------------------------------------------------------
    // Timing
    // -----------------------------------------------------------------------
    double startTime       = 0.0;
    double lastFrameTime   = 0.0;
    double lastWallTime    = 0.0;
    bool   framebufferResized = false;

    std::vector<const char *> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName};

    // =======================================================================
    // Window
    // =======================================================================
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan – Cloth Async Compute", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        startTime    = glfwGetTime();
        lastWallTime = startTime;
    }

    static void framebufferResizeCallback(GLFWwindow *w, int, int)
    {
        static_cast<AsyncComputeApplication *>(glfwGetWindowUserPointer(w))->framebufferResized = true;
    }

    // =======================================================================
    // Vulkan initialisation sequence
    // =======================================================================
    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createDepthResources();
        createComputeDescriptorSetLayout();
        createGraphicsPipelines();
        createComputePipeline();
        createCommandPools();
        createClothBuffers();
        createSphereBuffers();
        createUniformBuffers();
        createDescriptorPool();
        createComputeDescriptorSets();
        createCommandBuffers();
        createComputeCommandBuffers();
        createSyncObjects();
    }

    // =======================================================================
    // Main loop
    // =======================================================================
    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }
        device.waitIdle();
    }

    // =======================================================================
    // Cleanup helpers
    // =======================================================================
    void cleanupSwapChain()
    {
        depthImageView  = nullptr;
        depthImage      = nullptr;
        depthImageMemory = nullptr;
        swapChainImageViews.clear();
        swapChain = nullptr;
    }

    void cleanup()
    {
        cleanupSwapChain();
        surface = nullptr;  // must destroy VkSurfaceKHR before GLFW closes Wayland display
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void recreateSwapChain()
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        while (w == 0 || h == 0)
        {
            glfwGetFramebufferSize(window, &w, &h);
            glfwWaitEvents();
        }
        device.waitIdle();
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createDepthResources();
    }

    // =======================================================================
    // Instance
    // =======================================================================
    void createInstance()
    {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName   = "Cloth Async Compute",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = vk::ApiVersion13};

        std::vector<char const *> requiredLayers;
        if (kEnableValidation)
            requiredLayers.assign(kValidationLayers.begin(), kValidationLayers.end());

        auto layerProps = context.enumerateInstanceLayerProperties();
        auto missingIt  = std::ranges::find_if(requiredLayers, [&](const char *req) {
            return std::ranges::none_of(layerProps, [req](auto const &lp) {
                return strcmp(lp.layerName, req) == 0;
            });
        });
        if (missingIt != requiredLayers.end())
            throw std::runtime_error("Required layer not supported: " + std::string(*missingIt));

        auto requiredExtensions = getRequiredInstanceExtensions();

        auto extProps   = context.enumerateInstanceExtensionProperties();
        auto missingExt = std::ranges::find_if(requiredExtensions, [&](const char *req) {
            return std::ranges::none_of(extProps, [req](auto const &ep) {
                return strcmp(ep.extensionName, req) == 0;
            });
        });
        if (missingExt != requiredExtensions.end())
            throw std::runtime_error("Required extension not supported: " + std::string(*missingExt));

        vk::InstanceCreateInfo ci{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames     = requiredLayers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data()};
        instance = vk::raii::Instance(context, ci);
    }

    void setupDebugMessenger()
    {
        if (!kEnableValidation) return;
        vk::DebugUtilsMessageSeverityFlagsEXT sev(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT typ(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        debugMessenger = instance.createDebugUtilsMessengerEXT(
            {.messageSeverity = sev, .messageType = typ, .pfnUserCallback = &debugCallback});
    }

    void createSurface()
    {
        VkSurfaceKHR raw;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &raw) != VK_SUCCESS)
            throw std::runtime_error("failed to create window surface!");
        surface = vk::raii::SurfaceKHR(instance, raw);
    }

    // =======================================================================
    // Physical device
    // =======================================================================
    bool isDeviceSuitable(vk::raii::PhysicalDevice const &pd)
    {
        bool ok13 = pd.getProperties().apiVersion >= VK_API_VERSION_1_3;

        auto qfps         = pd.getQueueFamilyProperties();
        bool hasGraphics  = std::ranges::any_of(qfps, [](auto const &q) {
            return !!(q.queueFlags & vk::QueueFlagBits::eGraphics);
        });

        auto extProps = pd.enumerateDeviceExtensionProperties();
        bool hasExts  = std::ranges::all_of(requiredDeviceExtensions, [&](const char *req) {
            return std::ranges::any_of(extProps, [req](auto const &ep) {
                return strcmp(ep.extensionName, req) == 0;
            });
        });

        auto feats = pd.getFeatures2<vk::PhysicalDeviceFeatures2,
                                     vk::PhysicalDeviceVulkan12Features,
                                     vk::PhysicalDeviceVulkan13Features>();
        bool okFeats =
            feats.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore &&
            feats.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering  &&
            feats.get<vk::PhysicalDeviceVulkan13Features>().synchronization2;

        bool hasWorkgroupSize =
            pd.getProperties().limits.maxComputeWorkGroupInvocations >= CLOTH_N;

        return ok13 && hasGraphics && hasExts && okFeats && hasWorkgroupSize;
    }

    void pickPhysicalDevice()
    {
        // Among all suitable devices, prefer discrete > integrated > virtual > other.
        auto typeScore = [](vk::PhysicalDeviceType t) -> int {
            switch (t) {
                case vk::PhysicalDeviceType::eDiscreteGpu:   return 4;
                case vk::PhysicalDeviceType::eIntegratedGpu: return 3;
                case vk::PhysicalDeviceType::eVirtualGpu:    return 2;
                default:                                      return 1;
            }
        };
        int bestScore = 0;
        for (auto &pd : instance.enumeratePhysicalDevices())
        {
            if (!isDeviceSuitable(pd)) continue;
            int score = typeScore(pd.getProperties().deviceType);
            if (score > bestScore) { bestScore = score; physicalDevice = pd; }
        }
        if (bestScore == 0)
            throw std::runtime_error("failed to find a suitable GPU!");
        std::cout << "[Cloth] GPU: " << physicalDevice.getProperties().deviceName << "\n";
    }

    // =======================================================================
    // Logical device – two queues
    // =======================================================================
    void createLogicalDevice()
    {
        auto qfps = physicalDevice.getQueueFamilyProperties();

        // Find graphics+present family
        for (uint32_t i = 0; i < static_cast<uint32_t>(qfps.size()); ++i)
        {
            if ((qfps[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                physicalDevice.getSurfaceSupportKHR(i, *surface))
            {
                graphicsQueueFamily = i;
                break;
            }
        }
        if (graphicsQueueFamily == ~0u)
            throw std::runtime_error("No graphics+present queue family found.");

        // Prefer dedicated compute-only family
        for (uint32_t i = 0; i < static_cast<uint32_t>(qfps.size()); ++i)
        {
            bool hasC = !!(qfps[i].queueFlags & vk::QueueFlagBits::eCompute);
            bool hasG = !!(qfps[i].queueFlags & vk::QueueFlagBits::eGraphics);
            if (hasC && !hasG)
            {
                asyncComputeQueueFamily = i;
                break;
            }
        }

        if (asyncComputeQueueFamily == ~0u)
        {
            asyncComputeQueueFamily = graphicsQueueFamily;
            std::cout << "[AsyncCompute] No dedicated compute-only family found. "
                         "Falling back to graphics family " << graphicsQueueFamily << ".\n";
        }
        else
        {
            std::cout << "[AsyncCompute] Dedicated async compute family "
                      << asyncComputeQueueFamily << ".\n";
        }

        static const float prio[2] = {0.5f, 0.5f};
        std::vector<vk::DeviceQueueCreateInfo> qcis;
        if (graphicsQueueFamily != asyncComputeQueueFamily)
        {
            qcis.push_back({.queueFamilyIndex = graphicsQueueFamily,     .queueCount = 1, .pQueuePriorities = prio});
            qcis.push_back({.queueFamilyIndex = asyncComputeQueueFamily, .queueCount = 1, .pQueuePriorities = prio});
        }
        else
        {
            uint32_t avail   = qfps[graphicsQueueFamily].queueCount;
            uint32_t request = std::min(avail, 2u);
            if (request < 2u)
                std::cout << "[AsyncCompute] Only one queue in family " << graphicsQueueFamily
                          << "; graphics and compute share queue 0.\n";
            qcis.push_back({.queueFamilyIndex = graphicsQueueFamily, .queueCount = request, .pQueuePriorities = prio});
        }

        vk::StructureChain<vk::PhysicalDeviceFeatures2,
                           vk::PhysicalDeviceVulkan12Features,
                           vk::PhysicalDeviceVulkan13Features>
            featureChain = {
                {.features = {.samplerAnisotropy = true}},
                {.scalarBlockLayout = true, .timelineSemaphore = true},
                {.synchronization2 = true, .dynamicRendering = true}};

        vk::DeviceCreateInfo dci{
            .pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount    = static_cast<uint32_t>(qcis.size()),
            .pQueueCreateInfos       = qcis.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtensions.size()),
            .ppEnabledExtensionNames = requiredDeviceExtensions.data()};

        device = vk::raii::Device(physicalDevice, dci);

        graphicsQueue = vk::raii::Queue(device, graphicsQueueFamily, 0);

        if (graphicsQueueFamily != asyncComputeQueueFamily)
            asyncComputeQueue = vk::raii::Queue(device, asyncComputeQueueFamily, 0);
        else
        {
            uint32_t cqIdx = (qfps[graphicsQueueFamily].queueCount >= 2u) ? 1u : 0u;
            asyncComputeQueue = vk::raii::Queue(device, asyncComputeQueueFamily, cqIdx);
        }
    }

    // =======================================================================
    // Swapchain
    // =======================================================================
    void createSwapChain()
    {
        auto caps   = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        auto fmts   = physicalDevice.getSurfaceFormatsKHR(*surface);
        auto modes  = physicalDevice.getSurfacePresentModesKHR(*surface);

        swapChainExtent        = chooseExtent(caps);
        swapChainSurfaceFormat = chooseFormat(fmts);
        auto presentMode       = choosePresent(modes);
        uint32_t minImg        = chooseMinImageCount(caps);

        vk::SwapchainCreateInfoKHR sci{
            .surface          = *surface,
            .minImageCount    = minImg,
            .imageFormat      = swapChainSurfaceFormat.format,
            .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
            .imageExtent      = swapChainExtent,
            .imageArrayLayers = 1,
            .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform     = caps.currentTransform,
            .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode      = presentMode,
            .clipped          = true};

        swapChain       = vk::raii::SwapchainKHR(device, sci);
        swapChainImages = swapChain.getImages();
    }

    void createImageViews()
    {
        assert(swapChainImageViews.empty());
        vk::ImageViewCreateInfo ivci{
            .viewType         = vk::ImageViewType::e2D,
            .format           = swapChainSurfaceFormat.format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        for (auto &img : swapChainImages)
        {
            ivci.image = img;
            swapChainImageViews.emplace_back(device, ivci);
        }
    }

    // =======================================================================
    // Depth buffer
    // =======================================================================
    void createDepthResources()
    {
        vk::ImageCreateInfo ici{
            .imageType   = vk::ImageType::e2D,
            .format      = depthFormat,
            .extent      = {swapChainExtent.width, swapChainExtent.height, 1},
            .mipLevels   = 1,
            .arrayLayers = 1,
            .samples     = vk::SampleCountFlagBits::e1,
            .tiling      = vk::ImageTiling::eOptimal,
            .usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment};
        depthImage = vk::raii::Image(device, ici);

        auto memReqs = depthImage.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = memReqs.size,
            .memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits,
                                              vk::MemoryPropertyFlagBits::eDeviceLocal)};
        depthImageMemory = vk::raii::DeviceMemory(device, mai);
        depthImage.bindMemory(depthImageMemory, 0);

        vk::ImageViewCreateInfo ivci{
            .image            = *depthImage,
            .viewType         = vk::ImageViewType::e2D,
            .format           = depthFormat,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}};
        depthImageView = vk::raii::ImageView(device, ivci);
    }

    // =======================================================================
    // Descriptor set layout (compute)
    //
    //   binding 0 – uniform buffer (ClothUBO)
    //   binding 1 – storage buffer (positions,     RW)
    //   binding 2 – storage buffer (prevPositions, RW)
    //   binding 3 – storage buffer (normals,       RW)
    // =======================================================================
    void createComputeDescriptorSetLayout()
    {
        std::array bindings{
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eUniformBuffer,  1,
                                           vk::ShaderStageFlagBits::eCompute, nullptr},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer,  1,
                                           vk::ShaderStageFlagBits::eCompute, nullptr},
            vk::DescriptorSetLayoutBinding{2, vk::DescriptorType::eStorageBuffer,  1,
                                           vk::ShaderStageFlagBits::eCompute, nullptr},
            vk::DescriptorSetLayoutBinding{3, vk::DescriptorType::eStorageBuffer,  1,
                                           vk::ShaderStageFlagBits::eCompute, nullptr}};
        computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device,
            {.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});
    }

    // =======================================================================
    // Shader helper
    // =======================================================================
    [[nodiscard]] vk::raii::ShaderModule createShaderModule(std::vector<char> const &code) const
    {
        return vk::raii::ShaderModule(device,
            {.codeSize = code.size(),
             .pCode    = reinterpret_cast<const uint32_t *>(code.data())});
    }

    // =======================================================================
    // Graphics pipelines
    // =======================================================================
    void createGraphicsPipelines()
    {
        // Push-constant range (ClothPush = 2×mat4 + vec4 = 140 bytes)
        vk::PushConstantRange pcr{
            .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            .offset     = 0,
            .size       = sizeof(ClothPush)};
        graphicsPipelineLayout = vk::raii::PipelineLayout(device,
            {.pushConstantRangeCount = 1, .pPushConstantRanges = &pcr});

        auto shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertStage{
            .stage  = vk::ShaderStageFlagBits::eVertex,
            .module = shaderModule,
            .pName  = "vertMain"};
        vk::PipelineShaderStageCreateInfo fragStage{
            .stage  = vk::ShaderStageFlagBits::eFragment,
            .module = shaderModule,
            .pName  = "fragMain"};
        vk::PipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

        // Vertex input:
        //   binding 0: cloth positions SSBO (float4 posInvM, stride 16)
        //   binding 1: cloth UV buffer      (float2 uv,       stride 8)
        //   binding 2: cloth normals SSBO   (float4 normal,   stride 16, read as float3)
        std::array bindings{
            vk::VertexInputBindingDescription{0, sizeof(glm::vec4),   vk::VertexInputRate::eVertex},
            vk::VertexInputBindingDescription{1, sizeof(ClothUV),     vk::VertexInputRate::eVertex},
            vk::VertexInputBindingDescription{2, sizeof(glm::vec4),   vk::VertexInputRate::eVertex}};
        std::array attribs{
            vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32A32Sfloat, 0},
            vk::VertexInputAttributeDescription{1, 1, vk::Format::eR32G32Sfloat,       0},
            vk::VertexInputAttributeDescription{2, 2, vk::Format::eR32G32B32A32Sfloat, 0}};
        vk::PipelineVertexInputStateCreateInfo vis{
            .vertexBindingDescriptionCount   = static_cast<uint32_t>(bindings.size()),
            .pVertexBindingDescriptions      = bindings.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attribs.size()),
            .pVertexAttributeDescriptions    = attribs.data()};

        vk::PipelineInputAssemblyStateCreateInfo ia{
            .topology               = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False};
        vk::PipelineViewportStateCreateInfo vps{.viewportCount = 1, .scissorCount = 1};
        vk::PipelineRasterizationStateCreateInfo rast{
            .depthClampEnable        = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eNone,
            .frontFace               = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable         = vk::True,
            .depthBiasConstantFactor = -1.0f,
            .depthBiasSlopeFactor    = -1.0f,
            .lineWidth               = 1.0f};
        vk::PipelineMultisampleStateCreateInfo ms{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable  = vk::False};
        vk::PipelineDepthStencilStateCreateInfo ds{
            .depthTestEnable       = vk::True,
            .depthWriteEnable      = vk::True,
            .depthCompareOp        = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable     = vk::False};
        vk::PipelineColorBlendAttachmentState cba{
            .blendEnable    = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
        vk::PipelineColorBlendStateCreateInfo cbs{
            .logicOpEnable   = vk::False,
            .attachmentCount = 1,
            .pAttachments    = &cba};
        std::vector dynStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynState{
            .dynamicStateCount = static_cast<uint32_t>(dynStates.size()),
            .pDynamicStates    = dynStates.data()};

        // Rendering create info (dynamic rendering, no render pass object)
        vk::PipelineRenderingCreateInfo prc{
            .colorAttachmentCount    = 1,
            .pColorAttachmentFormats = &swapChainSurfaceFormat.format,
            .depthAttachmentFormat   = depthFormat};

        vk::GraphicsPipelineCreateInfo gpci{
            .pNext               = &prc,
            .stageCount          = 2,
            .pStages             = stages,
            .pVertexInputState   = &vis,
            .pInputAssemblyState = &ia,
            .pViewportState      = &vps,
            .pRasterizationState = &rast,
            .pMultisampleState   = &ms,
            .pDepthStencilState  = &ds,
            .pColorBlendState    = &cbs,
            .pDynamicState       = &dynState,
            .layout              = graphicsPipelineLayout,
            .renderPass          = nullptr};

        clothPipeline = vk::raii::Pipeline(device, nullptr, gpci);

        // Sphere pipeline: same shader, but single binding (float3 pos, stride 12)
        vk::VertexInputBindingDescription   sphBinding{0, sizeof(SphereVertex), vk::VertexInputRate::eVertex};
        vk::VertexInputAttributeDescription sphAttrib{0, 0, vk::Format::eR32G32B32Sfloat, 0};
        vk::PipelineVertexInputStateCreateInfo sphVis{
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &sphBinding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions    = &sphAttrib};

        // Sphere vertex shader reads posInvM as float3 via location 0 (w defaults to 1.0)
        // fragMain will see invMass=1.0, so it uses cloth color – we tint via lightDir.w flag.
        // We need a sphere-dedicated vert stage that passes a float3.
        // Reuse the same entry points: sphere just outputs posInvM.xyz, uv=0.
        // The sphere gets its own pipeline with the same shaders but a float3 vertex format.
        // Because the shader reads location 0 as float4, we use eR32G32B32A32Sfloat is
        // wrong for a float3 buffer – use a push constant to indicate sphere mode and
        // fill the w in the vertex shader from the buffer's float3 only.
        // Simplest: keep float3 vertex, shader reads float4 at location 0 → driver
        // zero-extends the missing w component to 0.0.  That gives invMass=0 → pinned
        // tint on the sphere, undesirable.  Use two separate attribute layouts.
        // Alternative (simpler): pack sphere verts as float4 with w=1.
        // We do that in buildSphere upload below.
        vk::VertexInputBindingDescription   sph4Binding{0, sizeof(glm::vec4), vk::VertexInputRate::eVertex};
        vk::VertexInputAttributeDescription sph4Attrib{0, 0, vk::Format::eR32G32B32A32Sfloat, 0};
        vk::PipelineVertexInputStateCreateInfo sph4Vis{
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &sph4Binding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions    = &sph4Attrib};

        // Sphere uses only binding 0 (no UV binding needed); vin.uv comes from
        // location 1 which is unbound – Slang won't read it because the sphere
        // pipeline needs only pos.  However we still need to provide the uv
        // attribute or the validation layer complains.  Supply a zero-stride
        // dummy binding for location 1.
        vk::VertexInputBindingDescription   dummy1Binding{1, sizeof(glm::vec2), vk::VertexInputRate::eVertex};
        vk::VertexInputAttributeDescription dummy1Attrib{1, 1, vk::Format::eR32G32Sfloat, 0};
        // binding 2: real per-vertex sphere normals (float3, stride 12)
        vk::VertexInputBindingDescription   sphNrmBinding{2, sizeof(glm::vec3), vk::VertexInputRate::eVertex};
        vk::VertexInputAttributeDescription sphNrmAttrib{2, 2, vk::Format::eR32G32B32Sfloat, 0};
        std::array sphAllBindings = {sph4Binding, dummy1Binding, sphNrmBinding};
        std::array sphAllAttribs  = {sph4Attrib,  dummy1Attrib,  sphNrmAttrib};
        vk::PipelineVertexInputStateCreateInfo sphFullVis{
            .vertexBindingDescriptionCount   = static_cast<uint32_t>(sphAllBindings.size()),
            .pVertexBindingDescriptions      = sphAllBindings.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(sphAllAttribs.size()),
            .pVertexAttributeDescriptions    = sphAllAttribs.data()};

        // Back-face culling for solid sphere
        vk::PipelineRasterizationStateCreateInfo sphRast{
            .depthClampEnable        = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eBack,
            .frontFace               = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable         = vk::False,
            .lineWidth               = 1.0f};

        gpci.pVertexInputState   = &sphFullVis;
        gpci.pRasterizationState = &sphRast;

        spherePipeline = vk::raii::Pipeline(device, nullptr, gpci);
    }

    // =======================================================================
    // Compute pipeline
    // =======================================================================
    void createComputePipeline()
    {
        auto shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo stage{
            .stage  = vk::ShaderStageFlagBits::eCompute,
            .module = shaderModule,
            .pName  = "constraintPass"};

        vk::PipelineLayoutCreateInfo pli{
            .setLayoutCount = 1,
            .pSetLayouts    = &*computeDescriptorSetLayout};
        computePipelineLayout = vk::raii::PipelineLayout(device, pli);

        computePipeline = vk::raii::Pipeline(device, nullptr,
            vk::ComputePipelineCreateInfo{.stage = stage, .layout = computePipelineLayout});
    }

    // =======================================================================
    // Command pools
    // =======================================================================
    void createCommandPools()
    {
        graphicsCommandPool = vk::raii::CommandPool(device,
            {.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
             .queueFamilyIndex = graphicsQueueFamily});
        asyncComputeCommandPool = vk::raii::CommandPool(device,
            {.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
             .queueFamilyIndex = asyncComputeQueueFamily});
    }

    // =======================================================================
    // Cloth buffers – two per frame (positions + prevPositions)
    // =======================================================================
    void createClothBuffers()
    {
        // Build initial cloth positions in world space.
        // The cloth lies HORIZONTALLY in the XZ plane at y = 1.4, spanning
        // X,Z in [-1, 1].  Its four corners are pinned (invMass = 0) like a
        // trampoline; gravity makes the middle sag, and the sphere bobbing up
        // from below pushes a clear bulge through it — an unambiguous drape.
        struct Vertex4 { glm::vec4 posInvM; };
        constexpr float CLOTH_Y = 1.4f;
        std::vector<Vertex4> initPos(CLOTH_N);
        for (uint32_t r = 0; r < CLOTH_H; ++r)
        {
            for (uint32_t c = 0; c < CLOTH_W; ++c)
            {
                uint32_t idx  = r * CLOTH_W + c;
                float    x    = -1.0f + 2.0f * float(c) / float(CLOTH_W - 1);
                float    z    = -1.0f + 2.0f * float(r) / float(CLOTH_H - 1);
                bool     cnrX = (c == 0 || c == CLOTH_W - 1);
                bool     cnrZ = (r == 0 || r == CLOTH_H - 1);
                float    invM = (cnrX && cnrZ) ? 0.0f : 1.0f;   // pin 4 corners
                initPos[idx].posInvM = glm::vec4(x, CLOTH_Y, z, invM);
            }
        }

        vk::DeviceSize bufSz = sizeof(Vertex4) * CLOTH_N;

        // Initial normals: flat horizontal cloth faces +Y.
        std::vector<glm::vec4> initNrm(CLOTH_N, glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

        clothPositionBuffers.clear();
        clothPositionMemory.clear();
        clothPrevBuffers.clear();
        clothPrevMemory.clear();
        clothNormalBuffers.clear();
        clothNormalMemory.clear();

        // Staging buffer for upload
        vk::raii::Buffer       staging({});
        vk::raii::DeviceMemory stagingMem({});
        createBuffer(bufSz,
                     vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     staging, stagingMem);
        void *dst = stagingMem.mapMemory(0, bufSz);
        memcpy(dst, initPos.data(), static_cast<size_t>(bufSz));
        stagingMem.unmapMemory();

        // Staging buffer for initial normals
        vk::raii::Buffer       nrmStaging({});
        vk::raii::DeviceMemory nrmStagingMem({});
        createBuffer(bufSz,
                     vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     nrmStaging, nrmStagingMem);
        void *ndst = nrmStagingMem.mapMemory(0, bufSz);
        memcpy(ndst, initNrm.data(), static_cast<size_t>(bufSz));
        nrmStagingMem.unmapMemory();

        for (int i = 0; i < MAX_FRAMES; ++i)
        {
            vk::raii::Buffer       buf({});
            vk::raii::DeviceMemory mem({});
            createBuffer(bufSz,
                         vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eVertexBuffer  |
                         vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         buf, mem);
            copyBuffer(staging, buf, bufSz);
            clothPositionBuffers.emplace_back(std::move(buf));
            clothPositionMemory.emplace_back(std::move(mem));

            vk::raii::Buffer       pbuf({});
            vk::raii::DeviceMemory pmem({});
            createBuffer(bufSz,
                         vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         pbuf, pmem);
            copyBuffer(staging, pbuf, bufSz);
            clothPrevBuffers.emplace_back(std::move(pbuf));
            clothPrevMemory.emplace_back(std::move(pmem));

            vk::raii::Buffer       nbuf({});
            vk::raii::DeviceMemory nmem({});
            createBuffer(bufSz,
                         vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eVertexBuffer  |
                         vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         nbuf, nmem);
            copyBuffer(nrmStaging, nbuf, bufSz);
            clothNormalBuffers.emplace_back(std::move(nbuf));
            clothNormalMemory.emplace_back(std::move(nmem));
        }

        // UV buffer (static)
        std::vector<ClothUV> uvs(CLOTH_N);
        for (uint32_t r = 0; r < CLOTH_H; ++r)
            for (uint32_t c = 0; c < CLOTH_W; ++c)
                uvs[r * CLOTH_W + c].uv = {float(c) / float(CLOTH_W - 1),
                                            float(r) / float(CLOTH_H - 1)};
        vk::DeviceSize uvSz = sizeof(ClothUV) * CLOTH_N;
        createBuffer(uvSz,
                     vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     clothUVBuffer, clothUVMemory);
        {
            vk::raii::Buffer       uvStage({});
            vk::raii::DeviceMemory uvStageMem({});
            createBuffer(uvSz, vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         uvStage, uvStageMem);
            void *p = uvStageMem.mapMemory(0, uvSz);
            memcpy(p, uvs.data(), static_cast<size_t>(uvSz));
            uvStageMem.unmapMemory();
            copyBuffer(uvStage, clothUVBuffer, uvSz);
        }

        // Index buffer for cloth triangle list
        std::vector<uint32_t> indices;
        indices.reserve(CLOTH_INDICES);
        for (uint32_t r = 0; r < CLOTH_H - 1; ++r)
        {
            for (uint32_t c = 0; c < CLOTH_W - 1; ++c)
            {
                uint32_t a = r * CLOTH_W + c;
                uint32_t b = a + 1;
                uint32_t d = a + CLOTH_W;
                uint32_t e = d + 1;
                indices.push_back(a); indices.push_back(d); indices.push_back(b);
                indices.push_back(b); indices.push_back(d); indices.push_back(e);
            }
        }
        vk::DeviceSize idxSz = sizeof(uint32_t) * indices.size();
        createBuffer(idxSz,
                     vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     clothIndexBuffer, clothIndexMemory);
        {
            vk::raii::Buffer       idxStage({});
            vk::raii::DeviceMemory idxStageMem({});
            createBuffer(idxSz, vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         idxStage, idxStageMem);
            void *p = idxStageMem.mapMemory(0, idxSz);
            memcpy(p, indices.data(), static_cast<size_t>(idxSz));
            idxStageMem.unmapMemory();
            copyBuffer(idxStage, clothIndexBuffer, idxSz);
        }
    }

    // =======================================================================
    // Sphere buffers
    // =======================================================================
    void createSphereBuffers()
    {
        std::vector<SphereVertex> verts;
        std::vector<uint32_t>     inds;
        const glm::vec3           sphCentre{0.0f, SPHERE_BASE_Y, 0.0f};
        buildSphere(SPHERE_RADIUS, sphCentre, verts, inds);
        sphereIndexCount = static_cast<uint32_t>(inds.size());

        // Pack as float4 so the vertex shader location 0 (float4 posInvM) works.
        // w = 1.0 so invMass != 0 → no pinned tint.
        std::vector<glm::vec4> sphVerts4;
        std::vector<glm::vec3> sphNrm;
        sphVerts4.reserve(verts.size());
        sphNrm.reserve(verts.size());
        for (auto &v : verts)
        {
            sphVerts4.push_back({v.position, 1.0f});
            sphNrm.push_back(glm::normalize(v.position - sphCentre));  // outward normal
        }

        // Dummy UV buffer for binding 1 (one vec2 = 8 bytes, stride doesn't matter
        // as no UV is actually needed for the sphere; we use the same frag shader)
        // We upload actual UVs so the hardware doesn't read garbage.
        // For simplicity, share the clothUVBuffer for binding 1 and read nothing.
        // The sphere fragment will use the same checker logic but with uv=(0,0).
        // We supply a zero-data UV buffer the same size as the sphere vertex buffer.

        vk::DeviceSize vSz = sizeof(glm::vec4) * sphVerts4.size();
        vk::DeviceSize iSz = sizeof(uint32_t)  * inds.size();

        createBuffer(vSz,
                     vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     sphereVertexBuffer, sphereVertexMemory);
        {
            vk::raii::Buffer       st({});  vk::raii::DeviceMemory sm({});
            createBuffer(vSz, vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         st, sm);
            void *p = sm.mapMemory(0, vSz);
            memcpy(p, sphVerts4.data(), static_cast<size_t>(vSz));
            sm.unmapMemory();
            copyBuffer(st, sphereVertexBuffer, vSz);
        }

        createBuffer(iSz,
                     vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     sphereIndexBuffer, sphereIndexMemory);
        {
            vk::raii::Buffer       st({});  vk::raii::DeviceMemory sm({});
            createBuffer(iSz, vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         st, sm);
            void *p = sm.mapMemory(0, iSz);
            memcpy(p, inds.data(), static_cast<size_t>(iSz));
            sm.unmapMemory();
            copyBuffer(st, sphereIndexBuffer, iSz);
        }

        // Sphere normals (binding 2)
        vk::DeviceSize nSz = sizeof(glm::vec3) * sphNrm.size();
        createBuffer(nSz,
                     vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     sphereNormalBuffer, sphereNormalMemory);
        {
            vk::raii::Buffer       st({});  vk::raii::DeviceMemory sm({});
            createBuffer(nSz, vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         st, sm);
            void *p = sm.mapMemory(0, nSz);
            memcpy(p, sphNrm.data(), static_cast<size_t>(nSz));
            sm.unmapMemory();
            copyBuffer(st, sphereNormalBuffer, nSz);
        }
    }

    // =======================================================================
    // Uniform buffers (per frame)
    // =======================================================================
    void createUniformBuffers()
    {
        vk::DeviceSize sz = sizeof(ClothUBO);
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (int i = 0; i < MAX_FRAMES; ++i)
        {
            vk::raii::Buffer       buf({});
            vk::raii::DeviceMemory mem({});
            createBuffer(sz,
                         vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         buf, mem);
            uniformBuffers.emplace_back(std::move(buf));
            uniformBuffersMemory.emplace_back(std::move(mem));
            uniformBuffersMapped.push_back(uniformBuffersMemory[i].mapMemory(0, sz));
        }
    }

    // =======================================================================
    // Descriptor pool + sets
    // =======================================================================
    void createDescriptorPool()
    {
        std::array pool{
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(MAX_FRAMES)},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(MAX_FRAMES * 3)}};
        vk::DescriptorPoolCreateInfo dpci{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = static_cast<uint32_t>(MAX_FRAMES),
            .poolSizeCount = static_cast<uint32_t>(pool.size()),
            .pPoolSizes    = pool.data()};
        descriptorPool = vk::raii::DescriptorPool(device, dpci);
    }

    void createComputeDescriptorSets()
    {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES, *computeDescriptorSetLayout);
        vk::DescriptorSetAllocateInfo        dsai{
            .descriptorPool     = *descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES),
            .pSetLayouts        = layouts.data()};
        computeDescSets.clear();
        computeDescSets = device.allocateDescriptorSets(dsai);

        for (int i = 0; i < MAX_FRAMES; ++i)
        {
            vk::DescriptorBufferInfo uboInfo{uniformBuffers[i], 0, sizeof(ClothUBO)};
            vk::DeviceSize           posSize = sizeof(glm::vec4) * CLOTH_N;
            vk::DescriptorBufferInfo posInfo{clothPositionBuffers[i], 0, posSize};
            vk::DescriptorBufferInfo prevInfo{clothPrevBuffers[i],    0, posSize};
            vk::DescriptorBufferInfo nrmInfo{clothNormalBuffers[i],   0, posSize};

            std::array writes{
                vk::WriteDescriptorSet{
                    .dstSet          = *computeDescSets[i],
                    .dstBinding      = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo     = &uboInfo},
                vk::WriteDescriptorSet{
                    .dstSet          = *computeDescSets[i],
                    .dstBinding      = 1,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo     = &posInfo},
                vk::WriteDescriptorSet{
                    .dstSet          = *computeDescSets[i],
                    .dstBinding      = 2,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo     = &prevInfo},
                vk::WriteDescriptorSet{
                    .dstSet          = *computeDescSets[i],
                    .dstBinding      = 3,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo     = &nrmInfo}};
            device.updateDescriptorSets(writes, {});
        }
    }

    // =======================================================================
    // Buffer helpers
    // =======================================================================
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags props,
                      vk::raii::Buffer &buf, vk::raii::DeviceMemory &mem) const
    {
        // When graphics and async-compute live in different queue families, the
        // cloth buffers are written by one family and read by the other.  Rather
        // than emitting explicit queue-family ownership transfers (acquire/release
        // barriers) every frame, we declare the buffers as CONCURRENT across both
        // families.  For buffers this carries no practical cost on real drivers,
        // and it removes the ownership-transfer requirement entirely.  When the two
        // families are the same, exclusive mode is used.
        std::array<uint32_t, 2> families{graphicsQueueFamily, asyncComputeQueueFamily};
        bool concurrent = (graphicsQueueFamily != asyncComputeQueueFamily);

        vk::BufferCreateInfo bci{.size = size, .usage = usage};
        if (concurrent)
        {
            bci.sharingMode           = vk::SharingMode::eConcurrent;
            bci.queueFamilyIndexCount = static_cast<uint32_t>(families.size());
            bci.pQueueFamilyIndices   = families.data();
        }
        else
        {
            bci.sharingMode = vk::SharingMode::eExclusive;
        }

        buf = vk::raii::Buffer(device, bci);
        auto req = buf.getMemoryRequirements();
        mem = vk::raii::DeviceMemory(device,
            {.allocationSize  = req.size,
             .memoryTypeIndex = findMemoryType(req.memoryTypeBits, props)});
        buf.bindMemory(mem, 0);
    }

    [[nodiscard]] vk::raii::CommandBuffer beginSingleTimeCommands() const
    {
        vk::raii::CommandBuffer cb = std::move(
            vk::raii::CommandBuffers(device,
                {.commandPool        = *graphicsCommandPool,
                 .level              = vk::CommandBufferLevel::ePrimary,
                 .commandBufferCount = 1}).front());
        cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return cb;
    }

    void endSingleTimeCommands(vk::raii::CommandBuffer const &cb) const
    {
        cb.end();
        vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
        graphicsQueue.submit(si, nullptr);
        graphicsQueue.waitIdle();
    }

    void copyBuffer(vk::raii::Buffer const &src, vk::raii::Buffer const &dst, vk::DeviceSize sz) const
    {
        auto cb = beginSingleTimeCommands();
        cb.copyBuffer(src, dst, vk::BufferCopy{0, 0, sz});
        endSingleTimeCommands(cb);
    }

    [[nodiscard]] uint32_t findMemoryType(uint32_t filter, vk::MemoryPropertyFlags props) const
    {
        auto mp = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("No suitable memory type found.");
    }

    // =======================================================================
    // Command buffers
    // =======================================================================
    void createCommandBuffers()
    {
        commandBuffers.clear();
        commandBuffers = vk::raii::CommandBuffers(device,
            {.commandPool        = *graphicsCommandPool,
             .level              = vk::CommandBufferLevel::ePrimary,
             .commandBufferCount = static_cast<uint32_t>(MAX_FRAMES)});
    }

    void createComputeCommandBuffers()
    {
        computeCommandBuffers.clear();
        computeCommandBuffers = vk::raii::CommandBuffers(device,
            {.commandPool        = *asyncComputeCommandPool,
             .level              = vk::CommandBufferLevel::ePrimary,
             .commandBufferCount = static_cast<uint32_t>(MAX_FRAMES)});
    }

    // =======================================================================
    // Synchronisation objects
    // =======================================================================
    void createSyncObjects()
    {
        // Two timeline semaphores — each only signalled by one queue
        vk::SemaphoreTypeCreateInfo stci{.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
        computeTimeline  = vk::raii::Semaphore(device, {.pNext = &stci});
        graphicsTimeline = vk::raii::Semaphore(device, {.pNext = &stci});
        frameCount = 0;

        // Binary acquire semaphores (MAX_FRAMES+1 rolling pool)
        acquireSemaphores.clear();
        for (int i = 0; i < MAX_FRAMES + 1; ++i)
            acquireSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});

        // Per-swapchain-image render-done binary semaphores
        // (created after swapchain images are known)
        renderDoneSems.clear();
        for (size_t i = 0; i < swapChainImages.size(); ++i)
            renderDoneSems.emplace_back(device, vk::SemaphoreCreateInfo{});

        // Per-frame in-flight fences (guard command buffer reuse)
        inFlightFences.clear();
        for (int i = 0; i < MAX_FRAMES; ++i)
            inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});

        // Per-frame compute fences (guard compute command-buffer reuse independently of graphics)
        computeInFlightFences.clear();
        for (int i = 0; i < MAX_FRAMES; ++i)
            computeInFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }

    // =======================================================================
    // Per-frame uniform update
    // =======================================================================
    void updateUniformBuffer()
    {
        double now = glfwGetTime();
        float  dt  = static_cast<float>(now - lastWallTime);
        dt = std::clamp(dt, 0.001f, 0.033f);
        lastWallTime = now;

        float t = static_cast<float>(now - startTime);
        // Sphere bobs up through the cloth from below and back down.
        float sphereY = sphereCentreY(t);
        ClothUBO ubo{
            .deltaTime    = dt,
            .iterCount    = CONSTRAINT_ITER,
            .time         = t,
            .pad          = 0.0f,
            .sphereCenter = {0.0f, sphereY, 0.0f},
            .sphereRadius = SPHERE_RADIUS};
        memcpy(uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
    }

    // =======================================================================
    // Image layout transition helper (Synchronisation2)
    // =======================================================================
    void transitionImage(vk::raii::CommandBuffer const &cb,
                         vk::Image image,
                         vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                         vk::AccessFlags2 srcAccess, vk::AccessFlags2 dstAccess,
                         vk::PipelineStageFlags2 srcStage, vk::PipelineStageFlags2 dstStage,
                         vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor)
    {
        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask        = srcStage,
            .srcAccessMask       = srcAccess,
            .dstStageMask        = dstStage,
            .dstAccessMask       = dstAccess,
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = image,
            .subresourceRange    = {aspect, 0, 1, 0, 1}};
        cb.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    }

    // =======================================================================
    // Command recording: compute (cloth physics)
    // =======================================================================
    void recordComputeCommandBuffer()
    {
        auto &cb = computeCommandBuffers[frameIndex];
        cb.reset();
        cb.begin({});

        cb.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);
        // Write to the slot that graphics(N) will NOT be reading this frame.
        // Graphics(N) reads clothPositionBuffers[frameIndex]; compute(N) writes
        // clothPositionBuffers[(frameIndex+1)%MAX_FRAMES] for graphics(N+1) to read.
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                              computePipelineLayout, 0,
                              {computeDescSets[(frameIndex + 1) % MAX_FRAMES]}, {});

        // Dispatch a SINGLE workgroup: the shader declares [numthreads(CLOTH_N,1,1)]
        // so all 1024 cloth vertices are solved together with a real workgroup
        // barrier between constraint iterations.
        cb.dispatch(1, 1, 1);

        cb.end();
    }

    // =======================================================================
    // Command recording: graphics (render cloth + sphere)
    // =======================================================================
    void recordCommandBuffer(uint32_t imageIndex)
    {
        auto &cb = commandBuffers[frameIndex];
        cb.reset();
        cb.begin({});

        // Transition colour attachment: undefined → color attachment write
        transitionImage(cb,
            swapChainImages[imageIndex],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        // Transition depth: undefined → depth attachment
        transitionImage(cb,
            *depthImage,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests,
            vk::ImageAspectFlagBits::eDepth);

        vk::ClearValue              clearColor{vk::ClearColorValue(0.05f, 0.05f, 0.12f, 1.0f)};
        vk::ClearValue              clearDepth{vk::ClearDepthStencilValue{1.0f, 0}};
        vk::RenderingAttachmentInfo colorAtt{
            .imageView   = *swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = clearColor};
        vk::RenderingAttachmentInfo depthAtt{
            .imageView   = *depthImageView,
            .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eDontCare,
            .clearValue  = clearDepth};
        cb.beginRendering({
            .renderArea           = {{0, 0}, swapChainExtent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAtt,
            .pDepthAttachment     = &depthAtt});

        cb.setViewport(0, vk::Viewport{
            0.0f, 0.0f,
            static_cast<float>(swapChainExtent.width),
            static_cast<float>(swapChainExtent.height),
            0.0f, 1.0f});
        cb.setScissor(0, vk::Rect2D{{0, 0}, swapChainExtent});

        // Build MVP: slow orbit camera gives a long face-on view before going edge-on.
        float aspect = float(swapChainExtent.width) / float(swapChainExtent.height);
        float t      = static_cast<float>(glfwGetTime() - startTime);

        // Slow orbit, elevated, looking down at the horizontal cloth so the
        // bulge created by the sphere is clearly visible.
        float camX   = 3.4f * std::sin(t * 0.12f);
        float camZ   = 3.4f * std::cos(t * 0.12f);
        glm::mat4 view = glm::lookAt(
            glm::vec3(camX, 3.2f, camZ),
            glm::vec3(0.0f, 0.8f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 20.0f);
        proj[1][1]    *= -1.0f;  // flip Y for Vulkan NDC

        ClothPush push{
            .mvp          = proj * view,
            .normalMatrix = glm::transpose(glm::inverse(view)),
            .lightDir     = glm::vec4(glm::normalize(glm::vec3(1.0f, 2.0f, 1.5f)), 0.0f)};

        // ---------------------------------------------------------------
        // Draw cloth
        // ---------------------------------------------------------------
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, clothPipeline);
        cb.pushConstants<ClothPush>(*graphicsPipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, push);
        std::array clothVBs  = {*clothPositionBuffers[frameIndex], *clothUVBuffer,
                                *clothNormalBuffers[frameIndex]};
        std::array clothOffs = {vk::DeviceSize{0}, vk::DeviceSize{0}, vk::DeviceSize{0}};
        cb.bindVertexBuffers(0, clothVBs, clothOffs);
        cb.bindIndexBuffer(clothIndexBuffer, 0, vk::IndexType::eUint32);
        cb.drawIndexed(CLOTH_INDICES, 1, 0, 0, 0);

        // ---------------------------------------------------------------
        // Draw sphere: translate MVP to match the animated sphere position.
        // The vertex buffer is baked with centre at (0, -0.2, 0); we shift
        // it to the current sphereY so the rendered sphere matches physics.
        // ---------------------------------------------------------------
        float sphereY    = sphereCentreY(t);
        glm::mat4 sphModel = glm::translate(glm::mat4(1.0f),
                                            glm::vec3(0.0f, sphereY - SPHERE_BASE_Y, 0.0f));
        ClothPush sphPush{
            .mvp          = proj * view * sphModel,
            .normalMatrix = glm::transpose(glm::inverse(view)),
            .lightDir     = glm::vec4(glm::normalize(glm::vec3(1.0f, 2.0f, 1.5f)), 1.0f)};
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, spherePipeline);
        cb.pushConstants<ClothPush>(*graphicsPipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, sphPush);
        // binding 0: sphere verts (float4), binding 1: clothUVBuffer dummy,
        // binding 2: sphere normals (float3)
        std::array sphVBs  = {*sphereVertexBuffer, *clothUVBuffer, *sphereNormalBuffer};
        std::array sphOffs = {vk::DeviceSize{0}, vk::DeviceSize{0}, vk::DeviceSize{0}};
        cb.bindVertexBuffers(0, sphVBs, sphOffs);
        cb.bindIndexBuffer(sphereIndexBuffer, 0, vk::IndexType::eUint32);
        cb.drawIndexed(sphereIndexCount, 1, 0, 0, 0);

        cb.endRendering();

        // Transition colour to present
        transitionImage(cb,
            swapChainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe);

        cb.end();
    }

    // =======================================================================
    // Draw frame
    //
    // Timeline semaphore sequence (frame N):
    //   compute[N] : wait  graphicsTimeline >= N  → solve cloth → signal computeTimeline  = N+1
    //   graphics[N]: wait  computeTimeline  >= N  → render      → signal graphicsTimeline = N+1
    // Both waits are >= 0 on frame 0 (initial value), so the first frame starts immediately.
    // Each semaphore is only ever signalled by one queue, so no out-of-order signal is possible.
    // CPU: inFlightFences[fi] guards graphics CB reuse; computeInFlightFences[fi] guards compute CB.
    // =======================================================================
    void drawFrame()
    {
        // Wait for the in-flight fence so we don't reuse command buffers that
        // are still executing from two frames ago.
        auto fenceWait = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
        if (fenceWait != vk::Result::eSuccess)
            throw std::runtime_error("waitForFences failed");
        device.resetFences(*inFlightFences[frameIndex]);

        // Acquire swapchain image using rotating binary semaphore pool
        auto &acqSem = acquireSemaphores[acquireSemIdx];
        acquireSemIdx = (acquireSemIdx + 1) % (MAX_FRAMES + 1);

        auto [acqResult, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *acqSem, nullptr);
        if (acqResult == vk::Result::eErrorOutOfDateKHR)
        {
            device.waitIdle();
            recreateSwapChain();
            return;
        }
        if (acqResult != vk::Result::eSuccess && acqResult != vk::Result::eSuboptimalKHR)
            throw std::runtime_error("acquireNextImage failed");

        // compute[N] waits graphicsTimeline >= N (graphics[N-1] released the write slot)
        // graphics[N] waits computeTimeline  >= N (compute[N-1] finished writing the read slot)
        // Both timelines start at 0, so N=0 imposes no GPU-side wait on either queue.
        uint64_t N          = frameCount;
        uint64_t cWaitVal   = N;        // compute waits graphicsTimeline >= N
        uint64_t cSignalVal = N + 1;    // compute signals computeTimeline  = N+1
        uint64_t gWaitVal   = N;        // graphics waits computeTimeline  >= N
        uint64_t gSignalVal = N + 1;    // graphics signals graphicsTimeline = N+1

        updateUniformBuffer();

        // ------------------------------------------------------------------
        // ASYNC COMPUTE SUBMIT
        // Wait for the previous use of computeCommandBuffers[frameIndex] before
        // resetting it.  The graphics inFlightFence only guards the graphics CB;
        // the async compute CB needs its own fence because the two queues run
        // independently and compute may outlive the graphics submission it
        // accompanied two frames ago.
        // ------------------------------------------------------------------
        {
            auto computeFenceWait = device.waitForFences(*computeInFlightFences[frameIndex], vk::True, UINT64_MAX);
            if (computeFenceWait != vk::Result::eSuccess)
                throw std::runtime_error("waitForFences (compute) failed");
            device.resetFences(*computeInFlightFences[frameIndex]);

            recordComputeCommandBuffer();

            // Wait graphicsTimeline >= N (graphics[N-1] released the write buffer slot).
            // Signal computeTimeline = N+1 so graphics[N] can proceed.
            vk::TimelineSemaphoreSubmitInfo cTssi{
                .waitSemaphoreValueCount   = 1,
                .pWaitSemaphoreValues      = &cWaitVal,
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues    = &cSignalVal};
            vk::PipelineStageFlags cWaitStage = vk::PipelineStageFlagBits::eComputeShader;
            vk::SubmitInfo cSi{
                .pNext                = &cTssi,
                .waitSemaphoreCount   = 1,
                .pWaitSemaphores      = &*graphicsTimeline,
                .pWaitDstStageMask    = &cWaitStage,
                .commandBufferCount   = 1,
                .pCommandBuffers      = &*computeCommandBuffers[frameIndex],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores    = &*computeTimeline};
            asyncComputeQueue.submit(cSi, *computeInFlightFences[frameIndex]);
        }

        // ------------------------------------------------------------------
        // GRAPHICS SUBMIT
        // Wait at eVertexInput for computeTimeline >= N (compute[N-1] finished
        // writing clothPositionBuffers[frameIndex]).
        // Wait at eColorAttachmentOutput for swapchain image acquisition.
        // Signal computeTimeline is NOT touched here; only graphicsTimeline = N+1.
        // ------------------------------------------------------------------
        {
            recordCommandBuffer(imageIndex);

            // Two wait semaphores:
            //   [0] computeTimeline >= N at eVertexInput
            //   [1] binary acquire semaphore at eColorAttachmentOutput
            std::array<vk::Semaphore,          2> gWaitSems   = {*computeTimeline, *acqSem};
            std::array<uint64_t,               2> gWaitVals   = {gWaitVal, 0};
            std::array<vk::PipelineStageFlags, 2> gWaitStages = {
                vk::PipelineStageFlagBits::eVertexInput,
                vk::PipelineStageFlagBits::eColorAttachmentOutput};

            // Two signal semaphores:
            //   [0] graphicsTimeline = N+1
            //   [1] binary renderDone for this swapchain image
            std::array<vk::Semaphore, 2> gSignalSems = {*graphicsTimeline, *renderDoneSems[imageIndex]};
            std::array<uint64_t,      2> gSignalVals = {gSignalVal, 0};

            vk::TimelineSemaphoreSubmitInfo gTssi{
                .waitSemaphoreValueCount   = 2,
                .pWaitSemaphoreValues      = gWaitVals.data(),
                .signalSemaphoreValueCount = 2,
                .pSignalSemaphoreValues    = gSignalVals.data()};
            vk::SubmitInfo gSi{
                .pNext                = &gTssi,
                .waitSemaphoreCount   = 2,
                .pWaitSemaphores      = gWaitSems.data(),
                .pWaitDstStageMask    = gWaitStages.data(),
                .commandBufferCount   = 1,
                .pCommandBuffers      = &*commandBuffers[frameIndex],
                .signalSemaphoreCount = 2,
                .pSignalSemaphores    = gSignalSems.data()};
            graphicsQueue.submit(gSi, *inFlightFences[frameIndex]);
        }

        // ------------------------------------------------------------------
        // Present.  The binary renderDoneSems[imageIndex] is signaled by the
        // graphics submit above; present waits on it.  The inFlightFence
        // (waited at the top of drawFrame) guards per-frame resource reuse,
        // so no additional CPU timeline wait is needed here.
        // ------------------------------------------------------------------
        {
            vk::PresentInfoKHR pi{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores    = &*renderDoneSems[imageIndex],
                .swapchainCount     = 1,
                .pSwapchains        = &*swapChain,
                .pImageIndices      = &imageIndex};
            auto result = graphicsQueue.presentKHR(pi);
            if ((result == vk::Result::eSuboptimalKHR) ||
                (result == vk::Result::eErrorOutOfDateKHR) ||
                framebufferResized)
            {
                framebufferResized = false;
                device.waitIdle();
                recreateSwapChain();
                // Re-create per-swapchain-image render-done semaphores
                renderDoneSems.clear();
                for (size_t i = 0; i < swapChainImages.size(); ++i)
                    renderDoneSems.emplace_back(device, vk::SemaphoreCreateInfo{});
            }
            else
            {
                assert(result == vk::Result::eSuccess);
            }
        }

        ++frameCount;
        frameIndex = (frameIndex + 1) % MAX_FRAMES;
    }

    // =======================================================================
    // Swap-chain choice helpers
    // =======================================================================
    static uint32_t chooseMinImageCount(vk::SurfaceCapabilitiesKHR const &c)
    {
        uint32_t n = std::max(3u, c.minImageCount);
        if (c.maxImageCount > 0 && c.maxImageCount < n) n = c.maxImageCount;
        return n;
    }

    static vk::SurfaceFormatKHR chooseFormat(std::vector<vk::SurfaceFormatKHR> const &formats)
    {
        assert(!formats.empty());
        auto it = std::ranges::find_if(formats, [](auto const &f) {
            return f.format == vk::Format::eB8G8R8A8Srgb &&
                   f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });
        return it != formats.end() ? *it : formats[0];
    }

    static vk::PresentModeKHR choosePresent(std::vector<vk::PresentModeKHR> const &modes)
    {
        return std::ranges::any_of(modes, [](auto m) { return m == vk::PresentModeKHR::eMailbox; })
                   ? vk::PresentModeKHR::eMailbox
                   : vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseExtent(vk::SurfaceCapabilitiesKHR const &c)
    {
        if (c.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return c.currentExtent;
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        return {std::clamp<uint32_t>(w, c.minImageExtent.width,  c.maxImageExtent.width),
                std::clamp<uint32_t>(h, c.minImageExtent.height, c.maxImageExtent.height)};
    }

    // =======================================================================
    // Misc helpers
    // =======================================================================
    [[nodiscard]] std::vector<const char *> getRequiredInstanceExtensions()
    {
        uint32_t n     = 0;
        auto     exts  = glfwGetRequiredInstanceExtensions(&n);
        std::vector<const char *> result(exts, exts + n);
        if (kEnableValidation)
            result.push_back(vk::EXTDebugUtilsExtensionName);
        return result;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT        type,
        const vk::DebugUtilsMessengerCallbackDataEXT *data,
        void *)
    {
        if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
            severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
            std::cerr << "[VL] " << vk::to_string(type) << ": " << data->pMessage << "\n";
        return vk::False;
    }

    static std::vector<char> readFile(std::string const &filename)
    {
        std::ifstream f(filename, std::ios::ate | std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("failed to open: " + filename);
        std::vector<char> buf(f.tellg());
        f.seekg(0);
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        return buf;
    }
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
#ifndef ANDROID_BUILD
int main()
{
    try
    {
        AsyncComputeApplication app;
        app.run();
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
#endif // ANDROID_BUILD

#ifdef ANDROID_BUILD
extern "C" void chapter08_run() {
    try { AsyncComputeApplication{}.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh08", "%s", e.what()); }
}
#endif
