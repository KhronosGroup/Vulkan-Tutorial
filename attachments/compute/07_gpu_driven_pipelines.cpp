// Chapter 7 – GPU-Driven Pipelines: LOD Asteroid Field
//
// Demonstrates:
//   • A compute cull pass that writes VkDrawIndexedIndirectCommand structs per LOD
//   • vkCmdDrawIndexedIndirectCount — GPU decides what to draw and how many
//   • LOD selection: compute selects high/mid/low detail mesh based on distance
//   • Frustum culling in compute — asteroids outside the view are never submitted
//   • 1024 procedurally-placed asteroids with orbit + self-rotation animation
//
// GPU-driven means: the GPU decides what to draw (frustum cull) and which
// level of detail to use (LOD selection).  The CPU submits one fixed dispatch
// call; the GPU fills the indirect draw buffers.

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
#include <memory>
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

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr uint32_t kWidth          = 1280;
constexpr uint32_t kHeight         = 720;
constexpr uint32_t kAsteroidCount  = 1024;
constexpr int      kMaxFrames      = 2;
constexpr int      kAcquireSemas   = kMaxFrames + 1;

// LOD distance thresholds (world-space units from camera)
constexpr float kLodDist0 = 18.0f;   // closer than this → LOD 0 (48-tri icosphere)
constexpr float kLodDist1 = 40.0f;   // closer than this → LOD 1 (20-tri icosahedron)
                                      // farther          → LOD 2 ( 8-tri octahedron)

const std::vector<const char *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// CPU-side data structures — must match Slang struct layouts exactly
// ---------------------------------------------------------------------------

// One asteroid instance.
struct Asteroid
{
    glm::vec3 position;    // world-space disk position (static base)
    float     size;
    glm::vec3 rotAxis;
    float     orbitSpeed;
    glm::vec3 color;
    float     rotSpeed;
};
static_assert(sizeof(Asteroid) == 48, "Asteroid struct size mismatch");

// Push constants for the cull compute pass.
struct CullPush
{
    glm::mat4 viewProj;
    glm::vec3 cameraPos;
    float     time;
    uint32_t  asteroidCount;
    float     lodDist0;
    float     lodDist1;
    float     pad;
};
static_assert(sizeof(CullPush) == 96, "CullPush struct size mismatch");

// Per-frame uniform buffer object read by the vertex shader.
struct FrameUBO
{
    glm::mat4 viewProj;
    glm::vec3 cameraPos;
    float     time;
};
static_assert(sizeof(FrameUBO) == 80, "FrameUBO struct size mismatch");

// Mesh vertex — position + normal.
struct MeshVertex
{
    glm::vec3 pos;
    glm::vec3 normal;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return {0, sizeof(MeshVertex), vk::VertexInputRate::eVertex};
    }
    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(MeshVertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(MeshVertex, normal)),
        };
    }
};

// ---------------------------------------------------------------------------
// Procedural mesh generation
// ---------------------------------------------------------------------------

// Octahedron (LOD 2) – 6 vertices, 8 triangles
static void buildOctahedron(std::vector<MeshVertex> &verts, std::vector<uint32_t> &inds)
{
    const float s = 1.0f;
    glm::vec3 pts[6] = {
        { 0, s, 0}, { s, 0, 0}, { 0, 0, s},
        {-s, 0, 0}, { 0, 0,-s}, { 0,-s, 0}
    };
    // 8 faces (CCW from outside)
    uint32_t faces[8][3] = {
        {0,1,2},{0,2,3},{0,3,4},{0,4,1},
        {5,2,1},{5,3,2},{5,4,3},{5,1,4}
    };
    verts.clear(); inds.clear();
    for (auto &f : faces)
    {
        glm::vec3 a = pts[f[0]], b = pts[f[1]], c = pts[f[2]];
        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        uint32_t base = static_cast<uint32_t>(verts.size());
        verts.push_back({a, n}); verts.push_back({b, n}); verts.push_back({c, n});
        inds.push_back(base); inds.push_back(base+1); inds.push_back(base+2);
    }
}

// Icosahedron base – 12 verts, 20 faces (LOD 1)
static void buildIcosahedron(std::vector<MeshVertex> &verts, std::vector<uint32_t> &inds)
{
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    glm::vec3 pts[12] = {
        glm::normalize(glm::vec3(-1, t, 0)), glm::normalize(glm::vec3(1, t, 0)),
        glm::normalize(glm::vec3(-1,-t, 0)), glm::normalize(glm::vec3(1,-t, 0)),
        glm::normalize(glm::vec3(0,-1, t)), glm::normalize(glm::vec3(0, 1, t)),
        glm::normalize(glm::vec3(0,-1,-t)), glm::normalize(glm::vec3(0, 1,-t)),
        glm::normalize(glm::vec3(t, 0,-1)), glm::normalize(glm::vec3(t, 0, 1)),
        glm::normalize(glm::vec3(-t,0,-1)), glm::normalize(glm::vec3(-t,0, 1))
    };
    uint32_t faces[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
    };
    verts.clear(); inds.clear();
    for (auto &f : faces)
    {
        glm::vec3 a = pts[f[0]], b = pts[f[1]], c = pts[f[2]];
        glm::vec3 n = glm::normalize(a + b + c);  // smooth normals for sphere
        uint32_t base = static_cast<uint32_t>(verts.size());
        verts.push_back({a, glm::normalize(a)});
        verts.push_back({b, glm::normalize(b)});
        verts.push_back({c, glm::normalize(c)});
        inds.push_back(base); inds.push_back(base+1); inds.push_back(base+2);
    }
}

// Subdivided icosahedron (LOD 0) – 1 subdivision → 80 tris, 2 subdivisions → 320 tris
// We do 1 subdivision for 80 triangles (240 verts flat-shaded) = 48 visible faces described
// as 48-poly in the chapter.  We'll actually do 48 triangles = 1 subdiv of 12 base faces
// but keep it simple: use icosahedron with per-vertex smooth normals + 1 subdiv pass.
static glm::vec3 midpoint(glm::vec3 a, glm::vec3 b) { return glm::normalize((a + b) * 0.5f); }

static void buildIcosphere(std::vector<MeshVertex> &verts, std::vector<uint32_t> &inds)
{
    // Start from icosahedron vertices
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<glm::vec3> pts = {
        glm::normalize(glm::vec3(-1, t, 0)), glm::normalize(glm::vec3(1, t, 0)),
        glm::normalize(glm::vec3(-1,-t, 0)), glm::normalize(glm::vec3(1,-t, 0)),
        glm::normalize(glm::vec3(0,-1, t)), glm::normalize(glm::vec3(0, 1, t)),
        glm::normalize(glm::vec3(0,-1,-t)), glm::normalize(glm::vec3(0, 1,-t)),
        glm::normalize(glm::vec3(t, 0,-1)), glm::normalize(glm::vec3(t, 0, 1)),
        glm::normalize(glm::vec3(-t,0,-1)), glm::normalize(glm::vec3(-t,0, 1))
    };
    std::vector<std::array<uint32_t,3>> faces = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
    };

    // 1 subdivision pass
    std::vector<std::array<uint32_t,3>> newFaces;
    for (auto &f : faces)
    {
        uint32_t a = f[0], b = f[1], c = f[2];
        uint32_t ab = static_cast<uint32_t>(pts.size()); pts.push_back(midpoint(pts[a], pts[b]));
        uint32_t bc = static_cast<uint32_t>(pts.size()); pts.push_back(midpoint(pts[b], pts[c]));
        uint32_t ca = static_cast<uint32_t>(pts.size()); pts.push_back(midpoint(pts[c], pts[a]));
        newFaces.push_back({a, ab, ca});
        newFaces.push_back({b, bc, ab});
        newFaces.push_back({c, ca, bc});
        newFaces.push_back({ab, bc, ca});
    }
    faces = std::move(newFaces);

    // Build flat vertex list (unique smooth normals = position on unit sphere)
    verts.clear(); inds.clear();
    for (auto &f : faces)
    {
        for (int i = 0; i < 3; ++i)
        {
            glm::vec3 p = pts[f[i]];
            uint32_t idx = static_cast<uint32_t>(verts.size());
            verts.push_back({p, glm::normalize(p)});
            inds.push_back(idx);
        }
    }
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------
class AsteroidFieldApp
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
    GLFWwindow                      *window    = nullptr;
    vk::raii::Context                context;
    vk::raii::Instance               instance       = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR             surface        = nullptr;
    vk::raii::PhysicalDevice         physicalDevice = nullptr;
    vk::raii::Device                 device         = nullptr;
    uint32_t                         queueIndex     = ~0u;
    vk::raii::Queue                  queue          = nullptr;

    vk::raii::SwapchainKHR           swapChain = nullptr;
    std::vector<vk::Image>           swapChainImages;
    vk::SurfaceFormatKHR             swapChainSurfaceFormat;
    vk::Extent2D                     swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    // Depth buffer (memory declared first so it is destroyed last by RAII)
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::Image        depthImage       = nullptr;
    vk::raii::ImageView    depthImageView   = nullptr;
    vk::Format             depthFormat      = vk::Format::eD32Sfloat;

    // ---- Compute pipeline (cull + LOD selection) ----
    vk::raii::DescriptorSetLayout        computeDescLayout = nullptr;
    vk::raii::PipelineLayout             computePipeLayout = nullptr;
    vk::raii::Pipeline                   computePipeline   = nullptr;
    vk::raii::DescriptorPool             computeDescPool   = nullptr;
    std::vector<vk::raii::DescriptorSet> computeDescSets;

    // ---- Graphics pipeline (Phong asteroid rendering) ----
    vk::raii::DescriptorSetLayout        graphicsDescLayout = nullptr;
    vk::raii::PipelineLayout             graphicsPipeLayout = nullptr;
    vk::raii::Pipeline                   graphicsPipeline   = nullptr;
    vk::raii::DescriptorPool             graphicsDescPool   = nullptr;
    std::vector<vk::raii::DescriptorSet> graphicsDescSets;

    // ---- Scene data (shared across frames, static) ----
    // Memory declared before buffer so RAII destroys buffer before freeing memory.
    vk::raii::DeviceMemory asteroidBufMemory = nullptr;
    vk::raii::Buffer       asteroidBuffer    = nullptr;

    // LOD 0 (icosphere, 80 tris)
    vk::raii::DeviceMemory lodVBM0   = nullptr;
    vk::raii::Buffer       lodVB0    = nullptr;
    vk::raii::DeviceMemory lodIBM0   = nullptr;
    vk::raii::Buffer       lodIB0    = nullptr;
    uint32_t               lodIdxCount0 = 0;

    // LOD 1 (icosahedron, 20 tris)
    vk::raii::DeviceMemory lodVBM1   = nullptr;
    vk::raii::Buffer       lodVB1    = nullptr;
    vk::raii::DeviceMemory lodIBM1   = nullptr;
    vk::raii::Buffer       lodIB1    = nullptr;
    uint32_t               lodIdxCount1 = 0;

    // LOD 2 (octahedron, 8 tris)
    vk::raii::DeviceMemory lodVBM2   = nullptr;
    vk::raii::Buffer       lodVB2    = nullptr;
    vk::raii::DeviceMemory lodIBM2   = nullptr;
    vk::raii::Buffer       lodIB2    = nullptr;
    uint32_t               lodIdxCount2 = 0;

    // ---- Per-frame GPU buffers ----
    // indirect draw buffers for each LOD (written by compute)
    std::vector<vk::raii::DeviceMemory> indirectMem0, indirectMem1, indirectMem2;
    std::vector<vk::raii::Buffer>       indirectBuf0, indirectBuf1, indirectBuf2;
    // per-LOD draw count (atomic counter, read by vkCmdDrawIndexedIndirectCount)
    std::vector<vk::raii::DeviceMemory> countMem0, countMem1, countMem2;
    std::vector<vk::raii::Buffer>       countBuf0, countBuf1, countBuf2;
    // per-frame UBO (viewProj + cameraPos + time)
    std::vector<vk::raii::DeviceMemory> uboMem;
    std::vector<vk::raii::Buffer>       uboBuf;
    std::vector<void *>                 uboMapped;

    vk::raii::CommandPool                commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::raii::CommandBuffer> computeCommandBuffers;

    // Synchronisation: timeline semaphore + acquire semaphore pool + per-image render-done binary semas
    vk::raii::Semaphore          timelineSema  = nullptr;
    uint64_t                     timelineValue = 0;
    std::vector<vk::raii::Semaphore> acquireSemas;
    std::vector<vk::raii::Semaphore> renderDoneSemas;   // indexed by swapchain image index
    uint32_t                     acquireSemaIdx = 0;
    std::vector<vk::raii::Fence> inFlightFences;        // guard per-frame resource reuse

    int      frameIndex        = 0;
    bool     framebufferResized = false;

    std::chrono::steady_clock::time_point startTime;

    // Camera orbit state – controlled by mouse drag and scroll
    float  camTheta   =  0.3f;   // horizontal angle (radians)
    float  camPhi     =  0.25f;  // vertical angle (radians), slight downward tilt
    float  camRadius  = 75.0f;   // distance from origin
    bool   camDragging = false;
    double lastMx = 0.0, lastMy = 0.0;

    std::vector<const char *> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName};

    // -----------------------------------------------------------------------
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);
        window = glfwCreateWindow(kWidth, kHeight,
            "LOD Asteroid Field  |  drag=orbit  scroll=zoom  R=reset  ESC=quit",
            nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetScrollCallback(window,      cbScroll);
        glfwSetMouseButtonCallback(window, cbMouseButton);
        glfwSetCursorPosCallback(window,   cbCursorPos);
        glfwSetKeyCallback(window,         cbKey);
    }

    static void framebufferResizeCallback(GLFWwindow *win, int, int)
    {
        static_cast<AsteroidFieldApp *>(glfwGetWindowUserPointer(win))->framebufferResized = true;
    }

    static void cbScroll(GLFWwindow *w, double, double dy)
    {
        auto *app    = static_cast<AsteroidFieldApp *>(glfwGetWindowUserPointer(w));
        float factor = (dy > 0.0) ? 0.9f : (1.0f / 0.9f);
        app->camRadius = std::clamp(app->camRadius * factor, 5.0f, 200.0f);
    }

    static void cbMouseButton(GLFWwindow *w, int button, int action, int)
    {
        auto *app = static_cast<AsteroidFieldApp *>(glfwGetWindowUserPointer(w));
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            app->camDragging = (action == GLFW_PRESS);
            glfwGetCursorPos(w, &app->lastMx, &app->lastMy);
        }
    }

    static void cbCursorPos(GLFWwindow *w, double mx, double my)
    {
        auto *app = static_cast<AsteroidFieldApp *>(glfwGetWindowUserPointer(w));
        if (app->camDragging)
        {
            float dx    = static_cast<float>(mx - app->lastMx) * 0.005f;
            float dy    = static_cast<float>(my - app->lastMy) * 0.005f;
            app->camTheta += dx;
            app->camPhi    = std::clamp(app->camPhi + dy, -1.4f, 1.4f);
        }
        app->lastMx = mx;
        app->lastMy = my;
    }

    static void cbKey(GLFWwindow *w, int key, int, int action, int)
    {
        if (action != GLFW_PRESS) return;
        auto *app = static_cast<AsteroidFieldApp *>(glfwGetWindowUserPointer(w));
        switch (key)
        {
            case GLFW_KEY_R:
                app->camTheta  =  0.3f;
                app->camPhi    =  0.25f;
                app->camRadius = 75.0f;
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(w, GLFW_TRUE);
                break;
            default: break;
        }
    }

    void initVulkan()
    {
        startTime = std::chrono::steady_clock::now();
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createDepthResources();
        createComputeDescriptorSetLayout();
        createGraphicsDescriptorSetLayout();
        createGraphicsPipeline();
        createComputePipeline();
        createCommandPool();
        buildLodMeshes();
        createAsteroidBuffer();
        createPerFrameBuffers();
        createDescriptorPools();
        createComputeDescriptorSets();
        createGraphicsDescriptorSets();
        createCommandBuffers();
        createComputeCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }
        device.waitIdle();
    }

    void cleanupSwapChain()
    {
        depthImageView   = nullptr;
        depthImage       = nullptr;
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
        while (w == 0 || h == 0) { glfwGetFramebufferSize(window, &w, &h); glfwWaitEvents(); }
        device.waitIdle();
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createDepthResources();
    }

    // -----------------------------------------------------------------------
    // Instance / device setup (identical to other chapters)
    // -----------------------------------------------------------------------
    void createInstance()
    {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName   = "GPU-Driven Pipelines",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = vk::ApiVersion13};

        std::vector<const char *> layers;
        if (kEnableValidation)
            layers.assign(kValidationLayers.begin(), kValidationLayers.end());

        auto required = getRequiredInstanceExtensions();
        vk::InstanceCreateInfo ci{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames     = layers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(required.size()),
            .ppEnabledExtensionNames = required.data()};
        instance = vk::raii::Instance(context, ci);
    }

    void setupDebugMessenger()
    {
        if (!kEnableValidation) return;
        vk::DebugUtilsMessengerCreateInfoEXT ci{
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                               vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            .messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral    |
                               vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                               vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            .pfnUserCallback = debugCallback};
        debugMessenger = instance.createDebugUtilsMessengerEXT(ci);
    }

    void createSurface()
    {
        VkSurfaceKHR raw;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &raw) != VK_SUCCESS)
            throw std::runtime_error("failed to create window surface");
        surface = vk::raii::SurfaceKHR(instance, raw);
    }

    bool isDeviceSuitable(const vk::raii::PhysicalDevice &pd)
    {
        if (pd.getProperties().apiVersion < VK_API_VERSION_1_3)
            return false;

        auto qfps = pd.getQueueFamilyProperties();
        bool hasQueue = std::ranges::any_of(qfps, [](auto &q) {
            return (q.queueFlags & vk::QueueFlagBits::eGraphics) &&
                   (q.queueFlags & vk::QueueFlagBits::eCompute);
        });
        if (!hasQueue) return false;

        auto exts = pd.enumerateDeviceExtensionProperties();
        bool hasAllExts = std::ranges::all_of(requiredDeviceExtensions, [&](auto req) {
            return std::ranges::any_of(exts, [req](auto &e) { return strcmp(e.extensionName, req) == 0; });
        });
        if (!hasAllExts) return false;

        auto chain = pd.template getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features>();
        return chain.template get<vk::PhysicalDeviceVulkan12Features>().drawIndirectCount &&
               chain.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
               chain.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2;
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
        if (bestScore == 0) throw std::runtime_error("no suitable GPU found");
        std::cout << "[GPU] " << physicalDevice.getProperties().deviceName << "\n";
    }

    void createLogicalDevice()
    {
        auto qfps = physicalDevice.getQueueFamilyProperties();
        for (uint32_t i = 0; i < qfps.size(); ++i)
        {
            if ((qfps[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                (qfps[i].queueFlags & vk::QueueFlagBits::eCompute) &&
                physicalDevice.getSurfaceSupportKHR(i, *surface))
            { queueIndex = i; break; }
        }
        if (queueIndex == ~0u)
            throw std::runtime_error("no graphics+compute+present queue");

        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features> chain = {
            {.features = {.samplerAnisotropy = true}},
            {.shaderDrawParameters = true},
            {.drawIndirectCount   = true,
             .scalarBlockLayout   = true,
             .timelineSemaphore   = true,
             .bufferDeviceAddress = true},
            {.synchronization2 = true, .dynamicRendering = true}
        };

        float prio = 0.5f;
        vk::DeviceQueueCreateInfo qci{.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &prio};
        vk::DeviceCreateInfo dci{
            .pNext                   = &chain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &qci,
            .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtensions.size()),
            .ppEnabledExtensionNames = requiredDeviceExtensions.data()};
        device = vk::raii::Device(physicalDevice, dci);
        queue  = vk::raii::Queue(device, queueIndex, 0);
    }

    void createSwapChain()
    {
        auto caps     = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        swapChainExtent = chooseExtent(caps);
        auto formats  = physicalDevice.getSurfaceFormatsKHR(*surface);
        swapChainSurfaceFormat = chooseFormat(formats);
        auto pmodes   = physicalDevice.getSurfacePresentModesKHR(*surface);
        auto pmode    = choosePresentMode(pmodes);
        uint32_t cnt  = std::max(3u, caps.minImageCount);
        if (caps.maxImageCount > 0 && cnt > caps.maxImageCount) cnt = caps.maxImageCount;

        vk::SwapchainCreateInfoKHR ci{
            .surface          = *surface,
            .minImageCount    = cnt,
            .imageFormat      = swapChainSurfaceFormat.format,
            .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
            .imageExtent      = swapChainExtent,
            .imageArrayLayers = 1,
            .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform     = caps.currentTransform,
            .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode      = pmode,
            .clipped          = true};
        swapChain       = vk::raii::SwapchainKHR(device, ci);
        swapChainImages = swapChain.getImages();
    }

    void createImageViews()
    {
        assert(swapChainImageViews.empty());
        vk::ImageViewCreateInfo ci{
            .viewType         = vk::ImageViewType::e2D,
            .format           = swapChainSurfaceFormat.format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        for (auto &img : swapChainImages)
        {
            ci.image = img;
            swapChainImageViews.emplace_back(device, ci);
        }
    }

    void createDepthResources()
    {
        vk::ImageCreateInfo ci{
            .imageType   = vk::ImageType::e2D,
            .format      = depthFormat,
            .extent      = {swapChainExtent.width, swapChainExtent.height, 1},
            .mipLevels   = 1,
            .arrayLayers = 1,
            .samples     = vk::SampleCountFlagBits::e1,
            .tiling      = vk::ImageTiling::eOptimal,
            .usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined};
        depthImage = vk::raii::Image(device, ci);

        auto req   = depthImage.getMemoryRequirements();
        vk::MemoryAllocateInfo ai{
            .allocationSize  = req.size,
            .memoryTypeIndex = findMemoryType(req.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)};
        depthImageMemory = vk::raii::DeviceMemory(device, ai);
        depthImage.bindMemory(depthImageMemory, 0);

        vk::ImageViewCreateInfo vci{
            .image            = *depthImage,
            .viewType         = vk::ImageViewType::e2D,
            .format           = depthFormat,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}};
        depthImageView = vk::raii::ImageView(device, vci);
    }

    // -----------------------------------------------------------------------
    // Descriptor set layouts
    // -----------------------------------------------------------------------

    // Compute: set 0 — asteroids(0), drawCmds0-2(1-3), drawCount0-2(4-6)
    void createComputeDescriptorSetLayout()
    {
        std::array bindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
            vk::DescriptorSetLayoutBinding(6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr),
        };
        vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings    = bindings.data()};
        computeDescLayout = vk::raii::DescriptorSetLayout(device, ci);
    }

    // Graphics: set 0 — asteroids(0), frameUBO(1)
    void createGraphicsDescriptorSetLayout()
    {
        std::array bindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1,
                vk::ShaderStageFlagBits::eVertex, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, 1,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
        };
        vk::DescriptorSetLayoutCreateInfo ci{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings    = bindings.data()};
        graphicsDescLayout = vk::raii::DescriptorSetLayout(device, ci);
    }

    // -----------------------------------------------------------------------
    void createGraphicsPipeline()
    {
        auto spv = readFile("shaders/slang.spv");
        vk::raii::ShaderModule mod = createShaderModule(spv);

        vk::PipelineShaderStageCreateInfo stages[2] = {
            {.stage = vk::ShaderStageFlagBits::eVertex,   .module = mod, .pName = "vertMain"},
            {.stage = vk::ShaderStageFlagBits::eFragment, .module = mod, .pName = "fragMain"}};

        auto binding    = MeshVertex::getBindingDescription();
        auto attributes = MeshVertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInput{
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &binding,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
            .pVertexAttributeDescriptions    = attributes.data()};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology               = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = vk::False};

        vk::PipelineViewportStateCreateInfo vpState{.viewportCount = 1, .scissorCount = 1};

        vk::PipelineRasterizationStateCreateInfo raster{
            .depthClampEnable        = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eBack,
            .frontFace               = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable         = vk::False,
            .lineWidth               = 1.0f};

        vk::PipelineMultisampleStateCreateInfo ms{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable  = vk::False};

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable       = vk::True,
            .depthWriteEnable      = vk::True,
            .depthCompareOp        = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable     = vk::False};

        vk::PipelineColorBlendAttachmentState blendAttachment{
            .blendEnable    = vk::False,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
        vk::PipelineColorBlendStateCreateInfo blending{
            .logicOpEnable  = vk::False,
            .attachmentCount = 1,
            .pAttachments    = &blendAttachment};

        std::vector<vk::DynamicState> dynStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynState{
            .dynamicStateCount = static_cast<uint32_t>(dynStates.size()),
            .pDynamicStates    = dynStates.data()};

        // Pipeline layout: set 0 = computeDescLayout (declared so Set 1 is valid),
        //                  set 1 = graphicsDescLayout (gfxAsteroids + frameUBO)
        // The vertex/fragment shaders use [[vk::binding(X,1)]] so they reference set 1.
        // Set 0 is bound at draw time with the per-frame compute descriptor set so the
        // layout is fully satisfied without a separate dummy pool.
        std::array<vk::DescriptorSetLayout, 2> gfxLayouts{*computeDescLayout, *graphicsDescLayout};
        vk::PipelineLayoutCreateInfo plci{
            .setLayoutCount = static_cast<uint32_t>(gfxLayouts.size()),
            .pSetLayouts    = gfxLayouts.data()};
        graphicsPipeLayout = vk::raii::PipelineLayout(device, plci);

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain = {
            {.stageCount          = 2,
             .pStages             = stages,
             .pVertexInputState   = &vertexInput,
             .pInputAssemblyState = &inputAssembly,
             .pViewportState      = &vpState,
             .pRasterizationState = &raster,
             .pMultisampleState   = &ms,
             .pDepthStencilState  = &depthStencil,
             .pColorBlendState    = &blending,
             .pDynamicState       = &dynState,
             .layout              = graphicsPipeLayout,
             .renderPass          = nullptr},
            {.colorAttachmentCount    = 1,
             .pColorAttachmentFormats = &swapChainSurfaceFormat.format,
             .depthAttachmentFormat   = depthFormat}};

        graphicsPipeline = vk::raii::Pipeline(device, nullptr, chain.get<vk::GraphicsPipelineCreateInfo>());
    }

    void createComputePipeline()
    {
        auto spv = readFile("shaders/slang.spv");
        vk::raii::ShaderModule mod = createShaderModule(spv);
        vk::PipelineShaderStageCreateInfo stage{
            .stage  = vk::ShaderStageFlagBits::eCompute,
            .module = mod,
            .pName  = "cullMain"};

        vk::PushConstantRange pcr{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(CullPush)};

        vk::PipelineLayoutCreateInfo plci{
            .setLayoutCount         = 1,
            .pSetLayouts            = &*computeDescLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcr};
        computePipeLayout = vk::raii::PipelineLayout(device, plci);

        vk::ComputePipelineCreateInfo ci{.stage = stage, .layout = *computePipeLayout};
        computePipeline = vk::raii::Pipeline(device, nullptr, ci);
    }

    // -----------------------------------------------------------------------
    // Command pool
    // -----------------------------------------------------------------------
    void createCommandPool()
    {
        vk::CommandPoolCreateInfo ci{
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueIndex};
        commandPool = vk::raii::CommandPool(device, ci);
    }

    // -----------------------------------------------------------------------
    // Mesh buffers
    // -----------------------------------------------------------------------
    void uploadMesh(const std::vector<MeshVertex> &verts, const std::vector<uint32_t> &inds,
                    vk::raii::Buffer &vb, vk::raii::DeviceMemory &vm,
                    vk::raii::Buffer &ib, vk::raii::DeviceMemory &im)
    {
        // Vertex buffer
        vk::DeviceSize vsize = sizeof(MeshVertex) * verts.size();
        {
            vk::raii::Buffer       stg({});
            vk::raii::DeviceMemory stgm({});
            createBuffer(vsize, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                stg, stgm);
            void *p = stgm.mapMemory(0, vsize);
            memcpy(p, verts.data(), vsize);
            stgm.unmapMemory();
            createBuffer(vsize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal, vb, vm);
            copyBuffer(stg, vb, vsize);
        }
        // Index buffer
        vk::DeviceSize isize = sizeof(uint32_t) * inds.size();
        {
            vk::raii::Buffer       stg({});
            vk::raii::DeviceMemory stgm({});
            createBuffer(isize, vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                stg, stgm);
            void *p = stgm.mapMemory(0, isize);
            memcpy(p, inds.data(), isize);
            stgm.unmapMemory();
            createBuffer(isize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal, ib, im);
            copyBuffer(stg, ib, isize);
        }
    }

    void buildLodMeshes()
    {
        // LOD 0: subdivided icosphere (80 triangles)
        {
            std::vector<MeshVertex> verts; std::vector<uint32_t> inds;
            buildIcosphere(verts, inds);
            lodIdxCount0 = static_cast<uint32_t>(inds.size());
            uploadMesh(verts, inds, lodVB0, lodVBM0, lodIB0, lodIBM0);
        }
        // LOD 1: icosahedron (20 triangles)
        {
            std::vector<MeshVertex> verts; std::vector<uint32_t> inds;
            buildIcosahedron(verts, inds);
            lodIdxCount1 = static_cast<uint32_t>(inds.size());
            uploadMesh(verts, inds, lodVB1, lodVBM1, lodIB1, lodIBM1);
        }
        // LOD 2: octahedron (8 triangles)
        {
            std::vector<MeshVertex> verts; std::vector<uint32_t> inds;
            buildOctahedron(verts, inds);
            lodIdxCount2 = static_cast<uint32_t>(inds.size());
            uploadMesh(verts, inds, lodVB2, lodVBM2, lodIB2, lodIBM2);
        }
    }

    void createAsteroidBuffer()
    {
        std::default_random_engine rng(12345u);
        std::uniform_real_distribution<float> radDist(8.0f, 55.0f);
        std::uniform_real_distribution<float> angleDist(0.0f, 6.2832f);
        std::uniform_real_distribution<float> heightDist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> sizeDist(0.5f, 2.0f);
        std::uniform_real_distribution<float> speedDist(0.02f, 0.15f);
        std::uniform_real_distribution<float> rotDist(0.1f, 1.5f);
        std::uniform_real_distribution<float> axisDist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> greyDist(0.35f, 0.65f);
        std::uniform_real_distribution<float> tintDist(-0.08f, 0.08f);

        std::vector<Asteroid> asteroidData(kAsteroidCount);
        for (auto &a : asteroidData)
        {
            float r     = radDist(rng);
            float theta = angleDist(rng);
            a.position  = glm::vec3(r * std::cos(theta), heightDist(rng), r * std::sin(theta));
            a.size      = sizeDist(rng);
            glm::vec3 ax = glm::normalize(glm::vec3(axisDist(rng), axisDist(rng), axisDist(rng)));
            a.rotAxis   = ax;
            a.orbitSpeed = speedDist(rng);
            a.rotSpeed  = rotDist(rng);
            float grey  = greyDist(rng);
            a.color     = glm::vec3(grey + tintDist(rng), grey + tintDist(rng), grey + tintDist(rng));
        }

        vk::DeviceSize sz = sizeof(Asteroid) * kAsteroidCount;
        vk::raii::Buffer       stg({});
        vk::raii::DeviceMemory stgm({});
        createBuffer(sz, vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stg, stgm);
        void *p = stgm.mapMemory(0, sz);
        memcpy(p, asteroidData.data(), sz);
        stgm.unmapMemory();
        createBuffer(sz, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal, asteroidBuffer, asteroidBufMemory);
        copyBuffer(stg, asteroidBuffer, sz);
    }

    void createPerFrameBuffers()
    {
        const vk::DeviceSize indirectSz = sizeof(VkDrawIndexedIndirectCommand) * kAsteroidCount;
        const vk::DeviceSize countSz    = sizeof(uint32_t);
        const vk::DeviceSize uboSz      = sizeof(FrameUBO);

        for (int i = 0; i < kMaxFrames; ++i)
        {
            // Indirect draw buffers (LOD 0, 1, 2)
            auto makeIndirect = [&](auto &bufs, auto &mems) {
                vk::raii::Buffer b({}); vk::raii::DeviceMemory m({});
                createBuffer(indirectSz,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, b, m);
                bufs.emplace_back(std::move(b)); mems.emplace_back(std::move(m));
            };
            makeIndirect(indirectBuf0, indirectMem0);
            makeIndirect(indirectBuf1, indirectMem1);
            makeIndirect(indirectBuf2, indirectMem2);

            // Count buffers (LOD 0, 1, 2) — need eTransferDst to be zeroed each frame
            auto makeCount = [&](auto &bufs, auto &mems) {
                vk::raii::Buffer b({}); vk::raii::DeviceMemory m({});
                createBuffer(countSz,
                    vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eIndirectBuffer |
                    vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, b, m);
                bufs.emplace_back(std::move(b)); mems.emplace_back(std::move(m));
            };
            makeCount(countBuf0, countMem0);
            makeCount(countBuf1, countMem1);
            makeCount(countBuf2, countMem2);

            // UBO — host-visible, persistently mapped
            {
                vk::raii::Buffer b({}); vk::raii::DeviceMemory m({});
                createBuffer(uboSz, vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                    b, m);
                uboMapped.push_back(m.mapMemory(0, uboSz));
                uboBuf.emplace_back(std::move(b));
                uboMem.emplace_back(std::move(m));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Descriptor pools and sets
    // -----------------------------------------------------------------------
    void createDescriptorPools()
    {
        // Compute: 7 storage buffers × kMaxFrames sets
        {
            vk::DescriptorPoolSize ps(vk::DescriptorType::eStorageBuffer, 7 * kMaxFrames);
            vk::DescriptorPoolCreateInfo ci{
                .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                .maxSets       = static_cast<uint32_t>(kMaxFrames),
                .poolSizeCount = 1,
                .pPoolSizes    = &ps};
            computeDescPool = vk::raii::DescriptorPool(device, ci);
        }
        // Graphics: 1 storage + 1 uniform × kMaxFrames sets
        {
            std::array<vk::DescriptorPoolSize, 2> ps = {
                vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, kMaxFrames),
                vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, kMaxFrames)};
            vk::DescriptorPoolCreateInfo ci{
                .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                .maxSets       = static_cast<uint32_t>(kMaxFrames),
                .poolSizeCount = static_cast<uint32_t>(ps.size()),
                .pPoolSizes    = ps.data()};
            graphicsDescPool = vk::raii::DescriptorPool(device, ci);
        }
    }

    void createComputeDescriptorSets()
    {
        std::vector<vk::DescriptorSetLayout> layouts(kMaxFrames, *computeDescLayout);
        vk::DescriptorSetAllocateInfo ai{
            .descriptorPool     = *computeDescPool,
            .descriptorSetCount = static_cast<uint32_t>(kMaxFrames),
            .pSetLayouts        = layouts.data()};
        computeDescSets = device.allocateDescriptorSets(ai);

        for (int i = 0; i < kMaxFrames; ++i)
        {
            vk::DescriptorBufferInfo astInfo(asteroidBuffer,  0, sizeof(Asteroid) * kAsteroidCount);
            vk::DescriptorBufferInfo dc0Info(indirectBuf0[i], 0, sizeof(VkDrawIndexedIndirectCommand) * kAsteroidCount);
            vk::DescriptorBufferInfo dc1Info(indirectBuf1[i], 0, sizeof(VkDrawIndexedIndirectCommand) * kAsteroidCount);
            vk::DescriptorBufferInfo dc2Info(indirectBuf2[i], 0, sizeof(VkDrawIndexedIndirectCommand) * kAsteroidCount);
            vk::DescriptorBufferInfo cnt0Info(countBuf0[i],   0, sizeof(uint32_t));
            vk::DescriptorBufferInfo cnt1Info(countBuf1[i],   0, sizeof(uint32_t));
            vk::DescriptorBufferInfo cnt2Info(countBuf2[i],   0, sizeof(uint32_t));

            std::array writes{
                vk::WriteDescriptorSet{.dstSet=*computeDescSets[i],.dstBinding=0,.descriptorCount=1,.descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&astInfo},
                vk::WriteDescriptorSet{.dstSet=*computeDescSets[i],.dstBinding=1,.descriptorCount=1,.descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&dc0Info},
                vk::WriteDescriptorSet{.dstSet=*computeDescSets[i],.dstBinding=2,.descriptorCount=1,.descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&dc1Info},
                vk::WriteDescriptorSet{.dstSet=*computeDescSets[i],.dstBinding=3,.descriptorCount=1,.descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&dc2Info},
                vk::WriteDescriptorSet{.dstSet=*computeDescSets[i],.dstBinding=4,.descriptorCount=1,.descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&cnt0Info},
                vk::WriteDescriptorSet{.dstSet=*computeDescSets[i],.dstBinding=5,.descriptorCount=1,.descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&cnt1Info},
                vk::WriteDescriptorSet{.dstSet=*computeDescSets[i],.dstBinding=6,.descriptorCount=1,.descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&cnt2Info},
            };
            device.updateDescriptorSets(writes, {});
        }
    }

    void createGraphicsDescriptorSets()
    {
        std::vector<vk::DescriptorSetLayout> layouts(kMaxFrames, *graphicsDescLayout);
        vk::DescriptorSetAllocateInfo ai{
            .descriptorPool     = *graphicsDescPool,
            .descriptorSetCount = static_cast<uint32_t>(kMaxFrames),
            .pSetLayouts        = layouts.data()};
        graphicsDescSets = device.allocateDescriptorSets(ai);

        for (int i = 0; i < kMaxFrames; ++i)
        {
            vk::DescriptorBufferInfo astInfo(asteroidBuffer, 0, sizeof(Asteroid) * kAsteroidCount);
            vk::DescriptorBufferInfo uboInfo(uboBuf[i],      0, sizeof(FrameUBO));
            std::array writes{
                vk::WriteDescriptorSet{.dstSet=*graphicsDescSets[i],.dstBinding=0,.descriptorCount=1,.descriptorType=vk::DescriptorType::eStorageBuffer,.pBufferInfo=&astInfo},
                vk::WriteDescriptorSet{.dstSet=*graphicsDescSets[i],.dstBinding=1,.descriptorCount=1,.descriptorType=vk::DescriptorType::eUniformBuffer, .pBufferInfo=&uboInfo},
            };
            device.updateDescriptorSets(writes, {});
        }
    }

    // -----------------------------------------------------------------------
    // Command buffers
    // -----------------------------------------------------------------------
    void createCommandBuffers()
    {
        commandBuffers.clear();
        vk::CommandBufferAllocateInfo ai{
            .commandPool        = *commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(kMaxFrames)};
        commandBuffers = vk::raii::CommandBuffers(device, ai);
    }

    void createComputeCommandBuffers()
    {
        computeCommandBuffers.clear();
        vk::CommandBufferAllocateInfo ai{
            .commandPool        = *commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(kMaxFrames)};
        computeCommandBuffers = vk::raii::CommandBuffers(device, ai);
    }

    // -----------------------------------------------------------------------
    // Per-frame UBO update
    // -----------------------------------------------------------------------
    float elapsedSeconds() const
    {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<float>(now - startTime).count();
    }

    glm::mat4 computeViewProj(float /*t*/, glm::vec3 &camPosOut) const
    {
        float sinPhi = std::sin(camPhi), cosPhi = std::cos(camPhi);
        float sinThe = std::sin(camTheta), cosThe = std::cos(camTheta);
        camPosOut = camRadius * glm::vec3(cosPhi * sinThe, sinPhi, cosPhi * cosThe);
        glm::vec3 up = {0.0f, (cosPhi >= 0.0f ? 1.0f : -1.0f), 0.0f};
        glm::mat4 view   = glm::lookAt(camPosOut, glm::vec3(0, 0, 0), up);
        float     aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
        glm::mat4 proj   = glm::perspective(glm::radians(45.0f), aspect, 0.5f, 200.0f);
        proj[1][1] *= -1; // Vulkan Y flip
        return proj * view;
    }

    void updateUBO(float t)
    {
        glm::vec3 camPos;
        glm::mat4 vp = computeViewProj(t, camPos);
        FrameUBO ubo{.viewProj = vp, .cameraPos = camPos, .time = t};
        memcpy(uboMapped[frameIndex], &ubo, sizeof(ubo));
    }

    // -----------------------------------------------------------------------
    // Command recording
    // -----------------------------------------------------------------------
    void recordComputeCommandBuffer(float t)
    {
        auto &cb = computeCommandBuffers[frameIndex];
        cb.reset();
        cb.begin({});

        // Zero the three count buffers
        cb.fillBuffer(*countBuf0[frameIndex], 0, sizeof(uint32_t), 0u);
        cb.fillBuffer(*countBuf1[frameIndex], 0, sizeof(uint32_t), 0u);
        cb.fillBuffer(*countBuf2[frameIndex], 0, sizeof(uint32_t), 0u);

        // Barrier: fill writes must complete before compute reads counts
        std::array<vk::BufferMemoryBarrier2, 3> fillBarriers{{
            {.srcStageMask=vk::PipelineStageFlagBits2::eTransfer,.srcAccessMask=vk::AccessFlagBits2::eTransferWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eComputeShader,.dstAccessMask=vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eShaderWrite,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*countBuf0[frameIndex],.offset=0,.size=sizeof(uint32_t)},
            {.srcStageMask=vk::PipelineStageFlagBits2::eTransfer,.srcAccessMask=vk::AccessFlagBits2::eTransferWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eComputeShader,.dstAccessMask=vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eShaderWrite,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*countBuf1[frameIndex],.offset=0,.size=sizeof(uint32_t)},
            {.srcStageMask=vk::PipelineStageFlagBits2::eTransfer,.srcAccessMask=vk::AccessFlagBits2::eTransferWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eComputeShader,.dstAccessMask=vk::AccessFlagBits2::eShaderRead|vk::AccessFlagBits2::eShaderWrite,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*countBuf2[frameIndex],.offset=0,.size=sizeof(uint32_t)},
        }};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = static_cast<uint32_t>(fillBarriers.size()),
            .pBufferMemoryBarriers    = fillBarriers.data()});

        // Bind compute pipeline
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipeLayout, 0, {*computeDescSets[frameIndex]}, {});

        // Build push constants with current VP matrix and camera pos
        glm::vec3 camPos;
        CullPush push{
            .viewProj      = computeViewProj(t, camPos),
            .cameraPos     = camPos,
            .time          = t,
            .asteroidCount = kAsteroidCount,
            .lodDist0      = kLodDist0,
            .lodDist1      = kLodDist1,
            .pad           = 0.0f};
        cb.pushConstants<CullPush>(*computePipeLayout, vk::ShaderStageFlagBits::eCompute, 0, push);

        // Dispatch: 1024 asteroids / 256 threads per workgroup = 4 groups
        cb.dispatch(kAsteroidCount / 256, 1, 1);

        // Barrier: compute writes to indirect + count buffers must be visible to DrawIndirect stage
        std::array<vk::BufferMemoryBarrier2, 6> postBarriers{{
            // indirect buffers
            {.srcStageMask=vk::PipelineStageFlagBits2::eComputeShader,.srcAccessMask=vk::AccessFlagBits2::eShaderWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eDrawIndirect,.dstAccessMask=vk::AccessFlagBits2::eIndirectCommandRead,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*indirectBuf0[frameIndex],.offset=0,.size=sizeof(VkDrawIndexedIndirectCommand)*kAsteroidCount},
            {.srcStageMask=vk::PipelineStageFlagBits2::eComputeShader,.srcAccessMask=vk::AccessFlagBits2::eShaderWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eDrawIndirect,.dstAccessMask=vk::AccessFlagBits2::eIndirectCommandRead,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*indirectBuf1[frameIndex],.offset=0,.size=sizeof(VkDrawIndexedIndirectCommand)*kAsteroidCount},
            {.srcStageMask=vk::PipelineStageFlagBits2::eComputeShader,.srcAccessMask=vk::AccessFlagBits2::eShaderWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eDrawIndirect,.dstAccessMask=vk::AccessFlagBits2::eIndirectCommandRead,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*indirectBuf2[frameIndex],.offset=0,.size=sizeof(VkDrawIndexedIndirectCommand)*kAsteroidCount},
            // count buffers
            {.srcStageMask=vk::PipelineStageFlagBits2::eComputeShader,.srcAccessMask=vk::AccessFlagBits2::eShaderWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eDrawIndirect,.dstAccessMask=vk::AccessFlagBits2::eIndirectCommandRead,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*countBuf0[frameIndex],.offset=0,.size=sizeof(uint32_t)},
            {.srcStageMask=vk::PipelineStageFlagBits2::eComputeShader,.srcAccessMask=vk::AccessFlagBits2::eShaderWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eDrawIndirect,.dstAccessMask=vk::AccessFlagBits2::eIndirectCommandRead,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*countBuf1[frameIndex],.offset=0,.size=sizeof(uint32_t)},
            {.srcStageMask=vk::PipelineStageFlagBits2::eComputeShader,.srcAccessMask=vk::AccessFlagBits2::eShaderWrite,
             .dstStageMask=vk::PipelineStageFlagBits2::eDrawIndirect,.dstAccessMask=vk::AccessFlagBits2::eIndirectCommandRead,
             .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
             .buffer=*countBuf2[frameIndex],.offset=0,.size=sizeof(uint32_t)},
        }};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = static_cast<uint32_t>(postBarriers.size()),
            .pBufferMemoryBarriers    = postBarriers.data()});

        cb.end();
    }

    void recordCommandBuffer(uint32_t imageIndex)
    {
        auto &cb = commandBuffers[frameIndex];
        cb.reset();
        cb.begin({});

        // Transition color image
        transitionImage(cb, swapChainImages[imageIndex],
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
            {}, vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);

        // Transition depth image
        transitionImage(cb, *depthImage,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            {}, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eEarlyFragmentTests,
            vk::ImageAspectFlagBits::eDepth);

        // Begin dynamic rendering
        vk::ClearValue clearColor = vk::ClearColorValue(0.02f, 0.02f, 0.05f, 1.0f);
        vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

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
        vk::RenderingInfo ri{
            .renderArea           = {.offset={0,0}, .extent=swapChainExtent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAtt,
            .pDepthAttachment     = &depthAtt};

        cb.beginRendering(ri);
        cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        cb.setViewport(0, vk::Viewport(0.f, 0.f,
            static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height),
            0.f, 1.f));
        cb.setScissor(0, vk::Rect2D({0,0}, swapChainExtent));
        // Bind set 0 (compute layout) and set 1 (graphics layout)
        cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *graphicsPipeLayout, 0,
            {*computeDescSets[frameIndex], *graphicsDescSets[frameIndex]}, {});

        // LOD 0 draw (icosphere)
        cb.bindVertexBuffers(0, {*lodVB0}, {vk::DeviceSize(0)});
        cb.bindIndexBuffer(*lodIB0, 0, vk::IndexType::eUint32);
        cb.drawIndexedIndirectCount(
            *indirectBuf0[frameIndex], 0,
            *countBuf0[frameIndex],    0,
            kAsteroidCount, sizeof(VkDrawIndexedIndirectCommand));

        // LOD 1 draw (icosahedron)
        cb.bindVertexBuffers(0, {*lodVB1}, {vk::DeviceSize(0)});
        cb.bindIndexBuffer(*lodIB1, 0, vk::IndexType::eUint32);
        cb.drawIndexedIndirectCount(
            *indirectBuf1[frameIndex], 0,
            *countBuf1[frameIndex],    0,
            kAsteroidCount, sizeof(VkDrawIndexedIndirectCommand));

        // LOD 2 draw (octahedron)
        cb.bindVertexBuffers(0, {*lodVB2}, {vk::DeviceSize(0)});
        cb.bindIndexBuffer(*lodIB2, 0, vk::IndexType::eUint32);
        cb.drawIndexedIndirectCount(
            *indirectBuf2[frameIndex], 0,
            *countBuf2[frameIndex],    0,
            kAsteroidCount, sizeof(VkDrawIndexedIndirectCommand));

        cb.endRendering();

        // Transition color image to present
        transitionImage(cb, swapChainImages[imageIndex],
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite, {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eBottomOfPipe,
            vk::ImageAspectFlagBits::eColor);

        cb.end();
    }

    static void transitionImage(
        const vk::raii::CommandBuffer &cb,
        vk::Image                      image,
        vk::ImageLayout                oldLayout,
        vk::ImageLayout                newLayout,
        vk::AccessFlags2               srcAccess,
        vk::AccessFlags2               dstAccess,
        vk::PipelineStageFlags2        srcStage,
        vk::PipelineStageFlags2        dstStage,
        vk::ImageAspectFlags           aspect)
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
        cb.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount=1, .pImageMemoryBarriers=&barrier});
    }

    // -----------------------------------------------------------------------
    // Sync objects
    // -----------------------------------------------------------------------
    void createSyncObjects()
    {
        vk::SemaphoreTypeCreateInfo timelineType{
            .semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
        timelineSema  = vk::raii::Semaphore(device, {.pNext = &timelineType});
        timelineValue = 0;

        for (int i = 0; i < kAcquireSemas; ++i)
            acquireSemas.emplace_back(device, vk::SemaphoreCreateInfo{});

        // One renderDone binary semaphore per swapchain image
        for (size_t i = 0; i < swapChainImages.size(); ++i)
            renderDoneSemas.emplace_back(device, vk::SemaphoreCreateInfo{});

        // Per-frame in-flight fences (pre-signaled so frame 0 doesn't deadlock)
        inFlightFences.clear();
        for (int i = 0; i < kMaxFrames; ++i)
            inFlightFences.emplace_back(device,
                vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }

    // -----------------------------------------------------------------------
    // Draw frame
    // -----------------------------------------------------------------------
    void drawFrame()
    {
        // Wait for the in-flight fence so we don't reuse per-frame resources
        // (command buffers, descriptor sets, UBO) still in use by the GPU.
        if (device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX)
                != vk::Result::eSuccess)
            throw std::runtime_error("waitForFences failed");
        device.resetFences(*inFlightFences[frameIndex]);

        float t = elapsedSeconds();
        updateUBO(t);

        // Acquire next swapchain image using a rotating acquire semaphore
        auto &acquireSema = acquireSemas[acquireSemaIdx];
        acquireSemaIdx = (acquireSemaIdx + 1) % kAcquireSemas;

        auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *acquireSema, nullptr);
        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            recreateSwapChain();
            return;
        }

        auto &renderDone = renderDoneSemas[imageIndex];

        // Timeline values for this frame
        uint64_t computeWait    = timelineValue;
        uint64_t computeSignal  = ++timelineValue;
        uint64_t graphicsWait   = computeSignal;
        uint64_t graphicsSignal = ++timelineValue;

        // ---- Compute pass ----
        recordComputeCommandBuffer(t);
        {
            vk::TimelineSemaphoreSubmitInfo tsi{
                .waitSemaphoreValueCount   = 1,
                .pWaitSemaphoreValues      = &computeWait,
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues    = &computeSignal};
            vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
            vk::SubmitInfo si{
                .pNext                = &tsi,
                .waitSemaphoreCount   = 1,
                .pWaitSemaphores      = &*timelineSema,
                .pWaitDstStageMask    = &waitStage,
                .commandBufferCount   = 1,
                .pCommandBuffers      = &*computeCommandBuffers[frameIndex],
                .signalSemaphoreCount = 1,
                .pSignalSemaphores    = &*timelineSema};
            queue.submit(si, nullptr);
        }

        // ---- Graphics pass ----
        recordCommandBuffer(imageIndex);
        {
            // Wait on: timeline semaphore (compute done) + acquire semaphore (image ready)
            std::array<vk::Semaphore, 2> waitSemas = {*timelineSema, *acquireSema};
            std::array<uint64_t, 2>      waitVals  = {graphicsWait,  0};   // 0 = binary sema ignores value
            std::array<vk::PipelineStageFlags, 2> waitStages = {
                vk::PipelineStageFlagBits::eDrawIndirect,
                vk::PipelineStageFlagBits::eColorAttachmentOutput};

            // Signal: timeline (for next frame ordering) + binary renderDone (for present)
            std::array<vk::Semaphore, 2> signalSemas = {*timelineSema, *renderDone};
            std::array<uint64_t, 2>      signalVals  = {graphicsSignal, 0};

            vk::TimelineSemaphoreSubmitInfo tsi{
                .waitSemaphoreValueCount   = static_cast<uint32_t>(waitVals.size()),
                .pWaitSemaphoreValues      = waitVals.data(),
                .signalSemaphoreValueCount = static_cast<uint32_t>(signalVals.size()),
                .pSignalSemaphoreValues    = signalVals.data()};
            vk::SubmitInfo si{
                .pNext                = &tsi,
                .waitSemaphoreCount   = static_cast<uint32_t>(waitSemas.size()),
                .pWaitSemaphores      = waitSemas.data(),
                .pWaitDstStageMask    = waitStages.data(),
                .commandBufferCount   = 1,
                .pCommandBuffers      = &*commandBuffers[frameIndex],
                .signalSemaphoreCount = static_cast<uint32_t>(signalSemas.size()),
                .pSignalSemaphores    = signalSemas.data()};
            // Signal the per-frame fence so the next iteration can reclaim these resources.
            queue.submit(si, *inFlightFences[frameIndex]);

            // Present using the binary renderDone semaphore
            vk::PresentInfoKHR pi{
                .waitSemaphoreCount = 1,
                .pWaitSemaphores    = &*renderDone,
                .swapchainCount     = 1,
                .pSwapchains        = &*swapChain,
                .pImageIndices      = &imageIndex};
            auto pr = queue.presentKHR(pi);
            if (pr == vk::Result::eSuboptimalKHR || pr == vk::Result::eErrorOutOfDateKHR || framebufferResized)
            {
                framebufferResized = false;
                recreateSwapChain();
            }
        }

        frameIndex = (frameIndex + 1) % kMaxFrames;
    }

    // -----------------------------------------------------------------------
    // Buffer helpers
    // -----------------------------------------------------------------------
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags props,
                      vk::raii::Buffer &buf, vk::raii::DeviceMemory &mem) const
    {
        vk::BufferCreateInfo bci{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
        buf = vk::raii::Buffer(device, bci);
        auto req = buf.getMemoryRequirements();
        vk::MemoryAllocateInfo ai{.allocationSize = req.size, .memoryTypeIndex = findMemoryType(req.memoryTypeBits, props)};
        mem = vk::raii::DeviceMemory(device, ai);
        buf.bindMemory(mem, 0);
    }

    [[nodiscard]] vk::raii::CommandBuffer beginOneShot() const
    {
        vk::CommandBufferAllocateInfo ai{.commandPool=*commandPool,.level=vk::CommandBufferLevel::ePrimary,.commandBufferCount=1};
        auto cb = std::move(vk::raii::CommandBuffers(device, ai).front());
        cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return cb;
    }

    void endOneShot(const vk::raii::CommandBuffer &cb) const
    {
        cb.end();
        vk::SubmitInfo si{.commandBufferCount=1,.pCommandBuffers=&*cb};
        queue.submit(si, nullptr);
        queue.waitIdle();
    }

    void copyBuffer(const vk::raii::Buffer &src, const vk::raii::Buffer &dst, vk::DeviceSize size) const
    {
        auto cb = beginOneShot();
        cb.copyBuffer(src, dst, vk::BufferCopy(0, 0, size));
        endOneShot(cb);
    }

    [[nodiscard]] uint32_t findMemoryType(uint32_t filter, vk::MemoryPropertyFlags props) const
    {
        auto mp = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("no suitable memory type");
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
    {
        vk::ShaderModuleCreateInfo ci{.codeSize=code.size(),.pCode=reinterpret_cast<const uint32_t*>(code.data())};
        return vk::raii::ShaderModule(device, ci);
    }

    // -----------------------------------------------------------------------
    // Swapchain helpers
    // -----------------------------------------------------------------------
    vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR &caps)
    {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return caps.currentExtent;
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        return {
            std::clamp<uint32_t>(w, caps.minImageExtent.width,  caps.maxImageExtent.width),
            std::clamp<uint32_t>(h, caps.minImageExtent.height, caps.maxImageExtent.height)};
    }

    static vk::SurfaceFormatKHR chooseFormat(const std::vector<vk::SurfaceFormatKHR> &formats)
    {
        assert(!formats.empty());
        for (auto &f : formats)
            if (f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return f;
        return formats[0];
    }

    static vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR> &modes)
    {
        for (auto m : modes) if (m == vk::PresentModeKHR::eMailbox) return m;
        return vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]] std::vector<const char *> getRequiredInstanceExtensions() const
    {
        uint32_t cnt = 0;
        auto     ext = glfwGetRequiredInstanceExtensions(&cnt);
        std::vector<const char *> exts(ext, ext + cnt);
        if (kEnableValidation) exts.push_back(vk::EXTDebugUtilsExtensionName);
        return exts;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT  severity,
        vk::DebugUtilsMessageTypeFlagsEXT         type,
        const vk::DebugUtilsMessengerCallbackDataEXT *data,
        void *)
    {
        if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
            std::cerr << "VK [" << to_string(type) << "]: " << data->pMessage << "\n";
        return vk::False;
    }

    static std::vector<char> readFile(const std::string &path)
    {
        std::ifstream f(path, std::ios::ate | std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("cannot open: " + path);
        std::vector<char> buf(f.tellg());
        f.seekg(0); f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        return buf;
    }
};

#ifndef ANDROID_BUILD
int main()
{
    try { AsteroidFieldApp{}.run(); }
    catch (const std::exception &e) { std::cerr << e.what() << "\n"; return EXIT_FAILURE; }
    return EXIT_SUCCESS;
}
#endif // ANDROID_BUILD

#ifdef ANDROID_BUILD
extern "C" void chapter07_run() {
    try { AsteroidFieldApp{}.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh07", "%s", e.what()); }
}
#endif
