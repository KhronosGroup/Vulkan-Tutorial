// Chapter 6 – Advanced Data Structures: BVH Ray Tracer
//
// Demonstrates:
//   • Buffer Device Address (BDA): BVH nodes and triangle data are stored in
//     device-local buffers with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
//     The GPU receives their raw 64-bit addresses as push constants and
//     traverses them entirely via pointer arithmetic — no descriptor bindings.
//   • GPU Work Queue with atomic counter: each primary-ray hit spawns a shadow
//     ray job into a shared atomic-indexed queue (RWStructuredBuffer + InterlockedAdd).
//     A second dispatch processes the shadow queue.
//   • Interactive windowed rendering of a Cornell-box scene.
//
// Scene: Cornell box (5 walls) + 2 colored boxes = ~34 triangles, tiny BVH.
//
// Controls:
//   Mouse drag  – orbit camera
//   Scroll      – zoom
//   R           – reset camera
//   ESC         – quit
//
// Build: see CMakeLists.txt — uses WINDOWED flag
// Shader: shaders/slang.spv  (compiled from 06_advanced_data_structures.slang)

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

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr uint32_t kWidth       = 1280;
constexpr uint32_t kHeight      = 720;
constexpr int      kMaxFrames   = 2;
constexpr int      kAcquireSemas = kMaxFrames + 1;

// Shadow queue capacity (max shadow rays per frame)
constexpr uint32_t kShadowQueueCap = 8192u;

const std::vector<char const *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// CPU-side data structures (must match shader)
// ---------------------------------------------------------------------------
struct alignas(16) BVHNode
{
    float    aabbMin[3];
    int32_t  leftChild;   // -1 if leaf
    float    aabbMax[3];
    int32_t  rightChild;  // -1 if leaf
    int32_t  triOffset;   // first triangle index (leaf only)
    int32_t  triCount;    // triangle count (leaf only)
    int32_t  _pad[2];     // keep 48-byte struct aligned to 16
};
static_assert(sizeof(BVHNode) == 48, "BVHNode size mismatch");

// Compact layout — no per-field padding — so the struct is exactly 64 bytes.
// Offsets: v0=0, v1=12, v2=24, normal=36, color=48
struct Triangle
{
    float v0[3];     // offset  0 (12 bytes)
    float v1[3];     // offset 12 (12 bytes)
    float v2[3];     // offset 24 (12 bytes)
    float normal[3]; // offset 36 (12 bytes)
    float color[4];  // offset 48 (16 bytes)
};
static_assert(sizeof(Triangle) == 64, "Triangle size mismatch");

// Push constants: must be byte-identical to the Slang struct
struct RayTracePush
{
    uint64_t bvhAddr;         // device address of BVHNode array
    uint64_t triAddr;         // device address of Triangle array
    uint64_t shadowQueueAddr; // device address of ShadowJob array
    uint64_t counterAddr;     // device address of atomic counter (uint)
    float    camPos[3];
    uint32_t frameWidth;
    float    camTarget[3];
    uint32_t frameHeight;
    float    camUp[3];
    float    fovY;            // vertical FoV in radians
    uint32_t nodeCount;
    uint32_t triCount;
    uint32_t queueCapacity;
    uint32_t _pad;
};
static_assert(sizeof(RayTracePush) == 96, "RayTracePush size mismatch");

// Shadow pass push constants
struct ShadowPush
{
    uint64_t bvhAddr;
    uint64_t triAddr;
    uint64_t shadowQueueAddr;
    uint64_t counterAddr;
    uint64_t outputImageAddr; // unused — output image is in descriptor set
    uint32_t frameWidth;
    uint32_t frameHeight;
    uint32_t queueCapacity;
    uint32_t _pad;
};

// ---------------------------------------------------------------------------
// BVH builder (CPU)
// ---------------------------------------------------------------------------
// Builds a simple flat two-level BVH from a triangle soup.
// Node 0 is the root; leaves directly reference ranges of triangles.

static void computeAABB(const std::vector<Triangle>& tris,
                         int offset, int count,
                         float outMin[3], float outMax[3])
{
    outMin[0] = outMin[1] = outMin[2] =  1e30f;
    outMax[0] = outMax[1] = outMax[2] = -1e30f;
    for (int i = offset; i < offset + count; ++i)
    {
        for (int v = 0; v < 3; ++v)
        {
            const float* verts[3] = {tris[i].v0, tris[i].v1, tris[i].v2};
            for (int c = 0; c < 3; ++c)
            {
                outMin[c] = std::min(outMin[c], verts[v][c]);
                outMax[c] = std::max(outMax[c], verts[v][c]);
            }
        }
    }
}

// Build a simple binary BVH splitting on the longest axis at the midpoint.
// Returns the number of nodes created (written into `nodes`).
static int buildBVH(std::vector<BVHNode>& nodes,
                    std::vector<Triangle>& tris,
                    int offset, int count, int depth = 0)
{
    int nodeIdx = static_cast<int>(nodes.size());
    nodes.push_back({});
    BVHNode& node = nodes.back();

    computeAABB(tris, offset, count, node.aabbMin, node.aabbMax);

    if (count <= 4 || depth >= 8)
    {
        // Leaf
        node.leftChild  = -1;
        node.rightChild = -1;
        node.triOffset  = offset;
        node.triCount   = count;
        return nodeIdx;
    }

    // Find longest axis
    float extents[3] = {
        node.aabbMax[0] - node.aabbMin[0],
        node.aabbMax[1] - node.aabbMin[1],
        node.aabbMax[2] - node.aabbMin[2]
    };
    int axis = 0;
    if (extents[1] > extents[axis]) axis = 1;
    if (extents[2] > extents[axis]) axis = 2;

    float mid = (node.aabbMin[axis] + node.aabbMax[axis]) * 0.5f;

    // Partition triangles by centroid on axis
    auto midIt = std::partition(
        tris.begin() + offset,
        tris.begin() + offset + count,
        [axis, mid](const Triangle& t) {
            float centroid = (t.v0[axis] + t.v1[axis] + t.v2[axis]) / 3.0f;
            return centroid < mid;
        });

    int leftCount  = static_cast<int>(midIt - (tris.begin() + offset));
    int rightCount = count - leftCount;

    // Degenerate: all on one side → make leaf
    if (leftCount == 0 || rightCount == 0)
    {
        node.leftChild  = -1;
        node.rightChild = -1;
        node.triOffset  = offset;
        node.triCount   = count;
        return nodeIdx;
    }

    // Use nodes[nodeIdx] (stable index) for both children — the 'node' reference
    // becomes dangling after push_back reallocates the vector during the left
    // child's recursive call.
    nodes[nodeIdx].leftChild  = buildBVH(nodes, tris, offset,             leftCount,  depth + 1);
    nodes[nodeIdx].rightChild = buildBVH(nodes, tris, offset + leftCount, rightCount, depth + 1);
    nodes[nodeIdx].triOffset  = -1;
    nodes[nodeIdx].triCount   = 0;

    return nodeIdx;
}

// ---------------------------------------------------------------------------
// Scene construction — Cornell box
// ---------------------------------------------------------------------------
static void pushQuad(std::vector<Triangle>& tris,
                     glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
                     glm::vec3 color)
{
    // Two triangles per quad, compute face normal
    glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    // Flip if pointing wrong way (for Cornell box normals should face inward)

    auto fillTri = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        Triangle t{};
        t.v0[0] = a.x;   t.v0[1] = a.y;   t.v0[2] = a.z;
        t.v1[0] = b.x;   t.v1[1] = b.y;   t.v1[2] = b.z;
        t.v2[0] = c.x;   t.v2[1] = c.y;   t.v2[2] = c.z;
        t.normal[0] = n.x; t.normal[1] = n.y; t.normal[2] = n.z;
        t.color[0] = color.r; t.color[1] = color.g;
        t.color[2] = color.b; t.color[3] = 1.0f;
        tris.push_back(t);
    };
    fillTri(v0, v1, v2);
    fillTri(v0, v2, v3);
}

static void buildCornellBox(std::vector<Triangle>& tris)
{
    // Cornell box dimensions: [-1,1] × [0,2] × [-1,1]
    // Walls: floor, ceiling, back, left(red), right(green)
    const glm::vec3 white  = {0.73f, 0.73f, 0.73f};
    const glm::vec3 red    = {0.65f, 0.05f, 0.05f};
    const glm::vec3 green  = {0.12f, 0.45f, 0.15f};

    // Floor (y=0, normal up)
    pushQuad(tris, {-1,0,-1}, {1,0,-1}, {1,0,1}, {-1,0,1}, white);
    // Ceiling (y=2, normal down)
    pushQuad(tris, {-1,2,1}, {1,2,1}, {1,2,-1}, {-1,2,-1}, white);
    // Back wall (z=-1, normal +z)
    pushQuad(tris, {-1,0,-1}, {-1,2,-1}, {1,2,-1}, {1,0,-1}, white);
    // Left wall (x=-1, normal +x) — red
    pushQuad(tris, {-1,0,1}, {-1,2,1}, {-1,2,-1}, {-1,0,-1}, red);
    // Right wall (x=1, normal -x) — green
    pushQuad(tris, {1,0,-1}, {1,2,-1}, {1,2,1}, {1,0,1}, green);

    // Tall box (5 quads: top + 4 sides), centered around (-0.35, 0, -0.35)
    const glm::vec3 boxColor = {0.73f, 0.73f, 0.73f};
    float bx = -0.35f, bz = -0.35f, bw = 0.3f, bh = 1.2f;
    // Top
    pushQuad(tris, {bx-bw,bh,bz-bw},{bx+bw,bh,bz-bw},{bx+bw,bh,bz+bw},{bx-bw,bh,bz+bw}, boxColor);
    // Front (+z)
    pushQuad(tris, {bx-bw,0,bz+bw},{bx+bw,0,bz+bw},{bx+bw,bh,bz+bw},{bx-bw,bh,bz+bw}, boxColor);
    // Back (-z)
    pushQuad(tris, {bx+bw,0,bz-bw},{bx-bw,0,bz-bw},{bx-bw,bh,bz-bw},{bx+bw,bh,bz-bw}, boxColor);
    // Left (-x)
    pushQuad(tris, {bx-bw,0,bz-bw},{bx-bw,0,bz+bw},{bx-bw,bh,bz+bw},{bx-bw,bh,bz-bw}, boxColor);
    // Right (+x)
    pushQuad(tris, {bx+bw,0,bz+bw},{bx+bw,0,bz-bw},{bx+bw,bh,bz-bw},{bx+bw,bh,bz+bw}, boxColor);

    // Area light patch on ceiling (y≈2, centred, slightly inset from the ceiling quad).
    // The shader identifies it by color.a == 2.0 and returns emission directly,
    // making it appear as a bright white patch — the canonical Cornell box light.
    {
        const glm::vec3 lightEmit = {6.0f, 6.0f, 5.5f}; // bright warm white (clamped in shader)
        // y=1.96 keeps the light patch clearly separated from the ceiling quad at y=2.0
        // to avoid floating-point precision issues in the BVH AABB slab test
        glm::vec3 lv0 = {-0.3f, 1.96f,  0.3f};
        glm::vec3 lv1 = { 0.3f, 1.96f,  0.3f};
        glm::vec3 lv2 = { 0.3f, 1.96f, -0.3f};
        glm::vec3 lv3 = {-0.3f, 1.96f, -0.3f};
        size_t before = tris.size();
        pushQuad(tris, lv0, lv1, lv2, lv3, lightEmit);
        // Mark emission: color.a = 2.0 distinguishes this from ordinary triangles (a=1.0)
        for (size_t ti = before; ti < tris.size(); ++ti)
            tris[ti].color[3] = 2.0f;
    }

    // Short box, centered around (0.35, 0, 0.2)
    float sx = 0.35f, sz = 0.2f, sw = 0.3f, sh = 0.6f;
    // Top
    pushQuad(tris, {sx-sw,sh,sz-sw},{sx+sw,sh,sz-sw},{sx+sw,sh,sz+sw},{sx-sw,sh,sz+sw}, boxColor);
    // Front (+z)
    pushQuad(tris, {sx-sw,0,sz+sw},{sx+sw,0,sz+sw},{sx+sw,sh,sz+sw},{sx-sw,sh,sz+sw}, boxColor);
    // Back (-z)
    pushQuad(tris, {sx+sw,0,sz-sw},{sx-sw,0,sz-sw},{sx-sw,sh,sz-sw},{sx+sw,sh,sz-sw}, boxColor);
    // Left (-x)
    pushQuad(tris, {sx-sw,0,sz-sw},{sx-sw,0,sz+sw},{sx-sw,sh,sz+sw},{sx-sw,sh,sz-sw}, boxColor);
    // Right (+x)
    pushQuad(tris, {sx+sw,0,sz+sw},{sx+sw,0,sz-sw},{sx+sw,sh,sz-sw},{sx+sw,sh,sz+sw}, boxColor);
}

// ---------------------------------------------------------------------------
// BVH Ray Tracer App
// ---------------------------------------------------------------------------
class BVHRayTracerApp
{
  public:
    void run()
    {
        initWindow();
        buildScene();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    // -----------------------------------------------------------------------
    // Window + camera state
    // -----------------------------------------------------------------------
    GLFWwindow *m_window    = nullptr;
    bool        m_resized   = false;
    bool        m_dragging  = false;
    double      m_lastMx    = 0.0, m_lastMy = 0.0;

    // Spherical camera — start from the front (+Z side) looking into the open box
    float m_theta    = 0.0f;      // horizontal angle (radians): 0 = front (+Z axis)
    float m_phi      = 0.1f;      // vertical angle (radians): slight downward tilt shows floor+ceiling
    float m_radius   = 3.0f;      // distance from target (closer reveals full floor/ceiling)
    glm::vec3 m_target = {0.0f, 1.0f, 0.0f};

    // -----------------------------------------------------------------------
    // Scene data
    // -----------------------------------------------------------------------
    std::vector<Triangle> m_triangles;
    std::vector<BVHNode>  m_bvhNodes;

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
    vk::raii::SwapchainKHR  m_swapchain  = nullptr;
    std::vector<vk::Image>  m_swapImages;
    vk::SurfaceFormatKHR    m_swapFormat{};
    vk::Extent2D            m_swapExtent{};

    // -----------------------------------------------------------------------
    // BDA buffers (persistent across frames)
    // -----------------------------------------------------------------------
    vk::raii::Buffer       m_bvhBuf     = nullptr;
    vk::raii::DeviceMemory m_bvhMem     = nullptr;
    vk::raii::Buffer       m_triBuf     = nullptr;
    vk::raii::DeviceMemory m_triMem     = nullptr;
    vk::raii::Buffer       m_shadowBuf  = nullptr;  // shadow job queue
    vk::raii::DeviceMemory m_shadowMem  = nullptr;
    vk::raii::Buffer       m_counterBuf = nullptr;  // atomic counter for shadow queue
    vk::raii::DeviceMemory m_counterMem = nullptr;

    // Device addresses
    uint64_t m_bvhAddr     = 0;
    uint64_t m_triAddr     = 0;
    uint64_t m_shadowAddr  = 0;
    uint64_t m_counterAddr = 0;

    // -----------------------------------------------------------------------
    // Pipelines / layouts
    // -----------------------------------------------------------------------
    vk::raii::DescriptorSetLayout m_dsLayout        = nullptr;
    vk::raii::PipelineLayout      m_primaryLayout   = nullptr;
    vk::raii::PipelineLayout      m_shadowLayout    = nullptr;
    vk::raii::Pipeline            m_primaryPipeline = nullptr;
    vk::raii::Pipeline            m_shadowPipeline  = nullptr;

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
        glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);

        m_window = glfwCreateWindow(kWidth, kHeight,
            "BVH Ray Tracer  |  drag=orbit  scroll=zoom  R=reset  ESC=quit",
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
        static_cast<BVHRayTracerApp *>(glfwGetWindowUserPointer(w))->m_resized = true;
    }

    static void cbScroll(GLFWwindow *w, double, double dy)
    {
        auto *app     = static_cast<BVHRayTracerApp *>(glfwGetWindowUserPointer(w));
        float factor  = (dy > 0.0) ? 0.9f : (1.0f / 0.9f);
        app->m_radius = std::clamp(app->m_radius * factor, 0.5f, 20.0f);
    }

    static void cbMouseButton(GLFWwindow *w, int button, int action, int)
    {
        auto *app = static_cast<BVHRayTracerApp *>(glfwGetWindowUserPointer(w));
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            app->m_dragging = (action == GLFW_PRESS);
            glfwGetCursorPos(w, &app->m_lastMx, &app->m_lastMy);
        }
    }

    static void cbCursorPos(GLFWwindow *w, double mx, double my)
    {
        auto *app = static_cast<BVHRayTracerApp *>(glfwGetWindowUserPointer(w));
        if (app->m_dragging)
        {
            float dx = static_cast<float>(mx - app->m_lastMx) * 0.005f;
            float dy = static_cast<float>(my - app->m_lastMy) * 0.005f;
            app->m_theta += dx;
            app->m_phi    = std::clamp(app->m_phi + dy, -1.5f, 1.5f);
        }
        app->m_lastMx = mx;
        app->m_lastMy = my;
    }

    static void cbKey(GLFWwindow *w, int key, int, int action, int)
    {
        if (action != GLFW_PRESS)
            return;
        auto *app = static_cast<BVHRayTracerApp *>(glfwGetWindowUserPointer(w));
        switch (key)
        {
            case GLFW_KEY_R:
                app->m_theta  = 0.0f;
                app->m_phi    = 0.1f;
                app->m_radius = 3.0f;
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(w, GLFW_TRUE);
                break;
            default: break;
        }
    }

    // =======================================================================
    // Scene + BVH construction
    // =======================================================================
    void buildScene()
    {
        buildCornellBox(m_triangles);

        buildBVH(m_bvhNodes, m_triangles, 0,
                 static_cast<int>(m_triangles.size()));

        std::cout << "=== BVH Ray Tracer Scene ===\n";
        std::cout << "  Triangles : " << m_triangles.size() << '\n';
        std::cout << "  BVH nodes : " << m_bvhNodes.size()  << '\n';
        std::cout << "============================\n";
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
        uploadSceneBuffers();
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
        m_cmdPool         = nullptr;
        m_shadowPipeline  = nullptr;
        m_primaryPipeline = nullptr;
        m_shadowLayout    = nullptr;
        m_primaryLayout   = nullptr;
        m_dsLayout        = nullptr;
        m_counterBuf      = nullptr;
        m_counterMem      = nullptr;
        m_shadowBuf       = nullptr;
        m_shadowMem       = nullptr;
        m_triBuf          = nullptr;
        m_triMem          = nullptr;
        m_bvhBuf          = nullptr;
        m_bvhMem          = nullptr;
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
            .pApplicationName   = "BVH Ray Tracer",
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
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT type(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
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

        auto props = m_physDev.getProperties();
        std::cout << "GPU: " << props.deviceName.data() << '\n';
    }

    // =======================================================================
    // Logical device  — enable bufferDeviceAddress + scalarBlockLayout
    // =======================================================================
    void createLogicalDevice()
    {
        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan12Features,
            vk::PhysicalDeviceVulkan13Features>
        featureChain = {
            {.features = {.shaderInt64 = true}},
            {.scalarBlockLayout   = true,
             .timelineSemaphore   = true,
             .bufferDeviceAddress = true},
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
    // Upload scene buffers (BVH nodes + triangles + shadow queue + counter)
    // =======================================================================
    void uploadSceneBuffers()
    {
        auto uploadBDA = [&](const void* data, vk::DeviceSize size,
                              vk::raii::Buffer& buf, vk::raii::DeviceMemory& mem,
                              uint64_t& addr)
        {
            // Create device-local buffer with BDA + storage + transfer-dst
            vk::BufferCreateInfo bci{
                .size        = size,
                .usage       = vk::BufferUsageFlagBits::eStorageBuffer
                             | vk::BufferUsageFlagBits::eShaderDeviceAddress
                             | vk::BufferUsageFlagBits::eTransferDst,
                .sharingMode = vk::SharingMode::eExclusive};
            buf = vk::raii::Buffer(m_device, bci);

            auto req = buf.getMemoryRequirements();
            vk::MemoryAllocateFlagsInfo mafi{
                .flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
            vk::MemoryAllocateInfo mai{
                .pNext           = &mafi,
                .allocationSize  = req.size,
                .memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eDeviceLocal)};
            mem = vk::raii::DeviceMemory(m_device, mai);
            buf.bindMemory(*mem, 0);

            // Staging upload
            vk::BufferCreateInfo stagCI{
                .size        = size,
                .usage       = vk::BufferUsageFlagBits::eTransferSrc,
                .sharingMode = vk::SharingMode::eExclusive};
            vk::raii::Buffer stagBuf(m_device, stagCI);
            auto stagReq = stagBuf.getMemoryRequirements();
            vk::MemoryAllocateInfo stagMai{
                .allocationSize  = stagReq.size,
                .memoryTypeIndex = findMemoryType(stagReq.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent)};
            vk::raii::DeviceMemory stagMem(m_device, stagMai);
            stagBuf.bindMemory(*stagMem, 0);

            if (data)
            {
                void *mapped = stagMem.mapMemory(0, size);
                std::memcpy(mapped, data, size);
                stagMem.unmapMemory();
            }
            else
            {
                // Zero the staging memory (for counter / shadow queue)
                void *mapped = stagMem.mapMemory(0, size);
                std::memset(mapped, 0, size);
                stagMem.unmapMemory();
            }

            // One-shot copy
            vk::CommandBufferAllocateInfo cbai{
                .commandPool        = *m_cmdPool,
                .level              = vk::CommandBufferLevel::ePrimary,
                .commandBufferCount = 1};
            auto cb = std::move(vk::raii::CommandBuffers(m_device, cbai).front());
            cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cb.copyBuffer(*stagBuf, *buf, vk::BufferCopy{0, 0, size});
            cb.end();
            vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
            m_queue.submit(si, nullptr);
            m_queue.waitIdle();

            addr = static_cast<uint64_t>(
                m_device.getBufferAddress(vk::BufferDeviceAddressInfo{.buffer = *buf}));
        };

        vk::DeviceSize bvhSize    = m_bvhNodes.size()  * sizeof(BVHNode);
        vk::DeviceSize triSize    = m_triangles.size()  * sizeof(Triangle);
        // ShadowJob in shader: two uint (pixelX, pixelY) + one uint (hitTriIdx) = 12 bytes each
        // We'll store a compact struct: { uint pixelX; uint pixelY; uint hitTriIdx; float hitDist; }
        // = 16 bytes, kShadowQueueCap entries + 4 bytes for counter
        vk::DeviceSize shadowSize = kShadowQueueCap * 16u;
        vk::DeviceSize counterSize = 4u;

        uploadBDA(m_bvhNodes.data(),  bvhSize,    m_bvhBuf,     m_bvhMem,     m_bvhAddr);
        uploadBDA(m_triangles.data(), triSize,    m_triBuf,     m_triMem,     m_triAddr);
        uploadBDA(nullptr,            shadowSize, m_shadowBuf,  m_shadowMem,  m_shadowAddr);
        uploadBDA(nullptr,            counterSize,m_counterBuf, m_counterMem, m_counterAddr);

        std::cout << "BVH buffer address  : 0x" << std::hex << m_bvhAddr    << '\n';
        std::cout << "Tri buffer address  : 0x" << m_triAddr    << '\n';
        std::cout << "Shadow queue address: 0x" << m_shadowAddr << '\n';
        std::cout << "Counter address     : 0x" << m_counterAddr << std::dec << '\n';
    }

    // =======================================================================
    // Descriptor set layout  (binding 0 = storage image)
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
    // Pipelines
    // =======================================================================
    vk::raii::Pipeline buildComputePipeline(vk::raii::PipelineLayout& layout,
                                             const char* entryName)
    {
        auto code = readFile("shaders/slang.spv");
        vk::ShaderModuleCreateInfo smci{
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<uint32_t const *>(code.data())};
        vk::raii::ShaderModule shaderModule(m_device, smci);

        vk::PipelineShaderStageCreateInfo stage{
            .stage  = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName  = entryName};
        vk::ComputePipelineCreateInfo pci{.stage = stage, .layout = *layout};
        return vk::raii::Pipeline(m_device, nullptr, pci);
    }

    void createPipelines()
    {
        // Primary ray pipeline layout: descriptor set (output image) + push constants
        vk::PushConstantRange pcPrimary{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(RayTracePush)};
        vk::PipelineLayoutCreateInfo primaryPlci{
            .setLayoutCount         = 1,
            .pSetLayouts            = &*m_dsLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcPrimary};
        m_primaryLayout = vk::raii::PipelineLayout(m_device, primaryPlci);

        // Shadow pipeline layout: descriptor set + shadow push constants
        vk::PushConstantRange pcShadow{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(ShadowPush)};
        vk::PipelineLayoutCreateInfo shadowPlci{
            .setLayoutCount         = 1,
            .pSetLayouts            = &*m_dsLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pcShadow};
        m_shadowLayout = vk::raii::PipelineLayout(m_device, shadowPlci);

        m_primaryPipeline = buildComputePipeline(m_primaryLayout, "primaryRayMain");
        m_shadowPipeline  = buildComputePipeline(m_shadowLayout,  "shadowQueueMain");
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
            .memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal)};
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

    void recordCommands(PerFrame &f, uint32_t imageIndex)
    {
        // Compute camera position from spherical coordinates
        float sinPhi = std::sin(m_phi), cosPhi = std::cos(m_phi);
        float sinThe = std::sin(m_theta), cosThe = std::cos(m_theta);
        glm::vec3 camPos = m_target + m_radius * glm::vec3(
            cosPhi * sinThe, sinPhi, cosPhi * cosThe);
        glm::vec3 camUp = {0.0f, (cosPhi > 0 ? 1.0f : -1.0f), 0.0f};

        auto &cb = f.cmdBuf;
        cb.reset();
        cb.begin({});

        // --- Reset atomic counter to 0 ---
        cb.fillBuffer(*m_counterBuf, 0, 4, 0u);

        // Barrier: fillBuffer → compute shader reads counter
        vk::BufferMemoryBarrier2 counterReset{
            .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                             vk::AccessFlagBits2::eShaderStorageWrite,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer        = *m_counterBuf,
            .offset        = 0,
            .size          = VK_WHOLE_SIZE};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &counterReset});

        // --- Primary ray dispatch ---
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_primaryPipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                              *m_primaryLayout, 0, {f.dsSet}, {});

        RayTracePush push{};
        push.bvhAddr         = m_bvhAddr;
        push.triAddr         = m_triAddr;
        push.shadowQueueAddr = m_shadowAddr;
        push.counterAddr     = m_counterAddr;
        push.camPos[0]       = camPos.x;
        push.camPos[1]       = camPos.y;
        push.camPos[2]       = camPos.z;
        push.frameWidth      = m_swapExtent.width;
        push.camTarget[0]    = m_target.x;
        push.camTarget[1]    = m_target.y;
        push.camTarget[2]    = m_target.z;
        push.frameHeight     = m_swapExtent.height;
        push.camUp[0]        = camUp.x;
        push.camUp[1]        = camUp.y;
        push.camUp[2]        = camUp.z;
        push.fovY            = 0.785398f; // 45 degrees
        push.nodeCount       = static_cast<uint32_t>(m_bvhNodes.size());
        push.triCount        = static_cast<uint32_t>(m_triangles.size());
        push.queueCapacity   = kShadowQueueCap;
        push._pad            = 0;

        cb.pushConstants<RayTracePush>(*m_primaryLayout,
                                       vk::ShaderStageFlagBits::eCompute, 0, push);

        uint32_t gx = (m_swapExtent.width  + 15u) / 16u;
        uint32_t gy = (m_swapExtent.height + 15u) / 16u;
        cb.dispatch(gx, gy, 1);

        // --- Barrier: primary writes → shadow reads ---
        vk::MemoryBarrier2 primaryToShadow{
            .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                             vk::AccessFlagBits2::eShaderStorageWrite};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers    = &primaryToShadow});

        // --- Shadow queue pass ---
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_shadowPipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                              *m_shadowLayout, 0, {f.dsSet}, {});

        ShadowPush sp{};
        sp.bvhAddr         = m_bvhAddr;
        sp.triAddr         = m_triAddr;
        sp.shadowQueueAddr = m_shadowAddr;
        sp.counterAddr     = m_counterAddr;
        sp.outputImageAddr = 0; // unused
        sp.frameWidth      = m_swapExtent.width;
        sp.frameHeight     = m_swapExtent.height;
        sp.queueCapacity   = kShadowQueueCap;
        sp._pad            = 0;

        cb.pushConstants<ShadowPush>(*m_shadowLayout,
                                      vk::ShaderStageFlagBits::eCompute, 0, sp);

        // One workgroup per 64 shadow jobs (flat dispatch, capped at kShadowQueueCap)
        uint32_t shadowGroups = (kShadowQueueCap + 63u) / 64u;
        cb.dispatch(shadowGroups, 1, 1);

        // --- Barrier: shadow compute → blit ---
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
            if ((filter & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("no suitable memory type");
    }

    static vk::SurfaceFormatKHR chooseFormat(std::vector<vk::SurfaceFormatKHR> const &fmts,
                                              vk::raii::PhysicalDevice const &physDev)
    {
        assert(!fmts.empty());
        auto supportsBlitDst = [&](vk::Format fmt) {
            return !!(physDev.getFormatProperties(fmt).optimalTilingFeatures &
                      vk::FormatFeatureFlagBits::eBlitDst);
        };
        for (auto const &f : fmts)
            if (f.format == vk::Format::eB8G8R8A8Unorm &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
                supportsBlitDst(f.format))
                return f;
        for (auto const &f : fmts)
            if (f.format == vk::Format::eB8G8R8A8Unorm &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return f;
        for (auto const &f : fmts)
            if (f.format == vk::Format::eB8G8R8A8Srgb &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return f;
        return fmts[0];
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
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT        type,
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
        BVHRayTracerApp app;
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
extern "C" void chapter06_run() {
    try { BVHRayTracerApp{}.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh06", "%s", e.what()); }
}
#endif
