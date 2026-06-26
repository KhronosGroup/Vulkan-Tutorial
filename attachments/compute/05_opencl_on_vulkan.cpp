// ===========================================================================
// 05_opencl_on_vulkan.cpp  —  Vulkan rendering powered by an OpenCL kernel
// ===========================================================================
// A normal *Vulkan* application — window, swapchain, present loop, free-fly
// camera — that draws its scene with an OpenCL C kernel instead of a hand-written
// Vulkan shader. This is the real value of the OpenCL-on-Vulkan toolchain: a
// Vulkan engine can reuse the huge body of existing OpenCL kernels.
//
// The scene is a raymarched, instanced forest (05_opencl_on_vulkan.cl): one tree
// SDF repeated across an infinite grid, with per-instance variation. You fly
// through it in real time.
//
// Two compute backends can drive the kernel, selected at startup:
//
//   * clspv AOT (preferred, zero-copy): the OpenCL C kernel is compiled by clspv
//     to forest.spv at build time and runs as the Vulkan engine's OWN compute
//     shader, writing straight into the buffer that is presented. The OpenCL
//     kernel becomes "just another shader" in the Vulkan pipeline — no copies,
//     no interop layer. This is Vulkan taking advantage of OpenCL, directly.
//
//   * clvk runtime (alternative): the SAME .cl is compiled and executed at run time
//     by clvk (OpenCL 3.0 layered on Vulkan). clvk does not expose
//     cl_khr_external_memory, so its result is bridged into the Vulkan buffer with
//     a per-frame readback. It demonstrates the runtime-layering tool at the cost
//     of one copy/frame.
//
// We deliberately refuse to use an unrelated OpenCL platform (e.g. a vendor's
// native CUDA/ROCm driver): the point is the Vulkan layer, so only a platform
// whose name contains "clvk" is accepted for the runtime path.
//
// Controls: W/A/S/D move, E/Q up/down, mouse-drag look, Shift boost, R reset,
//           ESC quit.   Force a backend with: --backend=aot | --backend=clvk
//
// See the chapter docs under en/Advanced_Vulkan_Compute/05_OpenCL_on_Vulkan/.
// ===========================================================================

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#  include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#ifdef ANDROID_BUILD
#  include "glfw_android_shim.h"
#else
#  define GLFW_INCLUDE_VULKAN
#  include <GLFW/glfw3.h>
#endif

#ifdef HAVE_OPENCL
#  define CL_TARGET_OPENCL_VERSION 300
#  include <CL/cl.h>
#endif

// ---------------------------------------------------------------------------
constexpr uint32_t kWidth        = 1280;
constexpr uint32_t kHeight       = 720;
constexpr int      kLocal        = 16;     // must match reqd_work_group_size in .cl
constexpr int      kAcquireSemas = 3;

const std::vector<const char*> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// Byte-compatible with the `Params` struct in the .cl file (8 packed 4-byte scalars).
// (The raymarch step count is a compile-time constant in the kernel — see MAX_STEPS.)
struct Params {
    int32_t width;
    int32_t height;
    float   camX, camY, camZ;
    float   camYaw, camPitch;
    float   fog;
};

static uint32_t roundUp(uint32_t v, uint32_t m) { return (v + m - 1) / m * m; }

// ===========================================================================
// clvk runtime backend: compile the .cl at run time, dispatch, and bridge the
// result into a host pointer (clvk has no external-memory sharing).
// ===========================================================================
#ifdef HAVE_OPENCL
struct ClvkBackend {
    cl_platform_id   platform = nullptr;
    cl_device_id     device   = nullptr;
    cl_context       context  = nullptr;
    cl_command_queue queue    = nullptr;
    cl_program       program  = nullptr;
    cl_kernel        kernel   = nullptr;
    cl_mem           paramsMem = nullptr;
    cl_mem           outMem    = nullptr;

    // Find the clvk platform (and ONLY clvk — never a CUDA/ROCm fallback).
    static bool probe(ClvkBackend& out) {
        cl_uint nplat = 0;
        if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return false;
        std::vector<cl_platform_id> plats(nplat);
        clGetPlatformIDs(nplat, plats.data(), nullptr);
        for (auto p : plats) {
            char name[256] = {};
            clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
            std::string lower(name);
            for (auto& c : lower) c = char(std::tolower((unsigned char)c));
            std::cout << "[clvk]   OpenCL platform: " << name << '\n';
            if (lower.find("clvk") == std::string::npos) continue;
            cl_device_id dev = nullptr;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_DEFAULT, 1, &dev, nullptr) != CL_SUCCESS) continue;
            char dname[256] = {};
            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
            std::cout << "[clvk]   clvk device: " << dname << '\n';
            out.platform = p; out.device = dev;
            return true;
        }
        std::cout << "[clvk]   no clvk platform found.\n";
        return false;
    }

    bool init(const std::string& source, size_t outBytes) {
        cl_int err = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err) { std::cerr << "[clvk] clCreateContext " << err << '\n'; return false; }
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        if (err) { std::cerr << "[clvk] queue " << err << '\n'; return false; }

        const char* src = source.c_str();
        const size_t len = source.size();
        program = clCreateProgramWithSource(context, 1, &src, &len, &err);
        if (err) { std::cerr << "[clvk] program " << err << '\n'; return false; }
        if (clBuildProgram(program, 1, &device, "", nullptr, nullptr) != CL_SUCCESS) {
            size_t ls = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ls);
            std::string log(ls, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ls, log.data(), nullptr);
            std::cerr << "[clvk] build failed:\n" << log << '\n';
            return false;
        }
        kernel = clCreateKernel(program, "render", &err);
        if (err) { std::cerr << "[clvk] clCreateKernel " << err << '\n'; return false; }

        paramsMem = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Params), nullptr, &err);
        if (err) { std::cerr << "[clvk] params buffer " << err << '\n'; return false; }
        outMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outBytes, nullptr, &err);
        if (err) { std::cerr << "[clvk] output buffer " << err << '\n'; return false; }
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &paramsMem);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &outMem);
        return true;
    }

    // Render one frame and copy the result into the Vulkan-visible host pointer.
    void compute(const Params& p, void* hostDst, size_t bytes) {
        clEnqueueWriteBuffer(queue, paramsMem, CL_FALSE, 0, sizeof(Params), &p, 0, nullptr, nullptr);
        const size_t global[2] = {roundUp(p.width, kLocal), roundUp(p.height, kLocal)};
        const size_t local[2]  = {kLocal, kLocal};
        clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, outMem, CL_TRUE, 0, bytes, hostDst, 0, nullptr, nullptr);
    }

    void destroy() {
        if (outMem)    clReleaseMemObject(outMem);
        if (paramsMem) clReleaseMemObject(paramsMem);
        if (kernel)    clReleaseKernel(kernel);
        if (program)   clReleaseProgram(program);
        if (queue)     clReleaseCommandQueue(queue);
        if (context)   clReleaseContext(context);
        *this = ClvkBackend{};
    }
};
#endif // HAVE_OPENCL

// ===========================================================================
class ForestApp {
public:
    explicit ForestApp(const std::string& forced) : m_forced(forced) {}

    void run() {
        initWindow();
        selectBackend();
        createInstance();
        setupDebug();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createCommandPool();
        createSwapchain();
        createIntermediateImage();
        createSharedBuffer();
        initComputeBackend();
        createSyncAndCmd();
        mainLoop();
        cleanup();
    }

private:
    enum class Backend { ClspvAot, ClvkRuntime };
    Backend     m_backend = Backend::ClspvAot;
    std::string m_forced;          // "aot", "clvk", or ""

    // ---- window / camera ----
    GLFWwindow* m_window = nullptr;
    bool   m_dragging = false;
    double m_lastMx = 0, m_lastMy = 0;
    float  m_camX = 0.0f, m_camY = 3.1f, m_camZ = 0.0f;
    float  m_yaw = 0.45f, m_pitch = -0.40f;
    double m_lastTime = 0.0;

    // ---- Vulkan core ----
    vk::raii::Context                m_ctx;
    vk::raii::Instance               m_instance    = nullptr;
    vk::raii::DebugUtilsMessengerEXT m_debug       = nullptr;
    vk::raii::SurfaceKHR             m_surface     = nullptr;
    vk::raii::PhysicalDevice         m_phys        = nullptr;
    vk::raii::Device                 m_device      = nullptr;
    uint32_t                         m_queueFamily = ~0u;
    vk::raii::Queue                  m_queue       = nullptr;
    vk::raii::CommandPool            m_cmdPool     = nullptr;

    // ---- swapchain ----
    vk::raii::SwapchainKHR  m_swapchain = nullptr;
    std::vector<vk::Image>  m_swapImages;
    vk::SurfaceFormatKHR    m_swapFormat{};
    vk::Extent2D            m_extent{};

    // ---- intermediate RGBA image (blit converts RGBA->BGRA) ----
    vk::raii::Image        m_interImg = nullptr;
    vk::raii::DeviceMemory m_interMem = nullptr;

    // ---- shared output buffer (compute writes it; present copies it) ----
    // Host-visible so the AOT compute shader writes it on the GPU and the clvk
    // backend can drop its readback straight into the same memory.
    vk::raii::Buffer       m_sharedBuf   = nullptr;
    vk::raii::DeviceMemory m_sharedMem   = nullptr;
    void*                  m_sharedMapped = nullptr;
    vk::DeviceSize         m_sharedBytes = 0;

    // ---- AOT compute pipeline ----
    vk::raii::DescriptorSetLayout m_dsLayout   = nullptr;
    vk::raii::PipelineLayout      m_pipeLayout = nullptr;
    vk::raii::Pipeline            m_pipeline   = nullptr;
    vk::raii::DescriptorPool      m_dsPool     = nullptr;
    vk::DescriptorSet             m_dsSet      = nullptr;
    vk::raii::Buffer              m_paramsBuf  = nullptr;
    vk::raii::DeviceMemory        m_paramsMem  = nullptr;
    void*                         m_paramsMapped = nullptr;

#ifdef HAVE_OPENCL
    ClvkBackend m_clvk;
    bool        m_clvkLive = false;
#endif

    // ---- per-frame sync ----
    std::vector<vk::raii::Semaphore> m_imageAvail;
    std::vector<vk::raii::Semaphore> m_renderDone;
    int                              m_acquireIdx = 0;
    vk::raii::CommandBuffer          m_cmd   = nullptr;
    vk::raii::Fence                  m_fence = nullptr;

    std::string m_kernelSource;

    // =======================================================================
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        m_window = glfwCreateWindow(kWidth, kHeight,
            "OpenCL-on-Vulkan Forest  |  WASD+EQ move  drag=look  Shift=boost  R=reset  ESC=quit",
            nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetMouseButtonCallback(m_window, cbMouseButton);
        glfwSetCursorPosCallback(m_window, cbCursorPos);
        glfwSetKeyCallback(m_window, cbKey);
    }

    static void cbMouseButton(GLFWwindow* w, int button, int action, int) {
        auto* a = static_cast<ForestApp*>(glfwGetWindowUserPointer(w));
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            a->m_dragging = (action == GLFW_PRESS);
            glfwGetCursorPos(w, &a->m_lastMx, &a->m_lastMy);
        }
    }
    static void cbCursorPos(GLFWwindow* w, double mx, double my) {
        auto* a = static_cast<ForestApp*>(glfwGetWindowUserPointer(w));
        if (a->m_dragging) {
            a->m_yaw   -= float(mx - a->m_lastMx) * 0.0040f;
            a->m_pitch -= float(my - a->m_lastMy) * 0.0040f;
            a->m_pitch  = std::clamp(a->m_pitch, -1.5f, 1.5f);
        }
        a->m_lastMx = mx; a->m_lastMy = my;
    }
    static void cbKey(GLFWwindow* w, int key, int, int action, int) {
        if (action != GLFW_PRESS) return;
        auto* a = static_cast<ForestApp*>(glfwGetWindowUserPointer(w));
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(w, GLFW_TRUE);
        if (key == GLFW_KEY_R) { a->m_camX = 0; a->m_camY = 3.1f; a->m_camZ = 0;
                                 a->m_yaw = 0.45f; a->m_pitch = -0.40f; }
    }

    void updateCamera() {
        double now = glfwGetTime();
        float dt = std::min(float(now - m_lastTime), 0.05f);
        m_lastTime = now;
        float speed = 6.0f * dt;
        if (glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) speed *= 4.0f;
        const float fx = std::sin(m_yaw), fz = std::cos(m_yaw);
        const float rx = std::cos(m_yaw), rz = -std::sin(m_yaw);
        auto mv = [&](float dx, float dy, float dz){ m_camX += dx; m_camY += dy; m_camZ += dz; };
        if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) mv(fx*speed, 0, fz*speed);
        if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) mv(-fx*speed, 0, -fz*speed);
        if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) mv(rx*speed, 0, rz*speed);
        if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) mv(-rx*speed, 0, -rz*speed);
        if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS) mv(0, speed, 0);
        if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS) mv(0, -speed, 0);
        m_camY = std::max(m_camY, 0.3f);
    }

    Params currentParams() const {
        return Params{
            .width = int32_t(kWidth), .height = int32_t(kHeight),
            .camX = m_camX, .camY = m_camY, .camZ = m_camZ,
            .camYaw = m_yaw, .camPitch = m_pitch, .fog = 0.05f};
    }

    // =======================================================================
    bool aotAvailable() const {
        return bool(std::ifstream("shaders/forest.spv", std::ios::binary));
    }

    void selectBackend() {
        m_kernelSource = readText("05_opencl_on_vulkan.cl");
        const bool wantClvk = (m_forced == "clvk");
        const bool wantAot  = (m_forced == "aot");

#ifdef HAVE_OPENCL
        if (!wantAot && !m_kernelSource.empty() && ClvkBackend::probe(m_clvk)) {
            if (wantClvk || !aotAvailable()) {       // prefer the zero-copy AOT path by default
                m_backend = Backend::ClvkRuntime;
                std::cout << "[backend] clvk runtime: OpenCL compiled at run time, "
                             "result bridged into the Vulkan buffer\n";
                return;
            }
        }
#endif
        if (wantClvk)
            throw std::runtime_error("--backend=clvk requested but no clvk platform / kernel source found.");
        if (!aotAvailable())
            throw std::runtime_error(
                "No compute backend: shaders/forest.spv missing AND no clvk platform.\n"
                "Install clspv (preferred) or clvk, then rebuild. See install_dependencies_*.");
        m_backend = Backend::ClspvAot;
        std::cout << "[backend] clspv AOT: the OpenCL kernel runs as the Vulkan compute shader "
                     "(zero-copy)\n";
    }

    // =======================================================================
    void createInstance() {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "OpenCL-on-Vulkan Forest",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine", .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion13};
        std::vector<const char*> layers;
        if (kEnableValidation) layers.assign(kValidationLayers.begin(), kValidationLayers.end());
        uint32_t gc = 0;
        auto gexts = glfwGetRequiredInstanceExtensions(&gc);
        std::vector<const char*> exts(gexts, gexts + gc);
        if (kEnableValidation) exts.push_back(vk::EXTDebugUtilsExtensionName);
        m_instance = vk::raii::Instance(m_ctx, vk::InstanceCreateInfo{
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = uint32_t(layers.size()), .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = uint32_t(exts.size()), .ppEnabledExtensionNames = exts.data()});
    }

    void setupDebug() {
        if (!kEnableValidation) return;
        m_debug = m_instance.createDebugUtilsMessengerEXT(vk::DebugUtilsMessengerCreateInfoEXT{
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                               vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            .pfnUserCallback = &debugCallback});
    }

    void createSurface() {
        VkSurfaceKHR raw;
        if (glfwCreateWindowSurface(*m_instance, m_window, nullptr, &raw) != VK_SUCCESS)
            throw std::runtime_error("glfwCreateWindowSurface failed");
        m_surface = vk::raii::SurfaceKHR(m_instance, raw);
    }

    void pickPhysicalDevice() {
        auto typeScore = [](vk::PhysicalDeviceType t) {
            switch (t) {
                case vk::PhysicalDeviceType::eDiscreteGpu:   return 4;
                case vk::PhysicalDeviceType::eIntegratedGpu: return 3;
                default:                                     return 1;
            }
        };
        int best = 0;
        for (auto& pd : m_instance.enumeratePhysicalDevices()) {
            auto qfps = pd.getQueueFamilyProperties();
            uint32_t qf = ~0u;
            for (uint32_t i = 0; i < qfps.size(); ++i)
                if ((qfps[i].queueFlags & vk::QueueFlagBits::eCompute) &&
                    pd.getSurfaceSupportKHR(i, *m_surface)) { qf = i; break; }
            if (qf == ~0u) continue;
            auto de = pd.enumerateDeviceExtensionProperties();
            if (!std::any_of(de.begin(), de.end(), [](auto& e){
                    return strcmp(e.extensionName, vk::KHRSwapchainExtensionName) == 0; }))
                continue;
            int s = typeScore(pd.getProperties().deviceType);
            if (s > best) { best = s; m_phys = pd; m_queueFamily = qf; }
        }
        if (!*m_phys) throw std::runtime_error("No suitable GPU found");
        std::cout << "[vulkan]  device: " << m_phys.getProperties().deviceName.data() << '\n';
    }

    void createLogicalDevice() {
        std::vector<const char*> exts = {vk::KHRSwapchainExtensionName};
        // variablePointers* are required by clspv-generated SPIR-V (AOT path).
        vk::StructureChain<vk::PhysicalDeviceFeatures2,
                           vk::PhysicalDeviceVulkan11Features,
                           vk::PhysicalDeviceVulkan13Features> chain = {
            {},
            {.variablePointersStorageBuffer = true, .variablePointers = true},
            {.synchronization2 = true}};
        float prio = 1.0f;
        vk::DeviceQueueCreateInfo qci{
            .queueFamilyIndex = m_queueFamily, .queueCount = 1, .pQueuePriorities = &prio};
        m_device = vk::raii::Device(m_phys, vk::DeviceCreateInfo{
            .pNext = &chain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
            .enabledExtensionCount = uint32_t(exts.size()), .ppEnabledExtensionNames = exts.data()});
        m_queue = vk::raii::Queue(m_device, m_queueFamily, 0);
    }

    void createCommandPool() {
        m_cmdPool = vk::raii::CommandPool(m_device, vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = m_queueFamily});
    }

    void createSwapchain() {
        auto caps = m_phys.getSurfaceCapabilitiesKHR(*m_surface);
        m_extent = (caps.currentExtent.width != UINT32_MAX)
            ? caps.currentExtent : vk::Extent2D{kWidth, kHeight};
        auto fmts = m_phys.getSurfaceFormatsKHR(*m_surface);
        m_swapFormat = fmts[0];
        // Prefer B8G8R8A8Unorm with blit-dst support; fall back to it without
        for (auto& f : fmts)
            if (f.format == vk::Format::eB8G8R8A8Unorm &&
                f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
                !!(m_phys.getFormatProperties(f.format).optimalTilingFeatures &
                   vk::FormatFeatureFlagBits::eBlitDst)) { m_swapFormat = f; break; }
        if (m_swapFormat.format != vk::Format::eB8G8R8A8Unorm)
            for (auto& f : fmts)
                if (f.format == vk::Format::eB8G8R8A8Unorm &&
                    f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) { m_swapFormat = f; break; }
        auto modes = m_phys.getSurfacePresentModesKHR(*m_surface);
        auto mode = vk::PresentModeKHR::eFifo;
        for (auto m : modes) if (m == vk::PresentModeKHR::eMailbox) mode = m;
        uint32_t imgCount = std::max(3u, caps.minImageCount);
        if (caps.maxImageCount > 0) imgCount = std::min(imgCount, caps.maxImageCount);
        m_swapchain = vk::raii::SwapchainKHR(m_device, vk::SwapchainCreateInfoKHR{
            .surface = *m_surface, .minImageCount = imgCount,
            .imageFormat = m_swapFormat.format, .imageColorSpace = m_swapFormat.colorSpace,
            .imageExtent = m_extent, .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eTransferDst,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = caps.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = mode, .clipped = true});
        m_swapImages = m_swapchain.getImages();
    }

    void createIntermediateImage() {
        m_interImg = vk::raii::Image(m_device, vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {m_extent.width, m_extent.height, 1}, .mipLevels = 1, .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1, .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive, .initialLayout = vk::ImageLayout::eUndefined});
        auto req = m_interImg.getMemoryRequirements();
        m_interMem = vk::raii::DeviceMemory(m_device, vk::MemoryAllocateInfo{
            .allocationSize = req.size,
            .memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                              vk::MemoryPropertyFlagBits::eDeviceLocal)});
        m_interImg.bindMemory(*m_interMem, 0);
    }

    void createSharedBuffer() {
        m_sharedBytes = vk::DeviceSize(m_extent.width) * m_extent.height * sizeof(uint32_t);
        m_sharedBuf = vk::raii::Buffer(m_device, vk::BufferCreateInfo{
            .size = m_sharedBytes,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive});
        auto req = m_sharedBuf.getMemoryRequirements();
        m_sharedMem = vk::raii::DeviceMemory(m_device, vk::MemoryAllocateInfo{
            .allocationSize = req.size,
            .memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)});
        m_sharedBuf.bindMemory(*m_sharedMem, 0);
        m_sharedMapped = m_sharedMem.mapMemory(0, m_sharedBytes);
    }

    void initComputeBackend() {
        if (m_backend == Backend::ClspvAot) { initAot(); return; }
#ifdef HAVE_OPENCL
        if (!m_clvk.init(m_kernelSource, size_t(m_sharedBytes))) {
            m_clvk.destroy();
            std::cerr << "[backend] clvk init failed; falling back to clspv AOT.\n";
            if (!aotAvailable())
                throw std::runtime_error("clvk failed and no forest.spv fallback present.");
            m_backend = Backend::ClspvAot;
            initAot();
            return;
        }
        m_clvkLive = true;
#endif
    }

    void initAot() {
        m_paramsBuf = vk::raii::Buffer(m_device, vk::BufferCreateInfo{
            .size = sizeof(Params), .usage = vk::BufferUsageFlagBits::eStorageBuffer,
            .sharingMode = vk::SharingMode::eExclusive});
        auto preq = m_paramsBuf.getMemoryRequirements();
        m_paramsMem = vk::raii::DeviceMemory(m_device, vk::MemoryAllocateInfo{
            .allocationSize = preq.size,
            .memoryTypeIndex = findMemoryType(preq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)});
        m_paramsBuf.bindMemory(*m_paramsMem, 0);
        m_paramsMapped = m_paramsMem.mapMemory(0, sizeof(Params));

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {.binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer,
             .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}}};
        m_dsLayout = vk::raii::DescriptorSetLayout(m_device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = 2, .pBindings = bindings.data()});
        m_pipeLayout = vk::raii::PipelineLayout(m_device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1, .pSetLayouts = &*m_dsLayout});

        auto code = readBinary("shaders/forest.spv");
        vk::raii::ShaderModule sm(m_device, vk::ShaderModuleCreateInfo{
            .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data())});
        vk::PipelineShaderStageCreateInfo stage{
            .stage = vk::ShaderStageFlagBits::eCompute, .module = *sm, .pName = "render"};
        m_pipeline = vk::raii::Pipeline(m_device, nullptr,
            vk::ComputePipelineCreateInfo{.stage = stage, .layout = *m_pipeLayout});

        std::array<vk::DescriptorPoolSize, 1> ps{{
            {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 2}}};
        m_dsPool = vk::raii::DescriptorPool(m_device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = ps.data()});
        m_dsSet = vk::raii::DescriptorSets(m_device, vk::DescriptorSetAllocateInfo{
            .descriptorPool = *m_dsPool, .descriptorSetCount = 1,
            .pSetLayouts = &*m_dsLayout})[0].release();

        vk::DescriptorBufferInfo pInfo{.buffer = *m_paramsBuf, .offset = 0, .range = sizeof(Params)};
        vk::DescriptorBufferInfo oInfo{.buffer = *m_sharedBuf, .offset = 0, .range = m_sharedBytes};
        std::array<vk::WriteDescriptorSet, 2> w{{
            {.dstSet = m_dsSet, .dstBinding = 0, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &pInfo},
            {.dstSet = m_dsSet, .dstBinding = 1, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &oInfo}}};
        m_device.updateDescriptorSets(w, {});
    }

    void createSyncAndCmd() {
        for (int i = 0; i < kAcquireSemas; ++i)
            m_imageAvail.emplace_back(m_device, vk::SemaphoreCreateInfo{});
        for (size_t i = 0; i < m_swapImages.size(); ++i)
            m_renderDone.emplace_back(m_device, vk::SemaphoreCreateInfo{});
        m_cmd = std::move(vk::raii::CommandBuffers(m_device, vk::CommandBufferAllocateInfo{
            .commandPool = *m_cmdPool, .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1}).front());
        m_fence = vk::raii::Fence(m_device, vk::FenceCreateInfo{
            .flags = vk::FenceCreateFlagBits::eSignaled});
    }

    // =======================================================================
    void mainLoop() {
        m_lastTime = glfwGetTime();
        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            updateCamera();
            drawFrame();
        }
        m_device.waitIdle();
    }

    void drawFrame() {
        auto _ = m_device.waitForFences(*m_fence, vk::True, UINT64_MAX);
        m_device.resetFences(*m_fence);

        const Params params = currentParams();
        if (m_backend == Backend::ClspvAot)
            std::memcpy(m_paramsMapped, &params, sizeof(Params));
#ifdef HAVE_OPENCL
        if (m_backend == Backend::ClvkRuntime)
            m_clvk.compute(params, m_sharedMapped, size_t(m_sharedBytes));  // fills the shared buffer
#endif

        auto& acq = m_imageAvail[m_acquireIdx];
        m_acquireIdx = (m_acquireIdx + 1) % kAcquireSemas;
        uint32_t imageIndex;
        {
            auto [res, idx] = m_swapchain.acquireNextImage(UINT64_MAX, *acq, nullptr);
            if (res == vk::Result::eErrorOutOfDateKHR) { m_device.waitIdle(); return; }
            imageIndex = idx;
        }
        recordCommands(imageIndex);

        auto& rd = m_renderDone[imageIndex];
        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;
        m_queue.submit(vk::SubmitInfo{
            .waitSemaphoreCount = 1, .pWaitSemaphores = &*acq, .pWaitDstStageMask = &waitStage,
            .commandBufferCount = 1, .pCommandBuffers = &*m_cmd,
            .signalSemaphoreCount = 1, .pSignalSemaphores = &*rd}, *m_fence);

        auto pres = m_queue.presentKHR(vk::PresentInfoKHR{
            .waitSemaphoreCount = 1, .pWaitSemaphores = &*rd,
            .swapchainCount = 1, .pSwapchains = &*m_swapchain, .pImageIndices = &imageIndex});
        if (pres == vk::Result::eErrorOutOfDateKHR) m_device.waitIdle();
    }

    void recordCommands(uint32_t imageIndex) {
        auto& cb = m_cmd;
        cb.reset();
        cb.begin({});

        if (m_backend == Backend::ClspvAot) {
            cb.bindPipeline(vk::PipelineBindPoint::eCompute, *m_pipeline);
            cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_pipeLayout, 0, {m_dsSet}, {});
            cb.dispatch((m_extent.width + 15) / 16, (m_extent.height + 15) / 16, 1);
        }

        // Make the buffer writes visible to the transfer read. AOT writes via the
        // compute shader; clvk writes via the host (clEnqueueReadBuffer above).
        const bool aot = (m_backend == Backend::ClspvAot);
        vk::BufferMemoryBarrier2 bufBar{
            .srcStageMask  = aot ? vk::PipelineStageFlagBits2::eComputeShader
                                 : vk::PipelineStageFlagBits2::eHost,
            .srcAccessMask = aot ? vk::AccessFlagBits2::eShaderWrite
                                 : vk::AccessFlagBits2::eHostWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = *m_sharedBuf, .offset = 0, .size = m_sharedBytes};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &bufBar,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = imgBar(*m_interImg,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite)});

        vk::BufferImageCopy2 copy{
            .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageOffset = {0, 0, 0}, .imageExtent = {m_extent.width, m_extent.height, 1}};
        cb.copyBufferToImage2(vk::CopyBufferToImageInfo2{
            .srcBuffer = *m_sharedBuf, .dstImage = *m_interImg,
            .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
            .regionCount = 1, .pRegions = &copy});

        std::array<vk::ImageMemoryBarrier2, 2> pre{
            *imgBar(*m_interImg, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead),
            *imgBar(m_swapImages[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite)};
        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = 2, .pImageMemoryBarriers = pre.data()});

        vk::ImageSubresourceLayers sub{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        vk::Offset3D ext{int32_t(m_extent.width), int32_t(m_extent.height), 1};
        vk::ImageBlit2 region{
            .srcSubresource = sub, .srcOffsets = std::array<vk::Offset3D, 2>{vk::Offset3D{0,0,0}, ext},
            .dstSubresource = sub, .dstOffsets = std::array<vk::Offset3D, 2>{vk::Offset3D{0,0,0}, ext}};
        cb.blitImage2(vk::BlitImageInfo2{
            .srcImage = *m_interImg, .srcImageLayout = vk::ImageLayout::eTransferSrcOptimal,
            .dstImage = m_swapImages[imageIndex], .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
            .regionCount = 1, .pRegions = &region, .filter = vk::Filter::eNearest});

        cb.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = imgBar(m_swapImages[imageIndex],
                vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eBottomOfPipe, vk::AccessFlagBits2::eNone)});
        cb.end();
    }

    vk::ImageMemoryBarrier2* imgBar(vk::Image img, vk::ImageLayout o, vk::ImageLayout n,
        vk::PipelineStageFlags2 ss, vk::AccessFlags2 sa,
        vk::PipelineStageFlags2 ds, vk::AccessFlags2 da) {
        static thread_local std::array<vk::ImageMemoryBarrier2, 8> ring;
        static thread_local int idx = 0;
        auto& b = ring[idx]; idx = (idx + 1) % int(ring.size());
        b = vk::ImageMemoryBarrier2{
            .srcStageMask = ss, .srcAccessMask = sa, .dstStageMask = ds, .dstAccessMask = da,
            .oldLayout = o, .newLayout = n,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = img, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        return &b;
    }

    void cleanup() {
#ifdef HAVE_OPENCL
        if (m_clvkLive) m_clvk.destroy();
#endif
        m_device.waitIdle();
        m_renderDone.clear(); m_imageAvail.clear();
        m_fence = nullptr; m_cmd = nullptr;
        m_dsPool = nullptr; m_pipeline = nullptr; m_pipeLayout = nullptr; m_dsLayout = nullptr;
        m_paramsBuf = nullptr; m_paramsMem = nullptr;
        m_sharedBuf = nullptr; m_sharedMem = nullptr;
        m_interImg = nullptr; m_interMem = nullptr;
        m_cmdPool = nullptr; m_swapchain = nullptr; m_queue = nullptr;
        m_device = nullptr; m_surface = nullptr; m_debug = nullptr; m_instance = nullptr;
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    uint32_t findMemoryType(uint32_t bits, vk::MemoryPropertyFlags props) const {
        auto mp = m_phys.getMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
        throw std::runtime_error("no suitable memory type");
    }
    static std::string readText(const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return {};
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    static std::vector<char> readBinary(const std::string& p) {
        std::ifstream f(p, std::ios::ate | std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + p);
        std::vector<char> b(f.tellg());
        f.seekg(0); f.read(b.data(), std::streamsize(b.size()));
        return b;
    }
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT sev, vk::DebugUtilsMessageTypeFlagsEXT,
        const vk::DebugUtilsMessengerCallbackDataEXT* d, void*) {
        if (sev >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
            std::cerr << "[VK] " << d->pMessage << '\n';
        return vk::False;
    }
};

// ---------------------------------------------------------------------------
#ifndef ANDROID_BUILD
int main(int argc, char** argv) {
    std::cout << "=====================================================\n"
                 " Chapter 05 — Vulkan rendering powered by OpenCL/clvk\n"
                 "=====================================================\n";
    std::string forced;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--backend=", 0) == 0) forced = a.substr(10);
    }
    try {
        ForestApp app(forced);
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
#endif // ANDROID_BUILD

#ifdef ANDROID_BUILD
extern "C" void chapter05_run() {
    // OpenCL/clspv path not available on Android without additional setup.
    // ForestApp will fall back to a static frame if no .spv is found.
    try { ForestApp app(""); app.run(); }
    catch (const std::exception& e) { __android_log_print(ANDROID_LOG_ERROR, "ComputeCh05", "%s", e.what()); }
}
#endif
