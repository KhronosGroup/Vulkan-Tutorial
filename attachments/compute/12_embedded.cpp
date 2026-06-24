// Chapter 12 – Embedded Compute: Parallel FABRIK IK Solver
//
// PURPOSE
//   Demonstrates running a purely headless Vulkan compute dispatch on
//   embedded hardware.  The sample solves 3-bone Inverse Kinematics (IK)
//   for N robot arm waypoints simultaneously — the kind of workload found
//   in industrial robot controllers, surgical arm systems, and real-time
//   animation engines.  IK applies to any articulated kinematic chain; it
//   is not related to drone flight, which uses path-planning algorithms
//   such as RRT or A* instead.
//
// ALGORITHM  — FABRIK (Forward And Backward Reaching IK)
//   Each GPU thread independently solves one arm configuration:
//     1. Forward pass: pull the end-effector onto the target, then
//        reproject each bone to restore its original length.
//     2. Backward pass: re-pin the root joint, then re-reproject
//        toward the end-effector.
//   Repeat until |end - target| < tolerance or maxIter is reached.
//   Convergence is typically 3–8 iterations for most pose configurations.
//
// ENTRY POINT
//   int main(int argc, char** argv)
//   Reads an optional "count" argument (default 4096).
//   Prints the report to stdout.
//
// PORTABILITY
//   • Requires Vulkan 1.3 — available on Raspberry Pi 5 (Mesa 24.0+),
//     Raspberry Pi 4 (Mesa 23.3+ / Vulkan 1.2 minimum), and NVIDIA Jetson
//     (JetPack ≥ 5.1).  Targeting 1.3 grants core synchronization2 without
//     separate extension loading.
//   • No GLFW / no display dependency.  Purely headless.
//   • Demonstrates UMA buffer detection: on unified-memory embedded devices
//     the result buffer is allocated in a heap that is both DEVICE_LOCAL and
//     HOST_VISIBLE, eliminating the staging copy.

// ---------------------------------------------------------------------------
// Includes
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#  include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define LOG_I(...) (void)(printf(__VA_ARGS__), putchar('\n'))
#define LOG_E(...) (void)(fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* pData,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        LOG_E("[VK] %s", pData->pMessage);
    return VK_FALSE;
}

// ---------------------------------------------------------------------------
// CPU-side struct layouts — must be byte-identical to the Slang shader structs
// ---------------------------------------------------------------------------
struct alignas(16) IKJob
{
    float    target[4];   // xyz = target position, w unused
    float    base[4];     // xyz = root joint position, w unused
    float    reach[4];    // xyz = bone lengths [r0, r1, r2], w unused
    uint32_t maxIter;
    float    tolerance;
    uint32_t pad[2];
};
static_assert(sizeof(IKJob) == 64, "IKJob layout mismatch");

struct alignas(16) IKSolution
{
    float    joint0[4];        // xyz = root (after solve)
    float    joint1[4];        // xyz = joint 1 solved position
    float    joint2[4];        // xyz = joint 2 solved position
    float    endEffector[4];   // xyz = end-effector solved position
    uint32_t iterations;
    float    residual;
    uint32_t reachable;        // 1 if residual < tolerance
    uint32_t _pad;
};
static_assert(sizeof(IKSolution) == 80, "IKSolution layout mismatch");

// Push constant sent to the shader
struct PushConst
{
    uint32_t count;
};

// ---------------------------------------------------------------------------
// Shader loading
// ---------------------------------------------------------------------------
static std::vector<char> loadShaderBytes(const char* path)
{
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::string("Cannot open shader: ") + path);
    std::vector<char> buf(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    return buf;
}

// ---------------------------------------------------------------------------
// EmbeddedContext — minimal self-contained Vulkan 1.3 headless context
//
// Vulkan 1.3 is supported on Raspberry Pi 5 (Mesa 24.0+) and NVIDIA Jetson
// (JetPack >= 5.1).  Targeting 1.3 gives core synchronization2 without needing
// to load extensions separately.  variablePointers (Vulkan 1.1 optional) is
// also requested when the device supports it, as it is required by
// clspv-generated SPIR-V.
// ---------------------------------------------------------------------------
struct EmbeddedContext
{
    vk::raii::Context                context;
    vk::raii::Instance               instance       = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::PhysicalDevice         physDev        = nullptr;
    vk::raii::Device                 device         = nullptr;
    uint32_t                         computeFamily  = ~0u;
    vk::raii::Queue                  computeQueue   = nullptr;
    vk::raii::CommandPool            cmdPool        = nullptr;
    bool                             isUMA          = false;
    uint32_t                         umaMemType     = ~0u;

    void init()
    {
        // ---- Instance ----
        vk::ApplicationInfo appInfo{
            .pApplicationName   = "IKSolverEmbedded",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = vk::ApiVersion13};

        std::vector<const char*> layers;
        std::vector<const char*> exts;
        if (kEnableValidation) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        instance = vk::raii::Instance(context, vk::InstanceCreateInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames     = layers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(exts.size()),
            .ppEnabledExtensionNames = exts.data()});

        if (kEnableValidation) {
            debugMessenger = instance.createDebugUtilsMessengerEXT(
                vk::DebugUtilsMessengerCreateInfoEXT{
                    .messageSeverity =
                        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                    .messageType =
                        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral    |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                    .pfnUserCallback = reinterpret_cast<vk::PFN_DebugUtilsMessengerCallbackEXT>(&debugCallback)});
        }

        // ---- Physical device selection ----
        // Prefer a family that has VK_QUEUE_COMPUTE_BIT but NOT VK_QUEUE_GRAPHICS_BIT
        // (dedicated compute queue, reduces scheduling contention on embedded parts).
        for (auto& pd : instance.enumeratePhysicalDevices()) {
            auto props = pd.getQueueFamilyProperties();
            uint32_t best = ~0u;
            for (uint32_t f = 0; f < props.size(); ++f) {
                bool hasCompute  = !!(props[f].queueFlags & vk::QueueFlagBits::eCompute);
                bool hasGraphics = !!(props[f].queueFlags & vk::QueueFlagBits::eGraphics);
                if (hasCompute && !hasGraphics) { best = f; break; }
                if (hasCompute && best == ~0u)    best = f;
            }
            if (best != ~0u) {
                physDev       = pd;
                computeFamily = best;
                break;
            }
        }
        if (computeFamily == ~0u)
            throw std::runtime_error("No compute queue family found");

        // Print key limits — demonstrates the embedded-limit-checking practice
        // from the tutorial text.
        auto limits = physDev.getProperties().limits;
        LOG_I("maxComputeWorkGroupInvocations : %u", limits.maxComputeWorkGroupInvocations);
        LOG_I("maxComputeSharedMemorySize     : %u bytes", limits.maxComputeSharedMemorySize);

        // ---- Query supported features ----
        // variablePointers are needed when loading clspv-generated SPIR-V
        // (Chapter 5 portability path).  synchronization2 is REQUIRED in
        // Vulkan 1.3, so we can always enable it when targeting 1.3.
        auto featChain = physDev.getFeatures2<vk::PhysicalDeviceFeatures2,
                                              vk::PhysicalDeviceVulkan11Features,
                                              vk::PhysicalDeviceVulkan13Features>();
        auto& v11Query = featChain.get<vk::PhysicalDeviceVulkan11Features>();
        bool haveVP = v11Query.variablePointers && v11Query.variablePointersStorageBuffer;

        // ---- Logical device ----
        // synchronization2 is core in Vulkan 1.3 — always available and always enabled.
        // variablePointers are optional; enable only when the device reports them.
        float prio = 1.0f;
        vk::DeviceQueueCreateInfo qCI{
            .queueFamilyIndex = computeFamily,
            .queueCount       = 1,
            .pQueuePriorities = &prio};

        vk::PhysicalDeviceVulkan13Features v13Enable{
            .synchronization2 = true};   // core in 1.3, always present
        vk::PhysicalDeviceVulkan11Features v11Enable{
            .pNext                         = &v13Enable,
            .variablePointersStorageBuffer = haveVP,
            .variablePointers              = haveVP};

        device = vk::raii::Device(physDev, vk::DeviceCreateInfo{
            .pNext               = &v11Enable,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos   = &qCI});

        computeQueue = vk::raii::Queue(device, computeFamily, 0);

        vk::CommandPoolCreateInfo poolCI{
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = computeFamily};
        cmdPool = vk::raii::CommandPool(device, poolCI);

        // ---- Detect UMA (Unified Memory Architecture) ----
        // On embedded targets the GPU and CPU share physical RAM.  A UMA heap is
        // both DEVICE_LOCAL and HOST_VISIBLE — we can skip the staging copy for
        // result readback.
        auto memProps = physDev.getMemoryProperties();
        for (uint32_t m = 0; m < memProps.memoryTypeCount; ++m) {
            auto flags = memProps.memoryTypes[m].propertyFlags;
            using F = vk::MemoryPropertyFlagBits;
            if ((flags & F::eDeviceLocal) && (flags & F::eHostVisible) &&
                (flags & F::eHostCoherent)) {
                isUMA       = true;
                umaMemType  = m;
                LOG_I("UMA device detected — result buffer will skip staging copy");
                break;
            }
        }
        if (!isUMA) LOG_I("Discrete device — result readback will use a staging buffer");
    }

    // -----------------------------------------------------------------------
    // Memory helpers
    // -----------------------------------------------------------------------
    [[nodiscard]] uint32_t findMemType(uint32_t bits, vk::MemoryPropertyFlags want) const
    {
        auto mp = physDev.getMemoryProperties();
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
                return i;
        throw std::runtime_error("No suitable memory type");
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags props,
                      vk::raii::Buffer& buf, vk::raii::DeviceMemory& mem) const
    {
        buf = vk::raii::Buffer(device, vk::BufferCreateInfo{
            .size        = size,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive});
        auto req = buf.getMemoryRequirements();
        mem = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
            .allocationSize  = req.size,
            .memoryTypeIndex = findMemType(req.memoryTypeBits, props)});
        buf.bindMemory(mem, 0);
    }

    // -----------------------------------------------------------------------
    // One-shot command buffer
    // -----------------------------------------------------------------------
    [[nodiscard]] vk::raii::CommandBuffer beginOneShot() const
    {
        auto cb = std::move(vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
            .commandPool        = *cmdPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1}).front());
        cb.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return cb;
    }

    void endOneShot(const vk::raii::CommandBuffer& cb, const vk::raii::Fence& fence) const
    {
        cb.end();
        vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
        computeQueue.submit(si, *fence);
    }
};

// ---------------------------------------------------------------------------
// IKSolverDemo
// ---------------------------------------------------------------------------
class IKSolverDemo
{
public:
    void run(uint32_t count, const char* shaderPath, std::string& outReport)
    {
        using namespace std::chrono;
        auto t0 = high_resolution_clock::now();

        // ---- 1. Initialise Vulkan ----
        EmbeddedContext ctx;
        ctx.init();

        // ---- 2. Generate IK jobs ----
        std::vector<IKJob> jobs(count);
        const float boneLen = 0.25f;          // each of the 3 bones = 25 cm
        const float reachIn  = 0.55f;         // inside total reach (0.75 m)
        const float reachOut = 0.85f;         // just outside total reach

        // Distribute targets over a Fibonacci sphere to get even coverage.
        const float phi = 3.14159265f * (3.0f - std::sqrtf(5.0f)); // golden angle
        for (uint32_t n = 0; n < count; ++n) {
            float y   = 1.0f - (static_cast<float>(n) / static_cast<float>(count - 1)) * 2.0f;
            float r   = std::sqrtf(std::max(0.0f, 1.0f - y * y));
            float ang = phi * static_cast<float>(n);
            float x   = std::cosf(ang) * r;
            float z   = std::sinf(ang) * r;
            // Every 8th target is placed outside reach — tests the stretch branch
            float dist = ((n % 8) == 0) ? reachOut : reachIn;

            jobs[n].target[0]  = x * dist;
            jobs[n].target[1]  = y * dist;
            jobs[n].target[2]  = z * dist;
            jobs[n].target[3]  = 0.0f;
            jobs[n].base[0] = jobs[n].base[1] = jobs[n].base[2] = jobs[n].base[3] = 0.0f;
            jobs[n].reach[0]   = boneLen;
            jobs[n].reach[1]   = boneLen;
            jobs[n].reach[2]   = boneLen;
            jobs[n].reach[3]   = 0.0f;
            jobs[n].maxIter    = 20;
            jobs[n].tolerance  = 0.001f;
            jobs[n].pad[0] = jobs[n].pad[1] = 0;
        }

        vk::DeviceSize jobBytes    = sizeof(IKJob)      * count;
        vk::DeviceSize solBytes    = sizeof(IKSolution) * count;

        using PF = vk::MemoryPropertyFlagBits;
        using BU = vk::BufferUsageFlagBits;

        // ---- 3. Upload jobs buffer ----
        // Jobs buffer: host uploads once, GPU reads once — DEVICE_LOCAL if possible.
        vk::raii::Buffer     jobBuf  = nullptr, jobStage = nullptr;
        vk::raii::DeviceMemory jobMem = nullptr, stageMem = nullptr;

        ctx.createBuffer(jobBytes,
            BU::eStorageBuffer | BU::eTransferDst,
            PF::eDeviceLocal,
            jobBuf, jobMem);

        // Upload via staging buffer (always needed for input on discrete GPU)
        ctx.createBuffer(jobBytes, BU::eTransferSrc,
            PF::eHostVisible | PF::eHostCoherent,
            jobStage, stageMem);
        {
            void* p = stageMem.mapMemory(0, jobBytes);
            std::memcpy(p, jobs.data(), static_cast<size_t>(jobBytes));
            stageMem.unmapMemory();
        }
        {
            auto cb = ctx.beginOneShot();
            cb.copyBuffer(*jobStage, *jobBuf, vk::BufferCopy{.size = jobBytes});
            cb.end();
            vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
            ctx.computeQueue.submit(si);
            ctx.computeQueue.waitIdle();
        }

        // ---- 4. Result buffer — skip staging on UMA ----
        vk::raii::Buffer      solBuf   = nullptr, solStage   = nullptr;
        vk::raii::DeviceMemory solMem  = nullptr, solStageMem = nullptr;

        if (ctx.isUMA) {
            // UMA: DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT — no staging needed.
            ctx.createBuffer(solBytes,
                BU::eStorageBuffer,
                PF::eDeviceLocal | PF::eHostVisible | PF::eHostCoherent,
                solBuf, solMem);
        } else {
            // Discrete: device-local for GPU writes, host-visible for readback.
            ctx.createBuffer(solBytes, BU::eStorageBuffer | BU::eTransferSrc,
                PF::eDeviceLocal,
                solBuf, solMem);
            ctx.createBuffer(solBytes, BU::eTransferDst,
                PF::eHostVisible | PF::eHostCoherent,
                solStage, solStageMem);
        }

        // ---- 5. Descriptor set layout ----
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {.binding=0, .descriptorType=vk::DescriptorType::eStorageBuffer,
             .descriptorCount=1, .stageFlags=vk::ShaderStageFlagBits::eCompute},
            {.binding=1, .descriptorType=vk::DescriptorType::eStorageBuffer,
             .descriptorCount=1, .stageFlags=vk::ShaderStageFlagBits::eCompute}}};

        auto dsLayout = vk::raii::DescriptorSetLayout(ctx.device,
            vk::DescriptorSetLayoutCreateInfo{
                .bindingCount = static_cast<uint32_t>(bindings.size()),
                .pBindings    = bindings.data()});

        // ---- 6. Pipeline layout (push constant for count) ----
        vk::PushConstantRange pcRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset     = 0,
            .size       = sizeof(PushConst)};

        auto pipeLayout = vk::raii::PipelineLayout(ctx.device,
            vk::PipelineLayoutCreateInfo{
                .setLayoutCount         = 1,
                .pSetLayouts            = &*dsLayout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges    = &pcRange});

        // ---- 7. Load SPIR-V and create compute pipeline ----
        auto spv = loadShaderBytes(shaderPath);

        auto shaderModule = vk::raii::ShaderModule(ctx.device,
            vk::ShaderModuleCreateInfo{
                .codeSize = spv.size(),
                .pCode    = reinterpret_cast<const uint32_t*>(spv.data())});

        vk::PipelineShaderStageCreateInfo stageCI{
            .stage  = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName  = "ikMain"};

        auto pipelines = ctx.device.createComputePipelines(nullptr,
            vk::ComputePipelineCreateInfo{
                .stage  = stageCI,
                .layout = *pipeLayout});
        auto& pipeline = pipelines.front();

        // ---- 8. Descriptor pool and set ----
        vk::DescriptorPoolSize poolSize{
            .type            = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 2};

        auto descPool = vk::raii::DescriptorPool(ctx.device,
            vk::DescriptorPoolCreateInfo{
                .maxSets       = 1,
                .poolSizeCount = 1,
                .pPoolSizes    = &poolSize});

        auto descSets = vk::raii::DescriptorSets(ctx.device,
            vk::DescriptorSetAllocateInfo{
                .descriptorPool     = *descPool,
                .descriptorSetCount = 1,
                .pSetLayouts        = &*dsLayout});
        auto& ds = descSets.front();

        vk::DescriptorBufferInfo jobDBI{ .buffer=*jobBuf, .offset=0, .range=jobBytes };
        vk::DescriptorBufferInfo solDBI{ .buffer=*solBuf, .offset=0, .range=solBytes };
        std::array<vk::WriteDescriptorSet, 2> writes{{
            {.dstSet=*ds, .dstBinding=0, .descriptorCount=1,
             .descriptorType=vk::DescriptorType::eStorageBuffer, .pBufferInfo=&jobDBI},
            {.dstSet=*ds, .dstBinding=1, .descriptorCount=1,
             .descriptorType=vk::DescriptorType::eStorageBuffer, .pBufferInfo=&solDBI}}};
        ctx.device.updateDescriptorSets(writes, {});

        // ---- 9. Dispatch ----
        auto fence = vk::raii::Fence(ctx.device,
            vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        ctx.device.resetFences(*fence);

        auto cb = ctx.beginOneShot();
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipeLayout, 0, *ds, {});

        PushConst pc{count};
        cb.pushConstants<PushConst>(*pipeLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);

        uint32_t groups = (count + 63u) / 64u;
        cb.dispatch(groups, 1, 1);

        // Barrier (sync2, core in Vulkan 1.3): make shader writes visible to
        // the host on UMA, or to the transfer engine on discrete hardware.
        vk::BufferMemoryBarrier2 barrier2{
            .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask        = ctx.isUMA
                                    ? vk::PipelineStageFlagBits2::eHost
                                    : vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask       = ctx.isUMA
                                    ? vk::AccessFlagBits2::eHostRead
                                    : vk::AccessFlagBits2::eTransferRead,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer              = *solBuf,
            .size                = VK_WHOLE_SIZE};
        vk::DependencyInfo dep{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers    = &barrier2};
        cb.pipelineBarrier2(dep);

        // On discrete GPUs: copy result to the host-visible staging buffer
        if (!ctx.isUMA)
            cb.copyBuffer(*solBuf, *solStage, vk::BufferCopy{.size = solBytes});

        ctx.endOneShot(cb, fence);
        (void)ctx.device.waitForFences(*fence, VK_TRUE, UINT64_MAX);

        auto t1 = high_resolution_clock::now();
        double elapsedMs = duration<double, std::milli>(t1 - t0).count();

        // ---- 10. Read back solutions ----
        std::vector<IKSolution> solutions(count);
        {
            auto& readMem = ctx.isUMA ? solMem : solStageMem;
            void* p = readMem.mapMemory(0, solBytes);
            // If not HOST_COHERENT, invalidate before reading (no-op on coherent memory).
            vk::MappedMemoryRange mmr{
                .memory = *readMem,
                .offset = 0,
                .size   = VK_WHOLE_SIZE};
            (void)ctx.device.invalidateMappedMemoryRanges(mmr);
            std::memcpy(solutions.data(), p, static_cast<size_t>(solBytes));
            readMem.unmapMemory();
        }

        // ---- 11. Build report ----
        uint32_t reachable = 0;
        uint64_t totalIter = 0;
        float    maxResid  = 0.0f;
        for (const auto& s : solutions) {
            reachable += s.reachable;
            totalIter += s.iterations;
            maxResid   = std::max(maxResid, s.residual);
        }

        std::ostringstream oss;
        oss << "=== Vulkan Compute IK Solver — " << count << " arms ===\n\n";
        oss << "First 5 solutions:\n";
        for (uint32_t n = 0; n < std::min(count, 5u); ++n) {
            const auto& s = solutions[n];
            oss << "  [" << n << "]  target=(" << jobs[n].target[0] << ", "
                << jobs[n].target[1] << ", " << jobs[n].target[2] << ")  "
                << (s.reachable ? "REACHED" : "stretch")
                << "  iters=" << s.iterations
                << "  resid=" << s.residual << " m\n";
        }
        oss << "\nSummary:\n";
        oss << "  Total jobs       : " << count << "\n";
        oss << "  Reachable        : " << reachable << " / " << count << "\n";
        oss << "  Avg iterations   : "
            << (count ? static_cast<double>(totalIter) / count : 0.0) << "\n";
        oss << "  Max residual     : " << maxResid << " m\n";
        oss << "  Elapsed          : " << elapsedMs << " ms\n";
        oss << "  (includes context init, buffer upload, dispatch, readback)\n";

        outReport = oss.str();
        LOG_I("%s", outReport.c_str());
    }
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    uint32_t count = 4096;
    if (argc > 1) count = static_cast<uint32_t>(std::atoi(argv[1]));

    std::string report;
    try {
        IKSolverDemo demo;
        demo.run(count, "shaders/12_embedded.spv", report);
        printf("%s", report.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
