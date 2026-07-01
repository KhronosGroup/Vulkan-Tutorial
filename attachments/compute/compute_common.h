#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#  include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#ifdef DEMO_WINDOWED
#  define GLFW_INCLUDE_VULKAN
#  include <GLFW/glfw3.h>
#endif

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
const std::vector<const char*> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif

// ---------------------------------------------------------------------------
// Debug messenger callback
// ---------------------------------------------------------------------------
static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT* pData,
    void*)
{
    if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
        std::cerr << "[VK] " << pData->pMessage << '\n';
    return vk::False;
}

// ---------------------------------------------------------------------------
// Headless Vulkan context – everything a compute-only demo needs
// ---------------------------------------------------------------------------
struct HeadlessContext
{
    vk::raii::Context                context;
    vk::raii::Instance               instance       = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::PhysicalDevice         physicalDevice = nullptr;
    vk::raii::Device                 device         = nullptr;
    uint32_t                         computeQueueFamily = ~0u;
    vk::raii::Queue                  computeQueue   = nullptr;
    vk::raii::CommandPool            commandPool    = nullptr;

    void init(const char* appName = "ComputeDemo")
    {
        // Instance
        vk::ApplicationInfo appInfo{
            .pApplicationName   = appName,
            .applicationVersion = VK_MAKE_VERSION(1,0,0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1,0,0),
            .apiVersion         = vk::ApiVersion14};

        std::vector<const char*> layers;
        std::vector<const char*> extensions;
        if (kEnableValidation) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }

        vk::InstanceCreateInfo instCI{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames     = layers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data()};
        instance = vk::raii::Instance(context, instCI);

        // Debug messenger
        if (kEnableValidation) {
            vk::DebugUtilsMessengerCreateInfoEXT dmCI{
                .messageSeverity =
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning  |
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                .messageType =
                    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral     |
                    vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation  |
                    vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                .pfnUserCallback = &debugCallback};
            debugMessenger = instance.createDebugUtilsMessengerEXT(dmCI);
        }

        // Physical device – pick first that has a compute queue
        for (auto& pd : instance.enumeratePhysicalDevices()) {
            auto qfps = pd.getQueueFamilyProperties();
            for (uint32_t i = 0; i < qfps.size(); ++i) {
                if (qfps[i].queueFlags & vk::QueueFlagBits::eCompute) {
                    physicalDevice    = pd;
                    computeQueueFamily = i;
                    break;
                }
            }
            if (computeQueueFamily != ~0u) break;
        }
        if (computeQueueFamily == ~0u)
            throw std::runtime_error("No compute queue family found");

        // Logical device
        float priority = 1.0f;
        vk::DeviceQueueCreateInfo qCI{
            .queueFamilyIndex = computeQueueFamily,
            .queueCount       = 1,
            .pQueuePriorities = &priority};

        // Feature chain: base → 1.1 → 1.2 → 1.3
        // ScalarBlockLayout and BufferDeviceAddress are promoted into Vulkan 1.2;
        // they must be set in VkPhysicalDeviceVulkan12Features, not in separate structs.
        vk::PhysicalDeviceVulkan13Features v13Features{
            .synchronization2 = true};
        vk::PhysicalDeviceVulkan12Features v12Features{
            .pNext                 = &v13Features,
            .drawIndirectCount     = true,
            .shaderFloat16         = true,
            .shaderInt8            = true,
            .scalarBlockLayout     = true,
            .bufferDeviceAddress   = true};
        // variablePointers* are required by clspv-generated SPIR-V (Chapter 05).
        // They are harmless for the other headless chapters.
        vk::PhysicalDeviceVulkan11Features v11Features{
            .pNext                          = &v12Features,
            .variablePointersStorageBuffer  = true,
            .variablePointers               = true,
            .shaderDrawParameters           = true};
        vk::PhysicalDeviceFeatures2 features2{
            .pNext    = &v11Features,
            .features = {.shaderInt64 = true}};

        std::vector<const char*> devExtensions;

        vk::DeviceCreateInfo devCI{
            .pNext                   = &features2,
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &qCI,
            .enabledExtensionCount   = static_cast<uint32_t>(devExtensions.size()),
            .ppEnabledExtensionNames = devExtensions.data()};
        device       = vk::raii::Device(physicalDevice, devCI);
        computeQueue = vk::raii::Queue(device, computeQueueFamily, 0);

        // Command pool
        vk::CommandPoolCreateInfo poolCI{
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = computeQueueFamily};
        commandPool = vk::raii::CommandPool(device, poolCI);
    }

    // -----------------------------------------------------------------------
    // Buffer helpers
    // -----------------------------------------------------------------------
    [[nodiscard]] uint32_t findMemoryType(uint32_t typeBits, vk::MemoryPropertyFlags props) const
    {
        auto memProps = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
            if ((typeBits & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        throw std::runtime_error("No suitable memory type");
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags props,
                      vk::raii::Buffer& buf, vk::raii::DeviceMemory& mem,
                      vk::MemoryAllocateFlags allocFlags = {}) const
    {
        buf = vk::raii::Buffer(device, vk::BufferCreateInfo{
            .size        = size,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive});

        auto req = buf.getMemoryRequirements();

        vk::MemoryAllocateFlagsInfo flagsInfo{.flags = allocFlags};
        vk::MemoryAllocateInfo allocInfo{
            .pNext           = allocFlags ? &flagsInfo : nullptr,
            .allocationSize  = req.size,
            .memoryTypeIndex = findMemoryType(req.memoryTypeBits, props)};
        mem = vk::raii::DeviceMemory(device, allocInfo);
        buf.bindMemory(mem, 0);
    }

    // One-shot command buffer
    [[nodiscard]] vk::raii::CommandBuffer beginOneShot() const
    {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool        = *commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1};
        auto cb = std::move(vk::raii::CommandBuffers(device, allocInfo).front());
        cb.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return cb;
    }

    void endOneShot(const vk::raii::CommandBuffer& cb) const
    {
        cb.end();
        vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &*cb};
        computeQueue.submit(si);
        computeQueue.waitIdle();
    }

    // -----------------------------------------------------------------------
    // SPIR-V loader
    // -----------------------------------------------------------------------
    static std::vector<char> readSPV(const std::string& path)
    {
        std::ifstream f(path, std::ios::ate | std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
        std::vector<char> buf(f.tellg());
        f.seekg(0); f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        return buf;
    }

    [[nodiscard]] vk::raii::ShaderModule loadShader(const std::string& path) const
    {
        auto code = readSPV(path);
        return vk::raii::ShaderModule(device, vk::ShaderModuleCreateInfo{
            .codeSize = code.size(),
            .pCode    = reinterpret_cast<const uint32_t*>(code.data())});
    }
};

// ---------------------------------------------------------------------------
// Simple wall-clock timer
// ---------------------------------------------------------------------------
struct Timer
{
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
    void start() { t0 = Clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
};
