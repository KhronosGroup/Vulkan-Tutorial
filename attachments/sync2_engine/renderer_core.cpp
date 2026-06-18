/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "renderer.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <ranges>
#include <set>
#include <thread>
#include <type_traits>
#include <unordered_set>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE; // In a .cpp file

#include <vulkan/vk_platform.h>
#include <vulkan/vulkan.h>          // For PFN_vkGetInstanceProcAddr and C types
#include <vulkan/vulkan_raii.hpp>

// Debug callback for vk::raii - uses raw Vulkan C types for cross-platform compatibility
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallbackVkRaii(
  VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
  const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
  [[maybe_unused]] void* pUserData) {
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    // Print a message to the console
    std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
  } else {
    // Print a message to the console
    std::cout << "Validation layer: " << pCallbackData->pMessage << std::endl;
  }

  return VK_FALSE;
}

// Vulkan-Hpp style callback signature for newer headers expecting vk:: types
static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallbackVkHpp(
  vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
  [[maybe_unused]] vk::DebugUtilsMessageTypeFlagsEXT messageType,
  const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
  [[maybe_unused]] void* pUserData) {
  if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
  } else {
    std::cout << "Validation layer: " << pCallbackData->pMessage << std::endl;
  }
  return vk::False;
}

// Watchdog thread function - monitors frame updates and aborts if application hangs
static void WatchdogThreadFunc(std::atomic<std::chrono::steady_clock::time_point>* lastFrameTime,
                               std::atomic<bool>* running,
                               std::atomic<bool>* suppressed,
                               std::atomic<const char *>* progressLabel,
                               std::atomic<uint32_t>* progressIndex) {
  while (running->load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::seconds(5));

    if (!running->load(std::memory_order_relaxed)) {
      break; // Shutdown requested
    }

    // Check if frame timestamp was updated recently.
    // Some operations (e.g., BLAS/TLAS builds in Debug on large scenes) can legitimately take
    // much longer than 5 or 10 seconds. When suppressed, allow a longer grace period.
    auto now = std::chrono::steady_clock::now();
    auto lastUpdate = lastFrameTime->load(std::memory_order_relaxed);
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate).count();
    const int64_t allowedSeconds = (suppressed && suppressed->load(std::memory_order_relaxed)) ? 300 : 60;

    if (elapsed >= allowedSeconds) {
      // APPLICATION HAS HUNG - no frame updates for 10+ seconds
      const char* label = nullptr;
      if (progressLabel) {
        label = progressLabel->load(std::memory_order_relaxed);
      }
      uint32_t idx = 0;
      if (progressIndex) {
        idx = progressIndex->load(std::memory_order_relaxed);
      }

      std::cerr << "\n\n";
      std::cerr << "========================================\n";
      std::cerr << "WATCHDOG: APPLICATION HAS HUNG!\n";
      std::cerr << "========================================\n";
      std::cerr << "Last frame update was " << elapsed << " seconds ago.\n";
      if (suppressed && suppressed->load(std::memory_order_relaxed)) {
        std::cerr << "Watchdog was SUPPRESSED (allowed " << allowedSeconds << "s)\n";
      }
      if (label && label[0] != '\0') {
        std::cerr << "Last progress marker: " << label << "\n";
      }
      if (progressIndex) {
        std::cerr << "Progress index: " << idx << "\n";
      }
      std::cerr << "The render loop is not progressing.\n";
      std::cerr << "Aborting to generate stack trace...\n";
      std::cerr << "========================================\n\n";
      std::abort(); // Force crash with stack trace
    }
  }

  std::cout << "[Watchdog] Stopped\n";
}

// Renderer core implementation for the "Rendering Pipeline" chapter of the tutorial.
Renderer::Renderer(Platform* platform) : platform(platform) {
  // Initialize deviceExtensions with required extensions only
  // Optional extensions will be added later after checking device support
  deviceExtensions = requiredDeviceExtensions;
}

// Destructor
Renderer::~Renderer() {
  Cleanup();
}

// Initialize the renderer
bool Renderer::Initialize(const std::string& appName, bool enableValidationLayers, bool debugSync) {
  // Initialize the Vulkan-Hpp default dispatcher using the global symbol directly.
  // This avoids differences across Vulkan-Hpp versions for DynamicLoader placement.
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
  // Create a Vulkan instance
  if (!createInstance(appName, enableValidationLayers, debugSync)) {
    std::cerr << "Failed to create Vulkan instance" << std::endl;
    return false;
  }

  // Setup debug messenger
  if (!setupDebugMessenger(enableValidationLayers)) {
    std::cerr << "Failed to setup debug messenger" << std::endl;
    return false;
  }

  // Create surface
  if (!createSurface()) {
    std::cerr << "Failed to create surface" << std::endl;
    return false;
  }

  // Pick the physical device
  if (!pickPhysicalDevice()) {
    std::cerr << "Failed to pick physical device" << std::endl;
    return false;
  }

  // Create logical device
  if (!createLogicalDevice(enableValidationLayers)) {
    std::cerr << "Failed to create logical device" << std::endl;
    return false;
  }

  // Initialize memory pool for efficient memory management
  try {
    memoryPool = std::make_unique<MemoryPool>(device, physicalDevice);
    if (!memoryPool->initialize()) {
      std::cerr << "Failed to initialize memory pool" << std::endl;
      return false;
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to create memory pool: " << e.what() << std::endl;
    return false;
  }

  // Create timeline semaphore for frame-in-flight and cross-system synchronization
  vk::SemaphoreTypeCreateInfo timelineCreateInfo{
    .semaphoreType = vk::SemaphoreType::eTimeline,
    .initialValue = 0
  };
  vk::SemaphoreCreateInfo semaphoreCreateInfo{
    .pNext = &timelineCreateInfo
  };
  frameTimeline = vk::raii::Semaphore(device, semaphoreCreateInfo);

  // Create swap chain
  if (!createSwapChain()) {
    std::cerr << "Failed to create swap chain" << std::endl;
    return false;
  }

  // Create image views
  if (!createImageViews()) {
    std::cerr << "Failed to create image views" << std::endl;
    return false;
  }

  // Setup dynamic rendering
  if (!setupDynamicRendering()) {
    std::cerr << "Failed to setup dynamic rendering" << std::endl;
    return false;
  }

  // Create the descriptor set layout
  if (!createDescriptorSetLayout()) {
    std::cerr << "Failed to create descriptor set layout" << std::endl;
    return false;
  }

  // Create the graphics pipeline
  if (!createGraphicsPipeline()) {
    std::cerr << "Failed to create graphics pipeline" << std::endl;
    return false;
  }

  // Create PBR pipeline
  if (!createPBRPipeline()) {
    std::cerr << "Failed to create PBR pipeline" << std::endl;
    return false;
  }

  // Create the lighting pipeline
  if (!createLightingPipeline()) {
    std::cerr << "Failed to create lighting pipeline" << std::endl;
    return false;
  }

  // Create composite pipeline (fullscreen pass for off-screen → swapchain)
  if (!createCompositePipeline()) {
    std::cerr << "Failed to create composite pipeline" << std::endl;
    return false;
  }

  // Create compute pipeline
  if (!createComputePipeline()) {
    std::cerr << "Failed to create compute pipeline" << std::endl;
    return false;
  }

  // Ensure light storage buffers exist before creating Forward+ resources
  // so that compute descriptor binding 0 (lights SSBO) can be populated safely.
  if (!createOrResizeLightStorageBuffers(1)) {
    std::cerr << "Failed to create initial light storage buffers" << std::endl;
    return false;
  }

  // Create Forward+ compute and depth pre-pass pipelines/resources
  if (useForwardPlus) {
    if (!createForwardPlusPipelinesAndResources()) {
      std::cerr << "Failed to create Forward+ resources" << std::endl;
      return false;
    }
  }

  // Create ray query descriptor set layout and pipeline (but not resources yet - need descriptor pool first)
  if (!createRayQueryDescriptorSetLayout()) {
    std::cerr << "Failed to create ray query descriptor set layout" << std::endl;
    return false;
  }
  if (!createRayQueryPipeline()) {
    std::cerr << "Failed to create ray query pipeline" << std::endl;
    return false;
  }

  // Create the command pool
  if (!createCommandPool()) {
    std::cerr << "Failed to create command pool" << std::endl;
    return false;
  }

  // Create depth resources
  if (!createDepthResources()) {
    std::cerr << "Failed to create depth resources" << std::endl;
    return false;
  }

  if (useForwardPlus) {
    if (!createDepthPrepassPipeline()) {
      std::cerr << "Failed to create depth prepass pipeline" << std::endl;
      return false;
    }
  }

  // Create the descriptor pool
  if (!createDescriptorPool()) {
    std::cerr << "Failed to create descriptor pool" << std::endl;
    return false;
  }

  // Create ray query resources AFTER descriptor pool (needs pool for descriptor set allocation)
  if (!createRayQueryResources()) {
    std::cerr << "Failed to create ray query resources" << std::endl;
    return false;
  }

  // Note: Acceleration structure build is requested by scene_loading.cpp after entities load
  // No need to request it here during init

  // Light storage buffers were already created earlier to satisfy Forward+ binding requirements

  if (!createOpaqueSceneColorResources()) {
    std::cerr << "Failed to create opaque scene color resources" << std::endl;
    return false;
  }

  createTransparentDescriptorSets();

  // Create default texture resources
  if (!createDefaultTextureResources()) {
    std::cerr << "Failed to create default texture resources" << std::endl;
    return false;
  }

  // Create fallback transparent descriptor sets (must occur after default textures exist)
  createTransparentFallbackDescriptorSets();

  // Create shared default PBR textures (to avoid creating hundreds of identical textures)
  if (!createSharedDefaultPBRTextures()) {
    std::cerr << "Failed to create shared default PBR textures" << std::endl;
    return false;
  }

  // Create command buffers
  if (!createCommandBuffers()) {
    std::cerr << "Failed to create command buffers" << std::endl;
    return false;
  }

  // Create sync objects
  if (!createSyncObjects()) {
    std::cerr << "Failed to create sync objects" << std::endl;
    return false;
  }

  // Initialize background thread pool for async tasks (textures, etc.) AFTER all Vulkan resources are ready
  try {
    // Size the thread pool based on hardware concurrency, clamped to a sensible range
    unsigned int hw = std::max(2u, std::min(8u, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4u));
    threadPool = std::make_unique<ThreadPool>(hw);
  } catch (const std::exception& e) {
    std::cerr << "Failed to create thread pool: " << e.what() << std::endl;
    return false;
  }

  // Start background uploads worker now that queues/semaphores exist
  StartUploadsWorker();

  // Start watchdog thread to detect application hangs
  lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
  watchdogRunning.store(true, std::memory_order_relaxed);
  watchdogThread = std::thread(WatchdogThreadFunc, &lastFrameUpdateTime, &watchdogRunning, &watchdogSuppressed, &watchdogProgressLabel, &watchdogProgressIndex);

  std::cout << "[Watchdog] Started - will abort if no frame updates for 10+ seconds\n";

  initialized = true;
  return true;
}

void Renderer::ensureThreadLocalVulkanInit() const {
  // Initialize Vulkan-Hpp dispatcher per-thread; required for multi-threaded RAII usage
  static thread_local bool s_tlsInitialized = false;
  if (s_tlsInitialized)
    return;
  try {
    // Initialize the dispatcher for this thread using the global symbol.
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    if (*instance) {
      VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
    }
    if (*device) {
      VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
    }
    s_tlsInitialized = true;
  } catch (...) {
    // best-effort
  }
}

// Clean up renderer resources
void Renderer::KickWatchdog() {
  lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
}

void Renderer::Cleanup() {
  // Stop watchdog thread first to prevent false hang detection during shutdown
  if (watchdogRunning.load(std::memory_order_relaxed)) {
    watchdogRunning.store(false, std::memory_order_relaxed);
    if (watchdogThread.joinable()) {
      watchdogThread.join();
    }
  }

  // Ensure background workers are stopped before tearing down Vulkan resources
  StopUploadsWorker();

  // Disallow any further descriptor writes during shutdown.
  // This prevents late updates/frees racing against pool destruction.
  descriptorSetsValid.store(false, std::memory_order_relaxed); {
    std::lock_guard<std::mutex> lk(pendingDescMutex);
    pendingDescOps.clear();
    descriptorRefreshPending.store(false, std::memory_order_relaxed);
  } {
    std::unique_lock<std::shared_mutex> lock(threadPoolMutex);
    if (threadPool) {
      threadPool.reset();
    }
  }

  if (!initialized) {
    return;
  }

  std::cout << "Starting renderer cleanup..." << std::endl;

  // Wait for the device to be idle before cleaning up
  try {
    WaitIdle();
  } catch (...) {
  }

  // 1) Clean up any swapchain-scoped resources first
  cleanupSwapChain();

  // 2) Clear per-entity resources (descriptor sets and buffers) while descriptor pools still exist
  for (auto& kv : entityResources) {
    auto& resources = kv.second;
    resources.basicDescriptorSets.clear();
    resources.pbrDescriptorSets.clear();
    resources.uniformBuffers.clear();
    resources.uniformBufferAllocations.clear();
    resources.uniformBuffersMapped.clear();
    resources.instanceBuffer = nullptr;
    resources.instanceBufferAllocation = nullptr;
    resources.instanceBufferMapped = nullptr;
  }
  entityResources.clear();

  // 3) Clear any global descriptor sets that are allocated from pools to avoid dangling refs
  transparentDescriptorSets.clear();
  transparentFallbackDescriptorSets.clear();
  compositeDescriptorSets.clear();
  computeDescriptorSets.clear();
  rqCompositeDescriptorSets.clear();

  // 3.5) Clear ray query descriptor sets BEFORE destroying descriptor pool
  // Without this, rayQueryDescriptorSets' RAII destructor tries to free them after
  // the pool is destroyed, causing "Invalid VkDescriptorPool Object" validation errors
  rayQueryDescriptorSets.clear();

  // Ray Query composite sampler/sets are allocated from the shared descriptor pool.
  // Ensure they are released before destroying the pool.
  rqCompositeSampler = nullptr;

  // 4) Destroy/Reset pipelines and pipeline layouts (graphics/compute/forward+)
  graphicsPipeline = nullptr;
  pbrGraphicsPipeline = nullptr;
  pbrBlendGraphicsPipeline = nullptr;
  pbrPremulBlendGraphicsPipeline = nullptr;
  pbrPrepassGraphicsPipeline = nullptr;
  glassGraphicsPipeline = nullptr;
  lightingPipeline = nullptr;
  compositePipeline = nullptr;
  forwardPlusPipeline = nullptr;
  depthPrepassPipeline = nullptr;

  pipelineLayout = nullptr;
  pbrPipelineLayout = nullptr;
  lightingPipelineLayout = nullptr;
  compositePipelineLayout = nullptr;
  pbrTransparentPipelineLayout = nullptr;
  forwardPlusPipelineLayout = nullptr;

  // 4.3) Ray query pipelines and layouts
  rayQueryPipeline = nullptr;
  rayQueryPipelineLayout = nullptr;

  // 4.5) Forward+ per-frame resources (including descriptor sets) must be released
  // BEFORE destroying descriptor pools to avoid vkFreeDescriptorSets with invalid pool
  for (auto& fp : forwardPlusPerFrame) {
    fp.tileHeaders = nullptr;
    fp.tileHeadersAlloc = nullptr;
    fp.tileLightIndices = nullptr;
    fp.tileLightIndicesAlloc = nullptr;
    fp.params = nullptr;
    fp.paramsAlloc = nullptr;
    fp.paramsMapped = nullptr;
    fp.debugOut = nullptr;
    fp.debugOutAlloc = nullptr;
    fp.probeOffscreen = nullptr;
    fp.probeOffscreenAlloc = nullptr;
    fp.probeSwapchain = nullptr;
    fp.probeSwapchainAlloc = nullptr;
    fp.computeSet = nullptr; // descriptor set allocated from compute/graphics pools
  }
  forwardPlusPerFrame.clear();

  // 5) Destroy descriptor set layouts and pools (compute + graphics)
  descriptorSetLayout = nullptr;
  pbrDescriptorSetLayout = nullptr;
  transparentDescriptorSetLayout = nullptr;
  compositeDescriptorSetLayout = nullptr;
  forwardPlusDescriptorSetLayout = nullptr;
  computeDescriptorSetLayout = nullptr;
  rayQueryDescriptorSetLayout = nullptr;

  // Pools last, after sets are cleared
  computeDescriptorPool = nullptr;
  descriptorPool = nullptr;

  // 6) Clear textures and aliases, including default resources
  {
    std::unique_lock<std::shared_mutex> lk(textureResourcesMutex);
    textureResources.clear();
    textureAliases.clear();
  }
  // Reset default texture resources
  defaultTextureResources.textureSampler = nullptr;
  defaultTextureResources.textureImageView = nullptr;
  defaultTextureResources.textureImage = nullptr;
  defaultTextureResources.textureImageAllocation = nullptr;

  // 7) Opaque scene color and related descriptors
  opaqueSceneColorSampler = nullptr;
  opaqueSceneColorImages.clear();
  opaqueSceneColorImageAllocations.clear();
  opaqueSceneColorImageViews.clear();
  opaqueSceneColorImageLayouts.clear();

  // 7.5) Ray query output image and acceleration structures
  rayQueryOutputImageView = nullptr;
  rayQueryOutputImage = nullptr;
  rayQueryOutputImageAllocation = nullptr;

  // Clear acceleration structures (BLAS and TLAS buffers)
  blasStructures.clear();
  tlasStructure = AccelerationStructure{};

  // 8) (moved above) Forward+ per-frame buffers cleared prior to pool destruction

  // 9) Command buffers/pools
  commandBuffers.clear();
  commandPool = nullptr;
  computeCommandPool = nullptr;

  // 10) Sync objects
  imageAvailableSemaphores.clear();
  renderFinishedSemaphores.clear();
  inFlightFences.clear();
  uploadsTimeline = nullptr;

  // 11) Queues and surface (RAII handles will release upon reset; keep device alive until the end)
  graphicsQueue = nullptr;
  presentQueue = nullptr;
  computeQueue = nullptr;
  transferQueue = nullptr;
  surface = nullptr;

  // 12) Memory pool last
  memoryPool.reset();

  // Finally mark uninitialized
  initialized = false;
  std::cout << "Renderer cleanup completed." << std::endl;
}

// Create instance
bool Renderer::createInstance(const std::string& appName, bool enableValidationLayers, bool debugSync) {
  try {
    // Create application info
    vk::ApplicationInfo appInfo{
      .pApplicationName = appName.c_str(),
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "Sync2 Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_4
    };

    // Get required extensions
    std::vector<const char *> extensions;

    // Add required extensions for GLFW
#if defined(PLATFORM_DESKTOP)
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    extensions.insert(extensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);
#endif

    // Add debug extension if validation layers are enabled
    if (enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Add validation features if debug sync is requested
    if (debugSync) {
      extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    }

    // Create instance info
    vk::InstanceCreateInfo createInfo{
      .pApplicationInfo = &appInfo,
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data()
    };

    // Set up validation features if requested
    vk::ValidationFeaturesEXT validationFeatures;
    std::vector<vk::ValidationFeatureEnableEXT> enabledValidationFeatures;

    if (debugSync) {
      enabledValidationFeatures.push_back(vk::ValidationFeatureEnableEXT::eSynchronizationValidation);
    }

    // Enable validation layers if requested
    if (enableValidationLayers) {
      if (!checkValidationLayerSupport()) {
        std::cerr << "Validation layers requested, but not available" << std::endl;
        return false;
      }

      createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    }

    if (!enabledValidationFeatures.empty()) {
      validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(enabledValidationFeatures.size());
      validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures.data();
      createInfo.pNext = &validationFeatures;
    }

    // Create instance
    instance = vk::raii::Instance(context, createInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create instance: " << e.what() << std::endl;
    return false;
  }
}

// Setup debug messenger
bool Renderer::setupDebugMessenger(bool enableValidationLayers) {
  if (!enableValidationLayers) {
    return true;
  }

  try {
    // Create debug messenger info
    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

    // Select callback via simple platform macro: Android typically expects C PFN types in headers
    // while desktop (newer Vulkan-Hpp) expects vk:: types.
#if defined(__ANDROID__)
    createInfo.pfnUserCallback = &debugCallbackVkRaii;
#else
    createInfo.pfnUserCallback = &debugCallbackVkHpp;
#endif

    // Create debug messenger
    debugMessenger = vk::raii::DebugUtilsMessengerEXT(instance, createInfo);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to set up debug messenger: " << e.what() << std::endl;
    return false;
  }
}

// Create surface
bool Renderer::createSurface() {
  try {
    // Create surface
    VkSurfaceKHR _surface;
    if (!platform->CreateVulkanSurface(*instance, &_surface)) {
      std::cerr << "Failed to create window surface" << std::endl;
      return false;
    }

    surface = vk::raii::SurfaceKHR(instance, _surface);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create surface: " << e.what() << std::endl;
    return false;
  }
}

// Pick a physical device
bool Renderer::pickPhysicalDevice() {
  try {
    // Get available physical devices
    std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

    if (devices.empty()) {
      std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
      return false;
    }

    // Prioritize discrete GPUs (like NVIDIA RTX 2080) over integrated GPUs (like Intel UHD Graphics)
    // First, collect all suitable devices with their suitability scores
    std::multimap<int, vk::raii::PhysicalDevice> suitableDevices;

    for (auto& _device : devices) {
      // Print device properties for debugging
      vk::PhysicalDeviceProperties deviceProperties = _device.getProperties();
      std::cout << "Checking device: " << deviceProperties.deviceName
          << " (Type: " << vk::to_string(deviceProperties.deviceType) << ")" << std::endl;

      // Check if the device supports Vulkan 1.3
      bool supportsVulkan1_3 = deviceProperties.apiVersion >= VK_API_VERSION_1_3;
      if (!supportsVulkan1_3) {
        std::cout << "  - Does not support Vulkan 1.3" << std::endl;
        continue;
      }

      // Check queue families
      QueueFamilyIndices indices = findQueueFamilies(_device);
      bool supportsGraphics = indices.isComplete();
      if (!supportsGraphics) {
        std::cout << "  - Missing required queue families" << std::endl;
        continue;
      }

      // Check device extensions
      bool supportsAllRequiredExtensions = checkDeviceExtensionSupport(_device);
      if (!supportsAllRequiredExtensions) {
        std::cout << "  - Missing required extensions" << std::endl;
        continue;
      }

      // Check swap chain support
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_device);
      bool swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
      if (!swapChainAdequate) {
        std::cout << "  - Inadequate swap chain support" << std::endl;
        continue;
      }

      // Check for required features
      auto features = _device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
      bool supportsRequiredFeatures = features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering;
      if (!supportsRequiredFeatures) {
        std::cout << "  - Does not support required features (dynamicRendering)" << std::endl;
        continue;
      }

      // Calculate suitability score - prioritize discrete GPUs
      int score = 0;

      // Discrete GPUs get the highest priority (NVIDIA RTX 2080, AMD, etc.)
      if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        score += 1000;
        std::cout << "  - Discrete GPU: +1000 points" << std::endl;
      }
      // Integrated GPUs get lower priority (Intel UHD Graphics, etc.)
      else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
        score += 100;
        std::cout << "  - Integrated GPU: +100 points" << std::endl;
      }

      // Add points for memory size (more VRAM is better)
      vk::PhysicalDeviceMemoryProperties memProperties = _device.getMemoryProperties();
      for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
        if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
          // Add 1 point per GB of VRAM
          score += static_cast<int>(memProperties.memoryHeaps[i].size / (1024 * 1024 * 1024));
          break;
        }
      }

      std::cout << "  - Device is suitable with score: " << score << std::endl;
      suitableDevices.emplace(score, _device);
    }

    if (!suitableDevices.empty()) {
      // Select the device with the highest score (discrete GPU with most VRAM)
      physicalDevice = suitableDevices.rbegin()->second;
      vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
      std::cout << "Selected device: " << deviceProperties.deviceName
          << " (Type: " << vk::to_string(deviceProperties.deviceType)
          << ", Score: " << suitableDevices.rbegin()->first << ")" << std::endl;

      // Store queue family indices for the selected device
      queueFamilyIndices = findQueueFamilies(physicalDevice);

      // Add supported optional extensions
      addSupportedOptionalExtensions();

      return true;
    }
    std::cerr << "Failed to find a suitable GPU. Make sure your GPU supports Vulkan and has the required extensions." << std::endl;
    return false;
  } catch (const std::exception& e) {
    std::cerr << "Failed to pick physical device: " << e.what() << std::endl;
    return false;
  }
}

// Add supported optional extensions
void Renderer::addSupportedOptionalExtensions() {
  try {
    // Get available extensions
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    // Build a set of available extension names for quick lookup
    std::unordered_set<std::string> avail;
    for (const auto& e : availableExtensions) {
      avail.insert(e.extensionName);
    }

    // Set of already added extensions to avoid duplicates
    std::unordered_set<std::string> added(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& optionalExt : optionalDeviceExtensions) {
      if (avail.contains(optionalExt) && !added.contains(optionalExt)) {
        deviceExtensions.push_back(optionalExt);
        added.insert(optionalExt);
        std::cout << "Adding optional extension: " << optionalExt << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to add optional extensions: " << e.what() << std::endl;
  }
}

// Create logical device
bool Renderer::createLogicalDevice(bool enableValidationLayers) {
  try {
    // 1. Setup Queues
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
      queueFamilyIndices.graphicsFamily.value(),
      queueFamilyIndices.presentFamily.value(),
      queueFamilyIndices.computeFamily.value(),
      queueFamilyIndices.transferFamily.value()
    };
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      queueCreateInfos.push_back({
        .queueFamilyIndex = queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
      });
    }

    // 2. Query Supported Features
    auto supported = physicalDevice.getFeatures2<
      vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan11Features,
      vk::PhysicalDeviceVulkan12Features,
      vk::PhysicalDeviceVulkan13Features,
      vk::PhysicalDeviceVulkan14Features,
      vk::PhysicalDeviceRobustness2FeaturesEXT,
      vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
      vk::PhysicalDeviceRayQueryFeaturesKHR
    >();

    // 3. Setup Enabled Features using StructureChain for stability and zero-init
    // Core versioned features are included in the chain.
    // Extensions that are NOT promoted to 1.4 are linked manually.
    vk::StructureChain<
      vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan11Features,
      vk::PhysicalDeviceVulkan12Features,
      vk::PhysicalDeviceVulkan13Features,
      vk::PhysicalDeviceVulkan14Features
    > enabledChain;

    auto& f2 = enabledChain.get<vk::PhysicalDeviceFeatures2>();
    auto const& sCore = supported.get<vk::PhysicalDeviceFeatures2>().features;
    f2.features.samplerAnisotropy = vk::True;
    f2.features.depthBiasClamp = sCore.depthBiasClamp;
    f2.features.shaderSampledImageArrayDynamicIndexing = sCore.shaderSampledImageArrayDynamicIndexing;
    f2.features.shaderInt64 = sCore.shaderInt64; // Required for RQ

    auto& f11 = enabledChain.get<vk::PhysicalDeviceVulkan11Features>();
    f11.shaderDrawParameters = vk::True;

    auto& f12 = enabledChain.get<vk::PhysicalDeviceVulkan12Features>();
    auto const& s12 = supported.get<vk::PhysicalDeviceVulkan12Features>();
    f12.descriptorIndexing = vk::True;
    f12.shaderSampledImageArrayNonUniformIndexing = s12.shaderSampledImageArrayNonUniformIndexing;
    f12.descriptorBindingPartiallyBound = s12.descriptorBindingPartiallyBound;
    f12.descriptorBindingUpdateUnusedWhilePending = s12.descriptorBindingUpdateUnusedWhilePending;
    f12.descriptorBindingSampledImageUpdateAfterBind = s12.descriptorBindingSampledImageUpdateAfterBind;
    f12.descriptorBindingUniformBufferUpdateAfterBind = s12.descriptorBindingUniformBufferUpdateAfterBind;
    f12.timelineSemaphore = vk::True;
    f12.vulkanMemoryModel = vk::True;
    f12.vulkanMemoryModelDeviceScope = s12.vulkanMemoryModelDeviceScope;
    f12.bufferDeviceAddress = vk::True;
    f12.storageBuffer8BitAccess = s12.storageBuffer8BitAccess;

    auto& f13 = enabledChain.get<vk::PhysicalDeviceVulkan13Features>();
    f13.dynamicRendering = vk::True;
    f13.synchronization2 = vk::True;

    auto& f14 = enabledChain.get<vk::PhysicalDeviceVulkan14Features>();
    auto const& s14 = supported.get<vk::PhysicalDeviceVulkan14Features>();
    f14.hostImageCopy = vk::True;
    f14.dynamicRenderingLocalRead = s14.dynamicRenderingLocalRead;
    f14.maintenance5 = s14.maintenance5;
    f14.maintenance6 = s14.maintenance6;
    f14.pushDescriptor = s14.pushDescriptor;

    // 4. Link Extensions (not in 1.4 core)
    void** lastNext = &f14.pNext;

    vk::PhysicalDeviceRobustness2FeaturesEXT enabledRobust2{};
    if (std::find_if(deviceExtensions.begin(), deviceExtensions.end(), [](const char* n){ return std::strcmp(n, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) == 0; }) != deviceExtensions.end()) {
        enabledRobust2 = supported.get<vk::PhysicalDeviceRobustness2FeaturesEXT>();
        *lastNext = &enabledRobust2;
        lastNext = &enabledRobust2.pNext;
        if (enabledRobust2.robustBufferAccess2) f2.features.robustBufferAccess = vk::True;
    }

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR enabledAS{};
    if (std::find_if(deviceExtensions.begin(), deviceExtensions.end(), [](const char* n){ return std::strcmp(n, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0; }) != deviceExtensions.end()) {
        enabledAS = supported.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
        *lastNext = &enabledAS;
        lastNext = &enabledAS.pNext;
    }

    vk::PhysicalDeviceRayQueryFeaturesKHR enabledRQ{};
    if (std::find_if(deviceExtensions.begin(), deviceExtensions.end(), [](const char* n){ return std::strcmp(n, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0; }) != deviceExtensions.end()) {
        enabledRQ = supported.get<vk::PhysicalDeviceRayQueryFeaturesKHR>();
        *lastNext = &enabledRQ;
        lastNext = &enabledRQ.pNext;
    }

#if !defined(PLATFORM_ANDROID)
    vk::PhysicalDeviceShaderTileImageFeaturesEXT enabledTile{};
    if (std::find_if(deviceExtensions.begin(), deviceExtensions.end(), [](const char* n){ return std::strcmp(n, VK_EXT_SHADER_TILE_IMAGE_EXTENSION_NAME) == 0; }) != deviceExtensions.end()) {
        auto s = physicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderTileImageFeaturesEXT>();
        enabledTile = s.get<vk::PhysicalDeviceShaderTileImageFeaturesEXT>();
        *lastNext = &enabledTile;
        lastNext = &enabledTile.pNext;
    }
#endif

    *lastNext = nullptr;

    // 5. Create Logical Device
    vk::DeviceCreateInfo createInfo{
      .pNext = &f2,
      .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
      .pQueueCreateInfos = queueCreateInfos.data(),
      .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
      .ppEnabledExtensionNames = deviceExtensions.data()
    };

    device = vk::raii::Device(physicalDevice, createInfo);

    // Query Acceleration Structure properties if supported
    if (std::find_if(deviceExtensions.begin(), deviceExtensions.end(), [](const char* n) {
          return std::strcmp(n, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0;
        }) != deviceExtensions.end()) {
      auto propertiesChain = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
      accelStructProperties = propertiesChain.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
    }

    // 6. Initialize Queues and semaphores
    graphicsQueue = vk::raii::Queue(device, queueFamilyIndices.graphicsFamily.value(), 0);
    presentQueue = vk::raii::Queue(device, queueFamilyIndices.presentFamily.value(), 0);
    computeQueue = vk::raii::Queue(device, queueFamilyIndices.computeFamily.value(), 0);
    transferQueue = vk::raii::Queue(device, queueFamilyIndices.transferFamily.value(), 0);

    // Record states
    robustness2Enabled = (enabledRobust2.robustBufferAccess2 || enabledRobust2.robustImageAccess2 || enabledRobust2.nullDescriptor);
    dynamicRenderingLocalReadEnabled = (f14.dynamicRenderingLocalRead == vk::True);
    accelerationStructureEnabled = (enabledAS.accelerationStructure == vk::True);
    rayQueryEnabled = (enabledRQ.rayQuery == vk::True);
    std::cout << "[Renderer] Ray Query supported: " << (rayQueryEnabled ? "YES" : "NO") << std::endl;
    std::cout << "[Renderer] Acceleration Structures supported: " << (accelerationStructureEnabled ? "YES" : "NO") << std::endl;
    descriptorIndexingEnabled = (f12.descriptorIndexing == vk::True);

    vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> timelineChain(
      {},
      {.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0});
    uploadsTimeline = vk::raii::Semaphore(device, timelineChain.get<vk::SemaphoreCreateInfo>());
    nextUploadTimelineValue.store(0, std::memory_order_relaxed);
    nextFrameTimelineValue.store(0, std::memory_order_relaxed);

    initialized = true;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create logical device: " << e.what() << std::endl;
    return false;
  }
}

// Check validation layer support
bool Renderer::checkValidationLayerSupport() const {
  // Get available layers
  std::vector<vk::LayerProperties> availableLayers = context.enumerateInstanceLayerProperties();

  // Check if all requested layers are available
  for (const char* layerName : validationLayers) {
    bool layerFound = false;

    for (const auto& layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}
