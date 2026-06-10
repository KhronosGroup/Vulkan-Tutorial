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

#ifdef PLATFORM_ANDROID
#include <vulkan/vulkan_android.h>
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <ranges>
#include <set>
#include <thread>
#include <type_traits>

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
    const int64_t allowedSeconds = (suppressed && suppressed->load(std::memory_order_relaxed)) ? 60 : 10;

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

  // Suppress watchdog by default during startup to allow for debugger attachment
  // and long initialization times on some mobile devices.
  watchdogSuppressed.store(false, std::memory_order_relaxed);

#if defined(PLATFORM_ANDROID)
  // Re-enable Ray Query and Forward+ for Android now that basic rendering is stabilized
  currentRenderMode = RenderMode::RayQuery;
  useForwardPlus = true;
  forwardPlusPerFrame.resize(MAX_FRAMES_IN_FLIGHT);
#endif
}

// Destructor
Renderer::~Renderer() {
  Cleanup();
}

// Initialize the renderer
bool Renderer::Initialize(const std::string& appName, bool enableValidationLayers) {
  LOGI("Renderer::Initialize start");
  // Initialize the Vulkan-Hpp default dispatcher.
  // On Android, use a dynamic loader to ensure we get the correct entry point.
#if defined(PLATFORM_ANDROID)
  LOGI("Initializing dispatcher with DynamicLoader...");
  static vk::detail::DynamicLoader dl;
  PFN_vkGetInstanceProcAddr pvkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  if (!pvkGetInstanceProcAddr) {
    LOGE("Failed to load vkGetInstanceProcAddr!");
    return false;
  }
  VULKAN_HPP_DEFAULT_DISPATCHER.init(pvkGetInstanceProcAddr);
  LOGI("Dispatcher initialized");
#else
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
#endif
  // Create a Vulkan instance
  LOGI("Creating Vulkan instance...");
  if (!createInstance(appName, enableValidationLayers)) {
    LOGE("Failed to create Vulkan instance");
    return false;
  }
  LOGI("Instance created successfully");

  // Setup debug messenger
  LOGI("Setting up debug messenger...");
  if (!setupDebugMessenger(enableValidationLayers)) {
    LOGE("Failed to setup debug messenger");
    return false;
  }
  LOGI("Debug messenger setup successfully");

  // Create surface
  LOGI("Creating surface...");
  if (!createSurface()) {
    LOGE("Failed to create surface");
    return false;
  }
  LOGI("Surface created successfully");

  // Pick the physical device
  LOGI("Picking physical device...");
  if (!pickPhysicalDevice()) {
    LOGE("Failed to pick physical device");
    return false;
  }
  LOGI("Physical device picked successfully");

  // Create logical device
  LOGI("Creating logical device...");
  if (!createLogicalDevice(enableValidationLayers)) {
    LOGE("Failed to create logical device");
    return false;
  }
  LOGI("Logical device created successfully");

  // Initialize memory pool for efficient memory management
  LOGI("Initializing memory pool...");
  try {
    memoryPool = std::make_unique<MemoryPool>(device, physicalDevice);
    if (!memoryPool->initialize()) {
      LOGE("Failed to initialize memory pool");
      return false;
    }
    LOGI("Memory pool initialized successfully");
  } catch (const std::exception& e) {
    LOGE("Failed to create memory pool: %s", e.what());
    return false;
  }

  // Create swap chain
  LOGI("Creating swap chain...");
  if (!createSwapChain()) {
    LOGE("Failed to create swap chain");
    return false;
  }
  LOGI("Swap chain created successfully");

  // Create image views
  LOGI("Creating image views...");
  if (!createImageViews()) {
    LOGE("Failed to create image views");
    return false;
  }
  LOGI("Image views created successfully");

  // Setup dynamic rendering
  LOGI("Setting up dynamic rendering...");
  if (!setupDynamicRendering()) {
    LOGE("Failed to setup dynamic rendering");
    return false;
  }
  LOGI("Dynamic rendering setup successfully");

  // Create the descriptor set layout
  LOGI("Creating descriptor set layout...");
  if (!createDescriptorSetLayout()) {
    LOGE("Failed to create descriptor set layout");
    return false;
  }
  LOGI("Descriptor set layout created successfully");

  // Create the graphics pipeline
  LOGI("Creating graphics pipeline...");
  if (!createGraphicsPipeline()) {
    LOGE("Failed to create graphics pipeline");
    return false;
  }
  LOGI("Graphics pipeline created successfully");

  // Create PBR pipeline
  LOGI("Creating PBR pipeline...");
  if (!createPBRPipeline()) {
    LOGE("Failed to create PBR pipeline");
    return false;
  }
  LOGI("PBR pipeline created successfully");

  // Create the lighting pipeline
  LOGI("Creating lighting pipeline...");
  if (!createLightingPipeline()) {
    LOGE("Failed to create lighting pipeline");
    return false;
  }
  LOGI("Lighting pipeline created successfully");

  // Create composite pipeline (fullscreen pass for off-screen → swapchain)
  LOGI("Creating composite pipeline...");
  if (!createCompositePipeline()) {
    LOGE("Failed to create composite pipeline");
    return false;
  }
  LOGI("Composite pipeline created successfully");

  // Create compute pipeline
  LOGI("Creating compute pipeline...");
  if (!createComputePipeline()) {
    LOGE("Failed to create compute pipeline");
    return false;
  }
  LOGI("Compute pipeline created successfully");

  // Ensure light storage buffers exist before creating Forward+ resources
  // so that compute descriptor binding 0 (lights SSBO) can be populated safely.
  LOGI("Creating light storage buffers...");
  if (!createOrResizeLightStorageBuffers(1)) {
    LOGE("Failed to create initial light storage buffers");
    return false;
  }
  LOGI("Light storage buffers created successfully");

  // Create Forward+ compute and depth pre-pass pipelines/resources
  if (useForwardPlus) {
    LOGI("Creating Forward+ resources...");
    if (!createForwardPlusPipelinesAndResources()) {
      LOGE("Failed to create Forward+ resources");
      return false;
    }
    LOGI("Forward+ resources created successfully");
  }

  // Create ray query descriptor set layout and pipeline (but not resources yet - need descriptor pool first)
  LOGI("Creating ray query descriptor set layout...");
  if (!createRayQueryDescriptorSetLayout()) {
    LOGE("Failed to create ray query descriptor set layout");
    return false;
  }
  LOGI("Ray query descriptor set layout created successfully");

  LOGI("Creating ray query pipeline...");
  if (!createRayQueryPipeline()) {
    LOGE("Failed to create ray query pipeline");
    return false;
  }
  LOGI("Ray query pipeline created successfully");

  // Create the command pool
  LOGI("Creating command pool...");
  if (!createCommandPool()) {
    LOGE("Failed to create command pool");
    return false;
  }
  LOGI("Command pool created successfully");

  // Create depth resources
  LOGI("Creating depth resources...");
  if (!createDepthResources()) {
    LOGE("Failed to create depth resources");
    return false;
  }
  LOGI("Depth resources created successfully");

  if (useForwardPlus) {
    LOGI("Creating depth prepass pipeline...");
    if (!createDepthPrepassPipeline()) {
      LOGE("Failed to create depth prepass pipeline");
      return false;
    }
    LOGI("Depth prepass pipeline created successfully");
  }

  // Create the descriptor pool
  LOGI("Creating descriptor pool...");
  if (!createDescriptorPool()) {
    LOGE("Failed to create descriptor pool");
    return false;
  }
  LOGI("Descriptor pool created successfully");

  // Create ray query resources AFTER descriptor pool (needs pool for descriptor set allocation)
  LOGI("Creating ray query resources...");
  if (!createRayQueryResources()) {
    LOGE("Failed to create ray query resources");
    return false;
  }
  LOGI("Ray query resources created successfully");

  // Note: Acceleration structure build is requested by scene_loading.cpp after entities load
  // No need to request it here during init

  // Light storage buffers were already created earlier to satisfy Forward+ binding requirements

  LOGI("Creating opaque scene color resources...");
  if (!createOpaqueSceneColorResources()) {
    LOGE("Failed to create opaque scene color resources");
    return false;
  }
  LOGI("Opaque scene color resources created successfully");

  LOGI("Creating transparent descriptor sets...");
  createTransparentDescriptorSets();
  LOGI("Transparent descriptor sets created");

  // Create default texture resources
  LOGI("Creating default texture resources...");
  if (!createDefaultTextureResources()) {
    LOGE("Failed to create default texture resources");
    return false;
  }
  LOGI("Default texture resources created successfully");

  // Create fallback transparent descriptor sets (must occur after default textures exist)
  LOGI("Creating fallback transparent descriptor sets...");
  createTransparentFallbackDescriptorSets();
  LOGI("Fallback transparent descriptor sets created");

  // Create shared default PBR textures (to avoid creating hundreds of identical textures)
  LOGI("Creating shared default PBR textures...");
  if (!createSharedDefaultPBRTextures()) {
    LOGE("Failed to create shared default PBR textures");
    return false;
  }
  LOGI("Shared default PBR textures created successfully");

  // Create command buffers
  LOGI("Creating command buffers...");
  if (!createCommandBuffers()) {
    LOGE("Failed to create command buffers");
    return false;
  }
  LOGI("Command buffers created successfully");

  // Create sync objects
  LOGI("Creating sync objects...");
  if (!createSyncObjects()) {
    LOGE("Failed to create sync objects");
    return false;
  }
  LOGI("Sync objects created successfully");

  // Initialize background thread pool for async tasks (textures, etc.) AFTER all Vulkan resources are ready
  try {
    // Size the thread pool based on hardware concurrency, clamped to a sensible range
    unsigned int hw = std::max(2u, std::min(8u, std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4u));
    threadPool = std::make_unique<ThreadPool>(hw);
  } catch (const std::exception& e) {
    LOGE("Failed to create thread pool: %s", e.what());
    return false;
  }

  // Start background uploads worker now that queues/semaphores exist
  StartUploadsWorker();

  // Start watchdog thread to detect application hangs
  lastFrameUpdateTime.store(std::chrono::steady_clock::now(), std::memory_order_relaxed);
  watchdogRunning.store(true, std::memory_order_relaxed);
  watchdogThread = std::thread(WatchdogThreadFunc, &lastFrameUpdateTime, &watchdogRunning, &watchdogSuppressed, &watchdogProgressLabel, &watchdogProgressIndex);

  std::cout << "[Watchdog] Started - will abort if no frame updates for 10+ seconds (60s during loading)\n";

  initialized = true;
  return true;
}

void Renderer::ensureThreadLocalVulkanInit() const {
  // Initialize Vulkan-Hpp dispatcher per-thread; required for multi-threaded RAII usage
  static thread_local bool s_tlsInitialized = false;
  if (s_tlsInitialized)
    return;
    // The dispatcher is global and initialized on the main thread during Renderer::Initialize.
    // Background threads inherit this global state. No per-thread init is required
    // for VULKAN_HPP_DEFAULT_DISPATCHER when using the default storage.
    s_tlsInitialized = true;
}

// Clean up renderer resources
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
bool Renderer::createInstance(const std::string& appName, bool enableValidationLayers) {
  try {
    // Create application info
    vk::ApplicationInfo appInfo{
      .pApplicationName = appName.c_str(),
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "Simple Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_3
    };

    // Get required extensions
    std::vector<const char *> extensions;

    // Add required extensions for GLFW
#if defined(PLATFORM_DESKTOP)
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    extensions.insert(extensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);
#elif defined(PLATFORM_ANDROID)
    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#endif

    // Add debug extension if validation layers are enabled
    if (enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Create instance info
    vk::InstanceCreateInfo createInfo{
      .pApplicationInfo = &appInfo,
      .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
      .ppEnabledExtensionNames = extensions.data()
    };

    // Enable validation layers if requested
    vk::ValidationFeaturesEXT validationFeatures{};
    std::vector<vk::ValidationFeatureEnableEXT> enabledValidationFeatures;

    bool actualEnableValidationLayers = enableValidationLayers;
    if (actualEnableValidationLayers) {
      if (!checkValidationLayerSupport()) {
        LOGW("Validation layers requested, but not available. Continuing without validation.");
        actualEnableValidationLayers = false;
      }
    }

    if (actualEnableValidationLayers) {
      createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();

      // Keep validation output quiet by default (no DebugPrintf feature).
      // Ray Query debugPrintf/printf diagnostics are intentionally removed.

      validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(enabledValidationFeatures.size());
      validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures.data();

      createInfo.pNext = &validationFeatures;
    }

    // Create instance
    instance = vk::raii::Instance(context, createInfo);
    // Initialize the dispatcher with the instance to load instance-level functions
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
    return true;
  } catch (const std::exception& e) {
    LOGE("Failed to create instance: %s", e.what());
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

    // Select callback: modern Vulkan-Hpp expects vk:: types.
    createInfo.pfnUserCallback = &debugCallbackVkHpp;

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
      LOGI("Checking device: %s (Type: %s)", deviceProperties.deviceName.data(), vk::to_string(deviceProperties.deviceType).c_str());

      // Check if the device supports Vulkan 1.3
      bool supportsVulkan1_3 = deviceProperties.apiVersion >= VK_API_VERSION_1_3;
      if (!supportsVulkan1_3) {
        LOGI("  - Does not support Vulkan 1.3");
        continue;
      }

      // Check queue families
      QueueFamilyIndices indices = findQueueFamilies(_device);
      bool supportsGraphics = indices.isComplete();
      if (!supportsGraphics) {
        LOGI("  - Missing required queue families");
        continue;
      }

      // Check device extensions
      bool supportsAllRequiredExtensions = checkDeviceExtensionSupport(_device);
      if (!supportsAllRequiredExtensions) {
        LOGI("  - Missing required extensions");
        continue;
      }

      // Check swap chain support
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_device);
      bool swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
      if (!swapChainAdequate) {
        LOGI("  - Inadequate swap chain support");
        continue;
      }

      // Check for required features
      auto features = _device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
      bool supportsRequiredFeatures = features.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering;
      if (!supportsRequiredFeatures) {
        LOGI("  - Does not support required features (dynamicRendering)");
        continue;
      }

      // Calculate suitability score - prioritize discrete GPUs
      int score = 0;

      // Discrete GPUs get the highest priority (NVIDIA RTX 2080, AMD, etc.)
      if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        score += 1000;
        LOGI("  - Discrete GPU: +1000 points");
      }
      // Integrated GPUs get lower priority (Intel UHD Graphics, etc.)
      else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
        score += 100;
        LOGI("  - Integrated GPU: +100 points");
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

      LOGI("  - Device is suitable with score: %d", score);
      suitableDevices.emplace(score, _device);
    }

    if (!suitableDevices.empty()) {
      // Select the device with the highest score (discrete GPU with most VRAM)
      physicalDevice = suitableDevices.rbegin()->second;
      vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
      LOGI("Selected device: %s (Type: %s, Score: %d)",
           deviceProperties.deviceName.data(),
           vk::to_string(deviceProperties.deviceType).c_str(),
           suitableDevices.rbegin()->first);

      // Store queue family indices for the selected device
      queueFamilyIndices = findQueueFamilies(physicalDevice);

      // Add supported optional extensions
      addSupportedOptionalExtensions();

      return true;
    }
    LOGE("Failed to find a suitable GPU. Make sure your GPU supports Vulkan and has the required extensions.");
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

    //add any extra extensions enabled by courses
#ifdef ENABLE_COURSE_OPACITY_MICROMAPS
    // Opacity micromap for hardware-accelerated alpha-tested shadow rays (Course: Opacity Micromaps)
    // vkCreateAccelerationStructure2KHR (KHR micromap build entry point) lives in
    // VK_KHR_device_address_commands — both must be present for the build path to work.
    optionalDeviceExtensions.push_back( VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME );
    optionalDeviceExtensions.push_back(VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME);
#endif

    // Build a set of available extension names for quick lookup
    std::set<std::string> avail;
    for (const auto& e : availableExtensions) {
      avail.insert(e.extensionName);
    }

    for (const auto& optionalExt : optionalDeviceExtensions) {
      if (avail.contains(optionalExt)) {
        deviceExtensions.push_back(optionalExt);
        std::cout << "Adding optional extension: " << optionalExt << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to add optional extensions: " << e.what() << std::endl;
  }
}

// Create logical device
bool Renderer::createLogicalDevice(bool enableValidationLayers) {
  LOGI("Entering createLogicalDevice");
  try {
    // Create queue create info for each unique queue family
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set uniqueQueueFamilies = {
      queueFamilyIndices.graphicsFamily.value(),
      queueFamilyIndices.presentFamily.value(),
      queueFamilyIndices.computeFamily.value(),
      queueFamilyIndices.transferFamily.value()
    };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueCreateInfo{
        .queueFamilyIndex = queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
      };
      queueCreateInfos.push_back(queueCreateInfo);
    }

    LOGI("Querying supported features...");
    // Query supported features before enabling them
    auto featureChainSupported = physicalDevice.getFeatures2<
      vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceTimelineSemaphoreFeatures,
      vk::PhysicalDeviceVulkanMemoryModelFeatures,
      vk::PhysicalDeviceBufferDeviceAddressFeatures,
      vk::PhysicalDevice8BitStorageFeatures,
      vk::PhysicalDeviceVulkan11Features,
      vk::PhysicalDeviceVulkan12Features,
      vk::PhysicalDeviceVulkan13Features,
      vk::PhysicalDeviceDescriptorIndexingFeatures,
      vk::PhysicalDeviceRobustness2FeaturesEXT,
      vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR,
      vk::PhysicalDeviceShaderTileImageFeaturesEXT,
      vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
      vk::PhysicalDeviceRayQueryFeaturesKHR>();
    LOGI("Features queried successfully");

    // Extract supported feature structs
    const auto& coreSupported = featureChainSupported.get<vk::PhysicalDeviceFeatures2>().features;
    const auto& timelineSupported = featureChainSupported.get<vk::PhysicalDeviceTimelineSemaphoreFeatures>();
    const auto& memoryModelSupported = featureChainSupported.get<vk::PhysicalDeviceVulkanMemoryModelFeatures>();
    const auto& bufferAddressSupported = featureChainSupported.get<vk::PhysicalDeviceBufferDeviceAddressFeatures>();
    const auto& storage8BitSupported = featureChainSupported.get<vk::PhysicalDevice8BitStorageFeatures>();
    const auto& vulkan11Supported = featureChainSupported.get<vk::PhysicalDeviceVulkan11Features>();
    const auto& vulkan13Supported = featureChainSupported.get<vk::PhysicalDeviceVulkan13Features>();
    const auto& indexingFeaturesSupported = featureChainSupported.get<vk::PhysicalDeviceDescriptorIndexingFeatures>();
    const auto& robust2Supported = featureChainSupported.get<vk::PhysicalDeviceRobustness2FeaturesEXT>();
    const auto& localReadSupported = featureChainSupported.get<vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>();
    const auto& tileImageSupported = featureChainSupported.get<vk::PhysicalDeviceShaderTileImageFeaturesEXT>();
    const auto& accelerationStructureSupported = featureChainSupported.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
    const auto& rayQuerySupported = featureChainSupported.get<vk::PhysicalDeviceRayQueryFeaturesKHR>();

    // Verify critical features are supported
    if (!coreSupported.samplerAnisotropy)
    LOGW("Missing feature: samplerAnisotropy");
    if (!timelineSupported.timelineSemaphore)
    LOGW("Missing feature: timelineSemaphore");
    if (!memoryModelSupported.vulkanMemoryModel)
    LOGW("Missing feature: vulkanMemoryModel");
    if (!bufferAddressSupported.bufferDeviceAddress)
    LOGW("Missing feature: bufferDeviceAddress");
    if (!vulkan13Supported.dynamicRendering)
    LOGW("Missing feature: dynamicRendering");
    if (!vulkan13Supported.synchronization2)
    LOGW("Missing feature: synchronization2");

    if (!coreSupported.samplerAnisotropy ||
      !timelineSupported.timelineSemaphore ||
      !memoryModelSupported.vulkanMemoryModel ||
      !bufferAddressSupported.bufferDeviceAddress ||
      !vulkan13Supported.dynamicRendering ||
      !vulkan13Supported.synchronization2) {
      throw std::runtime_error("Required Vulkan features not supported by physical device");
    }

    // Helper to check extension availability
    auto hasExtension = [&](const char* name) {
      return std::find_if(deviceExtensions.begin(),
                          deviceExtensions.end(),
                          [&](const char* ext) {
                            return std::strcmp(ext, name) == 0;
                          }) != deviceExtensions.end();
    };

    // Feature structures for the logical device
    vk::PhysicalDeviceFeatures2 features2{};
    features2.features.samplerAnisotropy = vk::True;
    features2.features.depthBiasClamp = coreSupported.depthBiasClamp ? vk::True : vk::False;
    if (coreSupported.shaderSampledImageArrayDynamicIndexing) {
      features2.features.shaderSampledImageArrayDynamicIndexing = vk::True;
    }

    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};
    timelineSemaphoreFeatures.timelineSemaphore = vk::True;

    vk::PhysicalDeviceVulkanMemoryModelFeatures memoryModelFeatures{};
    memoryModelFeatures.vulkanMemoryModel = vk::True;
    memoryModelFeatures.vulkanMemoryModelDeviceScope = memoryModelSupported.vulkanMemoryModelDeviceScope ? vk::True : vk::False;

    vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
    bufferDeviceAddressFeatures.bufferDeviceAddress = vk::True;

    vk::PhysicalDevice8BitStorageFeatures storage8BitFeatures{};
    storage8BitFeatures.storageBuffer8BitAccess = storage8BitSupported.storageBuffer8BitAccess ? vk::True : vk::False;

    vk::PhysicalDeviceVulkan11Features vulkan11Features{};
    if (vulkan11Supported.shaderDrawParameters) {
      vulkan11Features.shaderDrawParameters = vk::True;
    }

    vk::PhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.dynamicRendering = vk::True;
    vulkan13Features.synchronization2 = vk::True;

    vk::PhysicalDeviceDescriptorIndexingFeatures indexingFeaturesEnable{};
    descriptorIndexingEnabled = false;
    if (indexingFeaturesSupported.shaderSampledImageArrayNonUniformIndexing) {
      indexingFeaturesEnable.shaderSampledImageArrayNonUniformIndexing = vk::True;
      descriptorIndexingEnabled = true;
    }
    if (descriptorIndexingEnabled) {
      if (indexingFeaturesSupported.descriptorBindingPartiallyBound) indexingFeaturesEnable.descriptorBindingPartiallyBound = vk::True;
      if (indexingFeaturesSupported.descriptorBindingUpdateUnusedWhilePending) indexingFeaturesEnable.descriptorBindingUpdateUnusedWhilePending = vk::True;
      if (indexingFeaturesSupported.descriptorBindingSampledImageUpdateAfterBind) {
        indexingFeaturesEnable.descriptorBindingSampledImageUpdateAfterBind = vk::True;
        descriptorBindingSampledImageUpdateAfterBindEnabled = true;
      }
      if (indexingFeaturesSupported.descriptorBindingUniformBufferUpdateAfterBind) {
        indexingFeaturesEnable.descriptorBindingUniformBufferUpdateAfterBind = vk::True;
        descriptorBindingUniformBufferUpdateAfterBindEnabled = true;
      }
    }

    vk::PhysicalDeviceRobustness2FeaturesEXT robust2Enable{};
    bool hasRobust2 = hasExtension(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
    if (hasRobust2) {
      if (robust2Supported.robustBufferAccess2) robust2Enable.robustBufferAccess2 = vk::True;
      if (robust2Supported.robustImageAccess2) robust2Enable.robustImageAccess2 = vk::True;
      if (robust2Supported.nullDescriptor) robust2Enable.nullDescriptor = vk::True;
    }
    robustness2Enabled = hasRobust2 && (robust2Enable.robustBufferAccess2 || robust2Enable.robustImageAccess2 || robust2Enable.nullDescriptor);

    vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR localReadEnable{};
    bool hasLocalRead = hasExtension(VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME);
    if (hasLocalRead && localReadSupported.dynamicRenderingLocalRead) {
      localReadEnable.dynamicRenderingLocalRead = vk::True;
    }
    dynamicRenderingLocalReadEnabled = hasLocalRead && localReadEnable.dynamicRenderingLocalRead;

    vk::PhysicalDeviceShaderTileImageFeaturesEXT tileImageEnable{};
    bool hasTileImage = hasExtension(VK_EXT_SHADER_TILE_IMAGE_EXTENSION_NAME);
    if (hasTileImage && tileImageSupported.shaderTileImageColorReadAccess) {
      tileImageEnable.shaderTileImageColorReadAccess = vk::True;
    }
    shaderTileImageEnabled = hasTileImage && tileImageEnable.shaderTileImageColorReadAccess;

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR asFeaturesEnable{};
    bool hasAS = hasExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    if (hasAS && accelerationStructureSupported.accelerationStructure) {
      asFeaturesEnable.accelerationStructure = vk::True;
    }
    accelerationStructureEnabled = hasAS && asFeaturesEnable.accelerationStructure;

    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeaturesEnable{};
    bool hasRQ = hasExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    if (hasRQ && rayQuerySupported.rayQuery) {
      rayQueryFeaturesEnable.rayQuery = vk::True;
    }
    rayQueryEnabled = hasRQ && rayQueryFeaturesEnable.rayQuery;

    // Build the pNext chain
    features2.pNext = &timelineSemaphoreFeatures;
    timelineSemaphoreFeatures.pNext = &memoryModelFeatures;
    memoryModelFeatures.pNext = &bufferDeviceAddressFeatures;
    bufferDeviceAddressFeatures.pNext = &storage8BitFeatures;
    storage8BitFeatures.pNext = &vulkan11Features;
    vulkan11Features.pNext = &vulkan13Features;

    void** tailNext = reinterpret_cast<void **>(&vulkan13Features.pNext);
    if (descriptorIndexingEnabled) {
      *tailNext = &indexingFeaturesEnable;
      tailNext = reinterpret_cast<void **>(&indexingFeaturesEnable.pNext);
    }
    if (hasRobust2) {
      *tailNext = &robust2Enable;
      tailNext = reinterpret_cast<void **>(&robust2Enable.pNext);
    }
#if !defined(PLATFORM_ANDROID)
    if (hasLocalRead) {
      *tailNext = &localReadEnable;
      tailNext = reinterpret_cast<void **>(&localReadEnable.pNext);
    }
    if (hasTileImage) {
      *tailNext = &tileImageEnable;
      tailNext = reinterpret_cast<void **>(&tileImageEnable.pNext);
    }
#endif
    if (hasAS) {
      *tailNext = &asFeaturesEnable;
      tailNext = reinterpret_cast<void **>(&asFeaturesEnable.pNext);
    }
    if (hasRQ) {
      *tailNext = &rayQueryFeaturesEnable;
      tailNext = reinterpret_cast<void **>(&rayQueryFeaturesEnable.pNext);
    }

    // Opacity micromap — VK_KHR_opacity_micromap (Course: Opacity Micromaps)
    // Also requires VK_KHR_device_address_commands for vkCreateAccelerationStructure2KHR.
    auto hasOpacityMicromap = hasExtension(VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME)
        && hasExtension(VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME);
    vk::PhysicalDeviceOpacityMicromapFeaturesKHR opacityMicromapSupported{};
    vk::PhysicalDeviceOpacityMicromapFeaturesKHR opacityMicromapEnable{};
    if (hasOpacityMicromap) {
      auto featChain2 = physicalDevice.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceOpacityMicromapFeaturesKHR>();
      opacityMicromapSupported = featChain2.template get<vk::PhysicalDeviceOpacityMicromapFeaturesKHR>();
      if (opacityMicromapSupported.micromap) {
        opacityMicromapEnable.micromap = vk::True;
#ifdef ENABLE_COURSE_OPACITY_MICROMAPS
        opacityMicromapEnabled = true;
#endif
        *tailNext = &opacityMicromapEnable;
        tailNext = reinterpret_cast<void **>(&opacityMicromapEnable.pNext);
      }
    }

    // Record which features ended up enabled (for runtime decisions/tutorial diagnostics)
    robustness2Enabled = hasRobust2 && (robust2Enable.robustBufferAccess2 == vk::True ||
      robust2Enable.robustImageAccess2 == vk::True ||
      robust2Enable.nullDescriptor == vk::True);
#if !defined(PLATFORM_ANDROID)
    dynamicRenderingLocalReadEnabled = hasLocalRead && (localReadEnable.dynamicRenderingLocalRead == vk::True);
    shaderTileImageEnabled = hasTileImage && (tileImageEnable.shaderTileImageColorReadAccess == vk::True ||
      tileImageEnable.shaderTileImageDepthReadAccess == vk::True ||
      tileImageEnable.shaderTileImageStencilReadAccess == vk::True);
#else
    dynamicRenderingLocalReadEnabled = false;
    shaderTileImageEnabled = false;
#endif
    accelerationStructureEnabled = hasAS && (asFeaturesEnable.accelerationStructure == vk::True);
    rayQueryEnabled = hasRQ && (rayQueryFeaturesEnable.rayQuery == vk::True);

    // Create device info
    vk::DeviceCreateInfo createInfo{
      .pNext = &features2,
      .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
      .pQueueCreateInfos = queueCreateInfos.data(),
      .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
      .ppEnabledExtensionNames = deviceExtensions.data(),
      .pEnabledFeatures = nullptr
    };

    LOGI("Creating logical device...");
    device = vk::raii::Device(physicalDevice, createInfo);
    LOGI("Device created successfully");

    // Initialize the dispatcher with the device to load device-level functions
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

    // Get queue handles
    graphicsQueue = vk::raii::Queue(device, queueFamilyIndices.graphicsFamily.value(), 0);
    presentQueue = vk::raii::Queue(device, queueFamilyIndices.presentFamily.value(), 0);
    computeQueue = vk::raii::Queue(device, queueFamilyIndices.computeFamily.value(), 0);
    transferQueue = vk::raii::Queue(device, queueFamilyIndices.transferFamily.value(), 0);

    // Create global timeline semaphore for uploads early (needed before default texture creation)
    vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> timelineChain(
      {},
      {.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0});
    uploadsTimeline = vk::raii::Semaphore(device, timelineChain.get<vk::SemaphoreCreateInfo>());
    uploadTimelineLastSubmitted.store(0, std::memory_order_relaxed);

    return true;
  } catch (const std::exception& e) {
    LOGE("Failed to create logical device: %s", e.what());
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
