#include <algorithm>
#include <array>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN        // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

constexpr uint32_t WIDTH                = 800;
constexpr uint32_t HEIGHT               = 600;
const std::string  MODEL_PATH           = "models/viking_room.obj";
const std::string  TEXTURE_PATH         = "textures/viking_room.png";
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

// Validation layers are now managed by vulkanconfig instead of being hard-coded
// See the Ecosystem Utilities chapter for details on using vulkanconfig

// Application info structure to store feature support flags
struct AppInfo
{
	bool dynamicRenderingSupported   = false;
	bool timelineSemaphoresSupported = false;
	bool synchronization2Supported   = false;
};

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
	}

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
	{
		return {
		    vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
		    vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
		    vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))};
	}

	bool operator==(const Vertex &other) const
	{
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

template <>
struct std::hash<Vertex>
{
	size_t operator()(Vertex const &vertex) const noexcept
	{
		return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
	}
};

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication
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
	AppInfo appInfo;

	GLFWwindow                      *window = nullptr;
	vk::raii::Context                context;
	vk::raii::Instance               instance       = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::SurfaceKHR             surface        = nullptr;
	vk::raii::PhysicalDevice         physicalDevice = nullptr;
	vk::SampleCountFlagBits          msaaSamples    = vk::SampleCountFlagBits::e1;
	vk::raii::Device                 device         = nullptr;
	uint32_t                         queueIndex     = ~0;
	vk::raii::Queue                  queue          = nullptr;
	vk::raii::SwapchainKHR           swapChain      = nullptr;
	std::vector<vk::Image>           swapChainImages;
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;
	vk::Extent2D                     swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;

	// Traditional render pass (fallback for non-dynamic rendering)
	vk::raii::RenderPass               renderPass = nullptr;
	std::vector<vk::raii::Framebuffer> swapChainFramebuffers;

	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      pipelineLayout      = nullptr;
	vk::raii::Pipeline            graphicsPipeline    = nullptr;

	vk::raii::Image        colorImage       = nullptr;
	vk::raii::DeviceMemory colorImageMemory = nullptr;
	vk::raii::ImageView    colorImageView   = nullptr;

	vk::raii::Image        depthImage       = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView    depthImageView   = nullptr;

	uint32_t               mipLevels          = 0;
	vk::raii::Image        textureImage       = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView    textureImageView   = nullptr;
	vk::raii::Sampler      textureSampler     = nullptr;

	std::vector<Vertex>    vertices;
	std::vector<uint32_t>  indices;
	vk::raii::Buffer       vertexBuffer       = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	vk::raii::Buffer       indexBuffer        = nullptr;
	vk::raii::DeviceMemory indexBufferMemory  = nullptr;

	std::vector<vk::raii::Buffer>       uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void *>                 uniformBuffersMapped;

	vk::raii::DescriptorPool             descriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	vk::raii::CommandPool                commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	// Synchronization objects
	std::vector<vk::raii::Semaphore> presentCompleteSemaphore;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphore;
	std::vector<vk::raii::Fence>     inFlightFences;
	vk::raii::Semaphore              timelineSemaphore = nullptr;
	uint64_t                         timelineValue     = 0;
	uint32_t                         currentFrame      = 0;

	bool framebufferResized = false;

	std::vector<const char *> requiredDeviceExtension = {
	    vk::KHRSwapchainExtensionName};

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Compatibility Example", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
	{
		auto app                = static_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		detectFeatureSupport();
		createLogicalDevice();
		createSwapChain();
		createImageViews();

		// Create traditional render pass if dynamic rendering is not supported
		if (!appInfo.dynamicRenderingSupported)
		{
			createRenderPass();
			createFramebuffers();
		}

		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createColorResources();
		createDepthResources();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		loadModel();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();

		// Print feature support summary
		std::cout << "\nFeature support summary:\n";
		std::cout << "- Dynamic Rendering: " << (appInfo.dynamicRenderingSupported ? "Yes" : "No") << "\n";
		std::cout << "- Timeline Semaphores: " << (appInfo.timelineSemaphoresSupported ? "Yes" : "No") << "\n";
		std::cout << "- Synchronization2: " << (appInfo.synchronization2Supported ? "Yes" : "No") << "\n";
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
		swapChainFramebuffers.clear();
		swapChainImageViews.clear();
		swapChain = nullptr;
	}

	void cleanup() const
	{
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		device.waitIdle();

		cleanupSwapChain();
		createSwapChain();
		createImageViews();

		// Recreate traditional render pass and framebuffers if dynamic rendering is not supported
		if (!appInfo.dynamicRenderingSupported)
		{
			createRenderPass();
			createFramebuffers();
		}

		createColorResources();
		createDepthResources();
	}

	void createInstance()
	{
		// Validation layers are now managed by vulkanconfig instead of being hard-coded

		constexpr vk::ApplicationInfo appInfo{
		    .pApplicationName   = "Hello Triangle",
		    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		    .pEngineName        = "No Engine",
		    .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
		    .apiVersion         = vk::ApiVersion14};

		auto extensions = getRequiredExtensions();

		vk::InstanceCreateInfo createInfo{
		    .pApplicationInfo        = &appInfo,
		    .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
		    .ppEnabledExtensionNames = extensions.data()};

		instance = vk::raii::Instance(context, createInfo);
	}

	void setupDebugMessenger()
	{
		// Always set up the debug messenger
		// It will only be used if validation layers are enabled via vulkanconfig

		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
		    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
		    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);

		vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
		    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
		    vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		    vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

		vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
		    .messageSeverity = severityFlags,
		    .messageType     = messageTypeFlags,
		    .pfnUserCallback = &debugCallback};

		try
		{
			debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
		}
		catch (vk::SystemError &err)
		{
			// If the debug utils extension is not available, this will fail
			// That's okay; it just means validation layers aren't enabled
			std::cout << "Debug messenger not available. Validation layers may not be enabled." << std::endl;
		}
	}

	void createSurface()
	{
		VkSurfaceKHR _surface;
		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		{
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	void pickPhysicalDevice()
	{
		std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
		const auto                            devIter = std::ranges::find_if(
            devices,
            [&](auto const &device) {
                // Check if any of the queue families support graphics operations
                auto queueFamilies = device.getQueueFamilyProperties();
                bool supportsGraphics =
                    std::ranges::any_of(queueFamilies, [](auto const &qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });

                // Check if all required device extensions are available
                auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
                bool supportsAllRequiredExtensions =
                    std::ranges::all_of(requiredDeviceExtension,
			                                                       [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
                                            return std::ranges::any_of(availableDeviceExtensions,
				                                                                                  [requiredDeviceExtension](auto const &availableDeviceExtension) { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
                                        });

                return supportsGraphics && supportsAllRequiredExtensions;
            });
		if (devIter != devices.end())
		{
			physicalDevice = *devIter;
			msaaSamples    = getMaxUsableSampleCount();
		}
		else
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	void detectFeatureSupport()
	{
		// Get device properties to check Vulkan version
		vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();

		// Get available extensions
		std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

		// Check for dynamic rendering support
		if (deviceProperties.apiVersion >= VK_VERSION_1_3)
		{
			appInfo.dynamicRenderingSupported = true;
			std::cout << "Dynamic rendering supported via Vulkan 1.3\n";
		}
		else
		{
			// Check for the extension on older Vulkan versions
			for (const auto &extension : availableExtensions)
			{
				if (strcmp(extension.extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0)
				{
					appInfo.dynamicRenderingSupported = true;
					std::cout << "Dynamic rendering supported via extension\n";
					break;
				}
			}
		}

		// Check for timeline semaphores support
		if (deviceProperties.apiVersion >= VK_VERSION_1_2)
		{
			appInfo.timelineSemaphoresSupported = true;
			std::cout << "Timeline semaphores supported via Vulkan 1.2\n";
		}
		else
		{
			// Check for the extension on older Vulkan versions
			for (const auto &extension : availableExtensions)
			{
				if (strcmp(extension.extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0)
				{
					appInfo.timelineSemaphoresSupported = true;
					std::cout << "Timeline semaphores supported via extension\n";
					break;
				}
			}
		}

		// Check for synchronization2 support
		if (deviceProperties.apiVersion >= VK_VERSION_1_3)
		{
			appInfo.synchronization2Supported = true;
			std::cout << "Synchronization2 supported via Vulkan 1.3\n";
		}
		else
		{
			// Check for the extension on older Vulkan versions
			for (const auto &extension : availableExtensions)
			{
				if (strcmp(extension.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0)
				{
					appInfo.synchronization2Supported = true;
					std::cout << "Synchronization2 supported via extension\n";
					break;
				}
			}
		}

		// Add required extensions based on feature support
		if (appInfo.dynamicRenderingSupported && deviceProperties.apiVersion < VK_VERSION_1_3)
		{
			requiredDeviceExtension.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
		}

		if (appInfo.timelineSemaphoresSupported && deviceProperties.apiVersion < VK_VERSION_1_2)
		{
			requiredDeviceExtension.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
		}

		if (appInfo.synchronization2Supported && deviceProperties.apiVersion < VK_VERSION_1_3)
		{
			requiredDeviceExtension.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
		}
	}

	void createLogicalDevice()
	{
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		// get the first index into queueFamilyProperties which supports both graphics and present
		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
		{
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
			    physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
			{
				// found a queue family that supports both graphics and present
				queueIndex = qfpIndex;
				break;
			}
		}
		if (queueIndex == ~0)
		{
			throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
		}

		// Create device with appropriate features
		auto features = physicalDevice.getFeatures2();

		// Setup feature chain based on detected support
		void *pNext = nullptr;

		// Add dynamic rendering if supported
		vk::PhysicalDeviceVulkan13Features         vulkan13Features;
		vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures;

		if (appInfo.dynamicRenderingSupported)
		{
			if (appInfo.synchronization2Supported)
			{
				vulkan13Features.dynamicRendering = vk::True;
				vulkan13Features.synchronization2 = vk::True;
				vulkan13Features.pNext            = pNext;
				pNext                             = &vulkan13Features;
			}
			else
			{
				dynamicRenderingFeatures.dynamicRendering = vk::True;
				dynamicRenderingFeatures.pNext            = pNext;
				pNext                                     = &dynamicRenderingFeatures;
			}
		}

		// Add timeline semaphores if supported
		vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures;
		if (appInfo.timelineSemaphoresSupported)
		{
			timelineSemaphoreFeatures.timelineSemaphore = vk::True;
			timelineSemaphoreFeatures.pNext             = pNext;
			pNext                                       = &timelineSemaphoreFeatures;
		}

		features.pNext = pNext;

		// create a Device
		float                     queuePriority = 0.5f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
		vk::DeviceCreateInfo      deviceCreateInfo{
		         .pNext                   = &features,
		         .queueCreateInfoCount    = 1,
		         .pQueueCreateInfos       = &deviceQueueCreateInfo,
		         .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtension.size()),
		         .ppEnabledExtensionNames = requiredDeviceExtension.data()};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		queue  = vk::raii::Queue(device, queueIndex, 0);
	}

	void createSwapChain()
	{
		auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		swapChainExtent          = chooseSwapExtent(surfaceCapabilities);
		swapChainSurfaceFormat   = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));
		vk::SwapchainCreateInfoKHR swapChainCreateInfo{.surface          = *surface,
		                                               .minImageCount    = chooseSwapMinImageCount(surfaceCapabilities),
		                                               .imageFormat      = swapChainSurfaceFormat.format,
		                                               .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
		                                               .imageExtent      = swapChainExtent,
		                                               .imageArrayLayers = 1,
		                                               .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
		                                               .imageSharingMode = vk::SharingMode::eExclusive,
		                                               .preTransform     = surfaceCapabilities.currentTransform,
		                                               .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
		                                               .presentMode      = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(*surface)),
		                                               .clipped          = true};

		swapChain       = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	void createImageViews()
	{
		assert(swapChainImageViews.empty());

		vk::ImageViewCreateInfo imageViewCreateInfo{
		    .viewType         = vk::ImageViewType::e2D,
		    .format           = swapChainSurfaceFormat.format,
		    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
		for (auto image : swapChainImages)
		{
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}
	}

	void createRenderPass()
	{
		if (appInfo.dynamicRenderingSupported)
		{
			// No render pass needed with dynamic rendering
			std::cout << "Using dynamic rendering, skipping render pass creation\n";
			return;
		}

		std::cout << "Creating traditional render pass\n";

		// Color attachment description
		vk::AttachmentDescription colorAttachment{
		    .format         = swapChainSurfaceFormat.format,
		    .samples        = msaaSamples,
		    .loadOp         = vk::AttachmentLoadOp::eClear,
		    .storeOp        = vk::AttachmentStoreOp::eStore,
		    .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
		    .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		    .initialLayout  = vk::ImageLayout::eUndefined,
		    .finalLayout    = vk::ImageLayout::eColorAttachmentOptimal};

		vk::AttachmentDescription depthAttachment{
		    .format         = findDepthFormat(),
		    .samples        = msaaSamples,
		    .loadOp         = vk::AttachmentLoadOp::eClear,
		    .storeOp        = vk::AttachmentStoreOp::eDontCare,
		    .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
		    .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		    .initialLayout  = vk::ImageLayout::eUndefined,
		    .finalLayout    = vk::ImageLayout::eDepthStencilAttachmentOptimal};

		vk::AttachmentDescription colorAttachmentResolve{
		    .format         = swapChainSurfaceFormat.format,
		    .samples        = vk::SampleCountFlagBits::e1,
		    .loadOp         = vk::AttachmentLoadOp::eDontCare,
		    .storeOp        = vk::AttachmentStoreOp::eStore,
		    .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
		    .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		    .initialLayout  = vk::ImageLayout::eUndefined,
		    .finalLayout    = vk::ImageLayout::ePresentSrcKHR};

		// Subpass references
		vk::AttachmentReference colorAttachmentRef{
		    .attachment = 0,
		    .layout     = vk::ImageLayout::eColorAttachmentOptimal};

		vk::AttachmentReference depthAttachmentRef{
		    .attachment = 1,
		    .layout     = vk::ImageLayout::eDepthStencilAttachmentOptimal};

		vk::AttachmentReference colorAttachmentResolveRef{
		    .attachment = 2,
		    .layout     = vk::ImageLayout::eColorAttachmentOptimal};

		// Subpass description
		vk::SubpassDescription subpass{
		    .pipelineBindPoint       = vk::PipelineBindPoint::eGraphics,
		    .colorAttachmentCount    = 1,
		    .pColorAttachments       = &colorAttachmentRef,
		    .pResolveAttachments     = &colorAttachmentResolveRef,
		    .pDepthStencilAttachment = &depthAttachmentRef};

		// Dependency to ensure proper image layout transitions
		vk::SubpassDependency dependency{
		    .srcSubpass    = VK_SUBPASS_EXTERNAL,
		    .dstSubpass    = 0,
		    .srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
		    .dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
		    .srcAccessMask = vk::AccessFlagBits::eNone,
		    .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite};

		// Create the render pass
		std::array               attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
		vk::RenderPassCreateInfo renderPassInfo{
		    .attachmentCount = static_cast<uint32_t>(attachments.size()),
		    .pAttachments    = attachments.data(),
		    .subpassCount    = 1,
		    .pSubpasses      = &subpass,
		    .dependencyCount = 1,
		    .pDependencies   = &dependency};

		renderPass = vk::raii::RenderPass(device, renderPassInfo);
	}

	void createFramebuffers()
	{
		if (appInfo.dynamicRenderingSupported)
		{
			// No framebuffers needed with dynamic rendering
			std::cout << "Using dynamic rendering, skipping framebuffer creation\n";
			return;
		}

		std::cout << "Creating traditional framebuffers\n";

		swapChainFramebuffers.clear();

		for (size_t i = 0; i < swapChainImageViews.size(); i++)
		{
			std::array attachments = {
			    *colorImageView,
			    *depthImageView,
			    *swapChainImageViews[i]};

			vk::FramebufferCreateInfo framebufferInfo{
			    .renderPass      = *renderPass,
			    .attachmentCount = static_cast<uint32_t>(attachments.size()),
			    .pAttachments    = attachments.data(),
			    .width           = swapChainExtent.width,
			    .height          = swapChainExtent.height,
			    .layers          = 1};

			swapChainFramebuffers.emplace_back(device, framebufferInfo);
		}
	}

	void createDescriptorSetLayout()
	{
		std::array bindings = {
		    vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
		    vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	}

	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		auto                                   bindingDescription    = Vertex::getBindingDescription();
		auto                                   attributeDescriptions = Vertex::getAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		    .vertexBindingDescriptionCount   = 1,
		    .pVertexBindingDescriptions      = &bindingDescription,
		    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
		    .pVertexAttributeDescriptions    = attributeDescriptions.data()};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
		    .topology               = vk::PrimitiveTopology::eTriangleList,
		    .primitiveRestartEnable = vk::False};
		vk::PipelineViewportStateCreateInfo viewportState{
		    .viewportCount = 1,
		    .scissorCount  = 1};
		vk::PipelineRasterizationStateCreateInfo rasterizer{
		    .depthClampEnable        = vk::False,
		    .rasterizerDiscardEnable = vk::False,
		    .polygonMode             = vk::PolygonMode::eFill,
		    .cullMode                = vk::CullModeFlagBits::eBack,
		    .frontFace               = vk::FrontFace::eCounterClockwise,
		    .depthBiasEnable         = vk::False};
		rasterizer.lineWidth = 1.0f;
		vk::PipelineMultisampleStateCreateInfo multisampling{
		    .rasterizationSamples = msaaSamples,
		    .sampleShadingEnable  = vk::False};
		vk::PipelineDepthStencilStateCreateInfo depthStencil{
		    .depthTestEnable       = vk::True,
		    .depthWriteEnable      = vk::True,
		    .depthCompareOp        = vk::CompareOp::eLess,
		    .depthBoundsTestEnable = vk::False,
		    .stencilTestEnable     = vk::False};
		vk::PipelineColorBlendAttachmentState colorBlendAttachment;
		colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		colorBlendAttachment.blendEnable    = vk::False;

		vk::PipelineColorBlendStateCreateInfo colorBlending{
		    .logicOpEnable   = vk::False,
		    .logicOp         = vk::LogicOp::eCopy,
		    .attachmentCount = 1,
		    .pAttachments    = &colorBlendAttachment};

		std::vector dynamicStates = {
		    vk::DynamicState::eViewport,
		    vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0};

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		    {.stageCount          = 2,
		     .pStages             = shaderStages,
		     .pVertexInputState   = &vertexInputInfo,
		     .pInputAssemblyState = &inputAssembly,
		     .pViewportState      = &viewportState,
		     .pRasterizationState = &rasterizer,
		     .pMultisampleState   = &multisampling,
		     .pDepthStencilState  = &depthStencil,
		     .pColorBlendState    = &colorBlending,
		     .pDynamicState       = &dynamicState,
		     .layout              = pipelineLayout,
		     .renderPass          = nullptr},
		    {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format, .depthAttachmentFormat = findDepthFormat()}};

		if (appInfo.dynamicRenderingSupported)
		{
			std::cout << "Configuring pipeline for dynamic rendering\n";
		}
		else
		{
			std::cout << "Configuring pipeline for traditional render pass\n";
			pipelineCreateInfoChain.unlink<vk::PipelineRenderingCreateInfo>();
			pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().renderPass = *renderPass;
		}

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{
		    .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		    .queueFamilyIndex = queueIndex};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	void createColorResources()
	{
		vk::Format colorFormat = swapChainSurfaceFormat.format;

		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage, colorImageMemory);
		colorImageView = createImageView(colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
	}

	void createDepthResources()
	{
		vk::Format depthFormat = findDepthFormat();

		createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
	}

	vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const
	{
		for (const auto format : candidates)
		{
			vk::FormatProperties props = physicalDevice.getFormatProperties(format);

			if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
			{
				return format;
			}
			if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
			{
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	[[nodiscard]] vk::Format findDepthFormat() const
	{
		return findSupportedFormat(
		    {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
		    vk::ImageTiling::eOptimal,
		    vk::FormatFeatureFlagBits::eDepthStencilAttachment);
	}

	static bool hasStencilComponent(vk::Format format)
	{
		return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
	}

	void createTextureImage()
	{
		int            texWidth, texHeight, texChannels;
		stbi_uc       *pixels    = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		vk::DeviceSize imageSize = texWidth * texHeight * 4;
		mipLevels                = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

		if (!pixels)
		{
			throw std::runtime_error("failed to load texture image!");
		}

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void *data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, pixels, imageSize);
		stagingBufferMemory.unmapMemory();

		stbi_image_free(pixels);

		createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

		transitionImageLayout(textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

		generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
	}

	void generateMipmaps(vk::raii::Image &image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels)
	{
		// Check if image format supports linear blit-ing
		vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);

		if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
		{
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();

		vk::ImageMemoryBarrier barrier          = {.srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eTransferRead, .oldLayout = vk::ImageLayout::eTransferDstOptimal, .newLayout = vk::ImageLayout::eTransferSrcOptimal, .srcQueueFamilyIndex = vk::QueueFamilyIgnored, .dstQueueFamilyIndex = vk::QueueFamilyIgnored, .image = image};
		barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount     = 1;
		barrier.subresourceRange.levelCount     = 1;

		int32_t mipWidth  = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++)
		{
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout                     = vk::ImageLayout::eTransferDstOptimal;
			barrier.newLayout                     = vk::ImageLayout::eTransferSrcOptimal;
			barrier.srcAccessMask                 = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask                 = vk::AccessFlagBits::eTransferRead;

			commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

			vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
			offsets[0]          = vk::Offset3D(0, 0, 0);
			offsets[1]          = vk::Offset3D(mipWidth, mipHeight, 1);
			dstOffsets[0]       = vk::Offset3D(0, 0, 0);
			dstOffsets[1]       = vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1);
			vk::ImageBlit blit  = {.srcSubresource = {}, .srcOffsets = offsets, .dstSubresource = {}, .dstOffsets = dstOffsets};
			blit.srcSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
			blit.dstSubresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1);

			commandBuffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, {blit}, vk::Filter::eLinear);

			barrier.oldLayout     = vk::ImageLayout::eTransferSrcOptimal;
			barrier.newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal;
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

			if (mipWidth > 1)
				mipWidth /= 2;
			if (mipHeight > 1)
				mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout                     = vk::ImageLayout::eTransferDstOptimal;
		barrier.newLayout                     = vk::ImageLayout::eShaderReadOnlyOptimal;
		barrier.srcAccessMask                 = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask                 = vk::AccessFlagBits::eShaderRead;

		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

		endSingleTimeCommands(*commandBuffer);
	}

	vk::SampleCountFlagBits getMaxUsableSampleCount()
	{
		vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice.getProperties();

		vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & vk::SampleCountFlagBits::e64)
		{
			return vk::SampleCountFlagBits::e64;
		}
		if (counts & vk::SampleCountFlagBits::e32)
		{
			return vk::SampleCountFlagBits::e32;
		}
		if (counts & vk::SampleCountFlagBits::e16)
		{
			return vk::SampleCountFlagBits::e16;
		}
		if (counts & vk::SampleCountFlagBits::e8)
		{
			return vk::SampleCountFlagBits::e8;
		}
		if (counts & vk::SampleCountFlagBits::e4)
		{
			return vk::SampleCountFlagBits::e4;
		}
		if (counts & vk::SampleCountFlagBits::e2)
		{
			return vk::SampleCountFlagBits::e2;
		}

		return vk::SampleCountFlagBits::e1;
	}

	void createTextureImageView()
	{
		textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
	}

	void createTextureSampler()
	{
		vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
		vk::SamplerCreateInfo        samplerInfo{
		           .magFilter        = vk::Filter::eLinear,
		           .minFilter        = vk::Filter::eLinear,
		           .mipmapMode       = vk::SamplerMipmapMode::eLinear,
		           .addressModeU     = vk::SamplerAddressMode::eRepeat,
		           .addressModeV     = vk::SamplerAddressMode::eRepeat,
		           .addressModeW     = vk::SamplerAddressMode::eRepeat,
		           .mipLodBias       = 0.0f,
		           .anisotropyEnable = vk::True,
		           .maxAnisotropy    = properties.limits.maxSamplerAnisotropy,
		           .compareEnable    = vk::False,
		           .compareOp        = vk::CompareOp::eAlways};
		textureSampler = vk::raii::Sampler(device, samplerInfo);
	}

	[[nodiscard]] vk::raii::ImageView createImageView(const vk::raii::Image &image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const
	{
		vk::ImageViewCreateInfo viewInfo{
		    .image            = image,
		    .viewType         = vk::ImageViewType::e2D,
		    .format           = format,
		    .subresourceRange = {aspectFlags, 0, mipLevels, 0, 1}};
		return vk::raii::ImageView(device, viewInfo);
	}

	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image &image, vk::raii::DeviceMemory &imageMemory)
	{
		vk::ImageCreateInfo imageInfo{
		    .imageType     = vk::ImageType::e2D,
		    .format        = format,
		    .extent        = {width, height, 1},
		    .mipLevels     = mipLevels,
		    .arrayLayers   = 1,
		    .samples       = numSamples,
		    .tiling        = tiling,
		    .usage         = usage,
		    .sharingMode   = vk::SharingMode::eExclusive,
		    .initialLayout = vk::ImageLayout::eUndefined};
		image = vk::raii::Image(device, imageInfo);

		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
		    .allocationSize  = memRequirements.size,
		    .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
		imageMemory = vk::raii::DeviceMemory(device, allocInfo);
		image.bindMemory(imageMemory, 0);
	}

	void transitionImageLayout(const vk::raii::Image &image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, uint32_t mipLevels)
	{
		const auto commandBuffer = beginSingleTimeCommands();

		if (appInfo.synchronization2Supported)
		{
			// Use Synchronization2 API
			vk::ImageMemoryBarrier2 barrier{
			    .srcStageMask     = vk::PipelineStageFlagBits2::eAllCommands,
			    .dstStageMask     = vk::PipelineStageFlagBits2::eAllCommands,
			    .oldLayout        = oldLayout,
			    .newLayout        = newLayout,
			    .image            = image,
			    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1}};

			if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
			{
				barrier.srcAccessMask = vk::AccessFlagBits2::eNone;
				barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
				barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTopOfPipe;
				barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
			}
			else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
			{
				barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
				barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
				barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
				barrier.dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader;
			}
			else
			{
				throw std::invalid_argument("unsupported layout transition!");
			}

			vk::DependencyInfo dependencyInfo{
			    .imageMemoryBarrierCount = 1,
			    .pImageMemoryBarriers    = &barrier};

			commandBuffer->pipelineBarrier2(dependencyInfo);
		}
		else
		{
			// Use traditional synchronization API
			vk::ImageMemoryBarrier barrier{
			    .oldLayout        = oldLayout,
			    .newLayout        = newLayout,
			    .image            = image,
			    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1}};

			vk::PipelineStageFlags sourceStage;
			vk::PipelineStageFlags destinationStage;

			if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
			{
				barrier.srcAccessMask = {};
				barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

				sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;
				destinationStage = vk::PipelineStageFlagBits::eTransfer;
			}
			else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
			{
				barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
				barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

				sourceStage      = vk::PipelineStageFlagBits::eTransfer;
				destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
			}
			else
			{
				throw std::invalid_argument("unsupported layout transition!");
			}
			commandBuffer->pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
		}

		endSingleTimeCommands(*commandBuffer);
	}

	void copyBufferToImage(const vk::raii::Buffer &buffer, const vk::raii::Image &image, uint32_t width, uint32_t height)
	{
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
		vk::BufferImageCopy                      region{
		                         .bufferOffset      = 0,
		                         .bufferRowLength   = 0,
		                         .bufferImageHeight = 0,
		                         .imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
		                         .imageOffset       = {0, 0, 0},
		                         .imageExtent       = {width, height, 1}};
		commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
		endSingleTimeCommands(*commandBuffer);
	}

	void loadModel()
	{
		tinyobj::attrib_t                attrib;
		std::vector<tinyobj::shape_t>    shapes;
		std::vector<tinyobj::material_t> materials;
		std::string                      warn, err;

		if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
		{
			throw std::runtime_error(warn + err);
		}

		std::unordered_map<Vertex, uint32_t> uniqueVertices{};

		for (const auto &shape : shapes)
		{
			for (const auto &index : shape.mesh.indices)
			{
				Vertex vertex{};

				vertex.pos = {
				    attrib.vertices[3 * index.vertex_index + 0],
				    attrib.vertices[3 * index.vertex_index + 1],
				    attrib.vertices[3 * index.vertex_index + 2]};

				vertex.texCoord = {
				    attrib.texcoords[2 * index.texcoord_index + 0],
				    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};

				vertex.color = {1.0f, 1.0f, 1.0f};

				if (!uniqueVertices.contains(vertex))
				{
					uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
					vertices.push_back(vertex);
				}

				indices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	void createVertexBuffer()
	{
		vk::DeviceSize         bufferSize = sizeof(vertices[0]) * vertices.size();
		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, vertices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
	}

	void createIndexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void *data = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(data, indices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
	}

	void createUniformBuffers()
	{
		uniformBuffers.clear();
		uniformBuffersMemory.clear();
		uniformBuffersMapped.clear();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DeviceSize         bufferSize = sizeof(UniformBufferObject);
			vk::raii::Buffer       buffer({});
			vk::raii::DeviceMemory bufferMem({});
			createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
			uniformBuffers.emplace_back(std::move(buffer));
			uniformBuffersMemory.emplace_back(std::move(bufferMem));
			uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
		}
	}

	void createDescriptorPool()
	{
		std::array poolSize{
		    vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
		    vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT)};
		vk::DescriptorPoolCreateInfo poolInfo{
		    .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		    .maxSets       = MAX_FRAMES_IN_FLIGHT,
		    .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
		    .pPoolSizes    = poolSize.data()};
		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
	}

	void createDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{
		           .descriptorPool     = descriptorPool,
		           .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		           .pSetLayouts        = layouts.data()};

		descriptorSets.clear();
		descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DescriptorBufferInfo bufferInfo{
			    .buffer = uniformBuffers[i],
			    .offset = 0,
			    .range  = sizeof(UniformBufferObject)};
			vk::DescriptorImageInfo imageInfo{
			    .sampler     = textureSampler,
			    .imageView   = textureImageView,
			    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
			std::array descriptorWrites{
			    vk::WriteDescriptorSet{
			        .dstSet          = descriptorSets[i],
			        .dstBinding      = 0,
			        .dstArrayElement = 0,
			        .descriptorCount = 1,
			        .descriptorType  = vk::DescriptorType::eUniformBuffer,
			        .pBufferInfo     = &bufferInfo},
			    vk::WriteDescriptorSet{
			        .dstSet          = descriptorSets[i],
			        .dstBinding      = 1,
			        .dstArrayElement = 0,
			        .descriptorCount = 1,
			        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
			        .pImageInfo      = &imageInfo}};
			device.updateDescriptorSets(descriptorWrites, {});
		}
	}

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory)
	{
		vk::BufferCreateInfo bufferInfo{
		    .size        = size,
		    .usage       = usage,
		    .sharingMode = vk::SharingMode::eExclusive};
		buffer                                 = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
		    .allocationSize  = memRequirements.size,
		    .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
		bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
		buffer.bindMemory(bufferMemory, 0);
	}

	std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands()
	{
		vk::CommandBufferAllocateInfo allocInfo{
		    .commandPool        = commandPool,
		    .level              = vk::CommandBufferLevel::ePrimary,
		    .commandBufferCount = 1};
		std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers(device, allocInfo).front()));

		vk::CommandBufferBeginInfo beginInfo{
		    .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
		commandBuffer->begin(beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(const vk::raii::CommandBuffer &commandBuffer) const
	{
		commandBuffer.end();

		vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
		queue.submit(submitInfo, nullptr);
		queue.waitIdle();
	}

	void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size)
	{
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
		vk::raii::CommandBuffer       commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
		commandCopyBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.size = size});
		commandCopyBuffer.end();
		queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
		queue.waitIdle();
	}

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createCommandBuffers()
	{
		commandBuffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	void recordCommandBuffer(uint32_t imageIndex)
	{
		commandBuffers[currentFrame].begin({});

		vk::ClearValue                clearColor  = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::ClearValue                clearDepth  = vk::ClearDepthStencilValue(1.0f, 0);
		std::array<vk::ClearValue, 2> clearValues = {clearColor, clearDepth};

		if (appInfo.dynamicRenderingSupported)
		{
			// Transition attachments to the correct layout
			if (appInfo.synchronization2Supported)
			{
				// Use Synchronization2 API for image transitions
				vk::ImageMemoryBarrier2 colorBarrier{
				    .srcStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				    .srcAccessMask    = vk::AccessFlagBits2::eColorAttachmentWrite,
				    .dstStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				    .dstAccessMask    = vk::AccessFlagBits2::eColorAttachmentWrite,
				    .oldLayout        = vk::ImageLayout::eUndefined,
				    .newLayout        = vk::ImageLayout::eColorAttachmentOptimal,
				    .image            = *colorImage,
				    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

				vk::ImageMemoryBarrier2 depthBarrier{
				    .srcStageMask     = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
				    .srcAccessMask    = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
				    .dstStageMask     = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
				    .dstAccessMask    = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
				    .oldLayout        = vk::ImageLayout::eUndefined,
				    .newLayout        = vk::ImageLayout::eDepthStencilAttachmentOptimal,
				    .image            = *depthImage,
				    .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}};

				vk::ImageMemoryBarrier2 swapchainBarrier{
				    .srcStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				    .srcAccessMask    = vk::AccessFlagBits2::eNone,
				    .dstStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				    .dstAccessMask    = vk::AccessFlagBits2::eColorAttachmentWrite,
				    .oldLayout        = vk::ImageLayout::eUndefined,
				    .newLayout        = vk::ImageLayout::eColorAttachmentOptimal,
				    .image            = swapChainImages[imageIndex],
				    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

				std::array<vk::ImageMemoryBarrier2, 3> barriers = {colorBarrier, depthBarrier, swapchainBarrier};
				vk::DependencyInfo                     dependencyInfo{
				                        .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
				                        .pImageMemoryBarriers    = barriers.data()};

				commandBuffers[currentFrame].pipelineBarrier2(dependencyInfo);
			}
			else
			{
				// Use traditional synchronization API
				vk::ImageMemoryBarrier colorBarrier{
				    .srcAccessMask       = vk::AccessFlagBits::eNone,
				    .dstAccessMask       = vk::AccessFlagBits::eColorAttachmentWrite,
				    .oldLayout           = vk::ImageLayout::eUndefined,
				    .newLayout           = vk::ImageLayout::eColorAttachmentOptimal,
				    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				    .image               = *colorImage,
				    .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

				vk::ImageMemoryBarrier depthBarrier{
				    .srcAccessMask       = vk::AccessFlagBits::eNone,
				    .dstAccessMask       = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
				    .oldLayout           = vk::ImageLayout::eUndefined,
				    .newLayout           = vk::ImageLayout::eDepthStencilAttachmentOptimal,
				    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				    .image               = *depthImage,
				    .subresourceRange    = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}};

				vk::ImageMemoryBarrier swapchainBarrier{
				    .srcAccessMask       = vk::AccessFlagBits::eNone,
				    .dstAccessMask       = vk::AccessFlagBits::eColorAttachmentWrite,
				    .oldLayout           = vk::ImageLayout::eUndefined,
				    .newLayout           = vk::ImageLayout::eColorAttachmentOptimal,
				    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				    .image               = swapChainImages[imageIndex],
				    .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

				std::array<vk::ImageMemoryBarrier, 3> barriers = {colorBarrier, depthBarrier, swapchainBarrier};
				commandBuffers[currentFrame].pipelineBarrier(
				    vk::PipelineStageFlagBits::eTopOfPipe,
				    vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
				    vk::DependencyFlagBits::eByRegion,
				    {},
				    {},
				    barriers);
			}

			// Setup rendering attachments
			vk::RenderingAttachmentInfo colorAttachment{
			    .imageView          = *colorImageView,
			    .imageLayout        = vk::ImageLayout::eColorAttachmentOptimal,
			    .resolveMode        = vk::ResolveModeFlagBits::eAverage,
			    .resolveImageView   = *swapChainImageViews[imageIndex],
			    .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			    .loadOp             = vk::AttachmentLoadOp::eClear,
			    .storeOp            = vk::AttachmentStoreOp::eStore,
			    .clearValue         = clearColor};

			vk::RenderingAttachmentInfo depthAttachment{
			    .imageView   = *depthImageView,
			    .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
			    .loadOp      = vk::AttachmentLoadOp::eClear,
			    .storeOp     = vk::AttachmentStoreOp::eDontCare,
			    .clearValue  = clearDepth};

			vk::RenderingInfo renderingInfo{
			    .renderArea           = {{0, 0}, swapChainExtent},
			    .layerCount           = 1,
			    .colorAttachmentCount = 1,
			    .pColorAttachments    = &colorAttachment,
			    .pDepthAttachment     = &depthAttachment};

			commandBuffers[currentFrame].beginRendering(renderingInfo);
		}
		else
		{
			// Use traditional render pass
			std::cout << "Recording command buffer with traditional render pass\n";

			vk::RenderPassBeginInfo renderPassInfo{
			    .renderPass      = *renderPass,
			    .framebuffer     = *swapChainFramebuffers[imageIndex],
			    .renderArea      = {{0, 0}, swapChainExtent},
			    .clearValueCount = static_cast<uint32_t>(clearValues.size()),
			    .pClearValues    = clearValues.data()};

			commandBuffers[currentFrame].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
		}

		// Common rendering commands
		commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
		commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
		commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
		commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, {0});
		commandBuffers[currentFrame].bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
		commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
		commandBuffers[currentFrame].drawIndexed(indices.size(), 1, 0, 0, 0);

		if (appInfo.dynamicRenderingSupported)
		{
			commandBuffers[currentFrame].endRendering();

			// Transition swapchain image to present layout
			if (appInfo.synchronization2Supported)
			{
				vk::ImageMemoryBarrier2 barrier{
				    .srcStageMask     = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				    .srcAccessMask    = vk::AccessFlagBits2::eColorAttachmentWrite,
				    .dstStageMask     = vk::PipelineStageFlagBits2::eBottomOfPipe,
				    .dstAccessMask    = vk::AccessFlagBits2::eNone,
				    .oldLayout        = vk::ImageLayout::eColorAttachmentOptimal,
				    .newLayout        = vk::ImageLayout::ePresentSrcKHR,
				    .image            = swapChainImages[imageIndex],
				    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

				vk::DependencyInfo dependencyInfo{
				    .imageMemoryBarrierCount = 1,
				    .pImageMemoryBarriers    = &barrier};

				commandBuffers[currentFrame].pipelineBarrier2(dependencyInfo);
			}
			else
			{
				vk::ImageMemoryBarrier barrier{
				    .srcAccessMask       = vk::AccessFlagBits::eColorAttachmentWrite,
				    .dstAccessMask       = vk::AccessFlagBits::eNone,
				    .oldLayout           = vk::ImageLayout::eColorAttachmentOptimal,
				    .newLayout           = vk::ImageLayout::ePresentSrcKHR,
				    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				    .image               = swapChainImages[imageIndex],
				    .subresourceRange    = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

				commandBuffers[currentFrame].pipelineBarrier(
				    vk::PipelineStageFlagBits::eColorAttachmentOutput,
				    vk::PipelineStageFlagBits::eBottomOfPipe,
				    vk::DependencyFlagBits::eByRegion,
				    {},
				    {},
				    {barrier});
			}
		}
		else
		{
			commandBuffers[currentFrame].endRenderPass();
		}

		commandBuffers[currentFrame].end();
	}

	void createSyncObjects()
	{
		presentCompleteSemaphore.clear();
		renderFinishedSemaphore.clear();
		inFlightFences.clear();

		if (appInfo.timelineSemaphoresSupported)
		{
			// Create timeline semaphore
			std::cout << "Creating timeline semaphores\n";
			vk::SemaphoreTypeCreateInfo timelineCreateInfo{
			    .semaphoreType = vk::SemaphoreType::eTimeline,
			    .initialValue  = 0};

			vk::SemaphoreCreateInfo semaphoreInfo{
			    .pNext = &timelineCreateInfo};

			timelineSemaphore = vk::raii::Semaphore(device, semaphoreInfo);

			// Still need binary semaphores for swapchain operations
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			{
				presentCompleteSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
				renderFinishedSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
			}
		}
		else
		{
			// Create binary semaphores and fences
			std::cout << "Creating binary semaphores and fences\n";
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
			{
				presentCompleteSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
				renderFinishedSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
			}
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
		}
	}

	void updateUniformBuffer(uint32_t currentImage) const
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto  currentTime = std::chrono::high_resolution_clock::now();
		float time        = std::chrono::duration<float>(currentTime - startTime).count();

		UniformBufferObject ubo{};
		ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view  = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj  = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	void drawFrame()
	{
		while (vk::Result::eTimeout == device.waitForFences(*inFlightFences[currentFrame], vk::True, UINT64_MAX))
			;
		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphore[currentFrame], nullptr);

		if (result == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapChain();
			return;
		}
		if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
		{
			throw std::runtime_error("failed to acquire swap chain image!");
		}
		updateUniformBuffer(currentFrame);

		device.resetFences(*inFlightFences[currentFrame]);
		commandBuffers[currentFrame].reset();
		recordCommandBuffer(imageIndex);

		if (appInfo.timelineSemaphoresSupported)
		{
			// Use timeline semaphores for GPU synchronization
			uint64_t waitValue   = timelineValue;
			uint64_t signalValue = ++timelineValue;

			vk::TimelineSemaphoreSubmitInfo timelineInfo{
			    .waitSemaphoreValueCount   = 0,        // We'll still use binary semaphore for swapchain
			    .signalSemaphoreValueCount = 1,
			    .pSignalSemaphoreValues    = &signalValue};

			std::array<vk::Semaphore, 2>          waitSemaphores = {*presentCompleteSemaphore[currentFrame], *timelineSemaphore};
			std::array<vk::PipelineStageFlags, 2> waitStages     = {vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eVertexInput};
			std::array<uint64_t, 2>               waitValues     = {0, waitValue};        // Binary semaphore value is ignored

			std::array<vk::Semaphore, 2> signalSemaphores = {*renderFinishedSemaphore[currentFrame], *timelineSemaphore};
			std::array<uint64_t, 2>      signalValues     = {0, signalValue};        // Binary semaphore value is ignored

			timelineInfo.waitSemaphoreValueCount   = 1;        // Only for the timeline semaphore
			timelineInfo.pWaitSemaphoreValues      = &waitValues[1];
			timelineInfo.signalSemaphoreValueCount = 1;        // Only for the timeline semaphore
			timelineInfo.pSignalSemaphoreValues    = &signalValues[1];

			vk::SubmitInfo submitInfo{
			    .pNext                = &timelineInfo,
			    .waitSemaphoreCount   = 1,        // Only wait on the binary semaphore
			    .pWaitSemaphores      = &waitSemaphores[0],
			    .pWaitDstStageMask    = &waitStages[0],
			    .commandBufferCount   = 1,
			    .pCommandBuffers      = &*commandBuffers[currentFrame],
			    .signalSemaphoreCount = 2,        // Signal both semaphores
			    .pSignalSemaphores    = signalSemaphores.data()};

			queue.submit(submitInfo, *inFlightFences[currentFrame]);
		}
		else
		{
			// Use traditional binary semaphores
			vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
			const vk::SubmitInfo   submitInfo{
			      .waitSemaphoreCount   = 1,
			      .pWaitSemaphores      = &*presentCompleteSemaphore[currentFrame],
			      .pWaitDstStageMask    = &waitDestinationStageMask,
			      .commandBufferCount   = 1,
			      .pCommandBuffers      = &*commandBuffers[currentFrame],
			      .signalSemaphoreCount = 1,
			      .pSignalSemaphores    = &*renderFinishedSemaphore[currentFrame]};
			queue.submit(submitInfo, *inFlightFences[currentFrame]);
		}

		try
		{
			const vk::PresentInfoKHR presentInfoKHR{
			    .waitSemaphoreCount = 1,
			    .pWaitSemaphores    = &*renderFinishedSemaphore[currentFrame],
			    .swapchainCount     = 1,
			    .pSwapchains        = &*swapChain,
			    .pImageIndices      = &imageIndex};
			result = queue.presentKHR(presentInfoKHR);
			if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized)
			{
				framebufferResized = false;
				recreateSwapChain();
			}
			else if (result != vk::Result::eSuccess)
			{
				throw std::runtime_error("failed to present swap chain image!");
			}
		}
		catch (const vk::SystemError &e)
		{
			if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR))
			{
				recreateSwapChain();
				return;
			}
			else
			{
				throw;
			}
		}
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t *>(code.data())};
		vk::raii::ShaderModule     shaderModule{device, createInfo};

		return shaderModule;
	}

	static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
		{
			minImageCount = surfaceCapabilities.maxImageCount;
		}
		return minImageCount;
	}

	static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
	{
		assert(!availableFormats.empty());
		const auto formatIt = std::ranges::find_if(
		    availableFormats,
		    [](const auto &format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}

	static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
		return std::ranges::any_of(availablePresentModes,
		                           [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
		           vk::PresentModeKHR::eMailbox :
		           vk::PresentModeKHR::eFifo;
	}

	[[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const
	{
		if (capabilities.currentExtent.width != 0xFFFFFFFF)
		{
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {
		    std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		    std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
	}

	[[nodiscard]] std::vector<const char *> getRequiredExtensions() const
	{
		// Get the required extensions from GLFW
		uint32_t    glfwExtensionCount = 0;
		auto        glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		// Check if the debug utils extension is available
		std::vector<vk::ExtensionProperties> props               = context.enumerateInstanceExtensionProperties();
		bool                                 debugUtilsAvailable = std::ranges::any_of(props,
		                                                                               [](vk::ExtensionProperties const &ep) {
                                                           return strcmp(ep.extensionName, vk::EXTDebugUtilsExtensionName) == 0;
                                                       });

		// Always include the debug utils extension if available
		// This allows validation layers to be enabled via vulkanconfig
		if (debugUtilsAvailable)
		{
			extensions.push_back(vk::EXTDebugUtilsExtensionName);
		}
		else
		{
			std::cout << "VK_EXT_debug_utils extension not available. Validation layers may not work." << std::endl;
		}

		return extensions;
	}

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *)
	{
		if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
		{
			std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
		}

		return vk::False;
	}

	static std::vector<char> readFile(const std::string &filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}
		std::vector<char> buffer(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		file.close();

		return buffer;
	}
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
