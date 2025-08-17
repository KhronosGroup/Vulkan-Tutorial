#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef __INTELLISENSE__
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <vulkan/vk_platform.h>

#define GLFW_INCLUDE_VULKAN        // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

constexpr uint32_t WIDTH                = 800;
constexpr uint32_t HEIGHT               = 600;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

const std::vector validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
	}

	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		return {
		    vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
		    vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color))};
	}
};

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0};

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
	GLFWwindow                      *window = nullptr;
	vk::raii::Context                context;
	vk::raii::Instance               instance       = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::SurfaceKHR             surface        = nullptr;
	vk::raii::PhysicalDevice         physicalDevice = nullptr;
	vk::raii::Device                 device         = nullptr;
	uint32_t                         queueIndex     = ~0;
	vk::raii::Queue                  queue          = nullptr;

	vk::raii::SwapchainKHR           swapChain = nullptr;
	std::vector<vk::Image>           swapChainImages;
	vk::Format                       swapChainImageFormat = vk::Format::eUndefined;
	vk::Extent2D                     swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;

	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      pipelineLayout      = nullptr;
	vk::raii::Pipeline            graphicsPipeline    = nullptr;

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

	std::vector<vk::raii::Semaphore> presentCompleteSemaphore;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphore;
	std::vector<vk::raii::Fence>     inFlightFences;
	uint32_t                         semaphoreIndex = 0;
	uint32_t                         currentFrame   = 0;

	bool framebufferResized = false;

	std::vector<const char *> requiredDeviceExtension = {
	    vk::KHRSwapchainExtensionName,
	    vk::KHRSpirv14ExtensionName,
	    vk::KHRSynchronization2ExtensionName,
	    vk::KHRCreateRenderpass2ExtensionName};

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
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
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
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
		swapChainImageViews.clear();
		swapChain = nullptr;
	}

	void cleanup()
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
	}

	void createInstance()
	{
		constexpr vk::ApplicationInfo appInfo{.pApplicationName   = "Hello Triangle",
		                                      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		                                      .pEngineName        = "No Engine",
		                                      .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
		                                      .apiVersion         = vk::ApiVersion14};

		// Get the required layers
		std::vector<char const *> requiredLayers;
		if (enableValidationLayers)
		{
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		// Check if the required layers are supported by the Vulkan implementation.
		auto layerProperties = context.enumerateInstanceLayerProperties();
		for (auto const &requiredLayer : requiredLayers)
		{
			if (std::ranges::none_of(layerProperties,
			                         [requiredLayer](auto const &layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; }))
			{
				throw std::runtime_error("Required layer not supported: " + std::string(requiredLayer));
			}
		}

		// Get the required extensions.
		auto requiredExtensions = getRequiredExtensions();

		// Check if the required extensions are supported by the Vulkan implementation.
		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		for (auto const &requiredExtension : requiredExtensions)
		{
			if (std::ranges::none_of(extensionProperties,
			                         [requiredExtension](auto const &extensionProperty) { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; }))
			{
				throw std::runtime_error("Required extension not supported: " + std::string(requiredExtension));
			}
		}

		vk::InstanceCreateInfo createInfo{
		    .pApplicationInfo        = &appInfo,
		    .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
		    .ppEnabledLayerNames     = requiredLayers.data(),
		    .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
		    .ppEnabledExtensionNames = requiredExtensions.data()};
		instance = vk::raii::Instance(context, createInfo);
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;

		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
		vk::DebugUtilsMessageTypeFlagsEXT     messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
		vk::DebugUtilsMessengerCreateInfoEXT  debugUtilsMessengerCreateInfoEXT{
		     .messageSeverity = severityFlags,
		     .messageType     = messageTypeFlags,
		     .pfnUserCallback = &debugCallback};
		debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
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
                // Check if the device supports the Vulkan 1.3 API version
                bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

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

                auto features                 = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
                bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

                return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
            });
		if (devIter != devices.end())
		{
			physicalDevice = *devIter;
		}
		else
		{
			throw std::runtime_error("failed to find a suitable GPU!");
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

		// query for Vulkan 1.3 features
		vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
		    {},                                                          // vk::PhysicalDeviceFeatures2
		    {.synchronization2 = true, .dynamicRendering = true},        // vk::PhysicalDeviceVulkan13Features
		    {.extendedDynamicState = true}                               // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
		};

		// create a Device
		float                     queuePriority = 0.0f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
		vk::DeviceCreateInfo      deviceCreateInfo{.pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		                                           .queueCreateInfoCount    = 1,
		                                           .pQueueCreateInfos       = &deviceQueueCreateInfo,
		                                           .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtension.size()),
		                                           .ppEnabledExtensionNames = requiredDeviceExtension.data()};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		queue  = vk::raii::Queue(device, queueIndex, 0);
	}

	void createSwapChain()
	{
		auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
		swapChainImageFormat     = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
		swapChainExtent          = chooseSwapExtent(surfaceCapabilities);
		auto minImageCount       = std::max(3u, surfaceCapabilities.minImageCount);
		minImageCount            = (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;
		vk::SwapchainCreateInfoKHR swapChainCreateInfo{
		    .surface = surface, .minImageCount = minImageCount, .imageFormat = swapChainImageFormat, .imageColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear, .imageExtent = swapChainExtent, .imageArrayLayers = 1, .imageUsage = vk::ImageUsageFlagBits::eColorAttachment, .imageSharingMode = vk::SharingMode::eExclusive, .preTransform = surfaceCapabilities.currentTransform, .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque, .presentMode = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface)), .clipped = true};

		swapChain       = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	void createImageViews()
	{
		vk::ImageViewCreateInfo imageViewCreateInfo{.viewType = vk::ImageViewType::e2D, .format = swapChainImageFormat, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
		for (auto image : swapChainImages)
		{
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}
	}

	void createDescriptorSetLayout()
	{
		vk::DescriptorSetLayoutBinding    uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = 1, .pBindings = &uboLayoutBinding};
		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	}

	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		auto                                     bindingDescription    = Vertex::getBindingDescription();
		auto                                     attributeDescriptions = Vertex::getAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo   vertexInputInfo{.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDescription, .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()), .pVertexAttributeDescriptions = attributeDescriptions.data()};
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};
		vk::PipelineViewportStateCreateInfo      viewportState{.viewportCount = 1, .scissorCount = 1};

		vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable = vk::False, .rasterizerDiscardEnable = vk::False, .polygonMode = vk::PolygonMode::eFill, .cullMode = vk::CullModeFlagBits::eBack, .frontFace = vk::FrontFace::eCounterClockwise, .depthBiasEnable = vk::False, .depthBiasSlopeFactor = 1.0f, .lineWidth = 1.0f};

		vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{.blendEnable    = vk::False,
		                                                           .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

		vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

		std::vector dynamicStates = {
		    vk::DynamicState::eViewport,
		    vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0};

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainImageFormat};
		vk::GraphicsPipelineCreateInfo  pipelineInfo{.pNext               = &pipelineRenderingCreateInfo,
		                                             .stageCount          = 2,
		                                             .pStages             = shaderStages,
		                                             .pVertexInputState   = &vertexInputInfo,
		                                             .pInputAssemblyState = &inputAssembly,
		                                             .pViewportState      = &viewportState,
		                                             .pRasterizationState = &rasterizer,
		                                             .pMultisampleState   = &multisampling,
		                                             .pColorBlendState    = &colorBlending,
		                                             .pDynamicState       = &dynamicState,
		                                             .layout              = pipelineLayout,
		                                             .renderPass          = nullptr};

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
	}

	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		                                   .queueFamilyIndex = queueIndex};
		commandPool = vk::raii::CommandPool(device, poolInfo);
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
		memcpy(data, indices.data(), (size_t) bufferSize);
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
		vk::DescriptorPoolSize       poolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT);
		vk::DescriptorPoolCreateInfo poolInfo{.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, .maxSets = MAX_FRAMES_IN_FLIGHT, .poolSizeCount = 1, .pPoolSizes = &poolSize};
		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
	}

	void createDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool = descriptorPool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()), .pSetLayouts = layouts.data()};

		descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DescriptorBufferInfo bufferInfo{.buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject)};
			vk::WriteDescriptorSet   descriptorWrite{.dstSet = descriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo};
			device.updateDescriptorSets(descriptorWrite, {});
		}
	}

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer &buffer, vk::raii::DeviceMemory &bufferMemory)
	{
		vk::BufferCreateInfo bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
		buffer                                 = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
		bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
		buffer.bindMemory(bufferMemory, 0);
	}

	void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size)
	{
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
		vk::raii::CommandBuffer       commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
		commandCopyBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
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
		// Before starting rendering, transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
		transition_image_layout(
		    imageIndex,
		    vk::ImageLayout::eUndefined,
		    vk::ImageLayout::eColorAttachmentOptimal,
		    {},                                                       // srcAccessMask (no need to wait for previous operations)
		    vk::AccessFlagBits2::eColorAttachmentWrite,               // dstAccessMask
		    vk::PipelineStageFlagBits2::eTopOfPipe,                   // srcStage
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput        // dstStage
		);
		vk::ClearValue              clearColor     = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo = {
		    .imageView   = swapChainImageViews[imageIndex],
		    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		    .loadOp      = vk::AttachmentLoadOp::eClear,
		    .storeOp     = vk::AttachmentStoreOp::eStore,
		    .clearValue  = clearColor};
		vk::RenderingInfo renderingInfo = {
		    .renderArea           = {.offset = {0, 0}, .extent = swapChainExtent},
		    .layerCount           = 1,
		    .colorAttachmentCount = 1,
		    .pColorAttachments    = &attachmentInfo};
		commandBuffers[currentFrame].beginRendering(renderingInfo);
		commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
		commandBuffers[currentFrame].setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
		commandBuffers[currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
		commandBuffers[currentFrame].bindVertexBuffers(0, *vertexBuffer, {0});
		commandBuffers[currentFrame].bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);
		commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
		commandBuffers[currentFrame].drawIndexed(indices.size(), 1, 0, 0, 0);
		commandBuffers[currentFrame].endRendering();
		// After rendering, transition the swapchain image to PRESENT_SRC
		transition_image_layout(
		    imageIndex,
		    vk::ImageLayout::eColorAttachmentOptimal,
		    vk::ImageLayout::ePresentSrcKHR,
		    vk::AccessFlagBits2::eColorAttachmentWrite,                // srcAccessMask
		    {},                                                        // dstAccessMask
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		    vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
		);
		commandBuffers[currentFrame].end();
	}

	void transition_image_layout(
	    uint32_t                imageIndex,
	    vk::ImageLayout         old_layout,
	    vk::ImageLayout         new_layout,
	    vk::AccessFlags2        src_access_mask,
	    vk::AccessFlags2        dst_access_mask,
	    vk::PipelineStageFlags2 src_stage_mask,
	    vk::PipelineStageFlags2 dst_stage_mask)
	{
		vk::ImageMemoryBarrier2 barrier = {
		    .srcStageMask        = src_stage_mask,
		    .srcAccessMask       = src_access_mask,
		    .dstStageMask        = dst_stage_mask,
		    .dstAccessMask       = dst_access_mask,
		    .oldLayout           = old_layout,
		    .newLayout           = new_layout,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = swapChainImages[imageIndex],
		    .subresourceRange    = {
		           .aspectMask     = vk::ImageAspectFlagBits::eColor,
		           .baseMipLevel   = 0,
		           .levelCount     = 1,
		           .baseArrayLayer = 0,
		           .layerCount     = 1}};
		vk::DependencyInfo dependency_info = {
		    .dependencyFlags         = {},
		    .imageMemoryBarrierCount = 1,
		    .pImageMemoryBarriers    = &barrier};
		commandBuffers[currentFrame].pipelineBarrier2(dependency_info);
	}

	void createSyncObjects()
	{
		presentCompleteSemaphore.clear();
		renderFinishedSemaphore.clear();
		inFlightFences.clear();

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			presentCompleteSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
			renderFinishedSemaphore.emplace_back(device, vk::SemaphoreCreateInfo());
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
		}
	}

	void updateUniformBuffer(uint32_t currentImage)
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
		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphore[semaphoreIndex], nullptr);

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

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		const vk::SubmitInfo   submitInfo{.waitSemaphoreCount = 1, .pWaitSemaphores = &*presentCompleteSemaphore[semaphoreIndex], .pWaitDstStageMask = &waitDestinationStageMask, .commandBufferCount = 1, .pCommandBuffers = &*commandBuffers[currentFrame], .signalSemaphoreCount = 1, .pSignalSemaphores = &*renderFinishedSemaphore[imageIndex]};
		queue.submit(submitInfo, *inFlightFences[currentFrame]);

		const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1, .pWaitSemaphores = &*renderFinishedSemaphore[imageIndex], .swapchainCount = 1, .pSwapchains = &*swapChain, .pImageIndices = &imageIndex};
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
		semaphoreIndex = (semaphoreIndex + 1) % presentCompleteSemaphore.size();
		currentFrame   = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t *>(code.data())};
		vk::raii::ShaderModule     shaderModule{device, createInfo};

		return shaderModule;
	}

	static vk::Format chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
	{
		const auto formatIt = std::ranges::find_if(availableFormats,
		                                           [](const auto &format) {
			                                           return format.format == vk::Format::eB8G8R8A8Srgb &&
			                                                  format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
		                                           });
		return formatIt != availableFormats.end() ? formatIt->format : availableFormats[0].format;
	}

	static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
	{
		return std::ranges::any_of(availablePresentModes,
		                           [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
		           vk::PresentModeKHR::eMailbox :
		           vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {
		    std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		    std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
	}

	std::vector<const char *> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		auto     glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers)
		{
			extensions.push_back(vk::EXTDebugUtilsExtensionName);
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
