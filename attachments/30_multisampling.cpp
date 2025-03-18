#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono>

import vulkan_hpp;
#include <vulkan/vk_platform.h>

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
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

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint64_t FenceTimeout = 100000000;
const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription( 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) ),
            vk::VertexInputAttributeDescription( 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) ),
            vk::VertexInputAttributeDescription( 2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord) )
        };
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

template<> struct std::hash<Vertex> {
        size_t operator()(Vertex const& vertex) const noexcept {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context  context;
    std::unique_ptr<vk::raii::Instance> instance;
    std::unique_ptr<vk::raii::DebugUtilsMessengerEXT> debugMessenger;
    std::unique_ptr<vk::raii::SurfaceKHR> surface;

    std::unique_ptr<vk::raii::PhysicalDevice> physicalDevice;
    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
    std::unique_ptr<vk::raii::Device> device;

    std::unique_ptr<vk::raii::Queue> graphicsQueue;
    std::unique_ptr<vk::raii::Queue> presentQueue;

    std::unique_ptr<vk::raii::SwapchainKHR> swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;
    std::vector<vk::raii::Framebuffer> swapChainFramebuffers;

    std::unique_ptr<vk::raii::RenderPass> renderPass;
    std::unique_ptr<vk::raii::DescriptorSetLayout> descriptorSetLayout;
    std::unique_ptr<vk::raii::PipelineLayout> pipelineLayout;
    std::unique_ptr<vk::raii::Pipeline> graphicsPipeline;

    std::unique_ptr<vk::raii::Image> colorImage;
    std::unique_ptr<vk::raii::DeviceMemory> colorImageMemory;
    std::unique_ptr<vk::raii::ImageView> colorImageView;

    std::unique_ptr<vk::raii::Image> depthImage;
    std::unique_ptr<vk::raii::DeviceMemory> depthImageMemory;
    std::unique_ptr<vk::raii::ImageView> depthImageView;

    uint32_t mipLevels = 0;
    std::unique_ptr<vk::raii::Image> textureImage;
    std::unique_ptr<vk::raii::DeviceMemory> textureImageMemory;
    std::unique_ptr<vk::raii::ImageView> textureImageView;
    std::unique_ptr<vk::raii::Sampler> textureSampler;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::unique_ptr<vk::raii::Buffer> vertexBuffer;
    std::unique_ptr<vk::raii::DeviceMemory> vertexBufferMemory;
    std::unique_ptr<vk::raii::Buffer> indexBuffer;
    std::unique_ptr<vk::raii::DeviceMemory> indexBufferMemory;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    std::unique_ptr<vk::raii::DescriptorPool> descriptorPool;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    std::unique_ptr<vk::raii::CommandPool> commandPool;
    std::vector<std::unique_ptr<vk::raii::CommandBuffer>> commandBuffers;
    uint32_t graphicsIndex = 0;

    std::vector<std::unique_ptr<vk::raii::Semaphore>> presentCompleteSemaphore;
    std::vector<std::unique_ptr<vk::raii::Semaphore>> renderFinishedSemaphore;
    std::vector<std::unique_ptr<vk::raii::Fence>> inFlightFences;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createColorResources();
        createDepthResources();
        createFramebuffers();
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
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device->waitIdle();
    }

    void cleanupSwapChain() {
        swapChainFramebuffers.clear();
        swapChainImageViews.clear();
        swapChainImageViews.clear();
        }

    void cleanup() const {
        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device->waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createColorResources();
        createDepthResources();
        createFramebuffers();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        constexpr auto appInfo = vk::ApplicationInfo("Hello Triangle", 1, "No Engine", 1, vk::ApiVersion14);
        auto extensions = getRequiredExtensions();
        std::vector<char const *> enabledLayers;
        if (enableValidationLayers) {
            enabledLayers.assign(validationLayers.begin(), validationLayers.end());
        }
        vk::InstanceCreateInfo createInfo({}, &appInfo, enabledLayers.size(), enabledLayers.data(), extensions.size(), extensions.data());
        instance = std::make_unique<vk::raii::Instance>(context, createInfo);
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
        vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT({}, severityFlags, messageTypeFlags, &debugCallback);
        debugMessenger = std::make_unique<vk::raii::DebugUtilsMessengerEXT>( *instance, debugUtilsMessengerCreateInfoEXT );
    }

    void createSurface() {
        VkSurfaceKHR       _surface;
        if (glfwCreateWindowSurface(**instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = std::make_unique<vk::raii::SurfaceKHR>(*instance, _surface);
    }

    void pickPhysicalDevice() {
        physicalDevice = std::make_unique<vk::raii::PhysicalDevice>(vk::raii::PhysicalDevices( *instance ).front());
    }

    void createLogicalDevice() {
        // find the index of the first queue family that supports graphics
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice->getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty =
          std::find_if( queueFamilyProperties.begin(),
                        queueFamilyProperties.end(),
                        []( vk::QueueFamilyProperties const & qfp ) { return qfp.queueFlags & vk::QueueFlagBits::eGraphics; } );

        graphicsIndex = static_cast<uint32_t>( std::distance( queueFamilyProperties.begin(), graphicsQueueFamilyProperty ) );

        // determine a queueFamilyIndex that supports present
        // first check if the graphicsIndex is good enough
        auto presentIndex = physicalDevice->getSurfaceSupportKHR( graphicsIndex, *surface )
                                           ? graphicsIndex
                                           : static_cast<uint32_t>( queueFamilyProperties.size() );
        if ( presentIndex == queueFamilyProperties.size() )
        {
            // the graphicsIndex doesn't support present -> look for another family index that supports both
            // graphics and present
            for ( size_t i = 0; i < queueFamilyProperties.size(); i++ )
            {
                if ( ( queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics ) &&
                     physicalDevice->getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
                {
                    graphicsIndex = static_cast<uint32_t>( i );
                    presentIndex  = graphicsIndex;
                    break;
                }
            }
            if ( presentIndex == queueFamilyProperties.size() )
            {
                // there's nothing like a single family index that supports both graphics and present -> look for another
                // family index that supports present
                for ( size_t i = 0; i < queueFamilyProperties.size(); i++ )
                {
                    if ( physicalDevice->getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
                    {
                        presentIndex = static_cast<uint32_t>( i );
                break;
            }
        }
        }
    }
        if ( ( graphicsIndex == queueFamilyProperties.size() ) || ( presentIndex == queueFamilyProperties.size() ) )
        {
            throw std::runtime_error( "Could not find a queue for graphics or present -> terminating" );
        }

        // create a Device
        std::vector deviceExtensions = { vk::KHRSwapchainExtensionName, vk::KHRSpirv14ExtensionName };
        float                     queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo( {}, graphicsIndex, 1, &queuePriority );
        vk::PhysicalDeviceFeatures deviceFeatures;
        vk::DeviceCreateInfo      deviceCreateInfo( {}, deviceQueueCreateInfo, {}, deviceExtensions, &deviceFeatures );
        deviceFeatures.samplerAnisotropy = vk::True;
        deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
        deviceCreateInfo.enabledExtensionCount = deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        device = std::make_unique<vk::raii::Device>( *physicalDevice, deviceCreateInfo );
        msaaSamples = getMaxUsableSampleCount();
        graphicsQueue = std::make_unique<vk::raii::Queue>( *device, graphicsIndex, 0 );
        presentQueue = std::make_unique<vk::raii::Queue>( *device, presentIndex, 0 );
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice->getSurfaceCapabilitiesKHR( *surface );
        auto surfaceFormat = chooseSwapSurfaceFormat(physicalDevice->getSurfaceFormatsKHR( *surface ));
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        auto minImageCount = std::max( 3u, surfaceCapabilities.minImageCount );
        minImageCount = ( surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount ) ? surfaceCapabilities.maxImageCount : minImageCount;
        vk::SwapchainCreateInfoKHR swapChainCreateInfo( {}, *surface, minImageCount, surfaceFormat.format, surfaceFormat.colorSpace,
                                                    swapChainExtent, 1, vk::ImageUsageFlagBits::eColorAttachment,
                                                    vk::SharingMode::eExclusive, {}, surfaceCapabilities.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                    chooseSwapPresentMode(physicalDevice->getSurfacePresentModesKHR( *surface )), true, nullptr );

        swapChain = std::make_unique<vk::raii::SwapchainKHR>( *device, swapChainCreateInfo );
        swapChainImageFormat = surfaceFormat.format;
        swapChainImages = swapChain->getImages();
    }

    void createImageViews() {
        vk::ImageViewCreateInfo imageViewCreateInfo( {}, {}, vk::ImageViewType::e2D, swapChainImageFormat, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } );
        for ( auto image : swapChainImages )
        {
            imageViewCreateInfo.image = image;
            swapChainImageViews.emplace_back( *device, imageViewCreateInfo );
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment({}, swapChainImageFormat, msaaSamples, vk::AttachmentLoadOp::eClear,
         vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
          vk::ImageLayout::ePresentSrcKHR);
        vk::AttachmentDescription depthAttachment({}, findDepthFormat(), msaaSamples, vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal);
        vk::AttachmentDescription colorAttachmentResolve({}, swapChainImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
        vk::ImageLayout::ePresentSrcKHR);

        vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);
        vk::AttachmentReference depthAttachmentRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        vk::AttachmentReference colorAttachmentResolveRef(2, vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, 0, {}, 1, &colorAttachmentRef, &colorAttachmentResolveRef, &depthAttachmentRef);
        vk::SubpassDependency dependency(vk::SubpassExternal, {},
                    vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eLateFragmentTests,
                    vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eColorAttachmentOutput,
                    vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                    vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eColorAttachmentWrite
                    );
        std::array attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
        vk::RenderPassCreateInfo renderPassInfo({}, attachments, subpass, dependency);
        renderPass = std::make_unique<vk::raii::RenderPass>( *device, renderPassInfo );
    }

    void createDescriptorSetLayout() {
        std::array bindings = {
            vk::DescriptorSetLayoutBinding( 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr),
            vk::DescriptorSetLayoutBinding( 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr)
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings.size(), bindings.data());
        descriptorSetLayout = std::make_unique<vk::raii::DescriptorSetLayout>( *device, layoutInfo );
    }

    void createGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo({}, vk::ShaderStageFlagBits::eVertex, shaderModule, "vertMain");
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment, shaderModule, "fragMain");
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, { bindingDescription }, attributeDescriptions);
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, vk::False);
        vk::PipelineViewportStateCreateInfo viewportState({}, 1, {}, 1);
        vk::PipelineRasterizationStateCreateInfo rasterizer({}, vk::False, vk::False, vk::PolygonMode::eFill,
                        vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, vk::False);
        rasterizer.lineWidth = 1.0f;
        vk::PipelineMultisampleStateCreateInfo multisampling({}, msaaSamples, vk::False);
        vk::PipelineDepthStencilStateCreateInfo depthStencil( {}, vk::True, vk::True, vk::CompareOp::eLess, vk::False, vk::False);
        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = vk::False;

        vk::PipelineColorBlendStateCreateInfo colorBlending({},vk::False, vk::LogicOp::eCopy, colorBlendAttachment);

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, { **descriptorSetLayout });

        pipelineLayout = std::make_unique<vk::raii::PipelineLayout>( *device, pipelineLayoutInfo );

        vk::GraphicsPipelineCreateInfo pipelineInfo({}, shaderStages, &vertexInputInfo, &inputAssembly, {}, &viewportState, &rasterizer, &multisampling, &depthStencil, &colorBlending,
                    &dynamicState, *pipelineLayout, *renderPass, 0);

        graphicsPipeline = std::make_unique<vk::raii::Pipeline>(*device, nullptr, pipelineInfo);
    }

    void createFramebuffers() {
        for ( auto const & view : swapChainImageViews )
        {
            vk::ImageView attachments[] = { *colorImageView,  *depthImageView, view };
            vk::FramebufferCreateInfo framebufferCreateInfo( {}, *renderPass, attachments, swapChainExtent.width, swapChainExtent.height, 1 );
            swapChainFramebuffers.emplace_back(*device, framebufferCreateInfo );
        }
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsIndex);
        commandPool = std::make_unique<vk::raii::CommandPool>(*device, poolInfo);
    }

    void createColorResources() {
        vk::Format colorFormat = swapChainImageFormat;

        vk::raii::Image colorImage_({});
        vk::raii::DeviceMemory colorImageMemory_({});
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,  vk::MemoryPropertyFlagBits::eDeviceLocal, colorImage_, colorImageMemory_);
        colorImage = std::make_unique<vk::raii::Image>(std::move(colorImage_));
        colorImageMemory = std::make_unique<vk::raii::DeviceMemory>(std::move(colorImageMemory_));
        colorImageView = createImageView(*colorImage, colorFormat, vk::ImageAspectFlagBits::eColor, 1);
    }

    void createDepthResources() {
        vk::Format depthFormat = findDepthFormat();

        vk::raii::Image depthImage_({});
        vk::raii::DeviceMemory depthImageMemory_({});
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage_, depthImageMemory_);
        depthImage = std::make_unique<vk::raii::Image>(std::move(depthImage_));
        depthImageMemory = std::make_unique<vk::raii::DeviceMemory>(std::move(depthImageMemory_));
        depthImageView = createImageView(*depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth, 1);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) const {
        for (const auto format : candidates) {
            vk::FormatProperties props = physicalDevice->getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    [[nodiscard]] vk::Format findDepthFormat() const {
        return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    static bool hasStencilComponent(vk::Format format) {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;
        mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, imageSize);
        stagingBufferMemory.unmapMemory();

        stbi_image_free(pixels);

        vk::raii::Image textureImageTemp({});
        vk::raii::DeviceMemory textureImageMemoryTemp({});
        createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImageTemp, textureImageMemoryTemp);
        textureImage = std::make_unique<vk::raii::Image>(std::move(textureImageTemp));
        textureImageMemory = std::make_unique<vk::raii::DeviceMemory>(std::move(textureImageMemoryTemp));

        transitionImageLayout(*textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
        copyBufferToImage(stagingBuffer, *textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        //transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps

        generateMipmaps(*textureImage, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, mipLevels);
    }

    void generateMipmaps(vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
        // Check if image format supports linear blit-ing
        vk::FormatProperties formatProperties = physicalDevice->getFormatProperties(imageFormat);

        if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier (vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead
                               , vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal
                               , vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, image);
        barrier.image = image;
        barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
        barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

            vk::ImageBlit blit( {}, {vk::Offset3D(0,0, 0),vk::Offset3D(mipWidth, mipHeight, 1)},
                                {}, {vk::Offset3D(0,0, 0),vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1)} );
            blit.srcSubresource = vk::ImageSubresourceLayers( vk::ImageAspectFlagBits::eColor, i - 1, 0, 1);
            blit.dstSubresource = vk::ImageSubresourceLayers( vk::ImageAspectFlagBits::eColor, i, 0, 1);

            commandBuffer->blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, { blit }, vk::Filter::eLinear);

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

        endSingleTimeCommands(*commandBuffer);
    }

    vk::SampleCountFlagBits getMaxUsableSampleCount() {
        vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice->getProperties();

        vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
        if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
        if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
        if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
        if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
        if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

        return vk::SampleCountFlagBits::e1;
    }

    void createTextureImageView() {
        textureImageView = createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, mipLevels);
    }

    void createTextureSampler() {

        vk::PhysicalDeviceProperties properties = physicalDevice->getProperties();
        vk::SamplerCreateInfo samplerInfo( {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat,
                                            vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0, vk::True,
                                            properties.limits.maxSamplerAnisotropy, vk::False, vk::CompareOp::eAlways,
                                            0.0f, vk::LodClampNone, vk::BorderColor::eFloatOpaqueBlack, vk::False
                                        );
        textureSampler = std::make_unique<vk::raii::Sampler>(*device, samplerInfo);
    }

    [[nodiscard]] std::unique_ptr<vk::raii::ImageView> createImageView(const vk::raii::Image& image, vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const {
        vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e2D, format, {}, { aspectFlags, 0, mipLevels, 0, 1 });
        return std::make_unique<vk::raii::ImageView>( *device, viewInfo );
        }

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, vk::SampleCountFlagBits numSamples, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) const {
        vk::ImageCreateInfo imageInfo( {}, vk::ImageType::e2D, format, {width, height, 1}, mipLevels, 1, numSamples, tiling, usage, vk::SharingMode::eExclusive);
        imageInfo.initialLayout = vk::ImageLayout::eUndefined;
        image = vk::raii::Image( *device, imageInfo );

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo( memRequirements.size, findMemoryType(memRequirements.memoryTypeBits, properties) );
        imageMemory = vk::raii::DeviceMemory( *device, allocInfo );
        image.bindMemory(*imageMemory, 0);
    }

    void transitionImageLayout(const vk::raii::Image& image, const vk::ImageLayout oldLayout, const vk::ImageLayout newLayout, uint32_t mipLevels) const {
        const auto commandBuffer = beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier( {}, {}, oldLayout, newLayout, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, image, { vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 1 } );

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask =  vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask =  vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        commandBuffer->pipelineBarrier( sourceStage, destinationStage, {}, {}, nullptr, barrier );
        endSingleTimeCommands(*commandBuffer);
    }

    void copyBufferToImage(const vk::raii::Buffer& buffer, const vk::raii::Image& image, uint32_t width, uint32_t height) const {
        std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = beginSingleTimeCommands();
        vk::BufferImageCopy region( 0, 0, 0, { vk::ImageAspectFlagBits::eColor, 0, 0, 1 }, {0, 0, 0}, {width, height, 1});
        commandBuffer->copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});
        endSingleTimeCommands(*commandBuffer);
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.texCoord = {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = {1.0f, 1.0f, 1.0f};

                if (!uniqueVertices.contains(vertex)) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
    }

    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        vk::raii::Buffer vertexBufferTemp({});
        vk::raii::DeviceMemory vertexBufferMemoryTemp({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBufferTemp, vertexBufferMemoryTemp);
        vertexBuffer = std::make_unique<vk::raii::Buffer>(std::move(vertexBufferTemp));
        vertexBufferMemory = std::make_unique<vk::raii::DeviceMemory>(std::move(vertexBufferMemoryTemp));

        copyBuffer(stagingBuffer, *vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        vk::raii::Buffer indexBufferTemp({});
        vk::raii::DeviceMemory indexBufferMemoryTemp({});
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBufferTemp, indexBufferMemoryTemp);
        indexBuffer = std::make_unique<vk::raii::Buffer>(std::move(indexBufferTemp));
        indexBufferMemory = std::make_unique<vk::raii::DeviceMemory>(std::move(indexBufferMemoryTemp));

        copyBuffer(stagingBuffer, *indexBuffer, bufferSize);
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer, bufferMem);
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back( uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createDescriptorPool() {
        std::array poolSize {
            vk::DescriptorPoolSize( vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize(  vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT)
        };
        vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, MAX_FRAMES_IN_FLIGHT, poolSize);
        descriptorPool = std::make_unique<vk::raii::DescriptorPool>(*device, poolInfo);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool, layouts);

        descriptorSets.clear();
        descriptorSets = device->allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));
            vk::DescriptorImageInfo imageInfo( *textureSampler, *textureImageView, vk::ImageLayout::eShaderReadOnlyOptimal );
            std::array descriptorWrites{
                vk::WriteDescriptorSet( descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo ),
                vk::WriteDescriptorSet( descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo)
            };
            device->updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) const {
        vk::BufferCreateInfo bufferInfo({}, size, usage, vk::SharingMode::eExclusive);
        buffer = vk::raii::Buffer(*device, bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo(memRequirements.size, findMemoryType(memRequirements.memoryTypeBits, properties));
        bufferMemory = vk::raii::DeviceMemory(*device, allocInfo);
        buffer.bindMemory(bufferMemory, 0);
    }

    [[nodiscard]] std::unique_ptr<vk::raii::CommandBuffer> beginSingleTimeCommands() const {
        vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
        std::unique_ptr<vk::raii::CommandBuffer> commandBuffer = std::make_unique<vk::raii::CommandBuffer>(std::move(vk::raii::CommandBuffers( *device, allocInfo ).front()));

        vk::CommandBufferBeginInfo beginInfo( vk::CommandBufferUsageFlagBits::eOneTimeSubmit );
        commandBuffer->begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(const vk::raii::CommandBuffer& commandBuffer) const {
        commandBuffer.end();

        vk::SubmitInfo submitInfo( {}, {}, {*commandBuffer});
        graphicsQueue->submit(submitInfo, nullptr);
        graphicsQueue->waitIdle();
    }

    void copyBuffer(const vk::raii::Buffer & srcBuffer, const vk::raii::Buffer & dstBuffer, vk::DeviceSize size) const {
        std::unique_ptr<vk::raii::CommandBuffer> commandCopyBuffer = beginSingleTimeCommands();
        commandCopyBuffer->copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
        endSingleTimeCommands(*commandCopyBuffer);
    }

    [[nodiscard]] uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice->getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        vk::CommandBufferAllocateInfo allocInfo(*commandPool, vk::CommandBufferLevel::ePrimary, commandBuffers.size());
        auto commandBuffers_ = vk::raii::CommandBuffers( *device, allocInfo );
        for (int i = 0; i < commandBuffers.size(); i++) {
            commandBuffers[i] = std::make_unique<vk::raii::CommandBuffer>(std::move(commandBuffers_[i]));
        }
    }

    void recordCommandBuffer(uint32_t imageIndex) {
        commandBuffers[currentFrame]->begin( {} );
        vk::ClearValue clearColor[2] = { vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f), vk::ClearDepthStencilValue(1.0f, 0) };
        vk::RenderPassBeginInfo renderPassInfo( *renderPass, swapChainFramebuffers[imageIndex], {{0, 0}, swapChainExtent}, clearColor);
        commandBuffers[currentFrame]->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffers[currentFrame]->bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
        commandBuffers[currentFrame]->setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
        commandBuffers[currentFrame]->setScissor( 0, vk::Rect2D( vk::Offset2D( 0, 0 ), swapChainExtent ) );
        commandBuffers[currentFrame]->bindVertexBuffers(0, { **vertexBuffer }, {0});
        commandBuffers[currentFrame]->bindIndexBuffer( **indexBuffer, 0, vk::IndexType::eUint32 );
        commandBuffers[currentFrame]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
        commandBuffers[currentFrame]->drawIndexed(indices.size(), 1, 0, 0, 0);
        commandBuffers[currentFrame]->endRenderPass();
        commandBuffers[currentFrame]->end();
    }

    void createSyncObjects() {
        presentCompleteSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            presentCompleteSemaphore[i] = std::make_unique<vk::raii::Semaphore>(*device, vk::SemaphoreCreateInfo());
            renderFinishedSemaphore[i] = std::make_unique<vk::raii::Semaphore>(*device, vk::SemaphoreCreateInfo());
            inFlightFences[i] = std::make_unique<vk::raii::Fence>(*device, vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
        }
    }

    void updateUniformBuffer(uint32_t currentImage) const {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        while ( vk::Result::eTimeout == device->waitForFences( **inFlightFences[currentFrame], vk::True, FenceTimeout ) )
            ;
        auto [result, imageIndex] = swapChain->acquireNextImage( UINT64_MAX, **presentCompleteSemaphore[currentFrame], nullptr );

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        updateUniformBuffer(currentFrame);

        device->resetFences(  **inFlightFences[currentFrame] );
        commandBuffers[currentFrame]->reset();
        recordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitDestinationStageMask( vk::PipelineStageFlagBits::eColorAttachmentOutput );
        const vk::SubmitInfo submitInfo( **presentCompleteSemaphore[currentFrame], waitDestinationStageMask, **commandBuffers[currentFrame], **renderFinishedSemaphore[currentFrame] );
        graphicsQueue->submit(submitInfo, **inFlightFences[currentFrame]);

        const vk::PresentInfoKHR presentInfoKHR( **renderFinishedSemaphore[currentFrame], **swapChain, imageIndex );
        result = presentQueue->presentKHR( presentInfoKHR );
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo({}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) );
        vk::raii::ShaderModule shaderModule(*device, createInfo);

        return shaderModule;
    }

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]] vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            return {
                std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
            };
    }

    [[nodiscard]] std::vector<const char*> getRequiredExtensions() const {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<vk::ExtensionProperties> props = context.enumerateInstanceExtensionProperties();
        if (const auto propsIterator = std::ranges::find_if(props, []( vk::ExtensionProperties const & ep ) { return strcmp( ep.extensionName, vk::EXTDebugUtilsExtensionName ) == 0; } ); propsIterator == props.end() )
        {
            std::cout << "Something went very wrong, cannot find VK_EXT_debug_utils extension" << std::endl;
            exit( 1 );
        }
        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName );
        }

        return extensions;
    }

    [[nodiscard]] bool checkValidationLayerSupport() const {
        return (std::ranges::any_of(context.enumerateInstanceLayerProperties(),
        []( vk::LayerProperties const & lp ) { return ( strcmp( "VK_LAYER_KHRONOS_validation", lp.layerName ) == 0 ); } ) );
            }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
        if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
            std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
        }

        return vk::False;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }
        std::vector<char> buffer(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();

        return buffer;
    }
};

int main() {
    try {
    HelloTriangleApplication app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
