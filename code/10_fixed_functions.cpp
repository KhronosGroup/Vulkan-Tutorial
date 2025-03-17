#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <limits>

import vulkan_hpp;
#include <vulkan/vk_platform.h>

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

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
    std::unique_ptr<vk::raii::Device> device;

    std::unique_ptr<vk::raii::Queue> graphicsQueue;
    std::unique_ptr<vk::raii::Queue> presentQueue;

    std::unique_ptr<vk::raii::SwapchainKHR> swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    std::unique_ptr<vk::raii::PipelineLayout> pipelineLayout;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createGraphicsPipeline();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
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

        // get the first index into queueFamiliyProperties which supports graphics
        auto graphicsQueueFamilyProperty =
          std::find_if( queueFamilyProperties.begin(),
                        queueFamilyProperties.end(),
                        []( vk::QueueFamilyProperties const & qfp ) { return qfp.queueFlags & vk::QueueFlagBits::eGraphics; } );

        auto graphicsIndex = static_cast<uint32_t>( std::distance( queueFamilyProperties.begin(), graphicsQueueFamilyProperty ) );

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
        vk::DeviceCreateInfo      deviceCreateInfo( {}, deviceQueueCreateInfo );
        deviceCreateInfo.enabledExtensionCount = deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        device = std::make_unique<vk::raii::Device>( *physicalDevice, deviceCreateInfo );
        graphicsQueue = std::make_unique<vk::raii::Queue>( *device, graphicsIndex, 0 );
        presentQueue = std::make_unique<vk::raii::Queue>( *device, presentIndex, 0 );
    }

    void createSwapChain() {
        auto surfaceCapabilities = physicalDevice->getSurfaceCapabilitiesKHR( *surface );
        swapChainImageFormat = chooseSwapSurfaceFormat(physicalDevice->getSurfaceFormatsKHR( *surface ));
        swapChainExtent = chooseSwapExtent(surfaceCapabilities);
        auto minImageCount = std::max( 3u, surfaceCapabilities.minImageCount );
        minImageCount = ( surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount ) ? surfaceCapabilities.maxImageCount : minImageCount;
        vk::SwapchainCreateInfoKHR swapChainCreateInfo( vk::SwapchainCreateFlagsKHR(), *surface, minImageCount, swapChainImageFormat, vk::ColorSpaceKHR::eSrgbNonlinear,
                                                    swapChainExtent, 1, vk::ImageUsageFlagBits::eColorAttachment,
                                                    vk::SharingMode::eExclusive, {}, surfaceCapabilities.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                    chooseSwapPresentMode(physicalDevice->getSurfacePresentModesKHR( *surface )), true, nullptr );

        swapChain = std::make_unique<vk::raii::SwapchainKHR>( *device, swapChainCreateInfo );
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

    void createGraphicsPipeline() {
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo({}, vk::ShaderStageFlagBits::eVertex, shaderModule, "vertMain");
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment, shaderModule, "fragMain");
        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList);
        vk::PipelineViewportStateCreateInfo viewportState({}, 1, {}, 1);

        vk::PipelineRasterizationStateCreateInfo rasterizer({}, vk::False, vk::False, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise, vk::False, 0.0f, 0.0f, 1.0f, 1.0f);

        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, vk::False);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = vk::False;

        vk::PipelineColorBlendStateCreateInfo colorBlending({},vk::False, vk::LogicOp::eCopy, 1, &colorBlendAttachment);

        std::vector dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates.size(), dynamicStates.data());

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 0, nullptr, 0, nullptr);

        pipelineLayout = std::make_unique<vk::raii::PipelineLayout>( *device, pipelineLayoutInfo );
    }

     [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo({}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) );
        vk::raii::ShaderModule shaderModule(*device, createInfo);

        return shaderModule;
    }

    static vk::Format chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        return ( availableFormats[0].format == vk::Format::eUndefined ) ? vk::Format::eB8G8R8A8Unorm : availableFormats[0].format;
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
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

    std::vector<const char*> getRequiredExtensions() {
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

    bool checkValidationLayerSupport() {
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
