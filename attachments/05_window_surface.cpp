#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <algorithm>

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
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;

    std::vector<const char*> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName,
        vk::KHRSpirv14ExtensionName,
        vk::KHRSynchronization2ExtensionName,
        vk::KHRCreateRenderpass2ExtensionName
    };

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

        constexpr vk::ApplicationInfo appInfo{ .pApplicationName   = "Hello Triangle",
                    .applicationVersion = VK_MAKE_VERSION( 1, 0, 0 ),
                    .pEngineName        = "No Engine",
                    .engineVersion      = VK_MAKE_VERSION( 1, 0, 0 ),
                    .apiVersion         = vk::ApiVersion14 };
        auto extensions = getRequiredExtensions();
        std::vector<char const *> enabledLayers;
        if (enableValidationLayers) {
            enabledLayers.assign(validationLayers.begin(), validationLayers.end());
        }
        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames     = enabledLayers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data() };
        instance = vk::raii::Instance(context, createInfo);
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError );
        vk::DebugUtilsMessageTypeFlagsEXT    messageTypeFlags( vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
            .messageSeverity = severityFlags,
            .messageType = messageTypeFlags,
            .pfnUserCallback = &debugCallback
            };
        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    void createSurface() {
        VkSurfaceKHR       _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void pickPhysicalDevice() {
        std::vector<vk::raii::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        const auto                            devIter = std::ranges::find_if(
          devices,
          [&]( auto const & device )
          {
            // Check if the device supports the Vulkan 1.3 API version
            bool supportsVulkan1_3 = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

            // Check if any of the queue families support graphics operations
            auto queueFamilies = device.getQueueFamilyProperties();
            bool supportsGraphics =
              std::ranges::any_of( queueFamilies, []( auto const & qfp ) { return !!( qfp.queueFlags & vk::QueueFlagBits::eGraphics ); } );

            // Check if all required device extensions are available
            auto availableDeviceExtensions = device.enumerateDeviceExtensionProperties();
            bool supportsAllRequiredExtensions =
              std::ranges::all_of( requiredDeviceExtension,
                                   [&availableDeviceExtensions]( auto const & requiredDeviceExtension )
                                   {
                                     return std::ranges::any_of( availableDeviceExtensions,
                                                                 [requiredDeviceExtension]( auto const & availableDeviceExtension )
                                                                 { return strcmp( availableDeviceExtension.extensionName, requiredDeviceExtension ) == 0; } );
                                   } );

            auto features = device.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
            bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

            return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
          } );
        if ( devIter != devices.end() )
        {
            physicalDevice = *devIter;
        }
        else
        {
            throw std::runtime_error( "failed to find a suitable GPU!" );
        }
    }

    void createLogicalDevice() {
        // find the index of the first queue family that supports graphics
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty = std::ranges::find_if( queueFamilyProperties, []( auto const & qfp )
                        { return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); } );

        auto graphicsIndex = static_cast<uint32_t>( std::distance( queueFamilyProperties.begin(), graphicsQueueFamilyProperty ) );

        // determine a queueFamilyIndex that supports present
        // first check if the graphicsIndex is good enough
        auto presentIndex = physicalDevice.getSurfaceSupportKHR( graphicsIndex, *surface )
                                           ? graphicsIndex
                                           : ~0;
        if ( presentIndex == queueFamilyProperties.size() )
        {
            // the graphicsIndex doesn't support present -> look for another family index that supports both
            // graphics and present
            for ( size_t i = 0; i < queueFamilyProperties.size(); i++ )
            {
                if ( ( queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics ) &&
                     physicalDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
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
                    if ( physicalDevice.getSurfaceSupportKHR( static_cast<uint32_t>( i ), *surface ) )
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

        // query for Vulkan 1.3 features
        auto features = physicalDevice.getFeatures2();
        vk::PhysicalDeviceVulkan13Features vulkan13Features;
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures;
        vulkan13Features.dynamicRendering = vk::True;
        extendedDynamicStateFeatures.extendedDynamicState = vk::True;
        vulkan13Features.pNext = &extendedDynamicStateFeatures;
        features.pNext = &vulkan13Features;
        // create a Device
        float                     queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo { .queueFamilyIndex = graphicsIndex, .queueCount = 1, .pQueuePriorities = &queuePriority };
        vk::DeviceCreateInfo      deviceCreateInfo{ .pNext =  &features, .queueCreateInfoCount = 1, .pQueueCreateInfos = &deviceQueueCreateInfo };
        deviceCreateInfo.enabledExtensionCount = requiredDeviceExtension.size();
        deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();

        device = vk::raii::Device( physicalDevice, deviceCreateInfo );
        graphicsQueue = vk::raii::Queue( device, graphicsIndex, 0 );
        presentQueue = vk::raii::Queue( device, presentIndex, 0 );
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
