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
    std::unique_ptr<vk::raii::Instance> instance;
    std::unique_ptr<vk::raii::DebugUtilsMessengerEXT> debugMessenger;

    std::unique_ptr<vk::raii::PhysicalDevice> physicalDevice;
    std::unique_ptr<vk::raii::Device> device;

    std::unique_ptr<vk::raii::Queue> graphicsQueue;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
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

        // create a Device
        float                     queuePriority = 0.0f;
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo( {}, graphicsIndex, 1, &queuePriority );
        vk::DeviceCreateInfo      deviceCreateInfo( {}, deviceQueueCreateInfo );

        device = std::make_unique<vk::raii::Device>( *physicalDevice, deviceCreateInfo );
        graphicsQueue = std::make_unique<vk::raii::Queue>( *device, graphicsIndex, 0 );
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
