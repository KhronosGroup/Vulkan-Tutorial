/* Copyright (c) 2026, Khronos Group and contributors
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// The main objective of this course is to explain to developers how to use Vulkan.
// The current code is heavily documented, to explain most relevant Vulkan concepts.
//
// See the course page for more information.
// https://github.com/KhronosGroup/Vulkan-Tutorial/blob/main/en/courses/siggraph2026_vk_tutorial
//

#include "main.h"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

// Vulkan-Hpp loads extension functions through a dispatcher.
// This can create a dispatch table to cache function pointers,
// and can skip the loader when calling Vulkan functions.
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace siggraph {

// Main entry point.
void Application::run()
{
    initGLFWWindow();

    // Create the core Vulkan objects and the swapchain images we will render into.
    initVulkanVKB();

    // Initialize persistent Vulkan resources.
    {
        initCommandPool();
        initFramesInFlightResources();
        initSwapchainImageSyncObjects();

        beginHelperCommands();
        initDepthImages();
        initSceneResources();
        endHelperCommandsAndFlushUploads();

        initDescriptorHeaps();
        calculateVertexInputs();
        initShaderObjects();
    }

    // After initialization, we enter the main loop.
    // The application will run generating frames until the user closes the window or we reach a frame limit.
    mainLoop();
}

void Application::setFrameLimit(std::uint32_t frameLimit) { m_remainingFrameLimit = frameLimit; }

//****************
// Vulkan initialization
//****************

void Application::initGLFWWindow()
{
    util::log_msg("[Init] Init GLFW window");
    // GLFW handles cross-platform window creation for the tutorial.
    // We could use SDL or platform-specific code instead.
    util::require(glfwInit() == GLFW_TRUE, "Failed to initialize GLFW");

    // Vulkan creates the rendering surface; GLFW should not create an OpenGL context.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // Disable window resizing to avoid recreation of swapchain and related resources.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    m_window = glfwCreateWindow(windowWidth, windowHeight, "Vulkan SIGGRAPH 2026", nullptr, nullptr);
    util::require(m_window != nullptr, "Failed to create GLFW window");
}

// Initialize Vulkan using vk-bootstrap.
// This creates the instance, surface, physical device, logical device, queues, and swapchain.
void Application::initVulkanVKB()
{
    util::log_msg("[Init] Init Vulkan using vk-bootstrap");

    // vk-bootstrap reports failures as std::error_code values.
    // This helper throws an exception with a descriptive message if the result is an error.
    // If the object is created correctly it returns it
    const auto vkbGetIfValid = [](auto&& result, const std::string_view operation) {
        // Most Vulkan functions return a VkResult value that acts as an error code, indicating success or failure.
        // vk-bootstrap wraps the error code and the actual object we created.
        // vkbGetIfValid unwraps the result and checks if the object was created successfully.

        if (!result) {
            std::string errorMessage = std::format("{} failed: {}", operation, result.error().message());
            for (const std::string& reason : result.detailed_failure_reasons()) {
                errorMessage += "\n - " + reason;
            }
            throw std::runtime_error(errorMessage);
        }

        return std::move(result).value();
    };

    // Now we will initialize Vulkan using vk-bootstrap.
    // In order to initialize Vulkan we need to create the Vulkan instance, select a physical device,
    // create a logical device, create a swapchain and select queues to submit work.
    // This is relatively easy to do but requires a lot of repetitive code.
    // vk-bootstrap is a library that helps with Vulkan initialization by providing higher-level abstractions.
    // It allows developers to skip writing the same initialization code and focus on creating the application.
    //
    // We recommend vk-bootstrap for small projects and beginners to simplify Vulkan initialization.
    // vk-bootstrap allows people learning Vulkan to get a better idea of the relevant parts when developing Vulkan.
    // Vulkan initialization is a verbose and repetitive process that can be easily learned later.

    // Create the Vulkan instance using vk-bootstrap.
    {

        // This code creates a Vulkan instance using vk-bootstrap.
        // The instance is the root Vulkan object.
        // It represents the connection between the application and the Vulkan API and is used to create all other
        // Vulkan objects.

        // Vulkan layers are optional components that can be enabled to provide additional functionality.
        // They can hook Vulkan API calls and provide additional features like validation, debugging, and
        // profiling. Vulkan layers are employed by tools like RenderDoc or gfxreconstruct to offer debugging and
        // profiling capabilities for Vulkan applications.
        //
        // Vulkan validation layers are particularly important for beginners as they provide runtime validation of
        // Vulkan API usage. They help developers catch bugs and incorrect usage of the Vulkan API.
        //
        // Vulkan layers can be enabled by the application, but we recommend using tools like vkconfig.
        // [Vulkan Configurator](https://vulkan.lunarg.com/doc/view/latest/windows/vkconfig.html)
        // This tool makes it easy to enable and configure layers.
        // This is particularly useful for the validation layers, making it easier to enable additional validation
        // checks like GPU-assisted validation or synchronization validation.

        // This tutorial uses vk-bootstrap to create the Vulkan instance and enable validation layers if available.
        // Creating the instance in Vulkan is relatively easy.
        // vk-bootstrap abstracts the process of enabling the validation layers and creating the debug messenger.
        // In a normal app, we would first need to query all the available layers and check if the validation layers are
        // available. Then we would build a vector of layers to enable and pass it to the instance creation info.
        // A proper application should also use EXT_layer_settings to configure the validation layers and enable/disable
        // specific checks. Vk-bootstrap significantly simplifies this process,
        // checking for available layers and enabling the validation layers if requested.
        // For layer configuration, we recommend using vkconfig.

        // Query global Vulkan support before requesting optional validation layers.
        const vkb::SystemInfo systemInfo =
            vkbGetIfValid(vkb::SystemInfo::get_system_info(), "Querying Vulkan system info");

        // vk-bootstrap handles instance extension/layer selection and debug messenger setup.
        auto instanceBuilder =
            vkb::InstanceBuilder()
                .set_app_name("Vulkan SIGGRAPH tutorial")
                .set_engine_name("No Engine")  // Vulkan allows us to set an engine name and application name.
                .require_api_version(1, 4, 0); // Specify the minimum Vulkan API version.

        // Request validation layers.
        if (debugData.enableGpuDebug) {
            // vk-bootstrap abstracts enumerating and checking Vulkan validation layers.
            //
            // Note: When enabling use_default_debug_messenger(), vk-bootstrap requests the debug utils extension for
            // debug names.

            util::require(systemInfo.validation_layers_available,
                          "GPU debug is enabled, but Vulkan validation layers are not available");

            // We are also using a default debug messenger to handle validation messages.
            // Vulkan allows us to write a specific callback to handle validation messages,
            // but vk-bootstrap provides a default one that prints messages to the console.
            instanceBuilder.request_validation_layers(true).use_default_debug_messenger();
        }

        // Ensure the instance was created successfully by vk-bootstrap.
        m_vkbData.m_instance = vkbGetIfValid(instanceBuilder.build(), "Creating Vulkan instance");

        // Load function pointers for the Vulkan instance.
        {
            // It is complicated to explain how Vulkan is loaded, but as an oversimplification:
            // When calling a Vulkan function, the application does not call the function directly.
            // First it calls the Vulkan loader, and the loader then obtains this function from the device or instance.
            // Vulkan allows us to use vkGetInstanceProcAddr to cache the final function pointers, to call them directly
            // and skip the lookup work done by the loader.
            //
            // The process of creating a table of function pointers is a bit tedious, so it is handled automatically by
            // Vulkan-Hpp. Applications using the C API are encouraged to use a library like Volk for similar
            // functionality. Most third-party bindings for other languages also handle this automatically.
            //
            // The VULKAN_HPP_DEFAULT_DISPATCHER contains the function pointers for the Vulkan instance and device
            // functions. It is used by Vulkan-Hpp to call C API functions directly without going through the loader.
            VULKAN_HPP_DEFAULT_DISPATCHER.init(m_vkbData.m_instance.instance, vkGetInstanceProcAddr);
        }
    }

    // Create the window surface using GLFW.
    {
        // We need to create the surface before physical-device selection so we can check presentation support
        // when selecting the physical device.
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkResult result = glfwCreateWindowSurface(m_vkbData.m_instance.instance, m_window, nullptr, &surface);
        util::checkVk(result, "Creating GLFW Vulkan surface");
        m_surface = vk::SurfaceKHR{surface};
    }

    // Select a physical device and create the logical device with required features and extensions.
    {

        // In Vulkan, the machine issuing the Vulkan commands is the host, and the machine executing the commands is the
        // device. Vulkan can run on multiple hardware configurations, but in general the device is the GPU, and the
        // host is the CPU. Vulkan is designed to support systems with multiple GPUs, including laptops with integrated
        // and dedicated graphics, or multi-GPU clusters.
        //
        // Vulkan differentiates between physical devices and logical devices.
        // A physical device is a GPU that is available in the system, exposing the hardware capabilities.
        // A logical device is a handle to a physical device. It specifies the capabilities enabled for our application
        // and is used to issue commands.
        //
        // After creating the Vulkan instance, we need to select the device that we will use.
        // This is abstracted by vk-bootstrap, but we can also do it manually by enumerating the available physical
        // devices. Then we check which physical devices support the features and extensions we need, and select the
        // best one for our application. Finally we create a logical device from the selected physical device, enabling
        // the features and extensions we need. Checking capabilities for a physical device requires a lot of
        // boilerplate code. First we need to obtain all extensions supported by the device, then we need to check if
        // the extensions we want to use are supported. We need to build a pNext chain of feature structures for all the
        // features and extensions we want to check. After querying the physical device properties and features, we need
        // to build a new pNext chain only with the features and extensions we want to enable in the logical device.
        // This is mostly boilerplate code that is abstracted by vk-bootstrap.

        // This tutorial uses a lot of modern extensions.
        // We use this array to specify all the extensions we need in one place and pass it to vk-bootstrap when
        // selecting the physical device.
        const std::array requiredExtensions{
            VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
            VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
            VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME,
        };

        // vk-bootstrap uses our instance to enumerate all physical devices and filter them based on the criteria we
        // specify.
        vkb::PhysicalDeviceSelector physicalDeviceSelector{m_vkbData.m_instance};

        // Set all extensions and features required by the tutorial.
        {
            physicalDeviceSelector
                .add_required_extensions(requiredExtensions.size(), requiredExtensions.data())
                // Check that the device supports presentation to our surface, so we can present rendered images.
                .set_surface(static_cast<VkSurfaceKHR>(m_surface))
                // Vulkan 1.4 makes maintenance5 core; require the feature for bindIndexBuffer2.
                // Descriptor heap also has a requirement on maintenance5.
                .set_minimum_version(1, 4)
                .add_required_extension_features(VkPhysicalDeviceVulkan14Features{
                    // In the Vulkan C API, we need to specify the structure type in sType.
                    // Vulkan-Hpp handles this automatically, setting sType in the constructor of the structure,
                    // but we need to manually set sType when iterating with a C library like vk-bootstrap.
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
                    .maintenance5 = VK_TRUE,
                })
                // Vulkan 1.3 core features used by the tutorial: synchronization2 and dynamic rendering.
                .add_required_extension_features(VkPhysicalDeviceVulkan13Features{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                    .synchronization2 = VK_TRUE,
                    .dynamicRendering = VK_TRUE,
                })
                // Buffer device address is core in Vulkan 1.2.
                .add_required_extension_features(VkPhysicalDeviceVulkan12Features{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                    .bufferDeviceAddress = VK_TRUE,
                })
                // Draw passes objectIndex as firstInstance. Slang exposes that value through
                // SV_StartInstanceLocation so the shader can select one object transform per draw.
                .add_required_extension_features(VkPhysicalDeviceVulkan11Features{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                    .shaderDrawParameters = VK_TRUE,
                })
                // EXT_shader_object lets us bind shader objects directly instead of building a pipeline.
                .add_required_extension_features(VkPhysicalDeviceShaderObjectFeaturesEXT{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
                    .shaderObject = VK_TRUE,
                })
                // These dynamic-state features allow the command buffer to set rasterization/blending state.
                .add_required_extension_features(VkPhysicalDeviceExtendedDynamicState3FeaturesEXT{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
                    .extendedDynamicState3DepthClampEnable = VK_TRUE,
                    .extendedDynamicState3PolygonMode = VK_TRUE,
                    .extendedDynamicState3RasterizationSamples = VK_TRUE,
                    .extendedDynamicState3SampleMask = VK_TRUE,
                    .extendedDynamicState3AlphaToCoverageEnable = VK_TRUE,
                    .extendedDynamicState3AlphaToOneEnable = VK_TRUE,
                    .extendedDynamicState3LogicOpEnable = VK_TRUE,
                    .extendedDynamicState3ColorBlendEnable = VK_TRUE,
                    .extendedDynamicState3ColorWriteMask = VK_TRUE,
                })
                .add_required_extension_features(VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
                    .vertexInputDynamicState = VK_TRUE,
                })
                // This tutorial uses descriptor heaps to manage shader resources without pipeline layouts or descriptor
                // sets.
                .add_required_extension_features(VkPhysicalDeviceDescriptorHeapFeaturesEXT{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
                    .descriptorHeap = VK_TRUE,
                });
        }

        std::optional<vkb::PhysicalDevice> selectedCandidate;
        // Manually filter vk-bootstrap's candidates to select the best device/GPU possible.
        {
            // Get devices that meet criteria specified above; vk-bootstrap checks for required features and extensions.
            const std::vector<vkb::PhysicalDevice> physicalDeviceCandidates =
                vkbGetIfValid(physicalDeviceSelector.select_devices(), "Selecting Vulkan physical devices");

            // We iterate through the candidates and select the best one based on additional criteria.
            for (const vkb::PhysicalDevice& vkbPhysicalDeviceCandidate : physicalDeviceCandidates) {
                bool valid = true;
                // Create a Vulkan-Hpp wrapper around the raw C handle.
                vk::PhysicalDevice physicalDeviceCandidate{vkbPhysicalDeviceCandidate.physical_device};

                // Vk-bootstrap can return multiple candidates if multiple GPUs support the required features and
                // extensions. This is an example of how to use some properties to further filter GPUs.
                // This is a tutorial, but the best solution might be to sort GPU candidates based on a score calculated
                // from their properties and features, and select the best one.
                // It is good practice to offer configuration options to select a specific GPU if multiple are
                // available.
                //
                // Format properties are not device features, so we need to filter vk-bootstrap's candidates manually.
                // Check that the device supports linear blitting for the texture format used in the tutorial.
                {
                    const vk::FormatProperties2 formatProperties =
                        physicalDeviceCandidate.getFormatProperties2(mainTextureFormat);
                    const vk::FormatFeatureFlags requiredFeatures =
                        vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst |
                        vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
                    if ((formatProperties.formatProperties.optimalTilingFeatures & requiredFeatures) !=
                        requiredFeatures) {
                        valid = false;
                    }
                }
                // If this candidate supports everything the tutorial needs, select it and stop searching.
                // This is fine for a tutorial, but a proper application might score all valid devices and choose the
                // best one.
                if (valid) {
                    selectedCandidate = vkbPhysicalDeviceCandidate;
                    break;
                }
            }
            util::require(selectedCandidate.has_value(),
                          "No Vulkan physical device supports linear-blit mip generation for R8G8B8A8_UNORM textures");
        }

        // Store selected physical device and its properties.
        const vkb::PhysicalDevice physicalDevice = *selectedCandidate;
        const vk::PhysicalDevice selectedPhysicalDevice{physicalDevice.physical_device};

        // Load physical-device properties that are needed later.
        {
            // After selecting the physical device we will use,
            // we need to query its properties and features to cache the ones we will need later.
            // We are doing this during initialization to avoid querying properties and features multiple times later.

            // Cache memory type flags so buffer and image allocation can choose compatible memory.
            {
                const vk::PhysicalDeviceMemoryProperties2 memoryProperties =
                    selectedPhysicalDevice.getMemoryProperties2();
                m_memoryTypeFlags.clear();

                // Small optimization: reserve space in the vector to avoid multiple allocations.
                m_memoryTypeFlags.reserve(memoryProperties.memoryProperties.memoryTypeCount);

                for (std::uint32_t i = 0; i < memoryProperties.memoryProperties.memoryTypeCount; ++i) {
                    m_memoryTypeFlags.push_back(memoryProperties.memoryProperties.memoryTypes[i].propertyFlags);
                }
            }

            // Cache descriptor heap properties for allocation and alignment of descriptors.
            {
                // Query physical device properties with a pNext chain.
                vk::PhysicalDeviceDescriptorHeapPropertiesEXT heapProperties{};
                vk::PhysicalDeviceShaderObjectPropertiesEXT shaderObjectProperties{
                    .pNext = &heapProperties,
                };
                vk::PhysicalDeviceProperties2 properties2{
                    .pNext = &shaderObjectProperties,
                };
                selectedPhysicalDevice.getProperties2(&properties2);

                // Query descriptor heap descriptor sizes from physical-device properties.
                {
                    // These values are more specific than the ones in PhysicalDeviceDescriptorHeapPropertiesEXT.
                    // They are smaller and more granular, allowing descriptors to be packed more tightly in the heap.
                    m_descriptorHeaps.m_sampledImageSize =
                        selectedPhysicalDevice.getDescriptorSizeEXT(vk::DescriptorType::eSampledImage);
                    m_descriptorHeaps.m_samplerSize =
                        selectedPhysicalDevice.getDescriptorSizeEXT(vk::DescriptorType::eSampler);
                    m_descriptorHeaps.m_uniformBufferSize =
                        selectedPhysicalDevice.getDescriptorSizeEXT(vk::DescriptorType::eUniformBuffer);
                    m_descriptorHeaps.m_storageBufferSize =
                        selectedPhysicalDevice.getDescriptorSizeEXT(vk::DescriptorType::eStorageBuffer);
                }

                // Cache descriptor heap properties. These are mostly alignment requirements that we need to respect
                // when allocating and writing to descriptor heaps.
                {
                    // Alignment requirements for offsets of particular descriptor types.
                    m_descriptorHeaps.m_imageDescriptorAlignment = heapProperties.imageDescriptorAlignment;
                    m_descriptorHeaps.m_bufferDescriptorAlignment = heapProperties.bufferDescriptorAlignment;
                    m_descriptorHeaps.m_samplerDescriptorAlignment = heapProperties.samplerDescriptorAlignment;

                    // Alignment requirements for the actual heap in the GPU.
                    m_descriptorHeaps.m_resourceHeap.m_alignment = heapProperties.resourceHeapAlignment;
                    m_descriptorHeaps.m_samplerHeap.m_alignment = heapProperties.samplerHeapAlignment;

                    // The driver needs a minimum reserved range in the heap for internal operations.
                    m_descriptorHeaps.m_resourceHeap.m_minReservedRange = heapProperties.minResourceHeapReservedRange;
                    m_descriptorHeaps.m_samplerHeap.m_minReservedRange = heapProperties.minSamplerHeapReservedRange;

                    // Push data is supposed to be small. Most devices have a limited amount of data we can push.
                    m_descriptorHeaps.m_maxPushDataSize = heapProperties.maxPushDataSize;

                    // Camera and image descriptors are stored contiguously.
                    // We calculate an aligned stride for simplicity. This is the aligned size of the descriptor.
                    // This makes indexing in the descriptor heap easier.
                    {
                        m_descriptorHeaps.m_uniformBufferStride = util::safeCastToU32(util::alignUp(
                            m_descriptorHeaps.m_uniformBufferSize, m_descriptorHeaps.m_bufferDescriptorAlignment));

                        m_descriptorHeaps.m_sampledImageStride = util::safeCastToU32(util::alignUp(
                            m_descriptorHeaps.m_sampledImageSize, m_descriptorHeaps.m_imageDescriptorAlignment));
                    }
                }

                // We have a small shader cache. We need to store our current shader binary properties to know if the
                // cached binary is valid or not.
                {
                    std::copy(shaderObjectProperties.shaderBinaryUUID.begin(),
                              shaderObjectProperties.shaderBinaryUUID.end(),
                              m_shaderBinaryCacheProperties.m_shaderBinaryUUID.begin());
                    m_shaderBinaryCacheProperties.m_shaderBinaryVersion = shaderObjectProperties.shaderBinaryVersion;
                }
            }
        }

        // Create the logical device.
        {
            // After selecting the physical device, we need to create a logical device to interface with it.
            // We need to select and enable the features and extensions we need for our application.
            // We do so using vk-bootstrap, which reuses the physical device selector to create a device with the same
            // criteria as the physical device selection, allowing us to skip recreating a pNext chain with the features
            // we need to enable.

            m_vkbData.m_device =
                vkbGetIfValid(vkb::DeviceBuilder(physicalDevice).build(), "Creating Vulkan logical device");

            m_logicalDevice = vk::Device{m_vkbData.m_device.device};

            // Load device-level function pointers for Vulkan-Hpp wrappers.
            // Before we only loaded the function pointers for the instance-level functions.
            // After creating the logical device, we can load the function pointers that depend on it to skip the
            // loader. Loading the function pointers is a bit tedious, but is handled automatically by Vulkan-Hpp.
            // Other libraries like Volk provide similar functionality for the C API.
            VULKAN_HPP_DEFAULT_DISPATCHER.init(m_vkbData.m_instance.instance, vkGetInstanceProcAddr,
                                               m_vkbData.m_device.device);
        }
    }

    // Fetch queues using vk-bootstrap.
    {

        // Queues are the main way to submit work to the GPU in Vulkan.
        // In Vulkan, you record a command buffer with work and then submit it to a queue to execute it.
        // Queues are key for advanced GPU parallelism.
        // However, it is still viable for most applications to use a universal queue for all operations.
        //
        // Queues are created with the logical device creation, but vk-bootstrap handles that for us.

        // The graphics queue allows us to submit and execute draw commands.
        m_graphicsQueue =
            vk::Queue{vkbGetIfValid(m_vkbData.m_device.get_queue(vkb::QueueType::graphics), "Getting graphics queue")};

        // The present queue will allow us to present rendered images to the surface.
        m_presentQueue =
            vk::Queue{vkbGetIfValid(m_vkbData.m_device.get_queue(vkb::QueueType::present), "Getting present queue")};

        // Multiple Vulkan objects need the queue family index to specify which queue family they depend on.
        m_graphicsQueueFamily = vkbGetIfValid(m_vkbData.m_device.get_queue_index(vkb::QueueType::graphics),
                                              "Getting graphics queue family");

        const uint32_t presentQueueFamily =
            vkbGetIfValid(m_vkbData.m_device.get_queue_index(vkb::QueueType::present), "Getting present queue family");

        // This tutorial assumes graphics and present queues are from the same family for simplicity.
        // Separate-family presentation is intentionally omitted.
        // This beginner tutorial keeps image ownership simple by avoiding queue ownership transfers.
        util::require(m_graphicsQueueFamily == presentQueueFamily,
                      "This tutorial requires graphics and present queues from the same family");
    }

    // Create the swapchain using vk-bootstrap.
    {
        // Vulkan uses the swapchain to manage how images are presented to the surface.
        // The swapchain is a list of images that are presented to the surface.
        // The swapchain is designed to synchronize presentation latency and screen refresh.

        // Selecting the present mode is important for performance and latency.
        // Immediate mode presents images as soon as possible, but can cause visible tearing.
        // FIFO is used by this tutorial and by most applications, as it is widely supported.
        // FIFO will provide a queue of images to present, and display the next image at display refresh.
        // The application will render into an image in the queue, avoiding tearing.
        const VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;

        // The process of creating a swapchain is a bit tedious.
        // We need to query the surface capabilities and select the best format and present mode.
        // Then we need to create the swapchain, vk-bootstrap handles all of this for us.
        {
            vkb::SwapchainBuilder swapchainBuilder =
                vkb::SwapchainBuilder(m_vkbData.m_device)
                    .set_desired_extent(windowWidth, windowHeight)
                    .set_desired_present_mode(presentMode)
                    // The app renders directly into swapchain images as color attachments.
                    .add_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            m_vkbData.m_swapchain = vkbGetIfValid(swapchainBuilder.build(), "Creating Vulkan swapchain");
        }

        // Store format and extent of the swapchain images for later use.
        {
            m_swapchainFormat = vk::Format{m_vkbData.m_swapchain.image_format};
            m_swapchainExtent = vk::Extent2D{
                .width = m_vkbData.m_swapchain.extent.width,
                .height = m_vkbData.m_swapchain.extent.height,
            };
        }

        // Retrieve swapchain images and create views for them.
        {
            // Swapchain images are owned by the swapchain, so we only retrieve their handles.
            // This implies that we do not need to create or destroy the swapchain images,
            // but we do need to create image views for them.
            // Vulkan implementations have a minimum number of images they require for the swapchain.
            // On some devices like Android phones, the minimum number of images can be up to 10.
            // Remember that the number of swapchain images is independent of the number of frames in flight, which
            // is usually 2. vk-bootstrap handles the minimum number of swapchain images for us.
            const std::vector images = vkbGetIfValid(m_vkbData.m_swapchain.get_images(), "Getting swapchain images");

            // Small optimization: reserve space in the vector to avoid multiple allocations.
            m_swapchainImages.reserve(images.size());

            for (VkImage image : images) {

                // vk-bootstrap returns a Vulkan API C handle, so we wrap it in a Vulkan-Hpp wrapper.
                const vk::Image swapchainImage{image};

                // Image views are used to select the particular part of an image used by a resource.
                // They allow us to render or copy to a particular mip level or layer of an image.
                // Descriptor heaps remove the need to create image view objects for descriptors,
                // but we still need to create image views for some use cases like attachments.
                // We need to create image views for the swapchain images, so that we can render into them.
                const vk::ImageViewCreateInfo imageViewInfo{
                    .image = swapchainImage,
                    .viewType = vk::ImageViewType::e2D,
                    .format = m_swapchainFormat,
                    .components = identityComponentMapping,
                    .subresourceRange = CreateImageSubresourceRange(),
                };
                m_swapchainImages.emplace_back(SwapchainImageResources{
                    .m_image = swapchainImage,
                    .m_imageView = m_logicalDevice.createImageViewUnique(imageViewInfo),
                    .m_state = ImageState{.m_aspectMask = vk::ImageAspectFlagBits::eColor},
                });
                const std::size_t swapchainImageIndex = m_swapchainImages.size() - 1U;

                // Set a debug name for all Vulkan objects.
                setDebugName(swapchainImage, vk::ObjectType::eImage,
                             std::format("SwapchainImage[{}]", swapchainImageIndex));
                setDebugName(*m_swapchainImages.back().m_imageView, vk::ObjectType::eImageView,
                             std::format("SwapchainImageView[{}]", swapchainImageIndex));
            }
        }
    }
}

void Application::initCommandPool()
{
    // In Vulkan, we do not issue work to the GPU directly.
    // Instead we record commands into command buffers and submit them to queues.
    // Recording commands into command buffers allows Vulkan to record work in a multithreaded manner.
    //
    // Command buffers can be recorded once and submitted multiple times.
    // However, this is rarely useful in practice because command buffers depend on frames in flight and swapchain
    // images. Moreover, most games are very dynamic and need to record commands every frame.
    //
    // Command buffers are allocated from a command pool.
    // Command pools manage the memory used to store command buffers.
    // Command pools must be externally synchronized, so we need to use multiple pools for multithreaded command
    // recording.
    //
    // Note: command buffers and command pools are key for multithreaded rendering, but this tutorial does not use
    // multithreading for simplicity.

    util::log_msg("[Init] Init CommandPool");

    const vk::CommandPoolCreateInfo commandPoolInfo{
        // eResetCommandBuffer allows each command buffer to be reset and recorded again.
        // We re-use existing command buffers to avoid allocating new ones every frame.
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        // These command buffers will be submitted to the graphics queue, so we need to specify the graphics queue
        // family index.
        // Note: we will also use the present queue for presentation, but we checked that graphics and present
        // queues are from the same family for simplicity.
        .queueFamilyIndex = m_graphicsQueueFamily,
    };

    m_commandPool = m_logicalDevice.createCommandPoolUnique(commandPoolInfo);
    setDebugName(*m_commandPool, vk::ObjectType::eCommandPool, "MainCommandPool");
}

// Frames in flight are a common technique to improve performance.
// This creates objects like semaphores, fences, and command buffers for each frame in flight.
void Application::initFramesInFlightResources()
{
    // Frames in flight are a common technique to improve performance.
    // The idea is to work on multiple frames in parallel, so the GPU stays busy rendering,
    // avoiding stalls while waiting for presentation or CPU work from previous frames.
    //
    // In general, applications should have a fixed number of two frames in flight.
    // This allows them to prepare the next frame while the GPU is rendering and presenting the previous one.
    // Note: frames in flight is completely independent and orthogonal to the number of swapchain images,
    // we will still use 2 frames in flight even if the swapchain has more than 2 images.
    //
    // See Vulkanised 2026: Frames in Flight Demystified - https://www.youtube.com/watch?v=Khxc_Qky-fM
    //
    // Since the GPU has multiple frames in flight, some resources need to be duplicated for each frame in flight.
    // The tutorial duplicates the following resources for each frame in flight: semaphores, fences, command
    // buffers, depth images, and camera uniform buffers.

    util::log_msg("[Init] Init Frames In Flight Resources");

    // Vulkan is an explicit API, so it is the application's responsibility to synchronize resources.
    //
    // Synchronization primitives:
    // - Fences: let the CPU wait for finished GPU work.
    // - Semaphores: let the GPU wait for earlier GPU work, order GPU-side operations, and queue dependencies.
    // - Barriers: control GPU execution and memory dependencies, visibility, etc.
    //             Barriers order resource access and manage layout/state transitions.
    // - Events: allow more granular synchronization, are considered advanced, and are unused in this tutorial.
    //
    // Vulkan has multiple extensions to improve the developer experience when dealing with synchronization.
    // Classic synchronization primitives are still needed, but new features like timeline semaphores, synchronization2
    // and unified image layouts are recommended for most use cases.

    // Binary semaphores are one of the basic Vulkan synchronization primitives.
    // Semaphores are used to synchronize GPU work with other GPU work.
    // They can only be signaled or waited on, and are used to wait and signal between GPU operations.
    // Timeline semaphores are more advanced and solve many problems with binary semaphores.
    // However, swapchain presentation still requires binary semaphores.
    //
    // This semaphore is signaled by acquireNextImageKHR once a swapchain image can be rendered to.
    const vk::SemaphoreCreateInfo semaphoreInfo{};

    // Fences are one of the basic Vulkan synchronization primitives.
    // Fences synchronize GPU work with the CPU.
    //
    // This fence is used to ensure resources from the frame are not used until they are ready.
    const vk::FenceCreateInfo fenceInfo{
        // Start fences as signaled so the first frame does not wait forever for previous work.
        .flags = vk::FenceCreateFlagBits::eSignaled,
    };

    // Vulkan records commands into command buffers and submits them to queues for execution.
    // This allows Vulkan to select the work the GPU will be doing.
    //
    // Allocate all command buffers.
    const vk::CommandBufferAllocateInfo allocateInfo{
        .commandPool = *m_commandPool,
        // Primary buffers can be submitted directly to a queue.
        .level = vk::CommandBufferLevel::ePrimary,
        // Create one command buffer per frame plus one helper buffer for setup-time uploads.
        .commandBufferCount = util::safeCastToU32(m_framesInFlight.size() + 1),
    };

    // Obtain all command buffers.
    const std::vector<vk::CommandBuffer> commandBuffers = m_logicalDevice.allocateCommandBuffers(allocateInfo);
    util::require(commandBuffers.size() == (m_framesInFlight.size() + 1),
                  "Expected one command buffer per frame in flight plus one helper command buffer");

    // Loop to initialize resources for each frame in flight.
    for (std::size_t frameIndex = 0; frameIndex < m_framesInFlight.size(); ++frameIndex) {
        FrameInFlightResources& frame = m_framesInFlight[frameIndex];

        frame.m_commandBuffer = commandBuffers[frameIndex];

        // Camera data is usually updated on each frame, so we duplicate the resource.
        // This allows the CPU to update the camera while the previous frame is still rendering with the previous
        // camera buffer.
        frame.m_camera = createBuffer(
            sizeof(CameraData), vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            std::format("FrameCameraBuffer[{}]", frameIndex));

        // This semaphore is signaled by acquireNextImageKHR once a swapchain image can be rendered to.
        // This allows the graphics queue to wait on this per-frame semaphore before rendering into the acquired
        // swapchain image.
        frame.m_imageAvailableSemaphore = m_logicalDevice.createSemaphoreUnique(semaphoreInfo);

        // This fence is used to ensure resources from the frame are not used until they are ready.
        frame.m_inFlightFence = m_logicalDevice.createFenceUnique(fenceInfo);

        // Note: We are using the helper command buffer to transition the depth images after creating them.
        // So we create and initialize the depth images later with the other images, see initDepthImages.

        // Set debug names for all Vulkan objects related to the frame in flight for easier debugging.
        {
            setDebugName(frame.m_commandBuffer, vk::ObjectType::eCommandBuffer,
                         std::format("FrameCommandBuffer[{}]", frameIndex));
            setDebugName(*frame.m_imageAvailableSemaphore, vk::ObjectType::eSemaphore,
                         std::format("ImageAvailableSemaphore[{}]", frameIndex));
            setDebugName(*frame.m_inFlightFence, vk::ObjectType::eFence,
                         std::format("FrameInFlightFence[{}]", frameIndex));
        }
    }

    // Create the helper command buffer for immediate upload operations outside of the render loop.
    // This helper command buffer is used for operations like uploading textures or generating mipmaps.
    m_helperCommandBuffer.m_commandBuffer = commandBuffers.back();
    m_helperCommandBuffer.m_active = false;
    setDebugName(m_helperCommandBuffer.m_commandBuffer, vk::ObjectType::eCommandBuffer, "HelperCommandBuffer");
}

void Application::initSwapchainImageSyncObjects()
{
    // We need to initialize some synchronization objects for each swapchain image.
    util::log_msg("[Init] Init SwapchainImageSyncObjects");

    // Binary semaphores are one of the basic Vulkan synchronization primitives.
    // Semaphores are used to synchronize GPU work with other GPU work.
    const vk::SemaphoreCreateInfo semaphoreInfo{};

    for (std::uint32_t swapchainImageIndex = 0; swapchainImageIndex < m_swapchainImages.size(); ++swapchainImageIndex) {
        SwapchainImageResources& swapchainImage = m_swapchainImages[swapchainImageIndex];

        // Present waits on this semaphore before it displays the image.
        swapchainImage.m_renderFinishedSemaphore = m_logicalDevice.createSemaphoreUnique(semaphoreInfo);

        setDebugName(*swapchainImage.m_renderFinishedSemaphore, vk::ObjectType::eSemaphore,
                     std::format("RenderFinishedSemaphore[{}]", swapchainImageIndex));

        // This state tracks the swapchain image between frames and helps create barriers.
        // We need to select the default aspect mask for the swapchain image, which is color.
        // But we use the default layout of the swapchain image, which is undefined.
        // The first time we acquire the swapchain image, we will transition it to the correct layout for rendering.
        swapchainImage.m_state = ImageState{.m_aspectMask = vk::ImageAspectFlagBits::eColor};
    }
}

void Application::initDepthImages()
{
    // When we are rendering multiple objects, Vulkan needs to know which objects should overwrite others.
    // This is done using a depth buffer, which stores the depth of each pixel in the scene.
    //
    // In Vulkan, we need to create a depth image and a depth image view. During rendering, objects
    // will write their depth to the depth image, and we will decide which fragments to keep based on their depth
    // values. This process is called depth testing or Z testing.
    //
    // This tutorial uses a depth image per frame in flight.
    //
    // Note: If we wanted to implement resize, we would need to recreate the depth images along with the swapchain
    // images, since the depth images depend on the swapchain extent.
    // Note: The number of depth images depends on the number of frames in flight, not the number of swapchain
    // images. This is easy to miss in traditional framebuffer-per-swapchain-image implementations.
    util::log_msg("[Init] Init DepthImages");

    // Keep one depth attachment per frame in flight so overlapping frames never render into the same depth image.

    // Obtain the helper command buffer for the depth image layout transition.
    const vk::CommandBuffer commandBuffer = getHelperCommandBuffer();

    for (std::uint32_t frameIndex = 0; frameIndex < m_framesInFlight.size(); ++frameIndex) {
        GpuViewImage& depthImage = m_framesInFlight[frameIndex].m_depthImage;

        // Create the depth image and allocate memory for it.
        {
            // Depth images are normal device-local images that this app owns, unlike swapchain images.
            // Since the app owns the depth images, we need to create them and allocate memory for them.
            const vk::ImageCreateInfo imageInfo{
                .imageType = vk::ImageType::e2D,
                .format = depthFormat,
                .extent =
                    vk::Extent3D{.width = m_swapchainExtent.width, .height = m_swapchainExtent.height, .depth = 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .tiling = vk::ImageTiling::eOptimal,
                .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
                .sharingMode = vk::SharingMode::eExclusive,
                .initialLayout = vk::ImageLayout::eUndefined,
            };
            depthImage.m_image.m_image = m_logicalDevice.createImageUnique(imageInfo);
            depthImage.m_image.m_states.resize(1, ImageState{.m_aspectMask = vk::ImageAspectFlagBits::eDepth});
            setDebugName(*depthImage.m_image.m_image, vk::ObjectType::eImage,
                         std::format("DepthImage[{}]", frameIndex));
        }

        // Allocate memory and bind it to the image.
        allocateGpuImage(depthImage.m_image);

        // Create an image view for the depth image.
        {
            // Image views are used to select the particular part of an image used by a resource.
            // Rendering uses image views, not raw images, so create one depth view per depth image.
            const vk::ImageViewCreateInfo viewInfo{
                .image = *depthImage.m_image.m_image,
                .viewType = vk::ImageViewType::e2D,
                .format = depthFormat,
                .subresourceRange = CreateImageSubresourceRange(0, vk::ImageAspectFlagBits::eDepth),
            };
            depthImage.m_imageView = m_logicalDevice.createImageViewUnique(viewInfo);
            setDebugName(*depthImage.m_imageView, vk::ObjectType::eImageView,
                         std::format("DepthImageView[{}]", frameIndex));
        }

        // The depth attachment will be written by depth testing during rendering.
        // Each depth image is paired with one frame in flight and is only used as a depth attachment.
        // Because it never becomes a present image, or has any other use, this one setup-time transition is enough;
        // the per-frame fence prevents reuse while older rendering work may still touch it.
        depthImage.m_image.transition(commandBuffer,
                                      ImageState{
                                          .m_layout = depthAttachmentLayout,
                                          .m_stageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests,
                                          .m_accessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                                          .m_aspectMask = vk::ImageAspectFlagBits::eDepth,
                                      });
    }
}

//****************
// Scene initialization
//****************

void Application::initSceneResources()
{
    util::log_msg("[Init] Init Scene Resources");

    const std::filesystem::path dataDir = SIGGRAPH_DATA_DIR;
    std::array<std::filesystem::path, SceneData::sceneFileNames.size()> gltfPaths{};
    for (std::size_t i = 0; i < SceneData::sceneFileNames.size(); ++i) {
        gltfPaths[i] = dataDir / std::string{SceneData::sceneFileNames[i]};
        util::log_msg("[Init] glTF scene: {}", gltfPaths[i].string());
    }

    // This tutorial stores the data to draw the scene in multiple vectors.
    //  * m_meshes stores the GPU mesh data. The mesh data is the geometry of the scene.
    //      For each mesh, we store the vertex and index buffers on the GPU.
    //  * m_textures stores the GPU texture data.
    //  * m_drawData stores the draw data. Each draw data entry describes a single draw call, and contains the mesh
    //      id, object id, shader variant, and texture ids. When rendering, we will loop over the draw data and issue a
    //      draw call for each entry, binding the shaders and corresponding descriptors.
    //
    // glTF is a popular open format developed by the Khronos group for 3D scenes and models.
    // It is designed to be efficient to transmit and load, and is widely supported by 3D content creation tools.
    //
    // This function will parse the glTF files and load the scene data into CPU-side structures.
    // We have already implemented a glTF parser that returns an easy-to-use structure for our application, but we
    // still need to upload the data to the GPU and create the draw data.
    //
    // Note: our application supports loading and merging multiple glTF files.
    // Some assets are distributed as a series of glTF files with different layers.
    util::log_msg("[Init] Reading glTF files");
    const util::gltf::ParsedData gltfData = util::gltf::parseGltfFiles(gltfPaths);

    // We store the object data in this CPU-side vector.
    // The CPU copy can be deleted after uploading it to the GPU.
    std::vector<ObjectData> objects;

    util::log_msg("[Init] Uploading Scene Meshes");
    initSceneMeshesAndDrawData(gltfData, objects);

    util::log_msg("[Init] Init Scene Camera");
    initSceneCamera(gltfData);

    util::log_msg("[Init] Init Object Buffer");
    initObjectBuffer(objects);

    util::log_msg("[Init] Init Point Light");
    initPointLight(gltfData);

    util::log_msg("[Init] Init Solid Color");
    initSolidColor();

    util::log_msg("[Init] Init Scene Textures");
    initSceneTextures(gltfData);
}

void Application::initSceneMeshesAndDrawData(const util::gltf::ParsedData& gltfData, std::vector<ObjectData>& objects)
{
    // glTF parsing gives us a CPU-side scene description.
    // This step turns that description into data we will upload to the GPU for rendering.
    // We will also keep some data on the CPU to be able to issue the draw calls and bind descriptors.

    objects.resize(gltfData.nodes.size());

    // Small optimization: reserve space in the vectors to avoid reallocations.
    // Note: we are storing indices into the vectors instead of references.
    // This has slight overhead but indices will still be valid if the vector reallocates.
    m_scene.m_meshes.reserve(gltfData.meshes.size());
    m_scene.m_drawData.reserve(gltfData.nodes.size());

    // Get glTF data for meshes and upload it to the GPU.
    const std::uint32_t meshCount = util::safeCastToU32(gltfData.meshes.size());
    for (std::uint32_t meshId = 0; meshId < meshCount; ++meshId) {

        // Generate the buffers with the mesh data for this mesh.
        const util::MeshGeometryData geometry = util::gltf::buildMeshGeometryData(meshId, gltfData);

        // Note: we only keep the GPU copy of the mesh data, and discard the CPU copy after uploading it to the GPU.
        const GpuMesh& gpuMesh = m_scene.m_meshes.emplace_back(GpuMesh{
            // For simplicity we use a single vertex buffer per mesh in the tutorial.
            // Some GPU implementations would benefit if the application splits vertex data into position and
            // non-positional attributes.
            .m_vertices = uploadToNewGpuBuffer(std::as_bytes(std::span{geometry.vertices}),
                                               vk::BufferUsageFlagBits::eVertexBuffer,
                                               std::format("{} VertexBuffer", geometry.name)),
            // For most meshes, vertices are reused across multiple triangles, so we use an index buffer to avoid
            // duplicating vertex data.
            .m_indices =
                uploadToNewGpuBuffer(std::as_bytes(std::span{geometry.indices}), vk::BufferUsageFlagBits::eIndexBuffer,
                                     std::format("{} IndexBuffer", geometry.name)),
            .m_indexCount = util::safeCastToU32(geometry.indices.size()),
        });

        // Validate the mesh data.
        {
            util::require(gpuMesh.m_indexCount > 0, std::format("{} has a mesh without any indices", geometry.name));

            util::require(gpuMesh.m_indices.m_size ==
                              (static_cast<vk::DeviceSize>(gpuMesh.m_indexCount) * sizeof(std::uint32_t)),
                          std::format("{} index buffer byte size does not match stored index count", geometry.name));

            // Check that indices are inside the vertex range.
            // Note: m_indexCount > 0 so geometry.indices.size() > 0.
            util::require(*std::max_element(geometry.indices.begin(), geometry.indices.end()) <
                              geometry.vertices.size(),
                          std::format("{} has an index outside its vertex range.", geometry.name));

            util::require((gpuMesh.m_vertices.m_size > 0) &&
                              (gpuMesh.m_vertices.m_size == (sizeof(Vertex) * geometry.vertices.size())),
                          std::format("{} has a mesh with an invalid vertex buffer size of {} ({} * {})", geometry.name,
                                      gpuMesh.m_vertices.m_size, sizeof(Vertex), geometry.vertices.size()));
        }
    }

    // Get node data for the glTF and store it for draws and descriptors.
    //
    // In a normal game, it is common for the same mesh to be reused multiple times in the scene, but with different
    // transforms and materials. Similarly, it is also common for the same texture to be reused multiple times in
    // the scene, but by different meshes and materials. The easiest way to handle this is to store the mesh and
    // texture data once, and then have multiple draw calls that reference the same mesh data.
    //
    // In this tutorial draw data is stored in the CPU and used to issue the draw calls.
    const std::uint32_t nodeCount = util::safeCastToU32(gltfData.nodes.size());
    for (std::uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {

        const util::gltf::Node& node = gltfData.nodes[nodeIndex];

        objects[nodeIndex].model = util::math::generateModel(node.pos, node.eulerAngles, node.scale);

        util::require(node.meshId < m_scene.m_meshes.size(), "glTF node references an invalid mesh id");

        // We use a special shader for objects with an albedo texture but not a normal map.
        // Depending on your architecture and GPU, using a default texture for the normal map might be better.
        const ShaderVariant shaderVariant =
            node.albedoTextureId == util::gltf::invalidGltfId
                ? ShaderVariant::SolidColor
                : (node.normalTextureId == util::gltf::invalidGltfId ? ShaderVariant::Albedo
                                                                     : ShaderVariant::AlbedoAndNormal);

        m_scene.m_drawData.push_back(SceneDraw{
            .m_meshId = node.meshId,
            .m_objectIndex = nodeIndex,
            .m_shaderVariant = shaderVariant,
            .m_albedoTextureIndex = node.albedoTextureId,
            .m_normalTextureIndex = node.normalTextureId,
            .m_debugName = node.name,
        });
    }
}

void Application::initSceneCamera(const util::gltf::ParsedData& gltfData)
{
    // The parsed glTF camera becomes the runtime free-camera starting point.
    // Each frame has a persistently mapped camera buffer so the CPU can update one frame slot without waiting for other
    // frames that may still be executing on the GPU.
    m_cameraPos = gltfData.cameraPos;
    const auto [cameraYaw, cameraPitch] = util::math::calculateYawPitch(gltfData.cameraPos, gltfData.cameraLookAt);
    m_cameraYaw = cameraYaw;
    m_cameraPitch = cameraPitch;

    for (FrameInFlightResources& frame : m_framesInFlight) {
        uploadCameraData(frame);
    }
}

void Application::initObjectBuffer(std::span<const ObjectData> objects)
{
    // This buffer contains data for each node.
    // In this tutorial, the only data we store is the model matrix.
    // In a real application, we would also store other per-object data like animation data, etc.
    //
    // In our tutorial the vertex shader reads one model matrix per draw from this storage buffer.
    // The draw's firstInstance value is used to index into the buffer and select the correct object data.
    m_scene.m_objects = uploadToNewGpuBuffer(
        std::as_bytes(objects), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        "SceneObjectBuffer");
}

void Application::initPointLight(const util::gltf::ParsedData& gltfData)
{
    // In the tutorial, we hard-coded a single point light for the scene and store it as a uniform buffer on the GPU.
    const glm::vec3 lightPosition = gltfData.cameraLookAt + glm::vec3{0.0F, 4.0F, 2.0F};
    const LightData light{
        .pointPosition = lightPosition,
        .pointIntensity = 0.90F,
        .ambientColor = glm::vec3{1.0F, 0.92F, 0.78F},
        .ambientIntensity = 0.12F,
    };
    m_scene.m_pointLight =
        uploadToNewGpuBuffer(std::as_bytes(std::span{&light, std::size_t{1}}),
                             vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                             "ScenePointLightBuffer");
}

void Application::initSolidColor()
{
    // We have a shader using a fixed solid color.
    // Currently the tutorial sets this color to a hard-coded value in a UBO.
    // A real application would likely have a fixed color per object, and might send this data as push data, rather
    // than a UBO.
    const SolidColorData solidColor{
        // Official red tone of the Vulkan logo.
        .color = glm::vec4{util::sRgbToLinear(164, 30, 34), 1.0F},
    };
    m_scene.m_solidColor =
        uploadToNewGpuBuffer(std::as_bytes(std::span{&solidColor, std::size_t{1}}),
                             vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                             "SceneSolidColorBuffer");
}

void Application::initSceneTextures(const util::gltf::ParsedData& gltfData)
{
    // From the Vulkan point of view, normal, albedo, and all other textures are the same type of object and
    // resource. So they are treated the same way in this tutorial and stored together in this vector.

    // Small optimization: reserve space in the vector to avoid reallocations.
    // We store indices into this vector in the draw data.
    // This has slight overhead but allows us to avoid invalidating references when the vector is resized.
    m_scene.m_textures.reserve(gltfData.textures.size());

    // Read all textures in the glTF and store them as GPU images.
    for (const util::gltf::GltfTexture& textureInfo : gltfData.textures) {
        util::log_msg("[Init] Reading Texture: {}", textureInfo.filename);
        // Read pixel data and store it in the CPU. This will be deleted after uploading it to the GPU.
        const util::ImageRgba8 texture = util::readImageFileRgba8(textureInfo.filename);

        // Upload texture to the GPU and generate mips.
        m_scene.m_textures.push_back(createTexture(texture, textureInfo.name));
    }
    util::require(!m_scene.m_textures.empty(), "Textures expected. The glTF file appears incorrect");
}

//****************
// Shader and descriptor setup
//****************

void Application::initDescriptorHeaps()
{
    util::log_msg("[Init] Init Descriptor Heaps");

    // In Vulkan, descriptors describe resources such as buffers and images to a shader.
    // Descriptors are opaque data blobs that describe a resource like a buffer or an image.
    // They usually contain a pointer to the resource, and some metadata about how the resource is used.
    //
    // Classic Vulkan tutorials usually teach descriptor set layouts, descriptor pools, descriptor sets, and
    // pipeline layouts. This tutorial uses EXT_descriptor_heap instead. The shader still declares normal
    // set/binding resources, but shader-object creation receives a mapping table that translates those bindings
    // into byte offsets in these heap buffers.
    //
    // Descriptor heaps are more flexible than descriptor sets.
    // Descriptors are opaque blobs stored in memory. The application can change the descriptor bound to a shader by
    // writing to this memory, including copying descriptors and passing them around in shaders.
    //
    // Descriptor heaps are also more manual. We calculate the space needed for the descriptors and allocate GPU
    // buffers to hold them. The application is responsible for managing and writing heap memory.
    //
    // The mapping table links shader bindings to byte offsets in the heap.
    // Applications can write descriptors directly into heap buffers, then select which descriptor is accessed
    // from a shader by mapping bindings to offsets in this buffer. The data flow in this tutorial is:
    // shader binding -> descriptor heap mapping -> heap byte offset -> read descriptor bytes from heap -> resource.
    //
    // It is possible to use descriptor heaps with untyped pointers, and to skip mappings and shader sets and bindings.
    // This tutorial uses shader bindings and mapping tables to make the code easier to read and understand.
    // We only use constant offsets and push indices for resource mapping.
    // When porting existing applications, it is worth considering other mapping sources supported by descriptor heaps.
    //
    // It is important to highlight that descriptor heaps are a different way of working with descriptors.
    // Descriptor heaps are based on manual memory management, which requires a lot more manual work.
    // When selecting the physical device in initVulkanVKB, we need to cache the selected device's descriptor sizes
    // and alignments alongside the heap alignments. Then we need to manage the offsets in the heap manually, to
    // ensure that all descriptors are correctly aligned. Descriptor heaps allow more advanced use cases not
    // explored in this tutorial. For example, they allow descriptor data to be copied and moved using GPU shaders.

    // Calculate descriptor byte offsets in the descriptor heap.
    {
        // With descriptor heaps, we need to manage two heaps, one for resources and one for samplers.
        // We start by managing the resource heap and reserving space for each resource descriptor.
        //
        // We will calculate the offset of each descriptor in the resource descriptor heap.
        // Our code needs to ensure that the descriptor offsets are correctly aligned. After allocating all descriptors,
        // we will obtain the total size of the descriptor heap, which we can use to allocate the GPU buffer.

        // The tutorial uses a fixed resource-heap layout to keep the code simple:
        // We decided to use a fixed resource heap structure with the following blocks:
        // [reserved] Reserved space for internal driver usage, required by the extension.
        //            Can be placed anywhere in the heap, but this tutorial reserves the start of the heap.
        // [camera uniform 0][camera uniform 1][ ... ] Descriptors for the buffers containing the camera data.
        //            We have one camera buffer per frame in flight. These descriptors are mapped using
        //            **push indices**, so the shader can select the correct camera buffer for the current frame.
        // [object storage] This is a huge buffer containing all the object data for the scene.
        //            This is mapped using a **constant offset**. The shader indexes it using firstInstance.
        // [point light uniform] This buffer contains the point light data for the scene.
        //            It is mapped using a **constant offset** and does not change in this tutorial.
        // [solid color uniform] This buffer contains the solid color data for the shader variant.
        //            It is constant, so it is mapped using a **constant offset** and does not change in this tutorial.
        //            A more realistic use case might send it per draw call using push data.
        // [sampled image 0][sampled image 1][ ... ] Descriptors for the sampled images.
        //            We have a descriptor for each texture in the scene.
        //            Note how combined sampled-image shader variables still read their image
        //            descriptor bytes from the resource heap. The sampler descriptor bytes come from the sampler heap.
        //            These descriptors are mapped using **push indices**; we select the correct texture for each draw
        //            call using the draw data.
        //
        // Note: The previous diagram omits padding before each block, which is required to satisfy the alignment
        // requirements.

        // cursorResourceHeap is a local variable that tracks the current offset in the resource descriptor heap.
        // It is used to calculate each descriptor offset and the final heap size.
        //
        // The first block on the heap is reserved for internal driver usage, so we reserve that space.
        // Device properties report the minimum reserved range required by the implementation.
        // The application must reserve this space and not access it while the heap range is bound.
        // This tutorial reserves the start of the heap, but the reserved range can be placed elsewhere.
        VkDeviceSize cursorResourceHeap = m_descriptorHeaps.m_resourceHeap.m_minReservedRange;

        // Advance descriptor heap offsets to allocate the resources.
        {
            // The basic idea is to use cursorAllocateResourceDescriptorRange to advance the cursor and reserve
            // space for each descriptor in the heap. As explained above, the heap layout is fixed, so we can
            // calculate the offsets for each descriptor in the heap.

            // Small helper to advance the descriptor heap offset cursor.
            // It returns the aligned offset for the new descriptor,
            // and reserves space in the heap to store the new descriptor.
            // Note: we are not creating the descriptor bytes, just reserving space for them in the heap.
            const auto cursorAllocateResourceDescriptorRange = [&cursorResourceHeap](VkDeviceSize size,
                                                                                     VkDeviceSize alignment) {
                // Align the current position in the heap to the required descriptor alignment.
                // This aligned position is returned and will be the offset of the descriptor in the heap.
                cursorResourceHeap = util::alignUp(cursorResourceHeap, alignment);

                const std::uint32_t offset = util::safeCastToU32(cursorResourceHeap);

                // Reserve enough space in the heap for the new descriptor.
                cursorResourceHeap += size;

                return offset;
            };
            // Use the helper to advance the cursor and reserve space for each descriptor in the heap in our fixed
            // layout.
            //
            // Note: All descriptor sizes and alignments are cached in initVulkanVKB, so we can use them here to
            // calculate offsets. Descriptor alignment refers to descriptor offsets in the heap;
            // resource and sampler heap alignment requirements refer to the GPU address used when binding the heap.

            // Note: we use stride (aligned-up size) to allocate all camera buffers at once.
            // This makes it easier to index each buffer descriptor in the heap.
            m_descriptorHeaps.m_cameraOffset = cursorAllocateResourceDescriptorRange(
                static_cast<VkDeviceSize>(m_descriptorHeaps.m_uniformBufferStride) * m_framesInFlight.size(),
                m_descriptorHeaps.m_bufferDescriptorAlignment);

            m_descriptorHeaps.m_objectsOffset = cursorAllocateResourceDescriptorRange(
                m_descriptorHeaps.m_storageBufferSize, m_descriptorHeaps.m_bufferDescriptorAlignment);
            m_descriptorHeaps.m_pointLightOffset = cursorAllocateResourceDescriptorRange(
                m_descriptorHeaps.m_uniformBufferSize, m_descriptorHeaps.m_bufferDescriptorAlignment);
            m_descriptorHeaps.m_solidColorOffset = cursorAllocateResourceDescriptorRange(
                m_descriptorHeaps.m_uniformBufferSize, m_descriptorHeaps.m_bufferDescriptorAlignment);

            // Note: we use stride (aligned-up size) to allocate all images at once.
            // This is similar to how we handle camera buffer descriptors.
            m_descriptorHeaps.m_pushTextureOffset = cursorAllocateResourceDescriptorRange(
                static_cast<VkDeviceSize>(m_descriptorHeaps.m_sampledImageStride) * m_scene.m_textures.size(),
                m_descriptorHeaps.m_imageDescriptorAlignment);
        }

        // At the end, the cursor contains the total size of the resource descriptor heap.
        m_descriptorHeaps.m_resourceHeap.m_rangeSize = cursorResourceHeap;
    }
    // Calculate sampler heap offset.
    {
        // When interacting with descriptor heaps, we need to manage two heaps, one for resources and one for samplers.
        // We completed our resource heap. Now we need to deal with sampler descriptors. In this tutorial,
        // we only use two samplers, so the layout is even simpler than the resource heap.
        //
        // This tutorial only uses two samplers, so the sampler heap layout is simple:
        // [reserved] Implementation-reserved range. Can be placed anywhere in the heap.
        // [linear sampler]  A linear sampler descriptor,
        //            used by all combined sampled-image mappings for albedo textures.
        // [nearest sampler]  A nearest sampler descriptor,
        //            used by all combined sampled-image mappings for normal maps.

        // Reserve implementation space at the beginning of the sampler heap, similar to the resource heap.
        // Note: the reserved range can be placed anywhere in the heap.
        VkDeviceAddress cursorSamplerHeap = m_descriptorHeaps.m_samplerHeap.m_minReservedRange;

        // Align the cursor to the required alignment for sampler descriptors.
        cursorSamplerHeap = util::alignUp(cursorSamplerHeap, m_descriptorHeaps.m_samplerDescriptorAlignment);

        // Store the aligned offset of the first sampler descriptor in the heap.
        m_descriptorHeaps.m_linearSamplerOffset = util::safeCastToU32(cursorSamplerHeap);

        // Increase and align for the next sampler descriptor.
        cursorSamplerHeap = util::alignUp(cursorSamplerHeap + m_descriptorHeaps.m_samplerSize,
                                          m_descriptorHeaps.m_samplerDescriptorAlignment);

        m_descriptorHeaps.m_nearestSamplerOffset = util::safeCastToU32(cursorSamplerHeap);

        // Add the sampler descriptor size to the cursor to calculate the total size of the sampler heap.
        m_descriptorHeaps.m_samplerHeap.m_rangeSize = cursorSamplerHeap + m_descriptorHeaps.m_samplerSize;
    }

    // Allocate descriptor heap buffers.
    {

        // We calculated the total size of the resource and sampler descriptor heaps.
        // Now we need to allocate GPU buffers to hold the descriptor bytes.

        // This function allocates a GPU buffer for a descriptor heap.
        // Sets the required usage flags, adds padding for GPU address alignment,
        // and calculates the bind offset for the heap range.
        auto allocateAlignedBuffer = [this](DescriptorHeapResources::DescriptorHeapData& descriptorHeapData,
                                            const std::string& debugName) {
            // Allocating a GPU buffer and ensuring it is aligned to the required GPU address alignment is
            // a common operation in Vulkan.
            // A new buffer's device address is not guaranteed to satisfy any GPU alignment.
            // The common solution is to allocate extra padding and bind an aligned subrange inside the buffer.
            //
            // Vulkan specifies GPU-address alignment requirements for resource and sampler heap ranges.
            // initVulkanVKB caches the alignment requirements for the selected physical device.

            // Mark the buffer as descriptor-heap memory and request a shader device address for the heap range.
            const vk::BufferUsageFlags heapUsage{vk::BufferUsageFlagBits::eDescriptorHeapEXT |
                                                 vk::BufferUsageFlagBits::eShaderDeviceAddress};

            // We allocate enough extra space that we can bind an aligned subrange inside the buffer.
            // alignedSize = rangeSize + ((alignment > 0) ? (alignment - 1) : 0);
            const VkDeviceSize alignedSize =
                util::alignedAllocationSize(descriptorHeapData.m_rangeSize, descriptorHeapData.m_alignment);

            // We are creating a host-visible buffer so we can write descriptors directly into it.
            // A proper application would use a device-local buffer and a staging buffer to upload descriptors.
            const vk::MemoryPropertyFlags memoryProperties =
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

            // Create the GPU buffer for the descriptor heap.
            descriptorHeapData.m_buffer = createBuffer(alignedSize, heapUsage, memoryProperties, debugName);

            // Bound heap ranges must start at device-aligned GPU addresses.
            // We cached the offset required to move from the buffer's base address to the aligned heap base used by
            // Vulkan. We use this bindOffset to calculate the vk::DeviceAddressRangeEXT used when binding the heap.
            descriptorHeapData.m_bindOffset =
                util::alignedOffset(descriptorHeapData.m_buffer.m_addressGPU, descriptorHeapData.m_alignment);

            // Verify the alignment math.
            {
                // Check the main conditions needed for an aligned heap range.

                // Check that the bound heap base is correctly aligned.
                util::require(util::alignUp(descriptorHeapData.m_buffer.m_addressGPU + descriptorHeapData.m_bindOffset,
                                            descriptorHeapData.m_alignment) ==
                                  descriptorHeapData.m_buffer.m_addressGPU + descriptorHeapData.m_bindOffset,
                              "Descriptor heap bind address is not aligned");

                // Check that the buffer size is large enough.
                util::require(descriptorHeapData.m_bindOffset + descriptorHeapData.m_rangeSize <=
                                  descriptorHeapData.m_buffer.m_size,
                              "Aligned resource heap range does not fit in the allocated buffer");
            }
        };

        // Allocate aligned buffers for the resource and sampler descriptor heaps.
        {
            // We are allocating two GPU buffers: one for the resource heap and one for the sampler heap.
            // We can allocate a single GPU buffer for both heaps, but this tutorial keeps them separate for simplicity.
            allocateAlignedBuffer(m_descriptorHeaps.m_resourceHeap, "ResourceDescriptorHeap");
            allocateAlignedBuffer(m_descriptorHeaps.m_samplerHeap, "SamplerDescriptorHeap");
        }
    }

    // Write resource heap descriptors.
    {

        // Currently, the heap buffers are empty. We need to write the descriptors into the heap buffers.
        //
        // In previous steps we calculated the offsets and ranges for each descriptor in the heap.
        // Now we need to write the descriptors to those offsets in the heap buffers.
        // The descriptors are opaque bytes that describe a resource like a buffer or an image.
        // They are opaque but usually contain a GPU address and some metadata about the resource.
        // Before creating the descriptors, we need to create the actual resources they reference.
        // In previous functions we created the GPU buffers and images for the scene,
        // now we need to create the descriptors that reference those resources.
        // When creating descriptors we need to add some metadata about the resource,
        // like size, format, etc. In Vulkan 1.0 we needed to create ImageViews and descriptor set layouts.
        // Descriptor heaps greatly simplify the process, avoiding intermediate views and creating
        // descriptors directly.
        //
        // In summary, to write a descriptor we need to:
        //  - Specify the host address where writeResourceDescriptorsEXT writes the descriptor bytes.
        //    This tutorial uses the descriptor's final offset in the heap, instead of a temporary staging buffer.
        //  - Create the resource that the descriptor will reference (buffer or image).
        //  - Provide the metadata needed to describe the resource, such as size, format, and layout.
        //
        //  Note: it is also possible to write descriptors to temporary host-visible memory and copy them to the final
        //        heap location. That can allow more performant solutions like a device-local descriptor heap,
        //        using a staging buffer to upload descriptors. This tutorial uses a host-visible heap for simplicity.

        // We use vectors to store descriptor create data and keep pointer targets alive until
        // writeResourceDescriptorsEXT writes the descriptors.
        std::vector<vk::ResourceDescriptorInfoEXT> resources;
        std::vector<vk::HostAddressRangeEXT> descriptorRanges;
        std::vector<vk::DeviceAddressRangeEXT> bufferRanges;
        std::vector<vk::ImageViewCreateInfo> imageViewInfos;
        std::vector<vk::ImageDescriptorInfoEXT> imageInfos;

        // Fixed scene buffer descriptors: objects, point light, and solid color.
        constexpr std::size_t fixedSceneBufferDescriptorCount = 3;

        const std::size_t cameraDescriptorCount = m_framesInFlight.size();
        const std::size_t totalBufferDescriptorCount = cameraDescriptorCount + fixedSceneBufferDescriptorCount;
        const std::size_t totalImageDescriptorCount = m_scene.m_textures.size();
        const std::size_t totalDescriptorCount = totalBufferDescriptorCount + totalImageDescriptorCount;

        // Reserve all entries up front so adding elements does not invalidate those pointers.
        {
            // The descriptor create structs store pointers to other vector elements.
            // If the vectors grow, those pointers would be invalidated before writeResourceDescriptorsEXT is called.
            // Here we reserve enough space in the vectors to avoid reallocations and keep those pointers valid.
            // emplaceNoRealloc enforces the exact reserve sizes below, ensuring the pointers are not invalidated.

            resources.reserve(totalDescriptorCount);
            descriptorRanges.reserve(totalDescriptorCount);

            bufferRanges.reserve(totalBufferDescriptorCount);

            imageViewInfos.reserve(totalImageDescriptorCount);
            imageInfos.reserve(totalImageDescriptorCount);
        }

        // Helper function to emplace an element in a vector and ensure it does not reallocate.
        // This keeps pointers to vector elements valid.
        // We could also write each descriptor in a separate call to writeResourceDescriptorsEXT,
        // or have a vector of unique_ptrs, but this is simpler and more efficient.
        auto emplaceNoRealloc = [](auto& vec, auto&&... args) -> auto& {
            util::require(vec.size() < vec.capacity(), "Vector exceeded reserved capacity");
            return vec.emplace_back(std::forward<decltype(args)>(args)...);
        };

        // Write resource heap descriptors.
        {
            util::require(m_descriptorHeaps.m_resourceHeap.m_buffer.m_addressCPU != nullptr,
                          "Resource descriptor heap must be CPU-mapped before writing descriptors");

            // The heap buffer is host-visible and coherent, so writeResourceDescriptorsEXT can write descriptor
            // bytes directly into mapped memory. A proper application would use a device-local buffer and a staging
            // buffer to upload descriptors.
            std::byte* const resourceHeapBase =
                static_cast<std::byte*>(m_descriptorHeaps.m_resourceHeap.m_buffer.m_addressCPU) +
                m_descriptorHeaps.m_resourceHeap.m_bindOffset;

            // Add the create data and destination range for one buffer descriptor.
            auto addBufferDescriptor = [&bufferRanges, &descriptorRanges, &resources, emplaceNoRealloc,
                                        resourceHeapBase](vk::DescriptorType type, const GpuBuffer& buffer,
                                                          vk::DeviceSize bufferOffset, vk::DeviceSize size,
                                                          std::uint32_t heapOffset, vk::DeviceSize descriptorSize) {
                // Validate that the descriptor points at a valid GPU-addressable buffer range.
                {
                    // Ensure the buffer has a GPU address. Buffer descriptors reference device-address ranges.
                    util::require(buffer.m_addressGPU != 0, "Buffer descriptor requires a GPU address");
                    // Ensure the offset and size are within the buffer's range.
                    util::require(bufferOffset <= buffer.m_size && size <= buffer.m_size - bufferOffset,
                                  "Buffer descriptor range is outside the buffer");
                }

                // Store the buffer range, descriptor info, and destination range in vectors so all pointer targets
                // stay alive until writeResourceDescriptorsEXT writes the descriptor bytes.
                {
                    // To create a buffer descriptor we just use a Device Address Range.
                    // This allows us to specify the size and offset, but skips having to create a BufferView.
                    vk::DeviceAddressRangeEXT& bufferRange = emplaceNoRealloc(
                        bufferRanges,
                        vk::DeviceAddressRangeEXT{.address = buffer.m_addressGPU + bufferOffset, .size = size});

                    // To create a descriptor we just need to set the type and the buffer range.
                    // Notice that we are storing a reference to the buffer range in the descriptor,
                    // so we need to ensure the buffer range lives until we call writeResourceDescriptorsEXT.
                    vk::ResourceDescriptorInfoEXT& descriptor =
                        emplaceNoRealloc(resources, vk::ResourceDescriptorInfoEXT{
                                                        .type = type,
                                                    });
                    // Small workaround to initialize the union.
                    descriptor.data.pAddressRange = &bufferRange;

                    // Set the host address range where writeResourceDescriptorsEXT will write this descriptor.
                    // For simplicity, the tutorial writes to the final position using the offset in the heap,
                    // but it is possible to write the descriptor to a temporary location and then copy it to the
                    // final offset in the heap. This should be preferable for performance, since the heap buffer
                    // can be made device-local and not host-visible.
                    emplaceNoRealloc(descriptorRanges, vk::HostAddressRangeEXT{
                                                           .address = resourceHeapBase + heapOffset,
                                                           .size = static_cast<std::size_t>(descriptorSize),
                                                       });
                }
            };

            // Add all resource-heap descriptors using the helper above.
            // Note: We have a fixed layout for the resource heap, which makes this easy.
            //       Previously we calculated the offsets for each descriptor in the heap,
            //       and now we just need to write descriptors to those offsets.

            const std::uint32_t frameCount = util::safeCastToU32(m_framesInFlight.size());
            for (std::uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
                // The tutorial has one camera buffer descriptor per frame in flight.
                // We add one descriptor for each camera buffer.

                // Use the aligned stride calculated earlier to allocate the descriptors.
                // This makes it easier to index each camera descriptor in the heap.
                std::uint32_t uboWriteHeapOffset =
                    m_descriptorHeaps.m_cameraOffset + frameIndex * m_descriptorHeaps.m_uniformBufferStride;

                addBufferDescriptor(vk::DescriptorType::eUniformBuffer, m_framesInFlight[frameIndex].m_camera, 0,
                                    sizeof(CameraData), uboWriteHeapOffset, m_descriptorHeaps.m_uniformBufferSize);
            }

            // Storage buffer descriptor for the object data.
            addBufferDescriptor(vk::DescriptorType::eStorageBuffer, m_scene.m_objects, 0, m_scene.m_objects.m_size,
                                m_descriptorHeaps.m_objectsOffset, m_descriptorHeaps.m_storageBufferSize);

            // Uniform buffer descriptor for the point light data.
            addBufferDescriptor(vk::DescriptorType::eUniformBuffer, m_scene.m_pointLight, 0, sizeof(LightData),
                                m_descriptorHeaps.m_pointLightOffset, m_descriptorHeaps.m_uniformBufferSize);

            // Uniform buffer descriptor for the solid color data.
            // This might be better as push data, but for simplicity we use a UBO in this tutorial.
            addBufferDescriptor(vk::DescriptorType::eUniformBuffer, m_scene.m_solidColor, 0, sizeof(SolidColorData),
                                m_descriptorHeaps.m_solidColorOffset, m_descriptorHeaps.m_uniformBufferSize);

            const std::uint32_t textureCount = util::safeCastToU32(m_scene.m_textures.size());
            for (std::uint32_t i = 0; i < textureCount; ++i) {
                // In our tutorial, we are placing all image descriptors at the end of the heap.
                // This is not a requirement, but it makes it easier to index each texture descriptor in the heap
                // using the draw data.
                //
                // Image descriptors are a bit more complicated than buffer descriptors.
                // The descriptor still needs image-view metadata such as format, swizzles, and subresource range.
                // writeResourceDescriptorsEXT takes VkImageViewCreateInfo directly,
                // so descriptors can be created without creating an ImageView object.

                // Descriptor heaps require an ImageViewCreateInfo to create the image descriptor.
                // We reuse the same struct as in Vulkan 1.0 to set the format, mip levels, layers, etc.
                vk::ImageViewCreateInfo& imageViewInfo = emplaceNoRealloc(
                    imageViewInfos, vk::ImageViewCreateInfo{
                                        .image = *m_scene.m_textures[i].m_image,
                                        .viewType = vk::ImageViewType::e2D,
                                        .format = mainTextureFormat,
                                        .components = identityComponentMapping,
                                        .subresourceRange = CreateImageSubresourceRange(
                                            0, vk::ImageAspectFlagBits::eColor, m_scene.m_textures[i].m_mipLevels),
                                    });

                // Image descriptors store the image-view create info and the layout that shaders will sample from.
                // The create infos only need to live until writeResourceDescriptorsEXT copies descriptor bytes below.
                // Notice that with descriptor heaps, we do not need to create a separate ImageView.
                vk::ImageDescriptorInfoEXT& imageInfo =
                    emplaceNoRealloc(imageInfos, vk::ImageDescriptorInfoEXT{
                                                     .pView = &imageViewInfo,
                                                     .layout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                                 });

                // Initialize the descriptor with the type and image info. The descriptor stores a pointer to imageInfo,
                // and imageInfo stores a pointer to imageViewInfo, so both vector elements must remain stable.
                vk::ResourceDescriptorInfoEXT& descriptor =
                    emplaceNoRealloc(resources, vk::ResourceDescriptorInfoEXT{
                                                    .type = vk::DescriptorType::eSampledImage,
                                                });
                // Workaround to initialize the union.
                descriptor.data.pImage = &imageInfo;

                // Set the host address range where this image descriptor will be written.
                //
                // Similar to camera descriptors, we use the aligned stride during allocation to make it easier to index
                // each texture descriptor. Similar to buffers, we could write the descriptor to a temporary location
                // and then copy it to the final offset in the heap. For performance we might employ a staging buffer to
                // keep the heap device-local.
                const std::uint32_t imageWriteHeapOffset =
                    m_descriptorHeaps.m_pushTextureOffset + i * m_descriptorHeaps.m_sampledImageStride;
                emplaceNoRealloc(descriptorRanges,
                                 vk::HostAddressRangeEXT{
                                     .address = resourceHeapBase + imageWriteHeapOffset,
                                     .size = static_cast<std::size_t>(m_descriptorHeaps.m_sampledImageSize),
                                 });
            }
            util::require(resources.size() == descriptorRanges.size(), "Resource info and range size must match");

            util::checkVk(static_cast<VkResult>(m_logicalDevice.writeResourceDescriptorsEXT(
                              util::safeCastToU32(resources.size()), resources.data(), descriptorRanges.data())),
                          "Writing resource heap descriptors");
        }
    }

    // Write sampler heap descriptors.
    {
        util::require(m_descriptorHeaps.m_samplerHeap.m_buffer.m_addressCPU != nullptr,
                      "Sampler descriptor heap must be CPU-mapped before writing descriptors");

        // Samplers live in the sampler heap.
        // Similar to resources, we need to write sampler descriptor bytes into the sampler heap buffer.
        //
        // We have a fixed layout for the sampler heap, which makes this easy.
        // Currently the tutorial only uses two fixed samplers, so we need to write both descriptors into the heap.
        // The combined sampled-image mappings will pair these descriptors to create a combined image sampler.
        // vkWriteSamplerDescriptorsEXT takes VkSamplerCreateInfo directly, so we do not need to create a VkSampler
        // object for this descriptor.

        // Use VkSamplerCreateInfo to describe the sampler descriptor.
        // For teaching purposes the tutorial only uses two samplers.
        // A proper application would create multiple sampler descriptors for different filtering and address modes it
        // needs.
        std::array samplerInfo = {
            // Sampler with linear filtering and repeat address mode, which is what the Sponza albedo textures expect.
            // Linear filtering allows interpolating between mip levels and texels, improving visual quality.
            vk::SamplerCreateInfo{
                .magFilter = vk::Filter::eLinear,
                .minFilter = vk::Filter::eLinear,
                .mipmapMode = vk::SamplerMipmapMode::eLinear,
                .addressModeU = vk::SamplerAddressMode::eRepeat,
                .addressModeV = vk::SamplerAddressMode::eRepeat,
                .addressModeW = vk::SamplerAddressMode::eRepeat,
                .maxLod = VK_LOD_CLAMP_NONE,
            },
            // Sampler with nearest filtering and repeat address mode.
            // We use nearest filtering for normal map textures to avoid interpolation.
            vk::SamplerCreateInfo{
                .magFilter = vk::Filter::eNearest,
                .minFilter = vk::Filter::eNearest,
                .mipmapMode = vk::SamplerMipmapMode::eNearest,
                .addressModeU = vk::SamplerAddressMode::eRepeat,
                .addressModeV = vk::SamplerAddressMode::eRepeat,
                .addressModeW = vk::SamplerAddressMode::eRepeat,
                .maxLod = VK_LOD_CLAMP_NONE,
            }};

        // We use a host address range to specify where writeSamplerDescriptorsEXT will write sampler descriptors.
        // Similar to resources, we are directly writing the descriptor to its final offset in the heap.
        // It might be better to write it to a temporary address and then copy it to the correct offset in the heap.
        // This might allow us to use a staging buffer and make the heap buffer device-local for better performance.
        std::array samplerRange = {
            vk::HostAddressRangeEXT{
                .address = static_cast<std::byte*>(m_descriptorHeaps.m_samplerHeap.m_buffer.m_addressCPU) +
                           m_descriptorHeaps.m_samplerHeap.m_bindOffset + m_descriptorHeaps.m_linearSamplerOffset,
                .size = static_cast<std::size_t>(m_descriptorHeaps.m_samplerSize),
            },
            vk::HostAddressRangeEXT{
                .address = static_cast<std::byte*>(m_descriptorHeaps.m_samplerHeap.m_buffer.m_addressCPU) +
                           m_descriptorHeaps.m_samplerHeap.m_bindOffset + m_descriptorHeaps.m_nearestSamplerOffset,
                .size = static_cast<std::size_t>(m_descriptorHeaps.m_samplerSize),
            }};

        util::require(samplerInfo.size() == samplerRange.size(), "Sampler info and range size must match");
        util::checkVk(static_cast<VkResult>(m_logicalDevice.writeSamplerDescriptorsEXT(
                          util::safeCastToU32(samplerInfo.size()), samplerInfo.data(), samplerRange.data())),
                      "Writing sampler heap descriptor");
    }
}

std::vector<vk::DescriptorSetAndBindingMappingEXT> Application::buildShaderDescriptorMappings(
    const std::unordered_map<std::string, util::slang::ShaderResourceBinding>& shaderResourceBindings) const
{
    // In Vulkan, descriptors describe resources such as buffers and images so that shaders can access them.
    // Shaders usually decorate descriptor variables with set and binding numbers, IE [[vk::binding(bindingId, setId)]].
    // In Vulkan, descriptors are grouped into descriptor sets, and each slot in a descriptor set is called a binding.
    //
    // Descriptor sets are mostly a legacy concept from Vulkan 1.0, where descriptors were grouped into sets.
    // The idea was to group descriptors that are used or updated together into the same set, allowing applications to
    // minimize descriptor writes and updates.
    //
    // Descriptor sets require developers to think about how descriptors should be grouped, and they also introduce a
    // lot of boilerplate code for creating descriptor set layouts, allocating descriptor sets, and updating them.
    //
    // This tutorial uses descriptor heaps and shader objects. Descriptor heaps allow shaders to access descriptors
    // directly from a GPU buffer. This avoids the need to decide how to group descriptors into descriptor sets.
    //
    // With descriptor heaps, we have a GPU buffer that contains descriptors, but we still need a way to map shader
    // variables to the correct offsets in the descriptor heap buffer. A common way to do this is by using descriptor
    // mappings.
    //
    // When using mappings with descriptor heaps, shaders still use the old syntax to declare set and binding numbers
    // for their descriptors: [[vk::binding(bindingId, setId)]]. This allows existing shaders to work with descriptor
    // heaps without changes to the shader code, making it easy to migrate existing shaders to descriptor heaps.
    //
    // There are multiple ways to create descriptor mappings. This tutorial uses constant offsets and push indices.
    // We recommend that developers also read:
    // https://docs.vulkan.org/guide/latest/descriptor_heap.html#_mapping_the_heap_to_existing_shaders
    //
    // Descriptor heaps offer multiple ways to map shader variables, for example:
    // Descriptor heaps also offer push data as a replacement for push constants, and provide a simple way to specify
    // per-draw data for the shader. Similar inline and GPU-address-based data is also supported by the extension.
    //
    // Note: With descriptor heaps, mappings are completely optional. Shaders do not need to specify set and binding
    // numbers for their descriptors, because they can access the heap directly. This is commonly referred to as using
    // untyped pointers. It can be very flexible, but it can also be error-prone and complex.
    //
    // Note: Slang allows developers to avoid manually specifying set and binding numbers for descriptors, and can
    // automatically deduce valid bindings. Manually setting set and binding numbers is error-prone, so it is
    // recommended to deduce them automatically from Slang reflection.

    // This code will create a vector of vk::DescriptorSetAndBindingMappingEXT, which will be used to map shader
    // variables to the corresponding descriptor by selecting the correct offsets in the descriptor heap buffer.
    // This vector will be passed when creating the shader objects.
    // GPU compilers need to access the offset of each descriptor in the heap, so the offset needs to be given during
    // shader compilation.
    //
    // Most applications create shader bindings using reflection data from the shader compiler.
    // This tutorial uses Slang reflection to get the set and binding. The main idea is:
    // - Iterate over the reflection data and collect all descriptors in the shaders.
    // - Collect the set and binding for each descriptor using descriptor reflection data.
    // - Create a new binding mapping for each descriptor.
    //
    // Our tutorial uses the shader variable name to select hardcoded offsets and types to identify the descriptor.
    // In this tutorial, we are using a fixed layout for the descriptor heap, so we can use the offsets calculated
    // previously. This is not very flexible, but it is simple and works for this tutorial.
    // Most applications will have a more flexible layout for the descriptor heap, and will need to calculate and
    // store the offsets for each descriptor.
    //
    // Our application has two shader groups, and each shader group has its own set of descriptors.
    // Therefore, we create a separate mapping table for each shader group.

    // Store mappings for this group of shaders.
    // Each shader group will have its own mapping table, which will be passed when creating the shader object.
    std::vector<vk::DescriptorSetAndBindingMappingEXT> mappings;
    // Small optimization to reserve enough space in the vector to avoid reallocations.
    mappings.reserve(shaderResourceBindings.size());

    // Used for validation to ensure that shaders do not have duplicated shader variable names or set/binding numbers.
    std::vector<std::string> validateBindingNames;
    std::vector<std::pair<uint32_t, uint32_t>> validateBindingId;

    // Helper function to create a descriptor mapping.
    const auto makeDescriptorHeapMapping = [](const util::slang::ShaderResourceBinding& binding,
                                              vk::DescriptorMappingSourceEXT source,
                                              const vk::DescriptorMappingSourceDataEXT& sourceData) {
        // This helper function uses Slang reflection data to create a descriptor mapping.
        // The main idea is to select a descriptor for a shader variable by linking it to a specific offset in the
        // descriptor heap that we set in sourceData.

        return vk::DescriptorSetAndBindingMappingEXT{
            // Descriptor set and binding obtained from Slang [[vk::binding(bindingId, setId)]].
            // We use Slang reflection to reduce boilerplate and hardcoded values for the binding and set numbers.
            .descriptorSet = binding.set,
            .firstBinding = binding.binding,
            .bindingCount = 1,
            .resourceMask = vk::SpirvResourceTypeFlagsEXT{binding.resourceMask},
            .source = source,
            .sourceData = sourceData,
        };
    };

    // This helper calculates descriptor offsets for images using push indices with descriptor heaps.
    // It creates the source data for a combined sampled image.
    const auto makeCombinedImageSamplerPushIndexSourceData = [pushTextureOffset = m_descriptorHeaps.m_pushTextureOffset,
                                                              sampledImageStride =
                                                                  m_descriptorHeaps.m_sampledImageStride,
                                                              samplerOffset = m_descriptorHeaps.m_linearSamplerOffset,
                                                              samplerStride = m_descriptorHeaps.m_nearestSamplerOffset -
                                                                              m_descriptorHeaps.m_linearSamplerOffset](
                                                                 std::uint32_t pushOffset) {
        // For combined sampled images, we use push indices.
        // When using push-index mappings, the descriptor for each image is selected using data pushed to the GPU.
        // This makes it easy to select the correct descriptor in the heap per-draw call, allowing each draw call to
        // select a different texture per object. The final descriptor offset is calculated using:
        // GPU descriptor address = resource heap start + mapping heap offset + pushed index * mapping heap stride.
        // Note: we omit shader index since the tutorial only uses single-binding scalar mappings.
        //
        // Our application shaders declare a combined sampled image descriptor.
        // This binds both the image (from the resource heap) and the sampler (from the sampler heap) into a
        // single binding slot since they are a single shader variable.
        // We can edit the shader to have separate image and sampler descriptors.

        util::require((pushOffset % 4) == 0, std::format("Push offsets {} must be aligned to 4 bytes", pushOffset));

        return vk::DescriptorMappingSourceDataEXT{vk::DescriptorMappingSourcePushIndexEXT{
            // heapOffset is the constant offset added for byte calculations.
            // Usually, it points to the first descriptor we can bind.
            // In our case, we have a fixed layout for the heap, and all image descriptors are placed at the end of the
            // heap, so it is the offset of the first image descriptor.
            .heapOffset = pushTextureOffset,
            // pushOffset is the offset of the pushed index in the push data.
            // In our tutorial, we are using push-data texture indices to select the correct image descriptor in the
            // heap, so that is our push index, which pushOffset points to.
            // pushIndex = *(&PushData + pushOffset).
            .pushOffset = pushOffset,
            // heap index stride is the stride between each image descriptor in the heap.
            // This is the number of bytes added per push index increment.
            // Our stride is the aligned size of an image descriptor. This is common for use cases.
            // Our application is placing images using a fixed stride to make them easier to index.
            .heapIndexStride = sampledImageStride,
            // Heap array stride is the stride between each array of image descriptors in the heap.
            // Our mapped descriptors are single binding scalars, so our array index is always 0 and this is not
            // relevant for our tutorial. This is usually set to the aligned size of an image descriptor.
            // We can use a separate stride depending on the use case and layout of the heap.
            .heapArrayStride = 0,
            // We could specify a sampler create info in the mapping to embed a sampler in the mapping.
            // Our tutorial uses a sampler from the descriptor heap.
            .pEmbeddedSampler = nullptr,
            // Combined image sampler mappings can be used to pack the sampler and resource index in a single uint32_t.
            // If true, the combined index is set using pushOffset and ignores samplerPushOffset.
            // combinedPushIndex = *(&PushData + pushOffset) = ((samplerIndex&0xFFF) << 20) | (resourceIndex&0xFFFFF)
            .useCombinedImageSamplerIndex = vk::True,
            // Sampler descriptor offset is calculated the same way as the image descriptor offset,
            // but it is used to select the sampler descriptor in the sampler heap.
            // Note: Our application is ignoring samplerPushOffset,
            //       because we are combining the sampler and image indices.
            //       This reduces the data we send to the GPU.
            // Note: We could use samplerPushOffset to use different indices for the sampler and the image.
            // Note: These values refer to the sampler heap offset, not the resource heap offsets.
            .samplerHeapOffset = samplerOffset,
            .samplerPushOffset = 0, // Unused since we are using useCombinedImageSamplerIndex.
            .samplerHeapIndexStride = samplerStride,
            .samplerHeapArrayStride = 0, // Unused since we only have single-binding scalar descriptors.
        }};
    };

    // We use Slang reflection to iterate through all bindings in the shader and create a mapping for each one.
    for (const auto& bindingIt : shaderResourceBindings) {

        // Validation code.
        {
            // Check to ensure that shaders do not have duplicate set/binding names or set/binding numbers.
            // This is not required, but it helps catch mistakes in shader code.

            auto nameIt = std::find(validateBindingNames.begin(), validateBindingNames.end(), bindingIt.first);
            util::require(nameIt == validateBindingNames.end(),
                          std::format("duplicate shader resource binding name: {}", bindingIt.first));
            validateBindingNames.push_back(bindingIt.first);

            auto idPair = std::make_pair(bindingIt.second.set, bindingIt.second.binding);
            auto idIt = std::find(validateBindingId.begin(), validateBindingId.end(), idPair);
            util::require(idIt == validateBindingId.end(),
                          std::format("duplicate shader resource binding set: {} binding {}", bindingIt.second.set,
                                      bindingIt.second.binding));
            validateBindingId.push_back(idPair);
        }

        // The tutorial uses a fixed layout with few descriptors.
        // We have hardcoded mappings for each descriptor in the shader,
        // so we use the variable name to identify the descriptor and select the correct offset in the heap.

        // Create mapping for camera using push indices.
        if (bindingIt.first == "g_camera") {
            // We are using push indices so the logic is similar to our texture image mapping.
            const vk::DescriptorMappingSourceDataEXT cameraSourceData{vk::DescriptorMappingSourcePushIndexEXT{
                // This points to the first camera descriptor in the heap.
                .heapOffset = m_descriptorHeaps.m_cameraOffset,
                // We will select the camera buffer using the current frame in flight in the push data.
                .pushOffset = util::safeCastToU32(offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, cameraIndex)),
                .heapIndexStride = m_descriptorHeaps.m_uniformBufferStride,
            }};

            static_assert(
                offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, cameraIndex) ==
                    offsetof(DescriptorHeapDrawPushIndicesAlbedo, cameraIndex) &&
                offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, cameraIndex) ==
                    offsetof(DescriptorHeapDrawPushIndicesSolidColor, cameraIndex) &&
                "Camera index must be at the same offset in all push data structs. This makes detecting offsets in "
                "the tutorial simpler");

            mappings.push_back(makeDescriptorHeapMapping(
                bindingIt.second, vk::DescriptorMappingSourceEXT::eHeapWithPushIndex, cameraSourceData));
        }
        // Create mapping for object buffer.
        else if (bindingIt.first == "g_objects") {
            // We have a single object buffer descriptor in the heap.
            // We can use constant offset mappings to select the correct descriptor in the heap.
            const vk::DescriptorMappingSourceDataEXT objectsSourceData{vk::DescriptorMappingSourceConstantOffsetEXT{
                .heapOffset = m_descriptorHeaps.m_objectsOffset,
            }};
            mappings.push_back(makeDescriptorHeapMapping(
                bindingIt.second, vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset, objectsSourceData));
        }
        // Create mapping for point light buffer.
        else if (bindingIt.first == "g_pointLight") {
            // We also have a single point light buffer descriptor, so we can use constant offset mappings.
            const vk::DescriptorMappingSourceDataEXT pointLightSourceData{vk::DescriptorMappingSourceConstantOffsetEXT{
                .heapOffset = m_descriptorHeaps.m_pointLightOffset,
            }};
            mappings.push_back(makeDescriptorHeapMapping(
                bindingIt.second, vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset, pointLightSourceData));
        }
        // Create mapping for solid color buffer if needed.
        else if (bindingIt.first == "g_solidColor") {
            // We also have a single solid color buffer descriptor, so we can use constant offset mappings.
            // Solid color might be better as push data, but for simplicity we use a UBO in this tutorial.
            const vk::DescriptorMappingSourceDataEXT solidColorSourceData{vk::DescriptorMappingSourceConstantOffsetEXT{
                .heapOffset = m_descriptorHeaps.m_solidColorOffset,
            }};
            mappings.push_back(makeDescriptorHeapMapping(
                bindingIt.second, vk::DescriptorMappingSourceEXT::eHeapWithConstantOffset, solidColorSourceData));
        }
        // Create mapping for albedo texture if needed.
        else if (bindingIt.first == "g_albedoTexture") {

            // We select the offset of the albedo texture index.
            // Notice that we are using different indices for each texture.
            static_assert(offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, albedoTextureIndex) ==
                              offsetof(DescriptorHeapDrawPushIndicesAlbedo, albedoTextureIndex),
                          "Albedo texture index must be at the same offset in all albedo push data structs");
            const vk::DescriptorMappingSourceDataEXT albedoTextureSourceData =
                makeCombinedImageSamplerPushIndexSourceData(
                    util::safeCastToU32(offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, albedoTextureIndex)));

            // We use our helper function to create the mapping for textures.
            mappings.push_back(makeDescriptorHeapMapping(
                bindingIt.second, vk::DescriptorMappingSourceEXT::eHeapWithPushIndex, albedoTextureSourceData));
        }

        // Create mapping for normal texture if needed.
        else {
            // We select the offset of the normal texture index.
            // Notice that we are using different indices for each texture.
            util::require(bindingIt.first == "g_normalTexture",
                          std::format("Unexpected shader resource binding name: {}", bindingIt.first));
            const vk::DescriptorMappingSourceDataEXT normalTextureSourceData =
                makeCombinedImageSamplerPushIndexSourceData(
                    util::safeCastToU32(offsetof(DescriptorHeapDrawPushIndicesAlbedoAndNormal, normalTextureIndex)));

            // We use our helper function to create the mapping for textures.
            mappings.push_back(makeDescriptorHeapMapping(
                bindingIt.second, vk::DescriptorMappingSourceEXT::eHeapWithPushIndex, normalTextureSourceData));
        }
    }

    return mappings;
}

void Application::calculateVertexInputs()
{

    // In Vulkan, geometry data is passed to the vertex shader using vertex buffers.
    // Vertex buffers are GPU buffers that contain packed vertex data.
    // They store vertex attributes, such as position, normal, and texture coordinates.
    //
    // Vulkan allows control over how to specify the layout of vertex data in the buffer,
    // and how to map this vertex input data to the vertex shader input variables.
    // This allows applications to optimize how shaders receive vertex data.
    //
    // For example, this application uses a single vertex buffer, so all vertex data is packed into one
    // single struct stored in the same vertex buffer. This layout is usually called an interleaved vertex buffer.
    // Alternatively, an application can use multiple vertex buffers, with each attribute or group of attributes
    // coming from a separate buffer / binding. This is commonly called separate or multi-stream vertex input.
    //
    // The best layout for your vertex data depends on the GPU and the use case.
    // Some GPUs benefit from using separate buffers for positional and non-positional data;
    // check the recommendations from your GPU vendor about vertex buffers.
    //
    // It is important to note that most meshes reference the same vertices for multiple triangles,
    // so for good performance you should also use an index buffer to avoid duplicating vertex data.

    util::log_msg("[Init] Calculate Vertex Inputs");

    const std::filesystem::path reflectionPath{
        std::filesystem::path{SIGGRAPH_SHADER_DIR} /
        std::format("{}.json", SceneData::albedoAndNormalShaderFilePair.vertexName)};

    // Our tutorial uses Slang reflection data, which tells us which locations/formats the shader expects for the
    // vertex attributes.
    // We have a very simple vertex data layout. We bind a single buffer for all vertex attributes, and we pack all
    // vertex data in a single struct, so calculating the vertex input data is trivial. This function just:
    //  - Gets the location of the shader input variables from Slang
    //  - Uses hardcoded names to link them to offsetof(PackedVertex, attribute) to get the offset in the buffer.
    //  - We always set binding = 0 since we are using a single buffer for all vertex data.
    //  - Sets the format of the attribute using Slang reflection data.
    const util::slang::PackedVertexInputLayout layout = util::slang::calculatePackedVertexInputLayout(reflectionPath);

    // Vertex input binding descriptions reference the buffers we bound with the vertex data they contain.
    // .binding = Binding slot calculated using BindVertexBuffers2 first binding + buffer index
    //            This will allow us to use multiple vertex buffers.
    //            For example some GPUs benefit from using separate buffers for positional and non-positional data.
    //            This is a simple tutorial. We use a single buffer for all vertex data, and we always use binding = 0.
    // .stride = Distance between the start of two consecutive vertices in the buffer.
    //           We have all vertex data packed in a single struct, so the stride is sizeof(PackedVertex).
    //           When using dynamic input stride this can be modified by BindVertexBuffers2 at runtime.
    // .inputRate = We are using vertex rate rather than instance rate.
    m_vertexInput.m_vertexBindings = layout.bindings;

    // Vertex input attribute descriptions contain the mapping from vertex buffer data to shader input locations.
    // .location = The location set by [[vk::location(n)]] in the shader.
    //             This identifies the shader variable that will receive the vertex data.
    // .binding = binding slot set by BindVertexBuffers2.
    //            If we had multiple vertex buffers, this would allow us to select which buffer to read the data from.
    // .offset = This identifies which data in the GPU buffer will be read for this attribute.
    //           It is the offset in the vertex buffer to read the data.
    //           In our tutorial, the data is packed in a struct so it is a simple offsetof(PackedVertex, attribute).
    // Shader variable at ([[vk::location(locationId)]], for vertexIndex=vertexIndex) =
    //     (Bound buffer at binding=bindingId) + (bound offset) + attribute offset + bufferBinding.stride * vertexIndex
    // Note: we are using vertex rate rather than instance rate.
    // Note: We assume BindVertexBuffers2 binds the vertex buffer with offset=0.
    m_vertexInput.m_vertexAttributes = layout.attributes;
}

void Application::initShaderObjects()
{

    // This function will read our shaders and create shader objects.
    // Shaders are programs that run on the GPU. They are the core way for an application to control the GPU.
    // GPUs tend to have a SIMD architecture, so shaders are designed to run in parallel on many threads.
    // There are multiple ways to write shaders, but the most common way is to use a shading language.
    // Traditional languages such as GLSL or HLSL are relatively popular. This tutorial will use Slang.
    // Slang is a modern shading language hosted by Khronos, designed to modernize shader programming.
    //
    // Shader objects are the new way for Vulkan to interact with shaders. Similar to pipelines in Vulkan 1.0, they
    // provide a way to bind shaders to the GPU, but they are more flexible and easier to use than pipelines.

    util::log_msg("[Init] Creating shader objects");

    // We have three pairs of shaders, which is equivalent to three pipelines.
    // The current function will create three shader groups, one for each pair of shaders.
    //
    // The objective of the tutorial is to showcase how an application can handle multiple shader groups with shader
    // objects. Each draw call will store a reference to the shader objects it uses, which will be bound at run time if
    // necessary. This is a basic strategy, focused on fundamentals rather than performance and scalability.
    m_shaderObjects.m_albedoAndNormal = createShaderGroup(SceneData::albedoAndNormalShaderFilePair);
    m_shaderObjects.m_albedo = createShaderGroup(SceneData::albedoShaderFilePair);
    m_shaderObjects.m_solidColor = createShaderGroup(SceneData::solidColorShaderFilePair);
}

Application::ShaderGroup Application::createShaderGroup(const ShaderFilePair& shaderFilePair)
{
    // This tutorial uses shader objects to manage its pipelines / shader groups.
    //
    // Vulkan 1.0 is built around the idea of pipelines (`VkPipeline`).
    // When we create a pipeline, we need to specify the code for all shader stages and most graphics state upfront,
    // then the driver compiles them into a single object. Pipelines have proved problematic for many applications:
    // - Requiring all state during initialization is not very flexible, causing a huge number of PSOs.
    // - Pipeline caching has proved insufficient for most big applications.
    // - Creating and using `VkPipelines` requires a lot of boilerplate in Vulkan.
    //
    // This tutorial uses shader objects.
    // `EXT_shader_object` provides a more flexible and forward-looking way to use shaders in Vulkan.
    //  - Vertex and fragment shaders can be created independently, and combined during command buffer recording.
    //  - We use dynamic state to specify graphics state.
    //    The shader object extension requires applications to specify the required dynamic state before rendering.
    //    This reduces PSOs since we do not need to specify it at initialization.
    //  - The API is more direct and reduces the amount of boilerplate needed to create and bind shaders.
    //  - It enables advanced use cases, by giving applications direct access to shader binaries for internal caching,
    //    and control over shader linking.
    //  - It simplifies the API and reduces boilerplate.
    //
    // A pipeline state object (PSO) is an informal term for the data that defines a pipeline.
    // This includes all shaders used (vertex and fragment), fixed-function state, descriptor compatibility, etc.
    //
    // Pipelines describe the state, shaders, etc. that control operations on the GPU.
    // This tutorial uses vertex and fragment shaders, which are part of the graphics pipeline.
    // Beginners should understand the different parts of a graphics pipeline, including the fixed-function stages.
    // Note: Geometry and tessellation shaders are not commonly used and can be ignored by most developers.
    //
    // Beginners should also be aware of other pipeline types, especially compute pipelines, since
    // compute shaders are extremely popular for GPGPU and are becoming common in rendering use cases.
    // Note: With shader objects, applications use a different abstraction and no longer create pipeline objects.
    //       Pipelines still exist as a concept, even if we no longer create a Vulkan object for them.
    //
    //  With shader objects, we can create a shader group by creating the vertex and fragment shader objects
    //  independently. This significantly reduces boilerplate because we can avoid layouts and intermediate objects and
    //  directly use SPIR-V to create our shaders. We can then bind those shaders during command buffer recording.
    //  Note: Currently the tutorial has linking enabled, so shaders are not created independently.
    //
    //  Most Vulkan tutorials would create a graphics pipeline here.
    //       This tutorial relies on forward-looking strategies like shader objects and dynamic state instead.

    // Path to read the SPIR-V and reflection data, generated during compilation.
    const std::filesystem::path shaderDir = SIGGRAPH_SHADER_DIR;

    // Contains mappings from shader set/binding declarations to descriptor-heap mappings.
    // The main idea is to link binding data, specified in Slang by `[[vk::binding(bindingId, setId)]]`,
    // to descriptor heap offsets that select the correct descriptor.
    std::vector<vk::DescriptorSetAndBindingMappingEXT> mappings;
    {
        // We need to map shader resources to Vulkan descriptors.
        // This lets the application tell Vulkan which resources a shader will use for each shader variable.
        // We are using descriptor heaps, so we do not create descriptor sets or related layout objects.
        // To identify shader variables, shaders can still use classic set/binding declarations,
        // such as `[[vk::binding(0, 0)]] g_camera`. During shader creation, we attach a mapping that links those
        // bindings to offsets inside the bound heap buffers. That is why this sample does not create descriptor set
        // layouts, descriptor sets, or a pipeline layout.
        // Note: Descriptor heaps are compatible with untyped pointers. This could allow applications to use an untyped
        //       shader model where shaders directly index descriptors in heap memory.
        //       This is more flexible and skips mapping creation, but it is more complex and advanced.
        //       This tutorial ignores that path and uses classic set/binding decorations instead.

        // We use reflection data to generate our shader bindings.
        // Note: We are reading the reflection data from a JSON file at runtime.
        //       Normal applications would likely want to read this data at build time, usually using a script to
        //       generate C++ code with the mappings baked in. Most engines will also generate shader-specific structs
        //       like PackedVertex and the descriptor-heap push data structs from this reflection data at build time.
        const std::array shaderReflectionPaths{
            shaderDir / std::format("{}.json", shaderFilePair.vertexName),
            shaderDir / std::format("{}.json", shaderFilePair.fragmentName),
        };

        // This function will read the reflection data from the JSON files and collect all shader resource bindings.
        // It will ensure that bindings are compatible between shaders, and that there are no duplicate bindings.
        const std::unordered_map<std::string, util::slang::ShaderResourceBinding>& shaderResourceBindings =
            util::slang::collectShaderResourceBindings(shaderReflectionPaths);

        // Main function to create all mapping information.
        mappings = buildShaderDescriptorMappings(shaderResourceBindings);
    }

    // When creating shader objects, we use this to link shader bindings to descriptor heap mappings.
    // Note: Mappings need to be set at shader creation time,
    //       since the GPU compiler needs to know the offsets of each descriptor in the heap.
    // Note: We always create mappings, but they are ignored when using shader binaries rather than SPIR-V.
    const vk::ShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{
        .mappingCount = util::safeCastToU32(mappings.size()),
        .pMappings = mappings.data(),
    };

    // We need to use shader flags to enable descriptor heap and other options.
    vk::ShaderCreateFlagsEXT shaderFlags = vk::ShaderCreateFlagBitsEXT::eDescriptorHeap;

    // Shader Object Linking:
    //
    // With shader objects, we control linking: shader objects can be created independently or linked together.
    // Disabling linking lets us create each shader object independently. This can reduce the number of shader objects
    // we need to create because it lets us reuse individual vertex or fragment shaders in different pairs, reducing
    // the number of shader groups / PSOs. However, disabling linking prevents the driver from optimizing the
    // vertex/fragment pair together, which can significantly reduce GPU performance.
    //
    // There is a trade-off between enabling shader object linking to improve GPU performance,
    // and reusing shaders across different pipelines to reduce PSOs.
    // Advanced applications can combine both approaches. First, create the unlinked shaders. Individual shaders are
    // usually reused across multiple shader pairs, so they are likely to already be compiled, created, and ready for
    // use. Then re-create the shader-pair objects with linking enabled for better GPU performance.
    //
    // Our tutorial has the variable enable_linking to control linking.
    // We recommend that most applications enable linking for better GPU performance,
    // since PSO creation usually becomes a problem mainly in larger applications with many shader combinations.
    //
    // Note: If we enable linking, the shaders in the shader pair have to be used together.
    //       When we enable the linking flag, all shaders created in the same call must be bound in the same command
    //       buffer before being used, so applications are not allowed to use the vertex shader with a different
    //       fragment shader, or vice versa. We can still use multiple bindShadersEXT calls to bind the shaders.
    // Note: If linking is enabled, we need to send the pair of shaders to the driver at shader creation time.
    //       This enables link-time optimization (LTO) across both shaders to improve GPU performance.
    constexpr bool enable_linking = true;
    if constexpr (enable_linking) {
        shaderFlags |= vk::ShaderCreateFlagBitsEXT::eLinkStage;
    }

    constexpr std::size_t numShadersInPipeline = 2; // Vertex + fragment.

    // Store shader code until shader objects are created. vk::ShaderCreateInfoEXT points into these vectors.
    //
    // BinaryBuffer is just a wrapper around a byte vector.
    // It adds an offset to an internal vector to handle alignment requirements.
    // Note: In Vulkan, code used to create shader objects has alignment requirements.
    std::vector<util::BinaryBuffer> currentShadersCode;
    std::vector<vk::ShaderCreateInfoEXT> createInfos;

    // Small optimization: reserve space for the shader code and ShaderCreateInfoEXT vectors to avoid reallocations.
    //
    // ShaderCreateInfoEXT stores pointers into the inner byte buffers.
    // These are pointers to BinaryBuffer::storage, which is a vector.
    // Reallocating currentShadersCode can move BinaryBuffer objects, but moving a std::vector does not move its
    // contents, so pointers into the inner buffers remain valid after the outer vector reallocates.
    currentShadersCode.reserve(numShadersInPipeline);
    createInfos.reserve(numShadersInPipeline);

    // Read SPIR-V code from disk.
    // This helper reads binary SPIR-V data from disk and stores it in a byte vector.
    // It checks that the SPIR-V file exists and that its size is a multiple of 4 bytes.
    // The return byte vector has the alignment required to create the shader objects.
    const std::filesystem::path vertexSpirvPath = shaderDir / shaderFilePair.vertexName;
    const std::filesystem::path fragmentSpirvPath = shaderDir / shaderFilePair.fragmentName;
    util::BinaryBuffer vertexSpirvWords = util::readSpirvFile(vertexSpirvPath);
    util::BinaryBuffer fragmentSpirvWords = util::readSpirvFile(fragmentSpirvPath);

    //***********
    // Shader Binary Cache data start
    //***********

    // The tutorial implements a simple shader binary cache to store compiled shader binaries on disk.
    // This is an advanced option and can be skipped on first read.
    //
    // Vulkan has a pipeline cache object to help reuse pipeline compilation work.
    // This has proven insufficient for advanced use cases, since applications still have to manage their own pipeline
    // cache data. PipelineCache can be serialized and reused, but it is opaque, so it is difficult for applications
    // with large shader sets to manage, since it might require caching on top of PipelineCache.
    // Shader objects provide a more flexible way to cache shaders, allowing direct access to the shader binaries.
    //
    // The importance of a proper shader binary cache greatly depends on the platform. Some platforms have a
    // driver-internal or cloud shader cache that reduces the impact of an application shader cache.
    // However, on some platforms and applications, a shader cache can be critical for fast app initialization.
    //
    // It is also important to avoid recreating the same shaders multiple times.
    // Once an application reads a shader or shader pair and creates shader objects,
    // it should avoid recreating the same shader objects.
    constexpr bool enable_shader_binary_cache = true;

    // Path to store and read the optional binary shader cache.
    const std::filesystem::path shaderBinaryCacheDir = shaderDir / "shaderBinaryCache";

    // Wrapper for data we need to update in our cache.
    struct ShaderCacheMiss {
        std::filesystem::path cachePath;
        std::uint32_t shaderIndex;
    };

    // Store shaders that miss the binary cache so their binaries can be saved after creating a shader pair.
    // The list of cacheMisses has one entry for each shader we cannot find in the cache. After creating the shader
    // objects, we use this list to write all missing shader binaries to the cache.
    std::vector<ShaderCacheMiss> cacheMisses;

    // Key used to identify shader binaries in the cache.
    // Each entry in the cache will have a unique key, so this variable needs to contain all relevant data.
    std::uint64_t shaderPairCacheKey = 0;

    if (enable_shader_binary_cache) {

        // Keep an internal version number for cache keys and file formats.
        // If we ever change how our cache works and need to invalidate keys, we should update this value.
        // This will let us invalidate our current cache from the application side.
        // Note: keeping an internal version is a good programming best practice.
        constexpr std::uint32_t shaderBinaryCacheVersion = 1;

        // Calculate a hash for the shader binary cache.
        // Our key is a hash that uses all relevant data for the cache entry.
        // The final key will include mapping information, SPIR-V, linked shader and device properties.
        // Note: This part of the key is the same for all groups,
        //       we could move it to device initialization but we recalculate it here for simplicity.
        shaderPairCacheKey = util::combineHash(
            // Shader-object binaries are driver-specific.
            // The pair key hashes the shader binary UUID, version, etc. reported by the selected physical device.
            shaderBinaryCacheVersion, std::span<const std::uint8_t>{m_shaderBinaryCacheProperties.m_shaderBinaryUUID},
            m_shaderBinaryCacheProperties.m_shaderBinaryVersion);

        if (enable_linking) {
            // When creating linked shader objects, both shaders affect the compiled shader binary of each stage,
            // so the code for both shaders must be part of the key.
            // Note: For simplicity, the tutorial hashes the SPIR-V at runtime for each shader,
            //       but a real application would probably use a pre-calculated hash.
            shaderPairCacheKey = util::combineHash(vertexSpirvWords.as_byte_span(), fragmentSpirvWords.as_byte_span(),
                                                   shaderPairCacheKey);
        }

        // Add shader mapping data to the cache key.
        {
            // Descriptor heap mappings are part of shader-object creation, so they should be included in the key.
            // We are storing the values and device limits that affect how the offsets in the heap are calculated.
            // Layers and debugging tools can change device limits, so missing one of these values could make the cache
            // reuse a binary that was compiled with different descriptor heap assumptions.
            // We would need to update the shaderBinaryCacheVersion if we change how our mappings are created.
            // This is a relatively naive implementation. We should include all values and decisions that affect
            // buildShaderDescriptorMappings().
            shaderPairCacheKey =
                util::combineHash(shaderPairCacheKey, m_descriptorHeaps.m_resourceHeap.m_minReservedRange,
                                  m_descriptorHeaps.m_samplerHeap.m_minReservedRange, m_descriptorHeaps.m_cameraOffset,
                                  m_descriptorHeaps.m_uniformBufferStride, m_descriptorHeaps.m_objectsOffset,
                                  m_descriptorHeaps.m_pointLightOffset, m_descriptorHeaps.m_solidColorOffset,
                                  m_descriptorHeaps.m_pushTextureOffset, m_descriptorHeaps.m_sampledImageStride,
                                  m_descriptorHeaps.m_nearestSamplerOffset, m_descriptorHeaps.m_linearSamplerOffset);
        }
    }

    //***********
    // Shader Binary Cache data end
    //***********

    // Helper to initialize and add each stage shader create info.
    const auto addShaderCreateInfo = [&cacheMisses, &createInfos, &currentShadersCode, &mappingInfo,
                                      shaderBinaryCacheDir, shaderFlags, shaderPairCacheKey](
                                         std::string_view shaderName,
                                         // Note: shaderName is only used for logging purposes.
                                         vk::ShaderStageFlagBits stage, std::span<const std::byte> spirvBytes) {
        static_cast<void>(shaderPairCacheKey);
        std::optional<std::span<const std::byte>> shaderCode;

        vk::ShaderCodeTypeEXT codeType = vk::ShaderCodeTypeEXT::eSpirv;

        // Skipped when shader cache is disabled.
        // Check if disk has a binary for this shader and we can skip compilation.
        if (enable_shader_binary_cache) {

            // Generate the part of the binary cache key that identifies this shader.
            //
            // All common data for the pipeline was already stored in shaderPairCacheKey.
            // Now we only need to identify the shader in the pipeline.
            // If linking is disabled we store the hash of the SPIR-V bytes.
            // If linking is enabled, all shaders affect the output, so their SPIR-V was added during initialization,
            // and we only need to add the stage to the hash.
            uint64_t shaderBinaryCacheKey = enable_linking ? util::combineHash(stage, shaderPairCacheKey)
                                                           : util::combineHash(spirvBytes, shaderPairCacheKey);

            // We are using the cache key as the name of a binary file containing the shader binary.
            const std::filesystem::path shaderCachePath =
                shaderBinaryCacheDir / std::format("{:016x}.bin", shaderBinaryCacheKey);

            // Check whether a non-empty cache file exists and read it. This tutorial trusts the cache file once found.
            // We assume that if a cache file is read successfully, it contains the correct shader binary we want.
            //
            // This trusts the cache completely, which is not production-ready.
            // Production code should handle corrupt files and concurrent readers/writers more defensively.
            //
            // Driver updates can invalidate the cache. We include driver properties like the UUID
            // in the cache key to account for this, but this still depends on the cache key being complete.
            // Note: This is a tutorial. For simplicity, we have not implemented a way to clean up the cache.
            //       Our cache will grow with more shaders and driver updates, a proper application would need a way to
            //       clean up invalid and old entries.

            // Vulkan requires shader binary buffers used to create shader objects to have this byte alignment.
            std::size_t shaderBinaryAlignment = 16;

            // This wrapper will check if the file exists and read it.
            // It returns an aligned vector and checks that the file is not empty.
            std::optional<util::BinaryBuffer> cachedShaderBinary =
                util::readBinaryFile(shaderCachePath, shaderBinaryAlignment);

            const bool hasShaderBinary = cachedShaderBinary.has_value();

            // If a non-empty cache file was read, we have the shader binary in our cache.
            if (hasShaderBinary) {
                util::log_msg("[Init] Reading Cached Shader Binary: {}", shaderCachePath.string());

                // We can use the binary read from the file to create the shader object.
                // This shader binary data needs to live until shader creation, so we store it in a vector.
                // Note: The span points into an inner vector.
                //       Moving a std::vector does not invalidate references to its contents.
                //       So we can safely reallocate the outer vector.
                shaderCode = currentShadersCode.emplace_back(std::move(cachedShaderBinary.value())).as_byte_span();

                // If we found an entry in the cache, we will use the binary,
                // so we mark the current shader as binary rather than SPIR-V.
                codeType = vk::ShaderCodeTypeEXT::eBinary;
            }
            else {
                // If the shader is not in our cache, record the miss so it can be saved after shader creation.
                cacheMisses.push_back(ShaderCacheMiss{
                    .cachePath = shaderCachePath,
                    .shaderIndex = util::safeCastToU32(createInfos.size()),
                });
            }
        }

        // shaderCode only has a value when the shader binary cache has a hit.
        // If the shader is not in the cache, we create the shader object as a SPIR-V shader.
        if (!shaderCode.has_value()) {
            util::log_msg("[Init] Reading SPIR-V Shader: {}", shaderName);

            // Light validation of the SPIR-V bytes when reading the file.
            // We know the file is not empty and we checked it has a valid size and alignment (a multiple of 4 bytes).
            util::require(!spirvBytes.empty(), "Shader key generation must load SPIR-V before shader creation");

            // We can assume codeType is SPIR-V. The only way we modify codeType is after a shader cache hit,
            // and we will not enter this branch in that case because shaderCode has a value.
            util::require(codeType == vk::ShaderCodeTypeEXT::eSpirv, "SPIR-V shader code must use eSpirv code type");

            // We use the SPIR-V data to create our shaders.
            shaderCode = spirvBytes;
        }

        // Check that the input code is correctly aligned.
        {
            // When creating shader objects, the buffers containing our shader code have alignment requirements.
            // The spec defines the required alignment depending on whether they contain SPIR-V or binary data.
            // We are currently using BinaryBuffer, which handles these alignment requirements for us.
            util::require(reinterpret_cast<std::size_t>(shaderCode->data()) ==
                              util::alignUp(reinterpret_cast<std::size_t>(shaderCode->data()),
                                            codeType == vk::ShaderCodeTypeEXT::eSpirv ? size_t(4) : size_t(16)),
                          "When creating a shader object, code must be aligned to 16 bytes for binary objects and 4 "
                          "bytes for SPIR-V");
        }

        // Add one shader create info for each stage in the pipeline.
        // This is necessary for linking, since all shaders in the pipeline have to be created together.
        // For simplicity, the tutorial uses the same strategy when linking is disabled.
        // This might create the same shader object multiple times when linking is disabled,
        // and should be avoided by a real application.
        createInfos.emplace_back(vk::ShaderCreateInfoEXT{
            // Note: Mappings are ignored if code type is binary. We could still set them, but they would be ignored.
            .pNext = (codeType == vk::ShaderCodeTypeEXT::eSpirv) ? &mappingInfo : nullptr,
            .flags = shaderFlags,
            .stage = stage,
            // We need to specify the next stage in our pipeline.
            // The tutorial avoids tessellation and geometry shaders, so it is safe to assume that the next stage
            // after a vertex shader is a fragment shader.
            // We only support vertex and fragment shaders, and there is no next stage after a fragment shader.
            .nextStage = (stage == vk::ShaderStageFlagBits::eVertex) ? vk::ShaderStageFlagBits::eFragment
                                                                     : vk::ShaderStageFlagBits(0),
            // The only necessary change between creating a shader object using SPIR-V or binary data is codeType.
            // Our application stores either the SPIR-V or binary data in shaderCode.
            // Note: We can still specify mappings in pNext, but they will be ignored by binary shaders.
            // Note: shaderCode will store SPIR-V or binary data.
            .codeType = codeType,
            .codeSize = shaderCode->size(),
            .pCode = shaderCode->data(),
            // Slang reflection reports source function names. The generated SPIR-V entry point is "main".
            .pName = "main",
        });
    };

    // The tutorial creates shader objects using linking.
    // We need to specify all shaders in the pipeline in the same call.
    // When linking, shader objects require us to create both the vertex and fragment shaders in the same call.
    addShaderCreateInfo(shaderFilePair.vertexName, vk::ShaderStageFlagBits::eVertex, vertexSpirvWords.as_byte_span());
    addShaderCreateInfo(shaderFilePair.fragmentName, vk::ShaderStageFlagBits::eFragment,
                        fragmentSpirvWords.as_byte_span());

    util::log_msg("[Init] Create ShaderObjects: {} + {}", shaderFilePair.vertexName, shaderFilePair.fragmentName);

    // Check that all linked shaders use the same code type: all must be binary or SPIR-V.
    if (enable_linking) {
        // When linking shaders using shader objects, all shaders we send in the same call need to have the same
        // codeType. They all have to be either SPIR-V or binary; we are not allowed to combine them.
        // If our cache detects a hit for the vertex shader, but not for the fragment shader, this check throws before
        // we try to create shader objects with invalid API usage.
        // Under normal conditions, the vertex and fragment shaders cannot have mixed code types because we always save
        // both shaders to the cache. However, file-write failures or users deleting cache files can cause this.
        // Printing an error and failing is acceptable for a tutorial, but a real application would probably want to
        // update the cache and try to create both shaders using SPIR-V.
        util::require(
            cacheMisses.empty() || (createInfos.size() == cacheMisses.size()),
            std::format("Error in shader cache for shaders {} and {}. All shaders should be binary or SPIR-V. "
                        "Consider deleting the shader cache.",
                        shaderFilePair.vertexName, shaderFilePair.fragmentName));
    }

    // Linking requires creating all shaders in the pipeline in the same call.
    // For simplicity, always create both shaders in the same call.
    auto result = m_logicalDevice.createShadersEXTUnique(createInfos);

    // Ensure both shaders have been created correctly.
    //
    // Note: the tutorial is treating eIncompatibleShaderBinaryEXT as fatal.
    // A proper application would detect if the driver failed to generate shader objects from the shader binaries in the
    // cache, and if that happens it would fall back to trying to create the shaders using SPIR-V.
    util::require(result.has_value(),
                  std::format("Failed to create shader objects for {} and {}: {}", shaderFilePair.vertexName,
                              shaderFilePair.fragmentName, vk::to_string(result.result)));

    // Get all shader objects.
    std::vector<vk::UniqueShaderEXT> shaders = std::move(result.value);
    util::require(shaders.size() == 2, "Expected linked vertex and fragment shader objects");

    // Shader binary cache write.
    if (enable_shader_binary_cache) {

        // If the shader binary cache is enabled, update it.
        // This loop iterates through shaders that we did not find in the cache and saves their binaries to disk.
        for (const ShaderCacheMiss& cacheMiss : cacheMisses) {

            // Shader objects make managing a shader cache straightforward.
            // We can easily retrieve the compiled binary from a shader by calling getShaderBinaryDataEXT.
            // Then we can use this data when creating a new shader object.
            // This offers more direct control and an easier method for managing the cache than PipelineCache.
            const std::vector<std::uint8_t> shaderBinary =
                m_logicalDevice.getShaderBinaryDataEXT(*shaders[cacheMiss.shaderIndex]);

            // Save it to disk for future runs. This simple tutorial assumes the file write succeeds.
            // The cache key was already calculated to check if the file is in the cache.
            // Remember that it includes the shader SPIR-V to make it unique per shader, and the UUID/version so driver
            // updates use different cache files, alongside all other relevant data.
            util::writeBinaryFile(cacheMiss.cachePath, shaderBinary);

            util::log_msg("[Init] Writing Shader Binary Cache: {}", cacheMiss.cachePath.filename().string());
        }
    }

    ShaderGroup shaderGroup{
        .m_vertex = std::move(shaders[0]),
        .m_fragment = std::move(shaders[1]),
    };

    // Set debug names for the shaders.
    {
        const std::string vertexDebugName =
            std::format("Vertex Shader ({} + {})", shaderFilePair.vertexName, shaderFilePair.fragmentName);
        setDebugName(*shaderGroup.m_vertex, vk::ObjectType::eShaderEXT, vertexDebugName);

        const std::string fragmentDebugName =
            std::format("Fragment Shader ({} + {})", shaderFilePair.vertexName, shaderFilePair.fragmentName);
        setDebugName(*shaderGroup.m_fragment, vk::ObjectType::eShaderEXT, fragmentDebugName);
    }
    return shaderGroup;
}

//****************
// Runtime loop
//****************

void Application::mainLoop()
{
    // This is the main loop function of the application.
    //
    // Most games have a similar function, usually called a main loop or rendering loop.
    // This function starts and runs the main loop that the application will run.
    //
    // The common workflow for a game is:
    // - Initialization: create rendering and non-rendering resources.
    // - Main loop: run each frame to handle input, update game logic, and generate new rendering commands.
    // - Resource cleanup: destroy and free all game resources.
    //
    // This function will start a loop that runs until the window is closed.
    // We use this loop each frame to:
    // - Handle keyboard input.
    // - Update the camera.
    // - Issue new rendering commands.
    //
    // Real games follow a more complex but similar approach.
    // Our tutorial does not have logic, a physics system, or animations, but they are usually handled here.
    // Note: real games should consider separating logic update times and rendering times.
    //       For example, you might want to render at 60 FPS but update your game simulation every 25 ms.
    //       You might also want to consider different threads for rendering and logic updates.
    // Note: Game engine architecture is an extremely complex topic.
    //       You might want to check object hierarchies (tree/node systems) and Entity Component Systems (ECS).

    // Measure current time to calculate delta time for camera movement and FPS logging.
    // It is a best practice to use time deltas to update logic systems, like camera movement.
    // Proper applications might also use this for animation, physics, or game logic.
    // Note: GLFW returns the time in seconds, and this tutorial uses seconds for all times.
    double previousTime = glfwGetTime();
    double lastFpsLogTime = previousTime;

    // We print a simple FPS message every framesPerLog frames.
    constexpr uint32_t framesPerLog = 20;
    uint32_t remainingFramesLog = framesPerLog;

    // We have a CLI option to exit the application after a certain number of frames.
    // Log it at the beginning of the run so it is easy to see in the logs.
    // Logs and CLI flags can greatly help with automated testing and debugging.
    // Consider CLI flags like: --save_image_to_disk, --disable_cache, etc.
    if (m_remainingFrameLimit > 0) {
        util::log_msg("[Run] Frame limit: {}", m_remainingFrameLimit);
    }

    // Loop until window is closed.
    // This is the main loop. The application will run until the user closes the window or presses escape.
    while (glfwWindowShouldClose(m_window) == GLFW_FALSE) {

        // Update CPU time. We use GLFW to get the current time.
        // We store last frame time and calculate delta time for camera movement and FPS logging.
        const double currentTime = glfwGetTime();
        const float deltaSeconds = static_cast<float>(currentTime - previousTime);

        // Small validation that frame time is non-negative.
        util::require(deltaSeconds >= 0.0f, std::format("Invalid delta time {} between frames", deltaSeconds));

        // Next iteration will use the current time as previous time.
        previousTime = currentTime;

        // Simple FPS logging using FPS counter and delta times.
        if (--remainingFramesLog == 0) {
            // Seconds since last frame log.
            const float elapsedSeconds = static_cast<float>(currentTime - lastFpsLogTime);

            // Check to avoid division by 0.
            // This should not happen unless the timer did not advance while rendering frames.
            util::require(elapsedSeconds > 0.0f, std::format("Invalid elapsed seconds {}", elapsedSeconds));

            const float fps = framesPerLog / elapsedSeconds;
            util::log_msg("FPS: {}", fps);
            remainingFramesLog = framesPerLog;
            lastFpsLogTime = currentTime;
        }

        // Poll events. Inputs are handled in updateCamera() using GLFW.
        glfwPollEvents();

        // Detect if escape key is pressed and close the window if it is.
        if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            break;
        }

        // We use GLFW to read cached input, detecting key presses to update the camera position and orientation.
        // We use deltaSeconds to update the camera position and orientation independent of the frame rate.
        updateCamera(deltaSeconds);

        // Main Vulkan code.
        // This creates and submits all commands to render the current frame.
        renderFrame();

        // Simple CLI option to finish after a certain number of frames, useful for automated testing.
        if (m_remainingFrameLimit > 0) {
            --m_remainingFrameLimit;
            util::log_msg("Remaining frames: {}", m_remainingFrameLimit);
            if (m_remainingFrameLimit == 0) {
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
        }
    }
}

void Application::updateCamera(float deltaSeconds)
{
    // This function uses GLFW to detect pressed keys and use the keyboard to move the camera.
    // We are only updating the data we have in the CPU. Camera data is later uploaded to the GPU.
    // Note: For tutorial simplicity we do not handle mouse input.

    // Limit delta seconds to avoid jumps when the frame stutters.
    deltaSeconds = std::min(deltaSeconds, 0.3f);

    // Rotation using arrow keys.
    {
        // Rotate camera using arrow keys.
        // Mouse control is not implemented in this tutorial, but it would be simple to implement using GLFW.
        constexpr float cameraLookSpeed = 1.6F;
        const float lookStep = cameraLookSpeed * deltaSeconds;
        // Main loop calls glfwPollEvents to poll and process pending events.
        // We use glfwGetKey to read GLFW cached key state.
        if (glfwGetKey(m_window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            m_cameraYaw -= lookStep;
        }
        if (glfwGetKey(m_window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            m_cameraYaw += lookStep;
        }
        if (glfwGetKey(m_window, GLFW_KEY_UP) == GLFW_PRESS) {
            m_cameraPitch += lookStep;
        }
        if (glfwGetKey(m_window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            m_cameraPitch -= lookStep;
        }
        // calculateRight uses a cross product. Avoid looking straight up/down to avoid problems.
        m_cameraPitch = std::clamp(m_cameraPitch, -cameraPitchLimit, cameraPitchLimit);
    }

    // Movement using WASD and QE.
    {
        // We are implementing a free camera movement for our game.
        // Classic WASD movement controls: WS for forward/backward and AD for left/right.
        // E moves up; Q moves down.
        //
        // We are using GLFW for mouse and keyboard input.
        // This is a simple cross-platform library that can handle input. We can also use SDL as an alternative.
        const glm::vec3 forward = util::math::calculateForward(m_cameraPitch, m_cameraYaw);
        const glm::vec3 worldUp{0.0F, 1.0F, 0.0F};
        const glm::vec3 right = util::math::calculateRight(forward, worldUp);

        glm::vec3 movement{0.0F};
        if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) {
            movement += forward;
        }
        if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) {
            movement -= forward;
        }
        if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) {
            movement += right;
        }
        if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) {
            movement -= right;
        }
        if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS) {
            movement += worldUp;
        }
        if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS) {
            movement -= worldUp;
        }

        if (glm::length(movement) > 0.0F) {
            constexpr float cameraMoveSpeed = 4.0F;
            m_cameraPos += glm::normalize(movement) * cameraMoveSpeed * deltaSeconds;
        }
    }
}

void Application::renderFrame()
{
    // This function is executed during the main loop.
    // This function will:
    // - Update data on the CPU (update data in the camera buffer).
    // - Record commands to render the current frame.
    // - Submit the command buffer so it is executed on the GPU.

    // Summary of synchronization for the frame:
    // - Fence: lets the CPU wait before reusing a frame slot.
    // - Semaphores: order GPU-side acquire, render, and present work.
    // - Events: allow more granular synchronization, but are unused in this tutorial.
    // - Barriers: order image layout/access changes inside the recorded command buffer.

    // The helper command buffer is used to upload data to the GPU during initialization.
    // It should not be active during rendering.
    util::require(!m_helperCommandBuffer.m_active, "Helper command buffer must not be active while rendering");

    // Select the current frame-in-flight resources.
    //
    // Our render uses multiple frames in flight. This allows the application to start working on the next frame while
    // the previous frame is still in flight.
    // Frames in flight make us duplicate some resources, but help ensure correct utilization.
    const std::size_t frameIndex = m_currentFrameInFlight;
    FrameInFlightResources& frame = m_framesInFlight[frameIndex];

    // This ensures this frame's resources are available.
    waitForFrameResources(frame);

    // We use frame resources since the swapchain signals frame.m_imageAvailableSemaphore when the image is available.
    // It is important that frame and swapchain resources are kept independent.
    const std::uint32_t swapchainImageIndex = acquireSwapchainImage(frame);

    // Reset and begin the main command buffer so we can start recording commands.
    // Note: We need to call this before we start recording work in our command buffer.
    // In our tutorial we could call this after uploadCameraData,
    // but a real application is likely to use staging buffers with commands to update GPU resources.
    startRecordingCommandBuffer(frame);

    // Use CPU data to update GPU data.
    // We only have a camera buffer, but this usually includes animations, streaming new objects, etc.
    //
    // Because we are using host memory, we do not need to record commands to update resources.
    // But most apps will have a more optimized approach to update resources using commands.
    // Note: we still wait for a fence and submit work after writing to avoid race conditions.
    uploadCameraData(frame);

    // Record commands used to render.
    // This records commands on the CPU that are later executed on the GPU.
    // This only records rendering commands.
    recordRenderingCommandBuffer(frame, util::safeCastToU32(frameIndex), swapchainImageIndex);

    // Submit the command buffer with rendering commands to the GPU so that it can be executed.
    // This will mark the command buffers as ended before submission.
    // We will also present the image to the swapchain and handle inter-frame synchronization.
    finishAndSubmitMainCommandBuffer(frame, swapchainImageIndex);

    // Advance through the small ring of frame resources.
    m_currentFrameInFlight =
        util::safeCastToU32((static_cast<std::size_t>(m_currentFrameInFlight) + 1U) % m_framesInFlight.size());
}

void Application::waitForFrameResources(FrameInFlightResources& frame)
{
    // This ensures that we can reuse the per-frame resources.
    // This fence protects frame resources rather than swapchain data, allows the application to record new commands,
    // and ensures that intermediate resources like depth images or the camera buffer are safe to use.

    // Wait until this frame slot's previous GPU work is done before reusing its resources.
    // We cannot write previous resources unless we are done using them.
    // Note: We use a fence because the CPU needs to wait for the GPU. CPU waits to reuse command buffer and other frame
    // resources until the GPU finishes using them.
    const std::array frameFences{*frame.m_inFlightFence};

    const bool waitAll = true;
    const uint64_t timeout = UINT64_MAX;

    const vk::Result waitResult = m_logicalDevice.waitForFences(frameFences, waitAll, timeout);
    util::require(waitResult == vk::Result::eSuccess, "Failed to wait for frame fence");
}

std::uint32_t Application::acquireSwapchainImage(FrameInFlightResources& frame)
{
    // This function requests a new swapchain image so that we can use it for rendering.

    // Ask the swapchain for the next image index.
    // The swapchain image is not immediately available, so we need synchronization. We use imageAvailableSemaphore as
    // an acquire semaphore. The imageAvailableSemaphore is signaled when the presentation engine has released the
    // image and the image is available for application use. In our application, we use this semaphore in the graphics
    // queue to wait before rendering to the image (we render directly to the swapchain image).
    const auto acquireResult = m_logicalDevice.acquireNextImageKHR(
        vk::SwapchainKHR{m_vkbData.m_swapchain.swapchain}, UINT64_MAX, *frame.m_imageAvailableSemaphore, nullptr);

    // Tutorial simplifies swapchain management, ignoring most use cases.
    {
        // Tutorial skips recreating the swapchain, but a real app should properly deal with this.
        if (acquireResult.result == vk::Result::eSuboptimalKHR) {
            util::log_msg("Ignoring swapchain suboptimal: "
                          "Swapchain can still be presented but should be recreated soon.");
        }
        else {
            util::require(acquireResult.result != vk::Result::eErrorOutOfDateKHR,
                          "Swapchain is out of date. Resize omitted from the tutorial.");
            util::require(
                acquireResult.result == vk::Result::eSuccess,
                std::format("Failed to acquire swapchain image. Result {}\n", vk::to_string(acquireResult.result)));
        }
    }
    const std::uint32_t swapchainImageIndex = acquireResult.value;
    SwapchainImageResources& swapchainImage = m_swapchainImages[swapchainImageIndex];

    // finishAndSubmitMainCommandBuffer describes the real semaphore wait during submission.
    // This internal ImageState update only copies the wait stage into our tracked source state,
    // so the next image barrier describes the dependency from image acquisition to the first color attachment write.
    // This makes our dependency tracker consider the swapchain acquire semaphore wait when creating barriers.
    // We are not changing the layout here.
    swapchainImage.m_state.m_stageMask |= swapchainAcquireWaitStage;

    return swapchainImageIndex;
}

void Application::uploadCameraData(FrameInFlightResources& frame)
{
    // Update camera data on the GPU.
    // This updates the version of the camera data on the GPU.
    // Note: We have one camera resource per frame in flight, so we will not overwrite the camera data.

    // Ensure we do not have division by 0.
    util::require(m_swapchainExtent.height > 0, "Swapchain height must be non-zero before updating camera");

    // CPU has the data needed to compute the camera matrix.
    // We compute the camera matrix and send it to the GPU.
    const float aspectRatio =
        static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
    const CameraData camera{
        .viewProjection = util::math::calculateViewProjection(m_cameraPitch, m_cameraYaw, m_cameraPos, aspectRatio),
        .cameraPosition = glm::vec4{m_cameraPos, 1.0F}};

    // We will directly update the data in the camera buffer on the CPU, and then make it available to the GPU.
    // Note: We write from the CPU after GPU reads and before submission to ensure synchronization.
    // Note: Camera data is a host-coherent buffer, so we do not need to flush or invalidate.
    // Note: A proper application would probably want a staging buffer to keep the shader data device local.
    uploadBuffer(frame.m_camera, std::as_bytes(std::span{&camera, std::size_t{1}}));
}

void Application::startRecordingCommandBuffer(FrameInFlightResources& frame)
{
    // Prepare a command buffer to start recording to it.

    // Reset command buffer before we start recording to it.
    frame.m_commandBuffer.reset();

    // Begin recording commands. This indicates we will start recording work.
    frame.m_commandBuffer.begin(vk::CommandBufferBeginInfo{});
}

void Application::recordRenderingCommandBuffer(FrameInFlightResources& frame, std::uint32_t frameIndex,
                                               std::uint32_t swapchainImageIndex)
{
    // This function creates the commands needed to render one frame.
    // This function only handles rendering, resource management is handled by other functions.

    util::require(frameIndex < m_framesInFlight.size(), "Missing frame index");

    // Get the resources associated with the swapchain image we use for rendering.
    util::require(swapchainImageIndex < m_swapchainImages.size(), "Missing swapchain image");
    SwapchainImageResources& swapchainImage = m_swapchainImages[swapchainImageIndex];

    // Store aliases to the frame resources we will use.
    const GpuViewImage& depthImage = frame.m_depthImage;
    const vk::CommandBuffer commandBuffer = frame.m_commandBuffer;

    // Add debug labels to identify the current frame.
    // Debug labels are a key part of debugging and profiling applications. They interact well with tools like
    // RenderDoc, allowing developers to identify which part of the frame is being analyzed.
    const std::string frameDebugLabel =
        std::format("Rendering Frame {} SwapchainImage {}", frameIndex, swapchainImageIndex);
    beginDebugLabel(commandBuffer, frameDebugLabel, debugData.frameColor);

    // We are setting eAttachmentOptimal for the depth and color attachments.
    // With synchronization2 enabled, we can use a generic attachment optimal layout
    // instead of specialized layouts for both color and depth attachments.
    //
    // If unified image layouts were enabled, we could simplify this further, since the eGeneral layout
    // could be used in place of most specialized layouts, with no expected performance loss.
    // The current tutorial is too simple to benefit from unified layouts, but the extension
    // can simplify barriers and reduce layout transitions for images with multiple usages.
    // Note: unified image layouts do not remove the need to transition to/from present,
    // and undefined remains useful for discards.

    // Transition from "present/undefined" into a layout valid for color attachment writes.
    // In the tutorial, swapchain images move between rendering and presentation every frame,
    // unlike depth images which stay in the depth-attachment layout after setup.
    {
        const ImageState attachmentState{
            .m_layout = colorAttachmentLayout,
            .m_stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .m_accessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .m_aspectMask = vk::ImageAspectFlagBits::eColor,
        };
        transitionImage(commandBuffer, swapchainImage.m_image, swapchainImage.m_state, attachmentState,
                        CreateImageSubresourceRange());
    }

    // Record in the command buffer that rendering begins.
    {
        // We use beginRendering to indicate we can start doing draw calls. This is a command from dynamic rendering.
        // With dynamic rendering, we no longer need to create render pass objects and set subpasses. Instead, we can
        // set the information needed to start rendering in vk::RenderingInfo and call draw commands directly.
        // This significantly reduces boilerplate and helps to make applications more dynamic.
        //
        // Note: Tile Based Rendering (TBR) GPUs can benefit from dynamic rendering local read,
        // a Vulkan extension that allows subpass like local memory access with dynamic rendering.
        // This tutorial does not use local read. We recommend reading about it when targeting mobile.

        // In order to begin rendering we need to set up our attachments.
        // Attachments are image views connected to framebuffer operations during rendering.
        // There are multiple types of attachments in Vulkan:
        // color attachments: Store the outputs of the fragment shader.
        // depth attachment: Stores the depth, used during depth testing to decide which fragments we will overwrite.
        // stencil attachment: Stencil values work like a mask to discard fragments. Unused in this tutorial.

        // Set the clear values for the attachment in our application.
        // Clear the swapchain color image and depth attachment before drawing the scene.
        vk::ClearValue colorClearValue{};
        // Note: workaround to initialize a union.
        colorClearValue.color.float32 = std::array{0.55F, 0.35F, 0.35F, 1.0F}; // Greyish pink for sky/background.
        vk::ClearValue depthClearValue{};
        // Note: workaround to initialize a union
        depthClearValue.depthStencil.depth = 1.0F;

        const std::array colorAttachments = {vk::RenderingAttachmentInfo{
            .imageView = *swapchainImage.m_imageView, // We render directly to the swapchain.
            .imageLayout = colorAttachmentLayout,
            .resolveMode = vk::ResolveModeFlagBits::eNone,
            .resolveImageView = nullptr,
            .resolveImageLayout = vk::ImageLayout::eUndefined,
            .loadOp = vk::AttachmentLoadOp::eClear, // Clear before drawing.
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = colorClearValue,
        }};

        const vk::RenderingAttachmentInfo depthAttachment{
            .imageView = *depthImage.m_imageView, // Use per-frame depth image.
            .imageLayout = depthAttachmentLayout,
            // Clear depth attachment before rendering.
            .loadOp = vk::AttachmentLoadOp::eClear,
            // We do not use the depth attachment later.
            // We only use it for depth testing during the pass, so we can discard it.
            // Next pass will clear the current value, so we do not have to store it.
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = depthClearValue,
        };

        util::require(m_swapchainExtent.height > 0 && m_swapchainExtent.width > 0,
                      "Render area dimensions must be greater than 0");

        const vk::RenderingInfo renderingInfo{
            .renderArea = // Area to render.
            vk::Rect2D{
                .offset = vk::Offset2D{.x = 0, .y = 0},
                .extent = m_swapchainExtent,
            },
            .layerCount = 1, // This is commonly 1 except for VR or other layered rendering.
            .colorAttachmentCount = util::safeCastToU32(colorAttachments.size()),
            // Fragment shaders can have multiple outputs, so Vulkan supports multiple color attachments.
            .pColorAttachments = colorAttachments.data(),
            // We always have a single depth attachment.
            // In Vulkan, a render pass has a single depth and a single stencil attachment.
            .pDepthAttachment = &depthAttachment,
        };

        // Start rendering.
        // This is equivalent to beginning a new dynamic rendering render pass instance,
        // but a lot easier since we do not have to create all the intermediate objects.
        commandBuffer.beginRendering(renderingInfo);
    }

    // Set dynamic state.
    {
        // We are using dynamic state to set the graphics state of our application.
        // In a graphics pipeline, some stages execute in fixed-function hardware and cannot be programmed;
        // however, most of this state can be configured using graphics state.
        // Traditionally, Vulkan expects this state to be given upfront when creating a pipeline. This has proven
        // difficult for applications. Moreover, having the state as part of the pipeline has created an explosion of
        // PSOs, where small state changes require a full pipeline recompilation.
        //
        // Dynamic state allows us to record part of the state during command recording, rather than during resource
        // initialization. This delay helps applications manage resource creation. Removing the state from caching
        // reduces the number of PSOs applications need to manage.
        //
        // Dynamic state can greatly help applications, so Vulkan has created new extensions to allow implementations to
        // mark more state as supporting dynamic state.
        //
        // With shader objects, we no longer have pipelines, so all required graphics state has to be set using dynamic
        // state as in our application.
        //
        // Note: dynamic state can still be used with pipelines. When creating the pipeline, the state is marked as
        // dynamic. This will allow us to set the state during command recording rather than pipeline creation.
        beginDebugLabel(commandBuffer, "Set Dynamic Graphics State", debugData.renderColor);

        // With dynamic state, we just need to record commands in our command buffer setting all relevant state.

        // Set viewport using dynamic state.
        {
            // The viewport maps normalized device coordinates to pixels in the framebuffer.
            // We render to the entire swapchain image.
            const vk::Viewport viewport{
                .x = 0.0F,
                .y = 0.0F,
                .width = static_cast<float>(m_swapchainExtent.width),
                .height = static_cast<float>(m_swapchainExtent.height),
                .minDepth = 0.0F,
                .maxDepth = 1.0F,
            };
            commandBuffer.setViewportWithCountEXT(viewport);
        }

        // Set the scissor using dynamic state.
        {
            // The scissor controls which pixels are allowed to be written.
            // Pixels outside of the scissor rectangle are discarded.
            // We render to the entire swapchain image.
            const vk::Rect2D scissor{
                .offset = vk::Offset2D{.x = 0, .y = 0},
                .extent = m_swapchainExtent,
            };
            commandBuffer.setScissorWithCountEXT(scissor);
        }

        commandBuffer.setPrimitiveTopologyEXT(vk::PrimitiveTopology::eTriangleList);
        commandBuffer.setPrimitiveRestartEnableEXT(vk::False);

        commandBuffer.setRasterizerDiscardEnableEXT(vk::False);
        commandBuffer.setDepthClampEnableEXT(vk::False);
        commandBuffer.setPolygonModeEXT(vk::PolygonMode::eFill);

        commandBuffer.setCullModeEXT(vk::CullModeFlagBits::eNone);
        commandBuffer.setFrontFaceEXT(vk::FrontFace::eCounterClockwise);
        commandBuffer.setDepthBiasEnableEXT(vk::False);

        commandBuffer.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
        {
            const std::array sampleMasks{vk::SampleMask{0xFFFFFFFFU}};

            commandBuffer.setSampleMaskEXT(vk::SampleCountFlagBits::e1, sampleMasks);
        }
        commandBuffer.setAlphaToCoverageEnableEXT(vk::False);
        commandBuffer.setAlphaToOneEnableEXT(vk::False);

        commandBuffer.setLogicOpEnableEXT(vk::False);
        {
            const std::array colorBlendEnables{vk::False};
            commandBuffer.setColorBlendEnableEXT(0, colorBlendEnables);
        }
        {
            const std::array colorWriteMasks{
                vk::ColorComponentFlags{vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA}};
            commandBuffer.setColorWriteMaskEXT(0, colorWriteMasks);
        }

        commandBuffer.setDepthTestEnableEXT(vk::True);
        commandBuffer.setDepthWriteEnableEXT(vk::True);
        // Depth buffer is cleared to 1.0, so we keep fragments with a smaller depth value.
        commandBuffer.setDepthCompareOpEXT(vk::CompareOp::eLess);
        commandBuffer.setDepthBoundsTestEnableEXT(vk::False);
        commandBuffer.setStencilTestEnableEXT(vk::False);

        // End Set Dynamic Graphics State.
        endDebugLabel(commandBuffer);
    }

    // Bind vertex input using dynamic state.
    {
        // We are using dynamic state to select the format of our vertex input.
        // Usually, vertex input depends on the vertex shader, and might change per-draw.
        // This sample has one vertex layout, so this is enough.
        VertexInput& vertexInput = m_vertexInput;
        commandBuffer.setVertexInputEXT(vertexInput.m_vertexBindings, vertexInput.m_vertexAttributes);
    }

    // Validate descriptor heaps are ready.
    {
        util::require(m_descriptorHeaps.m_resourceHeap.m_buffer.m_addressGPU != 0,
                      "Resource heap must have a GPU address");
        util::require(m_descriptorHeaps.m_samplerHeap.m_buffer.m_addressGPU != 0,
                      "Sampler heap must have a GPU address");
    }

    // Bind descriptor heap buffers.
    {

        // The shader mappings interpret heap bytes as descriptors.
        // Now we need to bind our heaps so that we have a buffer from which to read these descriptors.
        beginDebugLabel(commandBuffer, "Bind Descriptor Heaps", debugData.setupColor);

        // Helper to bind a heap. Allows us to reuse code to bind the resource and sampler heap.
        auto createBindInfo = [](const DescriptorHeapResources::DescriptorHeapData& descriptorHeapData) {
            return vk::BindHeapInfoEXT{
                .heapRange =
                    vk::DeviceAddressRangeEXT{
                        // We do not bind the entire buffer. We have bindOffset as padding to ensure GPU alignment.
                        .address = descriptorHeapData.m_buffer.m_addressGPU + descriptorHeapData.m_bindOffset,
                        .size = descriptorHeapData.m_rangeSize,
                    },
                // Reserved range is a space reserved for implementation internal use.
                // The application reserves this space and cannot use it while the heap is bound.
                // Offset is 0 because this sample places the required reserved range at the start of the heap.
                // The size in the tutorial is the queried minimum reserved range required on current hardware.
                .reservedRangeOffset = 0,
                .reservedRangeSize = descriptorHeapData.m_minReservedRange,
            };
        };

        // Create the bind info of both heaps.
        const vk::BindHeapInfoEXT resourceHeapInfo = createBindInfo(m_descriptorHeaps.m_resourceHeap);
        const vk::BindHeapInfoEXT samplerHeapInfo = createBindInfo(m_descriptorHeaps.m_samplerHeap);

        // The application binds the sampler and resource heap to the command buffer.
        // Note: Switching heaps is expensive, for good performance applications are expected to bind the same sampler
        // and resource heap during the entire application lifetime.
        commandBuffer.bindResourceHeapEXT(resourceHeapInfo);
        commandBuffer.bindSamplerHeapEXT(samplerHeapInfo);

        // End Bind Descriptor Heaps.
        endDebugLabel(commandBuffer);
    }

    // Helper function called during the object loop to bind the shader objects.
    const auto bindShaderGroup = [commandBuffer](const ShaderGroup& shaderGroup) {
        beginDebugLabel(commandBuffer, "Bind Shader Objects", debugData.setupColor);

        // We bind all active shaders in a single call.
        const std::array stages{vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment};
        const std::array shaders{*shaderGroup.m_vertex, *shaderGroup.m_fragment};
        commandBuffer.bindShadersEXT(stages, shaders);

        // End Bind Shader Objects.
        endDebugLabel(commandBuffer);
    };

    // Helper function to push data.
    const auto pushDescriptorHeapShaderData = [maxPushDataSize = m_descriptorHeaps.m_maxPushDataSize,
                                               commandBuffer](const auto& pushData) {
        // This function uses push data to record some data that can be read by the GPU.
        // Descriptor heaps offer push data, allowing the application to record a binary blob into push-data command
        // state that GPU shaders will read.
        // This can be used to replace push constants, but it has some significant advantages.
        // It can be used to communicate some small data to the GPU that changes on each draw call.
        // Our tutorial uses it to select the texture indices, but it could be used to send arbitrary data.

        // Offset of the GPU push-data region we will update.
        // We are updating the entire copy on the GPU, so we set it to 0.
        constexpr std::uint32_t pushDataOffset = 0;

        constexpr std::uint32_t pushDataSize = sizeof(pushData);

        // Small data validation.
        {
            // Ensure the push data we are updating does not exceed device limits.
            // In a proper application, this check should be moved to initialization rather than recording.
            util::require(pushDataOffset <= maxPushDataSize && pushDataSize <= maxPushDataSize - pushDataOffset,
                          "Descriptor heap push data exceeds maxPushDataSize");

            static_assert((pushDataSize % 4) == 0, "push data size must be a multiple of 4");
            static_assert((pushDataOffset % 4) == 0, "push data offset must be a multiple of 4");
        }

        const vk::PushDataInfoEXT pushDataInfo{
            .offset = pushDataOffset,
            // CPU memory used to read the data we will send.
            .data =
                vk::HostAddressRangeConstEXT{
                    .address = &pushData,
                    .size = pushDataSize,
                },
        };
        // Sends push data from the CPU to the GPU.
        commandBuffer.pushDataEXT(pushDataInfo);
    };

    // Helper to create the index for an image.
    auto createCombinedImageSamplerIndex = [](std::uint32_t imageIndex, std::uint32_t samplerIndex) {
        // We are using a combined image-sampler index.
        // That allows us to set separate image and sampler descriptor indices using a single 32 bit push data value.

        util::require(imageIndex <= 0xFFFFF, "Image index must fit in 20 bits");
        util::require(samplerIndex <= 0xFFF, "Sampler index must fit in 12 bits");
        return (samplerIndex << 20) | imageIndex;
    };

    std::optional<ShaderVariant> boundShaderVariant;

    // Draw each scene node.
    // This main loop will iterate through all objects in our scene.
    // Check if state needs to be updated.
    // Issue a draw call.
    beginDebugLabel(commandBuffer, "Scene Draws", debugData.drawColor);
    for (const SceneDraw& draw : m_scene.m_drawData) {
        // Note: We keep track of the current bound shaders to avoid unnecessary binds.
        // We use the object name to identify the call.
        beginDebugLabel(commandBuffer, std::format("Draw {}", draw.m_debugName), debugData.drawColor);

        util::require(draw.m_meshId < m_scene.m_meshes.size(), "Scene draw references an invalid mesh");
        const GpuMesh& mesh = m_scene.m_meshes[draw.m_meshId];

        // Bind a new shader group if the shader group changes.
        // This is still a normal bind call, but without a pipeline object.
        // Note: We keep track of the current bound shaders to avoid unnecessary binds.
        if (!boundShaderVariant.has_value() || *boundShaderVariant != draw.m_shaderVariant) {
            switch (draw.m_shaderVariant) {
            case ShaderVariant::AlbedoAndNormal:
                bindShaderGroup(m_shaderObjects.m_albedoAndNormal);
                break;
            case ShaderVariant::Albedo:
                bindShaderGroup(m_shaderObjects.m_albedo);
                break;
            case ShaderVariant::SolidColor:
                bindShaderGroup(m_shaderObjects.m_solidColor);
                break;
            default:
                util::require(false, "Unsupported shader variant");
                break;
            }
            boundShaderVariant = draw.m_shaderVariant;
        }

        // Push draw-specific data.
        {
            // Push draw-specific data.
            // This allows us to easily set different textures for each draw call.
            // Note: we need to send different data depending on the call.

            // The tutorial hardcodes sampler indices depending on the use of the texture.
            // Linear sampler usually offers better quality for albedo textures.
            // Nearest sampler is used for normal maps in the tutorial, for illustration purposes.
            constexpr uint32_t linearSamplerIndex = 0;
            constexpr uint32_t nearestSamplerIndex = 1;

            // Albedo + normal-map draws push camera + albedo and normal texture indices.
            if (draw.m_shaderVariant == ShaderVariant::AlbedoAndNormal) {

                // Check that the texture is valid.
                {
                    util::require(draw.m_albedoTextureIndex < m_scene.m_textures.size(),
                                  "Scene draw references an invalid albedo texture");
                    util::require(draw.m_normalTextureIndex < m_scene.m_textures.size(),
                                  "Scene draw references an invalid normal texture");
                }

                // Create and set the value of the data we will push.
                const DescriptorHeapDrawPushIndicesAlbedoAndNormal pushData{
                    .cameraIndex = frameIndex,
                    .albedoTextureIndex =
                        createCombinedImageSamplerIndex(draw.m_albedoTextureIndex, linearSamplerIndex),
                    .normalTextureIndex =
                        createCombinedImageSamplerIndex(draw.m_normalTextureIndex, nearestSamplerIndex),
                };

                // This is a tutorial. We know that the bound buffers and textures have the correct state.
                // In a real application you might need to schedule transitions to handle intermediate resources.
                // Usually this is handled by a render graph, transitioning intermediate resources.

                pushDescriptorHeapShaderData(pushData);
            }
            // Albedo-only draws push camera + albedo texture index.
            else if (draw.m_shaderVariant == ShaderVariant::Albedo) {

                util::require(draw.m_albedoTextureIndex < m_scene.m_textures.size(),
                              "Scene draw references an invalid albedo texture");

                const DescriptorHeapDrawPushIndicesAlbedo pushData{
                    .cameraIndex = frameIndex,
                    .albedoTextureIndex =
                        createCombinedImageSamplerIndex(draw.m_albedoTextureIndex, linearSamplerIndex),
                };

                pushDescriptorHeapShaderData(pushData);
            }
            // Solid color draws only need to push the camera index.
            else {
                util::require(draw.m_shaderVariant == ShaderVariant::SolidColor, "Unsupported shader variant");
                const DescriptorHeapDrawPushIndicesSolidColor pushData{
                    .cameraIndex = frameIndex,
                };
                pushDescriptorHeapShaderData(pushData);
            }
        }

        // Geometry binding still uses the normal Vulkan vertex path.
        // Descriptor heaps in this sample are only for shader resources such as buffers and textures.
        {

            beginDebugLabel(commandBuffer, "Bind Geometry", debugData.setupColor);

            // We have a single vertex buffer in this tutorial.
            // Most applications benefit from separating position and non-position data into different vertex buffers.
            const std::array vertexBuffers{*mesh.m_vertices.m_buffer};
            const std::array vertexOffsets{vk::DeviceSize{0}};

            const std::array vertexSizes{mesh.m_vertices.m_size};
            const std::array vertexStrides{vk::DeviceSize{sizeof(Vertex)}};

            constexpr uint32_t firstBinding = 0;

            // Small validation. Ensure the vertex and index buffer are valid.
            {
                util::require(vertexSizes.size() == vertexOffsets.size() &&
                                  vertexSizes.size() == vertexBuffers.size() &&
                                  vertexSizes.size() == vertexStrides.size(),
                              "Binding buffer information must have the same size");

                util::require(mesh.m_indexCount > 0,
                              std::format("Draw {} has a mesh without any indices", draw.m_debugName));

                for (uint32_t a = 0; a < vertexSizes.size(); ++a) {
                    util::require(vertexSizes[a] > 0,
                                  std::format("Draw {} has a mesh with an empty vertex buffer", draw.m_debugName));
                    util::require(
                        (vertexSizes[a] % vertexStrides[a]) == 0,
                        std::format("Draw {} has a mesh with an unaligned vertex buffer size", draw.m_debugName));
                }
            }

            commandBuffer.bindVertexBuffers2(firstBinding, vertexBuffers, vertexOffsets, vertexSizes, vertexStrides);

            // Bind the index buffer with an explicit byte range.
            constexpr vk::DeviceSize indexBufferOffset = 0;
            commandBuffer.bindIndexBuffer2(*mesh.m_indices.m_buffer, indexBufferOffset, mesh.m_indices.m_size,
                                           vk::IndexType::eUint32);

            // End Bind Geometry.
            endDebugLabel(commandBuffer);
        }

        // Issue the draw call to render the object.
        {
            // This draw call renders the current mesh and inputs and writes the output image
            // for this mesh to the color attachment.
            util::require(mesh.m_indexCount > 0, "Scene draw references an empty mesh");

            const std::uint32_t firstIndex = 0;
            const std::int32_t vertexOffset = 0;

            // We draw a single instance of each object, but the object index can be read in the shader as
            // SV_StartInstanceLocation.
            const std::uint32_t instanceCount = 1;
            const std::uint32_t firstInstance = draw.m_objectIndex;

            commandBuffer.drawIndexed(mesh.m_indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
        }

        // End Draw {draw.m_debugName}
        endDebugLabel(commandBuffer);
    }

    // End Scene Draws.
    endDebugLabel(commandBuffer);

    // End dynamic rendering.
    commandBuffer.endRendering();

    // Transition the swapchain image back for presentation.
    {
        beginDebugLabel(commandBuffer, "Transition Swapchain To Present", debugData.barrierColor);
        const ImageState presentState{
            .m_layout = vk::ImageLayout::ePresentSrcKHR,
            // There is no later shader/attachment stage for presentation. The render-finished semaphore
            // submitted below is what orders rendering before the presentation engine reads the image.
            .m_stageMask = vk::PipelineStageFlagBits2::eNone,
            .m_accessMask = vk::AccessFlagBits2::eNone,
            .m_aspectMask = vk::ImageAspectFlagBits::eColor,
        };
        transitionImage(commandBuffer, swapchainImage.m_image, swapchainImage.m_state, presentState,
                        CreateImageSubresourceRange());
        // End Transition Swapchain To Present.
        endDebugLabel(commandBuffer);
    }

    // End Rendering Frame {frameIndex} SwapchainImage {swapchainImageIndex}
    endDebugLabel(commandBuffer);
}

void Application::finishAndSubmitMainCommandBuffer(FrameInFlightResources& frame, std::uint32_t swapchainImageIndex)
{
    // Main synchronization objects:
    // - frame.m_imageAvailableSemaphore:
    //      This semaphore is signaled when we acquire a new swapchain image,
    //      indicating that the current frame can use the swapchain image to start rendering to it.
    //      The current frame waits on this semaphore before writing to the swapchain, so we ensure
    //      swapchainAcquireWaitStage=eColorAttachmentOutput happens after the acquire.
    //      The semaphore is signaled when the swapchain image is acquired.
    // - swapchainImage.m_renderFinishedSemaphore:
    //      This semaphore is signaled when the application finishes rendering.
    //      It indicates that the graphics queue finished work and the present queue can present the swapchain image.
    //      On the current frame, we signal this when the graphics queue work completes and the output is rendered.
    //      On the current frame, the present queue waits on this until the graphics work finishes
    //      and the swapchain image has been written.
    //      It is a per-swapchain-image resource and not a per-frame resource.
    //      If we reacquire the image and wait for the acquire semaphore, we know that the previous presentation of this
    //      swapchain image is done. The graphics queue waits on frame.m_imageAvailableSemaphore, and the present queue
    //      waits on the graphics queue, so we know that the acquire semaphore has been waited on before presenting.
    //      This allows us to safely reuse m_renderFinishedSemaphore, as Vulkan indicates that it is safe to reuse
    //      the present wait semaphore after the corresponding swapchain image has been acquired again and we have
    //      waited on the acquire semaphore or fence after that.
    // - frame.m_inFlightFence:
    //      Indicates that the current frame's graphics queue submission is done and that we finished rendering to the
    //      image. This indicates we can reuse per-frame resources like command buffers. In a more complex application
    //      this will also allow us to reuse per-frame intermediate buffers or images. The current frame signals this
    //      fence when the graphics queue finishes all commands. Before acquiring the swapchain image, we can wait for
    //      this fence to protect per-frame resources like the command buffer. We need to ensure the previous frame
    //      using the same frame-in-flight slot completed before recording new commands for this frame.
    //
    // Per-frame (frameIndex): command buffer, intermediate buffers and images (camera buffer, depth images),
    //            image-available semaphore, in-flight fence.
    // Per-swapchain image (swapchainImageIndex): swapchain image, render-finished semaphore.
    //
    // Note: It is important that we store the render-finished semaphore per swapchain and not per frame.
    //       The semaphore can be reused after acquiring the swapchain image, so it depends on the swapchain.
    //       https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
    //
    // Main synchronization operations:
    // - CPU waits for frame.m_inFlightFence: Previous graphics submission finished so we can use its resources.
    //       Per-frame resources like command buffer, camera buffer, and depth images can be reused.
    // - Acquire swapchain image.
    //    - Signal frame.m_imageAvailableSemaphore when the swapchain is acquired.
    // - Submit rendering work to the graphics queue.
    //    - GPU waits for frame.m_imageAvailableSemaphore.
    //      The GPU waits for the swapchain image to be available before executing
    //      swapchainAcquireWaitStage=eColorAttachmentOutput before writing to the swapchain image.
    //    - Signal swapchainImage.m_renderFinishedSemaphore when all commands are completed.
    //    - Signal frame.m_inFlightFence when all commands are completed.
    // - Submit present operation to the present queue:
    //    - GPU waits for swapchainImage.m_renderFinishedSemaphore before presentation.
    //      This indicates rendering is complete before presenting the swapchain.
    //
    // Main timeline:
    // - CPU: Wait for frame fence.
    // - CPU: Acquire swapchain image.
    // - CPU: Record rendering command buffer.
    // - CPU: Submit rendering command buffer to the graphics queue.
    // - CPU: Submit present operation to the present queue.
    // - GPU graphics queue: Start executing commands in rendering command buffer.
    // - GPU graphics queue: Before reaching color attachment output,
    //                       wait for semaphore to know swapchain is available.
    // - GPU graphics queue: Complete all rendering commands.
    // - GPU graphics queue: Signal semaphore and fence that rendering is complete.
    // - Present queue / presentation engine: Wait for the semaphore that indicates rendering is complete.
    // - Present queue / presentation engine: Present the swapchain image.
    //
    // Note: KHR_swapchain_maintenance1 can offer more granular synchronization operations.
    util::require(swapchainImageIndex < m_swapchainImages.size(), "Missing swapchain image");
    SwapchainImageResources& swapchainImage = m_swapchainImages[swapchainImageIndex];

    // We end the command buffer before submitting it. This indicates all commands have been recorded.
    frame.m_commandBuffer.end();

    // Submit rendering operations to the graphics queue.
    {
        const std::array commandBufferInfos = {vk::CommandBufferSubmitInfo{
            .commandBuffer = frame.m_commandBuffer,
        }};

        // Wait until the acquired swapchain image is available before color attachment output.
        // The queue submission will use this list to wait for certain operations.
        // Wait semaphores are used by the queue to wait for semaphore signal operations.
        // In our case, commands in swapchainAcquireWaitStage=eColorAttachmentOutput will not execute
        // until imageAvailableSemaphore is signaled.
        // Notice that this only affects stages in the stageMask, and will not block other stages.
        const std::array waitInfos = {vk::SemaphoreSubmitInfo{
            .semaphore = *frame.m_imageAvailableSemaphore,
            .stageMask = swapchainAcquireWaitStage,
        }};

        // Signal after all recorded work.
        // In our case the submission signals m_renderFinishedSemaphore once all commands are done.
        // This includes transitioning the image to the present layout.
        // Present waits on this semaphore before the presentation engine reads the image.
        const std::array signalInfos = {vk::SemaphoreSubmitInfo{
            .semaphore = *swapchainImage.m_renderFinishedSemaphore,
            .stageMask = vk::PipelineStageFlagBits2::eAllCommands,
        }};

        const vk::SubmitInfo2 submitInfo{
            .waitSemaphoreInfoCount = util::safeCastToU32(waitInfos.size()),
            .pWaitSemaphoreInfos = waitInfos.data(),
            .commandBufferInfoCount = util::safeCastToU32(commandBufferInfos.size()),
            .pCommandBufferInfos = commandBufferInfos.data(),
            .signalSemaphoreInfoCount = util::safeCastToU32(signalInfos.size()),
            .pSignalSemaphoreInfos = signalInfos.data(),
        };

        // Reset frame fence before submission.
        {
            // Reset the fence before submit so this submission can signal it.
            const std::array frameFences{*frame.m_inFlightFence};
            m_logicalDevice.resetFences(frameFences);
        }

        // The frame fence is signaled when the GPU finishes this submitted work.
        m_graphicsQueue.submit2(submitInfo, *frame.m_inFlightFence);
    }

    // Submit present operation to a present queue.
    {
        // Note: When creating the device, we check that the graphics and present queues have the same queue family.
        //       This ensures that we do not need to create barriers to transfer family ownership.

        // Present waits on the per-image render-finished semaphore, then displays the image.
        const std::array waitSemaphores{*swapchainImage.m_renderFinishedSemaphore};

        const std::array swapchains{vk::SwapchainKHR{m_vkbData.m_swapchain.swapchain}};
        const std::array imageIndices{swapchainImageIndex};

        const vk::PresentInfoKHR presentInfo{
            .waitSemaphoreCount = util::safeCastToU32(waitSemaphores.size()),
            .pWaitSemaphores = waitSemaphores.data(),
            .swapchainCount = util::safeCastToU32(swapchains.size()),
            .pSwapchains = swapchains.data(),
            .pImageIndices = imageIndices.data(),
        };
        const vk::Result presentResult = m_presentQueue.presentKHR(presentInfo);
        util::require(presentResult == vk::Result::eSuccess || presentResult == vk::Result::eSuboptimalKHR,
                      "Failed to present swapchain image");
    }
}

//****************
// Helper command buffer utils
//****************

vk::CommandBuffer Application::getHelperCommandBuffer() const
{
    // Our application has a helper command buffer.
    // This command buffer is only used during initialization to move resources on the GPU.
    //
    // This extra command buffer is only for setup-time uploads and layout transitions.
    // Keeping it separate from frame command buffers makes initialization easier to explain:
    // record all one-time copy work, submit once, wait once, then start rendering later.

    // Wrapper that gets the helper command buffer and checks that it is valid.
    // The objective is to check that the command buffer exists and that it is active (we called beginHelperCommands).

    // Check that the helper command buffer exists.
    util::require(m_helperCommandBuffer.m_commandBuffer, "Helper command buffer has not been allocated");

    // Check that the command buffer is active and ready to record commands.
    // Note: This is an application guard.
    util::require(m_helperCommandBuffer.m_active, "Helper command buffer must be active before recording uploads");

    // Return the command buffer so that it can be used by the application.
    return m_helperCommandBuffer.m_commandBuffer;
}

void Application::beginHelperCommands()
{
    // Small wrapper to activate the helper command buffer.
    // This function resets and begins the helper command buffer, making it ready to start recording.

    // Check that the command buffer exists.
    util::require(m_helperCommandBuffer.m_commandBuffer, "Helper command buffer has not been allocated");

    // We are activating the command buffer. We need to check that it is not already active.
    util::require(!m_helperCommandBuffer.m_active, "Helper command buffer is already active");

    m_helperCommandBuffer.m_commandBuffer.reset();
    m_helperCommandBuffer.m_commandBuffer.begin(
        vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Set a debug label to easily identify this command buffer.
    beginDebugLabel(m_helperCommandBuffer.m_commandBuffer, "Setup Uploads", debugData.setupColor);

    // Mark active so upload helpers can verify that setup commands are being recorded.
    m_helperCommandBuffer.m_active = true;
}

void Application::endHelperCommandsAndFlushUploads()
{
    // Submit the batched initialization work and wait for it to finish before we free staging buffers.
    // This is intentionally conservative and beginner-friendly.
    //
    // A larger renderer would usually overlap upload work with other GPU work instead of calling waitIdle() here.
    const vk::CommandBuffer commandBuffer = getHelperCommandBuffer();

    // End recording the helper command buffer.

    // End Setup Uploads.
    endDebugLabel(commandBuffer);

    // Call end to stop recording commands in the command buffer.
    commandBuffer.end();

    // Submit the helper command buffer and wait for completion.
    {
        const std::array commandBufferInfo = {vk::CommandBufferSubmitInfo{.commandBuffer = commandBuffer}};
        const vk::SubmitInfo2 submitInfo{
            .commandBufferInfoCount = util::safeCastToU32(commandBufferInfo.size()),
            .pCommandBufferInfos = commandBufferInfo.data(),
        };

        // Use a null handle for our fence. waitIdle handles command buffer completion.
        vk::Fence nullFence{};

        m_graphicsQueue.submit2(submitInfo, nullFence);

        // Calling wait idle is not very efficient. It will probably cause CPU idle time.
        // When creating the helper command buffer we could create a helper fence, and use it here to wait,
        // but waiting immediately after submission is not recommended,
        // and will also cause CPU idle time.
        // This is not very relevant since it is only called during initialization, and will not cause stutters.
        // A proper application should defer waiting to schedule CPU work during GPU initialization.
        m_graphicsQueue.waitIdle();
    }

    // Reset helper command buffer state.
    {
        m_helperCommandBuffer.m_commandBuffer.reset();
        // Mark helper command buffer as inactive.
        m_helperCommandBuffer.m_active = false;
    }

    // Staging buffers were being kept alive until completion and can now be deleted.
    {
        // Staging buffers are only needed until their data has been copied to the correct GPU buffers.
        // After the data has been copied, they can be deleted to free resources.
        // We need to keep track of these buffers to delete them after the submitted commands complete.
        // Note: this has a very simple strategy for teaching and learning purposes.
        for (GpuBuffer& stagingBuffer : m_pendingUploadStagingBuffers) {
            stagingBuffer.destroy(m_logicalDevice);
        }
        m_pendingUploadStagingBuffers.clear();
    }
}

//****************
// Destruction and cleanup
//****************

Application::~Application()
{
    // When destroying the application, call cleanup to destroy Vulkan and other resources.
    cleanup();
}

void Application::cleanup()
{
    // This function is called when the application terminates to destroy all resources.
    // Note: we are assuming no errors during destruction.

    // Wait before destroying resources that the GPU may still be using.
    if (m_logicalDevice) {
        // We are using a simple path in this tutorial.
        // wait idle is the easiest way to ensure there is no outstanding work on the GPU queues.
        // We are going to destroy buffers and images so we need to ensure that there is no work on the GPU using them.
        //
        // Note: consider using KHR_swapchain_maintenance1 for better swapchain resource management.
        m_logicalDevice.waitIdle();
    }
    // Destroy Vulkan resources.
    {
        // Vulkan is a low-level API so we need to manually clean up our resources.
        // Vulkan-Hpp greatly helps manage resource destruction as it offers RAII wrappers.
        // Most of our Vulkan objects are owned by vk::Unique* Vulkan-Hpp wrappers like vk::UniqueSemaphore.
        // These wrappers will call Vulkan destroy functions like vkDestroySemaphore in their destructor.
        // Notice that we need to destroy and free our Vulkan objects in dependency order.
        // For example, we cannot destroy a device until we destroy all objects like buffers created by the device.
        // Vulkan-Hpp also offers vk::raii wrappers as a higher-level RAII wrapper.

        m_shaderObjects.m_solidColor.m_fragment.reset();
        m_shaderObjects.m_solidColor.m_vertex.reset();
        m_shaderObjects.m_albedo.m_fragment.reset();
        m_shaderObjects.m_albedo.m_vertex.reset();
        m_shaderObjects.m_albedoAndNormal.m_fragment.reset();
        m_shaderObjects.m_albedoAndNormal.m_vertex.reset();

        m_descriptorHeaps.m_samplerHeap.m_buffer.destroy(m_logicalDevice);
        m_descriptorHeaps.m_resourceHeap.m_buffer.destroy(m_logicalDevice);
        m_descriptorHeaps = {};

        for (GpuImage& image : m_scene.m_textures) {
            image.destroy();
        }
        m_scene.m_textures.clear();

        for (GpuMesh& mesh : m_scene.m_meshes) {
            mesh.m_indices.destroy(m_logicalDevice);
            mesh.m_vertices.destroy(m_logicalDevice);
        }
        m_scene.m_meshes.clear();
        m_scene.m_objects.destroy(m_logicalDevice);
        m_scene.m_pointLight.destroy(m_logicalDevice);
        m_scene.m_solidColor.destroy(m_logicalDevice);
        m_scene = {};

        for (GpuBuffer& stagingBuffer : m_pendingUploadStagingBuffers) {
            stagingBuffer.destroy(m_logicalDevice);
        }
        m_pendingUploadStagingBuffers.clear();

        // Command buffers are owned by the command pool, so clear the non-owning handles first.
        m_helperCommandBuffer = {};

        for (FrameInFlightResources& frame : m_framesInFlight) {
            frame.m_camera.destroy(m_logicalDevice);
            frame.m_commandBuffer = nullptr;
            frame.m_inFlightFence.reset();
            frame.m_imageAvailableSemaphore.reset();

            frame.m_depthImage.destroy();
        }

        // Destroying the command pool frees all command buffers allocated from it.
        m_commandPool.reset();

        // Clearing the vector destroys image views and per-image semaphores.
        m_swapchainImages.clear();
    }

    // Destroy Vulkan bootstrap resources.
    {
        if (m_vkbData.m_swapchain.swapchain != VK_NULL_HANDLE) {
            // Note: Swapchain images are owned by the swapchain. We destroy app images, but not the swapchain images.
            //       To destroy swapchain images we have to destroy the swapchain.
            vkb::destroy_swapchain(m_vkbData.m_swapchain);
            m_vkbData.m_swapchain = {};
        }

        // Destroy in dependency order, which is usually reverse creation order:
        // We need to destroy Vulkan objects in dependency order.
        // We cannot destroy an instance if the device still exists.
        if (m_vkbData.m_device.device != VK_NULL_HANDLE) {
            vkb::destroy_device(m_vkbData.m_device);
            m_vkbData.m_device = {};
            m_logicalDevice = nullptr;
        }

        if (m_surface) {
            vkb::destroy_surface(m_vkbData.m_instance, static_cast<VkSurfaceKHR>(m_surface));
            m_surface = nullptr;
        }

        if (m_vkbData.m_instance.instance != VK_NULL_HANDLE) {
            vkb::destroy_instance(m_vkbData.m_instance);
            m_vkbData.m_instance = {};
        }
    }

    // Destroy and terminate GLFW.
    {
        if (m_window != nullptr) {
            glfwDestroyWindow(m_window);
            m_window = nullptr;
        }
        glfwTerminate();
    }
}

//****************
// Resource utils
//****************

std::uint32_t Application::findMemoryType(std::uint32_t typeBits, vk::MemoryPropertyFlags properties) const
{
    // Find a valid memory type index to allocate memory for a buffer or image.
    // m_memoryTypeFlags was cached from vk::PhysicalDeviceMemoryProperties during device initialization.
    //
    // Iterate through the cached memory flags and select the first compatible one.
    for (std::uint32_t i = 0; i < m_memoryTypeFlags.size(); ++i) {
        const bool typeMatches = (typeBits & (1U << i)) != 0U;
        const bool flagsMatch = (m_memoryTypeFlags[i] & properties) == properties;
        if (typeMatches && flagsMatch) {
            return i;
        }
    }

    throw std::runtime_error("No compatible Vulkan memory type found");
}

Application::GpuBuffer Application::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                 vk::MemoryPropertyFlags memoryProperties,
                                                 std::string_view debugName) const
{
    // This function creates a buffer.
    // Buffers in Vulkan store linear data. They are commonly used to store scalars, arrays, or matrices.

    util::require(size > 0, "Vulkan requires buffer size to be greater than 0.");
    util::require(usage != vk::BufferUsageFlags{}, "Trying to create a buffer without any usage flags.");

    GpuBuffer buffer{.m_size = size};

    // Create the buffer object.
    {
        const vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };
        buffer.m_buffer = m_logicalDevice.createBufferUnique(bufferInfo);
    }

    // Name resources like buffers to help when debugging in tools like RenderDoc.
    setDebugName(*buffer.m_buffer, vk::ObjectType::eBuffer, std::string{debugName});

    // Only buffers created with eShaderDeviceAddress receive a GPU address. That keeps staging,
    // vertex, and index buffers simple while still supporting descriptor heap buffer descriptors.
    // Note: we cache the GPU address of all buffers that have one.
    const bool needsGpuAddress = (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) != vk::BufferUsageFlags{};

    // Allocate memory and bind it to the buffer.
    {
        // Use the created buffer object to query the memory requirements.
        const vk::BufferMemoryRequirementsInfo2 requirementsInfo{
            .buffer = *buffer.m_buffer,
        };
        const vk::MemoryRequirements2 requirements = m_logicalDevice.getBufferMemoryRequirements2(requirementsInfo);

        // Set allocation flags, enabling support for device addresses if needed.
        const vk::MemoryAllocateFlagsInfo allocateFlags{
            .flags = needsGpuAddress ? vk::MemoryAllocateFlagBits::eDeviceAddress : vk::MemoryAllocateFlags{},
        };

        // Use the queried requirements to allocate the buffer.
        const vk::MemoryAllocateInfo allocateInfo{
            .pNext = (allocateFlags.flags != vk::MemoryAllocateFlags{}) ? &allocateFlags : nullptr,
            .allocationSize = requirements.memoryRequirements.size,
            .memoryTypeIndex = findMemoryType(requirements.memoryRequirements.memoryTypeBits, memoryProperties),
        };

        // We allocate one memory block per buffer for simplicity.
        // Real applications suballocate from larger memory blocks to reduce overhead and fragmentation.
        // Consider using a library for memory management such as VMA instead.
        buffer.m_memory = m_logicalDevice.allocateMemoryUnique(allocateInfo);

        // We bind the memory to the buffer.
        // We created the buffer and allocated memory for the buffer.
        // This binds and links this memory to the buffer.
        const std::array bindInfos{vk::BindBufferMemoryInfo{
            .buffer = *buffer.m_buffer,
            .memory = *buffer.m_memory,
            .memoryOffset = 0,
        }};
        m_logicalDevice.bindBufferMemory2(bindInfos);
    }

    // Retrieve the GPU address.
    if (needsGpuAddress) {
        const vk::BufferDeviceAddressInfo addressInfo{.buffer = *buffer.m_buffer};
        buffer.m_addressGPU = m_logicalDevice.getBufferAddress(addressInfo);
        util::require(buffer.m_addressGPU != 0, "Failed to get buffer GPU address");
    }

    // Host-visible memory can be accessed by the CPU.
    // We keep those buffers persistently mapped so setup code can upload data with a simple memcpy.
    // This is a simple tutorial, so we avoid flush and invalidate operations by using eHostCoherent.
    const bool hostVisible = (memoryProperties & vk::MemoryPropertyFlagBits::eHostVisible) != vk::MemoryPropertyFlags{};

    if (hostVisible) {
        // We map the buffer and always keep it mapped.
        buffer.m_addressCPU = m_logicalDevice.mapMemory(*buffer.m_memory, 0, size);

        util::require(buffer.m_addressCPU != nullptr, "Failed to map buffer CPU address");

        util::require((memoryProperties & vk::MemoryPropertyFlagBits::eHostCoherent) ==
                          vk::MemoryPropertyFlagBits::eHostCoherent,
                      "Mapped buffers in this tutorial must be host coherent to avoid manual flushes");
    }

    return buffer;
}

Application::GpuBuffer Application::uploadToNewStagingBuffer(std::span<const std::byte> data,
                                                             std::string_view debugName) const
{
    // Staging buffers are temporary buffers used to transfer data from the CPU to the GPU.
    // They let us use a CPU-visible buffer before uploading data to a GPU-only buffer.

    // This is a tutorial, so they are created on demand for each upload.
    // Real applications will reuse them and suballocate from them to reduce overhead.

    // Staging buffers are host coherent to make CPU-to-GPU uploads simple.
    const vk::MemoryPropertyFlags stagingMemoryProperties =
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

    GpuBuffer stagingBuffer = createBuffer(static_cast<vk::DeviceSize>(data.size()),
                                           vk::BufferUsageFlagBits::eTransferSrc, stagingMemoryProperties, debugName);

    uploadBuffer(stagingBuffer, data);
    return stagingBuffer;
}

Application::GpuBuffer Application::uploadToNewGpuBuffer(std::span<const std::byte> data,
                                                         vk::BufferUsageFlags finalUsage, std::string_view debugName)
{
    // This function creates a GPU buffer and initializes it by copying data into it.

    // We use the helper command buffer to record copies to the buffer.
    const vk::CommandBuffer commandBuffer = getHelperCommandBuffer();

    // Static GPU data is uploaded through a temporary host-visible staging buffer,
    // then kept in device-local memory for rendering.

    // Allocate the staging buffer and upload data to it.
    const std::string stagingName = std::format("{} Staging", debugName);
    GpuBuffer stagingBuffer = uploadToNewStagingBuffer(data, stagingName);

    // Allocate GPU-only buffer.
    GpuBuffer gpuBuffer = createBuffer(static_cast<vk::DeviceSize>(data.size()),
                                       // We add eTransferDst usage to allow copying data to the buffer.
                                       finalUsage | vk::BufferUsageFlagBits::eTransferDst,
                                       // Mark it device local: GPU-only buffers are usually faster to access.
                                       vk::MemoryPropertyFlagBits::eDeviceLocal, debugName);

    // Add a debug label to help identify how the buffer is set up.
    const std::string uploadDebugLabel = std::format("Upload Buffer: {}", debugName);
    beginDebugLabel(commandBuffer, uploadDebugLabel, debugData.transferColor);

    // Copy staging buffer data to the GPU buffer using the active helper command buffer.
    {
        const std::array copyRegion = {vk::BufferCopy2{
            // Offsets are 0 since we copy the entire buffer.
            .srcOffset = 0,
            .dstOffset = 0,
            .size = static_cast<vk::DeviceSize>(data.size()),
        }};
        const vk::CopyBufferInfo2 copyInfo{
            .srcBuffer = *stagingBuffer.m_buffer,
            .dstBuffer = *gpuBuffer.m_buffer,
            .regionCount = util::safeCastToU32(copyRegion.size()),
            .pRegions = copyRegion.data(),
        };
        commandBuffer.copyBuffer2(copyInfo);
    }

    // Buffer memory barrier to ensure GPU buffer data is available.
    {
        const std::array uploadBarrier = {vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = *gpuBuffer.m_buffer,
            .offset = 0,
            .size = static_cast<vk::DeviceSize>(data.size()),
        }};
        const vk::DependencyInfo dependencyInfo{
            .bufferMemoryBarrierCount = util::safeCastToU32(uploadBarrier.size()),
            .pBufferMemoryBarriers = uploadBarrier.data(),
        };
        commandBuffer.pipelineBarrier2(dependencyInfo);
    }
    // End Upload Buffer.
    endDebugLabel(commandBuffer);

    // The GPU buffer is stored by the application, but staging buffers need to be alive until
    // the submitted GPU work that references them is done.
    // The tutorial keeps track of all staging buffers.
    // They will be freed once the helper command buffer finishes execution.
    m_pendingUploadStagingBuffers.push_back(std::move(stagingBuffer));
    return gpuBuffer;
}

void Application::uploadBuffer(const GpuBuffer& buffer, std::span<const std::byte> data) const
{
    // Mapped buffers uploaded by this tutorial use coherent host-visible memory, so memcpy is enough.
    // This allows us to make things simple and avoid invalidation and flushing.
    //
    // With coherent memory we still need to ensure the CPU does not overwrite data while the GPU reads it.
    // Staging buffers are written before submission.
    // Per-frame buffers are protected by a fence before submission.

    util::require(buffer.m_addressCPU != nullptr, "Buffer must be persistently mapped for CPU upload");
    util::require(data.size() <= buffer.m_size, "Upload is larger than destination buffer");

    std::memcpy(buffer.m_addressCPU, data.data(), data.size());
}

void Application::allocateGpuImage(GpuImage& image) const
{
    // Consider using a library like VMA to allocate and suballocate memory for images and buffers.
    util::require(static_cast<bool>(image.m_image), "GpuImage must exist before allocating memory");

    const vk::ImageMemoryRequirementsInfo2 requirementsInfo{
        .image = *image.m_image,
    };
    const vk::MemoryRequirements2 requirements = m_logicalDevice.getImageMemoryRequirements2(requirementsInfo);
    const vk::MemoryAllocateInfo allocateInfo{
        .allocationSize = requirements.memoryRequirements.size,
        .memoryTypeIndex =
            findMemoryType(requirements.memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    };
    image.m_memory = m_logicalDevice.allocateMemoryUnique(allocateInfo);

    const std::array bindInfos{vk::BindImageMemoryInfo{
        .image = *image.m_image,
        .memory = *image.m_memory,
        .memoryOffset = 0,
    }};
    m_logicalDevice.bindImageMemory2(bindInfos);
}

Application::GpuImage Application::createTexture(const util::ImageRgba8& sourceImage, std::string_view debugName)
{
    // This function creates a texture.
    // This creates a read-only texture image.

    util::require(sourceImage.m_width > 0 && sourceImage.m_height > 0,
                  "Texture images must have non-zero width and height");
    util::require(static_cast<std::size_t>(sourceImage.m_height) * static_cast<std::size_t>(sourceImage.m_width) ==
                      sourceImage.m_pixels.size(),
                  "Invalid image size.");

    // We create a GPU image to store our texture.
    GpuImage image;
    image.m_mipLevels = static_cast<std::uint32_t>(
        std::floor(std::log2(static_cast<float>(std::max(sourceImage.m_width, sourceImage.m_height)))) + 1.0F);
    image.m_states.resize(image.m_mipLevels, ImageState{.m_aspectMask = vk::ImageAspectFlagBits::eColor});

    // Create the GPU image object.
    {
        const vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = mainTextureFormat,
            .extent = vk::Extent3D{.width = sourceImage.m_width,
                                   .height = sourceImage.m_height,
                                   // Vulkan supports 3D images, but this tutorial only uses 2D, so depth is 1.
                                   .depth = 1},
            // Images are stored using mipmap levels with prefiltered versions of the image at smaller resolutions.
            // This improves memory access and allows texture interpolation for better image quality.
            .mipLevels = image.m_mipLevels,
            // Use more than 1 for texture arrays and cube maps.
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            // Avoid linear tiling.
            // Optimal tiling makes memory layout opaque but it enables proper access for sampling.
            .tiling = vk::ImageTiling::eOptimal,
            // Selecting the correct usage flags can have performance implications.
            // Only select the ones you need. For example, avoid eStorage if the image is not GPU written.
            .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eSampled,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        image.m_image = m_logicalDevice.createImageUnique(imageInfo);
    }
    setDebugName(*image.m_image, vk::ObjectType::eImage, std::string{debugName});

    // Allocate memory and bind it to the image.
    allocateGpuImage(image);

    // Upload the image from CPU memory to GPU memory, and generate mip levels.
    {
        // Create a temporary staging buffer that is visible to the CPU and copy the image there.
        // This tutorial creates a new staging buffer for each image so the upload path stays explicit.
        const std::string stagingName = std::format("{} Staging", debugName);
        GpuBuffer stagingBuffer = uploadToNewStagingBuffer(std::as_bytes(std::span{sourceImage.m_pixels}), stagingName);

        // Record setup-time copy work into the active helper command buffer.
        const vk::CommandBuffer commandBuffer = getHelperCommandBuffer();

        // Add a debug label to identify this work.
        const std::string uploadDebugLabel =
            debugName.empty() ? std::string{"Upload Texture"} : std::format("Upload Texture: {}", debugName);
        beginDebugLabel(commandBuffer, uploadDebugLabel, debugData.transferColor);

        const ImageState transferDstState{
            .m_layout = vk::ImageLayout::eTransferDstOptimal,
            .m_stageMask = vk::PipelineStageFlagBits2::eTransfer,
            .m_accessMask = vk::AccessFlagBits2::eTransferWrite,
            .m_aspectMask = vk::ImageAspectFlagBits::eColor,
        };
        const ImageState transferSrcState{
            .m_layout = vk::ImageLayout::eTransferSrcOptimal,
            .m_stageMask = vk::PipelineStageFlagBits2::eTransfer,
            .m_accessMask = vk::AccessFlagBits2::eTransferRead,
            .m_aspectMask = vk::ImageAspectFlagBits::eColor,
        };
        const ImageState shaderReadState{
            .m_layout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .m_stageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .m_accessMask = vk::AccessFlagBits2::eShaderSampledRead,
            .m_aspectMask = vk::ImageAspectFlagBits::eColor,
        };

        // Mip generation still starts with one whole-image transition.
        // Mip 0 can receive the upload.
        // Higher mips can start as blit destinations.
        image.transition(commandBuffer, transferDstState);

        // Copy the staging buffer into the image at mip 0.
        {
            const std::array copyRegion = {vk::BufferImageCopy2{
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource =
                    vk::ImageSubresourceLayers{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                .imageOffset = vk::Offset3D{.x = 0, .y = 0, .z = 0},
                .imageExtent = vk::Extent3D{.width = sourceImage.m_width, .height = sourceImage.m_height, .depth = 1},
            }};
            const vk::CopyBufferToImageInfo2 copyInfo{
                .srcBuffer = *stagingBuffer.m_buffer,
                .dstImage = *image.m_image,
                .dstImageLayout = transferDstState.m_layout,
                .regionCount = util::safeCastToU32(copyRegion.size()),
                .pRegions = copyRegion.data(),
            };
            commandBuffer.copyBufferToImage2(copyInfo);
        }

        // Use blit operations to generate all remaining mip levels for this image.
        //
        // The idea is to copy the previous mip level to the current mip level,
        // using a blit operation to resize the image.
        //
        // Mip levels let sampling choose a lower-resolution version of a texture when appropriate,
        // improving performance and reducing shimmering. This improves the application's visual quality.
        // We generate the mip levels by blitting down the image.
        {
            std::int32_t sourceWidth = util::safeCastTo<std::int32_t>(sourceImage.m_width);
            std::int32_t sourceHeight = util::safeCastTo<std::int32_t>(sourceImage.m_height);

            // Iterate through all mip levels.
            for (std::uint32_t mipLevel = 1; mipLevel < image.m_mipLevels; ++mipLevel) {
                // Transition the previous mip level from transfer dst to transfer src.
                // Note: blit is a transfer operation.
                image.transition(commandBuffer, transferSrcState, mipLevel - 1, 1);

                // Half the resolution of the previous mip level.
                const std::int32_t destinationWidth = std::max(1, sourceWidth / 2);
                const std::int32_t destinationHeight = std::max(1, sourceHeight / 2);

                // Issue a blit command to generate this level from the previous level.
                // A blit is a copy-and-resize operation.
                // Note: During device creation we ensure support for linear blit operations on this format.
                {
                    const vk::ImageBlit2 blitRegion{
                        .srcSubresource =
                            vk::ImageSubresourceLayers{
                                .aspectMask = vk::ImageAspectFlagBits::eColor,
                                .mipLevel = mipLevel - 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                            },
                        .srcOffsets = std::array{vk::Offset3D{.x = 0, .y = 0, .z = 0},
                                                 vk::Offset3D{.x = sourceWidth, .y = sourceHeight, .z = 1}},
                        .dstSubresource =
                            vk::ImageSubresourceLayers{
                                .aspectMask = vk::ImageAspectFlagBits::eColor,
                                .mipLevel = mipLevel,
                                .baseArrayLayer = 0,
                                .layerCount = 1,
                            },
                        .dstOffsets = std::array{vk::Offset3D{.x = 0, .y = 0, .z = 0},
                                                 vk::Offset3D{.x = destinationWidth, .y = destinationHeight, .z = 1}},
                    };
                    // These layout fields do not transition the image. They must match the layouts produced by the
                    // surrounding barriers, so we keep them sourced from the same states used for image transition.
                    const vk::BlitImageInfo2 blitInfo{
                        .srcImage = *image.m_image,
                        .srcImageLayout = transferSrcState.m_layout,
                        .dstImage = *image.m_image,
                        .dstImageLayout = transferDstState.m_layout,
                        .regionCount = 1,
                        .pRegions = &blitRegion,
                        .filter = vk::Filter::eLinear,
                    };
                    commandBuffer.blitImage2(blitInfo);
                }

                // Update current level dimensions.
                sourceWidth = destinationWidth;
                sourceHeight = destinationHeight;
            }

            image.transition(commandBuffer, shaderReadState);
        }
        // End Upload Texture.
        endDebugLabel(commandBuffer);

        // Keep the staging buffer alive until the batched helper command buffer has finished.
        m_pendingUploadStagingBuffers.push_back(std::move(stagingBuffer));
    }
    return image;
}

// Helper to set debug names on Vulkan objects.
// Debug names improve debugging in tools like RenderDoc and Nsight Graphics.
// They also make most validation-layer messages easier to interpret.
template <typename Handle>
void Application::setDebugName(Handle handle, vk::ObjectType objectType, const std::string& name) const
{
    // Return early if GPU debugging is disabled.
    if (!debugData.enableGpuDebug) {
        return;
    }

    util::require(static_cast<bool>(m_logicalDevice), "Debug naming requires a valid Vulkan device");
    util::require(!name.empty(), "Debug empty names are allowed by Vulkan, but not by the application");

    using RawHandle = typename Handle::CType;
    const RawHandle rawHandle = static_cast<RawHandle>(handle);
    util::require(rawHandle != VK_NULL_HANDLE, "Debug naming requires a valid Vulkan object handle");

    const vk::DebugUtilsObjectNameInfoEXT nameInfo{
        .objectType = objectType,
        .objectHandle = util::rawHandleToUint64(rawHandle),
        // Vulkan expects pObjectName to be null-terminated; std::string::c_str() guarantees that.
        .pObjectName = name.c_str(),
    };
    static_cast<void>(m_logicalDevice.setDebugUtilsObjectNameEXT(nameInfo));
}

void Application::beginDebugLabel(vk::CommandBuffer commandBuffer, const std::string& name,
                                  const std::array<float, 4>& color)
{
    // Return early if GPU debugging is disabled.
    if (!debugData.enableGpuDebug) {
        return;
    }

    const vk::DebugUtilsLabelEXT label{
        .pLabelName = name.c_str(),
        .color = color,
    };

    // We assume command buffer is recording
    commandBuffer.beginDebugUtilsLabelEXT(label);
}

void Application::endDebugLabel(vk::CommandBuffer commandBuffer)
{
    // Return early if GPU debugging is disabled.
    if constexpr (!debugData.enableGpuDebug) {
        return;
    }

    commandBuffer.endDebugUtilsLabelEXT();
}

// Helper function that records a barrier to transition one image range from one tracked state to another.
void Application::transitionImage(vk::CommandBuffer commandBuffer, vk::Image image, ImageState& currentState,
                                  const ImageState& newState, vk::ImageSubresourceRange subresourceRange)
{
    // This tutorial uses a simple implementation of how to track resources for synchronization, allowing us to create
    // automatic barriers.
    // A proper Render Hardware Interface (RHI) would use this function to automatically synchronize resources.
    //
    // Our implementation is very simple. For a production ready approach check:
    // - Vulkanised 2024: Vulkan Synchronization for WebGPU - https://www.youtube.com/watch?v=SH0N4QmioUw
    // - Vulkanised 2024: Vulkan Synchronization Made Easy - https://www.youtube.com/watch?v=d15RXWp1Rqo
    //
    // Many renderers implement a render graph or task graph.
    // A render graph is capable of creating barriers and other synchronization primitives automatically.
    // They can reorder work and passes to increase GPU overlap and avoid idle time on the GPU. Advanced
    // implementations can also reuse resources to reduce memory consumption. They are the standard solution
    // in the industry, so we recommend most developers study render graphs next.
    //
    // ImageState is tutorial-side bookkeeping: Vulkan tracks real image layouts during execution, but the app still
    // needs to remember the layout/access/stage it last requested so it can build the next correct barrier.

    // Skip only exact duplicate read-after-read barriers.
    // Widening or narrowing a read scope later would require remembering the earlier write that made the image contents
    // visible, which this small tutorial tracker intentionally does not keep.
    {
        const auto onlyRead = [](const ImageState& state) {
            const vk::AccessFlags2 writeAccessMask =
                vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eColorAttachmentWrite |
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eTransferWrite |
                vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eMemoryWrite;
            return static_cast<bool>(state.m_accessMask) && !static_cast<bool>(state.m_accessMask & writeAccessMask);
        };

        const bool duplicateReadOnlyState = currentState == newState && onlyRead(currentState);

        // Limited example of removing redundant read-only barriers.
        // We can improve this with a more complex state tracker and better heuristics.
        // Instead of per-resource state tracking, a lot of games implement a render-graph.
        // Render graphs allow reordering commands to increase overlap and reduce synchronization overhead.
        if (duplicateReadOnlyState) {
            return;
        }
    }

    // Vulkan requires images to be in a layout that matches their current use. This barrier
    // both changes the layout and describes which earlier writes must be visible to later work.
    const std::array barriers = {vk::ImageMemoryBarrier2{
        .srcStageMask = currentState.m_stageMask,
        .srcAccessMask = currentState.m_accessMask,
        .dstStageMask = newState.m_stageMask,
        .dstAccessMask = newState.m_accessMask,
        .oldLayout = currentState.m_layout,
        .newLayout = newState.m_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = subresourceRange,
    }};

    // Synchronization2 groups barriers in a DependencyInfo structure before recording them.
    const vk::DependencyInfo dependencyInfo{
        .imageMemoryBarrierCount = util::safeCastToU32(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    };
    commandBuffer.pipelineBarrier2(dependencyInfo);

    // Update the tracked state for this image range to the new state requested by this barrier.
    currentState = newState;
}

//****************
// GpuBuffer
//****************

void Application::GpuBuffer::destroy(vk::Device device)
{
    // Free GPU buffer and associated resources

    if (m_addressCPU != nullptr) {
        device.unmapMemory(*m_memory);
        m_addressCPU = nullptr;
    }
    m_buffer.reset();
    m_memory.reset();
    m_addressGPU = {};
    m_size = {};
}

//****************
// GpuImage
//****************

void Application::GpuImage::destroy()
{
    // Free GPU image and associated resources

    m_image.reset();
    m_memory.reset();
    m_mipLevels = 1;
    m_states.clear();
}

void Application::GpuImage::transition(vk::CommandBuffer commandBuffer, const ImageState& newState)
{
    // Wrapper to transition all mip levels at once
    transition(commandBuffer, newState, 0, m_mipLevels);
}

void Application::GpuImage::transition(vk::CommandBuffer commandBuffer, const ImageState& newState,
                                       std::uint32_t baseMipLevel, std::uint32_t mipLevelCount)
{
    // Wrapper to call transitionImage and change and synchronize an image
    // This will record and transition mip levels from baseMipLevel to mipLevelCount to new state

    // ImageState is not a Vulkan handle.
    // It is this tutorial's small state tracker so each transition can use the previous layout/access/stage values as
    // the source of the next barrier.

    // Small validation
    {
        util::require(static_cast<bool>(m_image), "GpuImage must exist before transitioning it");
        util::require(!m_states.empty(), "GpuImage must track at least one image state");
        util::require(mipLevelCount > 0, "GpuImage transition must cover at least one mip level");
        util::require(baseMipLevel < m_states.size(), "GpuImage transition base mip level is out of range");
        util::require((baseMipLevel + mipLevelCount) <= m_states.size(),
                      "GpuImage transition range exceeds tracked mip levels");
    }

    const std::uint32_t endMipLevel = baseMipLevel + mipLevelCount;

    // Transition images from runBaseMipLevel to runBaseMipLevel+runMipLevelCount
    // This is done in a single barrier.
    const auto transitionRun = [&](const std::uint32_t runBaseMipLevel, const std::uint32_t runMipLevelCount,
                                   ImageState runState) {
        transitionImage(commandBuffer, *m_image, runState, newState,
                        CreateImageSubresourceRange(runBaseMipLevel, newState.m_aspectMask, runMipLevelCount));
        for (std::uint32_t mipLevel = runBaseMipLevel; mipLevel < runBaseMipLevel + runMipLevelCount; ++mipLevel) {
            m_states[mipLevel] = runState;
        }
    };

    // Start handling the first mip level
    std::uint32_t runBaseMipLevel = baseMipLevel;
    ImageState runState = m_states[runBaseMipLevel];

    // The loop will store the first mip level it has to transition in runBaseMipLevel,
    for (std::uint32_t mipLevel = baseMipLevel + 1; mipLevel < endMipLevel; ++mipLevel) {
        // All mip levels between runBaseMipLevel and mipLevel have the same state
        if (m_states[mipLevel] != runState) {
            // Different state found in this mip level, so we need to create a new barrier.

            // Transition current mip levels, note that they share the same state
            // Consecutive mips that already share the same source state can use one wider barrier.
            transitionRun(runBaseMipLevel, mipLevel - runBaseMipLevel, runState);

            // Store current state of mip level and try to find more additional mip levels that are compatible.
            runBaseMipLevel = mipLevel;
            runState = m_states[mipLevel];
        }
    }

    // The loop will store the first mip level pending transition in runBaseMipLevel,
    // all mip levels between runBaseMipLevel and the end of the loop have the same state and can use the same
    // transition barrier. This last transition covers pending mips still not transitioned by the loop, including the
    // common case of all mips sharing the same state.
    transitionRun(runBaseMipLevel, endMipLevel - runBaseMipLevel, runState);
}

void Application::GpuViewImage::destroy()
{
    m_imageView.reset();
    m_image.destroy();
}

} // namespace siggraph

int main(int argc, char** argv)
{
    try {
        siggraph::Application app;
        app.setFrameLimit(siggraph::util::readFrameLimitCLI(argc, argv));
        app.run();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
