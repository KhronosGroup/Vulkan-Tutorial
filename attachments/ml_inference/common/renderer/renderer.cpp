/**
 * Minimal Vulkan Renderer implementation for MNIST GUI
 */
#include "renderer.h"
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

Renderer::Renderer(GLFWwindow* window, uint32_t width, uint32_t height)
    : window_(window), width_(width), height_(height) {

    createInstance();
    if (window_) {
        createSurface();
    }
    pickPhysicalDevice();
    createLogicalDevice();
    if (window_) {
        createSwapchain();
        createImageViews();
        createRenderPass();
        createFramebuffers();
    } else {
        createOffscreenResources();
        createRenderPass();
        createFramebuffers();
    }
    createCommandPool();
    createCommandBuffers();
    if (window_) {
        createSyncObjects();
    }
}

Renderer::~Renderer() {
    if (*device_) {
        device_.waitIdle();
    }
}

void Renderer::createInstance() {
    // Initialize dispatch loader
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    vk::ApplicationInfo appInfo{
        .pApplicationName = "Image Classifier",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2
    };

    uint32_t glfwExtensionCount = 0;
    std::vector<const char*> extensions;
    if (window_) {
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        extensions.assign(glfwExtensions, glfwExtensions + glfwExtensionCount);
    }

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()
    };
    instance_ = vk::raii::Instance(context_, createInfo);

    // Initialize instance-level functions
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance_);
}

void Renderer::createSurface() {
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(*instance_, window_, nullptr, &rawSurface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
    surface_ = vk::raii::SurfaceKHR(instance_, rawSurface);
}

Renderer::QueueFamilyIndices Renderer::findQueueFamilies(vk::PhysicalDevice device) {
    QueueFamilyIndices indices;
    auto queueFamilies = device.getQueueFamilyProperties();

    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }
        if (window_) {
            if (device.getSurfaceSupportKHR(i, *surface_)) {
                indices.presentFamily = i;
            }
        } else {
            // Headless mode: we don't need a present family
            indices.presentFamily = i; 
        }
        if (indices.isComplete()) break;
        i++;
    }
    return indices;
}

void Renderer::pickPhysicalDevice() {
    auto devices = instance_.enumeratePhysicalDevices();
    for (auto& device : devices) {
        auto indices = findQueueFamilies(*device);
        if (indices.isComplete()) {
            physicalDevice_ = std::move(device);
            graphicsFamily_ = indices.graphicsFamily.value();
            presentFamily_ = indices.presentFamily.value();
            return;
        }
    }
    throw std::runtime_error("Failed to find suitable GPU");
}

void Renderer::createLogicalDevice() {
    std::set<uint32_t> uniqueQueueFamilies = {graphicsFamily_, presentFamily_};
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };
        queueCreateInfos.push_back(queueCreateInfo);
    }

    std::vector<const char*> deviceExtensions;
    if (window_) {
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    vk::DeviceCreateInfo createInfo{
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data()
    };

    device_ = vk::raii::Device(physicalDevice_, createInfo);
    graphicsQueue_ = vk::raii::Queue(device_, graphicsFamily_, 0);
    if (window_) {
        presentQueue_ = vk::raii::Queue(device_, presentFamily_, 0);
    }
}

void Renderer::createSwapchain() {
    auto capabilities = physicalDevice_.getSurfaceCapabilitiesKHR(*surface_);
    auto formats = physicalDevice_.getSurfaceFormatsKHR(*surface_);
    auto presentModes = physicalDevice_.getSurfacePresentModesKHR(*surface_);

    vk::SurfaceFormatKHR surfaceFormat = formats[0];
    for (const auto& format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb &&
            format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            surfaceFormat = format;
            break;
        }
    }

    vk::Extent2D extent = capabilities.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width = std::clamp(width_, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(height_, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    uint32_t queueFamilyIndices[] = {graphicsFamily_, presentFamily_};
    vk::SwapchainCreateInfoKHR createInfo{
        .surface = *surface_,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = (graphicsFamily_ != presentFamily_) ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = (graphicsFamily_ != presentFamily_) ? 2u : 0u,
        .pQueueFamilyIndices = (graphicsFamily_ != presentFamily_) ? queueFamilyIndices : nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = vk::PresentModeKHR::eFifo,
        .clipped = VK_TRUE
    };

    swapchain_ = vk::raii::SwapchainKHR(device_, createInfo);
    auto images = swapchain_.getImages();
    swapchainImages_.assign(images.begin(), images.end());
    swapchainImageFormat_ = surfaceFormat.format;
    swapchainExtent_ = extent;
}

void Renderer::createImageViews() {
    if (!window_) return;
    swapchainImageViews_.clear();
    swapchainImageViews_.reserve(swapchainImages_.size());
    for (size_t i = 0; i < swapchainImages_.size(); i++) {
        vk::ImageViewCreateInfo createInfo{
            .image = swapchainImages_[i],
            .viewType = vk::ImageViewType::e2D,
            .format = swapchainImageFormat_,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        swapchainImageViews_.emplace_back(device_, createInfo);
    }
}

void Renderer::createRenderPass() {
    vk::AttachmentDescription colorAttachment{
        .format = swapchainImageFormat_,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = window_ ? vk::ImageLayout::ePresentSrcKHR : vk::ImageLayout::eColorAttachmentOptimal
    };

    vk::AttachmentReference colorAttachmentRef{
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal
    };

    vk::SubpassDescription subpass{
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef
    };

    vk::SubpassDependency dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .srcAccessMask = {},
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
    };

    vk::RenderPassCreateInfo renderPassInfo{
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency
    };
    renderPass_ = vk::raii::RenderPass(device_, renderPassInfo);
}

void Renderer::createFramebuffers() {
    framebuffers_.reserve(swapchainImageViews_.size());
    for (size_t i = 0; i < swapchainImageViews_.size(); i++) {
        vk::ImageView attachments[] = {*swapchainImageViews_[i]};
        vk::FramebufferCreateInfo framebufferInfo{
            .renderPass = *renderPass_,
            .attachmentCount = 1,
            .pAttachments = attachments,
            .width = swapchainExtent_.width,
            .height = swapchainExtent_.height,
            .layers = 1
        };
        framebuffers_.emplace_back(device_, framebufferInfo);
    }
}

void Renderer::createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = graphicsFamily_
    };
    commandPool_ = vk::raii::CommandPool(device_, poolInfo);
}

void Renderer::createCommandBuffers() {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT
    };
    commandBuffers_ = vk::raii::CommandBuffers(device_, allocInfo);
}

void Renderer::createSyncObjects() {
    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo{
        .flags = vk::FenceCreateFlagBits::eSignaled
    };

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        imageAvailableSemaphores_.emplace_back(device_, semaphoreInfo);
        renderFinishedSemaphores_.emplace_back(device_, semaphoreInfo);
        inFlightFences_.emplace_back(device_, fenceInfo);
    }
}

void Renderer::createOffscreenResources() {
    swapchainImageFormat_ = vk::Format::eB8G8R8A8Unorm;
    swapchainExtent_ = vk::Extent2D{width_, height_};

    vk::ImageCreateInfo imageInfo{
        .imageType = vk::ImageType::e2D,
        .format = swapchainImageFormat_,
        .extent = {width_, height_, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc
    };

    offscreenImage_ = vk::raii::Image(device_, imageInfo);

    vk::MemoryRequirements memReqs = offscreenImage_.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize = memReqs.size,
        .memoryTypeIndex = FindMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    };

    offscreenMemory_ = vk::raii::DeviceMemory(device_, allocInfo);
    offscreenImage_.bindMemory(*offscreenMemory_, 0);

    vk::ImageViewCreateInfo viewInfo{
        .image = *offscreenImage_,
        .viewType = vk::ImageViewType::e2D,
        .format = swapchainImageFormat_,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
    };

    offscreenView_ = vk::raii::ImageView(device_, viewInfo);
    
    // In headless mode, we only have one "swapchain" image
    swapchainImages_.push_back(*offscreenImage_);
    swapchainImageViews_.emplace_back(device_, viewInfo);
}

bool Renderer::BeginFrame() {
    if (!window_) {
        imageIndex_ = 0; // Always 0 for offscreen
        
        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        commandBuffers_[currentFrame_].begin(beginInfo);
        
        vk::ClearValue clearColor{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
        vk::RenderPassBeginInfo renderPassInfo{
            .renderPass = *renderPass_,
            .framebuffer = *framebuffers_[imageIndex_],
            .renderArea = {.offset = {0, 0}, .extent = swapchainExtent_},
            .clearValueCount = 1,
            .pClearValues = &clearColor
        };
        commandBuffers_[currentFrame_].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        return true;
    }

    auto& fence = inFlightFences_[currentFrame_];
    (void)device_.waitForFences(*fence, VK_TRUE, UINT64_MAX);

    try {
        auto [result, index] = swapchain_.acquireNextImage(UINT64_MAX, *imageAvailableSemaphores_[currentFrame_]);
        imageIndex_ = index;
    } catch (vk::OutOfDateKHRError&) {
        return false;
    }

    device_.resetFences(*fence);

    auto& cmd = commandBuffers_[currentFrame_];
    cmd.reset();
    vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };
    cmd.begin(beginInfo);

    vk::ClearValue clearColor(vk::ClearColorValue(std::array{0.1f, 0.1f, 0.1f, 1.0f}));
    vk::RenderPassBeginInfo renderPassInfo{
        .renderPass = *renderPass_,
        .framebuffer = *framebuffers_[imageIndex_],
        .renderArea = {{0, 0}, swapchainExtent_},
        .clearValueCount = 1,
        .pClearValues = &clearColor
    };

    cmd.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    return true;
}

void Renderer::EndFrame() {
    auto& cmd = commandBuffers_[currentFrame_];
    if (!window_) {
        cmd.endRenderPass();
        cmd.end();
        
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*cmd
        };
        graphicsQueue_.submit(submitInfo);
        graphicsQueue_.waitIdle();
        return;
    }

    cmd.endRenderPass();
    cmd.end();

    vk::Semaphore waitSemaphores[] = {*imageAvailableSemaphores_[currentFrame_]};
    vk::Semaphore signalSemaphores[] = {*renderFinishedSemaphores_[currentFrame_]};
    vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
    vk::CommandBuffer cmdBuffer = *cmd;

    vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = waitSemaphores,
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = signalSemaphores
    };

    graphicsQueue_.submit(submitInfo, *inFlightFences_[currentFrame_]);

    vk::SwapchainKHR swapchains[] = {*swapchain_};
    vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = signalSemaphores,
        .swapchainCount = 1,
        .pSwapchains = swapchains,
        .pImageIndices = &imageIndex_
    };

    try {
        (void)presentQueue_.presentKHR(presentInfo);
    } catch (vk::OutOfDateKHRError&) {
    }

    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

uint32_t Renderer::FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice_.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

void Renderer::TransitionImageLayout(vk::Image image, vk::Format,
                                    vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    auto cmdBuffers = vk::raii::CommandBuffers(device_, allocInfo);
    auto& cmd = cmdBuffers[0];

    vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };
    cmd.begin(beginInfo);

    vk::ImageSubresourceRange range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
    };

    vk::ImageMemoryBarrier barrier{
        .srcAccessMask = {},
        .dstAccessMask = {},
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = range
    };

    vk::PipelineStageFlags sourceStage, destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else if (oldLayout == vk::ImageLayout::eColorAttachmentOptimal && newLayout == vk::ImageLayout::eTransferSrcOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
        sourceStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eColorAttachmentOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    } else {
        throw std::runtime_error("Unsupported layout transition");
    }

    cmd.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
    cmd.end();

    vk::CommandBuffer cmdBuffer = *cmd;
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuffer
    };
    graphicsQueue_.submit(submitInfo);
    graphicsQueue_.waitIdle();
}

void Renderer::CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    auto cmdBuffers = vk::raii::CommandBuffers(device_, allocInfo);
    auto& cmd = cmdBuffers[0];

    vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };
    cmd.begin(beginInfo);

    vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1}
    };

    cmd.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    cmd.end();

    vk::CommandBuffer cmdBuffer = *cmd;
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuffer
    };
    graphicsQueue_.submit(submitInfo);
    graphicsQueue_.waitIdle();
}

void Renderer::CopyImageToBuffer(vk::Image image, vk::Buffer buffer, uint32_t width, uint32_t height) {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    auto cmdBuffers = vk::raii::CommandBuffers(device_, allocInfo);
    auto& cmd = cmdBuffers[0];

    cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::BufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, 1}
    };

    cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, region);
    cmd.end();

    vk::CommandBuffer cmdBuffer = *cmd;
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuffer
    };
    graphicsQueue_.submit(submitInfo);
    graphicsQueue_.waitIdle();
}

vk::raii::ShaderModule Renderer::CreateShaderModule(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = file.tellg();
    std::vector<char> code(fileSize);
    file.seekg(0);
    file.read(code.data(), fileSize);

    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data())
    };

    return vk::raii::ShaderModule(device_, createInfo);
}
