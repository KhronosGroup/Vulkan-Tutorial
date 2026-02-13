/**
 * Minimal Vulkan Renderer for MNIST GUI
 *
 * Provides the essential Vulkan infrastructure needed by ImGuiSystem.
 * Follows the architecture taught in the Simple Game Engine tutorial.
 */
#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <vector>
#include <optional>

class Renderer {
public:
    Renderer(GLFWwindow* window, uint32_t width, uint32_t height);
    ~Renderer();

    // Frame rendering
    bool BeginFrame();
    void EndFrame();

    // Getters for ImGuiSystem
    vk::raii::Device& GetRaiiDevice() { return device_; }
    vk::raii::PhysicalDevice& GetPhysicalDevice() { return physicalDevice_; }
    vk::raii::Queue& GetGraphicsQueue() { return graphicsQueue_; }
    vk::raii::CommandPool& GetCommandPool() { return commandPool_; }
    const vk::raii::Device& GetRaiiDevice() const { return device_; }
    const vk::raii::PhysicalDevice& GetPhysicalDevice() const { return physicalDevice_; }
    const vk::raii::Queue& GetGraphicsQueue() const { return graphicsQueue_; }
    const vk::raii::CommandPool& GetCommandPool() const { return commandPool_; }
    vk::RenderPass GetRenderPass() const { return *renderPass_; }
    uint32_t GetMaxFramesInFlight() const { return MAX_FRAMES_IN_FLIGHT; }
    uint32_t GetCurrentFrame() const { return currentFrame_; }
    uint32_t GetImageIndex() const { return imageIndex_; }
    vk::Extent2D GetSwapchainExtent() const { return swapchainExtent_; }
    const std::vector<VkImage>& GetSwapchainImages() const { return swapchainImages_; }

    // Command buffer for ImGui
    vk::raii::CommandBuffer& GetCurrentCommandBuffer() { return commandBuffers_[currentFrame_]; }

    // Utility functions
    uint32_t FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    void TransitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    void CopyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
    void CopyImageToBuffer(vk::Image image, vk::Buffer buffer, uint32_t width, uint32_t height);
    vk::raii::ShaderModule CreateShaderModule(const std::string& filename);
    void WaitIdle() { if (*device_) device_.waitIdle(); }

    // Dummy loading state (not used for MNIST)
    bool IsLoading() const { return false; }

private:
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void createOffscreenResources();

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
    };

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);

    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    GLFWwindow* window_;
    uint32_t width_;
    uint32_t height_;

    vk::raii::Context context_;
    vk::raii::Instance instance_ = nullptr;
    vk::raii::SurfaceKHR surface_ = nullptr;
    vk::raii::PhysicalDevice physicalDevice_ = nullptr;
    vk::raii::Device device_ = nullptr;
    vk::raii::Queue graphicsQueue_ = nullptr;
    vk::raii::Queue presentQueue_ = nullptr;
    vk::raii::SwapchainKHR swapchain_ = nullptr;
    std::vector<VkImage> swapchainImages_;
    std::vector<vk::raii::ImageView> swapchainImageViews_;
    vk::Format swapchainImageFormat_;
    vk::Extent2D swapchainExtent_;
    vk::raii::RenderPass renderPass_ = nullptr;
    std::vector<vk::raii::Framebuffer> framebuffers_;
    vk::raii::CommandPool commandPool_ = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers_;
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores_;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores_;
    std::vector<vk::raii::Fence> inFlightFences_;

    uint32_t currentFrame_ = 0;
    uint32_t imageIndex_ = 0;
    uint32_t graphicsFamily_ = 0;
    uint32_t presentFamily_ = 0;

    // Headless / Offscreen resources
    vk::raii::Image offscreenImage_ = nullptr;
    vk::raii::DeviceMemory offscreenMemory_ = nullptr;
    vk::raii::ImageView offscreenView_ = nullptr;
};
