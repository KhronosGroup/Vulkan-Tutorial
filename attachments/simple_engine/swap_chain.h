#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vk_platform.h>
#include <vector>
#include <memory>

#include "vulkan_device.h"
#include "platform.h"

/**
 * @brief Class for managing the Vulkan swap chain.
 */
class SwapChain {
public:
    /**
     * @brief Constructor.
     * @param device The Vulkan device.
     * @param platform The platform.
     */
    SwapChain(VulkanDevice& device, Platform* platform);

    /**
     * @brief Destructor.
     */
    ~SwapChain();

    /**
     * @brief Create the swap chain.
     * @return True if the swap chain was created successfully, false otherwise.
     */
    bool create();

    /**
     * @brief Create image views for the swap chain images.
     * @return True if the image views were created successfully, false otherwise.
     */
    bool createImageViews();

    /**
     * @brief Clean up the swap chain.
     */
    void cleanup();

    /**
     * @brief Recreate the swap chain.
     * @return True, if the swap chain was recreated successfully, false otherwise.
     */
    bool recreate();

    /**
     * @brief Get the swap chain.
     * @return The swap chain.
     */
    vk::raii::SwapchainKHR& getSwapChain() { return swapChain; }

    /**
     * @brief Get the swap chain images.
     * @return The swap chain images.
     */
    const std::vector<vk::Image>& getSwapChainImages() const { return swapChainImages; }

    /**
     * @brief Get the swap chain image format.
     * @return The swap chain image format.
     */
    vk::Format getSwapChainImageFormat() const { return swapChainImageFormat; }

    /**
     * @brief Get the swap chain extent.
     * @return The swap chain extent.
     */
    vk::Extent2D getSwapChainExtent() const { return swapChainExtent; }

    /**
     * @brief Get the swap chain image views.
     * @return The swap chain image views.
     */
    const std::vector<vk::raii::ImageView>& getSwapChainImageViews() const { return swapChainImageViews; }

private:
    // Vulkan device
    VulkanDevice& device;

    // Platform
    Platform* platform;

    // Swap chain
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D swapChainExtent = {0, 0};
    std::vector<vk::raii::ImageView> swapChainImageViews;

    // Helper functions
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
};
