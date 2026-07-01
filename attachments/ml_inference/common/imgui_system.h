/**
 * Minimal ImGui System for MNIST Application
 * Simplified version following simple_engine patterns but without AudioSystem dependencies
 */
#pragma once

#include <vulkan/vulkan_raii.hpp>
#include "imgui.h"
#include <vector>
#include <cstdint>

struct GLFWwindow;
class Renderer;
struct ImGuiContext;

class ImGuiSystem {
public:
    ImGuiSystem(Renderer* renderer, GLFWwindow* window, uint32_t width, uint32_t height);
    ~ImGuiSystem();

    void NewFrame();
    void Render(vk::raii::CommandBuffer& commandBuffer, uint32_t frameIndex);
    void UpdateDisplaySize(uint32_t width, uint32_t height);
    ImTextureID AddTexture(vk::raii::ImageView const& imageView, vk::raii::Sampler const& sampler);

private:
    bool initialize();
    bool createFontTexture();
    bool createDescriptorSetLayout();
    bool createDescriptorPool();
    bool createDescriptorSet();
    bool createPipelineLayout();
    bool createPipeline();
    void updateBuffers(uint32_t frameIndex);

    ImGuiContext* context_ = nullptr;
    Renderer* renderer_ = nullptr;
    GLFWwindow* window_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    // Vulkan resources
    vk::raii::DescriptorPool descriptorPool_ = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayout_ = nullptr;
    vk::raii::DescriptorSet descriptorSet_ = nullptr; // Font descriptor set
    std::vector<vk::raii::DescriptorSet> textureSets_; // Additional texture sets
    std::vector<ImTextureID> textureIds_;
    vk::raii::PipelineLayout pipelineLayout_ = nullptr;
    vk::raii::Pipeline pipeline_ = nullptr;
    vk::raii::Sampler fontSampler_ = nullptr;
    vk::raii::Image fontImage_ = nullptr;
    vk::raii::DeviceMemory fontMemory_ = nullptr;
    vk::raii::ImageView fontView_ = nullptr;

    // Per-frame buffers
    std::vector<vk::raii::Buffer> vertexBuffers_;
    std::vector<vk::raii::DeviceMemory> vertexMemories_;
    std::vector<vk::raii::Buffer> indexBuffers_;
    std::vector<vk::raii::DeviceMemory> indexMemories_;
};
