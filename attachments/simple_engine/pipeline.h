#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <string>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "vulkan_device.h"
#include "swap_chain.h"

/**
 * @brief Structure for material properties.
 */
struct MaterialProperties {
    alignas(16) glm::vec4 ambientColor;
    alignas(16) glm::vec4 diffuseColor;
    alignas(16) glm::vec4 specularColor;
    alignas(4) float shininess;
    alignas(4) float padding[3]; // Padding to ensure alignment
};

/**
 * @brief Class for managing Vulkan pipelines.
 */
class Pipeline {
public:
    /**
     * @brief Constructor.
     * @param device The Vulkan device.
     * @param swapChain The swap chain.
     */
    Pipeline(VulkanDevice& device, SwapChain& swapChain);

    /**
     * @brief Destructor.
     */
    ~Pipeline();

    /**
     * @brief Create the descriptor set layout.
     * @return True if the descriptor set layout was created successfully, false otherwise.
     */
    bool createDescriptorSetLayout();

    /**
     * @brief Create the PBR descriptor set layout.
     * @return True if the PBR descriptor set layout was created successfully, false otherwise.
     */
    bool createPBRDescriptorSetLayout();

    /**
     * @brief Create the graphics pipeline.
     * @return True if the graphics pipeline was created successfully, false otherwise.
     */
    bool createGraphicsPipeline();

    /**
     * @brief Create the PBR pipeline.
     * @return True if the PBR pipeline was created successfully, false otherwise.
     */
    bool createPBRPipeline();

    /**
     * @brief Create the lighting pipeline.
     * @return True if the lighting pipeline was created successfully, false otherwise.
     */
    bool createLightingPipeline();

    /**
     * @brief Push material properties to a command buffer.
     * @param commandBuffer The command buffer.
     * @param material The material properties.
     */
    void pushMaterialProperties(vk::CommandBuffer commandBuffer, const MaterialProperties& material);

    /**
     * @brief Get the descriptor set layout.
     * @return The descriptor set layout.
     */
    vk::raii::DescriptorSetLayout& getDescriptorSetLayout() { return descriptorSetLayout; }

    /**
     * @brief Get the pipeline layout.
     * @return The pipeline layout.
     */
    vk::raii::PipelineLayout& getPipelineLayout() { return pipelineLayout; }

    /**
     * @brief Get the graphics pipeline.
     * @return The graphics pipeline.
     */
    vk::raii::Pipeline& getGraphicsPipeline() { return graphicsPipeline; }

    /**
     * @brief Get the PBR pipeline layout.
     * @return The PBR pipeline layout.
     */
    vk::raii::PipelineLayout& getPBRPipelineLayout() { return pbrPipelineLayout; }

    /**
     * @brief Get the PBR graphics pipeline.
     * @return The PBR graphics pipeline.
     */
    vk::raii::Pipeline& getPBRGraphicsPipeline() { return pbrGraphicsPipeline; }

    /**
     * @brief Get the lighting pipeline layout.
     * @return The lighting pipeline layout.
     */
    vk::raii::PipelineLayout& getLightingPipelineLayout() { return lightingPipelineLayout; }

    /**
     * @brief Get the lighting pipeline.
     * @return The lighting pipeline.
     */
    vk::raii::Pipeline& getLightingPipeline() { return lightingPipeline; }

    /**
     * @brief Get the compute pipeline layout.
     * @return The compute pipeline layout.
     */
    vk::raii::PipelineLayout& getComputePipelineLayout() { return computePipelineLayout; }

    /**
     * @brief Get the compute pipeline.
     * @return The compute pipeline.
     */
    vk::raii::Pipeline& getComputePipeline() { return computePipeline; }

    /**
     * @brief Get the compute descriptor set layout.
     * @return The compute descriptor set layout.
     */
    vk::raii::DescriptorSetLayout& getComputeDescriptorSetLayout() { return computeDescriptorSetLayout; }

    /**
     * @brief Get the PBR descriptor set layout.
     * @return The PBR descriptor set layout.
     */
    vk::raii::DescriptorSetLayout& getPBRDescriptorSetLayout() { return pbrDescriptorSetLayout; }

private:
    // Vulkan device
    VulkanDevice& device;

    // Swap chain
    SwapChain& swapChain;

    // Pipelines
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::PipelineLayout pbrPipelineLayout = nullptr;
    vk::raii::Pipeline pbrGraphicsPipeline = nullptr;
    vk::raii::PipelineLayout lightingPipelineLayout = nullptr;
    vk::raii::Pipeline lightingPipeline = nullptr;

    // Compute pipeline
    vk::raii::PipelineLayout computePipelineLayout = nullptr;
    vk::raii::Pipeline computePipeline = nullptr;
    vk::raii::DescriptorSetLayout computeDescriptorSetLayout = nullptr;

    // Descriptor set layouts
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout pbrDescriptorSetLayout = nullptr;

    // Helper functions
    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code);
    std::vector<char> readFile(const std::string& filename);
};
