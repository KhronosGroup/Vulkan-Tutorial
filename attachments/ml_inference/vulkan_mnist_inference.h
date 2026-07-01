#pragma once

#include "renderer.h"
#include "weight_loader.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <string>

class VulkanMNISTInference {
public:
    VulkanMNISTInference(Renderer& renderer);
    ~VulkanMNISTInference();

    bool loadWeights(const std::string& path);
    std::vector<float> infer(const std::vector<float>& input);
    bool hasWeights() const { return weightsLoaded_; }

    static std::vector<float> preprocess(const std::vector<float>& input);

private:
    void createBuffers();
    void createShaderModules();
    void createPipelines();
    void destroyResources();

    vk::raii::ShaderModule loadShader(const std::string& path);

    Renderer& renderer_;
    bool weightsLoaded_ = false;

    // Model weights
    WeightLoader::ModelWeights weights_;

    // Vulkan resources
    vk::raii::ShaderModule denseShader_{nullptr};
    vk::raii::ShaderModule reluShader_{nullptr};
    vk::raii::ShaderModule softmaxShader_{nullptr};

    vk::raii::DescriptorSetLayout denseLayout_{nullptr};
    vk::raii::DescriptorSetLayout reluLayout_{nullptr};
    vk::raii::DescriptorSetLayout softmaxLayout_{nullptr};

    vk::raii::PipelineLayout densePipelineLayout_{nullptr};
    vk::raii::PipelineLayout reluPipelineLayout_{nullptr};
    vk::raii::PipelineLayout softmaxPipelineLayout_{nullptr};

    vk::raii::Pipeline densePipeline_{nullptr};
    vk::raii::Pipeline reluPipeline_{nullptr};
    vk::raii::Pipeline softmaxPipeline_{nullptr};

    vk::raii::DescriptorPool descriptorPool_{nullptr};
    vk::raii::DescriptorSet denseDescSet1_{nullptr};
    vk::raii::DescriptorSet reluDescSet_{nullptr};
    vk::raii::DescriptorSet denseDescSet2_{nullptr};
    vk::raii::DescriptorSet softmaxDescSet_{nullptr};

    // Buffers
    vk::raii::Buffer inputBuffer_{nullptr};
    vk::raii::DeviceMemory inputMemory_{nullptr};

    vk::raii::Buffer fc1WeightBuffer_{nullptr};
    vk::raii::DeviceMemory fc1WeightMemory_{nullptr};
    vk::raii::Buffer fc1BiasBuffer_{nullptr};
    vk::raii::DeviceMemory fc1BiasMemory_{nullptr};

    vk::raii::Buffer fc1OutputBuffer_{nullptr};
    vk::raii::DeviceMemory fc1OutputMemory_{nullptr};

    vk::raii::Buffer reluOutputBuffer_{nullptr};
    vk::raii::DeviceMemory reluOutputMemory_{nullptr};

    vk::raii::Buffer fc2WeightBuffer_{nullptr};
    vk::raii::DeviceMemory fc2WeightMemory_{nullptr};
    vk::raii::Buffer fc2BiasBuffer_{nullptr};
    vk::raii::DeviceMemory fc2BiasMemory_{nullptr};

    vk::raii::Buffer fc2OutputBuffer_{nullptr};
    vk::raii::DeviceMemory fc2OutputMemory_{nullptr};

    vk::raii::Buffer softmaxOutputBuffer_{nullptr};
    vk::raii::DeviceMemory softmaxOutputMemory_{nullptr};
};
