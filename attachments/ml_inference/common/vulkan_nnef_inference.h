#pragma once
#include "renderer.h"
#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <string>
#include <map>
#include "nnef.h"

class VulkanNNEFInference {
public:
    VulkanNNEFInference(Renderer& renderer);
    ~VulkanNNEFInference();

    bool loadModel(const std::string& path);
    std::vector<float> infer(const std::vector<float>& input);

private:
    struct Layer {
        std::string name;
        std::string type;
        std::map<std::string, std::string> inputs;
        std::string output;
        
        // Vulkan resources for this layer
        vk::raii::DescriptorSet descriptorSet = nullptr;
        
        // Operation parameters
        uint32_t groups = 1;
        uint32_t stride = 1;
        uint32_t padding = 0;
        uint32_t kernelSize = 1;
        uint32_t activation = 0; // 0: None, 1: ReLU, 2: ReLU6
    };

    void buildExecutionPlan();
    void createBuffers();
    void createDescriptorPool();
    void createPipelines();

    Renderer& renderer_;
    nnef::Graph graph_;
    std::string modelPath_;
    bool loaded_ = false;

    // Buffers for all tensors in the graph
    struct TensorBuffer {
        vk::raii::Buffer buffer = nullptr;
        vk::raii::DeviceMemory memory = nullptr;
        size_t size = 0;
        std::vector<int> shape;
    };
    std::map<std::string, TensorBuffer> tensors_;

    vk::raii::DescriptorPool descriptorPool_ = nullptr;
    vk::raii::DescriptorSetLayout layout_ = nullptr;
    vk::raii::PipelineLayout pipelineLayout_ = nullptr;
    
    // Dummy buffer for missing bindings
    vk::raii::Buffer dummyBuffer_ = nullptr;
    vk::raii::DeviceMemory dummyMemory_ = nullptr;
    
    // Shader modules and pipelines
    vk::raii::ShaderModule shaderModule_ = nullptr;
    vk::raii::Pipeline pipeline_ = nullptr;
    
    std::vector<Layer> layers_;
};
