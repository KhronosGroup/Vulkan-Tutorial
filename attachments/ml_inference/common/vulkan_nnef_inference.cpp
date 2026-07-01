#include "vulkan_nnef_inference.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

struct ConvParams {
    uint32_t srcWidth;
    uint32_t srcHeight;
    uint32_t srcChannels;
    uint32_t dstWidth;
    uint32_t dstHeight;
    uint32_t dstChannels;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t groups;
    uint32_t activation; // 0: None, 1: ReLU, 2: ReLU6
    uint32_t opType;     // 0: Conv, 1: GlobalAvgPool, 2: Add, 3: Linear, 4: Copy, 5: Softmax, 6: Clamp
    uint32_t hasBias;    // 1 if bias tensor is present
};

VulkanNNEFInference::VulkanNNEFInference(Renderer& renderer) : renderer_(renderer) {
    auto& device = renderer_.GetRaiiDevice();
    
    // Create descriptor set layout
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // Input/Src
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // Weights
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}, // Bias
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}  // Output/Dst
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    layout_ = vk::raii::DescriptorSetLayout(device, layoutInfo);

    // Create pipeline layout
    vk::PushConstantRange pushConstant;
    pushConstant.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(ConvParams);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &(*layout_);
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
    pipelineLayout_ = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    // Create dummy buffer
    vk::BufferCreateInfo bufferInfo{.size = 4, .usage = vk::BufferUsageFlagBits::eStorageBuffer};
    dummyBuffer_ = vk::raii::Buffer(device, bufferInfo);
    auto memReqs = dummyBuffer_.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{.allocationSize = memReqs.size, .memoryTypeIndex = renderer_.FindMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)};
    dummyMemory_ = vk::raii::DeviceMemory(device, allocInfo);
    dummyBuffer_.bindMemory(*dummyMemory_, 0);
}

VulkanNNEFInference::~VulkanNNEFInference() {}

bool VulkanNNEFInference::loadModel(const std::string& path) {
    std::cout << "NNEF: Loading model from " << path << "..." << std::endl;
    std::string error;
    if (!nnef::load_graph(path, graph_, error)) {
        std::cerr << "NNEF: Failed to load graph from " << path << ": " << error << std::endl;
        return false;
    }
    
    modelPath_ = path;
    
    // Load variables (weights)
    std::cout << "NNEF: Loading variables..." << std::endl;
    if (!nnef::load_variables(path, graph_, error)) {
        std::cerr << "NNEF: Failed to load variables: " << error << std::endl;
        return false;
    }

    // Infer shapes
    std::cout << "NNEF: Inferring shapes..." << std::endl;
    if (!nnef::infer_shapes(graph_, error)) {
        std::cerr << "NNEF: Failed to infer shapes: " << error << std::endl;
        return false;
    }

    std::cout << "NNEF: Building execution plan..." << std::endl;
    buildExecutionPlan();
    std::cout << "NNEF: Creating buffers..." << std::endl;
    createBuffers();
    std::cout << "NNEF: Creating descriptor pool..." << std::endl;
    createDescriptorPool();
    std::cout << "NNEF: Creating pipelines..." << std::endl;
    createPipelines();
    
    loaded_ = true;
    std::cout << "NNEF: Model loaded successfully." << std::endl;
    return true;
}

void VulkanNNEFInference::buildExecutionPlan() {
    layers_.clear();
    std::set<std::string> tensorsToEliminate;

    for (size_t i = 0; i < graph_.operations.size(); ++i) {
        auto& op = graph_.operations[i];
        
        if (op.name == "variable" || op.name == "external" || op.name == "constant") continue;

        // Handle activation folding: if next op is relu/relu6 and its only input is this op's output
        uint32_t activation = 0;
        if (op.outputs.empty()) continue;
        std::string next_output = op.outputs.begin()->second.identifier();
        
        // Peek next op for folding
        if (i + 1 < graph_.operations.size()) {
            auto& next_op = graph_.operations[i+1];
            if ((next_op.name == "relu" || next_op.name == "relu6") && 
                next_op.inputs.size() == 1 && next_op.inputs.begin()->second.identifier() == next_output) {
                activation = (next_op.name == "relu") ? 1 : 2;
                next_output = next_op.outputs.begin()->second.identifier();
                tensorsToEliminate.insert(op.outputs.begin()->second.identifier());
                i++; // Skip next op as it's folded
            }
        }

        Layer layer;
        layer.name = op.outputs.begin()->second.identifier(); // Use output name as layer name
        layer.type = op.name;
        layer.output = next_output;
        layer.activation = activation;
        
        for (auto& input : op.inputs) {
            if (input.second.kind() == nnef::Value::Identifier) {
                layer.inputs[input.first] = input.second.identifier();
            }
        }
        
        if (op.name == "conv") {
            if (op.attribs.contains("stride")) {
                auto stride = op.attribs.get("stride");
                if (stride.kind() == nnef::Value::Array && stride.size() >= 2) {
                    layer.stride = stride[0].integer();
                }
            }
            if (op.attribs.contains("groups")) {
                layer.groups = op.attribs.get("groups").integer();
            }
            if (layer.inputs.count("filter")) {
                auto& weightTensor = graph_.tensors[layer.inputs["filter"]];
                if (weightTensor.shape.size() >= 4) {
                    layer.kernelSize = weightTensor.shape[2];
                }
            }
            if (op.attribs.contains("padding")) {
                auto padding = op.attribs.get("padding");
                if (padding.kind() == nnef::Value::Array && padding.size() > 0) {
                    // NNEF padding: [(top, bottom), (left, right)]
                    layer.padding = padding[0][0].integer();
                }
            }
        }
        
        layers_.push_back(std::move(layer));
    }
}

void VulkanNNEFInference::createBuffers() {
    auto& device = renderer_.GetRaiiDevice();
    std::cout << "NNEF: Creating buffers for " << graph_.tensors.size() << " tensors..." << std::endl;

    // Build quick lookup sets for inputs/outputs to ensure host-visible where needed
    std::set<std::string> inputNames(graph_.inputs.begin(), graph_.inputs.end());
    std::set<std::string> outputNames(graph_.outputs.begin(), graph_.outputs.end());

    for (auto& pair : graph_.tensors) {
        auto& tensor = pair.second;
        TensorBuffer tb;
        tb.shape = tensor.shape;
        
        size_t count = 1;
        if (tb.shape.empty()) {
            std::cout << "NNEF: Warning: Tensor '" << pair.first << "' has empty shape." << std::endl;
        } else {
            for (int d : tb.shape) count *= d;
        }
        
        tb.size = std::max((size_t)4, count * sizeof(float));
        
        if (outputNames.count(pair.first)) {
             std::cout << "NNEF: Output tensor '" << pair.first << "' size=" << tb.size << " count=" << count << std::endl;
        }
        
        vk::BufferCreateInfo bufferInfo{
            .size = tb.size,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive
        };
        tb.buffer = vk::raii::Buffer(device, bufferInfo);
        
        vk::MemoryRequirements memReqs = tb.buffer.getMemoryRequirements();
        vk::MemoryPropertyFlags props = vk::MemoryPropertyFlagBits::eDeviceLocal;

        // Ensure host-visible for weights (have data), inputs, and outputs
        if (!tensor.data.empty() || inputNames.count(pair.first) || outputNames.count(pair.first)) {
            props = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        }
        
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = renderer_.FindMemoryType(memReqs.memoryTypeBits, props)
        };
        
        tb.memory = vk::raii::DeviceMemory(device, allocInfo);
        tb.buffer.bindMemory(*tb.memory, 0);

        // Zero-initialize the buffer if host-visible to avoid junk in inputs/outputs
        if (tensor.data.empty() && (props & vk::MemoryPropertyFlagBits::eHostVisible)) {
            void* dataPtr = tb.memory.mapMemory(0, VK_WHOLE_SIZE);
            std::memset(dataPtr, 0, tb.size);
            tb.memory.unmapMemory();
        }
        
        if (!tensor.data.empty()) {
            // Map the whole allocation to avoid edge cases with non-coherent atom sizes
            void* dataPtr = nullptr;
            try {
                dataPtr = tb.memory.mapMemory(0, VK_WHOLE_SIZE);
            } catch (const vk::SystemError& e) {
                std::cerr << "NNEF: Failed to map weight buffer for tensor '" << pair.first
                          << "': " << e.what() << std::endl;
                throw;
            }
            std::memcpy(dataPtr, tensor.data.data(), tensor.data.size());
            tb.memory.unmapMemory();
        }
        
        tensors_[pair.first] = std::move(tb);
    }
}

void VulkanNNEFInference::createDescriptorPool() {
    auto& device = renderer_.GetRaiiDevice();
    std::cout << "NNEF: Creating descriptor pool for " << layers_.size() << " layers..." << std::endl;
    
    vk::DescriptorPoolSize poolSize;
    poolSize.type = vk::DescriptorType::eStorageBuffer;
    poolSize.descriptorCount = (uint32_t)layers_.size() * 4 + 4;

    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.maxSets = (uint32_t)layers_.size() + 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    
    descriptorPool_ = vk::raii::DescriptorPool(device, poolInfo);
    
    // Allocate and update sets for each layer
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        vk::DescriptorSetAllocateInfo allocInfo;
        allocInfo.descriptorPool = *descriptorPool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &(*layout_);
        
        layer.descriptorSet = std::move(vk::raii::DescriptorSets(device, allocInfo)[0]);

        // Bind buffers to descriptor set
        std::vector<vk::DescriptorBufferInfo> bufferInfos;
        bufferInfos.reserve(4);
        std::vector<vk::WriteDescriptorSet> writes;
        writes.reserve(4);

        auto addBinding = [&](const std::string& tensorName, uint32_t binding) {
            vk::DescriptorBufferInfo info{};
            if (tensorName.empty() || tensors_.find(tensorName) == tensors_.end()) {
                info.buffer = *dummyBuffer_;
                info.offset = 0;
                info.range = VK_WHOLE_SIZE;
            } else {
                auto& tb = tensors_.at(tensorName);
                info.buffer = *tb.buffer;
                info.offset = 0;
                info.range = VK_WHOLE_SIZE;
            }
            bufferInfos.push_back(info);

            vk::WriteDescriptorSet write{};
            write.dstSet = *layer.descriptorSet;
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = vk::DescriptorType::eStorageBuffer;
            write.pBufferInfo = &bufferInfos.back();
            writes.push_back(write);
        };

        if (layer.type == "conv") {
            addBinding(layer.inputs["input"], 0);
            addBinding(layer.inputs["filter"], 1);
            addBinding(layer.inputs["bias"], 2);
            addBinding(layer.output, 3);
        } else if (layer.type == "add") {
            addBinding(layer.inputs["x"], 0);
            addBinding(layer.inputs["y"], 1);
            addBinding("", 2); 
            addBinding(layer.output, 3);
        } else if (layer.type == "box" || layer.type == "max_pool" || layer.type == "avg_pool") {
            addBinding(layer.inputs["input"], 0);
            addBinding("", 1);
            addBinding("", 2);
            addBinding(layer.output, 3);
        } else if (layer.type == "linear" || layer.type == "matmul") {
            addBinding(layer.inputs["input"], 0);
            addBinding(layer.inputs["filter"], 1);
            addBinding(layer.inputs["bias"], 2);
            addBinding(layer.output, 3);
        } else if (layer.type == "reshape" || layer.type == "squeeze" || layer.type == "unsqueeze") {
            addBinding(layer.inputs.begin()->second, 0);
            addBinding("", 1);
            addBinding("", 2);
            addBinding(layer.output, 3);
        } else if (layer.type == "softmax") {
            addBinding(layer.inputs["x"], 0);
            addBinding("", 1);
            addBinding("", 2);
            addBinding(layer.output, 3);
        } else if (layer.type == "clamp") {
            addBinding(layer.inputs["x"], 0);
            addBinding("", 1);
            addBinding("", 2);
            addBinding(layer.output, 3);
        } else if (layer.type == "mean_reduce") {
            addBinding(layer.inputs["input"], 0);
            addBinding("", 1);
            addBinding("", 2);
            addBinding(layer.output, 3);
        } else {
            addBinding("", 0); addBinding("", 1); addBinding("", 2); addBinding("", 3);
        }

        if (!writes.empty()) {
            device.updateDescriptorSets(writes, nullptr);
        }
    }
    std::cout << "NNEF: Descriptor pool created successfully." << std::endl;
}

void VulkanNNEFInference::createPipelines() {
    auto& device = renderer_.GetRaiiDevice();
    
    std::vector<std::string> searchPaths = {
        "shaders/nnef_ops.comp.spv",
        "rPi/shaders/nnef_ops.comp.spv",
        "cmake-build-debug/shaders/nnef_ops.comp.spv",
        "cmake-build-debug/rPi/shaders/nnef_ops.comp.spv"
    };

    bool found = false;
    for (const auto& path : searchPaths) {
        std::ifstream f(path, std::ios::binary);
        if (f.is_open()) {
            std::cout << "NNEF: Loading shader from " << path << std::endl;
            try {
                shaderModule_ = renderer_.CreateShaderModule(path);
                found = true;
                break;
            } catch (...) {}
        }
    }

    if (!found) {
        throw std::runtime_error("CRITICAL ERROR: NNEF shader 'nnef_ops.comp.spv' not found in any search path!");
    }

    vk::ComputePipelineCreateInfo info;
    info.stage.stage = vk::ShaderStageFlagBits::eCompute;
    info.stage.module = *shaderModule_;
    info.stage.pName = "main";
    info.layout = *pipelineLayout_;
    pipeline_ = vk::raii::Pipeline(device, nullptr, info);
}

std::vector<float> VulkanNNEFInference::infer(const std::vector<float>& input) {
    if (!loaded_) return {};
    
    // 1. Upload input
    if (graph_.inputs.empty()) {
        std::cerr << "NNEF: Graph has no inputs." << std::endl;
        return {};
    }

    float inputMax = -1e9;
    for (float v : input) {
        inputMax = std::max(inputMax, v);
    }

    auto& inputTensor = tensors_[graph_.inputs[0]];
    size_t inputBytes = input.size() * sizeof(float);

    if (inputBytes > 0) {
        // std::cout << "NNEF: Input sample [0..4]: ";
        // for (size_t i = 0; i < std::min((size_t)5, input.size()); ++i) std::cout << input[i] << " ";
        // std::cout << std::endl;
    }
    
    if (inputBytes > inputTensor.size) {
        auto& device = renderer_.GetRaiiDevice();
        vk::BufferCreateInfo bufferInfo{.size = inputBytes, .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc};
        inputTensor.buffer = vk::raii::Buffer(device, bufferInfo);
        auto memReqs = inputTensor.buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{.allocationSize = memReqs.size, .memoryTypeIndex = renderer_.FindMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)};
        inputTensor.memory = vk::raii::DeviceMemory(device, allocInfo);
        inputTensor.buffer.bindMemory(*inputTensor.memory, 0);
        inputTensor.size = inputBytes;
    }

    void* dataPtr = inputTensor.memory.mapMemory(0, VK_WHOLE_SIZE);
    std::memcpy(dataPtr, input.data(), inputBytes);
    inputTensor.memory.unmapMemory();

    // 2. Record and Submit Command Buffer
    auto& device = renderer_.GetRaiiDevice();
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *renderer_.GetCommandPool(),
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    auto cmdBuffers = vk::raii::CommandBuffers(device, allocInfo);
    auto& cmd = cmdBuffers[0];

    cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        
        uint32_t opType = 999;
        if (layer.type == "conv") opType = 0;
        else if (layer.type == "box" || layer.type == "max_pool" || layer.type == "avg_pool" || layer.type == "mean_reduce") opType = 1;
        else if (layer.type == "add") opType = 2;
        else if (layer.type == "linear" || layer.type == "matmul") opType = 3;
        else if (layer.type == "reshape" || layer.type == "squeeze" || layer.type == "unsqueeze") opType = 4;
        else if (layer.type == "softmax") opType = 5;
        else if (layer.type == "clamp") opType = 6;
        
        if (opType == 999) continue;

        if (*pipeline_ == VK_NULL_HANDLE) continue;

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout_, 0, {*layer.descriptorSet}, nullptr);

        ConvParams p{};
        p.stride = layer.stride;
        p.padding = layer.padding;
        p.groups = layer.groups;
        p.kernelSize = layer.kernelSize;
        p.activation = layer.activation;
        p.opType = opType;
        p.hasBias = (layer.inputs.count("bias") && !layer.inputs["bias"].empty() && tensors_.count(layer.inputs["bias"])) ? 1 : 0;

        if (layer.inputs.count("input") && tensors_.count(layer.inputs["input"])) {
            auto& in = tensors_[layer.inputs["input"]];
            if (in.shape.size() >= 4) { p.srcChannels = in.shape[1]; p.srcHeight = in.shape[2]; p.srcWidth = in.shape[3]; }
            else if (in.shape.size() == 2) { p.srcChannels = in.shape[1]; p.srcHeight = 1; p.srcWidth = 1; }
        } else if (layer.inputs.count("x") && tensors_.count(layer.inputs["x"])) {
             auto& in = tensors_[layer.inputs["x"]];
             if (in.shape.size() >= 4) { p.srcChannels = in.shape[1]; p.srcHeight = in.shape[2]; p.srcWidth = in.shape[3]; }
             else { p.srcChannels = (uint32_t)in.size/4; p.srcHeight = 1; p.srcWidth = 1; }
        }

        if (tensors_.count(layer.output)) {
            auto& out = tensors_[layer.output];
            if (out.shape.size() >= 4) { p.dstChannels = out.shape[1]; p.dstHeight = out.shape[2]; p.dstWidth = out.shape[3]; }
            else if (out.shape.size() == 2) { p.dstChannels = out.shape[1]; p.dstHeight = 1; p.dstWidth = 1; }
            else { p.dstChannels = (uint32_t)out.size/4; p.dstHeight = 1; p.dstWidth = 1; }
        }

        cmd.pushConstants<ConvParams>(*pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, p);

        if (opType == 0) { // Conv
            uint32_t totalThreads = p.dstWidth * p.dstHeight * p.dstChannels;
            cmd.dispatch((totalThreads + 255) / 256, 1, 1);
        } else if (opType == 1) { // Pool
            cmd.dispatch((p.srcChannels + 255) / 256, 1, 1); 
        } else if (opType == 2) { // Add
            cmd.dispatch((p.dstWidth * p.dstHeight * p.dstChannels + 255) / 256, 1, 1);
        } else if (opType == 3) { // Linear
            cmd.dispatch((p.dstChannels + 255) / 256, 1, 1);
        } else if (opType == 4) { // Copy
            cmd.dispatch((p.dstChannels + 255) / 256, 1, 1);
        } else if (opType == 5) { // Softmax
            cmd.dispatch(1, 1, 1); 
        } else if (opType == 6) { // Clamp
            cmd.dispatch((p.dstWidth * p.dstHeight * p.dstChannels + 255) / 256, 1, 1);
        }

        vk::MemoryBarrier barrier{
            .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead
        };
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, {barrier}, {}, {});
    }

    cmd.end();

    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd};
    renderer_.GetGraphicsQueue().submit(submitInfo);
    renderer_.GetGraphicsQueue().waitIdle();

    // 3. Download Result
    auto& outputTensor = tensors_[graph_.outputs[0]];
    std::vector<float> result(outputTensor.size / sizeof(float));
    void* outPtr = outputTensor.memory.mapMemory(0, VK_WHOLE_SIZE);
    std::memcpy(result.data(), outPtr, outputTensor.size);
    outputTensor.memory.unmapMemory();

    float outSum = 0;
    for (float v : result) outSum += std::abs(v);
    if (outSum == 0) {
        std::cout << "WARNING: NNEF output is all zeros. Intermediate junk or failed execution?" << std::endl;
    } else {
        // std::cout << "NNEF: Logits sample [0..4]: ";
        // for (int i = 0; i < std::min(5, (int)result.size()); ++i) std::cout << result[i] << " ";
        // std::cout << std::endl;
    }

    float maxVal = *std::max_element(result.begin(), result.end());
    float sum = 0.0f;
    for (float& val : result) {
        val = std::exp(val - maxVal);
        sum += val;
    }
    for (float& val : result) {
        val /= sum;
    }

    return result;
}
