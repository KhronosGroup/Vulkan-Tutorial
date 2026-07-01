#include "vulkan_mnist_inference.h"
#include "nnef_loader.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

VulkanMNISTInference::VulkanMNISTInference(Renderer& renderer)
    : renderer_(renderer) {
}

VulkanMNISTInference::~VulkanMNISTInference() {
    destroyResources();
}

bool VulkanMNISTInference::loadWeights(const std::string& path) {
    try {
        if (std::filesystem::is_directory(path) || path.ends_with(".nnef") || std::filesystem::exists(path + "/graph.nnef")) {
            weights_ = NNEFLoader::load(path);
        } else {
            weights_ = WeightLoader::load(path);
        }
        createShaderModules();
        createBuffers();
        createPipelines();
        weightsLoaded_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load weights: " << e.what() << "\n";
        return false;
    }
}

vk::raii::ShaderModule VulkanMNISTInference::loadShader(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open shader file: " + path);
    }

    size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    vk::ShaderModuleCreateInfo createInfo{
        .codeSize = buffer.size(),
        .pCode = reinterpret_cast<const uint32_t*>(buffer.data())
    };

    return vk::raii::ShaderModule(renderer_.GetRaiiDevice(), createInfo);
}

void VulkanMNISTInference::createShaderModules() {
    denseShader_ = loadShader("shaders/dense.comp.spv");
    reluShader_ = loadShader("shaders/relu.comp.spv");
    softmaxShader_ = loadShader("shaders/softmax.comp.spv");
}

void VulkanMNISTInference::createBuffers() {
    auto& device = renderer_.GetRaiiDevice();
    auto& physicalDevice = renderer_.GetPhysicalDevice();

    // Helper lambda to create buffer and memory
    auto createBuffer = [&](vk::DeviceSize size, vk::BufferUsageFlags usage,
                           vk::raii::Buffer& buffer, vk::raii::DeviceMemory& memory) {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };
        buffer = vk::raii::Buffer(device, bufferInfo);

        vk::MemoryRequirements memReqs = buffer.getMemoryRequirements();
        vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();

        uint32_t memoryTypeIndex = 0;
        vk::MemoryPropertyFlags requiredProps = vk::MemoryPropertyFlagBits::eHostVisible |
                                                 vk::MemoryPropertyFlagBits::eHostCoherent;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((memReqs.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & requiredProps) == requiredProps) {
                memoryTypeIndex = i;
                break;
            }
        }

        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReqs.size,
            .memoryTypeIndex = memoryTypeIndex
        };
        memory = vk::raii::DeviceMemory(device, allocInfo);
        buffer.bindMemory(*memory, 0);
    };

    // Create all buffers
    createBuffer(784 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer, inputBuffer_, inputMemory_);

    createBuffer(weights_.fc1_weights.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                fc1WeightBuffer_, fc1WeightMemory_);
    createBuffer(weights_.fc1_bias.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                fc1BiasBuffer_, fc1BiasMemory_);
    createBuffer(weights_.fc1_bias.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                fc1OutputBuffer_, fc1OutputMemory_);

    createBuffer(weights_.fc1_bias.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                reluOutputBuffer_, reluOutputMemory_);

    createBuffer(weights_.fc2_weights.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                fc2WeightBuffer_, fc2WeightMemory_);
    createBuffer(weights_.fc2_bias.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                fc2BiasBuffer_, fc2BiasMemory_);
    createBuffer(10 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                fc2OutputBuffer_, fc2OutputMemory_);

    createBuffer(10 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer,
                softmaxOutputBuffer_, softmaxOutputMemory_);

    // Upload weights to GPU
    void* data = fc1WeightMemory_.mapMemory(0, weights_.fc1_weights.size() * sizeof(float));
    std::memcpy(data, weights_.fc1_weights.data(), weights_.fc1_weights.size() * sizeof(float));
    fc1WeightMemory_.unmapMemory();

    data = fc1BiasMemory_.mapMemory(0, weights_.fc1_bias.size() * sizeof(float));
    std::memcpy(data, weights_.fc1_bias.data(), weights_.fc1_bias.size() * sizeof(float));
    fc1BiasMemory_.unmapMemory();

    data = fc2WeightMemory_.mapMemory(0, weights_.fc2_weights.size() * sizeof(float));
    std::memcpy(data, weights_.fc2_weights.data(), weights_.fc2_weights.size() * sizeof(float));
    fc2WeightMemory_.unmapMemory();

    data = fc2BiasMemory_.mapMemory(0, weights_.fc2_bias.size() * sizeof(float));
    std::memcpy(data, weights_.fc2_bias.data(), weights_.fc2_bias.size() * sizeof(float));
    fc2BiasMemory_.unmapMemory();
}

void VulkanMNISTInference::createPipelines() {
    auto& device = renderer_.GetRaiiDevice();

    // Create descriptor set layouts
    std::vector<vk::DescriptorSetLayoutBinding> denseBindings = {
        {.binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 3, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}
    };
    vk::DescriptorSetLayoutCreateInfo denseLayoutInfo{
        .bindingCount = static_cast<uint32_t>(denseBindings.size()),
        .pBindings = denseBindings.data()
    };
    denseLayout_ = vk::raii::DescriptorSetLayout(device, denseLayoutInfo);

    std::vector<vk::DescriptorSetLayoutBinding> reluBindings = {
        {.binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}
    };
    vk::DescriptorSetLayoutCreateInfo reluLayoutInfo{
        .bindingCount = static_cast<uint32_t>(reluBindings.size()),
        .pBindings = reluBindings.data()
    };
    reluLayout_ = vk::raii::DescriptorSetLayout(device, reluLayoutInfo);
    softmaxLayout_ = vk::raii::DescriptorSetLayout(device, reluLayoutInfo); // Same layout as ReLU

    // Create pipeline layouts with push constants
    vk::PushConstantRange densePushConstant{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 12 // 3 uint32_t
    };
    vk::PipelineLayoutCreateInfo densePipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*denseLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &densePushConstant
    };
    densePipelineLayout_ = vk::raii::PipelineLayout(device, densePipelineLayoutInfo);

    vk::PushConstantRange reluPushConstant{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 4 // 1 uint32_t
    };
    vk::PipelineLayoutCreateInfo reluPipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*reluLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &reluPushConstant
    };
    reluPipelineLayout_ = vk::raii::PipelineLayout(device, reluPipelineLayoutInfo);

    vk::PushConstantRange softmaxPushConstant{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 4 // 1 uint32_t
    };
    vk::PipelineLayoutCreateInfo softmaxPipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*softmaxLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &softmaxPushConstant
    };
    softmaxPipelineLayout_ = vk::raii::PipelineLayout(device, softmaxPipelineLayoutInfo);

    // Create compute pipelines
    vk::PipelineShaderStageCreateInfo denseStage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *denseShader_,
        .pName = "main"
    };
    vk::ComputePipelineCreateInfo densePipelineInfo{
        .stage = denseStage,
        .layout = *densePipelineLayout_
    };
    densePipeline_ = vk::raii::Pipeline(device, nullptr, densePipelineInfo);

    vk::PipelineShaderStageCreateInfo reluStage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *reluShader_,
        .pName = "main"
    };
    vk::ComputePipelineCreateInfo reluPipelineInfo{
        .stage = reluStage,
        .layout = *reluPipelineLayout_
    };
    reluPipeline_ = vk::raii::Pipeline(device, nullptr, reluPipelineInfo);

    vk::PipelineShaderStageCreateInfo softmaxStage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *softmaxShader_,
        .pName = "main"
    };
    vk::ComputePipelineCreateInfo softmaxPipelineInfo{
        .stage = softmaxStage,
        .layout = *softmaxPipelineLayout_
    };
    softmaxPipeline_ = vk::raii::Pipeline(device, nullptr, softmaxPipelineInfo);

    // Create descriptor pool and sets
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 12}
    };
    vk::DescriptorPoolCreateInfo poolInfo{
        .maxSets = 4,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    descriptorPool_ = vk::raii::DescriptorPool(device, poolInfo);

    // Allocate descriptor sets
    std::vector<vk::DescriptorSetLayout> layouts = {*denseLayout_, *reluLayout_, *denseLayout_, *softmaxLayout_};
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool_,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()
    };
    auto sets = vk::raii::DescriptorSets(device, allocInfo);
    denseDescSet1_ = std::move(sets[0]);
    reluDescSet_ = std::move(sets[1]);
    denseDescSet2_ = std::move(sets[2]);
    softmaxDescSet_ = std::move(sets[3]);

    // Update descriptor sets
    vk::DescriptorBufferInfo inputInfo{.buffer = *inputBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo fc1WeightInfo{.buffer = *fc1WeightBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo fc1BiasInfo{.buffer = *fc1BiasBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo fc1OutputInfo{.buffer = *fc1OutputBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo reluOutputInfo{.buffer = *reluOutputBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo fc2WeightInfo{.buffer = *fc2WeightBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo fc2BiasInfo{.buffer = *fc2BiasBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo fc2OutputInfo{.buffer = *fc2OutputBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo softmaxOutputInfo{.buffer = *softmaxOutputBuffer_, .offset = 0, .range = VK_WHOLE_SIZE};

    std::vector<vk::WriteDescriptorSet> writes = {
        // Dense layer 1
        {.dstSet = *denseDescSet1_, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &inputInfo},
        {.dstSet = *denseDescSet1_, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &fc1WeightInfo},
        {.dstSet = *denseDescSet1_, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &fc1BiasInfo},
        {.dstSet = *denseDescSet1_, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &fc1OutputInfo},
        // ReLU
        {.dstSet = *reluDescSet_, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &fc1OutputInfo},
        {.dstSet = *reluDescSet_, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &reluOutputInfo},
        // Dense layer 2
        {.dstSet = *denseDescSet2_, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &reluOutputInfo},
        {.dstSet = *denseDescSet2_, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &fc2WeightInfo},
        {.dstSet = *denseDescSet2_, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &fc2BiasInfo},
        {.dstSet = *denseDescSet2_, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &fc2OutputInfo},
        // Softmax
        {.dstSet = *softmaxDescSet_, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &fc2OutputInfo},
        {.dstSet = *softmaxDescSet_, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &softmaxOutputInfo}
    };
    device.updateDescriptorSets(writes, nullptr);
}

std::vector<float> VulkanMNISTInference::preprocess(const std::vector<float>& input) {
    // 1. Find bounding box
    int minX = 28, maxX = 0, minY = 28, maxY = 0;
    bool empty = true;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            if (input[y * 28 + x] > 0.05f) {
                minX = std::min(minX, x);
                maxX = std::max(maxX, x);
                minY = std::min(minY, y);
                maxY = std::max(maxY, y);
                empty = false;
            }
        }
    }

    if (empty) return input;

    int width = maxX - minX + 1;
    int height = maxY - minY + 1;

    // 2. Scale digit to fit in a 20x20 box (standard MNIST)
    float scale = 20.0f / std::max(width, height);
    
    std::vector<float> scaled(28 * 28, 0.0f);
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            // Source coordinates relative to bounding box center
            float sx = (x - 13.5f) / scale + (minX + maxX) / 2.0f;
            float sy = (y - 13.5f) / scale + (minY + maxY) / 2.0f;
            
            int isx = static_cast<int>(std::floor(sx));
            int isy = static_cast<int>(std::floor(sy));
            
            if (isx >= 0 && isx < 28 && isy >= 0 && isy < 28) {
                // Bilinear interpolation
                float fx = sx - isx;
                float fy = sy - isy;
                
                float v00 = input[isy * 28 + isx];
                float v10 = (isx + 1 < 28) ? input[isy * 28 + isx + 1] : 0.0f;
                float v01 = (isy + 1 < 28) ? input[(isy + 1) * 28 + isx] : 0.0f;
                float v11 = (isx + 1 < 28 && isy + 1 < 28) ? input[(isy + 1) * 28 + isx + 1] : 0.0f;
                
                scaled[y * 28 + x] = v00 * (1 - fx) * (1 - fy) +
                                    v10 * fx * (1 - fy) +
                                    v01 * (1 - fx) * fy +
                                    v11 * fx * fy;
            }
        }
    }

    // 3. Center by center of mass
    float xSum = 0, ySum = 0, totalMass = 0;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float val = scaled[y * 28 + x];
            if (val > 0.01f) {
                totalMass += val;
                xSum += x * val;
                ySum += y * val;
            }
        }
    }

    if (totalMass < 0.01f) return scaled;

    float xCoM = xSum / totalMass;
    float yCoM = ySum / totalMass;

    float dx = 13.5f - xCoM;
    float dy = 13.5f - yCoM;

    std::vector<float> centered(784, 0.0f);
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float ox = x - dx;
            float oy = y - dy;
            int iox = static_cast<int>(std::round(ox));
            int ioy = static_cast<int>(std::round(oy));
            if (iox >= 0 && iox < 28 && ioy >= 0 && ioy < 28) {
                centered[y * 28 + x] = scaled[ioy * 28 + iox];
            }
        }
    }

    return centered;
}

std::vector<float> VulkanMNISTInference::infer(const std::vector<float>& input) {
    std::vector<float> output(10, 0.0f);
    if (!weightsLoaded_ || input.size() != 784) {
        return output;
    }

    // Preprocess input (centering and scaling)
    std::vector<float> preprocessedInput = preprocess(input);

    // Upload input data
    std::vector<float> normalizedInput(784);
    for (size_t i = 0; i < 784; ++i) {
        normalizedInput[i] = (preprocessedInput[i] - 0.1307f) / 0.3081f;
    }

    void* data = inputMemory_.mapMemory(0, 784 * sizeof(float));
    std::memcpy(data, normalizedInput.data(), 784 * sizeof(float));
    inputMemory_.unmapMemory();

    // Create command buffer
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *renderer_.GetCommandPool(),
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    auto cmdBuffers = renderer_.GetRaiiDevice().allocateCommandBuffers(allocInfo);
    vk::raii::CommandBuffer& cmd = cmdBuffers[0];

    cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // FC1: 784 -> 128
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *densePipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *densePipelineLayout_, 0, {*denseDescSet1_}, nullptr);
    uint32_t fc1Params[3] = {784, static_cast<uint32_t>(weights_.fc1_bias.size()), 1};
    cmd.pushConstants<uint32_t>(*densePipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, fc1Params);
    cmd.dispatch((weights_.fc1_bias.size() + 255) / 256, 1, 1);

    vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead
    };
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                       {}, {barrier}, nullptr, nullptr);

    // ReLU
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *reluPipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *reluPipelineLayout_, 0, {*reluDescSet_}, nullptr);
    uint32_t reluSize = weights_.fc1_bias.size();
    cmd.pushConstants<uint32_t>(*reluPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, {reluSize});
    cmd.dispatch((reluSize + 255) / 256, 1, 1);

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                       {}, {barrier}, nullptr, nullptr);

    // FC2: 128 -> 10
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *densePipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *densePipelineLayout_, 0, {*denseDescSet2_}, nullptr);
    uint32_t fc2Params[3] = {static_cast<uint32_t>(weights_.fc1_bias.size()), 10, 1};
    cmd.pushConstants<uint32_t>(*densePipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, fc2Params);
    cmd.dispatch((10 + 255) / 256, 1, 1);

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                       {}, {barrier}, nullptr, nullptr);

    // Softmax (single pass)
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *softmaxPipeline_);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *softmaxPipelineLayout_, 0, {*softmaxDescSet_}, nullptr);
    cmd.pushConstants<uint32_t>(*softmaxPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, {10});
    cmd.dispatch(1, 1, 1);

    cmd.end();

    // Submit and wait
    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd
    };
    renderer_.GetGraphicsQueue().submit({submitInfo}, nullptr);
    renderer_.GetGraphicsQueue().waitIdle();

    // Read back results
    data = softmaxOutputMemory_.mapMemory(0, 10 * sizeof(float));
    std::memcpy(output.data(), data, 10 * sizeof(float));
    softmaxOutputMemory_.unmapMemory();

    return output;
}

void VulkanMNISTInference::destroyResources() {
    // RAII handles cleanup automatically
}
