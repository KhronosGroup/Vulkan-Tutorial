#include "renderer.h"
#include <fstream>
#include <stdexcept>
#include <array>
#include <iostream>

// This file contains compute-related methods from the Renderer class

// Create compute pipeline
bool Renderer::createComputePipeline() {
    try {
        // Read compute shader code
        auto computeShaderCode = readFile("shaders/compute.comp.spv");

        // Create shader module
        vk::raii::ShaderModule computeShaderModule = createShaderModule(computeShaderCode);

        // Create shader stage info
        vk::PipelineShaderStageCreateInfo computeShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *computeShaderModule,
            .pName = "main"
        };

        // Create compute descriptor set layout
        std::array<vk::DescriptorSetLayoutBinding, 4> computeBindings = {
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
                .pImmutableSamplers = nullptr
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
                .pImmutableSamplers = nullptr
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 2,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
                .pImmutableSamplers = nullptr
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 3,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
                .pImmutableSamplers = nullptr
            }
        };

        vk::DescriptorSetLayoutCreateInfo computeLayoutInfo{
            .bindingCount = static_cast<uint32_t>(computeBindings.size()),
            .pBindings = computeBindings.data()
        };

        computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, computeLayoutInfo);

        // Create compute pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*computeDescriptorSetLayout,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr
        };

        computePipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        // Create compute pipeline
        vk::ComputePipelineCreateInfo pipelineInfo{
            .stage = computeShaderStageInfo,
            .layout = *computePipelineLayout
        };

        computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);

        // Create compute descriptor pool
        std::array<vk::DescriptorPoolSize, 1> poolSizes = {
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 4u * MAX_FRAMES_IN_FLIGHT
            }
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        computeDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create compute pipeline: " << e.what() << std::endl;
        return false;
    }
}

// Dispatch compute shader
void Renderer::DispatchCompute(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
                              vk::Buffer inputBuffer, vk::Buffer outputBuffer,
                              vk::Buffer hrtfBuffer, vk::Buffer paramsBuffer) {
    try {
        // Create descriptor sets
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *computeDescriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &*computeDescriptorSetLayout
        };

        computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

        // Update descriptor sets
        vk::DescriptorBufferInfo inputBufferInfo{
            .buffer = inputBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        vk::DescriptorBufferInfo outputBufferInfo{
            .buffer = outputBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        vk::DescriptorBufferInfo hrtfBufferInfo{
            .buffer = hrtfBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        vk::DescriptorBufferInfo paramsBufferInfo{
            .buffer = paramsBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        std::array<vk::WriteDescriptorSet, 4> descriptorWrites = {
            vk::WriteDescriptorSet{
                .dstSet = computeDescriptorSets[0],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &inputBufferInfo
            },
            vk::WriteDescriptorSet{
                .dstSet = computeDescriptorSets[0],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &outputBufferInfo
            },
            vk::WriteDescriptorSet{
                .dstSet = computeDescriptorSets[0],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &hrtfBufferInfo
            },
            vk::WriteDescriptorSet{
                .dstSet = computeDescriptorSets[0],
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &paramsBufferInfo
            }
        };

        device.updateDescriptorSets(descriptorWrites, {});

        // Create command buffer
        vk::CommandBufferAllocateInfo cmdAllocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        auto commandBuffers = device.allocateCommandBuffers(cmdAllocInfo);
        vk::CommandBuffer cmdBuffer = commandBuffers[0];
        vk::raii::CommandBuffer commandBuffer(device, cmdBuffer, *commandPool);

        // Begin command buffer
        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        commandBuffer.begin(beginInfo);

        // Bind compute pipeline
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);

        // Bind descriptor sets
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, reinterpret_cast<std::vector<vk::DescriptorSet> &>(computeDescriptorSets), {});

        // Dispatch compute shader
        commandBuffer.dispatch(groupCountX, groupCountY, groupCountZ);

        // End command buffer
        commandBuffer.end();

        // Submit command buffer
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };

        computeQueue.submit(submitInfo, nullptr);

        // Wait for compute to complete
        computeQueue.waitIdle();
    } catch (const std::exception& e) {
        std::cerr << "Failed to dispatch compute shader: " << e.what() << std::endl;
    }
}
