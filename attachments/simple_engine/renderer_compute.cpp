#include "renderer.h"
#include <fstream>
#include <array>
#include <iostream>

// This file contains compute-related methods from the Renderer class

// Create compute pipeline
bool Renderer::createComputePipeline() {
    try {
        // Read compute shader code
        auto computeShaderCode = readFile("shaders/hrtf.spv");

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
                .descriptorType = vk::DescriptorType::eUniformBuffer,
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
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 3u * MAX_FRAMES_IN_FLIGHT
            },
            vk::DescriptorPoolSize{
                .type = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = 1u * MAX_FRAMES_IN_FLIGHT
            }
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = MAX_FRAMES_IN_FLIGHT,
            .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
            .pPoolSizes = poolSizes.data()
        };

        computeDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

        return createComputeCommandPool();
    } catch (const std::exception& e) {
        std::cerr << "Failed to create compute pipeline: " << e.what() << std::endl;
        return false;
    }
}

// Create compute command pool
bool Renderer::createComputeCommandPool() {
    try {
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueFamilyIndices.computeFamily.value()
        };

        computeCommandPool = vk::raii::CommandPool(device, poolInfo);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create compute command pool: " << e.what() << std::endl;
        return false;
    }
}

// Dispatch compute shader
vk::raii::Fence Renderer::DispatchCompute(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
                                         vk::Buffer inputBuffer, vk::Buffer outputBuffer,
                                         vk::Buffer hrtfBuffer, vk::Buffer paramsBuffer) {
    try {
        // Create fence for synchronization
        vk::FenceCreateInfo fenceInfo{};
        vk::raii::Fence computeFence(device, fenceInfo);

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
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &paramsBufferInfo
            }
        };

        device.updateDescriptorSets(descriptorWrites, {});

        // Create command buffer using dedicated compute command pool
        vk::CommandBufferAllocateInfo cmdAllocInfo{
            .commandPool = *computeCommandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        auto commandBuffers = device.allocateCommandBuffers(cmdAllocInfo);
        // Use RAII wrapper temporarily for recording to preserve dispatch loader
        vk::raii::CommandBuffer commandBufferRaii = std::move(commandBuffers[0]);

        // Begin command buffer
        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        commandBufferRaii.begin(beginInfo);

        // Bind compute pipeline
        commandBufferRaii.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);

        // Bind descriptor sets - properly convert RAII descriptor set to regular descriptor set
        std::vector<vk::DescriptorSet> descriptorSetsToBindRaw;
        descriptorSetsToBindRaw.reserve(1);
        descriptorSetsToBindRaw.push_back(*computeDescriptorSets[0]);
        commandBufferRaii.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, descriptorSetsToBindRaw, {});

        // Dispatch compute shader
        commandBufferRaii.dispatch(groupCountX, groupCountY, groupCountZ);

        // End command buffer
        commandBufferRaii.end();

        // Extract raw command buffer for submission and release RAII ownership
        // This prevents premature destruction while preserving the recorded commands
        vk::CommandBuffer rawCommandBuffer = *commandBufferRaii;
        commandBufferRaii.release(); // Release RAII ownership to prevent destruction

        // Submit command buffer with fence for synchronization
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &rawCommandBuffer
        };

        // Use mutex to ensure thread-safe access to compute queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            computeQueue.submit(submitInfo, *computeFence);
        }

        // Return fence for non-blocking synchronization
        return computeFence;
    } catch (const std::exception& e) {
        std::cerr << "Failed to dispatch compute shader: " << e.what() << std::endl;
        // Return a null fence on error
        vk::FenceCreateInfo fenceInfo{};
        return {device, fenceInfo};
    }
}
