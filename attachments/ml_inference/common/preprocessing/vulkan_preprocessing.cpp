#include "vulkan_preprocessing.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>

VulkanPreprocessor::VulkanPreprocessor(vk::raii::Device& dev,
                                       vk::raii::Queue& queue,
                                       vk::raii::CommandPool& pool,
                                       vk::raii::PhysicalDevice& physDev)
    : device(dev), computeQueue(queue), commandPool(pool), physicalDevice(physDev) {

    createBuffers();
    createDescriptorSetLayout();
    createPipelines();
    createDescriptorPool();
    createDescriptorSets();

    std::cout << "Vulkan preprocessor initialized\n";
}

void VulkanPreprocessor::createBuffers() {
    // Source image buffer (variable size, up to 16MB)
    createBuffer(MAX_SRC_IMAGE_SIZE,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                srcImageBuffer, srcImageMemory);

    // Resized image buffer (224x224x3 floats)
    size_t resizedSize = TARGET_SIZE * TARGET_SIZE * 3 * sizeof(float);
    createBuffer(resizedSize,
                vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                resizedBuffer, resizedMemory);

    // Normalized output (224x224x3 floats, NCHW)
    createBuffer(resizedSize,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                normalizedBuffer, normalizedMemory);

    // Staging buffer for upload/download
    createBuffer(MAX_SRC_IMAGE_SIZE,
                vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                stagingBuffer, stagingMemory);
}

void VulkanPreprocessor::createDescriptorSetLayout() {
    // Resize descriptor set layout (2 storage buffers)
    std::array<vk::DescriptorSetLayoutBinding, 2> resizeBindings{{
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    }};

    vk::DescriptorSetLayoutCreateInfo resizeLayoutInfo{
        .bindingCount = static_cast<uint32_t>(resizeBindings.size()),
        .pBindings = resizeBindings.data()
    };

    resizeSetLayout = vk::raii::DescriptorSetLayout(device, resizeLayoutInfo);

    // Normalize descriptor set layout (2 storage buffers)
    std::array<vk::DescriptorSetLayoutBinding, 2> normalizeBindings{{
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    }};

    vk::DescriptorSetLayoutCreateInfo normalizeLayoutInfo{
        .bindingCount = static_cast<uint32_t>(normalizeBindings.size()),
        .pBindings = normalizeBindings.data()
    };

    normalizeSetLayout = vk::raii::DescriptorSetLayout(device, normalizeLayoutInfo);
}

void VulkanPreprocessor::createPipelines() {
    // Load shader modules
    auto loadShader = [&](const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open shader: " + filename);
        }

        size_t fileSize = file.tellg();
        std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(code.data()), fileSize);
        file.close();

        vk::ShaderModuleCreateInfo createInfo{
            .codeSize = code.size() * sizeof(uint32_t),
            .pCode = code.data()
        };

        return vk::raii::ShaderModule(device, createInfo);
    };

    // Resize pipeline
    {
        auto shaderModule = loadShader("shaders/image_resize.comp.spv");

        struct ResizePushConstants {
            uint32_t srcWidth, srcHeight, dstWidth, dstHeight, isBgr, srcStep;
        };

        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(ResizePushConstants)
        };

        vk::PipelineLayoutCreateInfo layoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*resizeSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };

        resizeLayout = vk::raii::PipelineLayout(device, layoutInfo);

        vk::PipelineShaderStageCreateInfo stageInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName = "main"
        };

        vk::ComputePipelineCreateInfo pipelineInfo{
            .stage = stageInfo,
            .layout = *resizeLayout
        };

        resizePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    // Normalize pipeline
    {
        auto shaderModule = loadShader("shaders/normalize.comp.spv");

        struct NormalizePushConstants {
            float meanR, meanG, meanB;
            float stdR, stdG, stdB;
            uint32_t width, height;
        };

        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .offset = 0,
            .size = sizeof(NormalizePushConstants)
        };

        vk::PipelineLayoutCreateInfo layoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*normalizeSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange
        };

        normalizeLayout = vk::raii::PipelineLayout(device, layoutInfo);

        vk::PipelineShaderStageCreateInfo stageInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName = "main"
        };

        vk::ComputePipelineCreateInfo pipelineInfo{
            .stage = stageInfo,
            .layout = *normalizeLayout
        };

        normalizePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }
}

void VulkanPreprocessor::createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 1> poolSizes{{
        {vk::DescriptorType::eStorageBuffer, 4}  // 2 for resize, 2 for normalize
    }};

    vk::DescriptorPoolCreateInfo poolInfo{
        .maxSets = 2,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void VulkanPreprocessor::createDescriptorSets() {
    // Allocate descriptor sets
    std::array<vk::DescriptorSetLayout, 2> layouts = {*resizeSetLayout, *normalizeSetLayout};

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = 2,
        .pSetLayouts = layouts.data()
    };

    auto sets = vk::raii::DescriptorSets(device, allocInfo);
    resizeDescriptorSet = std::move(sets[0]);
    normalizeDescriptorSet = std::move(sets[1]);

    // Update resize descriptor set
    {
        vk::DescriptorBufferInfo srcInfo{
            .buffer = *srcImageBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        vk::DescriptorBufferInfo dstInfo{
            .buffer = *resizedBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        std::array<vk::WriteDescriptorSet, 2> writes{{
            {
                .dstSet = *resizeDescriptorSet,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &srcInfo
            },
            {
                .dstSet = *resizeDescriptorSet,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &dstInfo
            }
        }};

        device.updateDescriptorSets(writes, nullptr);
    }

    // Update normalize descriptor set
    {
        vk::DescriptorBufferInfo srcInfo{
            .buffer = *resizedBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        vk::DescriptorBufferInfo dstInfo{
            .buffer = *normalizedBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE
        };

        std::array<vk::WriteDescriptorSet, 2> writes{{
            {
                .dstSet = *normalizeDescriptorSet,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &srcInfo
            },
            {
                .dstSet = *normalizeDescriptorSet,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &dstInfo
            }
        }};

        device.updateDescriptorSets(writes, nullptr);
    }
}

PreprocessedImage VulkanPreprocessor::preprocess(const unsigned char* imageData,
                                                  int width, int height, size_t step, bool isBgr) {
    // auto startTime = std::chrono::high_resolution_clock::now();

    uploadImage(imageData, width, height, step);
    dispatchResize(width, height, step, isBgr);
    dispatchNormalize();
    auto result = downloadResult();

    // auto endTime = std::chrono::high_resolution_clock::now();
    /*
    float preprocessTime = std::chrono::duration<float, std::milli>(
        endTime - startTime).count();

    // std::cout << "Vulkan preprocessing: " << preprocessTime << " ms\n";
    */

    return result;
}

void VulkanPreprocessor::uploadImage(const unsigned char* imageData, int width, int height, size_t step) {
    size_t rowSize = width * 3;
    size_t imageSize = rowSize * height;

    // Map staging buffer and copy data
    void* mapped = stagingMemory.mapMemory(0, imageSize);
    if (step == rowSize) {
        std::memcpy(mapped, imageData, imageSize);
    } else {
        // Handle row padding (stride)
        for (int i = 0; i < height; ++i) {
            std::memcpy((uint8_t*)mapped + i * rowSize, imageData + i * step, rowSize);
        }
    }
    stagingMemory.unmapMemory();

    // Copy from staging to device buffer
    copyBuffer(stagingBuffer, srcImageBuffer, imageSize);
}

void VulkanPreprocessor::dispatchResize(int srcWidth, int srcHeight, size_t /*srcStep*/, bool isBgr) {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };

    auto cmdBuffers = vk::raii::CommandBuffers(device, allocInfo);
    auto& cmd = cmdBuffers[0];

    cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *resizePipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *resizeLayout, 0,
                          {*resizeDescriptorSet}, nullptr);

    struct {
        uint32_t srcWidth, srcHeight, dstWidth, dstHeight, isBgr, srcStep;
    } pushConstants{
        static_cast<uint32_t>(srcWidth),
        static_cast<uint32_t>(srcHeight),
        TARGET_SIZE,
        TARGET_SIZE,
        isBgr ? 1u : 0u,
        static_cast<uint32_t>(srcWidth * 3) // We already packed the image in uploadImage
    };

    cmd.pushConstants(*resizeLayout, vk::ShaderStageFlagBits::eCompute,
                     0, vk::ArrayProxy<const uint32_t>(6, &pushConstants.srcWidth));

    // Dispatch workgroups
    uint32_t workGroupsX = (TARGET_SIZE + 15) / 16;
    uint32_t workGroupsY = (TARGET_SIZE + 15) / 16;
    cmd.dispatch(workGroupsX, workGroupsY, 1);

    cmd.end();

    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd
    };

    computeQueue.submit(submitInfo);
    computeQueue.waitIdle();
}

void VulkanPreprocessor::dispatchNormalize() {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };

    auto cmdBuffers = vk::raii::CommandBuffers(device, allocInfo);
    auto& cmd = cmdBuffers[0];

    cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *normalizePipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *normalizeLayout, 0,
                          {*normalizeDescriptorSet}, nullptr);

    struct {
        float meanR, meanG, meanB;
        float stdR, stdG, stdB;
        uint32_t width, height;
    } pushConstants{
        MEAN[0], MEAN[1], MEAN[2],
        STD[0], STD[1], STD[2],
        TARGET_SIZE, TARGET_SIZE
    };

    cmd.pushConstants(*normalizeLayout, vk::ShaderStageFlagBits::eCompute,
                     0, vk::ArrayProxy<const float>(8, &pushConstants.meanR));

    // Dispatch workgroups
    uint32_t workGroupsX = (TARGET_SIZE + 15) / 16;
    uint32_t workGroupsY = (TARGET_SIZE + 15) / 16;
    cmd.dispatch(workGroupsX, workGroupsY, 1);

    cmd.end();

    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd
    };

    computeQueue.submit(submitInfo);
    computeQueue.waitIdle();
}

PreprocessedImage VulkanPreprocessor::downloadResult() {
    PreprocessedImage result;
    result.data.resize(1 * 3 * TARGET_SIZE * TARGET_SIZE);
    result.shape = {1, 3, TARGET_SIZE, TARGET_SIZE};

    // Copy from device to staging buffer
    size_t resultSize = result.data.size() * sizeof(float);
    copyBuffer(normalizedBuffer, stagingBuffer, resultSize);

    // Map and read
    void* mapped = stagingMemory.mapMemory(0, resultSize);
    std::memcpy(result.data.data(), mapped, resultSize);
    stagingMemory.unmapMemory();

    return result;
}

void VulkanPreprocessor::createBuffer(vk::DeviceSize size,
                                     vk::BufferUsageFlags usage,
                                     vk::MemoryPropertyFlags properties,
                                     vk::raii::Buffer& buffer,
                                     vk::raii::DeviceMemory& memory) {
    vk::BufferCreateInfo bufferInfo{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive
    };

    buffer = vk::raii::Buffer(device, bufferInfo);

    auto memReqs = buffer.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{
        .allocationSize = memReqs.size,
        .memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties)
    };

    memory = vk::raii::DeviceMemory(device, allocInfo);
    buffer.bindMemory(*memory, 0);
}

uint32_t VulkanPreprocessor::findMemoryType(uint32_t typeFilter,
                                            vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

void VulkanPreprocessor::copyBuffer(vk::raii::Buffer& src, vk::raii::Buffer& dst,
                                   vk::DeviceSize size) {
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };

    auto cmdBuffers = vk::raii::CommandBuffers(device, allocInfo);
    auto& cmd = cmdBuffers[0];

    cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::BufferCopy copyRegion{.size = size};
    cmd.copyBuffer(*src, *dst, copyRegion);

    cmd.end();

    vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd
    };

    computeQueue.submit(submitInfo);
    computeQueue.waitIdle();
}
