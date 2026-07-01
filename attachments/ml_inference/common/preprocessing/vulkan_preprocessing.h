#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <memory>
#include <array>

struct PreprocessedImage {
    std::vector<float> data;
    std::vector<int64_t> shape;
};

class VulkanPreprocessor {
public:
    VulkanPreprocessor(vk::raii::Device& device,
                      vk::raii::Queue& computeQueue,
                      vk::raii::CommandPool& commandPool,
                      vk::raii::PhysicalDevice& physicalDevice);

    ~VulkanPreprocessor() = default;

    PreprocessedImage preprocess(const unsigned char* imageData, int width, int height, size_t step, bool isBgr = false);

private:
    void createBuffers();
    void createDescriptorSetLayout();
    void createPipelines();
    void createDescriptorPool();
    void createDescriptorSets();

    void uploadImage(const unsigned char* imageData, int width, int height, size_t step);
    void dispatchResize(int srcWidth, int srcHeight, size_t srcStep, bool isBgr);
    void dispatchNormalize();
    PreprocessedImage downloadResult();

    void createBuffer(vk::DeviceSize size,
                     vk::BufferUsageFlags usage,
                     vk::MemoryPropertyFlags properties,
                     vk::raii::Buffer& buffer,
                     vk::raii::DeviceMemory& memory);

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);

    void copyBuffer(vk::raii::Buffer& src, vk::raii::Buffer& dst, vk::DeviceSize size);

    vk::raii::Device& device;
    vk::raii::Queue& computeQueue;
    vk::raii::CommandPool& commandPool;
    vk::raii::PhysicalDevice& physicalDevice;

    // Pipelines
    vk::raii::Pipeline resizePipeline = nullptr;
    vk::raii::Pipeline normalizePipeline = nullptr;
    vk::raii::PipelineLayout resizeLayout = nullptr;
    vk::raii::PipelineLayout normalizeLayout = nullptr;

    // Descriptor sets
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSetLayout resizeSetLayout = nullptr;
    vk::raii::DescriptorSetLayout normalizeSetLayout = nullptr;
    vk::raii::DescriptorSet resizeDescriptorSet = nullptr;
    vk::raii::DescriptorSet normalizeDescriptorSet = nullptr;

    // Buffers
    vk::raii::Buffer srcImageBuffer = nullptr;
    vk::raii::DeviceMemory srcImageMemory = nullptr;

    vk::raii::Buffer resizedBuffer = nullptr;
    vk::raii::DeviceMemory resizedMemory = nullptr;

    vk::raii::Buffer normalizedBuffer = nullptr;
    vk::raii::DeviceMemory normalizedMemory = nullptr;

    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingMemory = nullptr;

    static constexpr int TARGET_SIZE = 224;
    static constexpr size_t MAX_SRC_IMAGE_SIZE = 16 * 1024 * 1024;  // 16MB for source images

    // ImageNet normalization constants
    static constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
    static constexpr std::array<float, 3> STD = {0.229f, 0.224f, 0.225f};
};
