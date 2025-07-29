#include "memory_pool.h"
#include <iostream>
#include <algorithm>

MemoryPool::MemoryPool(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice)
    : device(device), physicalDevice(physicalDevice) {
}

MemoryPool::~MemoryPool() {
    // RAII will handle cleanup automatically
    std::lock_guard lock(poolMutex);
    pools.clear();
}

bool MemoryPool::initialize() {
    std::lock_guard lock(poolMutex);

    try {
        // Configure default pool settings based on typical usage patterns

        // Vertex buffer pool: Large allocations, device-local (increased for large models like bistro)
        configurePool(
            PoolType::VERTEX_BUFFER,
            128 * 1024 * 1024, // 128MB blocks (doubled)
            4096,               // 4KB allocation units
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            16                  // Max 16 blocks (2GB total, quadrupled)
        );

        // Index buffer pool: Medium allocations, device-local (increased for large models like bistro)
        configurePool(
            PoolType::INDEX_BUFFER,
            64 * 1024 * 1024,  // 64MB blocks (doubled)
            2048,               // 2KB allocation units
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            8                   // Max 8 blocks (512MB total, quadrupled)
        );

        // Uniform buffer pool: Small allocations, host-visible
        configurePool(
            PoolType::UNIFORM_BUFFER,
            4 * 1024 * 1024,   // 4MB blocks
            256,                // 256B allocation units
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            4                   // Max 4 blocks (16MB total)
        );

        // Staging buffer pool: Variable allocations, host-visible
        configurePool(
            PoolType::STAGING_BUFFER,
            16 * 1024 * 1024,  // 16MB blocks
            1024,               // 1KB allocation units
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            4                   // Max 4 blocks (64MB total)
        );

        // Texture image pool: Large allocations, device-local (significantly increased for large models like bistro)
        configurePool(
            PoolType::TEXTURE_IMAGE,
            256 * 1024 * 1024, // 256MB blocks (doubled)
            4096,               // 4KB allocation units
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            24                  // Max 24 blocks (6GB total, 6x increase)
        );

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize memory pool: " << e.what() << std::endl;
        return false;
    }
}

void MemoryPool::configurePool(
    const PoolType poolType,
    const vk::DeviceSize blockSize,
    const vk::DeviceSize allocationUnit,
    const vk::MemoryPropertyFlags properties,
    const uint32_t maxBlocks) {

    PoolConfig config;
    config.blockSize = blockSize;
    config.allocationUnit = allocationUnit;
    config.properties = properties;
    config.maxBlocks = maxBlocks;

    poolConfigs[poolType] = config;
}

uint32_t MemoryPool::findMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties) const {
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type");
}

std::unique_ptr<MemoryPool::MemoryBlock> MemoryPool::createMemoryBlock(PoolType poolType, vk::DeviceSize size) {
    auto configIt = poolConfigs.find(poolType);
    if (configIt == poolConfigs.end()) {
        throw std::runtime_error("Pool type not configured");
    }

    const PoolConfig& config = configIt->second;

    // Use the larger of the requested size or configured block size
    const vk::DeviceSize blockSize = std::max(size, config.blockSize);

    // Create a dummy buffer to get memory requirements for the memory type
    vk::BufferCreateInfo bufferInfo{
        .size = blockSize,
        .usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
                 vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                 vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive
    };

    vk::raii::Buffer dummyBuffer(device, bufferInfo);
    vk::MemoryRequirements memRequirements = dummyBuffer.getMemoryRequirements();

    uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, config.properties);

    // Allocate the memory block
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize = blockSize,
        .memoryTypeIndex = memoryTypeIndex
    };

    // Create MemoryBlock with proper initialization to avoid default constructor issues
    auto block = std::unique_ptr<MemoryBlock>(new MemoryBlock{
        .memory = vk::raii::DeviceMemory(device, allocInfo),
        .size = blockSize,
        .used = 0,
        .memoryTypeIndex = memoryTypeIndex,
        .isMapped = false,
        .mappedPtr = nullptr,
        .freeList = {},
        .allocationUnit = config.allocationUnit
    });

    // Map memory if it's host-visible
    block->isMapped = (config.properties & vk::MemoryPropertyFlagBits::eHostVisible) != vk::MemoryPropertyFlags{};
    if (block->isMapped) {
        block->mappedPtr = block->memory.mapMemory(0, blockSize);
    } else {
        block->mappedPtr = nullptr;
    }

    // Initialize a free list
    const size_t numUnits = blockSize / config.allocationUnit;
    block->freeList.resize(numUnits, true);  // All units initially free


    return block;
}

MemoryPool::MemoryBlock* MemoryPool::findSuitableBlock(PoolType poolType, vk::DeviceSize size, vk::DeviceSize alignment) {
    auto poolIt = pools.find(poolType);
    if (poolIt == pools.end()) {
        pools[poolType] = std::vector<std::unique_ptr<MemoryBlock>>();
        poolIt = pools.find(poolType);
    }

    auto& poolBlocks = poolIt->second;
    const PoolConfig& config = poolConfigs[poolType];

    // Calculate required units (accounting for alignment)
    const vk::DeviceSize alignedSize = ((size + alignment - 1) / alignment) * alignment;
    const size_t requiredUnits = (alignedSize + config.allocationUnit - 1) / config.allocationUnit;

    // Search existing blocks for sufficient free space
    for (const auto& block : poolBlocks) {
        // Find consecutive free units
        size_t consecutiveFree = 0;
        for (size_t i = 0; i < block->freeList.size(); ++i) {
            if (block->freeList[i]) {
                consecutiveFree++;
                if (consecutiveFree >= requiredUnits) {
                    return block.get();
                }
            } else {
                consecutiveFree = 0;
            }
        }
    }

    // No suitable block found, create a new one if we haven't reached the limit AND rendering is not active
    if (poolBlocks.size() < config.maxBlocks) {
        if (renderingActive) {
            std::cerr << "ERROR: Attempted to create new memory block during rendering! Pool type: "
                      << static_cast<int>(poolType) << ", required size: " << alignedSize << std::endl;
            std::cerr << "This violates the constraint that no new memory should be allocated during PBR rendering." << std::endl;
            return nullptr;
        }

        try {
            auto newBlock = createMemoryBlock(poolType, alignedSize);
            poolBlocks.push_back(std::move(newBlock));
            std::cout << "Created new memory block during initialization (pool type: "
                      << static_cast<int>(poolType) << ")" << std::endl;
            return poolBlocks.back().get();
        } catch (const std::exception& e) {
            std::cerr << "Failed to create new memory block: " << e.what() << std::endl;
            return nullptr;
        }
    }

    std::cerr << "Memory pool exhausted for pool type " << static_cast<int>(poolType) << std::endl;
    return nullptr;
}

std::unique_ptr<MemoryPool::Allocation> MemoryPool::allocate(PoolType poolType, vk::DeviceSize size, vk::DeviceSize alignment) {
    std::lock_guard<std::mutex> lock(poolMutex);

    MemoryBlock* block = findSuitableBlock(poolType, size, alignment);
    if (!block) {
        return nullptr;
    }

    const PoolConfig& config = poolConfigs[poolType];

    // Calculate required units (accounting for alignment)
    const vk::DeviceSize alignedSize = ((size + alignment - 1) / alignment) * alignment;
    const size_t requiredUnits = (alignedSize + config.allocationUnit - 1) / config.allocationUnit;

    // Find consecutive free units
    size_t startUnit = 0;
    size_t consecutiveFree = 0;
    bool found = false;

    for (size_t i = 0; i < block->freeList.size(); ++i) {
        if (block->freeList[i]) {
            if (consecutiveFree == 0) {
                startUnit = i;
            }
            consecutiveFree++;
            if (consecutiveFree >= requiredUnits) {
                found = true;
                break;
            }
        } else {
            consecutiveFree = 0;
        }
    }

    if (!found) {
        return nullptr;
    }

    // Mark units as used
    for (size_t i = startUnit; i < startUnit + requiredUnits; ++i) {
        block->freeList[i] = false;
    }

    // Create allocation info
    auto allocation = std::make_unique<Allocation>();
    allocation->memory = *block->memory;
    allocation->offset = startUnit * config.allocationUnit;
    allocation->size = alignedSize;
    allocation->memoryTypeIndex = block->memoryTypeIndex;
    allocation->isMapped = block->isMapped;
    allocation->mappedPtr = block->isMapped ?
        static_cast<char*>(block->mappedPtr) + allocation->offset : nullptr;

    block->used += alignedSize;

    return allocation;
}

void MemoryPool::deallocate(std::unique_ptr<Allocation> allocation) {
    if (!allocation) {
        return;
    }

    std::lock_guard<std::mutex> lock(poolMutex);

    // Find the block that contains this allocation
    for (auto& [poolType, poolBlocks] : pools) {
        const PoolConfig& config = poolConfigs[poolType];

        for (auto& block : poolBlocks) {
            if (*block->memory == allocation->memory) {
                // Calculate which units to free
                size_t startUnit = allocation->offset / config.allocationUnit;
                size_t numUnits = (allocation->size + config.allocationUnit - 1) / config.allocationUnit;

                // Mark units as free
                for (size_t i = startUnit; i < startUnit + numUnits; ++i) {
                    if (i < block->freeList.size()) {
                        block->freeList[i] = true;
                    }
                }

                block->used -= allocation->size;
                return;
            }
        }
    }

    std::cerr << "Warning: Could not find memory block for deallocation" << std::endl;
}

std::pair<vk::raii::Buffer, std::unique_ptr<MemoryPool::Allocation>> MemoryPool::createBuffer(
    const vk::DeviceSize size,
    const vk::BufferUsageFlags usage,
    const vk::MemoryPropertyFlags properties) {

    // Determine a pool type based on usage and properties
    PoolType poolType = PoolType::VERTEX_BUFFER;
    if (usage & vk::BufferUsageFlagBits::eVertexBuffer) {
        poolType = PoolType::VERTEX_BUFFER;
    } else if (usage & vk::BufferUsageFlagBits::eIndexBuffer) {
        poolType = PoolType::INDEX_BUFFER;
    } else if (usage & vk::BufferUsageFlagBits::eUniformBuffer) {
        poolType = PoolType::UNIFORM_BUFFER;
    } else if (properties & vk::MemoryPropertyFlagBits::eHostVisible) {
        poolType = PoolType::STAGING_BUFFER;
    }

    // Create the buffer
    const vk::BufferCreateInfo bufferInfo{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive
    };

    vk::raii::Buffer buffer(device, bufferInfo);

    // Get memory requirements
    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();

    // Allocate from pool
    auto allocation = allocate(poolType, memRequirements.size, memRequirements.alignment);
    if (!allocation) {
        throw std::runtime_error("Failed to allocate memory from pool");
    }

    // Bind memory to buffer
    buffer.bindMemory(allocation->memory, allocation->offset);

    return {std::move(buffer), std::move(allocation)};
}

std::pair<vk::raii::Image, std::unique_ptr<MemoryPool::Allocation>> MemoryPool::createImage(
    uint32_t width,
    uint32_t height,
    vk::Format format,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlags properties) {

    // Create the image
    vk::ImageCreateInfo imageInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined
    };

    vk::raii::Image image(device, imageInfo);

    // Get memory requirements
    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();

    // Allocate from texture pool
    auto allocation = allocate(PoolType::TEXTURE_IMAGE, memRequirements.size, memRequirements.alignment);
    if (!allocation) {
        throw std::runtime_error("Failed to allocate memory from texture pool");
    }

    // Bind memory to image
    image.bindMemory(allocation->memory, allocation->offset);

    return {std::move(image), std::move(allocation)};
}

std::pair<vk::DeviceSize, vk::DeviceSize> MemoryPool::getMemoryUsage(PoolType poolType) const {
    std::lock_guard<std::mutex> lock(poolMutex);

    auto poolIt = pools.find(poolType);
    if (poolIt == pools.end()) {
        return {0, 0};
    }

    vk::DeviceSize used = 0;
    vk::DeviceSize total = 0;

    for (const auto& block : poolIt->second) {
        used += block->used;
        total += block->size;
    }

    return {used, total};
}

std::pair<vk::DeviceSize, vk::DeviceSize> MemoryPool::getTotalMemoryUsage() const {
    std::lock_guard<std::mutex> lock(poolMutex);

    vk::DeviceSize totalUsed = 0;
    vk::DeviceSize totalAllocated = 0;

    for (const auto& [poolType, poolBlocks] : pools) {
        for (const auto& block : poolBlocks) {
            totalUsed += block->used;
            totalAllocated += block->size;
        }
    }

    return {totalUsed, totalAllocated};
}

bool MemoryPool::preAllocatePools() {
    std::lock_guard<std::mutex> lock(poolMutex);

    try {
        std::cout << "Pre-allocating memory pools to prevent allocation during rendering..." << std::endl;

        // Pre-allocate at least one block for each pool type
        for (const auto& [poolType, config] : poolConfigs) {
            auto poolIt = pools.find(poolType);
            if (poolIt == pools.end()) {
                pools[poolType] = std::vector<std::unique_ptr<MemoryBlock>>();
                poolIt = pools.find(poolType);
            }

            auto& poolBlocks = poolIt->second;
            if (poolBlocks.empty()) {
                // Create initial block for this pool type
                auto newBlock = createMemoryBlock(poolType, config.blockSize);
                poolBlocks.push_back(std::move(newBlock));
                std::cout << "  Pre-allocated block for pool type " << static_cast<int>(poolType) << std::endl;
            }
        }

        std::cout << "Memory pool pre-allocation completed successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to pre-allocate memory pools: " << e.what() << std::endl;
        return false;
    }
}

void MemoryPool::setRenderingActive(bool active) {
    std::lock_guard lock(poolMutex);
    renderingActive = active;
}

bool MemoryPool::isRenderingActive() const {
    std::lock_guard<std::mutex> lock(poolMutex);
    return renderingActive;
}
