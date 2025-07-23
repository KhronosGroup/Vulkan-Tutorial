#include "physics_system.h"
#include "entity.h"
#include "renderer.h"
#include <iostream>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>

// Concrete implementation of RigidBody
class ConcreteRigidBody : public RigidBody {
public:
    ConcreteRigidBody(Entity* entity, CollisionShape shape, float mass)
        : entity(entity), shape(shape), mass(mass) {
        // Initialize with entity's transform if available
        if (entity) {
            // This would normally get the position, rotation, and scale from the entity's transform component
            position = glm::vec3(0.0f);
            rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
            scale = glm::vec3(1.0f);
        }
    }

    ~ConcreteRigidBody() override = default;

    void SetPosition(const glm::vec3& position) override {
        this->position = position;
        std::cout << "Setting rigid body position to (" << position.x << ", " << position.y << ", " << position.z << ")" << std::endl;
    }

    void SetRotation(const glm::quat& rotation) override {
        this->rotation = rotation;
        std::cout << "Setting rigid body rotation" << std::endl;
    }

    void SetScale(const glm::vec3& scale) override {
        this->scale = scale;
        std::cout << "Setting rigid body scale to (" << scale.x << ", " << scale.y << ", " << scale.z << ")" << std::endl;
    }

    void SetMass(float mass) override {
        this->mass = mass;
        std::cout << "Setting rigid body mass to " << mass << std::endl;
    }

    void SetRestitution(float restitution) override {
        this->restitution = restitution;
        std::cout << "Setting rigid body restitution to " << restitution << std::endl;
    }

    void SetFriction(float friction) override {
        this->friction = friction;
        std::cout << "Setting rigid body friction to " << friction << std::endl;
    }

    void ApplyForce(const glm::vec3& force, const glm::vec3& localPosition) override {
        std::cout << "Applying force (" << force.x << ", " << force.y << ", " << force.z << ") "
                  << "at local position (" << localPosition.x << ", " << localPosition.y << ", " << localPosition.z << ")" << std::endl;

        // In a real implementation, this would apply the force to the rigid body
        linearVelocity += force / mass;
    }

    void ApplyImpulse(const glm::vec3& impulse, const glm::vec3& localPosition) override {
        std::cout << "Applying impulse (" << impulse.x << ", " << impulse.y << ", " << impulse.z << ") "
                  << "at local position (" << localPosition.x << ", " << localPosition.y << ", " << localPosition.z << ")" << std::endl;

        // In a real implementation, this would apply the impulse to the rigid body
        linearVelocity += impulse / mass;
    }

    void SetLinearVelocity(const glm::vec3& velocity) override {
        linearVelocity = velocity;
        std::cout << "Setting rigid body linear velocity to (" << velocity.x << ", " << velocity.y << ", " << velocity.z << ")" << std::endl;
    }

    void SetAngularVelocity(const glm::vec3& velocity) override {
        angularVelocity = velocity;
        std::cout << "Setting rigid body angular velocity to (" << velocity.x << ", " << velocity.y << ", " << velocity.z << ")" << std::endl;
    }

    glm::vec3 GetPosition() const override {
        return position;
    }

    glm::quat GetRotation() const override {
        return rotation;
    }

    glm::vec3 GetLinearVelocity() const override {
        return linearVelocity;
    }

    glm::vec3 GetAngularVelocity() const override {
        return angularVelocity;
    }

    void SetKinematic(bool kinematic) override {
        this->kinematic = kinematic;
        std::cout << "Setting rigid body kinematic to " << (kinematic ? "true" : "false") << std::endl;
    }

    bool IsKinematic() const override {
        return kinematic;
    }

    Entity* GetEntity() const {
        return entity;
    }

    CollisionShape GetShape() const {
        return shape;
    }

    float GetMass() const {
        return mass;
    }

    float GetInverseMass() const {
        return mass > 0.0f ? 1.0f / mass : 0.0f;
    }

    float GetRestitution() const {
        return restitution;
    }

    float GetFriction() const {
        return friction;
    }

    void Update(float deltaTime, const glm::vec3& gravity) {
        if (kinematic) {
            return;
        }

        // Apply gravity
        linearVelocity += gravity * deltaTime;

        // Update position
        position += linearVelocity * deltaTime;

        // Update rotation
        glm::quat angularVelocityQuat(0.0f, angularVelocity.x, angularVelocity.y, angularVelocity.z);
        rotation += 0.5f * deltaTime * angularVelocityQuat * rotation;
        rotation = glm::normalize(rotation);

        // Update entity transform if available
        if (entity) {
            // This would normally set the position, rotation, and scale on the entity's transform component
        }
    }

private:
    Entity* entity = nullptr;
    CollisionShape shape;

    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // Identity quaternion
    glm::vec3 scale = glm::vec3(1.0f);

    glm::vec3 linearVelocity = glm::vec3(0.0f);
    glm::vec3 angularVelocity = glm::vec3(0.0f);

    float mass = 1.0f;
    float restitution = 0.5f;
    float friction = 0.5f;

    bool kinematic = false;

    friend class PhysicsSystem;
};

PhysicsSystem::PhysicsSystem() {
    // Constructor implementation
}

PhysicsSystem::~PhysicsSystem() {
    // Destructor implementation
    if (initialized && gpuAccelerationEnabled) {
        CleanupVulkanResources();
    }
    rigidBodies.clear();
}

bool PhysicsSystem::Initialize() {
    // This is a placeholder implementation
    // In a real implementation, this would initialize the physics engine

    std::cout << "Initializing physics system" << std::endl;

    // Initialize Vulkan resources if GPU acceleration is enabled and renderer is set
    if (gpuAccelerationEnabled && renderer) {
        if (!InitializeVulkanResources()) {
            std::cerr << "Failed to initialize Vulkan resources for physics system" << std::endl;
            gpuAccelerationEnabled = false;
        }
    }

    initialized = true;
    return true;
}

void PhysicsSystem::Update(float deltaTime) {
    // If GPU acceleration is enabled and we have a renderer, use the GPU
    if (initialized && gpuAccelerationEnabled && renderer && rigidBodies.size() <= maxGPUObjects) {
        SimulatePhysicsOnGPU(deltaTime);
    } else {
        // Fall back to CPU physics
        // Update all rigid bodies
        for (auto& rigidBody : rigidBodies) {
            auto concreteRigidBody = static_cast<ConcreteRigidBody*>(rigidBody.get());
            concreteRigidBody->Update(deltaTime, gravity);
        }

        // Perform collision detection and resolution
        // This would be a complex algorithm in a real implementation
    }
}

RigidBody* PhysicsSystem::CreateRigidBody(Entity* entity, CollisionShape shape, float mass) {
    // Create a new rigid body
    auto rigidBody = std::make_unique<ConcreteRigidBody>(entity, shape, mass);

    // Store the rigid body
    RigidBody* rigidBodyPtr = rigidBody.get();
    rigidBodies.push_back(std::move(rigidBody));

    std::cout << "Rigid body created for entity: " << (entity ? entity->GetName() : "null") << std::endl;
    return rigidBodyPtr;
}

bool PhysicsSystem::RemoveRigidBody(RigidBody* rigidBody) {
    // Find the rigid body in the vector
    auto it = std::find_if(rigidBodies.begin(), rigidBodies.end(),
        [rigidBody](const std::unique_ptr<RigidBody>& rb) {
            return rb.get() == rigidBody;
        });

    if (it != rigidBodies.end()) {
        // Remove the rigid body
        rigidBodies.erase(it);

        std::cout << "Rigid body removed" << std::endl;
        return true;
    }

    std::cerr << "PhysicsSystem::RemoveRigidBody: Rigid body not found" << std::endl;
    return false;
}

void PhysicsSystem::SetGravity(const glm::vec3& gravity) {
    this->gravity = gravity;

    std::cout << "Setting gravity to (" << gravity.x << ", " << gravity.y << ", " << gravity.z << ")" << std::endl;
}

glm::vec3 PhysicsSystem::GetGravity() const {
    return gravity;
}

bool PhysicsSystem::Raycast(const glm::vec3& origin, const glm::vec3& direction, float maxDistance,
                          glm::vec3* hitPosition, glm::vec3* hitNormal, Entity** hitEntity) {
    std::cout << "Performing raycast from (" << origin.x << ", " << origin.y << ", " << origin.z << ") "
              << "in direction (" << direction.x << ", " << direction.y << ", " << direction.z << ") "
              << "with max distance " << maxDistance << std::endl;

    // Normalize the direction vector
    glm::vec3 normalizedDirection = glm::normalize(direction);

    // Variables to track the closest hit
    float closestHitDistance = maxDistance;
    bool hitFound = false;
    glm::vec3 closestHitPosition;
    glm::vec3 closestHitNormal;
    Entity* closestHitEntity = nullptr;

    // Check each rigid body for intersection
    for (const auto& rigidBody : rigidBodies) {
        auto concreteRigidBody = static_cast<ConcreteRigidBody*>(rigidBody.get());
        Entity* entity = concreteRigidBody->GetEntity();

        // Skip if the entity is null
        if (!entity) {
            continue;
        }

        // Get the position and shape of the rigid body
        glm::vec3 position = concreteRigidBody->GetPosition();
        CollisionShape shape = concreteRigidBody->GetShape();

        // Variables for hit detection
        float hitDistance = 0.0f;
        glm::vec3 localHitPosition;
        glm::vec3 localHitNormal;
        bool hit = false;

        // Check for intersection based on the shape
        switch (shape) {
            case CollisionShape::Sphere: {
                // Sphere intersection test
                float radius = 0.5f; // Default radius

                // Calculate coefficients for quadratic equation
                glm::vec3 oc = origin - position;
                float a = glm::dot(normalizedDirection, normalizedDirection);
                float b = 2.0f * glm::dot(oc, normalizedDirection);
                float c = glm::dot(oc, oc) - radius * radius;
                float discriminant = b * b - 4 * a * c;

                if (discriminant >= 0) {
                    // Calculate intersection distance
                    float t = (-b - std::sqrt(discriminant)) / (2.0f * a);

                    // Check if intersection is within range
                    if (t > 0 && t < closestHitDistance) {
                        hitDistance = t;
                        localHitPosition = origin + normalizedDirection * t;
                        localHitNormal = glm::normalize(localHitPosition - position);
                        hit = true;
                    }
                }
                break;
            }
            case CollisionShape::Box: {
                // Box intersection test (AABB)
                glm::vec3 halfExtents(0.5f, 0.5f, 0.5f); // Default box size

                // Calculate min and max bounds of the box
                glm::vec3 boxMin = position - halfExtents;
                glm::vec3 boxMax = position + halfExtents;

                // Calculate intersection with each slab
                float tmin = -INFINITY, tmax = INFINITY;

                for (int i = 0; i < 3; i++) {
                    if (std::abs(normalizedDirection[i]) < 0.0001f) {
                        // Ray is parallel to slab, check if origin is within slab
                        if (origin[i] < boxMin[i] || origin[i] > boxMax[i]) {
                            // No intersection
                            hit = false;
                            break;
                        }
                    } else {
                        // Calculate intersection distances
                        float ood = 1.0f / normalizedDirection[i];
                        float t1 = (boxMin[i] - origin[i]) * ood;
                        float t2 = (boxMax[i] - origin[i]) * ood;

                        // Ensure t1 <= t2
                        if (t1 > t2) {
                            std::swap(t1, t2);
                        }

                        // Update tmin and tmax
                        tmin = std::max(tmin, t1);
                        tmax = std::min(tmax, t2);

                        if (tmin > tmax) {
                            // No intersection
                            hit = false;
                            break;
                        }
                    }
                }

                // Check if intersection is within range
                if (tmin > 0 && tmin < closestHitDistance) {
                    hitDistance = tmin;
                    localHitPosition = origin + normalizedDirection * tmin;

                    // Calculate normal based on which face was hit
                    glm::vec3 center = position;
                    glm::vec3 d = localHitPosition - center;
                    float bias = 1.00001f; // Small bias to ensure we get the correct face

                    localHitNormal = glm::vec3(0.0f);
                    if (d.x > halfExtents.x * bias) localHitNormal = glm::vec3(1, 0, 0);
                    else if (d.x < -halfExtents.x * bias) localHitNormal = glm::vec3(-1, 0, 0);
                    else if (d.y > halfExtents.y * bias) localHitNormal = glm::vec3(0, 1, 0);
                    else if (d.y < -halfExtents.y * bias) localHitNormal = glm::vec3(0, -1, 0);
                    else if (d.z > halfExtents.z * bias) localHitNormal = glm::vec3(0, 0, 1);
                    else if (d.z < -halfExtents.z * bias) localHitNormal = glm::vec3(0, 0, -1);

                    hit = true;
                }
                break;
            }
            case CollisionShape::Capsule: {
                // Capsule intersection test
                // Simplified as a line segment with spheres at each end
                float radius = 0.5f; // Default radius
                float halfHeight = 0.5f; // Default half-height

                // Define capsule line segment
                glm::vec3 capsuleA = position + glm::vec3(0, -halfHeight, 0);
                glm::vec3 capsuleB = position + glm::vec3(0, halfHeight, 0);

                // Calculate closest point on line segment
                glm::vec3 ab = capsuleB - capsuleA;
                glm::vec3 ao = origin - capsuleA;

                float t = glm::dot(ao, ab) / glm::dot(ab, ab);
                t = glm::clamp(t, 0.0f, 1.0f);

                glm::vec3 closestPoint = capsuleA + ab * t;

                // Sphere intersection test with closest point
                glm::vec3 oc = origin - closestPoint;
                float a = glm::dot(normalizedDirection, normalizedDirection);
                float b = 2.0f * glm::dot(oc, normalizedDirection);
                float c = glm::dot(oc, oc) - radius * radius;
                float discriminant = b * b - 4 * a * c;

                if (discriminant >= 0) {
                    // Calculate intersection distance
                    float t = (-b - std::sqrt(discriminant)) / (2.0f * a);

                    // Check if intersection is within range
                    if (t > 0 && t < closestHitDistance) {
                        hitDistance = t;
                        localHitPosition = origin + normalizedDirection * t;
                        localHitNormal = glm::normalize(localHitPosition - closestPoint);
                        hit = true;
                    }
                }
                break;
            }
            case CollisionShape::Mesh: {
                // Mesh intersection test
                // In a real implementation, this would perform intersection tests with each triangle in the mesh
                // For simplicity, we'll just simulate a hit with a sphere

                float radius = 0.5f; // Default radius

                // Calculate coefficients for quadratic equation
                glm::vec3 oc = origin - position;
                float a = glm::dot(normalizedDirection, normalizedDirection);
                float b = 2.0f * glm::dot(oc, normalizedDirection);
                float c = glm::dot(oc, oc) - radius * radius;
                float discriminant = b * b - 4 * a * c;

                if (discriminant >= 0) {
                    // Calculate intersection distance
                    float t = (-b - std::sqrt(discriminant)) / (2.0f * a);

                    // Check if intersection is within range
                    if (t > 0 && t < closestHitDistance) {
                        hitDistance = t;
                        localHitPosition = origin + normalizedDirection * t;
                        localHitNormal = glm::normalize(localHitPosition - position);
                        hit = true;
                    }
                }
                break;
            }
            default:
                break;
        }

        // Update closest hit if a hit was found
        if (hit && hitDistance < closestHitDistance) {
            closestHitDistance = hitDistance;
            closestHitPosition = localHitPosition;
            closestHitNormal = localHitNormal;
            closestHitEntity = entity;
            hitFound = true;
        }
    }

    // Set output parameters if a hit was found
    if (hitFound) {
        if (hitPosition) {
            *hitPosition = closestHitPosition;
        }

        if (hitNormal) {
            *hitNormal = closestHitNormal;
        }

        if (hitEntity) {
            *hitEntity = closestHitEntity;
        }

        std::cout << "Hit found at distance " << closestHitDistance << std::endl;
        std::cout << "Hit position: (" << closestHitPosition.x << ", " << closestHitPosition.y << ", " << closestHitPosition.z << ")" << std::endl;
        std::cout << "Hit normal: (" << closestHitNormal.x << ", " << closestHitNormal.y << ", " << closestHitNormal.z << ")" << std::endl;
        std::cout << "Hit entity: " << (closestHitEntity ? closestHitEntity->GetName() : "null") << std::endl;
    }

    return hitFound;
}

// Helper function to read a shader file
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

// Helper function to create a shader module
static vk::raii::ShaderModule createShaderModule(const vk::raii::Device& device, const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    try {
        return vk::raii::ShaderModule(device, createInfo);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create shader module: " + std::string(e.what()));
    }
}

bool PhysicsSystem::InitializeVulkanResources() {
    if (!renderer) {
        std::cerr << "Renderer is not set" << std::endl;
        return false;
    }

    vk::Device device = renderer->GetDevice();
    if (!device) {
        std::cerr << "Vulkan device is not valid" << std::endl;
        return false;
    }

    try {
        // Create shader modules
        const vk::raii::Device& raiiDevice = renderer->GetRaiiDevice();

        std::vector<char> integrateShaderCode = readFile("shaders/physics.spv");
        vulkanResources.integrateShaderModule = createShaderModule(raiiDevice, integrateShaderCode);

        std::vector<char> broadPhaseShaderCode = readFile("shaders/physics.spv");
        vulkanResources.broadPhaseShaderModule = createShaderModule(raiiDevice, broadPhaseShaderCode);

        std::vector<char> narrowPhaseShaderCode = readFile("shaders/physics.spv");
        vulkanResources.narrowPhaseShaderModule = createShaderModule(raiiDevice, narrowPhaseShaderCode);

        std::vector<char> resolveShaderCode = readFile("shaders/physics.spv");
        vulkanResources.resolveShaderModule = createShaderModule(raiiDevice, resolveShaderCode);

        // Create descriptor set layout
        std::array<vk::DescriptorSetLayoutBinding, 5> bindings = {
            // Physics data buffer
            vk::DescriptorSetLayoutBinding(
                0,                                      // binding
                vk::DescriptorType::eStorageBuffer,     // descriptorType
                1,                                      // descriptorCount
                vk::ShaderStageFlagBits::eCompute,      // stageFlags
                nullptr                                 // pImmutableSamplers
            ),
            // Collision data buffer
            vk::DescriptorSetLayoutBinding(
                1,                                      // binding
                vk::DescriptorType::eStorageBuffer,     // descriptorType
                1,                                      // descriptorCount
                vk::ShaderStageFlagBits::eCompute,      // stageFlags
                nullptr                                 // pImmutableSamplers
            ),
            // Pair buffer
            vk::DescriptorSetLayoutBinding(
                2,                                      // binding
                vk::DescriptorType::eStorageBuffer,     // descriptorType
                1,                                      // descriptorCount
                vk::ShaderStageFlagBits::eCompute,      // stageFlags
                nullptr                                 // pImmutableSamplers
            ),
            // Counter buffer
            vk::DescriptorSetLayoutBinding(
                3,                                      // binding
                vk::DescriptorType::eStorageBuffer,     // descriptorType
                1,                                      // descriptorCount
                vk::ShaderStageFlagBits::eCompute,      // stageFlags
                nullptr                                 // pImmutableSamplers
            ),
            // Parameters buffer
            vk::DescriptorSetLayoutBinding(
                4,                                      // binding
                vk::DescriptorType::eUniformBuffer,     // descriptorType
                1,                                      // descriptorCount
                vk::ShaderStageFlagBits::eCompute,      // stageFlags
                nullptr                                 // pImmutableSamplers
            )
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        try {
            vulkanResources.descriptorSetLayout = vk::raii::DescriptorSetLayout(raiiDevice, layoutInfo);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create descriptor set layout: " + std::string(e.what()));
        }

        // Create pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setLayoutCount = 1;
        vk::DescriptorSetLayout descriptorSetLayout = *vulkanResources.descriptorSetLayout;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        try {
            vulkanResources.pipelineLayout = vk::raii::PipelineLayout(raiiDevice, pipelineLayoutInfo);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create pipeline layout: " + std::string(e.what()));
        }

        // Create compute pipelines
        vk::ComputePipelineCreateInfo pipelineInfo;
        pipelineInfo.layout = *vulkanResources.pipelineLayout;
        pipelineInfo.basePipelineHandle = nullptr;

        // Integrate pipeline
        vk::PipelineShaderStageCreateInfo integrateStageInfo;
        integrateStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
        integrateStageInfo.module = *vulkanResources.integrateShaderModule;
        integrateStageInfo.pName = "IntegrateCS";
        pipelineInfo.stage = integrateStageInfo;

        try {
            vulkanResources.integratePipeline = vk::raii::Pipeline(raiiDevice, nullptr, pipelineInfo);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create integrate compute pipeline: " + std::string(e.what()));
        }

        // Broad phase pipeline
        vk::PipelineShaderStageCreateInfo broadPhaseStageInfo;
        broadPhaseStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
        broadPhaseStageInfo.module = *vulkanResources.broadPhaseShaderModule;
        broadPhaseStageInfo.pName = "BroadPhaseCS";
        pipelineInfo.stage = broadPhaseStageInfo;

        try {
            vulkanResources.broadPhasePipeline = vk::raii::Pipeline(raiiDevice, nullptr, pipelineInfo);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create broad phase compute pipeline: " + std::string(e.what()));
        }

        // Narrow phase pipeline
        vk::PipelineShaderStageCreateInfo narrowPhaseStageInfo;
        narrowPhaseStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
        narrowPhaseStageInfo.module = *vulkanResources.narrowPhaseShaderModule;
        narrowPhaseStageInfo.pName = "NarrowPhaseCS";
        pipelineInfo.stage = narrowPhaseStageInfo;

        try {
            vulkanResources.narrowPhasePipeline = vk::raii::Pipeline(raiiDevice, nullptr, pipelineInfo);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create narrow phase compute pipeline: " + std::string(e.what()));
        }

        // Resolve pipeline
        vk::PipelineShaderStageCreateInfo resolveStageInfo;
        resolveStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
        resolveStageInfo.module = *vulkanResources.resolveShaderModule;
        resolveStageInfo.pName = "ResolveCS";
        pipelineInfo.stage = resolveStageInfo;

        try {
            vulkanResources.resolvePipeline = vk::raii::Pipeline(raiiDevice, nullptr, pipelineInfo);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create resolve compute pipeline: " + std::string(e.what()));
        }

        // Create buffers
        vk::DeviceSize physicsBufferSize = sizeof(GPUPhysicsData) * maxGPUObjects;
        vk::DeviceSize collisionBufferSize = sizeof(GPUCollisionData) * maxGPUCollisions;
        vk::DeviceSize pairBufferSize = sizeof(uint32_t) * 2 * maxGPUCollisions;
        vk::DeviceSize counterBufferSize = sizeof(uint32_t) * 2;
        vk::DeviceSize paramsBufferSize = sizeof(PhysicsParams);

        // Create physics buffer
        vk::BufferCreateInfo bufferInfo;
        bufferInfo.size = physicsBufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;

        try {
            vulkanResources.physicsBuffer = vk::raii::Buffer(raiiDevice, bufferInfo);

            vk::MemoryRequirements memRequirements = vulkanResources.physicsBuffer.getMemoryRequirements();

            vk::MemoryAllocateInfo allocInfo;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = renderer->FindMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            vulkanResources.physicsBufferMemory = vk::raii::DeviceMemory(raiiDevice, allocInfo);
            vulkanResources.physicsBuffer.bindMemory(*vulkanResources.physicsBufferMemory, 0);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create physics buffer: " + std::string(e.what()));
        }

        // Create collision buffer
        bufferInfo.size = collisionBufferSize;
        try {
            vulkanResources.collisionBuffer = vk::raii::Buffer(raiiDevice, bufferInfo);

            vk::MemoryRequirements memRequirements = vulkanResources.collisionBuffer.getMemoryRequirements();

            vk::MemoryAllocateInfo allocInfo;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = renderer->FindMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            vulkanResources.collisionBufferMemory = vk::raii::DeviceMemory(raiiDevice, allocInfo);
            vulkanResources.collisionBuffer.bindMemory(*vulkanResources.collisionBufferMemory, 0);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create collision buffer: " + std::string(e.what()));
        }

        // Create pair buffer
        bufferInfo.size = pairBufferSize;
        try {
            vulkanResources.pairBuffer = vk::raii::Buffer(raiiDevice, bufferInfo);

            vk::MemoryRequirements memRequirements = vulkanResources.pairBuffer.getMemoryRequirements();

            vk::MemoryAllocateInfo allocInfo;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = renderer->FindMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            vulkanResources.pairBufferMemory = vk::raii::DeviceMemory(raiiDevice, allocInfo);
            vulkanResources.pairBuffer.bindMemory(*vulkanResources.pairBufferMemory, 0);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create pair buffer: " + std::string(e.what()));
        }

        // Create counter buffer
        bufferInfo.size = counterBufferSize;
        try {
            vulkanResources.counterBuffer = vk::raii::Buffer(raiiDevice, bufferInfo);

            vk::MemoryRequirements memRequirements = vulkanResources.counterBuffer.getMemoryRequirements();

            vk::MemoryAllocateInfo allocInfo;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = renderer->FindMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            vulkanResources.counterBufferMemory = vk::raii::DeviceMemory(raiiDevice, allocInfo);
            vulkanResources.counterBuffer.bindMemory(*vulkanResources.counterBufferMemory, 0);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create counter buffer: " + std::string(e.what()));
        }

        // Create params buffer
        bufferInfo.size = paramsBufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
        try {
            vulkanResources.paramsBuffer = vk::raii::Buffer(raiiDevice, bufferInfo);

            vk::MemoryRequirements memRequirements = vulkanResources.paramsBuffer.getMemoryRequirements();

            vk::MemoryAllocateInfo allocInfo;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = renderer->FindMemoryType(
                memRequirements.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            vulkanResources.paramsBufferMemory = vk::raii::DeviceMemory(raiiDevice, allocInfo);
            vulkanResources.paramsBuffer.bindMemory(*vulkanResources.paramsBufferMemory, 0);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create params buffer: " + std::string(e.what()));
        }

        // Initialize counter buffer
        uint32_t initialCounters[2] = { 0, 0 }; // [0] = pair count, [1] = collision count
        void* data = vulkanResources.counterBufferMemory.mapMemory(0, sizeof(initialCounters));
        memcpy(data, initialCounters, sizeof(initialCounters));
        vulkanResources.counterBufferMemory.unmapMemory();

        // Create descriptor pool
        std::array<vk::DescriptorPoolSize, 2> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)
        };

        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 1;

        try {
            vulkanResources.descriptorPool = vk::raii::DescriptorPool(raiiDevice, poolInfo);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create descriptor pool: " + std::string(e.what()));
        }

        // Allocate descriptor sets
        vk::DescriptorSetAllocateInfo descriptorSetAllocInfo;
        descriptorSetAllocInfo.descriptorPool = *vulkanResources.descriptorPool;
        descriptorSetAllocInfo.descriptorSetCount = 1;
        vk::DescriptorSetLayout descriptorSetLayoutRef = *vulkanResources.descriptorSetLayout;
        descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayoutRef;

        try {
            std::vector<vk::raii::DescriptorSet> raiiDescriptorSets = raiiDevice.allocateDescriptorSets(descriptorSetAllocInfo);
            vulkanResources.descriptorSets.resize(raiiDescriptorSets.size());
            for (size_t i = 0; i < raiiDescriptorSets.size(); ++i) {
                vulkanResources.descriptorSets[i] = *raiiDescriptorSets[i];
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate descriptor sets: " + std::string(e.what()));
        }

        // Update descriptor sets
        vk::DescriptorBufferInfo physicsBufferInfo;
        physicsBufferInfo.buffer = *vulkanResources.physicsBuffer;
        physicsBufferInfo.offset = 0;
        physicsBufferInfo.range = physicsBufferSize;

        vk::DescriptorBufferInfo collisionBufferInfo;
        collisionBufferInfo.buffer = *vulkanResources.collisionBuffer;
        collisionBufferInfo.offset = 0;
        collisionBufferInfo.range = collisionBufferSize;

        vk::DescriptorBufferInfo pairBufferInfo;
        pairBufferInfo.buffer = *vulkanResources.pairBuffer;
        pairBufferInfo.offset = 0;
        pairBufferInfo.range = pairBufferSize;

        vk::DescriptorBufferInfo counterBufferInfo;
        counterBufferInfo.buffer = *vulkanResources.counterBuffer;
        counterBufferInfo.offset = 0;
        counterBufferInfo.range = counterBufferSize;

        vk::DescriptorBufferInfo paramsBufferInfo;
        paramsBufferInfo.buffer = *vulkanResources.paramsBuffer;
        paramsBufferInfo.offset = 0;
        paramsBufferInfo.range = paramsBufferSize;

        std::array<vk::WriteDescriptorSet, 5> descriptorWrites;

        // Physics buffer
        descriptorWrites[0].setDstSet(vulkanResources.descriptorSets[0])
                          .setDstBinding(0)
                          .setDstArrayElement(0)
                          .setDescriptorCount(1)
                          .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                          .setPBufferInfo(&physicsBufferInfo);

        // Collision buffer
        descriptorWrites[1].setDstSet(vulkanResources.descriptorSets[0])
                          .setDstBinding(1)
                          .setDstArrayElement(0)
                          .setDescriptorCount(1)
                          .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                          .setPBufferInfo(&collisionBufferInfo);

        // Pair buffer
        descriptorWrites[2].setDstSet(vulkanResources.descriptorSets[0])
                          .setDstBinding(2)
                          .setDstArrayElement(0)
                          .setDescriptorCount(1)
                          .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                          .setPBufferInfo(&pairBufferInfo);

        // Counter buffer
        descriptorWrites[3].setDstSet(vulkanResources.descriptorSets[0])
                          .setDstBinding(3)
                          .setDstArrayElement(0)
                          .setDescriptorCount(1)
                          .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                          .setPBufferInfo(&counterBufferInfo);

        // Params buffer
        descriptorWrites[4].setDstSet(vulkanResources.descriptorSets[0])
                          .setDstBinding(4)
                          .setDstArrayElement(0)
                          .setDescriptorCount(1)
                          .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                          .setPBufferInfo(&paramsBufferInfo);

        raiiDevice.updateDescriptorSets(descriptorWrites, nullptr);

        // Create command pool
        vk::CommandPoolCreateInfo commandPoolInfo;
        commandPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        commandPoolInfo.queueFamilyIndex = 0; // Assuming compute queue family index is 0

        try {
            vulkanResources.commandPool = vk::raii::CommandPool(raiiDevice, commandPoolInfo);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create command pool: " + std::string(e.what()));
        }

        // Allocate command buffer
        vk::CommandBufferAllocateInfo commandBufferInfo;
        commandBufferInfo.commandPool = *vulkanResources.commandPool;
        commandBufferInfo.level = vk::CommandBufferLevel::ePrimary;
        commandBufferInfo.commandBufferCount = 1;

        try {
            std::vector<vk::raii::CommandBuffer> commandBuffers = raiiDevice.allocateCommandBuffers(commandBufferInfo);
            vulkanResources.commandBuffer = std::move(commandBuffers.front());
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate command buffer: " + std::string(e.what()));
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing Vulkan resources: " << e.what() << std::endl;
        CleanupVulkanResources();
        return false;
    }
}

void PhysicsSystem::CleanupVulkanResources() {
    if (!renderer) {
        return;
    }

    // Wait for the device to be idle before cleaning up
    renderer->WaitIdle();

    // With RAII, we just need to set the resources to nullptr
    // The destructors will handle the cleanup
    vulkanResources.commandBuffer = nullptr;
    vulkanResources.commandPool = nullptr;
    vulkanResources.paramsBuffer = nullptr;
    vulkanResources.paramsBufferMemory = nullptr;
    vulkanResources.counterBuffer = nullptr;
    vulkanResources.counterBufferMemory = nullptr;
    vulkanResources.pairBuffer = nullptr;
    vulkanResources.pairBufferMemory = nullptr;
    vulkanResources.collisionBuffer = nullptr;
    vulkanResources.collisionBufferMemory = nullptr;
    vulkanResources.physicsBuffer = nullptr;
    vulkanResources.physicsBufferMemory = nullptr;
    vulkanResources.descriptorPool = nullptr;
    vulkanResources.resolvePipeline = nullptr;
    vulkanResources.narrowPhasePipeline = nullptr;
    vulkanResources.broadPhasePipeline = nullptr;
    vulkanResources.integratePipeline = nullptr;
    vulkanResources.pipelineLayout = nullptr;
    vulkanResources.descriptorSetLayout = nullptr;
    vulkanResources.resolveShaderModule = nullptr;
    vulkanResources.narrowPhaseShaderModule = nullptr;
    vulkanResources.broadPhaseShaderModule = nullptr;
    vulkanResources.integrateShaderModule = nullptr;
}

void PhysicsSystem::UpdateGPUPhysicsData() {
    if (!renderer) {
        return;
    }

    // TODO: Add validity checks for Vulkan resources if needed
    // Temporarily removed to focus on main validation error investigation

    const vk::raii::Device& raiiDevice = renderer->GetRaiiDevice();


    // Map the physics buffer
    void* data = vulkanResources.physicsBufferMemory.mapMemory(0, sizeof(GPUPhysicsData) * rigidBodies.size());

    // Copy physics data to the buffer
    GPUPhysicsData* gpuData = static_cast<GPUPhysicsData*>(data);
    for (size_t i = 0; i < rigidBodies.size(); i++) {
        auto concreteRigidBody = static_cast<ConcreteRigidBody*>(rigidBodies[i].get());

        gpuData[i].position = glm::vec4(concreteRigidBody->GetPosition(), concreteRigidBody->GetInverseMass());
        gpuData[i].rotation = glm::vec4(concreteRigidBody->GetRotation().x, concreteRigidBody->GetRotation().y,
                                      concreteRigidBody->GetRotation().z, concreteRigidBody->GetRotation().w);
        gpuData[i].linearVelocity = glm::vec4(concreteRigidBody->GetLinearVelocity(), concreteRigidBody->GetRestitution());
        gpuData[i].angularVelocity = glm::vec4(concreteRigidBody->GetAngularVelocity(), concreteRigidBody->GetFriction());
        gpuData[i].force = glm::vec4(glm::vec3(0.0f), concreteRigidBody->IsKinematic() ? 1.0f : 0.0f);
        gpuData[i].torque = glm::vec4(glm::vec3(0.0f), 1.0f); // Always use gravity

        // Set collider data based on collider type
        CollisionShape shape = concreteRigidBody->GetShape();
        switch (shape) {
            case CollisionShape::Sphere:
                gpuData[i].colliderData = glm::vec4(0.5f, 0.0f, 0.0f, static_cast<float>(0)); // 0 = Sphere
                gpuData[i].colliderData2 = glm::vec4(0.0f);
                break;
            case CollisionShape::Box:
                gpuData[i].colliderData = glm::vec4(0.5f, 0.5f, 0.5f, static_cast<float>(1)); // 1 = Box
                gpuData[i].colliderData2 = glm::vec4(0.0f);
                break;
            default:
                gpuData[i].colliderData = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f); // Invalid
                gpuData[i].colliderData2 = glm::vec4(0.0f);
                break;
        }
    }

    vulkanResources.physicsBufferMemory.unmapMemory();

    // Reset counters
    uint32_t initialCounters[2] = { 0, 0 }; // [0] = pair count, [1] = collision count
    data = vulkanResources.counterBufferMemory.mapMemory(0, sizeof(initialCounters));
    memcpy(data, initialCounters, sizeof(initialCounters));
    vulkanResources.counterBufferMemory.unmapMemory();

    // Update params buffer
    PhysicsParams params;
    params.deltaTime = 1.0f / 60.0f; // Fixed time step
    params.gravity = gravity;
    params.numBodies = static_cast<uint32_t>(rigidBodies.size());
    params.maxCollisions = maxGPUCollisions;

    data = vulkanResources.paramsBufferMemory.mapMemory(0, sizeof(PhysicsParams));
    memcpy(data, &params, sizeof(PhysicsParams));
    vulkanResources.paramsBufferMemory.unmapMemory();
}

void PhysicsSystem::ReadbackGPUPhysicsData() {
    if (!renderer) {
        return;
    }

    // TODO: Add validity checks for Vulkan resources if needed
    // Temporarily removed to focus on main validation error investigation

    const vk::raii::Device& raiiDevice = renderer->GetRaiiDevice();


    // Map the physics buffer
    void* data = vulkanResources.physicsBufferMemory.mapMemory(0, sizeof(GPUPhysicsData) * rigidBodies.size());

    // Copy physics data from the buffer
    GPUPhysicsData* gpuData = static_cast<GPUPhysicsData*>(data);
    for (size_t i = 0; i < rigidBodies.size(); i++) {
        auto concreteRigidBody = static_cast<ConcreteRigidBody*>(rigidBodies[i].get());

        // Skip kinematic bodies
        if (concreteRigidBody->IsKinematic()) {
            continue;
        }

        concreteRigidBody->SetPosition(glm::vec3(gpuData[i].position));
        concreteRigidBody->SetRotation(glm::quat(gpuData[i].rotation.w, gpuData[i].rotation.x,
                                               gpuData[i].rotation.y, gpuData[i].rotation.z));
        concreteRigidBody->SetLinearVelocity(glm::vec3(gpuData[i].linearVelocity));
        concreteRigidBody->SetAngularVelocity(glm::vec3(gpuData[i].angularVelocity));
    }

    vulkanResources.physicsBufferMemory.unmapMemory();
}

void PhysicsSystem::SimulatePhysicsOnGPU(float deltaTime) {
    if (!renderer) {
        return;
    }

    // TODO: Add validity checks for Vulkan resources if needed
    // Temporarily removed to focus on main validation error investigation

    const vk::raii::Device& raiiDevice = renderer->GetRaiiDevice();


    // Update physics data on the GPU
    UpdateGPUPhysicsData();

    // Reset command buffer before beginning (required for reuse)
    vulkanResources.commandBuffer.reset();

    // Begin command buffer
    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    vulkanResources.commandBuffer.begin(beginInfo);

    // Bind descriptor set
    vulkanResources.commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        *vulkanResources.pipelineLayout,
        0,
        vulkanResources.descriptorSets,
        nullptr
    );

    // Step 1: Integrate forces and velocities
    vulkanResources.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *vulkanResources.integratePipeline);
    vulkanResources.commandBuffer.dispatch((rigidBodies.size() + 63) / 64, 1, 1);

    // Memory barrier to ensure integration is complete before collision detection
    vk::MemoryBarrier memoryBarrier;
    memoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    memoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    vulkanResources.commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags(),
        memoryBarrier,
        nullptr,
        nullptr
    );

    // Step 2: Broad-phase collision detection
    vulkanResources.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *vulkanResources.broadPhasePipeline);
    // Each thread checks one pair of objects
    uint32_t numPairs = (rigidBodies.size() * (rigidBodies.size() - 1)) / 2;
    vulkanResources.commandBuffer.dispatch((numPairs + 63) / 64, 1, 1);

    // Memory barrier to ensure broad phase is complete before narrow phase
    vulkanResources.commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags(),
        memoryBarrier,
        nullptr,
        nullptr
    );

    // Step 3: Narrow-phase collision detection
    vulkanResources.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *vulkanResources.narrowPhasePipeline);
    // We don't know how many pairs were generated, so we use a conservative estimate
    vulkanResources.commandBuffer.dispatch((maxGPUCollisions + 63) / 64, 1, 1);

    // Memory barrier to ensure narrow phase is complete before resolution
    vulkanResources.commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags(),
        memoryBarrier,
        nullptr,
        nullptr
    );

    // Step 4: Collision resolution
    vulkanResources.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *vulkanResources.resolvePipeline);
    // We don't know how many collisions were detected, so we use a conservative estimate
    vulkanResources.commandBuffer.dispatch((maxGPUCollisions + 63) / 64, 1, 1);

    // End command buffer
    vulkanResources.commandBuffer.end();

    // Submit command buffer
    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    vk::CommandBuffer cmdBuffer = *vulkanResources.commandBuffer;
    submitInfo.pCommandBuffers = &cmdBuffer;

    vk::Queue computeQueue = renderer->GetComputeQueue();
    computeQueue.submit(submitInfo, nullptr);
    computeQueue.waitIdle();

    // Read back physics data from the GPU
    ReadbackGPUPhysicsData();
}
