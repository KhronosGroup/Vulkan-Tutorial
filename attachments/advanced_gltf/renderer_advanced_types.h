#pragma once
#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <memory>
#include <string>

// Forward declarations
class Renderer;
class Entity;
class Model;
class MeshComponent;
namespace tinygltf { class Model; }
#include "memory_pool.h"

struct OutputVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texcoord;
    glm::vec4 tangent;
};

struct SkinPushConstants {
    uint32_t vertexCount;
    uint32_t morphIndices[24];
    struct MorphWeightBlock {
        float weights[24];
        uint32_t activeCount;
        uint32_t applySkinning; // 1 = apply skeletal skinning to the position, 0 = morph only
        uint32_t pad[2];
    } morphWeights;
};

struct AdvancedRendererState {
    vk::raii::DescriptorSetLayout skinDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout morphDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSet dummyMorphDescriptorSet = nullptr;
    vk::raii::PipelineLayout skinPipelineLayout = nullptr;
    vk::raii::Pipeline skinPipeline = nullptr;

    // Track mapping from mesh component to BLAS index for refitting
    std::unordered_map<MeshComponent*, uint32_t> meshToBLAS;
};

struct AdvancedEntityResources {
    bool isDeformable = false;
    vk::raii::Buffer outputVertexBuffer = nullptr;
    std::unique_ptr<MemoryPool::Allocation> outputVertexBufferAllocation = nullptr;
    vk::raii::Buffer jointMatricesBuffer = nullptr;
    std::unique_ptr<MemoryPool::Allocation> jointMatricesBufferAllocation = nullptr;

    // Staging allocations to prevent redundant OS-level allocations
    std::unique_ptr<MemoryPool::Allocation> stagingVertexBufferAllocation = nullptr;
    std::unique_ptr<MemoryPool::Allocation> stagingIndexBufferAllocation = nullptr;

    // GPU buffers for parallel skinning data
    vk::raii::Buffer jointIndicesBuffer = nullptr;
    std::unique_ptr<MemoryPool::Allocation> jointIndicesBufferAllocation = nullptr;
    vk::raii::Buffer jointWeightsBuffer = nullptr;
    std::unique_ptr<MemoryPool::Allocation> jointWeightsBufferAllocation = nullptr;
    vk::raii::Buffer stagingJointIndicesBuffer = nullptr;
    std::unique_ptr<MemoryPool::Allocation> stagingJointIndicesAllocation = nullptr;
    vk::raii::Buffer stagingJointWeightsBuffer = nullptr;
    std::unique_ptr<MemoryPool::Allocation> stagingJointWeightsAllocation = nullptr;
    vk::DeviceSize jointIndicesSize = 0;
    vk::DeviceSize jointWeightsSize = 0;

    // GPU buffers for morph targets
    std::vector<vk::raii::Buffer> morphTargetBuffers;
    std::vector<std::unique_ptr<MemoryPool::Allocation>> morphTargetBufferAllocations;
    std::vector<vk::raii::Buffer> stagingMorphTargetBuffers;
    std::vector<std::unique_ptr<MemoryPool::Allocation>> stagingMorphTargetAllocations;
    std::vector<vk::DeviceSize> morphTargetSizes;

    std::vector<vk::raii::DescriptorSet> skinDescriptorSets;
    std::vector<vk::raii::DescriptorSet> morphDescriptorSets;

    // Scratch buffer for BLAS refits
    vk::raii::Buffer blasScratchBuffer = nullptr;
    std::unique_ptr<MemoryPool::Allocation> blasScratchBufferAllocation = nullptr;
    vk::DeviceSize blasScratchBufferSize = 0;

    // Cached flags for TLAS optimization
    bool isEnvironment = false;
    bool isEnvironmentChecked = false;
    bool isGeometryChecked = false;
};

struct Skin {
    std::string name;
    int skeletonRoot = -1;
    std::vector<int> joints;
    std::vector<glm::mat4> inverseBindMatrices;
};

struct AdvancedModelData {
    bool isDeformable = false;
    int skinIndex = -1;
    std::vector<Skin> skins;
    std::unordered_map<int, std::vector<int>> nodeChildren;
    std::unordered_map<int, glm::mat4> nodeLocalTransforms;
    std::unordered_map<int, glm::vec3> nodeLocalTranslations;
    std::unordered_map<int, glm::quat> nodeLocalRotations;
    std::unordered_map<int, glm::vec3> nodeLocalScales;
    std::vector<int> rootNodes;
    std::unordered_map<int, int> nodeSkins;
};

struct AdvancedAnimationState {
    std::unordered_map<int, std::vector<int>> nodeChildren;
    std::unordered_map<int, glm::mat4> initialLocalTransforms;
    std::unordered_map<int, glm::vec3> initialLocalTranslations;
    std::unordered_map<int, glm::quat> initialLocalRotations;
    std::unordered_map<int, glm::vec3> initialLocalScales;
    std::vector<int> rootNodes;
};

struct AdvancedMeshComponentData {
    bool isDeformable = false;
    int numMorphTargets = 0;
    std::vector<int> joints;
    std::vector<glm::mat4> inverseBindMatrices;
    std::vector<glm::mat4> jointMatrices;
    std::vector<float> morphWeights;
    std::vector<std::vector<glm::vec3>> morphTargetPositions;

    // Parallel buffers for skinning data
    std::vector<glm::uvec4> jointIndices;
    std::vector<glm::vec4> jointWeights;
};

extern std::unordered_map<const Renderer*, AdvancedRendererState> g_rendererStates;
extern std::unordered_map<const void*, AdvancedEntityResources> g_meshAdvancedResources; // Keyed by MeshComponent*
extern std::unordered_map<const Model*, AdvancedModelData> g_modelData;
extern std::unordered_map<const class AnimationComponent*, AdvancedAnimationState> g_animationAdvancedStates;
extern std::unordered_map<const class MeshComponent*, AdvancedMeshComponentData> g_meshComponentData;
extern std::unordered_map<const void*, bool> g_materialMeshDeformable; // Keyed by MaterialMesh*
extern std::unordered_map<const void*, std::vector<glm::uvec4>> g_materialMeshJointIndices;
extern std::unordered_map<const void*, std::vector<glm::vec4>> g_materialMeshJointWeights;
extern std::unordered_map<const void*, int> g_materialMeshMorphTargetCount;
extern std::unordered_map<const void*, std::vector<std::vector<glm::vec3>>> g_materialMeshMorphPositions;
extern std::shared_mutex g_advancedStateMutex;

// Global pointer for tracking the last spawned ball to optimize camera tracking and avoid O(N) string searches
extern Entity* g_lastSpawnedBall;

// Mark/query entities whose transform is owned by the physics system (e.g. a thrown Fox).
// While owned, the animation system must not overwrite the entity transform.
void SetEntityPhysicsOwned(const class Entity* entity, bool owned);
bool IsEntityPhysicsOwned(const class Entity* entity);

// Extension functions for Engine
std::vector<class Entity*> SnapshotEntities(const class Engine* engine);

// Extension functions for Entity
std::recursive_mutex& GetEntityMutex(const class Entity* entity);

// Extension functions for Renderer
bool AdvancedRenderer_createSkinningResources(Renderer* renderer);
void AdvancedRenderer_updateSkins(Renderer* renderer, vk::raii::CommandBuffer& cmd, uint32_t frameIndex, const std::vector<Entity*>& entities);
void AdvancedRenderer_Cleanup(Renderer* renderer);
void AdvancedRenderer_KickWatchdog(Renderer* renderer);

// Extension functions for Model
AdvancedModelData& GetAdvancedModelData(const Model* model);
void AdvancedModel_ProcessSkins(class ModelLoader* loader, const tinygltf::Model& gltfModel, Model* model);

// Extension functions for MeshComponent
void SetMeshComponentDeformable(class MeshComponent* mesh, bool deformable);
void SetMeshComponentMorphTargets(class MeshComponent* mesh, int numTargets);
int GetMeshComponentMorphTargets(const class MeshComponent* mesh);
void SetMeshComponentMorphWeights(class MeshComponent* mesh, const std::vector<float>& weights);
const std::vector<float>& GetMeshComponentMorphWeights(const class MeshComponent* mesh);
bool IsMeshComponentDeformable(const class MeshComponent* mesh);
void SetMeshComponentSkinData(class MeshComponent* mesh, const std::vector<int>& joints, const std::vector<glm::mat4>& inverseBindMatrices);
void SetMeshComponentJointMatrices(class MeshComponent* mesh, const std::vector<glm::mat4>& matrices);
const std::vector<glm::mat4>& GetMeshComponentJointMatrices(const class MeshComponent* mesh);
void SetMeshComponentJointsAndWeights(class MeshComponent* mesh, const std::vector<glm::uvec4>& joints, const std::vector<glm::vec4>& weights);
void SetMeshComponentMorphPositions(class MeshComponent* mesh, const std::vector<std::vector<glm::vec3>>& positions);
void SetMeshComponentEnvironment(class MeshComponent* mesh, bool isEnvironment);

// Extension functions for MaterialMesh
void SetMaterialMeshDeformable(const void* materialMesh, bool deformable);
bool IsMaterialMeshDeformable(const void* materialMesh);
void SetMaterialMeshJointsAndWeights(const void* materialMesh, const std::vector<glm::uvec4>& joints, const std::vector<glm::vec4>& weights);
const std::vector<glm::uvec4>& GetMaterialMeshJoints(const void* materialMesh);
const std::vector<glm::vec4>& GetMaterialMeshWeights(const void* materialMesh);
int GetMaterialMeshMorphTargetCount(const void* materialMesh);
void SetMaterialMeshMorphTargetCount(const void* materialMesh, int count);
void SetMaterialMeshMorphPositions(const void* materialMesh, const std::vector<std::vector<glm::vec3>>& positions);
const std::vector<std::vector<glm::vec3>>& GetMaterialMeshMorphPositions(const void* materialMesh);

// Extension functions for AnimationComponent
void AnimationComponent_SetHierarchy(class AnimationComponent* anim,
                                    const std::unordered_map<int, std::vector<int>>& nodeChildren,
                                    const std::unordered_map<int, glm::mat4>& initialLocalTransforms,
                                    const std::unordered_map<int, glm::vec3>& initialLocalTranslations,
                                    const std::unordered_map<int, glm::quat>& initialLocalRotations,
                                    const std::unordered_map<int, glm::vec3>& initialLocalScales,
                                    const std::vector<int>& rootNodes);
