#pragma once

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "mesh_component.h"

class Renderer;
class Mesh;
class Material;

// Forward declaration for tinygltf
namespace tinygltf {
    class Model;
}

class Material {
    public:
        explicit Material(std::string  name) : name(std::move(name)) {}
        ~Material() = default;

        [[nodiscard]] const std::string& GetName() const { return name; }

        // PBR properties
        glm::vec3 albedo = glm::vec3(1.0f);
        float metallic = 0.0f;
        float roughness = 1.0f;
        float ao = 1.0f;
        glm::vec3 emissive = glm::vec3(0.0f);
        float emissiveStrength = 1.0f;  // KHR_materials_emissive_strength extension
        float alpha = 1.0f;             // Base color alpha (from MR baseColorFactor or SpecGloss diffuseFactor)

        // Texture paths for PBR materials
        std::string albedoTexturePath;
        std::string normalTexturePath;
        std::string metallicRoughnessTexturePath;
        std::string occlusionTexturePath;
        std::string emissiveTexturePath;

    private:
        std::string name;
};


/**
 * @brief Structure representing a light source extracted from GLTF.
 */
struct ExtractedLight {
    enum class Type {
        Directional,
        Point,
        Spot,
        Emissive  // Light derived from emissive material
    };

    Type type = Type::Point;
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 direction = glm::vec3(0.0f, -1.0f, 0.0f);  // For directional/spotlights
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;
    float range = 100.0f;  // For point/spotlights
    float innerConeAngle = 0.0f;  // For spotlights
    float outerConeAngle = 0.785398f;  // For spotlights (45 degrees)
    std::string sourceMaterial;  // Name of source material (for emissive lights)
};

/**
 * @brief Structure representing camera data extracted from GLTF.
 */
struct CameraData {
    std::string name;
    bool isPerspective = true;

    // Perspective camera properties
    float fov = 0.785398f;  // 45 degrees in radians
    float aspectRatio = 1.0f;

    // Orthographic camera properties
    float orthographicSize = 1.0f;

    // Common properties
    float nearPlane = 0.1f;
    float farPlane = 1000.0f;

    // Transform properties
    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);  // Identity quaternion
};

/**
 * @brief Structure representing mesh data for a specific material.
 */
struct MaterialMesh {
    int materialIndex;
    std::string materialName;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // All PBR texture paths for this material
    std::string texturePath;           // Primary texture path (baseColor) - kept for backward compatibility
    std::string baseColorTexturePath;  // Base color (albedo) texture
    std::string normalTexturePath;     // Normal map texture
    std::string metallicRoughnessTexturePath;  // Metallic-roughness texture
    std::string occlusionTexturePath;  // Ambient occlusion texture
    std::string emissiveTexturePath;   // Emissive texture

    // Instancing support
    std::vector<InstanceData> instances;  // Instance data for instanced rendering
    bool isInstanced = false;             // Flag to indicate if this mesh uses instancing

    /**
     * @brief Add an instance with the given transform matrix.
     * @param transform The transform matrix for this instance.
     * @param matIndex The material index for this instance (default: use materialIndex).
     */
    void AddInstance(const glm::mat4& transform, uint32_t matIndex = 0) {
        if (matIndex == 0) matIndex = static_cast<uint32_t>(materialIndex);
        instances.emplace_back(transform, matIndex);
        isInstanced = instances.size() > 1;
    }

    /**
     * @brief Get the number of instances.
     * @return Number of instances (0 if not instanced, >= 1 if instanced).
     */
    [[nodiscard]] size_t GetInstanceCount() const {
        return instances.size();
    }

    /**
     * @brief Check if this mesh uses instancing.
     * @return True if instanced (more than 1 instance), false otherwise.
     */
    [[nodiscard]] bool IsInstanced() const {
        return isInstanced;
    }
};

/**
 * @brief Class representing a 3D model.
 */
class Model {
public:
    explicit Model(std::string  name) : name(std::move(name)) {}
    ~Model() = default;

    [[nodiscard]] const std::string& GetName() const { return name; }

    // Mesh data access methods
    [[nodiscard]] const std::vector<Vertex>& GetVertices() const { return vertices; }
    [[nodiscard]] const std::vector<uint32_t>& GetIndices() const { return indices; }

    // Methods to set mesh data (used by parser)
    void SetVertices(const std::vector<Vertex>& newVertices) { vertices = newVertices; }
    void SetIndices(const std::vector<uint32_t>& newIndices) { indices = newIndices; }

    // Camera data access methods
    [[nodiscard]] const std::vector<CameraData>& GetCameras() const { return cameras; }

public:
    // Public access to cameras for model loader
    std::vector<CameraData> cameras;

private:
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    // Other model data (meshes, materials, etc.)
};

/**
 * @brief Class for loading and managing 3D models.
 */
class ModelLoader {
public:
    /**
     * @brief Default constructor.
     */
    ModelLoader() = default;

    /**
     * @brief Destructor for proper cleanup.
     */
    ~ModelLoader();

    /**
     * @brief Initialize the model loader.
     * @param _renderer Pointer to the renderer.
     * @return True if initialization was successful, false otherwise.
     */
    bool Initialize(Renderer* _renderer);

    /**
     * @brief Load a model from a GLTF file.
     * @param filename The path to the GLTF file.
     * @return Pointer to the loaded model, or nullptr if loading failed.
     */
    Model* LoadGLTF(const std::string& filename);


    /**
     * @brief Get a model by name.
     * @param name The name of the model.
     * @return Pointer to the model, or nullptr if not found.
     */
    Model* GetModel(const std::string& name);



    /**
     * @brief Get extracted lights from a loaded model.
     * @param modelName The name of the model.
     * @return Vector of extracted lights from the model.
     */
    std::vector<ExtractedLight> GetExtractedLights(const std::string& modelName) const;

    /**
     * @brief Get material-specific meshes from a loaded model.
     * @param modelName The name of the model.
     * @return Vector of material meshes from the model.
     */
    const std::vector<MaterialMesh>& GetMaterialMeshes(const std::string& modelName) const;

    /**
     * @brief Get a material by name.
     * @param materialName The name of the material.
     * @return Pointer to the material, or nullptr if not found.
     */
    Material* GetMaterial(const std::string& materialName) const;


private:
    // Reference to the renderer
    Renderer* renderer = nullptr;

    // Loaded models
    std::unordered_map<std::string, std::unique_ptr<Model>> models;

    // Loaded materials
    std::unordered_map<std::string, std::unique_ptr<Material>> materials;

    // Extracted lights per model
    std::unordered_map<std::string, std::vector<ExtractedLight>> extractedLights;

    // Material meshes per model
    std::unordered_map<std::string, std::vector<MaterialMesh>> materialMeshes;

    float light_scale = 1.0f;

    /**
     * @brief Parse a GLTF file.
     * @param filename The path to the GLTF file.
     * @param model The model to populate.
     * @return True if parsing was successful, false otherwise.
     */
    bool ParseGLTF(const std::string& filename, Model* model);

    /**
     * @brief Load textures for a PBR material.
     * @param material The material to populate.
     * @param albedoMap The path to the albedo texture.
     * @param normalMap The path to the normal texture.
     * @param metallicRoughnessMap The path to the metallic-roughness texture.
     * @param aoMap The path to the ambient occlusion texture.
     * @param emissiveMap The path to the emissive texture.
     * @return True if loading was successful, false otherwise.
     */

    /**
     * @brief Extract lights from GLTF punctual lights extension.
     * @param gltfModel The loaded GLTF model.
     * @param modelName The name of the model.
     * @return True if extraction was successful, false otherwise.
     */
    bool ExtractPunctualLights(const class tinygltf::Model& gltfModel, const std::string& modelName);
};
