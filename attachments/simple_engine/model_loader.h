#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <glm/glm.hpp>
#include "mesh_component.h"

class Renderer;
class Mesh;
class Material;

// Forward declaration for tinygltf
namespace tinygltf {
    class Model;
}

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
    glm::vec3 direction = glm::vec3(0.0f, -1.0f, 0.0f);  // For directional/spot lights
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 1.0f;
    float range = 100.0f;  // For point/spot lights
    float innerConeAngle = 0.0f;  // For spot lights
    float outerConeAngle = 0.785398f;  // For spot lights (45 degrees)
    std::string sourceMaterial;  // Name of source material (for emissive lights)
};

/**
 * @brief Structure representing mesh data for a specific material.
 */
struct MaterialMesh {
    int materialIndex;
    std::string materialName;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::string texturePath;  // Primary texture path for this material
};

/**
 * @brief Class representing a 3D model.
 */
class Model {
public:
    Model(const std::string& name) : name(name) {}
    ~Model() = default;

    const std::string& GetName() const { return name; }

    // Mesh data access methods
    const std::vector<Vertex>& GetVertices() const { return vertices; }
    const std::vector<uint32_t>& GetIndices() const { return indices; }

    // Methods to set mesh data (used by parser)
    void SetVertices(const std::vector<Vertex>& newVertices) { vertices = newVertices; }
    void SetIndices(const std::vector<uint32_t>& newIndices) { indices = newIndices; }

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
    ModelLoader();

    /**
     * @brief Destructor for proper cleanup.
     */
    ~ModelLoader();

    /**
     * @brief Initialize the model loader.
     * @param renderer Pointer to the renderer.
     * @return True if initialization was successful, false otherwise.
     */
    bool Initialize(Renderer* renderer);

    /**
     * @brief Load a model from a GLTF file.
     * @param filename The path to the GLTF file.
     * @return Pointer to the loaded model, or nullptr if loading failed.
     */
    Model* LoadGLTF(const std::string& filename);

    /**
     * @brief Load a model from a GLTF file with PBR materials.
     * @param filename The path to the GLTF file.
     * @param albedoMap The path to the albedo texture.
     * @param normalMap The path to the normal texture.
     * @param metallicRoughnessMap The path to the metallic-roughness texture.
     * @param aoMap The path to the ambient occlusion texture.
     * @param emissiveMap The path to the emissive texture.
     * @return Pointer to the loaded model, or nullptr if loading failed.
     */
    Model* LoadGLTFWithPBR(const std::string& filename,
                          const std::string& albedoMap,
                          const std::string& normalMap,
                          const std::string& metallicRoughnessMap,
                          const std::string& aoMap,
                          const std::string& emissiveMap);

    /**
     * @brief Get a model by name.
     * @param name The name of the model.
     * @return Pointer to the model, or nullptr if not found.
     */
    Model* GetModel(const std::string& name);

    /**
     * @brief Create a new material with PBR properties.
     * @param name The name of the material.
     * @param albedo The albedo color.
     * @param metallic The metallic factor.
     * @param roughness The roughness factor.
     * @param ao The ambient occlusion factor.
     * @param emissive The emissive color.
     * @return Pointer to the created material, or nullptr if creation failed.
     */
    Material* CreatePBRMaterial(const std::string& name,
                               const glm::vec3& albedo,
                               float metallic,
                               float roughness,
                               float ao,
                               const glm::vec3& emissive);

    /**
     * @brief Get the first available material texture path for a model.
     * @param modelName The name of the model.
     * @return The texture path of the first material, or empty string if none found.
     */
    std::string GetFirstMaterialTexturePath(const std::string& modelName);

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
    std::vector<MaterialMesh> GetMaterialMeshes(const std::string& modelName) const;

    /**
     * @brief Parse GLTF file data without creating Vulkan resources (thread-safe).
     * @param filename The path to the GLTF file.
     * @return Vector of material meshes with raw data only.
     */
    std::vector<MaterialMesh> ParseGLTFDataOnly(const std::string& filename);

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
    bool LoadPBRTextures(Material* material,
                        const std::string& albedoMap,
                        const std::string& normalMap,
                        const std::string& metallicRoughnessMap,
                        const std::string& aoMap,
                        const std::string& emissiveMap);

    /**
     * @brief Extract lights from GLTF punctual lights extension.
     * @param gltfModel The loaded GLTF model.
     * @param modelName The name of the model.
     * @return True if extraction was successful, false otherwise.
     */
    bool ExtractPunctualLights(const class tinygltf::Model& gltfModel, const std::string& modelName);

    /**
     * @brief Extract lights from emissive materials.
     * @param gltfModel The loaded GLTF model.
     * @param modelName The name of the model.
     * @return True if extraction was successful, false otherwise.
     */
    bool ExtractEmissiveLights(const class tinygltf::Model& gltfModel, const std::string& modelName);
};
