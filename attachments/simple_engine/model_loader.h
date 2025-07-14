#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <glm/glm.hpp>

class Renderer;
class Mesh;
class Material;
class Model;

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

private:
    // Reference to the renderer
    Renderer* renderer = nullptr;

    // Loaded models
    std::unordered_map<std::string, std::unique_ptr<Model>> models;

    // Loaded materials
    std::unordered_map<std::string, std::unique_ptr<Material>> materials;

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
};
