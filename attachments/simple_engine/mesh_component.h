#pragma once

#include <vector>
#include <string>
#include <array>
#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>

#include "component.h"

/**
 * @brief Structure representing a vertex in a mesh.
 */
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 tangent;

    bool operator==(const Vertex& other) const {
        return position == other.position &&
               normal == other.normal &&
               texCoord == other.texCoord &&
               tangent == other.tangent;
    }

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription(
            0,                                  // binding
            sizeof(Vertex),                     // stride
            vk::VertexInputRate::eVertex        // inputRate
        );
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 4> attributeDescriptions = {
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, position)
            },
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = offsetof(Vertex, normal)
            },
            vk::VertexInputAttributeDescription{
                .location = 2,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = offsetof(Vertex, texCoord)
            },
            vk::VertexInputAttributeDescription{
                .location = 3,
                .binding = 0,
                .format = vk::Format::eR32G32B32A32Sfloat,
                .offset = offsetof(Vertex, tangent)
            }
        };
        return attributeDescriptions;
    }
};

/**
 * @brief Component that handles the mesh data for rendering.
 */
class MeshComponent : public Component {
private:
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // All PBR texture paths for this mesh
    std::string texturePath;           // Primary texture path (baseColor) - kept for backward compatibility
    std::string baseColorTexturePath;  // Base color (albedo) texture
    std::string normalTexturePath;     // Normal map texture
    std::string metallicRoughnessTexturePath;  // Metallic-roughness texture
    std::string occlusionTexturePath;  // Ambient occlusion texture
    std::string emissiveTexturePath;   // Emissive texture

    // Vulkan resources will be managed by the renderer
    // This component only stores the data

public:
    /**
     * @brief Constructor with optional name.
     * @param componentName The name of the component.
     */
    explicit MeshComponent(const std::string& componentName = "MeshComponent")
        : Component(componentName) {}

    /**
     * @brief Set the vertices of the mesh.
     * @param newVertices The new vertices.
     */
    void SetVertices(const std::vector<Vertex>& newVertices) {
        vertices = newVertices;
    }

    /**
     * @brief Get the vertices of the mesh.
     * @return The vertices.
     */
    [[nodiscard]] const std::vector<Vertex>& GetVertices() const {
        return vertices;
    }

    /**
     * @brief Set the indices of the mesh.
     * @param newIndices The new indices.
     */
    void SetIndices(const std::vector<uint32_t>& newIndices) {
        indices = newIndices;
    }

    /**
     * @brief Get the indices of the mesh.
     * @return The indices.
     */
    [[nodiscard]] const std::vector<uint32_t>& GetIndices() const {
        return indices;
    }

    /**
     * @brief Set the texture path for the mesh.
     * @param path The path to the texture file.
     */
    void SetTexturePath(const std::string& path) {
        texturePath = path;
        baseColorTexturePath = path; // Keep baseColor in sync for backward compatibility
    }

    /**
     * @brief Get the texture path for the mesh.
     * @return The path to the texture file.
     */
    [[nodiscard]] const std::string& GetTexturePath() const {
        return texturePath;
    }

    // PBR texture path setters
    void SetBaseColorTexturePath(const std::string& path) { baseColorTexturePath = path; }
    void SetNormalTexturePath(const std::string& path) { normalTexturePath = path; }
    void SetMetallicRoughnessTexturePath(const std::string& path) { metallicRoughnessTexturePath = path; }
    void SetOcclusionTexturePath(const std::string& path) { occlusionTexturePath = path; }
    void SetEmissiveTexturePath(const std::string& path) { emissiveTexturePath = path; }

    // PBR texture path getters
    [[nodiscard]] const std::string& GetBaseColorTexturePath() const { return baseColorTexturePath; }
    [[nodiscard]] const std::string& GetNormalTexturePath() const { return normalTexturePath; }
    [[nodiscard]] const std::string& GetMetallicRoughnessTexturePath() const { return metallicRoughnessTexturePath; }
    [[nodiscard]] const std::string& GetOcclusionTexturePath() const { return occlusionTexturePath; }
    [[nodiscard]] const std::string& GetEmissiveTexturePath() const { return emissiveTexturePath; }

    /**
     * @brief Create a simple quad mesh.
     * @param width The width of the quad.
     * @param height The height of the quad.
     * @param color The color of the quad.
     */
    void CreateQuad(float width = 1.0f, float height = 1.0f, const glm::vec3& color = glm::vec3(1.0f));

    /**
     * @brief Create a simple cube mesh.
     * @param size The size of the cube.
     * @param color The color of the cube.
     */
    void CreateCube(float size = 1.0f, const glm::vec3& color = glm::vec3(1.0f));

    /**
     * @brief Load mesh data from a Model.
     * @param model Pointer to the model to load from.
     */
    void LoadFromModel(const class Model* model);
};
