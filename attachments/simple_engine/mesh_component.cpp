#include "mesh_component.h"
#include "model_loader.h"
#include <cmath>

// Most of the MeshComponent class implementation is in the header file
// This file is mainly for any methods that might need additional implementation

void MeshComponent::CreateQuad(float width, float height, const glm::vec3& color) {
    float halfWidth = width * 0.5f;
    float halfHeight = height * 0.5f;

    // Quad facing forward (positive Z direction)
    glm::vec3 normal = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec4 tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

    vertices = {
        { {-halfWidth, -halfHeight, 0.0f}, normal, {0.0f, 0.0f}, tangent },
        { { halfWidth, -halfHeight, 0.0f}, normal, {1.0f, 0.0f}, tangent },
        { { halfWidth,  halfHeight, 0.0f}, normal, {1.0f, 1.0f}, tangent },
        { {-halfWidth,  halfHeight, 0.0f}, normal, {0.0f, 1.0f}, tangent }
    };

    indices = {
        0, 1, 2,
        2, 3, 0
    };
}

void MeshComponent::CreateCube(float size, const glm::vec3& color) {
    float halfSize = size * 0.5f;

    vertices = {
        // Front face (normal: +Z, tangent: +X)
        { {-halfSize, -halfSize,  halfSize}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { { halfSize, -halfSize,  halfSize}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { { halfSize,  halfSize,  halfSize}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { {-halfSize,  halfSize,  halfSize}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },

        // Back face (normal: -Z, tangent: -X)
        { {-halfSize, -halfSize, -halfSize}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f, 1.0f} },
        { {-halfSize,  halfSize, -halfSize}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f} },
        { { halfSize,  halfSize, -halfSize}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f, 1.0f} },
        { { halfSize, -halfSize, -halfSize}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f, 1.0f} },

        // Top face (normal: +Y, tangent: +X)
        { {-halfSize,  halfSize, -halfSize}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { {-halfSize,  halfSize,  halfSize}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { { halfSize,  halfSize,  halfSize}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { { halfSize,  halfSize, -halfSize}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },

        // Bottom face (normal: -Y, tangent: +X)
        { {-halfSize, -halfSize, -halfSize}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { { halfSize, -halfSize, -halfSize}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { { halfSize, -halfSize,  halfSize}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },
        { {-halfSize, -halfSize,  halfSize}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f} },

        // Right face (normal: +X, tangent: -Z)
        { { halfSize, -halfSize, -halfSize}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f} },
        { { halfSize,  halfSize, -halfSize}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 1.0f} },
        { { halfSize,  halfSize,  halfSize}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, -1.0f, 1.0f} },
        { { halfSize, -halfSize,  halfSize}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, -1.0f, 1.0f} },

        // Left face (normal: -X, tangent: +Z)
        { {-halfSize, -halfSize, -halfSize}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f} },
        { {-halfSize, -halfSize,  halfSize}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f} },
        { {-halfSize,  halfSize,  halfSize}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f} },
        { {-halfSize,  halfSize, -halfSize}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f} }
    };

    indices = {
        // Front face
        0, 1, 2, 2, 3, 0,
        // Back face
        4, 5, 6, 6, 7, 4,
        // Top face
        8, 9, 10, 10, 11, 8,
        // Bottom face
        12, 13, 14, 14, 15, 12,
        // Right face
        16, 17, 18, 18, 19, 16,
        // Left face
        20, 21, 22, 22, 23, 20
    };
}

void MeshComponent::CreateSphere(float radius, const glm::vec3& color, int segments) {
    vertices.clear();
    indices.clear();

    // Generate sphere vertices using parametric equations
    for (int lat = 0; lat <= segments; ++lat) {
        float theta = lat * M_PI / segments; // Latitude angle (0 to PI)
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int lon = 0; lon <= segments; ++lon) {
            float phi = lon * 2.0f * M_PI / segments; // Longitude angle (0 to 2*PI)
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            // Calculate position
            glm::vec3 position = {
                radius * sinTheta * cosPhi,
                radius * cosTheta,
                radius * sinTheta * sinPhi
            };

            // Normal is the same as normalized position for a sphere centered at origin
            glm::vec3 normal = glm::normalize(position);

            // Texture coordinates
            glm::vec2 texCoord = {
                (float)lon / segments,
                (float)lat / segments
            };

            // Calculate tangent (derivative with respect to longitude)
            glm::vec3 tangent = {
                -sinTheta * sinPhi,
                0.0f,
                sinTheta * cosPhi
            };
            tangent = glm::normalize(tangent);

            vertices.push_back({
                position,
                normal,
                texCoord,
                glm::vec4(tangent, 1.0f)
            });
        }
    }

    // Generate indices for triangles
    for (int lat = 0; lat < segments; ++lat) {
        for (int lon = 0; lon < segments; ++lon) {
            int current = lat * (segments + 1) + lon;
            int next = current + segments + 1;

            // Create two triangles for each quad
            indices.push_back(current);
            indices.push_back(next);
            indices.push_back(current + 1);

            indices.push_back(current + 1);
            indices.push_back(next);
            indices.push_back(next + 1);
        }
    }
}

void MeshComponent::LoadFromModel(const Model* model) {
    if (!model) {
        return;
    }

    // Copy vertex and index data from the model
    vertices = model->GetVertices();
    indices = model->GetIndices();
}
