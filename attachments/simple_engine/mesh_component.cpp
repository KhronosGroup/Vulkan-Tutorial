#include "mesh_component.h"
#include "model_loader.h"

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

void MeshComponent::LoadFromModel(const Model* model) {
    if (!model) {
        return;
    }

    // Copy vertex and index data from the model
    vertices = model->GetVertices();
    indices = model->GetIndices();
}
