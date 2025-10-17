#include "mesh_component.h"
#include "model_loader.h"
#include <cmath>

// Most of the MeshComponent class implementation is in the header file
// This file is mainly for any methods that might need additional implementation

void MeshComponent::CreateSphere(float radius, const glm::vec3& color, int segments) {
    vertices.clear();
    indices.clear();

    // Generate sphere vertices using parametric equations
    for (int lat = 0; lat <= segments; ++lat) {
        auto theta = static_cast<float>(lat * M_PI / segments); // Latitude angle (0 to PI)
        float sinTheta = sinf(theta);
        float cosTheta = cosf(theta);

        for (int lon = 0; lon <= segments; ++lon) {
            auto phi = static_cast<float>(lon * 2.0 * M_PI / segments); // Longitude angle (0 to 2*PI)
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

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
                static_cast<float>(lon) / static_cast<float>(segments),
                static_cast<float>(lat) / static_cast<float>(segments)
            };

            // Calculate tangent (derivative with respect to longitude). Handle poles robustly.
            glm::vec3 tangent = {
                -sinTheta * sinPhi,
                0.0f,
                sinTheta * cosPhi
            };
            float len2 = glm::dot(tangent, tangent);
            if (len2 < 1e-12f) {
                // At poles sinTheta ~ 0 -> fallback tangent orthogonal to normal
                glm::vec3 t = glm::cross(normal, glm::vec3(0.0f, 0.0f, 1.0f));
                if (glm::length(t) < 1e-12f) {
                    t = glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f));
                }
                tangent = glm::normalize(t);
            } else {
                tangent = glm::normalize(tangent);
            }

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

    RecomputeLocalAABB();
}

void MeshComponent::LoadFromModel(const Model* model) {
    if (!model) {
        return;
    }

    // Copy vertex and index data from the model
    vertices = model->GetVertices();
    indices = model->GetIndices();

    RecomputeLocalAABB();
}
