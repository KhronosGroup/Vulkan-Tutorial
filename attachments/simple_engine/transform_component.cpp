#include "transform_component.h"

// Most of the TransformComponent class implementation is in the header file
// This file is mainly for any methods that might need additional implementation
//
// This implementation corresponds to the Camera_Transformations chapter in the tutorial:
// @see en/Building_a_Simple_Engine/Camera_Transformations/04_transformation_matrices.adoc#model-matrix

// Returns the model matrix, updating it if necessary
// @see en/Building_a_Simple_Engine/Camera_Transformations/04_transformation_matrices.adoc#model-matrix
const glm::mat4& TransformComponent::GetModelMatrix() {
    if (matrixDirty) {
        UpdateModelMatrix();
    }
    return modelMatrix;
}

// Updates the model matrix based on position, rotation, and scale
// @see en/Building_a_Simple_Engine/Camera_Transformations/04_transformation_matrices.adoc#model-matrix
void TransformComponent::UpdateModelMatrix() {
    // Compose rotation with quaternions for stability and to avoid rad/deg ambiguity
    glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
    glm::quat qx = glm::angleAxis(rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat qy = glm::angleAxis(rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    glm::quat qz = glm::angleAxis(rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    glm::quat q = qz * qy * qx; // ZYX order is conventional for Euler composition
    glm::mat4 R = glm::mat4_cast(q);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    modelMatrix = T * R * S;
    matrixDirty = false;
}
