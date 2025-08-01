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
    modelMatrix = glm::mat4(1.0f);
    modelMatrix = glm::translate(modelMatrix, position);
    modelMatrix = glm::rotate(modelMatrix, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    modelMatrix = glm::rotate(modelMatrix, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    modelMatrix = glm::rotate(modelMatrix, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    modelMatrix = glm::scale(modelMatrix, scale);
    matrixDirty = false;
}
