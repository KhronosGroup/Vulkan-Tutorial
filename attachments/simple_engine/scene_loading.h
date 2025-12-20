#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "model_loader.h"

// Forward declarations
class Engine;
class ModelLoader;

/**
 * @brief Load a GLTF model synchronously on the main thread.
 * @param engine The engine to create entities in.
 * @param modelPath The path to the GLTF model file.
 * @param position The position to place the model.
 * @param rotation The rotation to apply to the model.
 * @param scale The scale to apply to the model.
 */
bool LoadGLTFModel(Engine* engine, const std::string& modelPath,
                   const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale);

/**
 * @brief Load a GLTF model with default transform values.
 * @param engine The engine to create entities in.
 * @param modelPath The path to the GLTF model file.
 */
void LoadGLTFModel(Engine* engine, const std::string& modelPath);
