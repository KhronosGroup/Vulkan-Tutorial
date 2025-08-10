#include "scene_loading.h"
#include "engine.h"
#include "transform_component.h"
#include "mesh_component.h"
#include "camera_component.h"
#include <iostream>
#include <set>
#include <map>
#include <filesystem>
#include <algorithm>
#include <glm/gtx/matrix_decompose.hpp>

/**
 * @brief Calculate bounding box dimensions for a MaterialMesh.
 * @param materialMesh The MaterialMesh to analyze.
 * @return The size of the bounding box (max - min for each axis).
 */
glm::vec3 CalculateBoundingBoxSize(const MaterialMesh& materialMesh) {
    if (materialMesh.vertices.empty()) {
        return glm::vec3(0.0f);
    }

    glm::vec3 minBounds = materialMesh.vertices[0].position;
    glm::vec3 maxBounds = materialMesh.vertices[0].position;

    for (const auto& vertex : materialMesh.vertices) {
        minBounds = glm::min(minBounds, vertex.position);
        maxBounds = glm::max(maxBounds, vertex.position);
    }

    return maxBounds - minBounds;
}

/**
 * @brief Load a GLTF model synchronously on the main thread.
 * @param engine The engine to create entities in.
 * @param modelPath The path to the GLTF model file.
 * @param position The position to place the model (default: origin with slight Y offset).
 * @param rotation The rotation to apply to the model (default: no rotation).
 * @param scale The scale to apply to the model (default: unit scale).
 */
void LoadGLTFModel(Engine* engine, const std::string& modelPath,
                   const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale) {
    // Get the model loader and renderer
    ModelLoader* modelLoader = engine->GetModelLoader();
    Renderer* renderer = engine->GetRenderer();

    if (!modelLoader || !renderer) {
        std::cerr << "Error: ModelLoader or Renderer is null" << std::endl;
        return;
    }

    // Extract model name from file path for entity naming
    std::filesystem::path modelFilePath(modelPath);
    std::string modelName = modelFilePath.stem().string(); // Get filename without extension

    try {
        // Load the complete GLTF model with all textures and lighting on the main thread
        Model* loadedModel = modelLoader->LoadGLTF(modelPath);
        if (!loadedModel) {
            std::cerr << "Failed to load GLTF model: " << modelPath << std::endl;
            return;
        }

        std::cout << "Successfully loaded GLTF model with all textures and lighting: " << modelPath << std::endl;

        // Extract lights from the model and transform them to world space
        std::vector<ExtractedLight> extractedLights = modelLoader->GetExtractedLights(modelPath);

        // Create a transformation matrix from position, rotation, and scale
        glm::mat4 transformMatrix = glm::mat4(1.0f);
        transformMatrix = glm::translate(transformMatrix, position);
        transformMatrix = glm::rotate(transformMatrix, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        transformMatrix = glm::rotate(transformMatrix, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        transformMatrix = glm::rotate(transformMatrix, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
        transformMatrix = glm::scale(transformMatrix, scale);

        // Transform all light positions from local model space to world space
        for (auto& light : extractedLights) {
            glm::vec4 worldPos = transformMatrix * glm::vec4(light.position, 1.0f);
            light.position = glm::vec3(worldPos);

            // Also transform the light direction (for directional lights)
            glm::mat3 normalMatrix = glm::mat3(glm::transpose(glm::inverse(transformMatrix)));
            light.direction = glm::normalize(normalMatrix * light.direction);
        }

        renderer->SetStaticLights(extractedLights);

        // Extract and apply cameras from the GLTF model
        const std::vector<CameraData>& cameras = loadedModel->GetCameras();
        if (!cameras.empty()) {
            const CameraData& gltfCamera = cameras[0]; // Use the first camera

            // Find or create a camera entity to replace the default one
            Entity* cameraEntity = engine->GetEntity("Camera");
            if (!cameraEntity) {
                // Create a new camera entity if none exists
                cameraEntity = engine->CreateEntity("Camera");
                if (cameraEntity) {
                    cameraEntity->AddComponent<TransformComponent>();
                    cameraEntity->AddComponent<CameraComponent>();
                }
            }

            if (cameraEntity) {
                // Update the camera transform with GLTF data
                auto* cameraTransform = cameraEntity->GetComponent<TransformComponent>();
                if (cameraTransform) {
                    // Apply the transformation matrix to the camera position
                    glm::vec4 worldPos = transformMatrix * glm::vec4(gltfCamera.position, 1.0f);
                    cameraTransform->SetPosition(glm::vec3(worldPos));

                    // Apply rotation from GLTF camera
                    glm::vec3 eulerAngles = glm::eulerAngles(gltfCamera.rotation);
                    cameraTransform->SetRotation(eulerAngles);
                }

                // Update the camera component with GLTF properties
                auto* camera = cameraEntity->GetComponent<CameraComponent>();
                if (camera) {
                    camera->ForceViewMatrixUpdate(); // Only sets viewMatrixDirty flag, doesn't change camera orientation
                    if (gltfCamera.isPerspective) {
                        camera->SetFieldOfView(glm::degrees(gltfCamera.fov)); // Convert radians to degrees
                        camera->SetClipPlanes(gltfCamera.nearPlane, gltfCamera.farPlane);
                        if (gltfCamera.aspectRatio > 0.0f) {
                            camera->SetAspectRatio(gltfCamera.aspectRatio);
                        }
                    } else {
                        // Handle orthographic camera if needed
                        camera->SetProjectionType(CameraComponent::ProjectionType::Orthographic);
                        camera->SetOrthographicSize(gltfCamera.orthographicSize, gltfCamera.orthographicSize);
                        camera->SetClipPlanes(gltfCamera.nearPlane, gltfCamera.farPlane);
                    }

                    // Set this as the active camera
                    engine->SetActiveCamera(camera);
                }
            }
        }

        // Get the material meshes from the loaded model
        const std::vector<MaterialMesh>& materialMeshes = modelLoader->GetMaterialMeshes(modelPath);
        if (materialMeshes.empty()) {
            std::cerr << "No material meshes found in loaded model: " << modelPath << std::endl;
            return;
        }

        int entitiesCreated = 0;
        for (const auto& materialMesh : materialMeshes) {
            // Create an entity name based on model and material
            std::string entityName = modelName + "_Material_" + std::to_string(materialMesh.materialIndex) +
                                    "_" + materialMesh.materialName;

            if (Entity* materialEntity = engine->CreateEntity(entityName)) {
                // Add a transform component with provided parameters
                auto* transform = materialEntity->AddComponent<TransformComponent>();
                transform->SetPosition(position);
                transform->SetRotation(glm::radians(rotation));
                transform->SetScale(scale);

                // Add a mesh component with material-specific data
                auto* mesh = materialEntity->AddComponent<MeshComponent>();
                mesh->SetVertices(materialMesh.vertices);
                mesh->SetIndices(materialMesh.indices);

                if (materialMesh.GetInstanceCount() > 0) {
                    const std::vector<InstanceData>& instances = materialMesh.instances;
                    for (const auto& instanceData : instances) {
                        // Reconstruct the transformation matrix from InstanceData column vectors
                        glm::mat4 instanceMatrix = instanceData.getModelMatrix();
                        mesh->AddInstance(instanceMatrix, static_cast<uint32_t>(materialMesh.materialIndex));
                    }
                }

                // Set ALL PBR texture paths for this material
                // Set primary texture path for backward compatibility
                if (!materialMesh.texturePath.empty()) {
                    mesh->SetTexturePath(materialMesh.texturePath);
                }

                // Set all PBR texture paths
                if (!materialMesh.baseColorTexturePath.empty()) {
                    mesh->SetBaseColorTexturePath(materialMesh.baseColorTexturePath);
                }
                if (!materialMesh.normalTexturePath.empty()) {
                    mesh->SetNormalTexturePath(materialMesh.normalTexturePath);
                }
                if (!materialMesh.metallicRoughnessTexturePath.empty()) {
                    mesh->SetMetallicRoughnessTexturePath(materialMesh.metallicRoughnessTexturePath);
                }
                if (!materialMesh.occlusionTexturePath.empty()) {
                    mesh->SetOcclusionTexturePath(materialMesh.occlusionTexturePath);
                }
                if (!materialMesh.emissiveTexturePath.empty()) {
                    mesh->SetEmissiveTexturePath(materialMesh.emissiveTexturePath);
                }

                // Fallback: Use material DB (from ModelLoader) if any PBR texture is still missing
                if (modelLoader) {
                    Material* mat = modelLoader->GetMaterial(materialMesh.materialName);
                    if (mat) {
                        if (mesh->GetBaseColorTexturePath().empty() && !mat->albedoTexturePath.empty()) {
                            mesh->SetBaseColorTexturePath(mat->albedoTexturePath);
                        }
                        if (mesh->GetNormalTexturePath().empty() && !mat->normalTexturePath.empty()) {
                            mesh->SetNormalTexturePath(mat->normalTexturePath);
                        }
                        if (mesh->GetMetallicRoughnessTexturePath().empty() && !mat->metallicRoughnessTexturePath.empty()) {
                            mesh->SetMetallicRoughnessTexturePath(mat->metallicRoughnessTexturePath);
                        }
                        if (mesh->GetOcclusionTexturePath().empty() && !mat->occlusionTexturePath.empty()) {
                            mesh->SetOcclusionTexturePath(mat->occlusionTexturePath);
                        }
                        if (mesh->GetEmissiveTexturePath().empty() && !mat->emissiveTexturePath.empty()) {
                            mesh->SetEmissiveTexturePath(mat->emissiveTexturePath);
                        }
                    }
                }

                // Pre-allocate all Vulkan resources for this entity
                if (!renderer->preAllocateEntityResources(materialEntity)) {
                    std::cerr << "Failed to pre-allocate resources for entity: " << entityName << std::endl;
                    // Continue with other entities even if one fails
                }

                // Create physics body for collision with balls
                // Use mesh collision shape for accurate geometry interaction
                PhysicsSystem* physicsSystem = engine->GetPhysicsSystem();
                if (physicsSystem) {
                    RigidBody* rigidBody = physicsSystem->CreateRigidBody(materialEntity, CollisionShape::Mesh, 0.0f); // Mass 0 = static
                    if (rigidBody) {
                        rigidBody->SetKinematic(true); // Static geometry doesn't move
                        rigidBody->SetRestitution(0.15f); // Very low bounce - balls lose 85%+ momentum
                        rigidBody->SetFriction(0.5f); // Moderate friction
                        std::cout << "Created physics body for geometry entity: " << entityName << std::endl;
                    } else {
                        std::cerr << "Failed to create physics body for entity: " << entityName << std::endl;
                    }
                }

                entitiesCreated++;
            } else {
                std::cerr << "Failed to create entity for material " << materialMesh.materialName << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading GLTF model: " << e.what() << std::endl;
    }
}

/**
 * @brief Load a GLTF model with default transform values.
 * @param engine The engine to create entities in.
 * @param modelPath The path to the GLTF model file.
 */
void LoadGLTFModel(Engine* engine, const std::string& modelPath) {
    // Use default transform values: slight Y offset, no rotation, unit scale
    LoadGLTFModel(engine, modelPath, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
}
