/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <filesystem>
#include <glm/gtx/matrix_decompose.hpp>
#include <iostream>
#include <chrono>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include "renderer.h"

#include "scene_loading.h"
#include "animation_component.h"
#include "camera_component.h"
#include "engine.h"
#include "mesh_component.h"
#include "renderer_advanced_types.h"
#include "transform_component.h"

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
 * @return success or fail on loading the GLTF model.
 * @param engine The engine to create entities in.
 * @param modelPath The path to the GLTF model file.
 * @param position The position to place the model (default: origin with slight Y offset).
 * @param rotation The rotation to apply to the model (default: no rotation).
 * @param scale The scale to apply to the model (default: unit scale).
 */
bool LoadGLTFModel(Engine* engine,
                   const std::string& modelPath,
                   const glm::vec3& position,
                   const glm::vec3& rotation,
                   const glm::vec3& scale,
                   float mass) {
  // Get the model loader and renderer
  ModelLoader* modelLoader = engine->GetModelLoader();
  Renderer* renderer = engine->GetRenderer();

  if (!modelLoader || !renderer) {
    std::cerr << "Error: ModelLoader or Renderer is null" << std::endl;
    return false;
  }

  // Only show the blocking loading overlay for the initial load.
  // Subsequent loads (Fox, Cube) happen in the background without UI interruption.
  bool wasInitialLoadComplete = renderer->initialLoadComplete.load(std::memory_order_relaxed);
  if (!wasInitialLoadComplete) {
    renderer->SetLoading(true);
  }
  renderer->SetLoadingPhase(Renderer::LoadingPhase::Textures);
  // Ensure loading flag is cleared on any exit from this function
  struct LoadingGuard {
    Renderer* r;
    bool needsClear;
    ~LoadingGuard() {
      if (needsClear) {
        r->SetLoading(false);
      }
    }
  } loadingGuard{renderer, !wasInitialLoadComplete};

  // Extract model name from file path for entity naming
  std::filesystem::path modelFilePath(modelPath);
  std::string modelName = modelFilePath.stem().string(); // Get filename without extension

  try {
    const auto loadStart = std::chrono::steady_clock::now();
    std::cout << "[Loading] Begin: " << modelPath << std::endl;

    // Suppress watchdog during heavy loading (allowed 60s instead of 10s)
    renderer->watchdogSuppressed.store(true, std::memory_order_relaxed);

    // Loading large scenes can produce tens of thousands of entities.
    // Avoid per-entity stdout spam (very slow on Windows consoles) and instead
    // keep counters + print occasional summaries.
    size_t physicsBodiesQueued = 0;
    size_t physicsBodiesSkipped = 0;
    size_t physicsNoGeometry = 0;
    auto maybeLogPhysicsProgress = [&]() {
      const size_t total = physicsBodiesQueued + physicsBodiesSkipped + physicsNoGeometry;
      // Log infrequently to keep visibility without tanking load time.
      if (total > 0 && (total % 5000u) == 0u) {
        std::cout << "[Loading] Physics bodies: queued=" << physicsBodiesQueued
            << ", skipped=" << physicsBodiesSkipped
            << ", noGeometry=" << physicsNoGeometry << std::endl;
      }
    };
    // Load the complete GLTF model with all textures and lighting on the main thread
    Model* loadedModel = modelLoader->LoadGLTF(modelPath);
    if (!loadedModel) {
      std::cerr << "Failed to load GLTF model: " << modelPath << std::endl;
      return false;
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
    // Also transform the light direction (for directional lights)
    glm::mat3 normalMatrix = glm::mat3(glm::transpose(glm::inverse(transformMatrix)));
    for (auto& light : extractedLights) {
      glm::vec4 worldPos = transformMatrix * glm::vec4(light.position, 1.0f);
      light.position = glm::vec3(worldPos);
      light.direction = glm::normalize(normalMatrix * light.direction);
    }

    // Append rather than overwrite
    for (const auto& light : extractedLights) {
      renderer->staticLights.push_back(light);
    }
    std::cout << "[Lights] staticLights appended: " << extractedLights.size() << " entries (total: " << renderer->staticLights.size() << ")" << std::endl;

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
      return false;
    }

    // Collect all geometry entities so we can batch Vulkan uploads for their meshes
    std::vector<Entity *> geometryEntities;
    geometryEntities.reserve(materialMeshes.size());

    // Phase: Physics (queue colliders / rigid bodies). This is CPU-side work that can
    // take noticeable time even after textures have finished scheduling.
    renderer->SetLoadingPhase(Renderer::LoadingPhase::Physics);
    renderer->SetLoadingPhaseProgress(0.0f);

    for (size_t meshIdx = 0; meshIdx < materialMeshes.size(); ++meshIdx) {
      AdvancedRenderer_KickWatchdog(renderer);
      const auto& materialMesh = materialMeshes[meshIdx];

      // Skip materials that have no geometry assigned to them
      if (materialMesh.vertices.empty() || materialMesh.indices.empty()) {
          continue;
      }

      if ((meshIdx % 64u) == 0u) {
        renderer->SetLoadingPhaseProgress(materialMeshes.empty() ? 0.0f : (static_cast<float>(meshIdx) / static_cast<float>(materialMeshes.size())));
      }
      // Create an entity name based on model and material. Use the globally-unique material
      // index so the ray-query material slot (parsed back out of this name) doesn't collide
      // with same-numbered materials from other models.
      const int entityMaterialIndex = (materialMesh.globalMaterialIndex >= 0)
          ? materialMesh.globalMaterialIndex : materialMesh.materialIndex;
      std::string entityName = modelName + "_Material_" + std::to_string(entityMaterialIndex) +
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

        // Optimization: Pre-calculate local AABB on the background thread.
        // This caches the AABB once per mesh and avoids slow vertex scans on the main thread
        // or during the physics ground check below.
        mesh->RecomputeLocalAABB();

        bool isDef = IsMaterialMeshDeformable(&materialMesh);
        SetMeshComponentDeformable(mesh, isDef);
        int numTargets = GetMaterialMeshMorphTargetCount(&materialMesh);
        SetMeshComponentMorphTargets(mesh, numTargets);
        SetMeshComponentEnvironment(mesh, entityName.find("bistro") != std::string::npos);
        if (numTargets > 0) {
            SetMeshComponentMorphPositions(mesh, GetMaterialMeshMorphPositions(&materialMesh));
        }
        if (isDef || numTargets > 0) {
            std::cout << "[Loading] Entity " << materialEntity->GetName() << " has deformable/morph data (skinned=" << isDef << ", morphTargets=" << numTargets << ")" << std::endl;
            if (isDef) {
                SetMeshComponentJointsAndWeights(mesh, GetMaterialMeshJoints(&materialMesh), GetMaterialMeshWeights(&materialMesh));
            }
        }

        if (materialMesh.GetInstanceCount() > 0) {
          mesh->SetInstances(materialMesh.instances);
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
          const Material* mat = modelLoader->GetMaterial(materialMesh.materialName);
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

        // Register all effective texture IDs this mesh uses so that when
        // textures finish streaming in, the renderer can refresh
        // descriptor sets for the appropriate entities. This must
        // happen *after* material fallbacks so we see the final IDs.
        auto registerTex = [&](const std::string& texId) {
          if (!texId.empty()) {
            renderer->RegisterTextureUser(texId, materialEntity);
          }
        };

        registerTex(mesh->GetTexturePath());
        registerTex(mesh->GetBaseColorTexturePath());
        registerTex(mesh->GetNormalTexturePath());
        registerTex(mesh->GetMetallicRoughnessTexturePath());
        registerTex(mesh->GetOcclusionTexturePath());
        registerTex(mesh->GetEmissiveTexturePath());

        // Track this entity for batched Vulkan resource pre-allocation later
        geometryEntities.push_back(materialEntity);

        // Create a physics body so dynamic objects (balls, the released Fox)
        // collide with this geometry.
        PhysicsSystem* physicsSystem = engine->GetPhysicsSystem();
        if (physicsSystem) {
          auto* mc = materialEntity->GetComponent<MeshComponent>();
          if (mc && !mc->GetVertices().empty() && !mc->GetIndices().empty()) {
            // Compute the world-space bounds (from the entity transform and the
            // mesh's local AABB, or the vertices as a fallback) to derive the
            // collider's center used for distance-based streaming.
            glm::vec3 minWS(std::numeric_limits<float>::max());
            glm::vec3 maxWS(-std::numeric_limits<float>::max());

            auto* xform = materialEntity->GetComponent<TransformComponent>();
            glm::mat4 model = xform ? xform->GetModelMatrix() : glm::mat4(1.0f);

            if (mc->HasLocalAABB()) {
              glm::vec3 localMin = mc->GetLocalAABBMin();
              glm::vec3 localMax = mc->GetLocalAABBMax();

              // Transform the 8 corners of the local AABB to world space
              for (int ix = 0; ix < 2; ++ix) {
                for (int iy = 0; iy < 2; ++iy) {
                  for (int iz = 0; iz < 2; ++iz) {
                    glm::vec3 corner(
                      ix ? localMax.x : localMin.x,
                      iy ? localMax.y : localMin.y,
                      iz ? localMax.z : localMin.z);
                    glm::vec3 cWS = glm::vec3(model * glm::vec4(corner, 1.0f));
                    minWS = glm::min(minWS, cWS);
                    maxWS = glm::max(maxWS, cWS);
                  }
                }
              }
            } else {
              // Fallback: compute bounds directly from vertices in world space
              const auto& verts = mc->GetVertices();
              for (const auto& v : verts) {
                glm::vec3 pWS = glm::vec3(model * glm::vec4(v.position, 1.0f));
                minWS = glm::min(minWS, pWS);
                maxWS = glm::max(maxWS, pWS);
              }
            }

            if (mass > 0.0f) {
              // Dynamic objects (Fox, balls, etc.) — created up-front since
              // they need to be active immediately.
              physicsSystem->EnqueueRigidBodyCreation(
                materialEntity,
                CollisionShape::Box,
                mass,
                false,
                0.15f,
                0.5f
              );
              ++physicsBodiesQueued;
              maybeLogPhysicsProgress();
            } else {
              // Static environment colliders use a triangle MESH shape so
              // dynamic objects collide with the actual surface rather than the
              // filled volume of an axis-aligned bounding box. They are
              // registered for distance-based streaming: each mesh is a
              // candidate, and only those near the camera are promoted to live
              // Jolt bodies, so we keep ~tens of active bodies instead of 500+.
              glm::vec3 center = (minWS + maxWS) * 0.5f;
              physicsSystem->RegisterStreamingCollider(
                materialEntity,
                CollisionShape::Mesh,
                mass,
                false,
                0.15f,
                0.5f,
                center
              );
              ++physicsBodiesQueued;
              maybeLogPhysicsProgress();
            }
          } else {
            ++physicsNoGeometry;
            maybeLogPhysicsProgress();
          }
        }
      } else {
        std::cerr << "Failed to create entity for material " << materialMesh.materialName << std::endl;
      }
    }
    renderer->SetLoadingPhaseProgress(1.0f);

    // Pre-allocate Vulkan resources for all geometry entities in a single batched pass
    if (!geometryEntities.empty()) {
      // Scene loading runs on a background thread. Do NOT perform Vulkan allocations
      // or mutate renderer resource maps here. Enqueue the batch so the render thread can
      // perform the GPU work safely at its frame-start safe point.
      renderer->EnqueueEntityPreallocationBatch(geometryEntities);
    }

    // Final loading summary (useful for profiling, low-noise)
    std::cout << "[Loading] Physics bodies summary: queued=" << physicsBodiesQueued
        << ", skipped=" << physicsBodiesSkipped
        << ", noGeometry=" << physicsNoGeometry << std::endl;

    const auto loadEnd = std::chrono::steady_clock::now();
    const auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart).count();
    const auto loadSecs = static_cast<double>(loadMs) / 1000.0;
    const bool loadFastOk = loadSecs <= 60.0;
    std::cout << "[Loading] End: " << modelPath << " in " << loadSecs << "s" << (loadFastOk ? "" : " (SLOW)") << std::endl;

    // Set up animations if the model has any
    const std::vector<Animation>& animations = loadedModel->GetAnimations();
    std::cout << "[Animation] Model has " << animations.size() << " animation(s)" << std::flush << std::endl;
    if (!animations.empty()) {
      std::cout << "[Animation] Setting up " << animations.size() << " animation(s) for playback" << std::flush << std::endl;

      // Create an animation controller entity
      Entity* animController = engine->CreateEntity(modelName + "_AnimController");
      if (animController) {
        auto* animTransform = animController->AddComponent<TransformComponent>();
        animTransform->SetPosition(position);
        animTransform->SetRotation(glm::radians(rotation));
        animTransform->SetScale(scale);

        auto* animComponent = animController->AddComponent<AnimationComponent>();
        animComponent->SetAnimations(animations);

        // Build node-to-entity mapping using actual glTF node indices
        // Get animated node mesh mappings to link geometry entities to animated nodes
        const auto& advanced = GetAdvancedModelData(loadedModel);
        const auto& animatedNodeMeshes = loadedModel->GetAnimatedNodeMeshes();

        // Get the base transforms for animated nodes
        const auto& animatedNodeTransforms = loadedModel->GetAnimatedNodeTransforms();
        const auto& nodeSkins = advanced.nodeSkins;
        const auto& modelSkins = advanced.skins;

        std::cout << "[Animation] Processing " << animatedNodeMeshes.size() << " animated nodes" << std::endl;

        // Build nodeToEntity mapping by creating or finding entities for each animated node
        std::unordered_map<int, std::vector<Entity *>> nodeToEntities;
        std::unordered_map<int, int> meshUsageCount; // Track how many times each mesh is used

        // First pass: count how many animated nodes use each mesh
        for (const auto& [nodeIndex, meshIndex] : animatedNodeMeshes) {
          meshUsageCount[meshIndex]++;
        }

        // Optimization: build a quick lookup map from sourceMeshIndex to materialMesh indices
        std::unordered_map<int, std::vector<size_t>> meshToMaterialIdx;
        for (size_t i = 0; i < materialMeshes.size(); ++i) {
            meshToMaterialIdx[materialMeshes[i].sourceMeshIndex].push_back(i);
        }

        // Second pass: create entities for animated nodes.
        // Each base geometry entity (created in the static pass above) may be claimed by
        // exactly ONE animated node — the first one that references that primitive — which
        // repurposes it as an animated entity and clears its static instances. Any further
        // nodes that share the primitive get their own _AnimNode_ entity. Tracking the
        // claimed base entities is essential: if a primitive used by several nodes (e.g. the
        // repeated bistro fans) never claims/clears its base entity, that base keeps
        // rendering a frozen, un-animated copy at its load-time pose alongside the animated
        // nodes — i.e. a ghost duplicate of every animated object.
        std::unordered_set<size_t> claimedGeometryIdx;
        for (const auto& [nodeIndex, meshIndex] : animatedNodeMeshes) {
          std::cout << "[Animation] Processing animated node " << nodeIndex << " with mesh " << meshIndex << std::endl;

          auto it = meshToMaterialIdx.find(meshIndex);
          if (it == meshToMaterialIdx.end()) continue;

          for (size_t sourceMaterialMeshIdx : it->second) {
            const MaterialMesh* sourceMaterialMesh = &materialMeshes[sourceMaterialMeshIdx];
            if (!sourceMaterialMesh || sourceMaterialMesh->vertices.empty() || sourceMaterialMesh->indices.empty()) continue;

            Entity* nodeEntity = nullptr;

            // Reuse the base geometry entity the first time this primitive is claimed by an
            // animated node; subsequent nodes (and out-of-range primitives) create fresh
            // _AnimNode_ entities.
            const bool firstClaim = (sourceMaterialMeshIdx < geometryEntities.size()) &&
                                    claimedGeometryIdx.insert(sourceMaterialMeshIdx).second;
            if (firstClaim) {
              nodeEntity = geometryEntities[sourceMaterialMeshIdx];
              auto* mesh = nodeEntity->GetComponent<MeshComponent>();
              if (mesh && mesh->GetInstanceCount() > 0) {
                mesh->ClearInstances();
                renderer->EnqueueInstanceBufferRecreation(nodeEntity);
              }
            } else {
              const int animMaterialIndex = (sourceMaterialMesh->globalMaterialIndex >= 0)
                  ? sourceMaterialMesh->globalMaterialIndex : sourceMaterialMesh->materialIndex;
              std::string entityName = modelName + "_AnimNode_" + std::to_string(nodeIndex) +
                  "_Material_" + std::to_string(animMaterialIndex);
              nodeEntity = engine->CreateEntity(entityName);
              if (nodeEntity) {
                nodeEntity->AddComponent<TransformComponent>();
                auto* mesh = nodeEntity->AddComponent<MeshComponent>();
                mesh->SetVertices(sourceMaterialMesh->vertices);
                mesh->SetIndices(sourceMaterialMesh->indices);

                bool isDef = IsMaterialMeshDeformable(sourceMaterialMesh);
                SetMeshComponentDeformable(mesh, isDef);
                if (isDef) {
                    SetMeshComponentJointsAndWeights(mesh, GetMaterialMeshJoints(sourceMaterialMesh), GetMaterialMeshWeights(sourceMaterialMesh));
                    SetMeshComponentMorphTargets(mesh, GetMaterialMeshMorphTargetCount(sourceMaterialMesh));
                    SetMeshComponentMorphPositions(mesh, GetMaterialMeshMorphPositions(sourceMaterialMesh));
                }

                if (!sourceMaterialMesh->baseColorTexturePath.empty()) mesh->SetBaseColorTexturePath(sourceMaterialMesh->baseColorTexturePath);
                if (!sourceMaterialMesh->normalTexturePath.empty()) mesh->SetNormalTexturePath(sourceMaterialMesh->normalTexturePath);
                if (!sourceMaterialMesh->metallicRoughnessTexturePath.empty()) mesh->SetMetallicRoughnessTexturePath(sourceMaterialMesh->metallicRoughnessTexturePath);
                if (!sourceMaterialMesh->occlusionTexturePath.empty()) mesh->SetOcclusionTexturePath(sourceMaterialMesh->occlusionTexturePath);
                if (!sourceMaterialMesh->emissiveTexturePath.empty()) mesh->SetEmissiveTexturePath(sourceMaterialMesh->emissiveTexturePath);

                renderer->RegisterTextureUser(mesh->GetBaseColorTexturePath(), nodeEntity);
                renderer->RegisterTextureUser(mesh->GetNormalTexturePath(), nodeEntity);
                renderer->RegisterTextureUser(mesh->GetMetallicRoughnessTexturePath(), nodeEntity);
                renderer->RegisterTextureUser(mesh->GetOcclusionTexturePath(), nodeEntity);
                renderer->RegisterTextureUser(mesh->GetEmissiveTexturePath(), nodeEntity);

                renderer->EnqueueEntityPreallocationBatch({nodeEntity});
                std::cout << "[Animation] Created new entity '" << entityName << "' for node " << nodeIndex << std::endl;
              }
            }

            if (nodeEntity) {
              auto transformIt = animatedNodeTransforms.find(nodeIndex);
              if (transformIt != animatedNodeTransforms.end()) {
                glm::mat4 worldNodeTransform = transformMatrix * transformIt->second;
                glm::vec3 nodePosition, nodeScale, skew;
                glm::quat nodeRotation;
                glm::vec4 perspective;
                glm::decompose(worldNodeTransform, nodeScale, nodeRotation, nodePosition, skew, perspective);

                auto* transform = nodeEntity->GetComponent<TransformComponent>();
                if (transform) {
                  transform->SetPosition(nodePosition);
                  transform->SetRotation(glm::eulerAngles(nodeRotation));
                  transform->SetScale(nodeScale);
                }
              }

              nodeToEntities[nodeIndex].push_back(nodeEntity);

              auto skinIt = nodeSkins.find(nodeIndex);
              if (skinIt != nodeSkins.end()) {
                int skinIndex = skinIt->second;
                if (skinIndex >= 0 && skinIndex < static_cast<int>(modelSkins.size())) {
                  const auto& skin = modelSkins[skinIndex];
                  auto* mesh = nodeEntity->GetComponent<MeshComponent>();
                  if (mesh) {
                    SetMeshComponentSkinData(mesh, skin.joints, skin.inverseBindMatrices);
                  }
                }
              }
            }
          }
        }

        animComponent->SetNodeToEntityMap(nodeToEntities);
        AnimationComponent_SetHierarchy(animComponent,
                                   advanced.nodeChildren,
                                   advanced.nodeLocalTransforms,
                                   advanced.nodeLocalTranslations,
                                   advanced.nodeLocalRotations,
                                   advanced.nodeLocalScales,
                                   advanced.rootNodes);

        std::cout << "[Animation] Node-to-entity mapping has " << nodeToEntities.size()
            << " entries (of " << animatedNodeMeshes.size() << " animated nodes)" << std::endl;

        // Auto-play the first animation
        if (!animations.empty()) {
          animComponent->Play(0, true); // Play first animation, looping
          std::cout << "Auto-playing animation: " << animations[0].name
              << " (duration: " << animations[0].GetDuration() << "s)" << std::endl;
        }
      }
    }
    renderer->watchdogSuppressed.store(false, std::memory_order_relaxed);
  } catch (const std::exception& e) {
    renderer->watchdogSuppressed.store(false, std::memory_order_relaxed);
    std::cerr << "Error loading GLTF model: " << e.what() << std::endl;
    return false;
  }

  // Request acceleration structure build at next safe frame point
  // Don't build here in background thread to avoid threading issues with command pools
  const bool needsAS = renderer->GetRayQueryEnabled() && renderer->GetAccelerationStructureEnabled();
  if (needsAS) {
    renderer->SetLoadingPhase(Renderer::LoadingPhase::AccelerationStructures);
    renderer->SetLoadingPhaseProgress(0.0f);
    std::cout << "Requesting acceleration structure build for loaded scene..." << std::endl;
    renderer->RequestAccelerationStructureBuild();
  }

  // Clear the scene loader flag so the render thread knows asset construction is done.
  // IMPORTANT: We deliberately do NOT call MarkInitialLoadComplete() here. Doing so would
  // hide the loading overlay before the acceleration structure (BLAS+TLAS) has actually
  // finished building. With chunked GPU resource preallocation (100 entities/frame), the
  // TLAS cannot be safely built until all BLAS are ready -- otherwise the GPU will hang
  // dereferencing instances that point to non-existent BLAS.
  //
  // Instead, transition to the Finalizing phase and let the render loop's auto-completion
  // check (renderer_rendering.cpp, around the "MarkInitialLoadComplete" call site) flip
  // the flag once asBuildRequested == false (i.e., a full successful AS build).
  renderer->SetLoading(false);
  if (needsAS) {
    // Stay in AS phase; render loop will switch to Finalizing on first successful build.
  } else {
    // No ray query: nothing to wait on, complete immediately.
    renderer->MarkInitialLoadComplete();
  }

  return true;
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
