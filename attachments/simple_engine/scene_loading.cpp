#include "scene_loading.h"
#include "engine.h"
#include "transform_component.h"
#include "mesh_component.h"
#include <iostream>
#include <thread>

// Global loading state definition
LoadingState g_loadingState;

/**
 * @brief Background thread function to load the Bistro model.
 * @param modelLoader Pointer to the model loader.
 */
void LoadBistroModelAsync(ModelLoader* modelLoader) {
    try {
        std::cout << "Starting thread-safe background loading of Bistro model..." << std::endl;
        g_loadingState.isLoading = true;
        g_loadingState.loadingComplete = false;
        g_loadingState.loadingFailed = false;

        // Parse GLTF data without creating Vulkan resources (thread-safe)
        std::vector<MaterialMesh> materialMeshes = modelLoader->ParseGLTFDataOnly("../Assets/Bistro.glb");
        if (materialMeshes.empty()) {
            g_loadingState.errorMessage = "Failed to parse Bistro.glb - no material meshes found";
            g_loadingState.loadingFailed = true;
            g_loadingState.isLoading = false;
            return;
        }

        g_loadingState.totalMaterials = static_cast<int>(materialMeshes.size());

        std::cout << "Parsed " << materialMeshes.size() << " materials in background thread (thread-safe)" << std::endl;

        // Store the loaded materials (thread-safe copy)
        {
            std::lock_guard<std::mutex> lock(g_loadingState.entityCreationMutex);
            g_loadingState.loadedMaterials = std::move(materialMeshes);
        }

        g_loadingState.loadingComplete = true;
        g_loadingState.isLoading = false;
        std::cout << "Thread-safe background loading completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        g_loadingState.errorMessage = std::string("Thread-safe loading error: ") + e.what();
        g_loadingState.loadingFailed = true;
        g_loadingState.isLoading = false;
        std::cerr << "Thread-safe background loading failed: " << e.what() << std::endl;
    }
}

/**
 * @brief Create entities from loaded materials (called from main thread).
 * @param engine The engine to create entities in.
 */
void CreateEntitiesFromLoadedMaterials(Engine* engine) {
    std::lock_guard<std::mutex> lock(g_loadingState.entityCreationMutex);

    if (g_loadingState.loadedMaterials.empty()) {
        return;
    }

    std::cout << "Creating " << g_loadingState.loadedMaterials.size() << " entities from loaded materials..." << std::endl;

    // Get the model loader and renderer for texture loading on main thread
    ModelLoader* modelLoader = engine->GetModelLoader();
    Renderer* renderer = engine->GetRenderer();

    // First, load the actual GLTF model with Vulkan resources on the main thread
    // This will create all the textures that the background thread couldn't create
    if (modelLoader && renderer) {
        std::cout << "Loading Bistro model with Vulkan resources on main thread..." << std::endl;
        Model* bistroModel = modelLoader->LoadGLTF("../Assets/Bistro.glb");
        if (bistroModel) {
            std::cout << "Successfully loaded Bistro model with Vulkan resources on main thread" << std::endl;
        } else {
            std::cerr << "Warning: Failed to load Bistro model with Vulkan resources on main thread" << std::endl;
        }
    }

    int entitiesCreated = 0;
    for (const auto& materialMesh : g_loadingState.loadedMaterials) {
        // Create entity name based on material
        std::string entityName = "Bistro_Material_" + std::to_string(materialMesh.materialIndex) +
                                "_" + materialMesh.materialName;

        if (Entity* materialEntity = engine->CreateEntity(entityName)) {
            // Add transform component
            auto* transform = materialEntity->AddComponent<TransformComponent>();
            transform->SetPosition(glm::vec3(0.0f, -1.5f, 0.0f));
            transform->SetRotation(glm::vec3(0.0f, 0.0f, 0.0f));
            transform->SetScale(glm::vec3(0.1f, 0.1f, 0.1f));

            // Add mesh component with material-specific data
            auto* mesh = materialEntity->AddComponent<MeshComponent>();
            mesh->SetVertices(materialMesh.vertices);
            mesh->SetIndices(materialMesh.indices);

            // Set the correct texture for this material
            // The texture should now be loaded since we called LoadGLTF on the main thread
            if (!materialMesh.texturePath.empty()) {
                mesh->SetTexturePath(materialMesh.texturePath);
                std::cout << "  Entity " << entityName << ": " << materialMesh.vertices.size()
                          << " vertices, texture: " << materialMesh.texturePath << std::endl;
            } else {
                std::cout << "  Entity " << entityName << ": " << materialMesh.vertices.size()
                          << " vertices, no texture" << std::endl;
            }

            entitiesCreated++;
            g_loadingState.materialsLoaded = entitiesCreated;
        } else {
            std::cerr << "Failed to create entity for material " << materialMesh.materialName << std::endl;
        }
    }

    std::cout << "Successfully created " << entitiesCreated << " entities from loaded materials" << std::endl;

    // Clear the loaded materials to free memory
    g_loadingState.loadedMaterials.clear();
}
