#pragma once

#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include "model_loader.h"

// Forward declarations
class Engine;
class ModelLoader;

// Structure to track threaded loading state
struct LoadingState {
    std::atomic<bool> isLoading{false};
    std::atomic<bool> loadingComplete{false};
    std::atomic<bool> loadingFailed{false};
    std::atomic<int> materialsLoaded{0};
    std::atomic<int> totalMaterials{0};
    std::mutex entityCreationMutex;
    std::vector<MaterialMesh> loadedMaterials;
    std::string errorMessage;
};

// Global loading state
extern LoadingState g_loadingState;

/**
 * @brief Background thread function to load the Bistro model.
 * @param modelLoader Pointer to the model loader.
 */
void LoadBistroModelAsync(ModelLoader* modelLoader);

/**
 * @brief Create entities from loaded materials (called from main thread).
 * @param engine The engine to create entities in.
 */
void CreateEntitiesFromLoadedMaterials(Engine* engine);
