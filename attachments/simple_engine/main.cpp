#include "engine.h"
#include "transform_component.h"
#include "camera_component.h"
#include "scene_loading.h"

#include <iostream>
#include <stdexcept>

// Constants
constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;
constexpr bool ENABLE_VALIDATION_LAYERS = true;


/**
 * @brief Set up a simple scene with a camera and some objects.
 * @param engine The engine to set up the scene in.
 */
void SetupScene(Engine* engine) {
    // Create a camera entity
    Entity* cameraEntity = engine->CreateEntity("Camera");
    if (!cameraEntity) {
        throw std::runtime_error("Failed to create camera entity");
    }

    // Add a transform component to the camera
    auto* cameraTransform = cameraEntity->AddComponent<TransformComponent>();
    cameraTransform->SetPosition(glm::vec3(0.0f, 0.0f, 3.0f));

    // Add a camera component to the camera entity
    auto* camera = cameraEntity->AddComponent<CameraComponent>();
    camera->SetAspectRatio(static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT));

    // Set the camera as the active camera
    engine->SetActiveCamera(camera);

    // Load GLTF model synchronously on the main thread
    std::cout << "Loading GLTF model synchronously..." << std::endl;
    LoadGLTFModel(engine, "../Assets/bistro_gltf/bistro.gltf");
    std::cout << "GLTF model loading completed." << std::endl;
}

#if PLATFORM_ANDROID
/**
 * @brief Android entry point.
 * @param app The Android app.
 */
void android_main(android_app* app) {
    try {
        // Create the engine
        Engine engine;

        // Initialize the engine
        if (!engine.InitializeAndroid(app, "Simple Engine", ENABLE_VALIDATION_LAYERS)) {
            throw std::runtime_error("Failed to initialize engine");
        }

        // Set up the scene
        SetupScene(&engine);

        // Run the engine
        engine.RunAndroid();
    } catch (const std::exception& e) {
        LOGE("Exception: %s", e.what());
    }
}
#else
/**
 * @brief Desktop entry point.
 * @return The exit code.
 */
int main(int, char*[]) {
    try {
        // Create the engine
        Engine engine;

        // Initialize the engine
        if (!engine.Initialize("Simple Engine", WINDOW_WIDTH, WINDOW_HEIGHT, ENABLE_VALIDATION_LAYERS)) {
            throw std::runtime_error("Failed to initialize engine");
        }

        // Set up the scene
        SetupScene(&engine);

        // Run the engine
        engine.Run();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
#endif
