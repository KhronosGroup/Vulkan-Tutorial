#include "engine.h"
#include "transform_component.h"
#include "mesh_component.h"
#include "camera_component.h"
#include "scene_loading.h"

#include <iostream>
#include <stdexcept>
#include <thread>

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

    // Create a cube entity
    Entity* cubeEntity = engine->CreateEntity("Cube");
    if (!cubeEntity) {
        throw std::runtime_error("Failed to create cube entity");
    }

    // Add a transform component to the cube
    auto* cubeTransform = cubeEntity->AddComponent<TransformComponent>();
    cubeTransform->SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    cubeTransform->SetRotation(glm::vec3(0.0f, 0.0f, 0.0f));
    cubeTransform->SetScale(glm::vec3(1.0f, 1.0f, 1.0f));

    // Add a mesh component to the cube
    auto* cubeMesh = cubeEntity->AddComponent<MeshComponent>();
    cubeMesh->CreateCube(1.0f, glm::vec3(1.0f, 0.0f, 0.0f));

    // Make the camera look at the red cube
    camera->LookAt(cubeTransform->GetPosition());

    // Create a second cube entity
    Entity* cube2Entity = engine->CreateEntity("Cube2");
    if (!cube2Entity) {
        throw std::runtime_error("Failed to create second cube entity");
    }

    // Add a transform component to the second cube
    auto* cube2Transform = cube2Entity->AddComponent<TransformComponent>();
    cube2Transform->SetPosition(glm::vec3(2.0f, 0.0f, 0.0f));
    cube2Transform->SetRotation(glm::vec3(0.0f, 0.0f, 0.0f));
    cube2Transform->SetScale(glm::vec3(0.5f, 0.5f, 0.5f));

    // Add a mesh component to the second cube
    auto* cube2Mesh = cube2Entity->AddComponent<MeshComponent>();
    cube2Mesh->CreateCube(1.0f, glm::vec3(0.0f, 1.0f, 0.0f));

    // Create a third cube entity
    Entity* cube3Entity = engine->CreateEntity("Cube3");
    if (!cube3Entity) {
        throw std::runtime_error("Failed to create third cube entity");
    }

    // Add a transform component to the third cube
    auto* cube3Transform = cube3Entity->AddComponent<TransformComponent>();
    cube3Transform->SetPosition(glm::vec3(-2.0f, 0.0f, 0.0f));
    cube3Transform->SetRotation(glm::vec3(0.0f, 0.0f, 0.0f));
    cube3Transform->SetScale(glm::vec3(0.5f, 0.5f, 0.5f));

    // Add a mesh component to the third cube
    auto* cube3Mesh = cube3Entity->AddComponent<MeshComponent>();
    cube3Mesh->CreateCube(1.0f, glm::vec3(0.0f, 0.0f, 1.0f));

    // Start loading Bistro.glb model in background thread
    if (ModelLoader* modelLoader = engine->GetModelLoader()) {
        std::cout << "Starting threaded loading of Bistro model..." << std::endl;
        std::thread loadingThread(LoadBistroModelAsync, modelLoader);
        loadingThread.detach(); // Let the thread run independently
        std::cout << "Background loading thread started. Application will continue running..." << std::endl;
    }
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
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments.
 * @return The exit code.
 */
int main(int argc, char* argv[]) {
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
