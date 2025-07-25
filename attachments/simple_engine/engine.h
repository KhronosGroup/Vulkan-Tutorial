#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "platform.h"
#include "renderer.h"
#include "resource_manager.h"
#include "entity.h"
#include "camera_component.h"
#include "model_loader.h"
#include "audio_system.h"
#include "physics_system.h"
#include "imgui_system.h"

/**
 * @brief Main engine class that manages the game loop and subsystems.
 *
 * This class implements the core engine architecture as described in the Engine_Architecture chapter:
 * @see en/Building_a_Simple_Engine/Engine_Architecture/02_architectural_patterns.adoc
 */
class Engine {
public:
    /**
     * @brief Default constructor.
     */
    Engine();

    /**
     * @brief Destructor for proper cleanup.
     */
    ~Engine();

    /**
     * @brief Initialize the engine.
     * @param appName The name of the application.
     * @param width The width of the window.
     * @param height The height of the window.
     * @param enableValidationLayers Whether to enable Vulkan validation layers.
     * @return True if initialization was successful, false otherwise.
     */
    bool Initialize(const std::string& appName, int width, int height, bool enableValidationLayers = true);

    /**
     * @brief Run the main game loop.
     */
    void Run();

    /**
     * @brief Clean up engine resources.
     */
    void Cleanup();

    /**
     * @brief Create a new entity.
     * @param name The name of the entity.
     * @return A pointer to the newly created entity.
     */
    Entity* CreateEntity(const std::string& name);

    /**
     * @brief Get an entity by name.
     * @param name The name of the entity.
     * @return A pointer to the entity, or nullptr if not found.
     */
    Entity* GetEntity(const std::string& name);

    /**
     * @brief Remove an entity.
     * @param entity The entity to remove.
     * @return True if the entity was removed, false otherwise.
     */
    bool RemoveEntity(Entity* entity);

    /**
     * @brief Remove an entity by name.
     * @param name The name of the entity to remove.
     * @return True if the entity was removed, false otherwise.
     */
    bool RemoveEntity(const std::string& name);

    /**
     * @brief Get all entities.
     * @return A vector of pointers to all entities.
     */
    std::vector<Entity*> GetAllEntities() const;

    /**
     * @brief Set the active camera.
     * @param cameraComponent The camera component to set as active.
     */
    void SetActiveCamera(CameraComponent* cameraComponent);

    /**
     * @brief Get the active camera.
     * @return A pointer to the active camera component, or nullptr if none is set.
     */
    CameraComponent* GetActiveCamera() const;

    /**
     * @brief Get the resource manager.
     * @return A pointer to the resource manager.
     */
    ResourceManager* GetResourceManager() const;

    /**
     * @brief Get the platform.
     * @return A pointer to the platform.
     */
    Platform* GetPlatform() const;

    /**
     * @brief Get the renderer.
     * @return A pointer to the renderer.
     */
    Renderer* GetRenderer() const;

    /**
     * @brief Get the model loader.
     * @return A pointer to the model loader.
     */
    ModelLoader* GetModelLoader() const;

    /**
     * @brief Get the audio system.
     * @return A pointer to the audio system.
     */
    AudioSystem* GetAudioSystem() const;

    /**
     * @brief Get the physics system.
     * @return A pointer to the physics system.
     */
    PhysicsSystem* GetPhysicsSystem() const;

    /**
     * @brief Get the ImGui system.
     * @return A pointer to the ImGui system.
     */
    ImGuiSystem* GetImGuiSystem() const;

#if PLATFORM_ANDROID
    /**
     * @brief Initialize the engine for Android.
     * @param app The Android app.
     * @param appName The name of the application.
     * @param enableValidationLayers Whether to enable Vulkan validation layers.
     * @return True if initialization was successful, false otherwise.
     */
    bool InitializeAndroid(android_app* app, const std::string& appName, bool enableValidationLayers = true);

    /**
     * @brief Run the engine on Android.
     */
    void RunAndroid();
#endif

private:
    // Subsystems
    std::unique_ptr<Platform> platform;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<ResourceManager> resourceManager;
    std::unique_ptr<ModelLoader> modelLoader;
    std::unique_ptr<AudioSystem> audioSystem;
    std::unique_ptr<PhysicsSystem> physicsSystem;
    std::unique_ptr<ImGuiSystem> imguiSystem;

    // Entities
    std::vector<std::unique_ptr<Entity>> entities;
    std::unordered_map<std::string, Entity*> entityMap;

    // Active camera
    CameraComponent* activeCamera = nullptr;

    // Engine state
    bool initialized = false;
    bool running = false;

    // Delta time calculation
    float deltaTime = 0.0f;
    uint64_t lastFrameTime = 0;

    // Camera control state
    struct CameraControlState {
        bool moveForward = false;
        bool moveBackward = false;
        bool moveLeft = false;
        bool moveRight = false;
        bool moveUp = false;
        bool moveDown = false;
        bool mouseLeftPressed = false;
        float lastMouseX = 0.0f;
        float lastMouseY = 0.0f;
        float yaw = 0.0f;   // Horizontal rotation
        float pitch = 0.0f; // Vertical rotation
        bool firstMouse = true;
        float cameraSpeed = 5.0f;
        float mouseSensitivity = 0.1f;
    } cameraControl;

    /**
     * @brief Update the engine state.
     * @param deltaTime The time elapsed since the last update.
     */
    void Update(float deltaTime);

    /**
     * @brief Render the scene.
     */
    void Render();

    /**
     * @brief Calculate the delta time between frames.
     * @return The delta time in seconds.
     */
    float CalculateDeltaTime();

    /**
     * @brief Handle window resize events.
     * @param width The new width of the window.
     * @param height The new height of the window.
     */
    void HandleResize(int width, int height);

    /**
     * @brief Update camera controls based on input state.
     * @param deltaTime The time elapsed since the last update.
     */
    void UpdateCameraControls(float deltaTime);

    /**
     * @brief Check for completed background loading and create entities if ready.
     */
    void CheckAndCreateLoadedEntities();
};
