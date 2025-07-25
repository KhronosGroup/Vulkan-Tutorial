#include "engine.h"
#include "scene_loading.h"

#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>

// This implementation corresponds to the Engine_Architecture chapter in the tutorial:
// @see en/Building_a_Simple_Engine/Engine_Architecture/02_architectural_patterns.adoc

Engine::Engine()
    : resourceManager(std::make_unique<ResourceManager>()),
      modelLoader(std::make_unique<ModelLoader>()),
      audioSystem(std::make_unique<AudioSystem>()),
      physicsSystem(std::make_unique<PhysicsSystem>()),
      imguiSystem(std::make_unique<ImGuiSystem>()) {
}

Engine::~Engine() {
    Cleanup();
}

bool Engine::Initialize(const std::string& appName, int width, int height, bool enableValidationLayers) {
    // Create platform
#if PLATFORM_ANDROID
    // For Android, the platform is created with the android_app
    // This will be handled in the android_main function
    return false;
#else
    platform = CreatePlatform();
    if (!platform->Initialize(appName, width, height)) {
        return false;
    }

    // Set resize callback
    platform->SetResizeCallback([this](int width, int height) {
        HandleResize(width, height);
    });

    // Set mouse callback
    platform->SetMouseCallback([this](float x, float y, uint32_t buttons) {
        // Handle camera rotation when left mouse button is pressed
        if (buttons & 1) { // Left mouse button (bit 0)
            if (!cameraControl.mouseLeftPressed) {
                cameraControl.mouseLeftPressed = true;
                cameraControl.firstMouse = true;
            }

            if (cameraControl.firstMouse) {
                cameraControl.lastMouseX = x;
                cameraControl.lastMouseY = y;
                cameraControl.firstMouse = false;
            }

            float xOffset = x - cameraControl.lastMouseX;
            float yOffset = cameraControl.lastMouseY - y; // Reversed since y-coordinates go from bottom to top
            cameraControl.lastMouseX = x;
            cameraControl.lastMouseY = y;

            xOffset *= cameraControl.mouseSensitivity;
            yOffset *= cameraControl.mouseSensitivity;

            cameraControl.yaw += xOffset;
            cameraControl.pitch += yOffset;

            // Constrain pitch to avoid gimbal lock
            if (cameraControl.pitch > 89.0f) cameraControl.pitch = 89.0f;
            if (cameraControl.pitch < -89.0f) cameraControl.pitch = -89.0f;
        } else {
            cameraControl.mouseLeftPressed = false;
        }

        if (imguiSystem) {
            imguiSystem->HandleMouse(x, y, buttons);
        }
    });

    // Set keyboard callback
    platform->SetKeyboardCallback([this](uint32_t key, bool pressed) {
        // Handle camera movement keys (WASD + Arrow keys)
        switch (key) {
            case GLFW_KEY_W:
            case GLFW_KEY_UP:
                cameraControl.moveForward = pressed;
                break;
            case GLFW_KEY_S:
            case GLFW_KEY_DOWN:
                cameraControl.moveBackward = pressed;
                break;
            case GLFW_KEY_A:
            case GLFW_KEY_LEFT:
                cameraControl.moveLeft = pressed;
                break;
            case GLFW_KEY_D:
            case GLFW_KEY_RIGHT:
                cameraControl.moveRight = pressed;
                break;
            case GLFW_KEY_Q:
            case GLFW_KEY_PAGE_UP:
                cameraControl.moveUp = pressed;
                break;
            case GLFW_KEY_E:
            case GLFW_KEY_PAGE_DOWN:
                cameraControl.moveDown = pressed;
                break;
        }

        if (imguiSystem) {
            imguiSystem->HandleKeyboard(key, pressed);
        }
    });

    // Set char callback
    platform->SetCharCallback([this](uint32_t c) {
        if (imguiSystem) {
            imguiSystem->HandleChar(c);
        }
    });

    // Create renderer
    renderer = std::make_unique<Renderer>(platform.get());
    if (!renderer->Initialize(appName, enableValidationLayers)) {
        return false;
    }

    // Initialize model loader
    if (!modelLoader->Initialize(renderer.get())) {
        return false;
    }

    // Connect model loader to renderer for light extraction
    renderer->SetModelLoader(modelLoader.get());

    // Initialize audio system
    if (!audioSystem->Initialize(this, renderer.get())) {
        return false;
    }

    // Initialize physics system
    physicsSystem->SetRenderer(renderer.get());
    if (!physicsSystem->Initialize()) {
        return false;
    }

    // Initialize ImGui system
    if (!imguiSystem->Initialize(renderer.get(), width, height)) {
        return false;
    }

    // Connect ImGui system to audio system for UI controls
    imguiSystem->SetAudioSystem(audioSystem.get());

    initialized = true;
    return true;
#endif
}

void Engine::Run() {
    if (!initialized) {
        throw std::runtime_error("Engine not initialized");
    }

    running = true;

    // Main loop
    while (running) {
        // Process platform events
        if (!platform->ProcessEvents()) {
            running = false;
            break;
        }

        // Calculate delta time
        deltaTime = CalculateDeltaTime();

        // Update
        Update(deltaTime);

        // Render
        Render();
    }
}

void Engine::Cleanup() {
    if (initialized) {
        // Wait for the device to be idle before cleaning up
        if (renderer) {
            renderer->WaitIdle();
        }

        // Clear entities
        entities.clear();
        entityMap.clear();

        // Clean up subsystems in reverse order of creation
        imguiSystem.reset();
        physicsSystem.reset();
        audioSystem.reset();
        modelLoader.reset();
        renderer.reset();
        platform.reset();

        initialized = false;
    }
}

Entity* Engine::CreateEntity(const std::string& name) {
    // Check if an entity with this name already exists
    if (entityMap.find(name) != entityMap.end()) {
        return nullptr;
    }

    // Create the entity
    auto entity = std::make_unique<Entity>(name);
    Entity* entityPtr = entity.get();

    // Add to the map and vector
    entityMap[name] = entityPtr;
    entities.push_back(std::move(entity));

    return entityPtr;
}

Entity* Engine::GetEntity(const std::string& name) {
    auto it = entityMap.find(name);
    if (it != entityMap.end()) {
        return it->second;
    }
    return nullptr;
}

bool Engine::RemoveEntity(Entity* entity) {
    if (!entity) {
        return false;
    }

    // Find the entity in the vector
    auto it = std::find_if(entities.begin(), entities.end(),
        [entity](const std::unique_ptr<Entity>& e) {
            return e.get() == entity;
        });

    if (it != entities.end()) {
        // Remove from the map
        entityMap.erase(entity->GetName());

        // Remove from the vector
        entities.erase(it);

        return true;
    }

    return false;
}

bool Engine::RemoveEntity(const std::string& name) {
    Entity* entity = GetEntity(name);
    if (entity) {
        return RemoveEntity(entity);
    }
    return false;
}

std::vector<Entity*> Engine::GetAllEntities() const {
    std::vector<Entity*> result;
    result.reserve(entities.size());

    for (const auto& entity : entities) {
        result.push_back(entity.get());
    }

    return result;
}

void Engine::SetActiveCamera(CameraComponent* cameraComponent) {
    activeCamera = cameraComponent;
}

CameraComponent* Engine::GetActiveCamera() const {
    return activeCamera;
}

ResourceManager* Engine::GetResourceManager() const {
    return resourceManager.get();
}

Platform* Engine::GetPlatform() const {
    return platform.get();
}

Renderer* Engine::GetRenderer() const {
    return renderer.get();
}

ModelLoader* Engine::GetModelLoader() const {
    return modelLoader.get();
}

AudioSystem* Engine::GetAudioSystem() const {
    return audioSystem.get();
}

PhysicsSystem* Engine::GetPhysicsSystem() const {
    return physicsSystem.get();
}

ImGuiSystem* Engine::GetImGuiSystem() const {
    return imguiSystem.get();
}

void Engine::Update(float deltaTime) {
    // Check for completed background loading and create entities if ready
    CheckAndCreateLoadedEntities();

    // Update physics system
    physicsSystem->Update(deltaTime);

    // Update audio system
    audioSystem->Update(deltaTime);

    // Update ImGui system
    imguiSystem->NewFrame();

    // Update camera controls
    if (activeCamera) {
        UpdateCameraControls(deltaTime);
    }

    // Update all entities
    for (auto& entity : entities) {
        if (entity->IsActive()) {
            entity->Update(deltaTime);
        }
    }
}

void Engine::Render() {

    // Check if we have an active camera
    if (!activeCamera) {
        return;
    }

    // Get all active entities
    std::vector<Entity*> activeEntities;
    for (auto& entity : entities) {
        if (entity->IsActive()) {
            activeEntities.push_back(entity.get());
        }
    }

    // Render the scene (ImGui will be rendered within the render pass)
    renderer->Render(activeEntities, activeCamera, imguiSystem.get());
}

float Engine::CalculateDeltaTime() {
    // Get current time
    auto currentTime = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count());

    // Initialize lastFrameTime on first call
    if (lastFrameTime == 0) {
        lastFrameTime = currentTime;
        return 0.016f; // Return ~16ms (60 FPS) for first frame
    }

    // Calculate delta time
    uint64_t delta = currentTime - lastFrameTime;

    // Update last frame time
    lastFrameTime = currentTime;

    // Clamp delta time to reasonable values (prevent huge jumps)
    if (delta > 10) { // Cap at 100ms (10 FPS minimum)
        delta = 16; // Use 16ms instead
    }

    return delta / 1000.0f; // Convert to seconds
}

void Engine::HandleResize(int width, int height) {
    // Update the active camera's aspect ratio
    if (activeCamera) {
        activeCamera->SetAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    }

    // Notify the renderer that the framebuffer has been resized
    if (renderer) {
        renderer->SetFramebufferResized();
    }

    // Notify ImGui system about the resize
    if (imguiSystem) {
        imguiSystem->HandleResize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    }
}

void Engine::UpdateCameraControls(float deltaTime) {
    if (!activeCamera) return;

    // Get camera transform component
    TransformComponent* cameraTransform = activeCamera->GetOwner()->GetComponent<TransformComponent>();
    if (!cameraTransform) return;

    // Calculate movement speed
    float velocity = cameraControl.cameraSpeed * deltaTime;

    // Calculate camera direction vectors based on yaw and pitch
    glm::vec3 front;
    front.x = cos(glm::radians(cameraControl.yaw)) * cos(glm::radians(cameraControl.pitch));
    front.y = sin(glm::radians(cameraControl.pitch));
    front.z = sin(glm::radians(cameraControl.yaw)) * cos(glm::radians(cameraControl.pitch));
    front = glm::normalize(front);

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(front, up));
    up = glm::normalize(glm::cross(right, front));

    // Get current camera position
    glm::vec3 position = cameraTransform->GetPosition();

    // Apply movement based on input
    if (cameraControl.moveForward) {
        position += front * velocity;
    }
    if (cameraControl.moveBackward) {
        position -= front * velocity;
    }
    if (cameraControl.moveLeft) {
        position -= right * velocity;
    }
    if (cameraControl.moveRight) {
        position += right * velocity;
    }
    if (cameraControl.moveUp) {
        position += up * velocity;
    }
    if (cameraControl.moveDown) {
        position -= up * velocity;
    }

    // Update camera position
    cameraTransform->SetPosition(position);

    // Update camera target based on direction
    glm::vec3 target = position + front;
    activeCamera->SetTarget(target);
}

void Engine::CheckAndCreateLoadedEntities() {
    // Check if background loading is complete
    if (g_loadingState.loadingComplete && !g_loadingState.loadedMaterials.empty()) {
        // Create entities from loaded materials on the main thread
        CreateEntitiesFromLoadedMaterials(this);

        // Reset the loading complete flag
        g_loadingState.loadingComplete = false;
    }

    // Check for loading errors
    if (g_loadingState.loadingFailed) {
        std::cerr << "Background loading failed: " << g_loadingState.errorMessage << std::endl;
        g_loadingState.loadingFailed = false; // Reset the flag
    }
}

#if PLATFORM_ANDROID
// Android-specific implementation
bool Engine::InitializeAndroid(android_app* app, const std::string& appName, bool enableValidationLayers) {
    // Create platform
    platform = CreatePlatform(app);
    if (!platform->Initialize(appName, 0, 0)) {
        return false;
    }

    // Set resize callback
    platform->SetResizeCallback([this](int width, int height) {
        HandleResize(width, height);
    });

    // Set mouse callback
    platform->SetMouseCallback([this](float x, float y, uint32_t buttons) {
        if (imguiSystem) {
            imguiSystem->HandleMouse(x, y, buttons);
        }
    });

    // Set keyboard callback
    platform->SetKeyboardCallback([this](uint32_t key, bool pressed) {
        if (imguiSystem) {
            imguiSystem->HandleKeyboard(key, pressed);
        }
    });

    // Set char callback
    platform->SetCharCallback([this](uint32_t c) {
        if (imguiSystem) {
            imguiSystem->HandleChar(c);
        }
    });

    // Create renderer
    renderer = std::make_unique<Renderer>(platform.get());
    if (!renderer->Initialize(appName, enableValidationLayers)) {
        return false;
    }

    // Initialize model loader
    if (!modelLoader->Initialize(renderer.get())) {
        return false;
    }

    // Connect model loader to renderer for light extraction
    renderer->SetModelLoader(modelLoader.get());

    // Initialize audio system
    if (!audioSystem->Initialize(this, renderer.get())) {
        return false;
    }

    // Initialize physics system
    physicsSystem->SetRenderer(renderer.get());
    if (!physicsSystem->Initialize()) {
        return false;
    }

    // Get window dimensions from platform
    int width, height;
    platform->GetWindowSize(&width, &height);

    // Initialize ImGui system
    if (!imguiSystem->Initialize(renderer.get(), width, height)) {
        return false;
    }

    // Connect ImGui system to audio system for UI controls
    imguiSystem->SetAudioSystem(audioSystem.get());

    initialized = true;
    return true;
}

void Engine::RunAndroid() {
    if (!initialized) {
        throw std::runtime_error("Engine not initialized");
    }

    running = true;

    // Main loop is handled by the platform
    // We just need to update and render when the platform is ready

    // Calculate delta time
    deltaTime = CalculateDeltaTime();

    // Update
    Update(deltaTime);

    // Render
    Render();
}
#endif
