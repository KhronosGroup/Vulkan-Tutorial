#include "engine.h"

#include <chrono>
#include <algorithm>
#include <stdexcept>

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

    // Initialize audio system
    if (!audioSystem->Initialize()) {
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
    // Update physics system
    physicsSystem->Update(deltaTime);

    // Update audio system
    audioSystem->Update(deltaTime);

    // Update ImGui system
    imguiSystem->NewFrame();

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
    auto currentTime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count()) / 1000.0f;

    // Calculate delta time
    float delta = currentTime - lastFrameTime;

    // Update last frame time
    lastFrameTime = currentTime;

    return delta;
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

    // Initialize audio system
    if (!audioSystem->Initialize()) {
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
