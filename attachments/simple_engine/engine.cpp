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
        // Check if ImGui wants to capture mouse input first
        bool imguiWantsMouse = imguiSystem && imguiSystem->WantCaptureMouse();

        if (!imguiWantsMouse) {
            // Handle mouse click for poke functionality (right mouse button)
            if (buttons & 2) { // Right mouse button (bit 1)
                if (!cameraControl.mouseRightPressed) {
                    cameraControl.mouseRightPressed = true;
                    // Perform poke on mouse click
                    HandleMousePoke(x, y);
                }
            } else {
                cameraControl.mouseRightPressed = false;
            }

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
        }

        if (imguiSystem) {
            imguiSystem->HandleMouse(x, y, buttons);
        }

        // Always perform hover detection (even when ImGui is active)
        HandleMouseHover(x, y);
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
            default: break;
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

    // Initialize a physics system
    physicsSystem->SetRenderer(renderer.get());
    if (!physicsSystem->Initialize()) {
        return false;
    }

    // Initialize ImGui system
    if (!imguiSystem->Initialize(renderer.get(), width, height)) {
        return false;
    }

    // Connect ImGui system to an audio system for UI controls
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
    if (entityMap.contains(name)) {
        return nullptr;
    }

    // Create the entity
    auto entity = std::make_unique<Entity>(name);
    // Add to the map and vector
    entities.push_back(std::move(entity));

    return entities.back().get();
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
    // Update a physics system
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

void Engine::HandleResize(int width, int height) const {
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

void Engine::UpdateCameraControls(float deltaTime) const {
    if (!activeCamera) return;

    // Get a camera transform component
    auto* cameraTransform = activeCamera->GetOwner()->GetComponent<TransformComponent>();
    if (!cameraTransform) return;

    // Calculate movement speed
    float velocity = cameraControl.cameraSpeed * deltaTime;

    // Calculate camera direction vectors based on yaw and pitch
    glm::vec3 front;
    front.x = cosf(glm::radians(cameraControl.yaw)) * cosf(glm::radians(cameraControl.pitch));
    front.y = sinf(glm::radians(cameraControl.pitch));
    front.z = sinf(glm::radians(cameraControl.yaw)) * cosf(glm::radians(cameraControl.pitch));
    front = glm::normalize(front);

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::normalize(glm::cross(front, up));
    up = glm::normalize(glm::cross(right, front));

    // Get the current camera position
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

    // Update camera target based on a direction
    glm::vec3 target = position + front;
    activeCamera->SetTarget(target);
}

void Engine::HandleMousePoke(float mouseX, float mouseY) const {
    if (!activeCamera || !physicsSystem) {
        return;
    }

    // Get window dimensions
    int windowWidth, windowHeight;
    platform->GetWindowSize(&windowWidth, &windowHeight);

    // Convert mouse coordinates to normalized device coordinates (-1 to 1)
    float ndcX = (2.0f * mouseX) / static_cast<float>(windowWidth) - 1.0f;
    float ndcY = 1.0f - (2.0f * mouseY) / static_cast<float>(windowHeight);

    // Get camera matrices
    glm::mat4 viewMatrix = activeCamera->GetViewMatrix();
    glm::mat4 projMatrix = activeCamera->GetProjectionMatrix();

    // Calculate inverse matrices
    glm::mat4 invView = glm::inverse(viewMatrix);
    glm::mat4 invProj = glm::inverse(projMatrix);

    // Convert NDC to world space
    glm::vec4 rayClip = glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 rayEye = invProj * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    glm::vec4 rayWorld = invView * rayEye;

    // Get ray origin and direction
    glm::vec3 rayOrigin = activeCamera->GetPosition();
    glm::vec3 rayDirection = glm::normalize(glm::vec3(rayWorld));

    // Perform raycast
    glm::vec3 hitPosition;
    glm::vec3 hitNormal;
    Entity* hitEntity = nullptr;

    if (physicsSystem->Raycast(rayOrigin, rayDirection, 1000.0f, &hitPosition, &hitNormal, &hitEntity)) {
        if (hitEntity) {
            std::cout << "Mouse poke hit entity: " << hitEntity->GetName() << std::endl;

            // Find or create rigid body for the entity
            RigidBody* rigidBody = nullptr;

            // Check if entity already has a rigid body (this is a simplified approach)
            // In a real implementation, you'd have a component system to track this
            rigidBody = physicsSystem->CreateRigidBody(hitEntity, CollisionShape::Box, 1.0f);

            if (rigidBody) {
                // Apply a small impulse in the direction of the ray
                glm::vec3 impulse = rayDirection * 0.5f; // Small force magnitude as requested
                rigidBody->ApplyImpulse(impulse, glm::vec3(0.0f));

                std::cout << "Applied poke impulse to " << hitEntity->GetName() << std::endl;
            }
        }
    } else {
        std::cout << "Mouse poke missed - no entity hit" << std::endl;
    }
}

void Engine::HandleMouseHover(float mouseX, float mouseY) {
    if (!activeCamera || !physicsSystem) {
        return;
    }

    // Update current mouse position
    currentMouseX = mouseX;
    currentMouseY = mouseY;

    // Get window dimensions
    int windowWidth, windowHeight;
    platform->GetWindowSize(&windowWidth, &windowHeight);

    // Convert mouse coordinates to normalized device coordinates (-1 to 1)
    float ndcX = (2.0f * mouseX) / static_cast<float>(windowWidth) - 1.0f;
    float ndcY = 1.0f - (2.0f * mouseY) / static_cast<float>(windowHeight);

    // Get camera matrices
    glm::mat4 viewMatrix = activeCamera->GetViewMatrix();
    glm::mat4 projMatrix = activeCamera->GetProjectionMatrix();

    // Calculate inverse matrices
    glm::mat4 invView = glm::inverse(viewMatrix);
    glm::mat4 invProj = glm::inverse(projMatrix);

    // Convert NDC to world space
    glm::vec4 rayClip = glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 rayEye = invProj * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    glm::vec4 rayWorld = invView * rayEye;

    // Get ray origin and direction
    glm::vec3 rayOrigin = activeCamera->GetPosition();
    glm::vec3 rayDirection = glm::normalize(glm::vec3(rayWorld));

    // Perform raycast
    glm::vec3 hitPosition;
    glm::vec3 hitNormal;
    Entity* hitEntity = nullptr;

    if (physicsSystem->Raycast(rayOrigin, rayDirection, 1000.0f, &hitPosition, &hitNormal, &hitEntity)) {
        if (hitEntity) {
            // Check if this entity is pokeable (has "_SMALL_POKEABLE" suffix)
            std::string entityName = hitEntity->GetName();

            if (entityName.find("_SMALL_POKEABLE") != std::string::npos) {
                // Update a hovered entity if it's different from the current one
                if (hoveredEntity != hitEntity) {
                    hoveredEntity = hitEntity;
                    renderer->SetHighlightedEntity(hoveredEntity);
                    std::cout << "Now hovering over pokeable entity: " << entityName << std::endl;
                }
            } else {
                // Clear hover if we're over a non-pokeable entity
                if (hoveredEntity != nullptr) {
                    std::cout << "No longer hovering over pokeable entity" << std::endl;
                    hoveredEntity = nullptr;
                    renderer->SetHighlightedEntity(nullptr);
                }
            }
        }
    } else {
        // Clear hover if no entity is hit
        if (hoveredEntity != nullptr) {
            std::cout << "No longer hovering over pokeable entity" << std::endl;
            hoveredEntity = nullptr;
            renderer->SetHighlightedEntity(nullptr);
        }
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
