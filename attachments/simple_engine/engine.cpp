#include "engine.h"
#include "scene_loading.h"
#include "mesh_component.h"

#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <random>

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
#if defined(PLATFORM_ANDROID)
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
            // Handle mouse click for ball throwing (right mouse button)
            if (buttons & 2) { // Right mouse button (bit 1)
                if (!cameraControl.mouseRightPressed) {
                    cameraControl.mouseRightPressed = true;
                    // Throw a ball on mouse click
                    ThrowBall(x, y);
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

    // Enable GPU acceleration for physics calculations to drastically speed up computations
    physicsSystem->SetGPUAccelerationEnabled(true);

    if (!physicsSystem->Initialize()) {
        return false;
    }

    // Initialize ImGui system
    if (!imguiSystem->Initialize(renderer.get(), width, height)) {
        return false;
    }

    // Connect ImGui system to an audio system for UI controls
    imguiSystem->SetAudioSystem(audioSystem.get());

    // Generate ball material properties once at load time
    GenerateBallMaterial();

    // Initialize physics scaling system
    InitializePhysicsScaling();


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
    int loopCount = 0;
    while (running) {
        loopCount++;
        // Process platform events
        if (!platform->ProcessEvents()) {
            running = false;
            break;
        }

        // Calculate delta time
        deltaTime = CalculateDeltaTime();

        // Update frame counter and FPS
        frameCount++;
        fpsUpdateTimer += deltaTime;

        // Update window title with FPS and frame time every second
        if (fpsUpdateTimer >= 1.0f) {
            uint64_t framesSinceLastUpdate = frameCount - lastFPSUpdateFrame;
            currentFPS = framesSinceLastUpdate / fpsUpdateTimer;
            // Average frame time in milliseconds over the last interval
            double avgMs = (fpsUpdateTimer / static_cast<double>(framesSinceLastUpdate)) * 1000.0;

            // Update window title with frame count, FPS, and frame time
            std::string title = "Simple Engine - Frame: " + std::to_string(frameCount) +
                               " | FPS: " + std::to_string(static_cast<int>(currentFPS)) +
                               " | ms: " + std::to_string(static_cast<int>(avgMs));
            platform->SetWindowTitle(title);

            // Reset timer and frame counter for next update
            fpsUpdateTimer = 0.0f;
            lastFPSUpdateFrame = frameCount;
        }

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
    // Debug: Verify Update method is being called
    static int updateCallCount = 0;
    updateCallCount++;
    // Process pending ball creations (outside rendering loop to avoid memory pool constraints)
    ProcessPendingBalls();


    if (activeCamera) {
        glm::vec3 currentCameraPosition = activeCamera->GetPosition();
        physicsSystem->SetCameraPosition(currentCameraPosition);
    }

    // Use real deltaTime for physics to maintain proper timing
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
    // Get current time using a steady clock to avoid system time jumps
    uint64_t currentTime = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );

    // Initialize lastFrameTime on first call
    if (lastFrameTime == 0) {
        lastFrameTime = currentTime;
        return 0.016f; // ~16ms as a sane initial guess
    }

    // Calculate delta time in milliseconds
    uint64_t delta = currentTime - lastFrameTime;

    // Update last frame time
    lastFrameTime = currentTime;

    return static_cast<float>(delta) / 1000.0f;
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

    // Check if camera tracking is enabled
    if (imguiSystem && imguiSystem->IsCameraTrackingEnabled()) {
        // Find the first active ball entity
        Entity* ballEntity = nullptr;
        for (const auto& entity : entities) {
            if (entity->IsActive() && entity->GetName().find("Ball_") != std::string::npos) {
                ballEntity = entity.get();
                break;
            }
        }

        if (ballEntity) {
            // Get ball's transform component
            auto* ballTransform = ballEntity->GetComponent<TransformComponent>();
            if (ballTransform) {
                glm::vec3 ballPosition = ballTransform->GetPosition();

                // Position camera at a fixed offset from the ball for good viewing
                glm::vec3 cameraOffset = glm::vec3(2.0f, 1.5f, 2.0f); // Behind and above the ball
                glm::vec3 cameraPosition = ballPosition + cameraOffset;

                // Update camera position and target
                cameraTransform->SetPosition(cameraPosition);
                activeCamera->SetTarget(ballPosition);

                return; // Skip manual controls when tracking
            }
        }
    }

    // Manual camera controls (only when tracking is disabled)
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

void Engine::GenerateBallMaterial() {
    // Generate 8 random material properties for PBR
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Generate bright, vibrant albedo colors for better visibility
    std::uniform_real_distribution<float> brightDis(0.6f, 1.0f); // Ensure bright colors
    ballMaterial.albedo = glm::vec3(brightDis(gen), brightDis(gen), brightDis(gen));

    // Random metallic value (0.0 to 1.0)
    ballMaterial.metallic = dis(gen);

    // Random roughness value (0.0 to 1.0)
    ballMaterial.roughness = dis(gen);

    // Random ambient occlusion (typically 0.8 to 1.0 for good lighting)
    ballMaterial.ao = 0.8f + dis(gen) * 0.2f;

    // Random emissive color (usually subtle)
    ballMaterial.emissive = glm::vec3(dis(gen) * 0.3f, dis(gen) * 0.3f, dis(gen) * 0.3f);

    // Decent bounciness (0.6 to 0.9) so bounces are clearly visible
    ballMaterial.bounciness = 0.6f + dis(gen) * 0.3f;
}

void Engine::InitializePhysicsScaling() {
    // Based on issue analysis: balls reaching 120+ m/s and extreme positions like (-244, -360, -244)
    // The previous 200.0f force scale was causing supersonic speeds and balls flying out of scene
    // Need much more conservative scaling for realistic visual gameplay

    // Use smaller game unit scale for more controlled physics
    physicsScaling.gameUnitsToMeters = 0.1f;  // 1 game unit = 0.1 meter (10cm) - smaller scale

    // Much reduced force scaling to prevent extreme speeds
    // With base forces 0.01f-0.05f, this gives final forces of 0.001f-0.005f
    physicsScaling.forceScale = 1.0f;         // Minimal force scaling for realistic movement
    physicsScaling.physicsTimeScale = 1.0f;   // Keep time scale normal
    physicsScaling.gravityScale = 1.0f;       // Keep gravity proportional to scale

    // Apply scaled gravity to physics system
    glm::vec3 realWorldGravity(0.0f, -9.81f, 0.0f);
    glm::vec3 scaledGravity = ScaleGravityForPhysics(realWorldGravity);
    physicsSystem->SetGravity(scaledGravity);
}


float Engine::ScaleForceForPhysics(float gameForce) const {
    // Scale force based on the relationship between game units and real world
    // and the force scaling factor to make physics feel right
    return gameForce * physicsScaling.forceScale * physicsScaling.gameUnitsToMeters;
}

glm::vec3 Engine::ScaleGravityForPhysics(const glm::vec3& realWorldGravity) const {
    // Scale gravity based on game unit scale and gravity scaling factor
    // If 1 game unit = 1 meter, then gravity should remain -9.81
    // If 1 game unit = 0.1 meter, then gravity should be -0.981
    return realWorldGravity * physicsScaling.gravityScale * physicsScaling.gameUnitsToMeters;
}

float Engine::ScaleTimeForPhysics(float deltaTime) const {
    // Scale time for physics simulation if needed
    // This can be used to slow down or speed up physics relative to rendering
    return deltaTime * physicsScaling.physicsTimeScale;
}

void Engine::ThrowBall(float mouseX, float mouseY) {
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

    // Convert NDC to world space for direction
    glm::vec4 rayClip = glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 rayEye = invProj * rayClip;
    rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
    glm::vec4 rayWorld = invView * rayEye;

    // Calculate screen center in world coordinates
    // Screen center is at NDC (0, 0) which corresponds to the center of the view
    glm::vec4 screenCenterClip = glm::vec4(0.0f, 0.0f, -1.0f, 1.0f);
    glm::vec4 screenCenterEye = invProj * screenCenterClip;
    screenCenterEye = glm::vec4(screenCenterEye.x, screenCenterEye.y, -1.0f, 0.0f);
    glm::vec4 screenCenterWorld = invView * screenCenterEye;
    glm::vec3 screenCenterDirection = glm::normalize(glm::vec3(screenCenterWorld));

    // Calculate world position for screen center at a reasonable distance from camera
    glm::vec3 cameraPosition = activeCamera->GetPosition();
    glm::vec3 screenCenterWorldPos = cameraPosition + screenCenterDirection * 2.0f; // 2 units in front of camera

    // Calculate throw direction from screen center toward mouse position
    glm::vec3 throwDirection = glm::normalize(glm::vec3(rayWorld));

    // Add upward component for realistic arc trajectory
    throwDirection.y += 0.3f; // Add upward bias for throwing arc
    throwDirection = glm::normalize(throwDirection); // Re-normalize after modification

    // Generate ball properties now
    static int ballCounter = 0;
    std::string ballName = "Ball_" + std::to_string(ballCounter++);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Launch balls from screen center toward mouse cursor
    glm::vec3 spawnPosition = screenCenterWorldPos;

    // Add small random variation to avoid identical paths
    std::uniform_real_distribution<float> posDis(-0.1f, 0.1f);
    spawnPosition.x += posDis(gen);
    spawnPosition.y += posDis(gen);
    spawnPosition.z += posDis(gen);

    std::uniform_real_distribution<float> spinDis(-10.0f, 10.0f);
    std::uniform_real_distribution<float> forceDis(15.0f, 35.0f); // Stronger force range for proper throwing feel

    // Store ball creation data for processing outside rendering loop
    PendingBall pendingBall;
    pendingBall.spawnPosition = spawnPosition;
    pendingBall.throwDirection = throwDirection; // This is now the corrected direction toward geometry
    pendingBall.throwForce = ScaleForceForPhysics(forceDis(gen)); // Apply physics scaling to force
    pendingBall.randomSpin = glm::vec3(spinDis(gen), spinDis(gen), spinDis(gen));
    pendingBall.ballName = ballName;

    pendingBalls.push_back(pendingBall);
}

void Engine::ProcessPendingBalls() {
    if (pendingBalls.empty()) {
        return;
    }

    // Process all pending balls
    for (const auto& pendingBall : pendingBalls) {
        // Create ball entity
        Entity* ballEntity = CreateEntity(pendingBall.ballName);
        if (!ballEntity) {
            std::cerr << "Failed to create ball entity: " << pendingBall.ballName << std::endl;
            continue;
        }

        // Add transform component
        auto* transform = ballEntity->AddComponent<TransformComponent>();
        if (!transform) {
            std::cerr << "Failed to add TransformComponent to ball: " << pendingBall.ballName << std::endl;
            continue;
        }
        transform->SetPosition(pendingBall.spawnPosition);
        transform->SetScale(glm::vec3(1.0f)); // Tennis ball size scale

        // Add mesh component with sphere geometry
        auto* mesh = ballEntity->AddComponent<MeshComponent>();
        if (!mesh) {
            std::cerr << "Failed to add MeshComponent to ball: " << pendingBall.ballName << std::endl;
            continue;
        }
        // Create tennis ball-sized, bright red sphere
        glm::vec3 brightRed(1.0f, 0.0f, 0.0f);
        mesh->CreateSphere(0.0335f, brightRed, 32); // Tennis ball radius, bright color, high detail
        mesh->SetTexturePath(renderer->SHARED_BRIGHT_RED_ID); // Use bright red texture for visibility

        // Verify mesh geometry was created
        const auto& vertices = mesh->GetVertices();
        const auto& indices = mesh->GetIndices();
        if (vertices.empty() || indices.empty()) {
            std::cerr << "ERROR: CreateSphere failed to generate geometry!" << std::endl;
            continue;
        }

        // Pre-allocate Vulkan resources for this entity (now outside rendering loop)
        if (!renderer->preAllocateEntityResources(ballEntity)) {
            std::cerr << "Failed to pre-allocate resources for ball: " << pendingBall.ballName << std::endl;
            continue;
        }

        // Create rigid body with sphere collision shape
        RigidBody* rigidBody = physicsSystem->CreateRigidBody(ballEntity, CollisionShape::Sphere, 1.0f);
        if (rigidBody) {
            // Set bounciness from material
            rigidBody->SetRestitution(ballMaterial.bounciness);

            // Apply throw force and spin
            glm::vec3 throwImpulse = pendingBall.throwDirection * pendingBall.throwForce;
            rigidBody->ApplyImpulse(throwImpulse, glm::vec3(0.0f));
            rigidBody->SetAngularVelocity(pendingBall.randomSpin);
        }
    }

    // Clear processed balls
    pendingBalls.clear();
}

void Engine::HandleMouseHover(float mouseX, float mouseY) {
    // Update current mouse position for any systems that might need it
    currentMouseX = mouseX;
    currentMouseY = mouseY;
}


#if defined(PLATFORM_ANDROID)
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
        // Check if ImGui wants to capture mouse input first
        bool imguiWantsMouse = imguiSystem && imguiSystem->WantCaptureMouse();

        if (!imguiWantsMouse) {
            // Handle mouse click for ball throwing (right mouse button)
            if (buttons & 2) { // Right mouse button (bit 1)
                if (!cameraControl.mouseRightPressed) {
                    cameraControl.mouseRightPressed = true;
                    // Throw a ball on mouse click
                    ThrowBall(x, y);
                }
            } else {
                cameraControl.mouseRightPressed = false;
            }
        }

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

    // Enable GPU acceleration for physics calculations to drastically speed up computations
    physicsSystem->SetGPUAccelerationEnabled(true);

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

    // Generate ball material properties once at load time
    GenerateBallMaterial();

    // Initialize physics scaling system
    InitializePhysicsScaling();

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
