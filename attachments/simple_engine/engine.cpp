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
#include "engine.h"
#include "mesh_component.h"
#include "scene_loading.h"
#include <cmath>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <ranges>
#include <stdexcept>

// This implementation corresponds to the Engine_Architecture chapter in the tutorial:
// @see en/Building_a_Simple_Engine/Engine_Architecture/02_architectural_patterns.adoc

Engine::Engine() : resourceManager(std::make_unique<ResourceManager>()) {
}

bool Engine::IsMainThread() const {
  return std::this_thread::get_id() == mainThreadId;
}

void Engine::ProcessPendingEntityRemovals() {
  std::vector<std::string> names; {
    std::lock_guard<std::mutex> lk(pendingEntityRemovalsMutex);
    if (pendingEntityRemovalNames.empty())
      return;
    names.swap(pendingEntityRemovalNames);
  }

  // Process on the main thread only (safety)
  if (!IsMainThread()) {
    // Put them back; we'll retry next main-thread tick
    std::lock_guard<std::mutex> lk(pendingEntityRemovalsMutex);
    pendingEntityRemovalNames.insert(pendingEntityRemovalNames.end(), names.begin(), names.end());
    return;
  }

  // Apply removals using the normal API (which takes the appropriate locks).
  for (const auto& name : names) {
    (void) RemoveEntity(name);
  }
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
  // Record main thread identity for deferring destructive operations from background threads
  mainThreadId = std::this_thread::get_id();

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
    handleMouseInput(x, y, buttons);
  });

  // Set keyboard callback
  platform->SetKeyboardCallback([this](uint32_t key, bool pressed) {
    handleKeyInput(key, pressed);
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

  try {
    // Model loader via constructor; also wire into renderer
    modelLoader = std::make_unique<ModelLoader>(renderer.get());
    renderer->SetModelLoader(modelLoader.get());

    // Audio system via constructor
    audioSystem = std::make_unique<AudioSystem>(this, renderer.get());

    // Physics system via constructor (GPU enabled)
    physicsSystem = std::make_unique<PhysicsSystem>(renderer.get(), true);

#ifdef ENABLE_COURSE_OPACITY_MICROMAPS
    // OMM integration via constructor
    ommIntegration = std::make_unique<OmmIntegration>();
    ommIntegration->init(*renderer, *modelLoader);
#endif

    // ImGui via constructor, then connect audio system
    imguiSystem = std::make_unique<ImGuiSystem>(renderer.get(), width, height);
    imguiSystem->SetAudioSystem(audioSystem.get());
  } catch (const std::exception& e) {
    std::cerr << "Subsystem initialization failed: " << e.what() << std::endl;
    return false;
  }

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
  while (running) {
    // Process platform events
    if (!platform->ProcessEvents()) {
      running = false;
      break;
    }

    // Calculate delta time
    deltaTimeMs = CalculateDeltaTimeMs();

    // Update frame counter and FPS
    frameCount++;
    fpsUpdateTimer += deltaTimeMs.count() * 0.001f;

    // Update window title with FPS and frame time every second
    if (fpsUpdateTimer >= 1.0f) {
      uint64_t framesSinceLastUpdate = frameCount - lastFPSUpdateFrame;
      double avgMs = 0.0;
      if (framesSinceLastUpdate > 0 && fpsUpdateTimer > 0.0f) {
        currentFPS = static_cast<float>(static_cast<double>(framesSinceLastUpdate) / static_cast<double>(fpsUpdateTimer));
        avgMs = (fpsUpdateTimer / static_cast<double>(framesSinceLastUpdate)) * 1000.0;
      } else {
        // Avoid divide-by-zero; keep previous FPS and estimate avgMs from last delta
        currentFPS = std::max(currentFPS, 1.0f);
        avgMs = static_cast<double>(deltaTimeMs.count());
      }

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
    Update(deltaTimeMs);

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
    {
      std::unique_lock<std::shared_mutex> lk(entitiesMutex);
      entities.clear();
      entityMap.clear();
    }

    // Clean up subsystems in reverse order of creation
    imguiSystem.reset();
#ifdef ENABLE_COURSE_OPACITY_MICROMAPS
    ommIntegration.reset();
#endif
    physicsSystem.reset();
    audioSystem.reset();
    modelLoader.reset();
    renderer.reset();
    platform.reset();

    initialized = false;
  }
}

Entity* Engine::CreateEntity(const std::string& name) {
  std::unique_lock<std::shared_mutex> lk(entitiesMutex);
  // Always allow duplicate names; map stores a representative entity
  // Create the entity
  auto entity = std::make_unique<Entity>(name);
  // Add to the vector and map
  entities.push_back(std::move(entity));
  Entity* rawPtr = entities.back().get();
  // Update the map to point to the most recently created entity with this name
  entityMap[name] = rawPtr;

  return rawPtr;
}

Entity* Engine::GetEntity(const std::string& name) {
  std::shared_lock<std::shared_mutex> lk(entitiesMutex);
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

  // If called from a background thread, defer removal to avoid deleting entities
  // while the render thread may be iterating a snapshot.
  if (!IsMainThread()) {
    std::lock_guard<std::mutex> lk(pendingEntityRemovalsMutex);
    pendingEntityRemovalNames.push_back(entity->GetName());
    return true;
  }

  std::unique_lock<std::shared_mutex> lk(entitiesMutex);

  // Remember the name before erasing ownership
  std::string name = entity->GetName();

  // Find the entity in the vector
  auto it = std::ranges::find_if(entities,
                                 [entity](const std::unique_ptr<Entity>& e) {
                                   return e.get() == entity;
                                 });

  if (it != entities.end()) {
    // Remove from the vector (ownership)
    entities.erase(it);

    // Update the map: point to another entity with the same name if one exists
    auto remainingIt = std::ranges::find_if(entities,
                                            [&name](const std::unique_ptr<Entity>& e) {
                                              return e->GetName() == name;
                                            });

    if (remainingIt != entities.end()) {
      entityMap[name] = remainingIt->get();
    } else {
      entityMap.erase(name);
    }

    return true;
  }

  return false;
}

bool Engine::RemoveEntity(const std::string& name) {
  // If called from a background thread, defer removal to avoid deleting entities
  // while the render thread may be iterating a snapshot.
  if (!IsMainThread()) {
    std::lock_guard<std::mutex> lk(pendingEntityRemovalsMutex);
    pendingEntityRemovalNames.push_back(name);
    return true;
  }

  std::unique_lock<std::shared_mutex> lk(entitiesMutex);
  auto it = entityMap.find(name);
  if (it == entityMap.end())
    return false;
  Entity* entity = it->second;
  if (!entity)
    return false;

  // Find the entity in the vector
  auto vecIt = std::ranges::find_if(entities,
                                    [entity](const std::unique_ptr<Entity>& e) {
                                      return e.get() == entity;
                                    });
  if (vecIt == entities.end()) {
    entityMap.erase(name);
    return false;
  }

  entities.erase(vecIt);

  // Update the map: point to another entity with the same name if one exists
  auto remainingIt = std::ranges::find_if(entities,
                                          [&name](const std::unique_ptr<Entity>& e) {
                                            return e && e->GetName() == name;
                                          });
  if (remainingIt != entities.end()) {
    entityMap[name] = remainingIt->get();
  } else {
    entityMap.erase(name);
  }
  return true;
}

void Engine::SetActiveCamera(CameraComponent* cameraComponent) {
  activeCamera = cameraComponent;
}

const CameraComponent* Engine::GetActiveCamera() const {
  return activeCamera;
}

const ResourceManager* Engine::GetResourceManager() const {
  return resourceManager.get();
}

const Platform* Engine::GetPlatform() const {
  return platform.get();
}

Renderer* Engine::GetRenderer() {
  return renderer.get();
}

ModelLoader* Engine::GetModelLoader() {
  return modelLoader.get();
}

const AudioSystem* Engine::GetAudioSystem() const {
  return audioSystem.get();
}

PhysicsSystem* Engine::GetPhysicsSystem() {
  return physicsSystem.get();
}

#ifdef ENABLE_COURSE_OPACITY_MICROMAPS
OmmIntegration* Engine::GetOmmIntegration() {
  return ommIntegration.get();
}
#endif

const ImGuiSystem* Engine::GetImGuiSystem() const {
  return imguiSystem.get();
}

void Engine::handleMouseInput(float x, float y, uint32_t buttons) {
  // Update ImGui system with current mouse state immediately.
  // This pushes events to the ImGui IO queue for processing in NewFrame().
  if (imguiSystem) {
    imguiSystem->HandleMouse(x, y, buttons);
  }

  // Handle LEFT button (Touch DOWN/MOVE/UP)
  if (buttons & 1) {
    if (!cameraControl.mouseLeftPressed) {
      // Finger just went down
      cameraControl.mouseLeftPressed = true;
      cameraControl.firstMouse = true;
      cameraControl.touchTotalDistance = 0.0f;
      cameraControl.touchDownX = x;
      cameraControl.touchDownY = y;
      cameraControl.touchStartTime = 0.0; // We'll increment this in Update
    }

    if (cameraControl.firstMouse) {
      cameraControl.lastMouseX = x;
      cameraControl.lastMouseY = y;
      cameraControl.firstMouse = false;
    }

    // Accumulate movement deltas. These will be applied in UpdateCameraControls
    // AFTER ImGui has updated its capture state (post-NewFrame).
    float dx = (x - cameraControl.lastMouseX);
    float dy = (y - cameraControl.lastMouseY);
    cameraControl.pendingXOffset += dx;
    cameraControl.pendingYOffset += dy;
    cameraControl.touchTotalDistance += std::sqrt(dx*dx + dy*dy);

#if defined(PLATFORM_ANDROID)
    // On Android, we map SWIPE to MOVEMENT (Forward/Backward, Left/Right)
    // if the touch didn't start on UI.
    if (!cameraControl.isFirstFrameOfInteraction && !cameraControl.startedOnImGui) {
       cameraControl.touchMoveX = dx;
       cameraControl.touchMoveY = dy;
    }
#endif

    cameraControl.lastMouseX = x;
    cameraControl.lastMouseY = y;
  } else {
    // Finger lifted
    cameraControl.mouseLeftPressed = false;
  }

  // Handle RIGHT button (Ball throwing)
  if (buttons & 2) {
    if (!cameraControl.mouseRightPressed) {
      cameraControl.mouseRightPressed = true;
      // Note: We check capture status in NewFrame/Update for consistent behavior
      // but for discrete clicks, we use the stale capture status or wait.
      // On Android, we don't currently generate right clicks easily.
      if (imguiSystem && !imguiSystem->WantCaptureMouse()) {
        ThrowBall(x, y);
      }
    }
  } else {
    cameraControl.mouseRightPressed = false;
  }

  // Update hover detection
  HandleMouseHover(x, y);
}
void Engine::handleKeyInput(uint32_t key, bool pressed) {
#if !defined(PLATFORM_ANDROID)
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
    default:
      break;
  }
#else
  // Android uses different input handling via touch events
  (void) key;
  (void) pressed;
#endif

  if (imguiSystem) {
    imguiSystem->HandleKeyboard(key, pressed);
  }
}

void Engine::Update(TimeDelta deltaTime) {
  // Apply any entity removals requested by background threads.
  ProcessPendingEntityRemovals();

  // During background scene loading we avoid touching the live entity
  // list from the main thread. This lets the loading thread construct
  // entities/components safely while the main thread only drives the
  // UI/loading overlay.
  if (renderer && renderer->IsLoading()) {
    if (imguiSystem) {
      uint32_t rw, rh;
      renderer->GetSwapChainExtent(&rw, &rh);
      if (rw > 0 && rh > 0) {
        imguiSystem->HandleResize(rw, rh);
      }
      imguiSystem->NewFrame();
    }
    return;
  }

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
  if (imguiSystem) {
    uint32_t rw, rh;
    renderer->GetSwapChainExtent(&rw, &rh);
    if (rw > 0 && rh > 0) {
      imguiSystem->HandleResize(rw, rh);
    }
    imguiSystem->NewFrame();
  }

  // Update camera controls
  if (activeCamera) {
    UpdateCameraControls(deltaTime);
  }

  // Update all entities.
  // Do not hold `entitiesMutex` while calling `Entity::Update()`.
  // Background threads may need the unique lock to add entities during loading,
  // and holding a shared lock for a long time can starve them.
  std::vector<Entity *> snapshot; {
    std::shared_lock<std::shared_mutex> lk(entitiesMutex);
    snapshot.reserve(entities.size());
    for (auto& uptr : entities) {
      snapshot.push_back(uptr.get());
    }
  }
  for (Entity* entity : snapshot) {
    if (!entity || !entity->IsActive())
      continue;
    entity->Update(deltaTime);
  }
}

void Engine::Render() {
  // Ensure renderer is ready
  if (!renderer || !renderer->IsInitialized()) {
    return;
  }

  // Check if we have an active camera
  if (!activeCamera) {
    return;
  }

  // Apply any entity removals requested by background threads before taking a snapshot.
  ProcessPendingEntityRemovals();

  // Snapshot entity pointers under a short shared lock, then release the lock
  // before rendering. This prevents starving the background loader/physics threads
  // that need the unique lock to create entities/components.
  std::vector<Entity *> snapshot; {
    std::shared_lock<std::shared_mutex> lk(entitiesMutex);
    snapshot.reserve(entities.size());
    for (auto& uptr : entities) {
      snapshot.push_back(uptr.get());
    }
  }

  // Render the scene (ImGui will be rendered within the render pass)
  renderer->Render(snapshot, activeCamera, imguiSystem.get());
}

std::chrono::milliseconds Engine::CalculateDeltaTimeMs() {
  // Get current time using a steady clock to avoid system time jumps
  uint64_t currentTime = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch())
    .count());

  // Initialize lastFrameTimeMs on first call
  if (lastFrameTimeMs == 0) {
    lastFrameTimeMs = currentTime;
    return std::chrono::milliseconds(16); // ~16ms as a sane initial guess
  }

  // Calculate delta time in milliseconds
  uint64_t delta = currentTime - lastFrameTimeMs;

  // Update last frame time
  lastFrameTimeMs = currentTime;

  return std::chrono::milliseconds(static_cast<long long>(delta));
}

void Engine::HandleResize(int width, int height) const {
  if (height <= 0 || width <= 0) {
    return;
  }
  LOGI("Engine: HandleResize %dx%d", width, height);

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

void Engine::UpdateCameraControls(TimeDelta deltaTime) {
  if (!activeCamera)
    return;

  // Get a camera transform component
  auto* cameraTransform = activeCamera->GetOwner()->GetComponent<TransformComponent>();
  if (!cameraTransform)
    return;

  // Check if camera tracking is enabled
  if (imguiSystem&& imguiSystem
  
  ->
  IsCameraTrackingEnabled()
  ) {
    // Find the first active ball entity
    Entity* ballEntity = nullptr; {
      std::shared_lock<std::shared_mutex> lk(entitiesMutex);
      auto ballEntityIt = std::ranges::find_if(entities,
                                               [](auto const& entity) {
                                                 return entity && entity->IsActive() && (entity->GetName().find("Ball_") != std::string::npos);
                                               });
      ballEntity = (ballEntityIt != entities.end()) ? ballEntityIt->get() : nullptr;
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
  float velocity = cameraControl.cameraSpeed * deltaTime.count() * .001f;

  // Check if ImGui wants to capture mouse input (updated in NewFrame)
  bool imguiWantsMouse = imguiSystem && imguiSystem->WantCaptureMouse();

#if defined(PLATFORM_ANDROID)
  // --- Android: Ambitious Controls ---
  // 1. Accelerometer -> Rotation (Tilting)
  float ax, ay, az;
  float androidPitchOffset = 0.0f;
  float androidYawOffset = 0.0f;
  if (platform->GetAccelerometerData(&ax, &ay, &az)) {
    // Correct for display rotation (Portrait vs Landscape vs Reversed)
    ax = -ax;
    ay = -ay;
    float rawX = ax;
    float rawY = ay;
    int rotation = platform->GetDisplayRotation();
    switch (rotation) {
      case 1: // ROTATION_90 (Landscape Left)
        ax = -rawY; ay = rawX; break;
      case 2: // ROTATION_180 (Portrait Upside Down)
        ax = -rawX; ay = -rawY; break;
      case 3: // ROTATION_270 (Landscape Right)
        ax = rawY; ay = -rawX; break;
      default: // ROTATION_0 (Portrait)
        break;
    }

    // If not calibrated, take current values as neutral
    if (!cameraControl.tiltCalibrated) {
      cameraControl.tiltCenterX = ax;
      cameraControl.tiltCenterY = ay;
      cameraControl.tiltCalibrated = true;
    }

    float dax = ax - cameraControl.tiltCenterX;
    float day = ay - cameraControl.tiltCenterY;

    // Auto-recalibration: If the phone is held steady (small delta from current center),
    // we slowly drift the center point towards the current reading.
    // This allows the "rest" position to adapt to the user's hands.
    float distFromCenter = std::sqrt(dax*dax + day*day);
    float dt = deltaTime.count() * 0.001f;

    if (distFromCenter < 1.5f) {
      // Steady detection: If we're close to the center for a while, snap it.
      cameraControl.tiltSteadyTime += dt;
      if (cameraControl.tiltSteadyTime > 0.5f) {
        // Drift the center towards current value to "establish a new deadzone"
        float driftRate = 2.0f * dt;
        cameraControl.tiltCenterX += dax * driftRate;
        cameraControl.tiltCenterY += day * driftRate;
      }
    } else {
      cameraControl.tiltSteadyTime = 0.0f;
    }

    // Deadzone and immediate stop: if within deadzone, motion is ZERO.
    // Increased deadzone to 0.8f for more stability.
    if (std::abs(dax) < 0.8f) dax = 0.0f;
    if (std::abs(day) < 0.8f) day = 0.0f;

    // We multiply by deltaTime to ensure consistent rotation speed across different frame rates.
    const float tiltSensitivity = 20.0f; // Degrees per second at max tilt
    androidYawOffset = dax * tiltSensitivity * dt;
    androidPitchOffset = -day * tiltSensitivity * dt;
  }

  // 2. Swipe -> Movement
  float androidMoveForward = 0.0f;
  float androidMoveRight = 0.0f;
  if (cameraControl.mouseLeftPressed && !cameraControl.startedOnImGui) {
    const float moveSensitivity = 0.15f;
    androidMoveRight = cameraControl.touchMoveX * moveSensitivity;
    androidMoveForward = -cameraControl.touchMoveY * moveSensitivity;
  }
  // Clear touch movement frame delta
  cameraControl.touchMoveX = 0.0f;
  cameraControl.touchMoveY = 0.0f;

  // 3. Tap and Hold -> Reset Camera & Recalibrate Tilt
  bool isHoldingToReset = false;
  if (cameraControl.mouseLeftPressed && !cameraControl.startedOnImGui) {
    cameraControl.touchStartTime += deltaTime.count() * 0.001f;
    // If held for more than 0.5s without moving more than 10 pixels
    if (cameraControl.touchStartTime > 0.5f && cameraControl.touchTotalDistance < 10.0f) {
      cameraControl.yaw = 0.0f;
      cameraControl.pitch = 0.0f;

      // Force the camera to be level (horizon-aligned) during reset.
      // We extract the yaw from the base orientation and discard pitch/roll.
      glm::vec3 euler = glm::eulerAngles(cameraControl.baseOrientation);
      cameraControl.baseOrientation = glm::angleAxis(euler.y, glm::vec3(0.0f, 1.0f, 0.0f));

      // Recalibrate: current physical orientation becomes the new "zero"
      // We use the rotation-corrected values already calculated in ax/ay above.
      cameraControl.tiltCenterX = ax;
      cameraControl.tiltCenterY = ay;

      androidYawOffset = 0.0f;
      androidPitchOffset = 0.0f;
      isHoldingToReset = true;
    }
  } else {
    cameraControl.touchStartTime = 0.0f;
  }
#endif

  // INTERACTION LOCKING LOGIC:
  // If a touch began, we wait until ImGui has processed the first DOWN event (in NewFrame)
  // before deciding whether this drag belongs to the GUI or the 3D Scene.
  if (cameraControl.mouseLeftPressed) {
    if (cameraControl.isFirstFrameOfInteraction) {
      // This is the first frame (Update call) where the finger is DOWN.
      // ImGui's WantCaptureMouse now accurately reflects if the tap was on a window.
      cameraControl.startedOnImGui = imguiWantsMouse;
      cameraControl.isFirstFrameOfInteraction = false;
    }

    // Only apply rotation if the interaction started on the scene background
    if (!cameraControl.startedOnImGui) {
#if !defined(PLATFORM_ANDROID)
      float xOffset = cameraControl.pendingXOffset * cameraControl.mouseSensitivity;
      float yOffset = cameraControl.pendingYOffset * cameraControl.mouseSensitivity;

      cameraControl.yaw -= xOffset;
      cameraControl.pitch -= yOffset;
#endif
    }
  } else {
    // Reset locking state when finger is lifted
    cameraControl.isFirstFrameOfInteraction = true;
    cameraControl.startedOnImGui = false;
  }

#if defined(PLATFORM_ANDROID)
  // Apply Android tilt and swiping
  if (!isHoldingToReset) {
    cameraControl.yaw += androidYawOffset;
    cameraControl.pitch += androidPitchOffset;
  }
#endif

  // Constrain pitch to avoid gimbal lock
  if (cameraControl.pitch > 89.0f)
    cameraControl.pitch = 89.0f;
  if (cameraControl.pitch < -89.0f)
    cameraControl.pitch = -89.0f;

  // Clear accumulated offsets after processing
  cameraControl.pendingXOffset = 0.0f;
  cameraControl.pendingYOffset = 0.0f;

  // Capture base orientation from GLTF camera once and then apply mouse deltas relative to it
  if (!cameraControl.baseOrientationCaptured) {
    // TransformComponent stores Euler in radians; convert to quaternion
    glm::vec3 baseEuler = cameraTransform->GetRotation();
    const glm::quat qx = glm::angleAxis(baseEuler.x, glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::quat qy = glm::angleAxis(baseEuler.y, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat qz = glm::angleAxis(baseEuler.z, glm::vec3(0.0f, 0.0f, 1.0f));
    // Match CameraComponent::UpdateViewMatrix composition (q = qz * qy * qx)
    cameraControl.baseOrientation = qz * qy * qx;
    cameraControl.baseOrientationCaptured = true;
  }

  // Build delta orientation from yaw/pitch mouse deltas (degrees -> radians)
  const float yawRad = glm::radians(cameraControl.yaw);
  const float pitchRad = glm::radians(cameraControl.pitch);
  const glm::quat qDeltaY = glm::angleAxis(yawRad, glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::quat qDeltaX = glm::angleAxis(pitchRad, glm::vec3(1.0f, 0.0f, 0.0f));
  // Apply yaw then pitch in the same convention as CameraComponent (ZYX overall), so delta = Ry * Rx
  glm::quat qDelta = qDeltaY * qDeltaX;
  glm::quat qFinal = cameraControl.baseOrientation * qDelta;

  // Derive camera basis directly from rotated axes to avoid ambiguity
  glm::vec3 right = glm::normalize(qFinal * glm::vec3(1.0f, 0.0f, 0.0f));
  glm::vec3 up = glm::normalize(qFinal * glm::vec3(0.0f, 1.0f, 0.0f));
  // Camera forward in world space.
  // Our view/projection conventions assume the camera looks down -Z in its local space.
  glm::vec3 front = glm::normalize(qFinal * glm::vec3(0.0f, 0.0f, -1.0f));

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

#if defined(PLATFORM_ANDROID)
  // Apply Android swipe-to-walk displacement
  // We use the same front/right vectors but apply the swipe deltas.
  // Note: androidMoveForward/Right are already calculated in the Android block above.
  position += front * androidMoveForward * cameraControl.cameraSpeed * 0.02f;
  position += right * androidMoveRight * cameraControl.cameraSpeed * 0.02f;
#endif

#if defined(PLATFORM_ANDROID)
  // Apply Android swipe-based movement
  position += front * androidMoveForward;
  position += right * androidMoveRight;
#endif

  // Update camera position
  cameraTransform->SetPosition(position);
  // Apply rotation to the camera transform based on GLTF base orientation plus mouse deltas
  // TransformComponent expects radians Euler (ZYX order in our CameraComponent).
  cameraTransform->SetRotation(glm::eulerAngles(qFinal));

  // Update camera target based on a direction
  glm::vec3 target = position + front;
  activeCamera->SetTarget(target);

  // Ensure the camera view matrix reflects the new transform immediately this frame
  activeCamera->ForceViewMatrixUpdate();
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
  physicsScaling.gameUnitsToMeters = 0.1f; // 1 game unit = 0.1 meter (10cm) - smaller scale

  // Much reduced force scaling to prevent extreme speeds
  // With base forces 0.01f-0.05f, this gives final forces of 0.001f-0.005f
  physicsScaling.forceScale = 1.0f; // Minimal force scaling for realistic movement
  physicsScaling.physicsTimeScale = 1.0f; // Keep time scale normal
  physicsScaling.gravityScale = 1.0f; // Keep gravity proportional to scale

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

      // Request an acceleration structure build so the new ball is included in Ray Query mode.
      // We do this after creating the rigid body and initializing the entity.
      renderer->RequestAccelerationStructureBuild("Ball spawned");

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
  // Record main thread identity
  mainThreadId = std::this_thread::get_id();

  // Create platform
  platform = CreatePlatform(app);

  // Wait for the window to be initialized before continuing
  while (app->window == nullptr) {
    if (!platform->ProcessEvents()) {
      return false; // Exit requested
    }
  }

  if (!platform->Initialize(appName, 0, 0)) {
    return false;
  }

  // Set resize callback
  platform->SetResizeCallback([this](int width, int height) {
    HandleResize(width, height);
  });

  // Set mouse callback
  platform->SetMouseCallback([this](float x, float y, uint32_t buttons) {
    handleMouseInput(x, y, buttons);
  });

  // Set keyboard callback
  platform->SetKeyboardCallback([this](uint32_t key, bool pressed) {
    handleKeyInput(key, pressed);
  });

  // Set char callback
  platform->SetCharCallback([this](uint32_t c) {
    if (imguiSystem) {
      imguiSystem->HandleChar(c);
    }
  });

  // Create renderer
  LOGI("Engine: Initializing Renderer...");
  renderer = std::make_unique<Renderer>(platform.get());
  if (!renderer->Initialize(appName, enableValidationLayers)) {
    LOGE("Engine: Renderer initialization failed");
    return false;
  }
  LOGI("Engine: Renderer initialized successfully");

  // Get window dimensions from platform for ImGui initialization
  int width, height;
  platform->GetWindowSize(&width, &height);
  LOGI("Engine: Initial window size: %dx%d", width, height);

  // Ensure initial size is applied to camera and renderer
  HandleResize(width, height);

  try {
    // Model loader via constructor; also wire into renderer
    LOGI("Engine: Initializing ModelLoader...");
    modelLoader = std::make_unique<ModelLoader>(renderer.get());
    renderer->SetModelLoader(modelLoader.get());

    // Audio system via constructor
    LOGI("Engine: Initializing AudioSystem...");
    audioSystem = std::make_unique<AudioSystem>(this, renderer.get());

    // Physics system via constructor (GPU enabled)
    LOGI("Engine: Initializing PhysicsSystem...");
    physicsSystem = std::make_unique<PhysicsSystem>(renderer.get(), true);

#ifdef ENABLE_COURSE_OPACITY_MICROMAPS
    // OMM integration via constructor
LOGI("Engine: Initializing OmmIntegration...");
ommIntegration = std::make_unique<OmmIntegration>();
    ommIntegration->init(*renderer, *modelLoader);
#endif

    // ImGui via constructor, then connect audio system
    LOGI("Engine: Initializing ImGuiSystem...");
    imguiSystem = std::make_unique<ImGuiSystem>(renderer.get(), width, height);
    imguiSystem->SetAudioSystem(audioSystem.get());
    LOGI("Engine: Subsystems initialized successfully");
  } catch (const std::exception& e) {
    LOGE("Subsystem initialization failed: %s", e.what());
    return false;
  }

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

  while (running) {
    // Process Android events
    if (!platform->ProcessEvents()) {
      running = false;
      break;
    }

    // Only update and render if we have a valid window size
    int width, height;
    platform->GetWindowSize(&width, &height);
    if (width > 0 && height > 0) {
      // Calculate delta time
      deltaTimeMs = CalculateDeltaTimeMs();

      // Update
      Update(deltaTimeMs);

      // Render
      Render();
    } else {
      // If the window is not ready or minimized, yield to the system
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
  }
}
#endif
