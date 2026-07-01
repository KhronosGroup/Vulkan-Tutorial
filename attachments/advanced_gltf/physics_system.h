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
#pragma once

#include <chrono>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

class Entity;
class Renderer;

/**
 * @brief Enum for different collision shapes.
 */
enum class CollisionShape {
  Box,
  Sphere,
  Capsule,
  Mesh
};

/**
 * @brief Class representing a rigid body for physics simulation.
 */
class RigidBody {
  public:
    /**
	 * @brief Default constructor.
	 */
    RigidBody() = default;

    /**
	 * @brief Destructor for proper cleanup.
	 */
    virtual ~RigidBody() = default;

    /**
	 * @brief Set the position of the rigid body.
	 * @param position The position.
	 */
    virtual void SetPosition(const glm::vec3& position) = 0;

    /**
	 * @brief Set the rotation of the rigid body.
	 * @param rotation The rotation quaternion.
	 */
    virtual void SetRotation(const glm::quat& rotation) = 0;

    /**
	 * @brief Set the scale of the rigid body.
	 * @param scale The scale.
	 */
    virtual void SetScale(const glm::vec3& scale) = 0;

    /**
	 * @brief Set the mass of the rigid body.
	 * @param mass The mass.
	 */
    virtual void SetMass(float mass) = 0;

    /**
	 * @brief Set the restitution (bounciness) of the rigid body.
	 * @param restitution The restitution (0.0f to 1.0f).
	 */
    virtual void SetRestitution(float restitution) = 0;

    /**
	 * @brief Set the friction of the rigid body.
	 * @param friction The friction (0.0f to 1.0f).
	 */
    virtual void SetFriction(float friction) = 0;

    /**
	 * @brief Apply a force to the rigid body.
	 * @param force The force vector.
	 * @param localPosition The local position to apply the force at.
	 */
    virtual void ApplyForce(const glm::vec3& force, const glm::vec3& localPosition = glm::vec3(0.0f)) = 0;

    /**
	 * @brief Apply an impulse to the rigid body.
	 * @param impulse The impulse vector.
	 * @param localPosition The local position to apply the impulse at.
	 */
    virtual void ApplyImpulse(const glm::vec3& impulse, const glm::vec3& localPosition = glm::vec3(0.0f)) = 0;

    /**
	 * @brief Set the linear velocity of the rigid body.
	 * @param velocity The linear velocity.
	 */
    virtual void SetLinearVelocity(const glm::vec3& velocity) = 0;

    /**
	 * @brief Set the angular velocity of the rigid body.
	 * @param velocity The angular velocity.
	 */
    virtual void SetAngularVelocity(const glm::vec3& velocity) = 0;

    /**
	 * @brief Get the position of the rigid body.
	 * @return The position.
	 */
    [[nodiscard]] virtual glm::vec3 GetPosition() const = 0;

    /**
	 * @brief Get the rotation of the rigid body.
	 * @return The rotation quaternion.
	 */
    [[nodiscard]] virtual glm::quat GetRotation() const = 0;

    /**
	 * @brief Get the linear velocity of the rigid body.
	 * @return The linear velocity.
	 */
    [[nodiscard]] virtual glm::vec3 GetLinearVelocity() const = 0;

    /**
	 * @brief Get the angular velocity of the rigid body.
	 * @return The angular velocity.
	 */
    [[nodiscard]] virtual glm::vec3 GetAngularVelocity() const = 0;

    /**
	 * @brief Set whether the rigid body is kinematic.
	 * @param kinematic Whether the rigid body is kinematic.
	 */
    virtual void SetKinematic(bool kinematic) = 0;

    /**
	 * @brief Check if the rigid body is kinematic.
	 * @return True if kinematic, false otherwise.
	 */
    [[nodiscard]] virtual bool IsKinematic() const = 0;
};


/**
 * @brief Class for managing physics simulation.
 *
 * This class implements the physics system as described in the Subsystems chapter:
 * @see en/Building_a_Simple_Engine/Subsystems/04_physics_basics.adoc
 * @see en/Building_a_Simple_Engine/Subsystems/05_vulkan_physics.adoc
 */
class PhysicsSystem {
  public:
    /**
	 * @brief Default constructor.
	 */
    /**
	 * @brief Constructor.
	 * @param _renderer Optional renderer (unused in Jolt implementation).
	 * @param _enableGPU Optional GPU flag (unused in Jolt implementation).
	 */
    explicit PhysicsSystem(Renderer* _renderer = nullptr, bool _enableGPU = true) {
      if (!Initialize()) {
        throw std::runtime_error("PhysicsSystem: initialization failed");
      }
    }

    /**
	 * @brief Destructor for proper cleanup.
	 */
    ~PhysicsSystem();

    /**
	 * @brief Update the physics system.
	 * @param deltaTime The time elapsed since the last update.
	 */
    void Update(std::chrono::milliseconds deltaTime);

    /**
	 * @brief Create a rigid body.
	 * @param entity The entity to attach the rigid body to.
	 * @param shape The collision shape.
	 * @param mass The mass.
	 * @return Pointer to the created rigid body, or nullptr if creation failed.
	 */
    RigidBody* CreateRigidBody(Entity* entity, CollisionShape shape, float mass);

    /**
	 * @brief Destroy a rigid body.
	 * @param rigidBody The rigid body to destroy.
	 * @return True if destruction was successful, false otherwise.
	 */
    bool DestroyRigidBody(RigidBody* rigidBody);

    /**
	 * @brief Set the gravity of the physics world.
	 * @param _gravity The gravity vector.
	 */
    void SetGravity(const glm::vec3& _gravity);

    /**
	 * @brief Get the gravity of the physics world.
	 * @return The gravity vector.
	 */
    [[nodiscard]] glm::vec3 GetGravity() const;

    /**
	 * @brief Perform a raycast.
	 * @param origin The origin of the ray.
	 * @param direction The direction of the ray.
	 * @param maxDistance The maximum distance of the ray.
	 * @param hitPosition Output parameter for the hit position.
	 * @param hitNormal Output parameter for the hit normal.
	 * @param hitEntity Output parameter for the hit entity.
	 * @return True if the ray hit something, false otherwise.
	 */
    bool Raycast(const glm::vec3& origin,
                 const glm::vec3& direction,
                 float maxDistance,
                 glm::vec3* hitPosition,
                 glm::vec3* hitNormal,
                 Entity** hitEntity) const;


    /**
	 * @brief Set the current camera position for geometry-relative ball checking.
	 * @param _cameraPosition The current camera position.
	 */
    void SetCameraPosition(const glm::vec3& _cameraPosition) {
      std::lock_guard<std::mutex> lock(cameraPositionMutex);
      cameraPosition = _cameraPosition;
    }

    // Thread-safe enqueue for rigid body creation from any thread
    void EnqueueRigidBodyCreation(Entity* entity,
                                  CollisionShape shape,
                                  float mass,
                                  bool kinematic,
                                  float restitution,
                                  float friction);

  private:
    /**
	 * @brief Initialize the physics system (called by constructor).
	 * @return True if initialization was successful, false otherwise.
	 */
    bool Initialize();

    /**
	 * @brief Clean up rigid bodies that are marked for removal.
	 */
    void CleanupMarkedBodies();

    // Pending rigid body creations queued from background threads
    struct PendingCreation {
      Entity* entity;
      CollisionShape shape;
      float mass;
      bool kinematic;
      float restitution;
      float friction;
    };
    std::mutex pendingMutex;
    std::vector<PendingCreation> pendingCreations;

    // ------------------------------------------------------------------
    // Streaming colliders: for very large scenes (Bistro has ~500+ static
    // colliders), creating Jolt bodies for everything up-front causes
    // multi-second hitches and burns memory. Instead, static colliders are
    // registered as "streaming candidates" with a precomputed world-space
    // center; PhysicsSystem::Update promotes candidates within
    // `streamingRadius` of the camera to real bodies, and removes bodies
    // that drift beyond `streamingEvictRadius`.
    // ------------------------------------------------------------------
    struct StreamingCandidate {
      Entity* entity;
      CollisionShape shape;
      float mass;
      bool kinematic;
      float restitution;
      float friction;
      glm::vec3 center;
      bool active = false; // true while a real body exists for this candidate
    };
    std::mutex streamingMutex;
    std::vector<StreamingCandidate> streamingCandidates;
    // Map entity -> index into streamingCandidates for fast eviction lookup.
    std::unordered_map<Entity*, size_t> entityToStreamingIndex;
    // Streaming distances (squared, in world units). Defaults tuned for Bistro
    // where ~15m surrounds the immediate room with walls/floor.
    float streamingRadius = 20.0f;
    float streamingEvictRadius = 30.0f;

  public:
    // Register a static collider for distance-based streaming. The body is NOT
    // created until the camera moves within `streamingRadius` of `center`.
    void RegisterStreamingCollider(Entity* entity,
                                   CollisionShape shape,
                                   float mass,
                                   bool kinematic,
                                   float restitution,
                                   float friction,
                                   const glm::vec3& center);

  private:

    // Rigid bodies
    mutable std::mutex rigidBodiesMutex; // Protect concurrent access to rigidBodies
    std::vector<std::unique_ptr<RigidBody>> rigidBodies;

    // Gravity
    glm::vec3 gravity = glm::vec3(0.0f, -9.81f, 0.0f);

    // Whether the physics system is initialized
    bool initialized = false;

    // Camera position for geometry-relative ball checking
    mutable std::mutex cameraPositionMutex;
    glm::vec3 cameraPosition = glm::vec3(0.0f, 0.0f, 0.0f);
};