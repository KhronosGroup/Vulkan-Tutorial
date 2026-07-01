/* Copyright (c) 2026 Holochip Corporation
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
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <cstdint>
#include <memory>

// Forward declarations — callers that only use PhysicsWorld* need not include Jolt headers.
namespace JPH {
    class BodyID;
    class BodyCreationSettings;
    enum class EMotionType : std::uint8_t;
}

// POD transform used to exchange pose data across the physics boundary.
struct PhysicsPose {
    glm::vec3 position;
    glm::quat orientation;

    glm::mat4 to_matrix() const {
        return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(orientation);
    }
};

// Abstract physics world interface — canonical from appendix_types.adoc.
// JoltPhysicsWorld (physics_world_jolt.cpp) is the only concrete implementation in this tutorial.
class PhysicsWorld {
public:
    virtual ~PhysicsWorld() = default;

    // --- Global lifecycle ---
    static void global_init();
    static void global_shutdown();
    static std::unique_ptr<PhysicsWorld> create();

    // --- Body management ---
    virtual JPH::BodyID create_body(const JPH::BodyCreationSettings& settings) = 0;
    virtual void        destroy_body(JPH::BodyID body_id) = 0;
    virtual void        set_motion_type(JPH::BodyID body_id, JPH::EMotionType type) = 0;
    virtual void        set_object_layer(JPH::BodyID body_id, uint16_t layer) = 0;
    virtual void        activate_body(JPH::BodyID body_id) = 0;

    // --- Kinematic/dynamic sync ---
    virtual void        move_kinematic(JPH::BodyID body_id, const PhysicsPose& pose) = 0;
    virtual PhysicsPose get_body_pose(JPH::BodyID body_id) const = 0;
    virtual glm::vec3   get_linear_velocity(JPH::BodyID body_id) const = 0;
    virtual void        set_linear_velocity(JPH::BodyID body_id, const glm::vec3& velocity) = 0;

    // --- Constraints ---
    // swing/twist limits are in radians.
    virtual void create_ball_socket_constraint(JPH::BodyID p1, JPH::BodyID p2,
                                               float swing_rad, float twist_rad) = 0;
    // min_angle/max_angle are in radians.
    virtual void create_hinge_constraint(JPH::BodyID p1, JPH::BodyID p2,
                                         const glm::vec3& axis,
                                         float min_angle_rad, float max_angle_rad) = 0;

    // --- Simulation ---
    virtual void step(float delta_seconds) = 0;

    // --- Queries ---
    virtual bool raycast(const glm::vec3& origin, const glm::vec3& direction, float max_distance,
                         float& out_distance, glm::vec3& out_normal, JPH::BodyID& out_body_id) const = 0;
};
