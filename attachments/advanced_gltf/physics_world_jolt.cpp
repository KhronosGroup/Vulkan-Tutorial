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
// Jolt Physics 5.x wrapper implementing PhysicsWorld.
// Tested against JoltPhysics commit pinned in CMakeLists.txt (tag v5.2.0).
// The Jolt API is not ABI-stable across major versions — check their CHANGELOG if you update.
#include "physics_interface.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>

#include <cassert>

// Broad-phase layers: static world geometry vs everything else.
namespace BPLayers {
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr uint32_t NUM_LAYERS = 2;
}

// Object layers: maps to broad-phase layers.
namespace ObjLayers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING     = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
}

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    BPLayerInterfaceImpl() {
        obj_to_bp[ObjLayers::NON_MOVING] = BPLayers::NON_MOVING;
        obj_to_bp[ObjLayers::MOVING]     = BPLayers::MOVING;
    }
    uint32_t GetNumBroadPhaseLayers() const override { return BPLayers::NUM_LAYERS; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        assert(layer < ObjLayers::NUM_LAYERS);
        return obj_to_bp[layer];
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
        return layer == BPLayers::NON_MOVING ? "NON_MOVING" : "MOVING";
    }
#endif
private:
    JPH::BroadPhaseLayer obj_to_bp[ObjLayers::NUM_LAYERS];
};

class ObjVsBPFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer obj, JPH::BroadPhaseLayer bp) const override {
        if (obj == ObjLayers::NON_MOVING) return bp == BPLayers::MOVING;
        return true; // MOVING collides with everything
    }
};

class ObjVsObjFilter final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        if (a == ObjLayers::NON_MOVING) return b == ObjLayers::MOVING;
        return true;
    }
};

// Concrete implementation — owns the JPH::PhysicsSystem.
class JoltPhysicsWorld final : public PhysicsWorld {
public:
    // Initializes Jolt's global factory and registers built-in types.
    // Call this once at application startup before constructing any JoltPhysicsWorld.
    static void global_init() {
        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }

    // Call at application shutdown after all JoltPhysicsWorld instances are destroyed.
    static void global_shutdown() {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }

    explicit JoltPhysicsWorld(
        uint32_t max_bodies        = 20480,
        uint32_t max_body_pairs    = 65536,
        uint32_t max_contact_constraints = 32768,
        uint32_t num_worker_threads = 4)
    {
        temp_allocator_ = std::make_unique<JPH::TempAllocatorImpl>(32 * 1024 * 1024);
        job_system_     = std::make_unique<JPH::JobSystemThreadPool>(
            JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, num_worker_threads);

        system_.Init(max_bodies, 0, max_body_pairs, max_contact_constraints,
                     bp_layer_interface_, obj_vs_bp_filter_, obj_vs_obj_filter_);
        body_interface_ = &system_.GetBodyInterface();
    }

    JPH::BodyID create_body(const JPH::BodyCreationSettings& settings) override {
        JPH::Body* body = body_interface_->CreateBody(settings);
        if (!body) return JPH::BodyID();
        body_interface_->AddBody(body->GetID(), JPH::EActivation::Activate);
        return body->GetID();
    }

    void destroy_body(JPH::BodyID id) override {
        body_interface_->RemoveBody(id);
        body_interface_->DestroyBody(id);
    }

    void set_motion_type(JPH::BodyID id, JPH::EMotionType type) override {
        body_interface_->SetMotionType(id, type, JPH::EActivation::Activate);
    }

    void set_object_layer(JPH::BodyID id, uint16_t layer) override {
        body_interface_->SetObjectLayer(id, layer);
    }

    void activate_body(JPH::BodyID id) override {
        body_interface_->ActivateBody(id);
    }

    void move_kinematic(JPH::BodyID id, const PhysicsPose& pose) override {
        body_interface_->SetPositionAndRotation(
            id,
            JPH::RVec3(pose.position.x, pose.position.y, pose.position.z),
            JPH::Quat(pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w),
            JPH::EActivation::Activate);
    }

    PhysicsPose get_body_pose(JPH::BodyID id) const override {
        JPH::RVec3 pos;
        JPH::Quat  rot;
        body_interface_->GetPositionAndRotation(id, pos, rot);
        return {
            glm::vec3(pos.GetX(), pos.GetY(), pos.GetZ()),
            glm::quat(rot.GetW(), rot.GetX(), rot.GetY(), rot.GetZ()),
        };
    }

    glm::vec3 get_linear_velocity(JPH::BodyID id) const override {
        JPH::Vec3 v = body_interface_->GetLinearVelocity(id);
        return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
    }

    void set_linear_velocity(JPH::BodyID id, const glm::vec3& v) override {
        body_interface_->SetLinearVelocity(id, JPH::Vec3(v.x, v.y, v.z));
    }

    // Approximates a ball-socket using SwingTwistConstraint.
    void create_ball_socket_constraint(JPH::BodyID p1, JPH::BodyID p2,
                                       float swing_rad, float twist_rad) override {
        JPH::SwingTwistConstraintSettings s;
        s.mSpace          = JPH::EConstraintSpace::LocalToBodyCOM;
        s.mNormalHalfConeAngle = swing_rad;
        s.mPlaneHalfConeAngle  = swing_rad;
        s.mTwistMinAngle  = -twist_rad;
        s.mTwistMaxAngle  =  twist_rad;

        JPH::BodyLockWrite lock1(system_.GetBodyLockInterface(), p1);
        JPH::BodyLockWrite lock2(system_.GetBodyLockInterface(), p2);
        if (lock1.Succeeded() && lock2.Succeeded()) {
            auto* c = static_cast<JPH::SwingTwistConstraint*>(
                s.Create(lock1.GetBody(), lock2.GetBody()));
            system_.AddConstraint(c);
        }
    }

    void create_hinge_constraint(JPH::BodyID p1, JPH::BodyID p2,
                                 const glm::vec3& axis,
                                 float min_angle_rad, float max_angle_rad) override {
        JPH::HingeConstraintSettings s;
        s.mSpace      = JPH::EConstraintSpace::WorldSpace;
        s.mHingeAxis1 = s.mHingeAxis2 = JPH::Vec3(axis.x, axis.y, axis.z);
        s.mNormalAxis1 = s.mNormalAxis2 = JPH::Vec3(0, 1, 0); // arbitrary perp
        s.mLimitsMin  = min_angle_rad;
        s.mLimitsMax  = max_angle_rad;

        JPH::BodyLockWrite lock1(system_.GetBodyLockInterface(), p1);
        JPH::BodyLockWrite lock2(system_.GetBodyLockInterface(), p2);
        if (lock1.Succeeded() && lock2.Succeeded()) {
            auto* c = static_cast<JPH::HingeConstraint*>(
                s.Create(lock1.GetBody(), lock2.GetBody()));
            system_.AddConstraint(c);
        }
    }

    void step(float delta_seconds) override {
        // cCollisionSteps=1 is fine for game-rate updates (60 Hz).
        // Increase to 2 or 3 for ragdoll simulations to improve constraint stability.
        system_.Update(delta_seconds, /*cCollisionSteps=*/1, temp_allocator_.get(), job_system_.get());
    }

    bool raycast(const glm::vec3& origin, const glm::vec3& direction, float max_distance,
                 float& out_distance, glm::vec3& out_normal, JPH::BodyID& out_body_id) const override {
        JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                          JPH::Vec3(direction.x * max_distance, direction.y * max_distance, direction.z * max_distance));
        JPH::RayCastResult result;
        if (system_.GetNarrowPhaseQuery().CastRay(ray, result)) {
            out_distance = result.mFraction * max_distance;
            out_body_id  = result.mBodyID;

            JPH::BodyLockRead lock(system_.GetBodyLockInterface(), out_body_id);
            if (lock.Succeeded()) {
                JPH::Vec3 normal = lock.GetBody().GetWorldSpaceSurfaceNormal(result.mSubShapeID2, ray.GetPointOnRay(result.mFraction));
                out_normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
            } else {
                out_normal = glm::vec3(0, 1, 0);
            }
            return true;
        }
        return false;
    }

    JPH::PhysicsSystem& get_physics_system() { return system_; }

private:
    BPLayerInterfaceImpl bp_layer_interface_;
    ObjVsBPFilter        obj_vs_bp_filter_;
    ObjVsObjFilter       obj_vs_obj_filter_;

    std::unique_ptr<JPH::TempAllocatorImpl>    temp_allocator_;
    std::unique_ptr<JPH::JobSystemThreadPool>  job_system_;

    JPH::PhysicsSystem   system_;
    JPH::BodyInterface*  body_interface_ = nullptr;
};

void PhysicsWorld::global_init() {
    JoltPhysicsWorld::global_init();
}

void PhysicsWorld::global_shutdown() {
    JoltPhysicsWorld::global_shutdown();
}

std::unique_ptr<PhysicsWorld> PhysicsWorld::create() {
    return std::make_unique<JoltPhysicsWorld>();
}
