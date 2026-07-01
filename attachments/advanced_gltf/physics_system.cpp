#include <Jolt/Jolt.h>
#include "physics_system.h"
#include "entity.h"
#include "mesh_component.h"
#include "renderer.h"
#include "renderer_advanced_types.h"
#include "transform_component.h"
#include "physics_interface.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>


namespace {
    // Broad-phase layers: static world geometry vs everything else.
    namespace ObjLayers {
        static constexpr uint16_t NON_MOVING = 0;
        static constexpr uint16_t MOVING     = 1;
    }

    // Static map to associate PhysicsSystem instances with their Jolt worlds
    static std::unordered_map<const PhysicsSystem*, std::unique_ptr<PhysicsWorld>> g_physicsWorlds;
    static std::mutex g_worldsMutex;

    PhysicsWorld* GetWorld(const PhysicsSystem* system) {
        std::lock_guard<std::mutex> lock(g_worldsMutex);
        auto it = g_physicsWorlds.find(system);
        return it != g_physicsWorlds.end() ? it->second.get() : nullptr;
    }
}

class ConcreteRigidBody final : public RigidBody {
public:
    ConcreteRigidBody(Entity* entity, PhysicsWorld* world, JPH::BodyID bodyId, bool isStatic)
        : entity(entity), physicsWorld(world), bodyId(bodyId), isStatic(isStatic) {}

    ~ConcreteRigidBody() override {
        if (physicsWorld && !bodyId.IsInvalid()) {
            physicsWorld->destroy_body(bodyId);
        }
    }

    void SetPosition(const glm::vec3& position) override {
        PhysicsPose pose = physicsWorld->get_body_pose(bodyId);
        pose.position = position;
        physicsWorld->move_kinematic(bodyId, pose);
    }

    void SetRotation(const glm::quat& rotation) override {
        PhysicsPose pose = physicsWorld->get_body_pose(bodyId);
        pose.orientation = rotation;
        physicsWorld->move_kinematic(bodyId, pose);
    }

    void SetScale(const glm::vec3& scale) override {
        // Jolt doesn't support dynamic scaling of shapes easily. 
        // Scale should be set during creation.
    }

    void SetMass(float mass) override {
        // Mass is handled via EMotionType and mass properties in Jolt.
        // For simplicity, we toggle Kinematic/Dynamic in PhysicsSystem::Update.
    }

    void SetRestitution(float restitution) override {
        // Jolt handles this in BodyCreationSettings.
    }

    void SetFriction(float friction) override {
        // Jolt handles this in BodyCreationSettings.
    }

    void ApplyForce(const glm::vec3& force, const glm::vec3& localPosition) override {
        // Not implemented in the minimal wrapper, but could be added.
    }

    void ApplyImpulse(const glm::vec3& impulse, const glm::vec3& localPosition) override {
        // Not implemented in the minimal wrapper.
    }

    void SetLinearVelocity(const glm::vec3& velocity) override {
        physicsWorld->set_linear_velocity(bodyId, velocity);
    }

    void SetAngularVelocity(const glm::vec3& velocity) override {
        // Not implemented in the minimal wrapper.
    }

    [[nodiscard]] glm::vec3 GetPosition() const override {
        return physicsWorld->get_body_pose(bodyId).position;
    }

    [[nodiscard]] glm::quat GetRotation() const override {
        return physicsWorld->get_body_pose(bodyId).orientation;
    }

    [[nodiscard]] glm::vec3 GetLinearVelocity() const override {
        return physicsWorld->get_linear_velocity(bodyId);
    }

    [[nodiscard]] glm::vec3 GetAngularVelocity() const override {
        return glm::vec3(0.0f);
    }

    void SetKinematic(bool kinematic) override {
        if (isStatic) return;
        isKinematic = kinematic;
        physicsWorld->set_motion_type(bodyId, kinematic ? JPH::EMotionType::Kinematic : JPH::EMotionType::Dynamic);
    }

    [[nodiscard]] bool IsKinematic() const override {
        return isKinematic;
    }

    [[nodiscard]] bool IsStatic() const { return isStatic; }

    JPH::BodyID GetBodyID() const { return bodyId; }
    Entity* GetEntity() const { return entity; }

private:
    Entity* entity;
    PhysicsWorld* physicsWorld;
    JPH::BodyID bodyId;
    bool isKinematic = false;
    bool isStatic = false;
};

PhysicsSystem::~PhysicsSystem() {
    std::lock_guard<std::mutex> lock(rigidBodiesMutex);
    rigidBodies.clear(); // Destroy all bodies before shutting down the world
    {
        std::lock_guard<std::mutex> lockW(g_worldsMutex);
        g_physicsWorlds.erase(this);
    }
    PhysicsWorld::global_shutdown();
}

bool PhysicsSystem::Initialize() {
    PhysicsWorld::global_init();
    {
        std::lock_guard<std::mutex> lock(g_worldsMutex);
        g_physicsWorlds[this] = PhysicsWorld::create();
    }
    initialized = true;
    return true;
}

void PhysicsSystem::Update(std::chrono::milliseconds deltaTime) {
    PhysicsWorld* physicsWorld = GetWorld(this);
    if (!physicsWorld) return;

    // 0. Streaming: promote/evict static colliders based on camera distance.
    // This keeps the active body count bounded (only colliders near the camera),
    // which avoids the multi-second hitch caused by creating 500+ Jolt bodies
    // up-front and the associated per-frame sync cost.
    glm::vec3 camPos; {
        std::lock_guard<std::mutex> lk(cameraPositionMutex);
        camPos = cameraPosition;
    }
    const float promoteR2 = streamingRadius * streamingRadius;
    const float evictR2 = streamingEvictRadius * streamingEvictRadius;
    std::vector<Entity*> toEvict;
    {
        std::lock_guard<std::mutex> lk(streamingMutex);
        // Bounded work per frame: only promote up to N candidates / frame so we
        // never block the physics thread on a sudden flood (e.g. teleport).
        constexpr size_t MAX_PROMOTIONS_PER_FRAME = 32;
        size_t promoted = 0;
        for (auto& sc : streamingCandidates) {
            float d2 = glm::length2(sc.center - camPos);
            if (!sc.active) {
                if (d2 <= promoteR2 && promoted < MAX_PROMOTIONS_PER_FRAME) {
                    std::lock_guard<std::mutex> lk2(pendingMutex);
                    pendingCreations.push_back({sc.entity, sc.shape, sc.mass, sc.kinematic, sc.restitution, sc.friction});
                    sc.active = true;
                    ++promoted;
                }
            } else {
                if (d2 > evictR2) {
                    toEvict.push_back(sc.entity);
                    sc.active = false;
                }
            }
        }
    }
    // Evict outside the streaming lock to avoid nested-lock acquisition on rigidBodiesMutex.
    if (!toEvict.empty()) {
        std::lock_guard<std::mutex> lock(rigidBodiesMutex);
        for (Entity* e : toEvict) {
            auto it = std::find_if(rigidBodies.begin(), rigidBodies.end(),
                [e](const auto& rb) {
                    auto* crb = static_cast<ConcreteRigidBody*>(rb.get());
                    return crb && crb->GetEntity() == e && crb->IsStatic();
                });
            if (it != rigidBodies.end()) {
                rigidBodies.erase(it);
            }
        }
    }

    // 1. Process pending creations (batched to avoid hanging the first frame)
    static constexpr uint32_t MAX_CREATIONS_PER_FRAME = 100;
    std::vector<PendingCreation> toCreate; {
        std::lock_guard<std::mutex> lk(pendingMutex);
        size_t count = std::min<size_t>(pendingCreations.size(), MAX_CREATIONS_PER_FRAME);
        if (count > 0) {
            toCreate.assign(pendingCreations.begin(), pendingCreations.begin() + count);
            pendingCreations.erase(pendingCreations.begin(), pendingCreations.begin() + count);
        }
    }
    
    for (auto& pc : toCreate) {
        RigidBody* rb = CreateRigidBody(pc.entity, pc.shape, pc.mass);
        if (rb) {
            rb->SetKinematic(pc.kinematic);
            rb->SetRestitution(pc.restitution);
            rb->SetFriction(pc.friction);
        }
    }

    // 2. Step the simulation
    float dt = deltaTime.count() * 0.001f;
    physicsWorld->step(dt);

    // 3. Sync physics back to entities
    std::lock_guard<std::mutex> lock(rigidBodiesMutex);
    for (auto& rb : rigidBodies) {
        auto* crb = static_cast<ConcreteRigidBody*>(rb.get());
        
        if (crb->IsStatic()) continue;

        Entity* entity = crb->GetEntity();
        if (!entity) continue;

        auto* transform = entity->GetComponent<TransformComponent>();
        if (!transform) continue;

        // If it's a dynamic body (Fox when released), update the entity
        if (!crb->IsKinematic()) {
            PhysicsPose pose = physicsWorld->get_body_pose(crb->GetBodyID());
            if (pose.position != transform->GetPosition() || 
                pose.orientation != glm::quat(transform->GetRotation())) {
                transform->SetPosition(pose.position);
                transform->SetRotation(glm::eulerAngles(pose.orientation));
            }
        } else {
            // If it's kinematic, move the physics body to follow the entity (e.g. Fox when grabbed)
            PhysicsPose pose;
            pose.position = transform->GetPosition();
            pose.orientation = glm::quat(transform->GetRotation());
            physicsWorld->move_kinematic(crb->GetBodyID(), pose);
        }
    }
}

RigidBody* PhysicsSystem::CreateRigidBody(Entity* entity, CollisionShape shape, float mass) {
    PhysicsWorld* physicsWorld = GetWorld(this);
    if (!entity || !physicsWorld) return nullptr;

    auto* transform = entity->GetComponent<TransformComponent>();
    glm::vec3 pos = transform ? transform->GetPosition() : glm::vec3(0.0f);
    glm::quat rot = transform ? glm::quat(transform->GetRotation()) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = transform ? transform->GetScale() : glm::vec3(1.0f);

    JPH::BodyCreationSettings settings;
    settings.mPosition = JPH::RVec3(pos.x, pos.y, pos.z);
    settings.mRotation = JPH::Quat(rot.x, rot.y, rot.z, rot.w);
    
    // Choose layer based on mass/kinematic
    bool isStatic = (mass <= 0.0f);
    settings.mObjectLayer = isStatic ? ObjLayers::NON_MOVING : ObjLayers::MOVING;
    settings.mMotionType = isStatic ? JPH::EMotionType::Static : JPH::EMotionType::Dynamic;
    
    if (!isStatic) {
        settings.mAllowSleeping = false;
        settings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    }

    if (shape == CollisionShape::Box) {
        auto* mc = entity->GetComponent<MeshComponent>();
        if (mc && mc->HasLocalAABB()) {
            glm::vec3 localMin = mc->GetLocalAABBMin();
            glm::vec3 localMax = mc->GetLocalAABBMax();
            glm::vec3 center = (localMin + localMax) * 0.5f;
            glm::vec3 halfExtents = (localMax - localMin) * 0.5f;

            // Add minimum thickness (especially for floors/walls) to prevent tunnelling
            float hx = std::max(0.01f, std::abs(halfExtents.x * scale.x));
            float hy = std::max(0.01f, std::abs(halfExtents.y * scale.y));
            float hz = std::max(0.01f, std::abs(halfExtents.z * scale.z));
            
            if (isStatic) {
                // For static environment objects, ensure at least 10cm thickness in each dimension
                // to make them "solid" enough for fast-moving dynamic objects.
                hx = std::max(hx, 0.1f);
                hy = std::max(hy, 0.1f);
                hz = std::max(hz, 0.1f);
            }

            auto* boxSettings = new JPH::BoxShapeSettings(JPH::Vec3(hx, hy, hz));
            boxSettings->mConvexRadius = 0.0f;
            
            // Use RotatedTranslatedShape to offset the box geometry so the body's origin matches the entity's origin.
            // This ensures that syncing physics back to the entity (and moving kinematic bodies) works correctly without manual offsets.
            auto* s = new JPH::RotatedTranslatedShapeSettings(JPH::Vec3(center.x * scale.x, center.y * scale.y, center.z * scale.z), JPH::Quat::sIdentity(), boxSettings);
            settings.SetShapeSettings(s);
        } else {
            auto* s = new JPH::BoxShapeSettings(JPH::Vec3(std::max(0.01f, std::abs(scale.x) * 0.5f),
                                                         std::max(0.01f, std::abs(scale.y) * 0.5f),
                                                         std::max(0.01f, std::abs(scale.z) * 0.5f)));
            s->mConvexRadius = 0.0f;
            settings.SetShapeSettings(s);
        }
    } else if (shape == CollisionShape::Sphere) {
        auto* mc = entity->GetComponent<MeshComponent>();
        if (mc && mc->HasLocalAABB()) {
            glm::vec3 localMin = mc->GetLocalAABBMin();
            glm::vec3 localMax = mc->GetLocalAABBMax();
            glm::vec3 halfExtents = (localMax - localMin) * 0.5f;
            float hx = std::abs(halfExtents.x * scale.x);
            float hy = std::abs(halfExtents.y * scale.y);
            float hz = std::abs(halfExtents.z * scale.z);
            float radius = std::max(hx, std::max(hy, hz));
            settings.SetShapeSettings(new JPH::SphereShapeSettings(std::max(0.01f, radius)));
        } else {
            settings.SetShapeSettings(new JPH::SphereShapeSettings(std::max(0.01f, std::abs(scale.x) * 0.5f)));
        }
    } else if (shape == CollisionShape::Capsule) {
        settings.SetShapeSettings(new JPH::CapsuleShapeSettings(std::max(0.01f, std::abs(scale.y) * 0.5f),
                                                               std::max(0.01f, std::abs(scale.x) * 0.5f)));
    } else if (shape == CollisionShape::Mesh) {
        // Static triangle-mesh collider (mesh shapes are static-only). Bistro
        // geometry is heavily GPU-instanced: a mesh authored in a small local
        // space is placed, often many times, via per-instance model matrices.
        // The renderer draws each instance at
        //   worldPos = entityModel * instanceModel * localPos
        // so we bake that same transform into the collider vertices and create
        // the body at the identity pose. Baking the full transform (rather than
        // using the TransformComponent's position/scale alone) is what keeps the
        // collider aligned with the visible geometry; it also scales tiny local
        // triangles up to world size so they survive the degenerate-area filter.
        // A non-instanced mesh is treated as a single identity instance.
        auto* mc = entity->GetComponent<MeshComponent>();
        if (mc) {
            const auto& vertices = mc->GetVertices();
            const auto& indices = mc->GetIndices();

            if (!vertices.empty() && !indices.empty()) {
                const glm::mat4 entityModel = transform ? transform->GetModelMatrix() : glm::mat4(1.0f);

                const auto& instances = mc->GetInstances();
                std::vector<glm::mat4> worldXforms;
                if (instances.empty()) {
                    worldXforms.push_back(entityModel);
                } else {
                    worldXforms.reserve(instances.size());
                    for (const auto& inst : instances) {
                        worldXforms.push_back(entityModel * inst.getModelMatrix());
                    }
                }

                // Cap total triangles so dense instanced foliage (hundreds of
                // copies of a high-poly bush) doesn't build a multi-million-tri
                // shape on the physics thread. Such decorative meshes simply get
                // no collider, which is fine for a walkable environment.
                const size_t triCount = indices.size() / 3;
                constexpr size_t MAX_TOTAL_TRIS = 200000;
                if (triCount * worldXforms.size() <= MAX_TOTAL_TRIS) {
                    JPH::VertexList joltVertices;
                    joltVertices.reserve(vertices.size() * worldXforms.size());
                    JPH::IndexedTriangleList joltTriangles;
                    joltTriangles.reserve(triCount * worldXforms.size());

                    for (const glm::mat4& M : worldXforms) {
                        const uint32_t base = static_cast<uint32_t>(joltVertices.size());
                        for (const auto& v : vertices) {
                            glm::vec3 wp = glm::vec3(M * glm::vec4(v.position, 1.0f));
                            joltVertices.push_back(JPH::Float3(wp.x, wp.y, wp.z));
                        }
                        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                            const uint32_t i0 = base + indices[i];
                            const uint32_t i1 = base + indices[i + 1];
                            const uint32_t i2 = base + indices[i + 2];
                            JPH::Vec3 v0(joltVertices[i0].x, joltVertices[i0].y, joltVertices[i0].z);
                            JPH::Vec3 v1(joltVertices[i1].x, joltVertices[i1].y, joltVertices[i1].z);
                            JPH::Vec3 v2(joltVertices[i2].x, joltVertices[i2].y, joltVertices[i2].z);
                            // Filter out degenerate (zero-area) triangles in world space.
                            if ((v1 - v0).Cross(v2 - v0).LengthSq() > 1e-12f) {
                                joltTriangles.push_back(JPH::IndexedTriangle(i0, i1, i2));
                            }
                        }
                    }

                    if (!joltTriangles.empty()) {
                        settings.SetShapeSettings(new JPH::MeshShapeSettings(joltVertices, joltTriangles));
                        // Vertices are already in world space — the body must sit
                        // at the identity pose so the collider lands on the geometry.
                        settings.mPosition = JPH::RVec3(0.0f, 0.0f, 0.0f);
                        settings.mRotation = JPH::Quat::sIdentity();
                    }
                }
            }
        }
    }

    if (!settings.GetShape()) {
        std::cerr << "PhysicsSystem: Failed to create shape for entity " << entity->GetName() << std::endl;
        return nullptr;
    }

    JPH::BodyID bodyId = physicsWorld->create_body(settings);
    if (bodyId.IsInvalid()) return nullptr;

    auto rb = std::make_unique<ConcreteRigidBody>(entity, physicsWorld, bodyId, isStatic);
    RigidBody* ptr = rb.get();
    
    std::lock_guard<std::mutex> lock(rigidBodiesMutex);
    rigidBodies.push_back(std::move(rb));
    
    return ptr;
}

bool PhysicsSystem::DestroyRigidBody(RigidBody* rigidBody) {
    if (!rigidBody) return false;
    std::lock_guard<std::mutex> lock(rigidBodiesMutex);
    auto it = std::find_if(rigidBodies.begin(), rigidBodies.end(), [rigidBody](const auto& rb) { return rb.get() == rigidBody; });
    if (it != rigidBodies.end()) {
        rigidBodies.erase(it);
        return true;
    }
    return false;
}

void PhysicsSystem::SetGravity(const glm::vec3& _gravity) {
    // Jolt physics world doesn't have a simple SetGravity in our wrapper, 
    // but it's usually handled in JPH::PhysicsSystem.
}

glm::vec3 PhysicsSystem::GetGravity() const {
    return glm::vec3(0.0f, -9.81f, 0.0f);
}

bool PhysicsSystem::Raycast(const glm::vec3& origin, const glm::vec3& direction, float maxDistance,
                           glm::vec3* hitPosition, glm::vec3* hitNormal, Entity** hitEntity) const {
    PhysicsWorld* physicsWorld = GetWorld(this);
    if (!physicsWorld) return false;
    
    JPH::BodyID bodyId;
    float distance;
    glm::vec3 normal;
    if (physicsWorld->raycast(origin, direction, maxDistance, distance, normal, bodyId)) {
        if (hitPosition) *hitPosition = origin + direction * distance;
        if (hitNormal) *hitNormal = normal;
        if (hitEntity) {
            std::lock_guard<std::mutex> lock(rigidBodiesMutex);
            for (const auto& rb : rigidBodies) {
                auto* crb = static_cast<ConcreteRigidBody*>(rb.get());
                if (crb->GetBodyID() == bodyId) {
                    *hitEntity = crb->GetEntity();
                    break;
                }
            }
        }
        return true;
    }
    return false;
}

void PhysicsSystem::EnqueueRigidBodyCreation(Entity* entity, CollisionShape shape, float mass, bool kinematic, float restitution, float friction) {
    std::lock_guard<std::mutex> lk(pendingMutex);
    pendingCreations.push_back({entity, shape, mass, kinematic, restitution, friction});
}

void PhysicsSystem::RegisterStreamingCollider(Entity* entity,
                                              CollisionShape shape,
                                              float mass,
                                              bool kinematic,
                                              float restitution,
                                              float friction,
                                              const glm::vec3& center) {
    std::lock_guard<std::mutex> lk(streamingMutex);
    StreamingCandidate sc;
    sc.entity = entity;
    sc.shape = shape;
    sc.mass = mass;
    sc.kinematic = kinematic;
    sc.restitution = restitution;
    sc.friction = friction;
    sc.center = center;
    sc.active = false;
    entityToStreamingIndex[entity] = streamingCandidates.size();
    streamingCandidates.push_back(sc);
}


void PhysicsSystem::CleanupMarkedBodies() {
    // Handled via DestroyRigidBody
}