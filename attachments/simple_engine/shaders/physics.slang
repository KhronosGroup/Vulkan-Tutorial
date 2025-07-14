// Compute shader for physics simulation
// This shader processes rigid body physics data to simulate physical interactions

// Workgroup size
[[vk::compute_shader_input(local_size_x = 64)]]

// Physics data structure
struct PhysicsData {
    float4 position;        // xyz = position, w = inverse mass
    float4 rotation;        // quaternion
    float4 linearVelocity;  // xyz = velocity, w = restitution
    float4 angularVelocity; // xyz = angular velocity, w = friction
    float4 force;           // xyz = force, w = is kinematic (0 or 1)
    float4 torque;          // xyz = torque, w = use gravity (0 or 1)
    float4 colliderData;    // type-specific data (e.g., radius for spheres)
    float4 colliderData2;   // additional collider data (e.g., box half extents)
};

// Collision data structure
struct CollisionData {
    uint bodyA;
    uint bodyB;
    float4 contactNormal;   // xyz = normal, w = penetration depth
    float4 contactPoint;    // xyz = contact point, w = unused
};

// Input/output buffer bindings
[[vk::binding(0, 0)]] RWStructuredBuffer<PhysicsData> physicsBuffer;  // Physics data
[[vk::binding(1, 0)]] RWStructuredBuffer<CollisionData> collisionBuffer; // Collision data
[[vk::binding(2, 0)]] RWStructuredBuffer<uint2> pairBuffer; // Potential collision pairs
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> counterBuffer; // [0] = pair count, [1] = collision count

// Parameters for physics simulation
[[vk::binding(4, 0)]] ConstantBuffer<PhysicsParams> params;

struct PhysicsParams {
    float deltaTime;        // Time step
    float3 gravity;         // Gravity vector
    uint numBodies;         // Number of rigid bodies
    uint maxCollisions;     // Maximum number of collisions
};

// Quaternion multiplication
float4 quatMul(float4 q1, float4 q2) {
    return float4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// Quaternion normalization
float4 quatNormalize(float4 q) {
    float len = length(q);
    if (len > 0.0001) {
        return q / len;
    }
    return float4(0, 0, 0, 1);
}

// Integration shader - updates positions and velocities
[shader("compute")]
void IntegrateCS(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;

    // Check if this thread is within the number of bodies
    if (index >= params.numBodies) {
        return;
    }

    // Get physics data for this body
    PhysicsData body = physicsBuffer[index];

    // Skip kinematic bodies
    if (body.force.w > 0.5) {
        return;
    }

    // Apply gravity if enabled
    if (body.torque.w > 0.5) {
        body.force.xyz += params.gravity / body.position.w;
    }

    // Integrate forces
    body.linearVelocity.xyz += body.force.xyz * body.position.w * params.deltaTime;
    body.angularVelocity.xyz += body.torque.xyz * params.deltaTime; // Simplified, should use inertia tensor

    // Apply damping
    const float linearDamping = 0.01;
    const float angularDamping = 0.01;
    body.linearVelocity.xyz *= (1.0 - linearDamping);
    body.angularVelocity.xyz *= (1.0 - angularDamping);

    // Integrate velocities
    body.position.xyz += body.linearVelocity.xyz * params.deltaTime;

    // Update rotation
    float4 angularVelocityQuat = float4(body.angularVelocity.xyz * 0.5, 0.0);
    float4 rotationDelta = quatMul(angularVelocityQuat, body.rotation);
    body.rotation = quatNormalize(body.rotation + rotationDelta * params.deltaTime);

    // Write updated data back to buffer
    physicsBuffer[index] = body;
}

// Compute AABB for a body
void computeAABB(PhysicsData body, out float3 min, out float3 max) {
    // Default to a small AABB
    min = body.position.xyz - float3(0.1, 0.1, 0.1);
    max = body.position.xyz + float3(0.1, 0.1, 0.1);

    // Check collider type
    int colliderType = int(body.colliderData.w);

    if (colliderType == 0) { // Sphere
        float radius = body.colliderData.x;
        float3 center = body.position.xyz + body.colliderData2.xyz;
        min = center - float3(radius, radius, radius);
        max = center + float3(radius, radius, radius);
    }
    else if (colliderType == 1) { // Box
        float3 halfExtents = body.colliderData.xyz;
        float3 center = body.position.xyz + body.colliderData2.xyz;
        // This is simplified - should account for rotation
        min = center - halfExtents;
        max = center + halfExtents;
    }
}

// Check if two AABBs overlap
bool aabbOverlap(float3 minA, float3 maxA, float3 minB, float3 maxB) {
    return all(minA < maxB) && all(minB < maxA);
}

// Broad phase collision detection - identifies potential collision pairs
[shader("compute")]
void BroadPhaseCS(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;

    // Calculate total number of pairs
    uint numPairs = (params.numBodies * (params.numBodies - 1)) / 2;

    if (index >= numPairs) {
        return;
    }

    // Convert linear index to pair indices (i, j) where i < j
    uint i = 0;
    uint j = 0;

    // This is a mathematical formula to convert a linear index to a pair of indices
    uint row = uint(floor(sqrt(float(2 * index + 0.25)) - 0.5));
    i = row;
    j = index - (row * (row + 1)) / 2;

    // Ensure j > i
    j += i + 1;

    // Get physics data for both bodies
    PhysicsData bodyA = physicsBuffer[i];
    PhysicsData bodyB = physicsBuffer[j];

    // Skip if both bodies are kinematic
    if (bodyA.force.w > 0.5 && bodyB.force.w > 0.5) {
        return;
    }

    // Skip if either body doesn't have a collider
    if (bodyA.colliderData.w < 0 || bodyB.colliderData.w < 0) {
        return;
    }

    // Compute AABBs
    float3 minA, maxA, minB, maxB;
    computeAABB(bodyA, minA, maxA);
    computeAABB(bodyB, minB, maxB);

    // Check for AABB overlap
    if (aabbOverlap(minA, maxA, minB, maxB)) {
        // Add to potential collision pairs
        uint pairIndex;
        InterlockedAdd(counterBuffer[0], 1, pairIndex);

        if (pairIndex < params.maxCollisions) {
            pairBuffer[pairIndex] = uint2(i, j);
        }
    }
}

// Narrow phase collision detection - detailed collision detection for potential pairs
[shader("compute")]
void NarrowPhaseCS(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;

    // Check if this thread is within the number of potential pairs
    uint numPairs = counterBuffer[0];
    if (index >= numPairs || index >= params.maxCollisions) {
        return;
    }

    // Get the pair of bodies
    uint2 pair = pairBuffer[index];
    uint bodyIndexA = pair.x;
    uint bodyIndexB = pair.y;

    PhysicsData bodyA = physicsBuffer[bodyIndexA];
    PhysicsData bodyB = physicsBuffer[bodyIndexB];

    // Determine collision shapes
    int shapeA = int(bodyA.colliderData.w);
    int shapeB = int(bodyB.colliderData.w);

    // Only handle sphere-sphere collisions for simplicity
    if (shapeA == 0 && shapeB == 0) { // Both are spheres
        float radiusA = bodyA.colliderData.x;
        float radiusB = bodyB.colliderData.x;

        float3 posA = bodyA.position.xyz + bodyA.colliderData2.xyz;
        float3 posB = bodyB.position.xyz + bodyB.colliderData2.xyz;

        float3 direction = posB - posA;
        float distance = length(direction);
        float minDistance = radiusA + radiusB;

        if (distance < minDistance) {
            // Collision detected
            uint collisionIndex;
            InterlockedAdd(counterBuffer[1], 1, collisionIndex);

            if (collisionIndex < params.maxCollisions) {
                // Normalize direction
                float3 normal = direction / max(distance, 0.0001);

                // Create collision data
                CollisionData collision;
                collision.bodyA = bodyIndexA;
                collision.bodyB = bodyIndexB;
                collision.contactNormal = float4(normal, minDistance - distance); // penetration depth
                collision.contactPoint = float4(posA + normal * radiusA, 0);

                // Store collision data
                collisionBuffer[collisionIndex] = collision;
            }
        }
    }
    // Add other collision types here (box-box, sphere-box, etc.)
}

// Collision resolution - resolves detected collisions
[shader("compute")]
void ResolveCS(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;

    // Check if this thread is within the number of collisions
    uint numCollisions = counterBuffer[1];
    if (index >= numCollisions || index >= params.maxCollisions) {
        return;
    }

    // Get collision data
    CollisionData collision = collisionBuffer[index];

    // Get the bodies involved in the collision
    PhysicsData bodyA = physicsBuffer[collision.bodyA];
    PhysicsData bodyB = physicsBuffer[collision.bodyB];

    // Skip if both bodies are kinematic
    if (bodyA.force.w > 0.5 && bodyB.force.w > 0.5) {
        return;
    }

    // Calculate relative velocity
    float3 relativeVelocity = bodyB.linearVelocity.xyz - bodyA.linearVelocity.xyz;

    // Calculate velocity along normal
    float velocityAlongNormal = dot(relativeVelocity, collision.contactNormal.xyz);

    // Don't resolve if velocities are separating
    if (velocityAlongNormal > 0) {
        return;
    }

    // Calculate restitution (bounciness)
    float restitution = min(bodyA.linearVelocity.w, bodyB.linearVelocity.w);

    // Calculate impulse scalar
    float j = -(1.0 + restitution) * velocityAlongNormal;
    j /= bodyA.position.w + bodyB.position.w;

    // Apply impulse
    float3 impulse = collision.contactNormal.xyz * j;

    // Update velocities
    if (bodyA.force.w < 0.5) { // If not kinematic
        bodyA.linearVelocity.xyz -= impulse * bodyA.position.w;
        physicsBuffer[collision.bodyA] = bodyA;
    }

    if (bodyB.force.w < 0.5) { // If not kinematic
        bodyB.linearVelocity.xyz += impulse * bodyB.position.w;
        physicsBuffer[collision.bodyB] = bodyB;
    }

    // Position correction to prevent sinking
    const float percent = 0.2; // usually 20% to 80%
    const float slop = 0.01; // small penetration allowed
    float3 correction = max(collision.contactNormal.w - slop, 0.0) * percent * collision.contactNormal.xyz / (bodyA.position.w + bodyB.position.w);

    if (bodyA.force.w < 0.5) { // If not kinematic
        bodyA.position.xyz -= correction * bodyA.position.w;
        physicsBuffer[collision.bodyA] = bodyA;
    }

    if (bodyB.force.w < 0.5) { // If not kinematic
        bodyB.position.xyz += correction * bodyB.position.w;
        physicsBuffer[collision.bodyB] = bodyB;
    }
}
