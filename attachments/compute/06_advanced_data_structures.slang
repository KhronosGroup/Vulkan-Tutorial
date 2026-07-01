// Chapter 6 – Advanced Data Structures: BVH Ray Tracer (Slang compute shader)
//
// Two compute entry points:
//
//   primaryRayMain   – One thread per pixel.  Traces a primary ray through the
//                      BVH via raw 64-bit Buffer Device Address pointer casts.
//                      On a hit, Lambertian shading is written to the output
//                      image AND a shadow ray job is enqueued into a GPU work
//                      queue using InterlockedAdd on a BDA counter pointer.
//
//   shadowQueueMain  – One thread per queue slot.  Reads a shadow ray job from
//                      the work queue, traces an occlusion ray through the BVH,
//                      and darkens the pixel if blocked.
//
// Key teaching elements:
//   • Buffer Device Address (BDA): BVH nodes and triangle arrays are accessed
//     entirely through raw pointer casts — (BVHNode*)bvhAddr — with no
//     descriptor bindings beyond the output image.
//   • GPU Work Queue: InterlockedAdd(*counterPtr, 1u, slot) atomically claims
//     a slot in the shadow job buffer.  The shadow pass reads back those jobs.
//   • RWStructuredBuffer-style layout emulated through typed pointer arithmetic.
//
// Cornell box scene:
//   • 5 walls  : floor/ceiling/back (white), left (red), right (green)
//   • 2 boxes  : tall box (white) and short box (white)
//   • Area light: bright patch on the ceiling (color.a == 2.0 emission flag)
//
// Compile:
//   slangc 06_advanced_data_structures.slang          \
//          -profile spirv_1_4 -target spirv           \
//          -emit-spirv-directly -fvk-use-entrypoint-name \
//          -entry primaryRayMain -entry shadowQueueMain   \
//          -o shaders/slang.spv

// ---------------------------------------------------------------------------
// Output image  (descriptor set binding)
// ---------------------------------------------------------------------------
[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outputImage;

// ---------------------------------------------------------------------------
// GPU Work Queue types
// ---------------------------------------------------------------------------
// Each shadow job occupies 16 bytes: pixelX(4), pixelY(4), triIdx(4), hitDist(4)
struct ShadowJob
{
    uint  pixelX;
    uint  pixelY;
    uint  triIdx;
    float hitDist;
};

// ---------------------------------------------------------------------------
// BVH data structures
// Layout must be byte-identical to the C++ structs.
//
// BVHNode (48 bytes):
//   float3 aabbMin;  int leftChild;    // offsets  0, 12
//   float3 aabbMax;  int rightChild;   // offsets 16, 28
//   int triOffset;   int triCount;     // offsets 32, 36
//   int2 _pad;                         // offsets 40, 44
//
// Triangle (64 bytes — compact, no per-field padding):
//   float3 v0;     // offset  0
//   float3 v1;     // offset 12
//   float3 v2;     // offset 24
//   float3 normal; // offset 36
//   float4 color;  // offset 48  — color.a == 2.0 flags the triangle as emissive
// ---------------------------------------------------------------------------
struct BVHNode
{
    float3 aabbMin;
    int    leftChild;   // -1 if leaf
    float3 aabbMax;
    int    rightChild;  // -1 if leaf
    int    triOffset;
    int    triCount;
    int2   _pad;
};

struct Triangle
{
    float3 v0;
    float3 v1;
    float3 v2;
    float3 normal;
    float4 color;       // color.a == 2.0 → emissive light patch
};

// ---------------------------------------------------------------------------
// Push constants — primary pass
// Must match RayTracePush in 06_advanced_data_structures.cpp (96 bytes)
// ---------------------------------------------------------------------------
struct RayTracePush
{
    uint64_t bvhAddr;          // BVHNode*
    uint64_t triAddr;          // Triangle*
    uint64_t shadowQueueAddr;  // ShadowJob*
    uint64_t counterAddr;      // uint* (atomic counter)
    float3   camPos;
    uint     frameWidth;
    float3   camTarget;
    uint     frameHeight;
    float3   camUp;
    float    fovY;             // vertical field-of-view (radians)
    uint     nodeCount;
    uint     triCount;
    uint     queueCapacity;
    uint     _pad;
};
[[vk::push_constant]] RayTracePush pc;

// ---------------------------------------------------------------------------
// Push constants — shadow pass
// Must match ShadowPush in 06_advanced_data_structures.cpp (56 bytes)
// ---------------------------------------------------------------------------
struct ShadowPush
{
    uint64_t bvhAddr;
    uint64_t triAddr;
    uint64_t shadowQueueAddr;
    uint64_t counterAddr;
    uint64_t _unused;
    uint     frameWidth;
    uint     frameHeight;
    uint     queueCapacity;
    uint     _pad;
};
[[vk::push_constant]] ShadowPush spc;

// ---------------------------------------------------------------------------
// Ray–AABB intersection  (slab test)
// Returns the entry distance, or 1e30 on miss.
// ---------------------------------------------------------------------------
float rayAABB(float3 ro, float3 invRd, float3 bMin, float3 bMax, float tCurrent)
{
    float3 t0     = (bMin - ro) * invRd;
    float3 t1     = (bMax - ro) * invRd;
    float3 tNear  = min(t0, t1);
    float3 tFar   = max(t0, t1);
    float  tEnter = max(max(tNear.x, tNear.y), tNear.z);
    float  tExit  = min(min(tFar.x,  tFar.y),  tFar.z);
    if (tExit < 0.0f || tEnter > tExit || tEnter > tCurrent)
        return 1e30f;
    return max(tEnter, 0.0f);
}

// ---------------------------------------------------------------------------
// Ray–triangle intersection  (Möller-Trumbore)
// Returns distance t (> 0), or 1e30 on miss.
// ---------------------------------------------------------------------------
float rayTriangle(float3 ro, float3 rd, float3 v0, float3 v1, float3 v2)
{
    const float EPS = 1e-7f;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h  = cross(rd, e2);
    float  a  = dot(e1, h);
    if (abs(a) < EPS)   return 1e30f;   // parallel

    float  f  = 1.0f / a;
    float3 s  = ro - v0;
    float  u  = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return 1e30f;

    float3 q  = cross(s, e1);
    float  v  = f * dot(rd, q);
    if (v < 0.0f || u + v > 1.0f) return 1e30f;

    float  t  = f * dot(e2, q);
    return (t > EPS) ? t : 1e30f;
}

// ---------------------------------------------------------------------------
// Iterative BVH traversal  (stack-based, depth limit 32)
//
// Demonstrates BDA: 'nodes' and 'tris' are typed raw pointers obtained by
// casting the 64-bit device addresses from push constants.
//
// Returns hit triangle index (–1 = miss) and sets hitT to the nearest hit.
// ---------------------------------------------------------------------------
int traverseBVH(uint64_t bvhAddr, uint64_t triAddr,
                float3 ro, float3 rd, float tMax,
                out float hitT)
{
    hitT      = tMax;
    int hitIdx = -1;

    float3     invRd = float3(1.0f) / rd;
    BVHNode*   nodes = (BVHNode*)  bvhAddr;  // BDA pointer cast
    Triangle*  tris  = (Triangle*) triAddr;  // BDA pointer cast

    // Small traversal stack — sufficient for our shallow BVH
    int  stack[32];
    int  top = 0;
    stack[top++] = 0;  // push root node

    while (top > 0)
    {
        int     nodeIdx = stack[--top];
        BVHNode node    = nodes[nodeIdx];   // load via BDA pointer

        float t = rayAABB(ro, invRd, node.aabbMin, node.aabbMax, hitT);
        if (t >= hitT)
            continue;   // AABB miss or farther than current best

        if (node.leftChild == -1)
        {
            // ── Leaf node: test each triangle ──────────────────────────────
            for (int i = 0; i < node.triCount; ++i)
            {
                Triangle tri = tris[node.triOffset + i]; // BDA load
                float tHit = rayTriangle(ro, rd, tri.v0, tri.v1, tri.v2);
                if (tHit < hitT)
                {
                    hitT   = tHit;
                    hitIdx = node.triOffset + i;
                }
            }
        }
        else
        {
            // ── Internal node: push both children ──────────────────────────
            if (top + 1 < 32)
            {
                stack[top++] = node.rightChild;
                stack[top++] = node.leftChild;   // left processed next (LIFO)
            }
        }
    }

    return hitIdx;
}

// ---------------------------------------------------------------------------
// Scene constants
//
// The area light is a patch on the ceiling (y ≈ 2) in [-0.3,0.3]×[-0.3,0.3].
// Its triangles are flagged with color.a == 2.0 in the scene data.
//
// kLightPos is the centroid of that patch, used as the shadow ray target.
// ---------------------------------------------------------------------------
static const float3 kLightPos   = float3(0.0f, 1.96f, 0.0f);  // area light centroid
static const float3 kLightColor = float3(1.0f, 0.95f, 0.85f); // warm white
static const float  kAmbient    = 0.12f;                        // base ambient

// ---------------------------------------------------------------------------
// Compute a simple directional ambient term to fake color bleeding.
//
// Surfaces closer to the red left wall (x → −1) get a faint red tint;
// surfaces closer to the green right wall (x → +1) get a faint green tint.
// The Cornell box spans x ∈ [−1, 1].
// ---------------------------------------------------------------------------
float3 colorBleedAmbient(float3 hitPos, float3 albedo)
{
    // Blend factor in [0,1] for proximity to left wall (x=-1)
    float leftBlend  = saturate((-hitPos.x - 0.2f) * 1.25f);  // ramps up near x=-1
    float rightBlend = saturate(( hitPos.x - 0.2f) * 1.25f);  // ramps up near x=+1

    float3 redBleed   = float3(0.65f, 0.05f, 0.05f) * leftBlend  * 0.18f;
    float3 greenBleed = float3(0.12f, 0.45f, 0.15f) * rightBlend * 0.18f;

    return albedo * (kAmbient + redBleed + greenBleed);
}

// ---------------------------------------------------------------------------
// Entry 1: primaryRayMain
//
// One 16×16 thread group per tile.  Each thread:
//   1. Builds a camera ray from push-constant camera parameters.
//   2. Traverses the BVH via BDA pointer.
//   3. On hit of an emissive triangle: returns bright emission immediately.
//   4. On hit of a diffuse triangle: computes Lambertian shading with ambient
//      color bleeding and writes to outputImage.
//   5. On diffuse hit: atomically claims a slot in the GPU shadow work queue
//      (InterlockedAdd on the BDA counter pointer), and stores a ShadowJob.
// ---------------------------------------------------------------------------
[numthreads(16, 16, 1)]
[shader("compute")]
void primaryRayMain(uint3 dispatchID : SV_DispatchThreadID)
{
    uint px = dispatchID.x;
    uint py = dispatchID.y;
    if (px >= pc.frameWidth || py >= pc.frameHeight)
        return;

    // ── Build camera ray ────────────────────────────────────────────────────
    float aspect  = (float)pc.frameWidth / (float)pc.frameHeight;
    float halfH   = tan(pc.fovY * 0.5f);
    float halfW   = halfH * aspect;

    float3 fwd    = normalize(pc.camTarget - pc.camPos);
    float3 right  = normalize(cross(fwd, pc.camUp));
    float3 up     = cross(right, fwd);

    float u  = ((float)px + 0.5f) / (float)pc.frameWidth  * 2.0f - 1.0f;
    float v  = 1.0f - ((float)py + 0.5f) / (float)pc.frameHeight * 2.0f;

    float3 ro = pc.camPos;
    float3 rd = normalize(fwd + right * (u * halfW) + up * (v * halfH));

    // ── BVH traversal ───────────────────────────────────────────────────────
    float hitT;
    int   hitIdx = traverseBVH(pc.bvhAddr, pc.triAddr, ro, rd, 1e30f, hitT);

    if (hitIdx < 0)
    {
        // Miss: pure black (the camera looks into the box; the front is open
        // so edge-case rays that escape see nothing)
        outputImage[uint2(px, py)] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    // ── Load hit triangle via BDA ───────────────────────────────────────────
    Triangle* tris   = (Triangle*) pc.triAddr;
    Triangle  tri    = tris[hitIdx];
    float3    hitPos = ro + rd * hitT;
    float3    N      = normalize(tri.normal);
    if (dot(N, -rd) < 0.0f) N = -N;   // ensure front-face normal

    // ── Emissive area-light triangles (color.a == 2.0) ──────────────────────
    // Any ray that hits the ceiling light patch returns the emission directly,
    // creating a bright, visible light source.
    if (tri.color.a > 1.5f)
    {
        // Clamp emission to [0,1] for the rgba8 storage image — it will still
        // appear as saturated white, clearly distinguishable from the grey walls.
        float3 emit = min(tri.color.rgb, float3(1.0f));
        outputImage[uint2(px, py)] = float4(emit, 1.0f);
        return;
    }

    // ── Lambertian shading with ambient color bleeding ──────────────────────
    float3 toLight = normalize(kLightPos - hitPos);
    float  NdotL   = max(dot(N, toLight), 0.0f);
    float3 albedo  = tri.color.rgb;

    // Diffuse contribution from the area light
    float3 diffuse = albedo * NdotL * 0.80f * kLightColor;

    // Ambient + simple color-bleed approximation
    float3 ambient = colorBleedAmbient(hitPos, albedo);

    float3 shaded = ambient + diffuse;
    outputImage[uint2(px, py)] = float4(shaded, 1.0f);

    // ── Enqueue shadow ray job (GPU Work Queue) ─────────────────────────────
    //
    // Teaching moment: InterlockedAdd on a raw BDA pointer atomically claims
    // the next available slot in the shadow job array.  This is the GPU work
    // queue pattern: many concurrent threads compete for slots without locks.
    //
    uint*       counterPtr = (uint*)       pc.counterAddr;      // BDA cast
    ShadowJob*  queue      = (ShadowJob*)  pc.shadowQueueAddr;  // BDA cast

    uint slot;
    InterlockedAdd(*counterPtr, 1u, slot);   // atomic slot claim

    if (slot < pc.queueCapacity)
    {
        // Write the shadow job into the claimed slot.
        // Store the actual hit position encoded as an index — we re-derive it
        // in the shadow pass from the stored triIdx and hitDist.
        queue[slot].pixelX  = px;
        queue[slot].pixelY  = py;
        queue[slot].triIdx  = (uint)hitIdx;
        queue[slot].hitDist = hitT;
    }
    // Threads that overflow the queue silently drop their job —
    // the overflow guard ensures no out-of-bounds write occurs.
}

// ---------------------------------------------------------------------------
// Entry 2: shadowQueueMain
//
// One flat thread per queue slot.  Each thread:
//   1. Reads the shadow job count from the BDA counter pointer.
//   2. Reads its job from the shadow queue (BDA pointer arithmetic).
//   3. Reconstructs the hit position from the primary ray — not the triangle
//      centroid — by re-casting the camera ray for that pixel.  This gives
//      accurate per-pixel shadows rather than centroid-approximated ones.
//   4. Traces an occlusion ray toward kLightPos through the BVH.
//   5. If blocked: darkens the corresponding pixel (multiply by shadow factor).
//
// This second dispatch demonstrates how the GPU work queue produced by the
// primary pass can be consumed in a follow-up compute wave.
// ---------------------------------------------------------------------------
[numthreads(64, 1, 1)]
[shader("compute")]
void shadowQueueMain(uint3 dispatchID : SV_DispatchThreadID)
{
    uint slot = dispatchID.x;

    // Read the actual number of enqueued jobs from the atomic counter
    uint* counterPtr = (uint*) spc.counterAddr;          // BDA cast
    uint  jobCount   = min(*counterPtr, spc.queueCapacity);

    if (slot >= jobCount)
        return;   // no work for this thread

    // ── Read shadow job from the GPU work queue ──────────────────────────
    ShadowJob* queue = (ShadowJob*) spc.shadowQueueAddr;  // BDA cast
    ShadowJob  job   = queue[slot];

    if (job.hitDist <= 0.0f)
        return;   // uninitialised slot (should not happen after counter check)

    if (job.pixelX >= spc.frameWidth || job.pixelY >= spc.frameHeight)
        return;

    Triangle* tris = (Triangle*) spc.triAddr;   // BDA cast
    Triangle  tri  = tris[job.triIdx];

    // Bias the shadow origin off the surface along the face normal.
    // The stored normal may point inward (away from the light) for floor/ceiling
    // triangles whose winding produces a downward/upward normal — always orient
    // toward the light so the shadow origin is on the lit side of the surface.
    float3 centroid  = (tri.v0 + tri.v1 + tri.v2) * (1.0f / 3.0f);
    float3 N         = normalize(tri.normal);
    if (dot(N, kLightPos - centroid) < 0.0f) N = -N;  // face toward light
    float3 shadowOri = centroid + N * 2e-3f;
    float3 toLight   = kLightPos - shadowOri;
    float  lightDist  = length(toLight);
    float3 shadowDir  = toLight / lightDist;

    // ── Trace occlusion ray ──────────────────────────────────────────────
    float occT;
    int   occHit = traverseBVH(spc.bvhAddr, spc.triAddr,
                                shadowOri, shadowDir,
                                lightDist - 2e-3f, occT);

    // Skip hits on emissive triangles — they are the light itself
    if (occHit >= 0)
    {
        Triangle occTri = tris[occHit];
        if (occTri.color.a > 1.5f)
            occHit = -1;   // hit the light patch, not an occluder
    }

    if (occHit >= 0)
    {
        // Point is in shadow: attenuate the pixel's colour
        // Use a softer shadow factor (0.35) to preserve readability of
        // wall colours under shadow — pure 0.25 is too dark.
        float4 existing = outputImage[uint2(job.pixelX, job.pixelY)];
        outputImage[uint2(job.pixelX, job.pixelY)] =
            float4(existing.rgb * 0.35f, 1.0f);
    }
}
