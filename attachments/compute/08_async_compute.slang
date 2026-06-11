// Chapter 8 – Asynchronous Compute: Cloth Physics Simulation
//
// This file contains all three shader entry points compiled into one SPIR-V
// binary.  The async compute queue runs cloth Verlet integration and spring-
// constraint solving every frame.
//
// Entry points
//   constraintPass  – compute: Verlet integration + spring constraints + normals
//   vertMain        – vertex:  transform cloth positions, pass through normal
//   fragMain        – fragment: diffuse shading for cloth and sphere
//
// IMPORTANT – why a single workgroup:
//   Iterative constraint solving needs ALL vertices synchronised between each
//   relaxation step.  A barrier in Vulkan compute (GroupMemoryBarrierWithGroup-
//   Sync) only synchronises threads WITHIN ONE workgroup – never across the
//   whole dispatch.  We therefore size the cloth so the entire 32×32 = 1024
//   grid lives in a single 1024-thread workgroup, hold the working set in
//   groupshared memory, and use red-black (checkerboard) Gauss-Seidel so that
//   no two simultaneously-processed edges ever touch the same vertex.  This is
//   completely race free and converges far faster than the naive parallel
//   Jacobi sweep.

// ==========================================================================
// Shared constants
// ==========================================================================

static const uint CLOTH_W  = 32u;
static const uint CLOTH_H  = 32u;
static const uint CLOTH_N  = CLOTH_W * CLOTH_H;   // 1024 vertices = 1 workgroup

// Rest length between adjacent grid nodes (cloth spans 2 units in X and Y)
static const float REST_X  = 2.0 / float(CLOTH_W - 1);
static const float REST_Y  = 2.0 / float(CLOTH_H - 1);

// Physics constants
static const float GRAVITY = -9.8;
static const float DAMPING = 0.99;

// ==========================================================================
// Buffers (set 0)
//   0 – ClothUBO          (uniforms)
//   1 – positions         (RW float4: xyz = world pos, w = inverse mass)
//   2 – prevPositions     (RW float4: previous-frame positions for Verlet)
//   3 – normals           (RW float4: xyz = smooth per-vertex normal)
// ==========================================================================

struct ClothUBO {
    float  deltaTime;      // seconds since last frame
    uint   iterCount;      // constraint iterations per frame
    float  time;           // total elapsed time for animation
    float  pad;
    float3 sphereCenter;   // animated sphere centre in world space
    float  sphereRadius;   // sphere collision radius
};

[[vk::binding(0, 0)]] ConstantBuffer<ClothUBO>   ubo;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> positions;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> prevPositions;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> normals;

// Working copy of all cloth positions for the duration of one dispatch.
// xyz = position, w = inverse mass (0 = pinned).  16 KB total.
groupshared float4 sPos[CLOTH_N];

// ==========================================================================
// Helper: push a point outside the collision sphere (with a small skin).
// ==========================================================================
float3 resolveSphereColl(float3 p, float3 sc, float sr)
{
    float3 d = p - sc;
    float  r = length(d);
    if (r < sr) {
        p = sc + (r > 1e-6 ? d / r : float3(0, 1, 0)) * sr;
    }
    return p;
}

// ==========================================================================
// Helper: satisfy one distance constraint between shared vertices a and b.
// Only ever called by the thread that OWNS vertex a, and red-black colouring
// guarantees b is not being written by any other active thread this phase, so
// the two stores are race free.
// ==========================================================================
void satisfy(uint a, uint b, float rest)
{
    float4 A = sPos[a];
    float4 B = sPos[b];
    float  wa = A.w;          // inverse mass
    float  wb = B.w;
    float  wsum = wa + wb;
    if (wsum <= 0.0) return;  // both pinned

    float3 d = B.xyz - A.xyz;
    float  L = length(d);
    if (L < 1e-6) return;

    float  diff = (L - rest) / L;
    float3 corr = d * diff;
    sPos[a] = float4(A.xyz + corr * (wa / wsum), wa);
    sPos[b] = float4(B.xyz - corr * (wb / wsum), wb);
}

// ==========================================================================
// COMPUTE PASS – Verlet integration + constraints + smooth normals
// One dispatch = one workgroup of CLOTH_N threads.
// ==========================================================================

[shader("compute")]
[numthreads(CLOTH_N, 1, 1)]
void constraintPass(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= CLOTH_N) return;

    uint row = idx / CLOTH_W;
    uint col = idx % CLOTH_W;

    float3 sc = ubo.sphereCenter;
    float  sr = ubo.sphereRadius;
    float  dt = ubo.deltaTime;

    // -----------------------------------------------------------------------
    // Step 1: Verlet integration  →  groupshared working set
    // -----------------------------------------------------------------------
    float4 p4   = positions[idx];
    float4 pv   = prevPositions[idx];
    float  invM = p4.w;                 // 0 = pinned, 1 = free
    float3 pos  = p4.xyz;
    float3 prev = pv.xyz;

    float3 vel  = (pos - prev) * DAMPING;
    float3 next = pos + vel + float3(0.0, GRAVITY, 0.0) * invM * dt * dt;
    if (invM == 0.0) next = pos;        // keep pinned vertex anchored
    next = resolveSphereColl(next, sc, sr);

    // prev for the next frame is the current (pre-integration) position
    prevPositions[idx] = float4(pos, invM);
    sPos[idx]          = float4(next, invM);
    GroupMemoryBarrierWithGroupSync();

    // -----------------------------------------------------------------------
    // Step 2: red-black Gauss-Seidel constraint relaxation
    // Four colour phases per iteration so concurrent edges never share a vertex:
    //   horizontal-even, horizontal-odd, vertical-even, vertical-odd
    // -----------------------------------------------------------------------
    for (uint it = 0u; it < ubo.iterCount; ++it)
    {
        if ((col & 1u) == 0u && col + 1u < CLOTH_W) satisfy(idx, idx + 1u, REST_X);
        GroupMemoryBarrierWithGroupSync();

        if ((col & 1u) == 1u && col + 1u < CLOTH_W) satisfy(idx, idx + 1u, REST_X);
        GroupMemoryBarrierWithGroupSync();

        if ((row & 1u) == 0u && row + 1u < CLOTH_H) satisfy(idx, idx + CLOTH_W, REST_Y);
        GroupMemoryBarrierWithGroupSync();

        if ((row & 1u) == 1u && row + 1u < CLOTH_H) satisfy(idx, idx + CLOTH_W, REST_Y);
        GroupMemoryBarrierWithGroupSync();

        // Sphere collision per vertex (skip pinned)
        float4 me = sPos[idx];
        if (me.w != 0.0) {
            me.xyz   = resolveSphereColl(me.xyz, sc, sr);
            sPos[idx] = me;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // -----------------------------------------------------------------------
    // Step 3: smooth per-vertex normal from neighbouring shared positions
    // -----------------------------------------------------------------------
    uint l = (col > 0u)             ? idx - 1u     : idx;
    uint r = (col + 1u < CLOTH_W)   ? idx + 1u     : idx;
    uint u = (row > 0u)             ? idx - CLOTH_W : idx;
    uint d = (row + 1u < CLOTH_H)   ? idx + CLOTH_W : idx;

    float3 dx  = sPos[r].xyz - sPos[l].xyz;   // tangent along grid columns (+X)
    float3 dy  = sPos[d].xyz - sPos[u].xyz;   // tangent along grid rows    (+Z)
    float3 nrm = cross(dy, dx);               // faces +Y for the flat horizontal rest pose
    float  nl  = length(nrm);
    nrm = (nl > 1e-6) ? nrm / nl : float3(0.0, 1.0, 0.0);

    // -----------------------------------------------------------------------
    // Step 4: write results back to device memory
    // -----------------------------------------------------------------------
    positions[idx] = sPos[idx];
    normals[idx]   = float4(nrm, 0.0);
}

// ==========================================================================
// GRAPHICS PASS – render cloth + sphere as lit triangle meshes
// ==========================================================================

struct ClothPush {
    float4x4 mvp;
    float4x4 normalMatrix;
    float4   lightDir;     // xyz = light direction (world space), w = renderSphere flag
};

[[vk::push_constant]] ClothPush push;

// Vertex input
//   location 0 – float4 posInvM  (xyz = position, w = inverse mass)
//   location 1 – float2 uv       (cloth (u,v) coordinates)
//   location 2 – float3 normal   (smooth world-space normal)
struct VSIn {
    [[vk::location(0)]] float4 posInvM;
    [[vk::location(1)]] float2 uv;
    [[vk::location(2)]] float3 normal;
};

struct VSOut {
    float4 clipPos  : SV_Position;
    float3 worldPos : TEXCOORD0;
    float2 uv       : TEXCOORD1;
    float  invMass  : TEXCOORD2;
    float3 normal   : TEXCOORD3;
};

[shader("vertex")]
VSOut vertMain(VSIn vin)
{
    VSOut vout;
    float3 pos    = vin.posInvM.xyz;
    vout.worldPos = pos;
    vout.clipPos  = mul(push.mvp, float4(pos, 1.0));
    vout.uv       = vin.uv;
    vout.invMass  = vin.posInvM.w;
    vout.normal   = vin.normal;
    return vout;
}

// Fragment shader – diffuse cloth/sphere shading
struct PSIn {
    float4 clipPos  : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float2 uv       : TEXCOORD1;
    float  invMass  : TEXCOORD2;
    float3 normal   : TEXCOORD3;
    bool   isFront  : SV_IsFrontFace;
};

[shader("fragment")]
float4 fragMain(PSIn pin) : SV_TARGET
{
    // Smooth interpolated normal (two-sided lighting for the cloth).
    float3 N = normalize(pin.normal);
    if (!pin.isFront) N = -N;

    float3 L       = normalize(push.lightDir.xyz);
    float  NdotL   = max(dot(N, L), 0.0);
    float  ambient = 0.2;
    float  diff    = ambient + (1.0 - ambient) * NdotL;

    float3 color;
    if (push.lightDir.w > 0.5) {
        // Sphere: warm grey with a soft specular sheen
        float3 V = normalize(-pin.worldPos);     // approx view dir
        float3 H = normalize(L + V);
        float  spec = pow(max(dot(N, H), 0.0), 48.0) * 0.3;
        color = float3(0.78, 0.74, 0.70) * diff + spec;
    } else {
        // Cloth: cream / teal checker pattern based on UV
        float2 uvScaled = pin.uv * 8.0;
        int2   cell     = int2(floor(uvScaled));
        float  checker  = float((cell.x + cell.y) & 1);
        float3 colorA   = float3(0.90, 0.84, 0.72);  // cream
        float3 colorB   = float3(0.20, 0.50, 0.60);  // teal
        color = lerp(colorA, colorB, checker) * diff;

        // Pinned corners get a red tint
        float pinned = (pin.invMass == 0.0) ? 1.0 : 0.0;
        color = lerp(color, float3(0.9, 0.2, 0.1), pinned * 0.6);
    }

    return float4(color, 1.0);
}
