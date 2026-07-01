// Chapter 7 – GPU-Driven Pipelines: LOD Asteroid Field
// Three entry points:
//   cullMain  – compute: frustum-cull 1024 asteroids, LOD-select, write per-LOD indirect draw commands
//   vertMain  – vertex:  transform LOD mesh vertex by per-instance asteroid data (pos/rot/size)
//   fragMain  – fragment: Phong shading with a distant sun

// ---------------------------------------------------------------------------
// Shared GPU structures  (CPU layout in 07_gpu_driven_pipelines.cpp must match)
// ---------------------------------------------------------------------------

// One asteroid instance – uploaded once from CPU, never modified on GPU.
struct Asteroid
{
    float3 position;     // world-space centre of the asteroid
    float  size;         // radius scale (0.5 – 2.0 units)
    float3 rotAxis;      // normalised rotation axis
    float  orbitSpeed;   // radians/second orbit around Y
    float3 color;        // grey + slight tint
    float  rotSpeed;     // radians/second self-rotation
};

// Matches VkDrawIndexedIndirectCommand exactly.
struct DrawCommand
{
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;   // stores asteroid index → vertex shader looks up Asteroid[firstInstance]
};

// Push constants for the cull compute pass.
struct CullPush
{
    float4x4 viewProj;         // view-projection matrix
    float3   cameraPos;
    float    time;             // elapsed seconds (for orbit/spin animation)
    uint     asteroidCount;    // total number of asteroids (1024)
    float    lodDist0;         // distance threshold LOD0→LOD1
    float    lodDist1;         // distance threshold LOD1→LOD2
    float    _pad;
};

// Per-frame UBO read by the vertex shader.
struct FrameUBO
{
    float4x4 viewProj;
    float3   cameraPos;
    float    time;
};

// ---------------------------------------------------------------------------
// Compute bindings  (set 0)
// ---------------------------------------------------------------------------
[[vk::binding(0, 0)]] StructuredBuffer<Asteroid>          asteroids;
[[vk::binding(1, 0)]] RWStructuredBuffer<DrawCommand>     drawCmds0;   // LOD 0 commands
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawCommand>     drawCmds1;   // LOD 1 commands
[[vk::binding(3, 0)]] RWStructuredBuffer<DrawCommand>     drawCmds2;   // LOD 2 commands
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>            drawCount0;  // LOD 0 atomic counter
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>            drawCount1;  // LOD 1 atomic counter
[[vk::binding(6, 0)]] RWStructuredBuffer<uint>            drawCount2;  // LOD 2 atomic counter

[[vk::push_constant]] CullPush cullPush;

// ---------------------------------------------------------------------------
// Compute: frustum cull + LOD selection
// ---------------------------------------------------------------------------

// Extract the 6 frustum planes (Gribb–Hartmann) from a view-projection matrix.
// Returns planes as float4(normal.xyz, d); dot(p, normal) + d >= 0 means inside.
//
// IMPORTANT — matrix-layout consistency with the vertex shader:
//   The vertex shader transforms vertices with `mul(viewProj, float4(pos,1))`,
//   and that render is correct.  In Slang, `mul(M, v)` computes result[i] =
//   dot(row i of M, v), and indexing `M[i]` returns that SAME logical row i.
//   So the Gribb–Hartmann planes — which are built from the ROWS of VP — must
//   use `vp[i]` directly.  The previous version instead assembled COLUMNS
//   (vp[0][0], vp[1][0], …), i.e. the transpose of VP, which produced a bogus
//   frustum: asteroids survived culling only in a thin mis-oriented slab,
//   bunched into one corner of the screen.  Using the rows fixes the field.
//
// Planes are normalised so the sphere test `dist < -radius` uses true
// world-space distances regardless of the matrix's overall scale.
void extractFrustumPlanes(float4x4 vp, out float4 planes[6])
{
    float4 r0 = vp[0];   // row 0 of VP
    float4 r1 = vp[1];   // row 1
    float4 r2 = vp[2];   // row 2
    float4 r3 = vp[3];   // row 3 (w-row)

    float4 raw[6];
    raw[0] = r3 + r0;    // left
    raw[1] = r3 - r0;    // right
    raw[2] = r3 + r1;    // bottom
    raw[3] = r3 - r1;    // top
    raw[4] = r2;         // near  (Vulkan [0,1] depth)
    raw[5] = r3 - r2;    // far

    for (int i = 0; i < 6; ++i)
    {
        float len = length(raw[i].xyz);
        planes[i] = (len > 1e-8f) ? raw[i] / len : raw[i];
    }
}

// Sphere-frustum test. Returns true if the sphere is (potentially) visible.
bool sphereInFrustum(float4 planes[6], float3 centre, float radius)
{
    for (int i = 0; i < 6; ++i)
    {
        float dist = dot(planes[i].xyz, centre) + planes[i].w;
        if (dist < -radius)
            return false;
    }
    return true;
}

// Animate an asteroid's world-space position given the elapsed time.
float3 animatedPosition(Asteroid a, float t)
{
    // Orbit around Y-axis at orbitSpeed
    float angle = a.orbitSpeed * t;
    float ca = cos(angle), sa = sin(angle);
    float3 p = a.position;
    return float3(p.x * ca - p.z * sa, p.y, p.x * sa + p.z * ca);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cullMain(uint3 threadId : SV_DispatchThreadID)
{
    uint idx = threadId.x;
    if (idx >= cullPush.asteroidCount)
        return;

    Asteroid a = asteroids[idx];

    // Animated world position
    float3 worldPos = animatedPosition(a, cullPush.time);

    // Frustum cull using a bounding sphere of radius = size * 1.5 (generous for jagged meshes)
    float4 planes[6];
    extractFrustumPlanes(cullPush.viewProj, planes);
    float cullRadius = a.size * 1.5f;
    if (!sphereInFrustum(planes, worldPos, cullRadius))
        return;

    // LOD selection based on camera distance
    float dist = length(worldPos - cullPush.cameraPos);
    uint  lod  = 2u;
    if (dist < cullPush.lodDist0)
        lod = 0u;
    else if (dist < cullPush.lodDist1)
        lod = 1u;

    // Write an indirect draw command for the chosen LOD
    DrawCommand cmd;
    cmd.instanceCount = 1u;
    cmd.firstIndex    = 0u;
    cmd.vertexOffset  = 0;
    cmd.firstInstance = idx;   // vertex shader uses this to look up Asteroid data

    if (lod == 0u)
    {
        cmd.indexCount = 144u;   // 48-tri icosphere: 48 * 3
        uint slot;
        InterlockedAdd(drawCount0[0], 1u, slot);
        drawCmds0[slot] = cmd;
    }
    else if (lod == 1u)
    {
        cmd.indexCount = 60u;    // 20-tri icosahedron: 20 * 3
        uint slot;
        InterlockedAdd(drawCount1[0], 1u, slot);
        drawCmds1[slot] = cmd;
    }
    else
    {
        cmd.indexCount = 24u;    // 8-tri octahedron: 8 * 3
        uint slot;
        InterlockedAdd(drawCount2[0], 1u, slot);
        drawCmds2[slot] = cmd;
    }
}

// ---------------------------------------------------------------------------
// Graphics bindings (set 1 — graphicsDescLayout; set 0 is computeDescLayout)
// The graphics pipeline layout declares both set 0 (compute layout, unused by
// vertex/fragment shaders) and set 1 (graphics layout) so there is no overlap
// between the compute-only bindings (set 0, bindings 0-6) and these.
// ---------------------------------------------------------------------------
[[vk::binding(0, 1)]] StructuredBuffer<Asteroid> gfxAsteroids;
[[vk::binding(1, 1)]] ConstantBuffer<FrameUBO>   frameUBO;

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

struct VSIn
{
    float3 pos    : POSITION;
    float3 normal : NORMAL;
};

struct VSOut
{
    float4 sv_pos   : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float3 color    : TEXCOORD2;
};

// Build a rotation matrix from axis-angle.
float3x3 axisAngleRotation(float3 axis, float angle)
{
    float c  = cos(angle);
    float s  = sin(angle);
    float t  = 1.0f - c;
    float x  = axis.x, y = axis.y, z = axis.z;
    return float3x3(
        t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
        t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
        t*x*z - s*y, t*y*z + s*x, t*z*z + c
    );
}

// Animate an asteroid's world-space position (same logic as compute side).
float3 animatedPositionV(Asteroid a, float t)
{
    float angle = a.orbitSpeed * t;
    float ca = cos(angle), sa = sin(angle);
    float3 p = a.position;
    return float3(p.x * ca - p.z * sa, p.y, p.x * sa + p.z * ca);
}

[shader("vertex")]
VSOut vertMain(VSIn vsIn,
               uint iid  : SV_InstanceID,
               uint base : SV_StartInstanceLocation)
{
    // SV_InstanceID = gl_InstanceIndex - gl_BaseInstance (always 0 when instanceCount=1).
    // SV_StartInstanceLocation = gl_BaseInstance = firstInstance = asteroid index.
    // Together they reconstruct gl_InstanceIndex, which is the correct asteroid index.
    uint instanceIndex = iid + base;
    Asteroid a = gfxAsteroids[instanceIndex];

    float t = frameUBO.time;

    // Self-rotation angle
    float spinAngle = a.rotSpeed * t;
    float3x3 spin   = axisAngleRotation(normalize(a.rotAxis), spinAngle);

    // Orbit rotation (around Y-axis)
    float orbitAngle = a.orbitSpeed * t;
    float ca = cos(orbitAngle), sa = sin(orbitAngle);
    float3x3 orbit = float3x3(
         ca, 0, sa,
          0, 1,  0,
        -sa, 0, ca
    );

    // Transform mesh vertex: scale → spin → orbit → translate
    float3 localPos     = vsIn.pos * a.size;
    float3 rotatedPos   = mul(orbit, mul(spin, localPos));
    float3 worldPos     = rotatedPos + a.position; // orbit of base position (pre-computed on CPU side)
    // Re-apply the orbit to the stored (disk) position
    float3 baseOrbit    = float3(a.position.x * ca - a.position.z * sa,
                                  a.position.y,
                                  a.position.x * sa + a.position.z * ca);
    worldPos = mul(orbit, mul(spin, localPos)) + baseOrbit;

    float3 worldNormal  = normalize(mul(orbit, mul(spin, vsIn.normal)));

    VSOut o;
    o.sv_pos   = mul(frameUBO.viewProj, float4(worldPos, 1.0f));
    o.worldPos = worldPos;
    o.normal   = worldNormal;
    o.color    = a.color;
    return o;
}

// ---------------------------------------------------------------------------
// Fragment shader – Phong shading
// ---------------------------------------------------------------------------

struct PSIn
{
    float3 worldPos : TEXCOORD0;
    float3 normal   : TEXCOORD1;
    float3 color    : TEXCOORD2;
};

[shader("fragment")]
float4 fragMain(PSIn psIn) : SV_Target
{
    // Distant sun direction (fixed, from upper-right-front)
    float3 sunDir    = normalize(float3(0.6f, 0.8f, 0.3f));
    float3 N         = normalize(psIn.normal);

    // Diffuse
    float  NdotL     = max(dot(N, sunDir), 0.0f);
    float3 diffuse   = psIn.color * NdotL;

    // Ambient
    float3 ambient   = psIn.color * 0.15f;

    // Specular (Blinn-Phong)
    float3 viewDir   = normalize(frameUBO.cameraPos - psIn.worldPos);
    float3 halfDir   = normalize(sunDir + viewDir);
    float  spec      = pow(max(dot(N, halfDir), 0.0f), 32.0f);
    float3 specular  = float3(0.4f, 0.38f, 0.35f) * spec;

    float3 finalColor = ambient + diffuse + specular;
    return float4(finalColor, 1.0f);
}
