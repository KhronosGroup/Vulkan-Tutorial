// Chapter 4 – Subgroup/Wave Operations: Hair Strands on a Sphere in Wind
//
// Two compute entry points:
//   physicsMain – one thread per strand; uses WaveActiveSum + WavePrefixSum
//                 (with the atomic slot counter pattern for Intel Arc ARL
//                  variable-subgroup-size safety) to compute strand tip
//                 displacement from animated wind forces.
//   renderMain  – one thread per pixel; 2D view of strands radiating from
//                 a centre circle, blown by wind.  Uses WaveActiveBallot
//                 to cull strands facing away from wind.
//
// Design note (Intel Arc ARL / variable subgroup sizes):
//   Intel Arc ARL (Alchemist) may use SIMD8 / SIMD16 / SIMD32 within the
//   same workgroup.  Therefore  groupIndex / WaveGetLaneCount()  gives a
//   WRONG wave index.  Instead we use an atomic slot counter: the first
//   lane of each wave claims a slot via InterlockedAdd, then stores its
//   groupIndex in waveScanFirstLanes[slot].  Thread 0 can sort/use the
//   per-wave data after a GroupMemoryBarrierWithGroupSync().

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
// Binding 0: render output (RGBA8 storage image)
[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;

// Binding 1: strand tip displacement array  (one float per strand, physics output)
[[vk::binding(1, 0)]] RWStructuredBuffer<float> strandTip;

// ---------------------------------------------------------------------------
// Push constants
// ---------------------------------------------------------------------------
struct PushConst
{
    uint  width;        // image width  (pixels)
    uint  height;       // image height (pixels)
    uint  numStrands;   // total number of strands (must be <= 1024)
    float time;         // animation time in seconds
    float windStrength; // base wind strength
};
[[vk::push_constant]] PushConst pc;

// ---------------------------------------------------------------------------
// Module-scope groupshared  (groupshared is illegal inside function bodies)
// ---------------------------------------------------------------------------
static const uint kMaxWaves = 64u;   // 256 threads / min-wave-size-4

// physicsMain groupshared
groupshared uint  physSlotCtr;
groupshared uint  physFirstLane[kMaxWaves];
groupshared float physWaveForce[kMaxWaves];     // per-wave WaveActiveSum result
groupshared float physWaveOffset[kMaxWaves];    // exclusive scan of wave totals
groupshared float physGroupWindSum;             // total wind force in this group

// ---------------------------------------------------------------------------
// Entry 1: physicsMain
//
// Workgroup = (256, 1, 1).  Each thread corresponds to one strand index.
// We demonstrate:
//   WaveActiveSum    – aggregate per-strand wind forces across the wave
//   WavePrefixSum    – chain propagation: each strand's "inherited" force
//                      from all strands before it in the wave
//   Atomic slot counter – safe wave-index assignment on variable-SIMD GPUs
// ---------------------------------------------------------------------------
[numthreads(256, 1, 1)]
[shader("compute")]
void physicsMain(uint3 gid   : SV_DispatchThreadID,
                 uint  gIdx  : SV_GroupIndex)
{
    // Initialise slot counter once per group
    if (gIdx == 0u)
    {
        physSlotCtr     = 0u;
        physGroupWindSum = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    uint strandIdx = gid.x;
    bool active    = (strandIdx < pc.numStrands);

    // ---- Per-strand wind force ----
    // Strands are distributed around a circle; the wind force is a sinusoidal
    // function of the strand's angular position and time.
    float angle     = (float(strandIdx) / float(pc.numStrands)) * 6.28318f;
    float windForce = active
        ? (0.5f + 0.5f * sin(angle * 2.0f + pc.time * 1.5f)) * pc.windStrength
        : 0.0f;

    // ---- WaveActiveSum: aggregate wind forces within this wave ----
    float waveForceSum = WaveActiveSum(windForce);

    // ---- WavePrefixSum: how much force precedes this strand in its wave ----
    float wavePrefix = WavePrefixSum(windForce);

    // ---- Atomic slot counter pattern (Intel Arc ARL safe) ----
    uint mySlot = 0u;
    if (WaveIsFirstLane())
    {
        InterlockedAdd(physSlotCtr, 1u, mySlot);
        physFirstLane[mySlot]  = gIdx;       // first-lane groupIndex for sorting
        physWaveForce[mySlot]  = waveForceSum;
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- Thread 0: insertion-sort slots by firstLane, then exclusive scan ----
    if (gIdx == 0u)
    {
        uint n = physSlotCtr;

        // Insertion sort: sort (physFirstLane[], physWaveForce[]) by firstLane
        for (uint i = 1u; i < n; ++i)
        {
            uint  fl  = physFirstLane[i];
            float wf  = physWaveForce[i];
            int   j   = int(i) - 1;
            while (j >= 0 && physFirstLane[j] > fl)
            {
                physFirstLane[j + 1] = physFirstLane[j];
                physWaveForce[j + 1] = physWaveForce[j];
                j--;
            }
            physFirstLane[j + 1] = fl;
            physWaveForce[j + 1] = wf;
        }

        // Exclusive scan of wave force totals → physWaveOffset
        physWaveOffset[0] = 0.0f;
        for (uint w = 1u; w < n; ++w)
            physWaveOffset[w] = physWaveOffset[w - 1u] + physWaveForce[w - 1u];

        // Group total
        float total = physWaveOffset[n - 1u] + physWaveForce[n - 1u];
        physGroupWindSum = total;
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- Each first lane finds its sorted position ----
    uint mySortedIdx = 0u;
    if (WaveIsFirstLane())
    {
        uint fl = gIdx;
        uint n  = physSlotCtr;
        for (uint k = 0u; k < n; ++k)
        {
            if (physFirstLane[k] == fl) { mySortedIdx = k; break; }
        }
    }
    uint waveIdx = WaveReadLaneFirst(mySortedIdx);

    // ---- Final displacement = wave prefix offset + intra-wave prefix ----
    float displacement = physWaveOffset[waveIdx] + wavePrefix;

    // Normalise by group total so displacement stays in [0, 1] range
    float norm = physGroupWindSum > 0.0f ? displacement / physGroupWindSum : 0.0f;

    // Write result: strand tip displacement in [-1, 1] with a sinusoidal modulation
    if (active)
        strandTip[strandIdx] = norm * pc.windStrength
                             * (0.8f + 0.2f * sin(pc.time * 3.0f + angle));
}

// ---------------------------------------------------------------------------
// Entry 2: renderMain
//
// Workgroup = (16, 16, 1).  One thread per pixel.
// Renders a 2D "top view":
//   - A circle (sphere cross-section) in the centre with gradient shading
//   - Thick hair strands radiating outward from the circle, blown by wind
//     drawn via SDF distance-to-segment for visible thickness
// Demonstrates WaveActiveBallot to cull strands that are in the "wind shadow"
// (dot(strandDir, windDir) < 0 → the strand is facing away from the wind).
// ---------------------------------------------------------------------------

// SDF helper: distance from point p to line segment [a, b]
float distToSegment(float2 p, float2 a, float2 b)
{
    float2 ab = b - a;
    float2 ap = p - a;
    float  t  = clamp(dot(ap, ab) / (dot(ab, ab) + 1e-8f), 0.0f, 1.0f);
    return length(ap - t * ab);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void renderMain(uint3 gid : SV_DispatchThreadID)
{
    uint px = gid.x;
    uint py = gid.y;
    if (px >= pc.width || py >= pc.height)
        return;

    // Normalised pixel coordinate in [-1, 1] with correct aspect ratio
    float aspect = float(pc.width) / float(pc.height);
    float2 uv    = float2(
        (float(px) / float(pc.width)  - 0.5f) * 2.0f * aspect,
        (float(py) / float(pc.height) - 0.5f) * 2.0f);

    // Pixel size in UV space — used to convert pixel thickness to UV units
    float pixelSizeUV = 2.0f / float(pc.height);

    // Background: deep dark blue-grey with a subtle vignette
    float vignette = 1.0f - 0.4f * saturate((uv.x * uv.x + uv.y * uv.y) * 0.5f);
    float4 color   = float4(0.04f * vignette, 0.05f * vignette, 0.10f * vignette, 1.0f);

    // Sphere circle with solid grey/silver shading + specular highlight
    const float sphereR = 0.30f;
    float distSphere = length(uv);
    if (distSphere < sphereR)
    {
        float nx = uv.x / sphereR;
        float ny = uv.y / sphereR;
        float nz = sqrt(max(0.0f, 1.0f - nx*nx - ny*ny));
        float3 lightDir = normalize(float3(0.6f + 0.2f*sin(pc.time), 0.4f, 0.8f));
        float3 norm     = float3(nx, ny, nz);
        float  diffuse  = max(0.15f, dot(norm, lightDir));
        // Specular: Blinn-Phong
        float3 viewDir  = float3(0.0f, 0.0f, 1.0f);
        float3 halfVec  = normalize(lightDir + viewDir);
        float  spec     = pow(max(0.0f, dot(norm, halfVec)), 40.0f);
        float3 baseCol  = float3(0.55f, 0.58f, 0.62f);  // silver-grey
        color = float4(baseCol * diffuse + float3(0.6f, 0.6f, 0.7f) * spec, 1.0f);
    }
    // Thin edge ring to make sphere boundary crisp
    else if (distSphere < sphereR + pixelSizeUV * 1.5f)
    {
        color = float4(0.5f, 0.52f, 0.55f, 1.0f);
    }

    // Wind direction (animated sinusoidal — slowly rotating)
    float windAngle = pc.time * 0.4f;
    float2 windDir  = normalize(float2(cos(windAngle), sin(windAngle * 0.6f)));

    // Strand thickness: 2.5 pixels expressed in UV units
    const float strandThickness = pixelSizeUV * 2.5f;
    const int   numSeg          = 8;

    // Track closest strand hit for smooth alpha blending
    float  minDist    = 1e10f;
    float3 hairColor  = float3(0.0f, 0.0f, 0.0f);

    for (uint s = 0u; s < pc.numStrands; ++s)
    {
        float  angle    = (float(s) / float(pc.numStrands)) * 6.28318f;
        float2 rootDir  = float2(cos(angle), sin(angle));

        // ---- WaveActiveBallot: cull strands facing away from wind ----
        // A strand "in wind shadow" points significantly away from the wind.
        // We use a softer threshold (-0.3) so more strands are visible, but
        // strands strongly facing away from wind are still culled for realism.
        bool inWindShadow = (dot(rootDir, windDir) < -0.3f);
        uint4 ballot      = WaveActiveBallot(inWindShadow);
        // ballot is used for potential subgroup-wide culling decisions;
        // here we act per-strand on the individual boolean.
        if (inWindShadow)
            continue;

        // Root sits just outside the sphere
        float2 root = rootDir * (sphereR + pixelSizeUV);

        // Wind deflection: strands on windward side bend more dramatically
        float windDot    = dot(rootDir, windDir);   // in [-1, 1], positive = windward
        float tipDisp    = strandTip[s];             // [0,1]-ish from physics
        // Exaggerate the bend: windward strands (windDot > 0) get strong bend,
        // cross-wind strands get moderate bend, and we use tipDisp as modulation.
        float bendAmount = (0.4f + windDot * 0.5f) * (0.8f + 0.4f * tipDisp)
                         * pc.windStrength;

        // Build strand tip via quadratic Bezier-like approach:
        // strand grows in rootDir, then curves toward wind
        // Total strand length = 2.2 × sphereR (well beyond the sphere)
        float strandLen = sphereR * 2.2f;

        float2 prevPt = root;
        for (int seg = 0; seg < numSeg; ++seg)
        {
            float t       = float(seg + 1) / float(numSeg);
            float blend   = t * t;   // quadratic bend — tip bends more
            // Direction evolves: starts radially outward, curves toward wind at tip
            float2 segDir  = normalize(rootDir + windDir * (bendAmount * blend));
            // Slight wave motion along the strand
            float  wave    = 0.04f * sin(pc.time * 2.5f + angle + t * 6.28318f);
            float2 perpDir = float2(-segDir.y, segDir.x);
            float2 nextPt  = root + segDir * (strandLen * t)
                           + perpDir * (wave * strandLen);

            // SDF: distance from this pixel to segment [prevPt, nextPt]
            float dist = distToSegment(uv, prevPt, nextPt);

            if (dist < minDist)
            {
                minDist = dist;
                // Warm golden-amber at root, cooler at tip
                float hairT = t;
                float r = lerp(0.95f, 0.70f, hairT);
                float g = lerp(0.75f, 0.50f, hairT);
                float b = lerp(0.15f, 0.05f, hairT);
                hairColor = float3(r, g, b);
            }

            prevPt = nextPt;
        }
    }

    // Blend hair onto background/sphere using smooth anti-aliased coverage
    if (minDist < strandThickness && distSphere >= sphereR)
    {
        // Smooth falloff for anti-aliasing: full opaque in centre, fade at edge
        float alpha = 1.0f - smoothstep(strandThickness * 0.5f, strandThickness, minDist);
        color.rgb   = lerp(color.rgb, hairColor, alpha);
    }

    // Grid overlay: subtle dot pattern visualising subgroup lanes on Intel Arc ARL
    // One faint dot per wave-first-lane position (8×8 pixel grid)
    if (px % 8u == 0u && py % 8u == 0u)
    {
        color.rgb += float3(0.03f, 0.04f, 0.05f);
    }

    outImage[int2(int(px), int(py))] = color;
}
