// Chapter 9 – Cooperative Matrices & Specialized Math (Slang compute shader)
//
// TWO entry points demonstrating FP16 arithmetic and tiled computation:
//
//   noiseMain    – animated value-noise generation using half-precision (FP16) math
//                  writes packed half2 values to an intermediate SSBO (binding 0)
//
//   denoiseMain  – tile-based FP16 Gaussian blur using 16×16 groupshared memory
//                  reads packed half2 from the noise SSBO (binding 1)
//                  writes blurred rgba8 result to a storage image (binding 0)
//
// The groupshared tile accumulation in denoiseMain mirrors the data-sharing
// pattern that VK_KHR_cooperative_matrix hardware-accelerates on GPUs with
// tensor/matrix cores.  On supported hardware, the per-thread FP16 accumulation
// loops would be replaced by a single CooperativeMatrixMulAdd instruction.
//
// FP16 feature requirements (both must be enabled in the Vulkan device):
//   shaderFloat16         → VkPhysicalDeviceVulkan12Features (promoted from
//                           VK_KHR_shader_float16_int8)
//   storageBuffer16BitAccess → VkPhysicalDeviceVulkan11Features (promoted from
//                           VK_KHR_16bit_storage)
//
// Compile:
//   slangc 09_specialized_math.slang \
//          -profile spirv_1_4 -target spirv -emit-spirv-directly \
//          -fvk-use-entrypoint-name \
//          -entry noiseMain -entry denoiseMain \
//          -o shaders/slang.spv

// ============================================================================
// Push constants (shared by both passes)
// ============================================================================
struct NoisePush
{
    float time;           // animation clock (seconds)
    float frequency;      // noise frequency multiplier
    uint  blurRadius;     // Gaussian blur half-width (0..4)
    uint  width;
    uint  height;
};
[[vk::push_constant]] NoisePush pc;

// ============================================================================
// Pass 1 resources
//   binding 0: noise output buffer — one uint32 per pixel, packed as two FP16
//              values (R in low 16 bits, G in high 16 bits).
//              Using uint instead of half2 avoids RWStructuredBuffer<half>
//              alignment edge cases across Slang/SPIR-V versions.
// ============================================================================
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> noiseBuf;

// ============================================================================
// Pass 2 resources
//   binding 0: output storage image (rgba8, eGeneral)
//   binding 1: input noise buffer (same packed uint layout as above)
// ============================================================================
[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> noiseIn;

// ============================================================================
// FP16 pack / unpack helpers
//   We store two half values packed into one uint32 so that the SSBO can
//   use a plain uint element type — no 16-bit storage alignment concerns.
// ============================================================================

/// Pack two float values into a uint32 as two 16-bit halves (R=lo, G=hi).
uint packHalf2(float r, float g)
{
    return (f32tof16(r) & 0xFFFFu) | ((f32tof16(g) & 0xFFFFu) << 16u);
}

/// Unpack the two half values from a packed uint32.
float2 unpackHalf2(uint packed)
{
    return float2(f16tof32(packed & 0xFFFFu),
                  f16tof32((packed >> 16u) & 0xFFFFu));
}

// ============================================================================
// Value-noise helpers (all arithmetic in FP16 to exercise shaderFloat16)
//
// Theory:
//   Value noise interpolates randomly-assigned values at integer lattice points.
//   Each lattice point gets a pseudo-random value; neighbouring cells are
//   smoothly interpolated using a quintic (fade) curve.  This produces smooth,
//   organic-looking noise at low cost — cheaper than Perlin noise while still
//   demonstrating FP16 vector arithmetic.
//
// FP16 benefit:
//   The interpolation maths uses half2 vectors.  On hardware with native FP16
//   vector units (Intel Xe, NVIDIA Maxwell+, AMD GCN3+) this executes at twice
//   the throughput of equivalent float32 code.
// ============================================================================

/// Pseudo-random float in [0,1] from a 2D integer seed.
half rand2(half2 p)
{
    // Mix the two coordinates using irrational-number dot products
    half2 k = half2((half)0.3183h, (half)0.3678h);
    p = p * k + k.yx;
    return (half)frac((float)(p.x * p.y * (p.x + p.y)));
}

/// Quintic fade: 6t^5 - 15t^4 + 10t^3  (zero first and second derivative at 0 and 1)
half2 fade(half2 t)
{
    return t * t * t * ((half)6.0h * t * t - (half)15.0h * t + (half)10.0h);
}

/// 2D value noise sample in FP16.
half valueNoise(half2 uv)
{
    half2 i = half2(floor(float2(uv)));   // integer cell
    half2 f = uv - i;                     // fractional position within cell

    half v00 = rand2(i + half2((half)0.0h, (half)0.0h));
    half v10 = rand2(i + half2((half)1.0h, (half)0.0h));
    half v01 = rand2(i + half2((half)0.0h, (half)1.0h));
    half v11 = rand2(i + half2((half)1.0h, (half)1.0h));

    half2 u = fade(f);

    // Bilinear interpolation of the four corner values
    half bottom = lerp(v00, v10, u.x);
    half top    = lerp(v01, v11, u.x);
    return       lerp(bottom, top, u.y);
}

/// Fractal Brownian Motion: sum of 4 octaves of value noise, all in FP16.
half fbm(half2 p)
{
    half sum  = (half)0.0h;
    half amp  = (half)0.5h;
    half freq = (half)1.0h;
    for (int i = 0; i < 4; ++i)
    {
        sum  += amp  * valueNoise(p * freq);
        amp  *= (half)0.5h;
        freq *= (half)2.0h;
    }
    return sum;
}

// ============================================================================
// Pass 1: noiseMain — FP16 animated noise generation
//
// Each thread computes one pixel's noise value in FP16 and stores it as a
// packed half2 into the intermediate noiseBuf.  The R channel holds the base
// noise, the G channel holds an octave-shifted variant for colour.
//
// Workgroup: 16×16 threads (256 total — a common GPU occupancy sweet spot)
// ============================================================================
[numthreads(16, 16, 1)]
[shader("compute")]
void noiseMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= pc.width || id.y >= pc.height)
        return;

    // Convert pixel coordinates to normalised noise space in FP16
    half2 uv = half2(
        (half)(float(id.x) / float(pc.width)),
        (half)(float(id.y) / float(pc.height)));

    // Scale by user-controlled frequency
    half2 p = uv * (half)pc.frequency;

    // Animate by rotating in noise space (FP16 trig approximation)
    half t      = (half)pc.time;
    half sinT   = (half)sin((float)t * 0.37f);
    half cosT   = (half)cos((float)t * 0.37f);
    half2 pAnim = half2(p.x * cosT - p.y * sinT,
                        p.x * sinT + p.y * cosT);

    // Two noise layers for R and G channels
    half noiseR = fbm(pAnim);
    half noiseG = fbm(pAnim + half2((half)3.1h, (half)1.7h));

    // Store as packed uint32 (half R in low 16, half G in high 16)
    uint idx = id.y * pc.width + id.x;
    noiseBuf[idx] = packHalf2((float)noiseR, (float)noiseG);
}

// ============================================================================
// Pass 2: denoiseMain — tile-based FP16 Gaussian blur
//
// Theory:
//   This is the "cooperative matrix emulation" pass.  A real cooperative matrix
//   kernel would load A and B tiles into hardware tensor registers and produce
//   C = A*B in a single instruction.  Here we do the equivalent manually using
//   groupshared (LDS) memory:
//
//     1. All 256 threads cooperatively load a 16×16 tile into groupshared memory.
//     2. Each thread reads a radius×radius neighbourhood from LDS.
//     3. FP16 multiply-accumulate computes the weighted Gaussian sum.
//     4. The result is written to the output storage image.
//
//   The key insight: the LDS tile load + cooperative accumulation has the same
//   data-access structure as a cooperative matrix operation — the difference is
//   just that VK_KHR_cooperative_matrix does it in dedicated hardware.
//
// Workgroup: 16×16 = 256 threads.
// ============================================================================

// Tile stored in groupshared (LDS) memory.
// We keep half2 values for R and G channels, packed per-thread.
// groupshared must be at module scope in Slang.
// The tile is 16×16 (workgroup size). Blur accesses a 5×5 neighbourhood
// clamped to [0,15] — no halo padding needed since we clamp at load time.
groupshared half2 gTile[16][16];

[numthreads(16, 16, 1)]
[shader("compute")]
void denoiseMain(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID,
                 uint3 id  : SV_DispatchThreadID)
{
    // ----------------------------------------------------------------
    // Step 1: Cooperative tile load into groupshared memory
    //
    // All threads load their own pixel's noise value.
    // This is the LDS-load phase that mirrors cooperative matrix LoadMatrix().
    // ----------------------------------------------------------------
    {
        uint px = min(id.x, pc.width  - 1u);
        uint py = min(id.y, pc.height - 1u);
        float2 rg = unpackHalf2(noiseIn[py * pc.width + px]);
        gTile[tid.y][tid.x] = half2((half)rg.x, (half)rg.y);
    }

    // Synchronise: all threads in the workgroup must finish loading before any
    // thread reads from gTile.  This is the groupshared barrier that would be
    // implicit inside a cooperative matrix MulAdd instruction.
    GroupMemoryBarrierWithGroupSync();

    // ----------------------------------------------------------------
    // Step 2: FP16 Gaussian blur over the tile neighbourhood
    //
    // This is the "multiply-accumulate" phase.  In a cooperative matrix kernel
    // this entire loop would collapse to:
    //   matC = CooperativeMatrixMulAdd(matA, matB, matC);
    // with the weights in matA and the tile values in matB.
    //
    // The blur radius is user-controlled (0..4).  A radius of 0 passes through
    // the raw noise; higher radii produce progressively smoother output.
    // The 5×5 Gaussian kernel (sigma≈1.5) approximated as separable weights:
    //   [0.0625, 0.25, 0.375, 0.25, 0.0625]  (normalised so sum = 1)
    // ----------------------------------------------------------------

    // Gaussian kernel weights in FP16 (sigma ≈ 1.0, 5-tap)
    const half kW[5] = { (half)0.0625h, (half)0.25h, (half)0.375h,
                         (half)0.25h,   (half)0.0625h };

    half2 accum = half2((half)0.0h, (half)0.0h);
    half  wSum  = (half)0.0h;

    int r = int(pc.blurRadius);

    for (int dy = -r; dy <= r; ++dy)
    {
        int sy = clamp(int(tid.y) + dy, 0, 15);
        // Pick the appropriate Gaussian weight (map dy ∈ [-4,4] → index ∈ [0,8])
        // For the 5-tap kernel only |dy| ≤ 2 are weighted; beyond that clamp to edge.
        half wy = (r > 0) ? kW[clamp(dy + 2, 0, 4)] : (half)1.0h;

        for (int dx = -r; dx <= r; ++dx)
        {
            int sx = clamp(int(tid.x) + dx, 0, 15);
            half wx = (r > 0) ? kW[clamp(dx + 2, 0, 4)] : (half)1.0h;

            half w = wy * wx;
            // FP16 multiply-accumulate — uses the hardware FP16 ALU
            accum += w * gTile[sy][sx];
            wSum  += w;
        }
    }

    // Normalise
    if (wSum > (half)0.0h)
        accum /= wSum;

    // ----------------------------------------------------------------
    // Step 3: Colour mapping and output
    //
    // Map the two noise channels to a pleasing colour palette.
    // The R channel drives luminance; G introduces a hue shift.
    // ----------------------------------------------------------------
    if (id.x >= pc.width || id.y >= pc.height)
        return;

    float noiseR = (float)accum.x;
    float noiseG = (float)accum.y;

    // IQ cosine palette: colour = a + b * cos(2π*(c*t + d))
    //   channel 0 (R): warm orange tones
    //   channel 1 (G): cool blue-green tones
    float3 col;
    col.r = 0.5f + 0.5f * cos(6.2832f * (0.0f + noiseR * 1.0f));
    col.g = 0.5f + 0.5f * cos(6.2832f * (0.33f + noiseG * 1.0f));
    col.b = 0.5f + 0.5f * cos(6.2832f * (0.67f + (noiseR + noiseG) * 0.5f));

    outImage[int2(id.xy)] = float4(col, 1.0f);
}
