// Chapter 10 – Performance Optimization: GPU Heatmap
//
// Single-entry shader that runs one of four benchmark kernels based on
// a push-constant `kernelType` (0-3), writing a colored rectangle to the
// output storage image for its assigned quadrant tile.
//
//   kernelType 0 – divergent     : per-thread branch, colors red/orange
//   kernelType 1 – non-divergent : branchless, colors green
//   kernelType 2 – LDS reduction : groupshared tree reduce, colors cyan
//   kernelType 3 – wave reduction : WaveActiveSum, colors blue/violet
//
// The tile region [tileX, tileY, tileW, tileH] is also passed via push
// constants so the host can dispatch each tile separately with timestamps.
//
// Compiled with:
//   slangc 10_performance_optimization.slang -o shaders/slang.spv \
//       -profile spirv_1_4 -target spirv -emit-spirv-directly \
//       -fvk-use-entrypoint-name -entry heatmapMain

// Storage image – rgba8 matches R8G8B8A8Unorm on the C++ side
[[vk::binding(0, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outputImage;

struct HeatmapPush
{
    uint kernelType;  // 0=divergent, 1=non-divergent, 2=LDS, 3=wave
    uint tileX;       // pixel offset X for this tile
    uint tileY;       // pixel offset Y for this tile
    uint tileW;       // tile width  in pixels
    uint tileH;       // tile height in pixels
    uint frameIndex;  // incremented each frame (used to animate the benchmark)
};
[[vk::push_constant]] HeatmapPush pc;

// ---------------------------------------------------------------------------
// Module-scope groupshared – required by Slang (must be at module scope)
// ---------------------------------------------------------------------------
groupshared float ldsData[256];

groupshared uint  waveSlotCtr;
groupshared float wavePartials[64];   // 256 threads / min-wave-size-4 = 64 max waves

// ---------------------------------------------------------------------------
// Heatmap color: maps a normalized intensity [0,1] to a cool→hot color ramp
//   0.0 = blue, 0.25 = cyan, 0.5 = green, 0.75 = yellow, 1.0 = red
// ---------------------------------------------------------------------------
float4 heatColor(float t)
{
    t = clamp(t, 0.0f, 1.0f);
    float r = clamp(2.0f * t - 0.5f,  0.0f, 1.0f);
    float g = clamp(2.0f * t,          0.0f, 1.0f) * clamp(2.0f - 2.0f * t, 0.0f, 1.0f);
    float b = clamp(1.5f - 2.0f * t,  0.0f, 1.0f);
    return float4(r, g, b, 1.0f);
}

// ---------------------------------------------------------------------------
// heatmapMain – dispatched once per tile per frame
// Thread (x,y) maps to pixel (tileX + x, tileY + y)
// ---------------------------------------------------------------------------
[numthreads(16, 16, 1)]
[shader("compute")]
void heatmapMain(
    uint3 groupID    : SV_GroupID,
    uint3 threadID   : SV_GroupThreadID,
    uint3 dispatchID : SV_DispatchThreadID)
{
    uint px = pc.tileX + dispatchID.x;
    uint py = pc.tileY + dispatchID.y;

    uint imgW, imgH;
    outputImage.GetDimensions(imgW, imgH);
    bool validPixel = (px < imgW) && (py < imgH) &&
                      (dispatchID.x < pc.tileW) && (dispatchID.y < pc.tileH);

    // Kernels 0 and 1 have no barriers — out-of-bounds threads can exit early.
    // Kernels 2 and 3 contain GroupMemoryBarrierWithGroupSync: ALL threads in the
    // workgroup must reach every barrier uniformly. Only skip the final image write.
    if (pc.kernelType < 2u && !validPixel)
        return;

    // -----------------------------------------------------------------------
    // Benchmark computation – the result drives the output color
    // -----------------------------------------------------------------------

    // Seed value that varies per pixel and per frame to avoid trivial
    // compiler constant-folding of the benchmark work
    float val = sin(float(px) * 0.031f + float(py) * 0.017f + float(pc.frameIndex) * 0.01f);

    float4 color;

    if (pc.kernelType == 0u)
    {
        // ---- Kernel 0: Divergent ----
        // Branches based on a per-thread value → produces warp divergence.
        // The modulo is normalised to [0,2] with the signed-safe trick from prior work.
        float result = 0.0f;
        int t = int(val * 10.0f);
        int branch = ((t % 3) + 3) % 3;

        if (branch == 0)
            result = val * val + sqrt(abs(val));
        else if (branch == 1)
            result = val * val * val - val;
        else
            result = sin(val) * cos(val);

        // Color: warm (yellow→red) showing it is the "slow" kernel
        float intensity = abs(result);
        color = float4(0.9f + 0.1f * intensity, 0.4f - 0.2f * intensity, 0.1f, 1.0f);
    }
    else if (pc.kernelType == 1u)
    {
        // ---- Kernel 1: Non-divergent ----
        // Same result as kernel 0 but computed branchlessly via lerp/step.
        int t    = int(val * 10.0f);
        int mod3 = ((t % 3) + 3) % 3;

        float r0 = val * val + sqrt(abs(val));
        float r1 = val * val * val - val;
        float r2 = sin(val) * cos(val);

        float s01 = step(0.5f, float(mod3 >= 1));
        float s12 = step(0.5f, float(mod3 >= 2));
        float result = lerp(r0, lerp(r1, r2, s12), s01);

        // Color: cool (green) showing it is the "fast" kernel
        float intensity = abs(result);
        color = float4(0.05f, 0.7f + 0.2f * intensity, 0.2f, 1.0f);
    }
    else if (pc.kernelType == 2u)
    {
        // ---- Kernel 2: LDS Reduction ----
        // Tree reduction using groupshared memory.
        // Each 16x16 block = 256 threads; flatten to 1-D for the reduction.
        uint tid = threadID.y * 16u + threadID.x;

        // Load one value per thread
        ldsData[tid] = sin(float(tid) * 0.01f + val);
        GroupMemoryBarrierWithGroupSync();

        // Binary tree reduction
        // NOTE: [unroll] must NOT be used here — the loop contains a
        // GroupMemoryBarrierWithGroupSync() barrier, and unrolling a
        // barrier-containing loop causes GPU hangs (VK_ERROR_DEVICE_LOST)
        // on Intel ARL and some other Vulkan drivers.
        for (uint stride = 128u; stride > 0u; stride >>= 1u)
        {
            if (tid < stride)
                ldsData[tid] += ldsData[tid + stride];
            GroupMemoryBarrierWithGroupSync();
        }

        float groupSum = ldsData[0];

        // Color: cyan/teal to distinguish LDS path
        float intensity = clamp(abs(groupSum) / 256.0f, 0.0f, 1.0f);
        color = float4(0.0f, 0.6f + 0.4f * intensity, 0.8f + 0.2f * intensity, 1.0f);
    }
    else
    {
        // ---- Kernel 3: Wave Reduction ----
        // Intra-wave sum then groupshared accumulation across waves.
        uint tid = threadID.y * 16u + threadID.x;

        if (tid == 0u)
            waveSlotCtr = 0u;
        GroupMemoryBarrierWithGroupSync();

        float elem     = sin(float(tid) * 0.01f + val);
        float waveSum  = WaveActiveSum(elem);

        uint mySlot = 0u;
        if (WaveIsFirstLane())
        {
            InterlockedAdd(waveSlotCtr, 1u, mySlot);
            wavePartials[mySlot] = waveSum;
        }
        GroupMemoryBarrierWithGroupSync();

        float groupSum = 0.0f;
        if (tid == 0u)
        {
            for (uint w = 0u; w < waveSlotCtr; ++w)
                groupSum += wavePartials[w];
        }
        // Broadcast groupSum to all threads via a final LDS slot
        if (tid == 0u)
            ldsData[0] = groupSum;
        GroupMemoryBarrierWithGroupSync();
        groupSum = ldsData[0];

        // Color: blue/violet to distinguish wave reduction path
        float intensity = clamp(abs(groupSum) / 256.0f, 0.0f, 1.0f);
        color = float4(0.3f + 0.3f * intensity, 0.1f, 0.7f + 0.3f * intensity, 1.0f);
    }

    if (validPixel)
        outputImage[int2(px, py)] = color;
}
