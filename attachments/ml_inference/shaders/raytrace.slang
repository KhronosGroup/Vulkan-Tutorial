// raytrace.slang
// Simple sphere ray tracer for optimization demonstration

struct PushConstants {
    uint width;
    uint height;
    uint frameCount;
    uint samplesPerPixel;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> outputImage; // RGB image
[[vk::binding(1, 0)]] StructuredBuffer<float> heatmap;      // AI Heatmap (normalized 0-1)

// Simple random function
float rand(uint2 pixel, uint frame, uint sample) {
    return frac(sin(dot(float2(pixel) + float2(frame, sample), float2(12.9898, 78.233))) * 43758.5453);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pc.width || id.y >= pc.height) return;

    float3 color = float3(0, 0, 0);
    float2 uv = float2(id.xy) / float2(pc.width, pc.height);

    // AI-Guided Adaptive Sampling
    uint totalSamples = pc.samplesPerPixel;
    if (pc.samplesPerPixel > 1) {
        // We use the AI heatmap to determine how many of the requested additional samples to actually trace.
        // heatmap is 256x256, so we map coordinates.
        uint2 hCoord = uint2(id.x * 256 / pc.width, id.y * 256 / pc.height);
        float hWeight = heatmap[hCoord.y * 256 + hCoord.x];
        totalSamples = max(1u, uint(float(pc.samplesPerPixel) * clamp(hWeight, 0.05, 1.0)));
    }

    for (uint i = 0; i < totalSamples; i++) {
        float2 jitter = float2(rand(id.xy, pc.frameCount, i), rand(id.xy, pc.frameCount, i + 100)) - 0.5;
        float2 p = (float2(id.xy) + jitter) / float2(pc.width, pc.height) * 2.0 - 1.0;
        p.x *= float(pc.width) / float(pc.height);

        // Ray setup
        float3 ro = float3(0, 0, 2);
        float3 rd = normalize(float3(p, -1));

        // Sphere intersection
        float3 sc = float3(0, 0, 0);
        float sr = 0.8;
        float3 oc = ro - sc;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - sr * sr;
        float h = b * b - c;

        float3 sampleColor = float3(0.1, 0.1, 0.1); // Background
        if (h > 0.0) {
            float t = -b - sqrt(h);
            float3 pos = ro + t * rd;
            float3 norm = normalize(pos - sc);
            float diff = max(0.0, dot(norm, normalize(float3(1, 1, 1))));
            sampleColor = float3(0.8, 0.4, 0.2) * diff + float3(0.1, 0.1, 0.1);
        }
        
        color += sampleColor;
    }

    color /= float(totalSamples);

    // Write to planar RGB output (NCHW-like or interleaved? Let's use interleaved for simplicity in output)
    uint idx = (id.y * pc.width + id.x) * 3;
    outputImage[idx + 0] = color.r;
    outputImage[idx + 1] = color.g;
    outputImage[idx + 2] = color.b;
}
