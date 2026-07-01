// normalize.slang
// Normalize image with ImageNet statistics and convert HWC to NCHW layout

struct NormalizeParams {
    float meanR;
    float meanG;
    float meanB;
    float stdR;
    float stdG;
    float stdB;
    uint width;
    uint height;
};

[[vk::push_constant]] NormalizeParams params;

[[vk::binding(0, 0)]] StructuredBuffer<float> srcImage;   // HWC, [0-255]
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dstImage; // NCHW, normalized

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    uint y = id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    // Read from HWC layout: [height][width][channel]
    uint srcIdx = (y * params.width + x) * 3;

    // Process R channel
    {
        float value = srcImage[srcIdx + 0] / 255.0f;  // Normalize to [0, 1]
        value = (value - params.meanR) / params.stdR;  // Apply ImageNet stats
        uint dstIdx = 0 * params.width * params.height + y * params.width + x;
        dstImage[dstIdx] = value;
    }

    // Process G channel
    {
        float value = srcImage[srcIdx + 1] / 255.0f;
        value = (value - params.meanG) / params.stdG;
        uint dstIdx = 1 * params.width * params.height + y * params.width + x;
        dstImage[dstIdx] = value;
    }

    // Process B channel
    {
        float value = srcImage[srcIdx + 2] / 255.0f;
        value = (value - params.meanB) / params.stdB;
        uint dstIdx = 2 * params.width * params.height + y * params.width + x;
        dstImage[dstIdx] = value;
    }
}
