// image_resize.slang
// Bilinear interpolation shader for resizing images

struct PushConstants {
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
    uint isBgr;
    uint srcStep;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] ByteAddressBuffer srcImage;   // Source: BGR/RGB uint8 buffer
[[vk::binding(1, 0)]] RWStructuredBuffer<float> dstImage; // Dest: HWC float format

float fetch(uint x, uint y, uint channel) {
    uint byteOffset = y * pc.srcStep + x * 3 + channel;
    // ByteAddressBuffer.Load reads 4 bytes, we need to extract one
    uint wordOffset = byteOffset & ~3;
    uint bitShift = (byteOffset & 3) * 8;
    return float((srcImage.Load(wordOffset) >> bitShift) & 0xFF);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint dstX = id.x;
    uint dstY = id.y;

    if (dstX >= pc.dstWidth || dstY >= pc.dstHeight) {
        return;
    }

    // Map destination coords to source coords (center-aligned)
    float srcX = (dstX + 0.5f) * float(pc.srcWidth) / float(pc.dstWidth) - 0.5f;
    float srcY = (dstY + 0.5f) * float(pc.srcHeight) / float(pc.dstHeight) - 0.5f;

    // Clamp to valid range
    srcX = clamp(srcX, 0.0f, float(pc.srcWidth - 1));
    srcY = clamp(srcY, 0.0f, float(pc.srcHeight - 1));

    // Get integer coords for bilinear interpolation
    uint x0 = uint(srcX);
    uint y0 = uint(srcY);
    uint x1 = min(x0 + 1, pc.srcWidth - 1);
    uint y1 = min(y0 + 1, pc.srcHeight - 1);

    // Calculate interpolation weights
    float wx = srcX - float(x0);
    float wy = srcY - float(y0);

    // Bilinear interpolation for all 3 channels
    // We also handle BGR to RGB conversion here if requested
    for (uint c = 0; c < 3; ++c) {
        uint srcC = c;
        if (pc.isBgr != 0) {
            srcC = 2 - c; // Swap R and B
        }

        float p00 = fetch(x0, y0, srcC);
        float p01 = fetch(x1, y0, srcC);
        float p10 = fetch(x0, y1, srcC);
        float p11 = fetch(x1, y1, srcC);

        // Bilinear interpolation
        float interpolated =
            p00 * (1.0f - wx) * (1.0f - wy) +
            p01 * wx * (1.0f - wy) +
            p10 * (1.0f - wx) * wy +
            p11 * wx * wy;

        // Store result in HWC layout (floats)
        dstImage[(dstY * pc.dstWidth + dstX) * 3 + c] = interpolated;
    }
}
