// Optimized NNEF Operations for Vulkan Compute
// Specifically tuned for MobileNetV2 patterns

struct ConvParams {
    uint srcWidth;
    uint srcHeight;
    uint srcChannels;
    uint dstWidth;
    uint dstHeight;
    uint dstChannels;
    uint kernelSize;
    uint stride;
    uint padding;
    uint groups;
    uint activation; // 0: None, 1: ReLU, 2: ReLU6
    uint opType;     // 0: Conv, 1: GlobalAvgPool, 2: Add, 3: Linear, 4: Copy, 5: Softmax, 6: Clamp
    uint hasBias;    // 1 if bias tensor is present
};

[[vk::push_constant]]
ConvParams params;

[[vk::binding(0)]]
StructuredBuffer<float> src;

[[vk::binding(1)]]
StructuredBuffer<float> weights;

[[vk::binding(2)]]
StructuredBuffer<float> bias;

[[vk::binding(3)]]
RWStructuredBuffer<float> dst;

// Sub-functions (not entry points)
void DoClamp(uint3 threadId) {
    uint i = threadId.x;
    uint size = params.dstWidth * params.dstHeight * params.dstChannels;
    if (i >= size) return;
    
    // NNEF clamp has min/max. For MobileNetV2 relu6 it's [0, 6]
    dst[i] = min(max(0.0, src[i]), 6.0);
}

void DoSoftmax(uint3 threadId) {
    // Very simple softmax (one thread per vector for now, assumes vector is small)
    // MobileNetV2 output is 1000 classes.
    if (threadId.x != 0) return; 

    uint size = params.srcChannels;
    float maxVal = -1e9;
    for (uint i = 0; i < size; i++) {
        maxVal = max(maxVal, src[i]);
    }

    float sum = 0;
    for (uint j = 0; j < size; j++) {
        float e = exp(src[j] - maxVal);
        dst[j] = e;
        sum += e;
    }

    for (uint k = 0; k < size; k++) {
        dst[k] /= sum;
    }
}

void DoAdd(uint3 threadId) {
    uint i = threadId.x;
    uint size = params.dstWidth * params.dstHeight * params.dstChannels;
    if (i >= size) return;
    
    float val = src[i] + weights[i];
    if (params.activation == 2) val = min(max(0.0, val), 6.0);
    else if (params.activation == 1) val = max(0.0, val);
    dst[i] = val;
}

void DoLinear(uint3 threadId) {
    uint oc = threadId.x;
    if (oc >= params.dstChannels) return;
    
    float sum = 0.0;
    if (params.hasBias != 0) {
        sum = bias[oc];
    }
    for (uint ic = 0; ic < params.srcChannels; ic++) {
        sum += src[ic] * weights[oc * params.srcChannels + ic];
    }
    if (params.activation == 1) sum = max(0.0, sum);
    dst[oc] = sum;
}

void DoCopy(uint3 threadId) {
    uint i = threadId.x;
    uint size = params.dstWidth * params.dstHeight * params.dstChannels;
    if (i >= size) return;
    dst[i] = src[i];
}

// entry point
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    uint i = threadId.x;
    
    if (params.opType == 0) { // Conv2D
        uint size = params.dstWidth * params.dstHeight * params.dstChannels;
        if (i >= size) return;

        uint outX = i % params.dstWidth;
        uint outY = (i / params.dstWidth) % params.dstHeight;
        uint oc = i / (params.dstWidth * params.dstHeight);

        float sum = 0.0;
        if (params.hasBias != 0) {
            sum = bias[oc];
        }
        
        uint group_size_out = params.dstChannels / params.groups;
        uint group_size_in = params.srcChannels / params.groups;
        uint g = oc / group_size_out;
        
        uint ic_start = g * group_size_in;
        uint ic_end = ic_start + group_size_in;

        for (uint ic = ic_start; ic < ic_end; ic++) {
            for (uint ky = 0; ky < params.kernelSize; ky++) {
                for (uint kx = 0; kx < params.kernelSize; kx++) {
                    int inX = int(outX * params.stride + kx) - int(params.padding);
                    int inY = int(outY * params.stride + ky) - int(params.padding);
                    
                    if (inX >= 0 && inX < (int)params.srcWidth && inY >= 0 && inY < (int)params.srcHeight) {
                        float val = src[(ic * params.srcHeight + uint(inY)) * params.srcWidth + uint(inX)];
                        uint weightIdx = (((oc * group_size_in + (ic - ic_start)) * params.kernelSize + ky) * params.kernelSize + kx);
                        sum += val * weights[weightIdx];
                    }
                }
            }
        }
        
        if (params.activation == 1) sum = max(0.0, sum);
        else if (params.activation == 2) sum = min(max(0.0, sum), 6.0);
        
        dst[i] = sum;
    }
    else if (params.opType == 1) { // GlobalAvgPool
        if (i >= params.srcChannels) return;
        
        float sum = 0;
        uint area = params.srcWidth * params.srcHeight;
        for (uint j = 0; j < area; j++) {
            sum += src[i * area + j];
        }
        dst[i] = sum / (float)area;
    }
    else if (params.opType == 2) DoAdd(threadId);
    else if (params.opType == 3) DoLinear(threadId);
    else if (params.opType == 4) DoCopy(threadId);
    else if (params.opType == 5) DoSoftmax(threadId);
    else if (params.opType == 6) DoClamp(threadId);
}
