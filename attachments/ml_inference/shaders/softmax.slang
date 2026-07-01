// Softmax Activation Compute Shader
// Applies softmax: output[i] = exp(input[i]) / sum(exp(input[j]))
// Simplified implementation for small outputs (e.g., 10 classes)

struct PushConstants {
    uint size;
};

[[vk::push_constant]] PushConstants params;

[[vk::binding(0, 0)]] StructuredBuffer<float> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;

[numthreads(1, 1, 1)]
void main(uint3 globalID : SV_DispatchThreadID) {
    // Single-threaded for simplicity with small outputs
    // Find max for numerical stability
    float maxVal = input[0];
    for (uint i = 1; i < params.size; ++i) {
        maxVal = max(maxVal, input[i]);
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (uint i = 0; i < params.size; ++i) {
        float expVal = exp(input[i] - maxVal);
        output[i] = expVal;
        sum += expVal;
    }

    // Normalize
    if (sum > 0.0f) {
        for (uint i = 0; i < params.size; ++i) {
            output[i] = output[i] / sum;
        }
    }
}
