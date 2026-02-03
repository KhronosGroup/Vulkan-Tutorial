// ReLU Activation Compute Shader
// Applies ReLU activation: output = max(0, input)

struct PushConstants {
    uint size;
};

[[vk::push_constant]] PushConstants params;

[[vk::binding(0, 0)]] StructuredBuffer<float> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> output;

[numthreads(256, 1, 1)]
void main(uint3 globalID : SV_DispatchThreadID) {
    uint idx = globalID.x;

    if (idx >= params.size) {
        return;
    }

    output[idx] = max(0.0f, input[idx]);
}
