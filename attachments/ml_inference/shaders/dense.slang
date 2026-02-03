// Dense (Fully Connected) Layer Compute Shader
// Performs matrix multiplication: output = input * weights + bias
// input: [batch_size, input_size]
// weights: [input_size, output_size]
// bias: [output_size]
// output: [batch_size, output_size]

struct PushConstants {
    uint inputSize;
    uint outputSize;
    uint batchSize;
};

[[vk::push_constant]] PushConstants params;

[[vk::binding(0, 0)]] StructuredBuffer<float> input;
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;
[[vk::binding(2, 0)]] StructuredBuffer<float> bias;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> output;

[numthreads(256, 1, 1)]
void main(uint3 globalID : SV_DispatchThreadID) {
    uint batchIdx = globalID.x / params.outputSize;
    uint outIdx = globalID.x % params.outputSize;

    if (batchIdx >= params.batchSize || outIdx >= params.outputSize) {
        return;
    }

    float sum = bias[outIdx];

    for (uint i = 0; i < params.inputSize; ++i) {
        uint inputIdx = batchIdx * params.inputSize + i;
        uint weightIdx = outIdx * params.inputSize + i;
        sum += input[inputIdx] * weights[weightIdx];
    }

    output[batchIdx * params.outputSize + outIdx] = sum;
}
