struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
    float3 inNormal;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float3   cameraPos;
};
[[vk::binding(0,0)]]
ConstantBuffer<UniformBuffer> ubo;

// TASK04: Acceleration structure binding
[[vk::binding(1,0)]]
RaytracingAccelerationStructure accelerationStructure;

[[vk::binding(2,0)]]
StructuredBuffer<uint> indexBuffer;

[[vk::binding(3,0)]]
StructuredBuffer<float2> uvBuffer;

// TASK09: Instance look-up table
struct InstanceLUT {
    uint materialID;
    uint indexBufferOffset;
};
[[vk::binding(4,0)]]
StructuredBuffer<InstanceLUT> instanceLUTBuffer;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
    float3 fragNormal;
    float3 worldPos;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    output.fragNormal = input.inNormal;
    output.worldPos = mul(ubo.model, float4(input.inPosition, 1.0)).xyz;
    return output;
}

// TASK09: Bindless resources
[[vk::binding(0,1)]]
SamplerState textureSampler;

[[vk::binding(1,1)]]
Texture2D<float4> textures[];

struct PushConstant {
    uint materialIndex;
};
[push_constant]
PushConstant pc;

static const float3 lightDir = float3(-6.0, 0.0, 6.0);

// Small epsilon to avoid self-intersection
static const float EPSILON = 0.01;

float2 intersection_uv(uint instanceID, uint primIndex, float2 barycentrics) {
    // TASK10
    return float2(0.0, 0.0);
}

void apply_reflection(float3 P, float3 N, inout float4 baseColor) {
    // TASK11
}

// TASK05: Implement ray query shadows
bool in_shadow(float3 P)
{
    bool hit = false;

    return hit;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
   float4 baseColor = textures[pc.materialIndex].Sample(textureSampler, vertIn.fragTexCoord);

   return baseColor;
}