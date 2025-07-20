// Combined vertex and fragment shader for textured mesh rendering
// This shader provides basic textured rendering with a uniform color

// Input from vertex buffer
struct VSInput {
    float3 Position : POSITION;
    float3 Color : COLOR;
    float2 TexCoord : TEXCOORD0;
};

// Output from vertex shader / Input to fragment shader
struct VSOutput {
    float4 Position : SV_POSITION;
    float3 Color : COLOR;
    float2 TexCoord : TEXCOORD0;
};

// Uniform buffer
struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

// Bindings
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] Sampler2D texSampler;

// Vertex shader entry point
[[shader("vertex")]]
VSOutput VSMain(VSInput input)
{
    VSOutput output;

    // Transform position to clip space
    output.Position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.Position, 1.0))));

    // Pass color and texture coordinates to fragment shader
    output.Color = input.Color;
    output.TexCoord = input.TexCoord;

    return output;
}

// Fragment shader entry point
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    // Sample texture and multiply by color
    return texSampler.Sample(input.TexCoord) * float4(input.Color, 1.0);
}
