// Combined vertex and fragment shader for ImGui rendering

// Input from vertex buffer
struct VSInput {
    float2 Position : POSITION;
    float2 UV : TEXCOORD0;
    float4 Color : COLOR0;
};

// Output from vertex shader / Input to fragment shader
struct VSOutput {
    float4 Position : SV_POSITION;
    float2 UV : TEXCOORD0;
    float4 Color : COLOR0;
};

// Push constants for transformation
struct PushConstants {
    float2 Scale;
    float2 Translate;
};

// Bindings
[[vk::push_constant]] PushConstants pushConstants;
[[vk::binding(0, 0)]] Sampler2D fontTexture;

// Vertex shader entry point
[[shader("vertex")]]
VSOutput VSMain(VSInput input)
{
    VSOutput output;

    // Transform position
    output.Position = float4(input.Position * pushConstants.Scale + pushConstants.Translate, 0.0, 1.0);

    // Pass UV and color to fragment shader
    output.UV = input.UV;
    output.Color = input.Color;

    return output;
}

// Fragment shader entry point
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    // Sample font texture and multiply by color
    float4 color = input.Color * fontTexture.Sample(input.UV);
    return color;
}
