// Combined vertex and fragment shader for textured mesh rendering
// This shader provides basic textured rendering with simple lighting

// Input from vertex buffer
struct VSInput {
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 Tangent : TANGENT;
};

// Output from vertex shader / Input to fragment shader
struct VSOutput {
    float4 Position : SV_POSITION;
    float3 WorldPos;
    float3 Normal : NORMAL;
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
    float4 worldPos = mul(ubo.model, float4(input.Position, 1.0));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));

    // Pass world position and transformed normal to fragment shader
    output.WorldPos = worldPos.xyz;
    output.Normal = normalize(mul((float3x3)ubo.model, input.Normal));
    output.TexCoord = input.TexCoord;

    return output;
}

// Fragment shader entry point
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    // Sample the texture
    float4 texColor = texSampler.Sample(input.TexCoord);

    // Simple directional lighting
    float3 lightDir = normalize(float3(0.5, 1.0, 0.3)); // Fixed light direction
    float3 normal = normalize(input.Normal);
    float lightIntensity = max(dot(normal, lightDir), 0.2); // Minimum ambient of 0.2

    // Check if texture is pure white (indicates no meaningful texture data)
    float whiteness = (texColor.r + texColor.g + texColor.b) / 3.0;
    bool isPureWhite = whiteness > 0.95; // Threshold for "pure white"

    if (isPureWhite) {
        // No texture or pure white texture: use a default color with lighting
        float3 defaultColor = float3(0.8, 0.8, 0.8); // Light gray
        return float4(defaultColor * lightIntensity, 1.0);
    } else {
        // Apply simple lighting to texture
        return float4(texColor.rgb * lightIntensity, texColor.a);
    }
}
