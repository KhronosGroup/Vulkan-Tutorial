// Combined vertex and fragment shader for textured mesh rendering
// This shader provides basic textured rendering with simple lighting

// Input from vertex buffer
struct VSInput {
    [[vk::location(0)]] float3 Position;
    [[vk::location(1)]] float3 Normal;
    [[vk::location(2)]] float2 TexCoord;
    [[vk::location(3)]] float4 Tangent;

    // Per-instance data as true matrices; occupy locations 4..7 and 8..10 respectively
    [[vk::location(4)]] column_major float4x4 InstanceModelMatrix;  // binding 1 (consumes 4 locations)
    [[vk::location(8)]] column_major float4x3 InstanceNormalMatrix; // binding 1 (consumes 3 locations)
};

// Output from vertex shader / Input to fragment shader
struct VSOutput {
    float4 Position : SV_POSITION;
    float3 WorldPos;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 Tangent : TANGENT; // Pass through tangent to satisfy validation layer
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

    // Use instance matrices directly (CPU uploads column-major matrices in attributes 4..10)
    float4x4 instanceModelMatrix = input.InstanceModelMatrix;
    float3x3 normalMatrix3x3 = (float3x3)input.InstanceNormalMatrix;

    // Transform position to world space: entity model * instance model
    float4 worldPos = mul(ubo.model, mul(instanceModelMatrix, float4(input.Position, 1.0)));

    // Final clip space position
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));

    // Pass world position and transformed normal to fragment shader (apply entity model to normals too)
    float3x3 model3x3 = (float3x3)ubo.model;
    output.WorldPos = worldPos.xyz;
    output.Normal = normalize(mul(model3x3, mul(normalMatrix3x3, input.Normal)));
    output.TexCoord = input.TexCoord;
    output.Tangent = input.Tangent; // Pass through tangent (unused in basic rendering)

    return output;
}

// Fragment shader entry point
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    // Sample the texture with flipped V coordinate (glTF UV origin vs Vulkan)
    float2 uv = float2(input.TexCoord.x, 1.0 - input.TexCoord.y);
    float4 texColor = texSampler.Sample(uv);

    // Simple directional lighting
    float3 lightDir = normalize(float3(0.5, 1.0, 0.3)); // Fixed light direction
    float3 normal = normalize(input.Normal);
    float lightIntensity = max(dot(normal, lightDir), 0.2); // Minimum ambient of 0.2

    // If texture is nearly white, use a default color to avoid washed-out look
    float whiteness = (texColor.r + texColor.g + texColor.b) / 3.0;
    float4 finalColor = (whiteness > 0.95)
        ? float4(float3(0.8, 0.8, 0.8) * lightIntensity, 1.0)
        : float4(texColor.rgb * lightIntensity, texColor.a);

    return finalColor;
}
