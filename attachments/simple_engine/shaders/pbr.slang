// Combined vertex and fragment shader for PBR rendering

// Input from vertex buffer
struct VSInput {
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD0;
    float4 Tangent : TANGENT;
};

// Output from vertex shader / Input to fragment shader
struct VSOutput {
    float4 Position : SV_POSITION;
    float3 WorldPos : POSITION;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD0;
    float4 Tangent : TANGENT;
};

// Uniform buffer
struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 lightPositions[4];
    float4 lightColors[4];
    float4 camPos;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
};

// Push constants for material properties
struct PushConstants {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureSet;
    int physicalDescriptorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    float alphaMask;
    float alphaMaskCutoff;
};

// Constants
static const float PI = 3.14159265359;

// Bindings
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] Texture2D baseColorMap;
[[vk::binding(1, 0)]] SamplerState baseColorSampler;
[[vk::binding(2, 0)]] Texture2D metallicRoughnessMap;
[[vk::binding(2, 0)]] SamplerState metallicRoughnessSampler;
[[vk::binding(3, 0)]] Texture2D normalMap;
[[vk::binding(3, 0)]] SamplerState normalSampler;
[[vk::binding(4, 0)]] Texture2D occlusionMap;
[[vk::binding(4, 0)]] SamplerState occlusionSampler;
[[vk::binding(5, 0)]] Texture2D emissiveMap;
[[vk::binding(5, 0)]] SamplerState emissiveSampler;

[[vk::push_constant]] PushConstants material;

// PBR functions
float DistributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float ggx1 = NdotV / (NdotV * (1.0 - k) + k);
    float ggx2 = NdotL / (NdotL * (1.0 - k) + k);

    return ggx1 * ggx2;
}

float3 FresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Vertex shader entry point
[[shader("vertex")]]
VSOutput VSMain(VSInput input)
{
    VSOutput output;

    // Transform position to clip space
    float4 worldPos = mul(ubo.model, float4(input.Position, 1.0));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));

    // Pass world position to fragment shader
    output.WorldPos = worldPos.xyz;

    // Transform normal to world space
    output.Normal = normalize(mul((float3x3)ubo.model, input.Normal));

    // Pass texture coordinates
    output.UV = input.UV;

    // Pass tangent
    output.Tangent = input.Tangent;

    return output;
}

// Fragment shader entry point
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    // Sample material textures
    float4 baseColor = baseColorMap.Sample(baseColorSampler, input.UV) * material.baseColorFactor;
    float2 metallicRoughness = metallicRoughnessMap.Sample(metallicRoughnessSampler, input.UV).bg;
    float metallic = metallicRoughness.x * material.metallicFactor;
    float roughness = metallicRoughness.y * material.roughnessFactor;
    float ao = occlusionMap.Sample(occlusionSampler, input.UV).r;
    float3 emissive = emissiveMap.Sample(emissiveSampler, input.UV).rgb;

    // Calculate normal in tangent space
    float3 N = normalize(input.Normal);
    if (material.normalTextureSet >= 0) {
        // Apply normal mapping
        float3 tangentNormal = normalMap.Sample(normalSampler, input.UV).xyz * 2.0 - 1.0;
        float3 T = normalize(input.Tangent.xyz);
        float3 B = normalize(cross(N, T)) * input.Tangent.w;
        float3x3 TBN = float3x3(T, B, N);
        N = normalize(mul(tangentNormal, TBN));
    }

    // Calculate view and reflection vectors
    float3 V = normalize(ubo.camPos.xyz - input.WorldPos);
    float3 R = reflect(-V, N);

    // Calculate F0 (base reflectivity)
    float3 F0 = float3(0.04, 0.04, 0.04);
    F0 = lerp(F0, baseColor.rgb, metallic);

    // Initialize lighting
    float3 Lo = float3(0.0, 0.0, 0.0);

    // Calculate lighting for each light
    for (int i = 0; i < 4; i++) {
        float3 lightPos = ubo.lightPositions[i].xyz;
        float3 lightColor = ubo.lightColors[i].rgb;

        // Calculate light direction and distance
        float3 L = normalize(lightPos - input.WorldPos);
        float distance = length(lightPos - input.WorldPos);
        float attenuation = 1.0 / (distance * distance);
        float3 radiance = lightColor * attenuation;

        // Calculate half vector
        float3 H = normalize(V + L);

        // Calculate BRDF terms
        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        // Specular BRDF
        float D = DistributionGGX(NdotH, roughness);
        float G = GeometrySmith(NdotV, NdotL, roughness);
        float3 F = FresnelSchlick(HdotV, F0);

        float3 numerator = D * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001;
        float3 specular = numerator / denominator;

        // Energy conservation
        float3 kS = F;
        float3 kD = float3(1.0, 1.0, 1.0) - kS;
        kD *= 1.0 - metallic;

        // Add to outgoing radiance
        Lo += (kD * baseColor.rgb / PI + specular) * radiance * NdotL;
    }

    // Add ambient and emissive
    float3 ambient = float3(0.03, 0.03, 0.03) * baseColor.rgb * ao;
    float3 color = ambient + Lo + emissive;

    // HDR tonemapping and gamma correction
    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(color, float3(1.0 / ubo.gamma, 1.0 / ubo.gamma, 1.0 / ubo.gamma));

    return float4(color, baseColor.a);
}
