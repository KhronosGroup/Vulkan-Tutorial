// Combined vertex and fragment shader for PBR rendering

// Input from vertex buffer
struct VSInput {
    [[vk::location(0)]] float3 Position;
    [[vk::location(1)]] float3 Normal;
    [[vk::location(2)]] float2 UV;
    [[vk::location(3)]] float4 Tangent;

    // Per-instance data as true matrices
    [[vk::location(4)]] column_major float4x4 InstanceModelMatrix;  // binding 1 (uses 4 locations)
    [[vk::location(8)]] column_major float4x3 InstanceNormalMatrix; // binding 1 (uses 3 locations)
};

// Output from vertex shader / Input to fragment shader
struct VSOutput {
    float4 Position : SV_POSITION;
    float3 WorldPos;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD0;
    float4 Tangent : TANGENT;
};

// Light data structure for storage buffer
// Explicit offsets ensure exact match with CPU-side layout
struct LightData {
    [[vk::offset(0)]]   float4 position;      // Directional: direction (w=0); Point/Spot/Emissive: world position (w=1)
    [[vk::offset(16)]]  float4 color;         // Light color and intensity
    // Match GLM column-major matrices in CPU
    [[vk::offset(32)]]  column_major float4x4 lightSpaceMatrix; // Light space matrix for shadow mapping
    [[vk::offset(96)]]  int   lightType;     // 0=Point, 1=Directional, 2=Spot, 3=Emissive
    [[vk::offset(100)]] float range;         // Light range
    [[vk::offset(104)]] float innerConeAngle;// For spot lights
    [[vk::offset(108)]] float outerConeAngle;// For spot lights
};

// Uniform buffer (now without fixed light arrays)
struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 camPos;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    int lightCount;             // Number of active lights (dynamic)
    int padding0;               // Padding for alignment
    float padding1;             // Padding for alignment
    float padding2;             // Padding for alignment
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
    float3 emissiveFactor;  // Emissive factor for HDR emissive sources
    float emissiveStrength; // KHR_materials_emissive_strength extension
};

// Constants
static const float PI = 3.14159265359;

// Bindings
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] Sampler2D baseColorMap;
[[vk::binding(2, 0)]] Sampler2D metallicRoughnessMap;
[[vk::binding(3, 0)]] Sampler2D normalMap;
[[vk::binding(4, 0)]] Sampler2D occlusionMap;
[[vk::binding(5, 0)]] Sampler2D emissiveMap;
[[vk::binding(6, 0)]] StructuredBuffer<LightData> lightBuffer; // Dynamic light storage buffer

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

    // Use instance matrices directly
    float4x4 instanceModelMatrix = input.InstanceModelMatrix;
    float3x3 normalMatrix3x3 = (float3x3)input.InstanceNormalMatrix;

    // Transform position to world space: entity model * instance model
    float4 worldPos = mul(ubo.model, mul(instanceModelMatrix, float4(input.Position, 1.0)));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));

    // Pass world position to fragment shader
    output.WorldPos = worldPos.xyz;

    // Transform normal to world space using reconstructed normal matrix and entity model
    float3x3 model3x3 = (float3x3)ubo.model;
    output.Normal = normalize(mul(model3x3, mul(normalMatrix3x3, input.Normal)));

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
    // Sample material textures (flip V to match glTF UV origin)
    float2 uv = float2(input.UV.x, 1.0 - input.UV.y);
    float4 baseColor = baseColorMap.Sample(uv) * material.baseColorFactor;
    float2 metallicRoughness = metallicRoughnessMap.Sample(uv).bg;
    float metallic = metallicRoughness.x * material.metallicFactor;
    float roughness = metallicRoughness.y * material.roughnessFactor;
    float ao = occlusionMap.Sample(uv).r;
    float3 emissive = emissiveMap.Sample(uv).rgb * material.emissiveFactor * material.emissiveStrength;

    // Calculate normal in tangent space
    float3 N = normalize(input.Normal);
    if (material.normalTextureSet >= 0) {
        // Apply normal mapping
        float3 tangentNormal = normalMap.Sample(uv).xyz * 2.0 - 1.0;
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

    // Calculate lighting for each light (dynamic count - no limit)
    for (int i = 0; i < ubo.lightCount; i++) {
        LightData light = lightBuffer[i];
        float3 lightVec = light.position.xyz; // w=0 indicates direction (directional), w=1 indicates position (point/spot/emissive)
        float3 lightColor = light.color.rgb;

        // Handle emissive lights differently
        if (light.lightType == 3) { // Emissive light
            // Treat emissive like a positional contributor from its stored position
          float3 L = normalize(lightVec - input.WorldPos);
            float distance = length(lightVec - input.WorldPos);
            float attenuation = 1.0 / (distance * distance);
            float3 radiance = lightColor * attenuation;

            float3 H = normalize(V + L);

            float NdotL = max(dot(N, L), 0.0);
            float NdotV = max(dot(N, V), 0.0);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);

            float D = DistributionGGX(NdotH, roughness);
            float G = GeometrySmith(NdotV, NdotL, roughness);
            float3 F = FresnelSchlick(HdotV, F0);

            float3 numerator = D * G * F;
            float denominator = 4.0 * NdotV * NdotL + 0.0001;
            float3 specular = numerator / denominator;

            float3 kS = F;
            float3 kD = float3(1.0, 1.0, 1.0) - kS;
            kD *= 1.0 - metallic;

            Lo += (kD * baseColor.rgb / PI + specular) * radiance * NdotL;

        } else if (light.lightType == 1) { // Directional light
            // For directional lights, position field stores direction; use no distance attenuation
            float3 L = normalize(-lightVec); // light direction towards the surface
            float3 radiance = lightColor;    // No attenuation with distance

            float3 H = normalize(V + L);

            float NdotL = max(dot(N, L), 0.0);
            float NdotV = max(dot(N, V), 0.0);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);

            float D = DistributionGGX(NdotH, roughness);
            float G = GeometrySmith(NdotV, NdotL, roughness);
            float3 F = FresnelSchlick(HdotV, F0);

            float3 numerator = D * G * F;
            float denominator = 4.0 * NdotV * NdotL + 0.0001;
            float3 specular = numerator / denominator;

            float3 kS = F;
            float3 kD = float3(1.0, 1.0, 1.0) - kS;
            kD *= 1.0 - metallic;

            Lo += (kD * baseColor.rgb / PI + specular) * radiance * NdotL;

        } else { // Point/Spot lights
            float3 L = normalize(lightVec - input.WorldPos);
            float distance = length(lightVec - input.WorldPos);
            float attenuation = 1.0 / (distance * distance);
            float3 radiance = lightColor * attenuation;

            float3 H = normalize(V + L);

            float NdotL = max(dot(N, L), 0.0);
            float NdotV = max(dot(N, V), 0.0);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);

            float D = DistributionGGX(NdotH, roughness);
            float G = GeometrySmith(NdotV, NdotL, roughness);
            float3 F = FresnelSchlick(HdotV, F0);

            float3 numerator = D * G * F;
            float denominator = 4.0 * NdotV * NdotL + 0.0001;
            float3 specular = numerator / denominator;

            float3 kS = F;
            float3 kD = float3(1.0, 1.0, 1.0) - kS;
            kD *= 1.0 - metallic;

            Lo += (kD * baseColor.rgb / PI + specular) * radiance * NdotL;
        }
    }

    // Add only emissive (no hardcoded ambient - use only model-defined lights)
    // Use full emissive contribution for proper HDR rendering
    float3 color = Lo + emissive;

    // Apply exposure before tone mapping for proper HDR workflow
    color *= ubo.exposure;

    // Standard Reinhard tone mapping - simple and effective
    // This prevents excessive brightness while preserving color relationships
    color = color / (1.0 + color);

    // Gamma correction (convert from linear to sRGB)
    color = pow(color, float3(1.0 / ubo.gamma, 1.0 / ubo.gamma, 1.0 / ubo.gamma));

    return float4(color, baseColor.a);
}
