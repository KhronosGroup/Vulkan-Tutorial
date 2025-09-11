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
    float transmissionFactor; // KHR_materials_transmission
    int useSpecGlossWorkflow;   // 1 if using KHR_materials_pbrSpecularGlossiness
    float glossinessFactor;     // SpecGloss glossiness scalar
    float3 specularFactor;      // SpecGloss specular color factor
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
    float3x3 instanceNormalMatrix3x3 = (float3x3)input.InstanceNormalMatrix;

    // Transform position to world space: entity model * instance model
    float4 worldPos = mul(ubo.model, mul(instanceModelMatrix, float4(input.Position, 1.0)));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));

    // Pass world position to fragment shader
    output.WorldPos = worldPos.xyz;

    // Transform normal and tangent to world space (apply instance normal then entity model)
    float3x3 model3x3 = (float3x3)ubo.model;
    output.Normal = normalize(mul(model3x3, mul(instanceNormalMatrix3x3, input.Normal)));

    float3 instTangent = mul(instanceNormalMatrix3x3, input.Tangent.xyz);
    float3 worldTangent = normalize(mul(model3x3, instTangent));

    // Pass texture coordinates
    output.UV = input.UV;

    // Pass world-space tangent (preserve handedness in w)
    output.Tangent = float4(worldTangent, input.Tangent.w);

    return output;
}

// Fragment shader entry point
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    // Sample material textures (flip V to match glTF UV origin)
    float2 uv = float2(input.UV.x, 1.0 - input.UV.y);
    float4 baseColor = baseColorMap.Sample(uv) * material.baseColorFactor;
    // For MR workflow: metallic in B, roughness in G; For SpecGloss: RGB=specular color, A=glossiness
    float4 mrOrSpecGloss = (material.physicalDescriptorTextureSet < 0) ? float4(1.0, 1.0, 1.0, 1.0) : metallicRoughnessMap.Sample(uv);
    float metallic = 0.0;
    float roughness = 1.0;
    float3 specColorSG = float3(0.0, 0.0, 0.0);
    if (material.useSpecGlossWorkflow != 0) {
        // Specular-Glossiness workflow
        specColorSG = mrOrSpecGloss.rgb * material.specularFactor;
        float gloss = clamp(mrOrSpecGloss.a * material.glossinessFactor, 0.0, 1.0);
        roughness = clamp(1.0 - gloss, 0.0, 1.0);
        metallic = 0.0; // not used in SpecGloss
    } else {
        // Metallic-Roughness workflow
        float2 metallicRoughness = mrOrSpecGloss.bg;
        metallic = clamp(metallicRoughness.x * material.metallicFactor, 0.0, 1.0);
        roughness = clamp(metallicRoughness.y * material.roughnessFactor, 0.0, 1.0);
    }
    float ao = occlusionMap.Sample(uv).r;
    float3 emissiveTex = (material.emissiveTextureSet < 0) ? float3(1.0, 1.0, 1.0) : emissiveMap.Sample(uv).rgb;
    float3 emissive = emissiveTex * material.emissiveFactor * material.emissiveStrength;

    // Early alpha masking discard for masked materials only (glTF alphaMode == MASK)
    if (material.alphaMask > 0.5 && baseColor.a < material.alphaMaskCutoff) {
        discard;
    }

    // Calculate normal in tangent space
    float3 N = normalize(input.Normal);
    if (material.normalTextureSet >= 0) {
        // Apply normal mapping
        float3 tangentNormal = normalMap.Sample(uv).xyz * 2.0 - 1.0;

        float3 T = input.Tangent.xyz;
        bool hasTangent = dot(T, T) > 1e-6;
        if (hasTangent) {
            // Orthonormalize T against N to reduce shading artifacts
            T = normalize(T);
            T = normalize(T - N * dot(N, T));
        } else {
            // Fallback: derive tangent from screen-space derivatives of position and UVs
            float3 dp1 = ddx(input.WorldPos);
            float3 dp2 = ddy(input.WorldPos);
            float2 duv1 = ddx(uv);
            float2 duv2 = ddy(uv);
            float det = duv1.x * duv2.y - duv1.y * duv2.x;
            if (abs(det) > 1e-8) {
                float r = 1.0 / det;
                T = normalize((dp1 * duv2.y - dp2 * duv1.y) * r);
            } else {
                // Degenerate UV derivatives; fall back to a stable orthogonal vector
                float3 up = (abs(N.y) < 0.999) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
                T = normalize(cross(up, N));
            }
        }
        float handedness = hasTangent ? input.Tangent.w : 1.0;
        float3 B = normalize(cross(N, T)) * handedness;
        // Construct tangent-to-world with T, B, N basis
        float3x3 TBN = float3x3(T, B, N);
        // Transform from tangent to world space using column-vector convention
        N = normalize(mul(TBN, tangentNormal));
    }

    // Calculate view and reflection vectors
    float3 V = normalize(ubo.camPos.xyz - input.WorldPos);
    float3 R = reflect(-V, N);

    // Calculate F0 (base reflectivity)
    float3 F0;
    if (material.useSpecGlossWorkflow != 0) {
        // SpecGloss: use specular color directly
        F0 = specColorSG;
    } else {
        // MR: interpolate between dielectric and baseColor by metallic
        F0 = float3(0.04, 0.04, 0.04);
        F0 = lerp(F0, baseColor.rgb, metallic);
    }

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

    float3 ambient = baseColor.rgb * ao * (0.03 * ubo.scaleIBLAmbient);
    // Base lit color from direct lighting and emissive
    float3 opaqueLit = ambient + Lo + emissive;

    // Transmission respecting roughness and Fresnel, but without environment map
    float T = clamp(material.transmissionFactor, 0.0, 1.0);
    float NdotV_glass = max(dot(N, V), 0.0);
    float3 Fv = FresnelSchlick(NdotV_glass, F0);
    float Favg = (Fv.x + Fv.y + Fv.z) / 3.0;
    float roughTrans = clamp(1.0 - (roughness * roughness), 0.0, 1.0);
    float T_eff = T * (1.0 - Favg) * roughTrans;

    // Energy-conserving mix between opaque lighting and transmitted base color tint
    float3 color = lerp(opaqueLit, baseColor.rgb, T_eff);

    // Apply exposure before tone mapping for proper HDR workflow
    color *= ubo.exposure;

    // Standard Reinhard tone mapping - simple and effective
    color = color / (1.0 + color);

    // Gamma correction (convert from linear to sRGB)
    color = pow(color, float3(1.0 / ubo.gamma, 1.0 / ubo.gamma, 1.0 / ubo.gamma));

    // Alpha approximates remaining opacity (higher transmission -> lower alpha), clamped for readability
    float alphaOut = baseColor.a;
    if (T > 0.001) {
        alphaOut = clamp(1.0 - T_eff, 0.08, 0.60);
    }
    return float4(color, alphaOut);
}
