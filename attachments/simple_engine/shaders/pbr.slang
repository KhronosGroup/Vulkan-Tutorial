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
    float3 GeometricNormal : NORMAL1;
    float2 UV : TEXCOORD0;
    float4 Tangent : TANGENT;
};

// Light data structure for storage buffer
struct LightData {
    [[vk::offset(0)]]   float4 position;
    [[vk::offset(16)]]  float4 color;
    [[vk::offset(32)]]  column_major float4x4 lightSpaceMatrix;
    [[vk::offset(96)]]  int   lightType;
    [[vk::offset(100)]] float range;
    [[vk::offset(104)]] float innerConeAngle;
    [[vk::offset(108)]] float outerConeAngle;
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
    int lightCount;
    int padding0;
    float padding1;
    float padding2;
    float2 screenDimensions;
};


[[vk::binding(0, 1)]] Sampler2D opaqueSceneColor;

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
    float3 emissiveFactor;
    float emissiveStrength;
    float transmissionFactor;
    int useSpecGlossWorkflow;
    float glossinessFactor;
    float3 specularFactor;
    float ior;
    bool hasEmissiveStrengthExt;
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
[[vk::binding(6, 0)]] StructuredBuffer<LightData> lightBuffer;

[[vk::push_constant]] PushConstants material;

// PBR functions
float DistributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / max(denom, 0.000001);
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float ggx1 = NdotV / (NdotV * (1.0 - k) + k);
    float ggx2 = NdotL / (NdotL * (1.0 - k) + k);
    return ggx1 * ggx2;
}

float3 FresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float3 Fresnel_Dielectric(float cosTheta, float ior) {
    float r0 = (1.0 - ior) / (1.0 + ior);
    float3 F0 = float3(r0 * r0);
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

// Vertex shader entry point
[[shader("vertex")]]
VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float4x4 instanceModelMatrix = input.InstanceModelMatrix;
    float3x3 instanceNormalMatrix3x3 = (float3x3)input.InstanceNormalMatrix;
    float4 worldPos = mul(ubo.model, mul(instanceModelMatrix, float4(input.Position, 1.0)));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));
    output.WorldPos = worldPos.xyz;
    float3x3 model3x3 = (float3x3)ubo.model;
    output.Normal = normalize(mul(model3x3, mul(instanceNormalMatrix3x3, input.Normal)));
    output.GeometricNormal = normalize(mul((float3x3)input.InstanceNormalMatrix, input.Normal));
    output.GeometricNormal = normalize(mul(model3x3, output.GeometricNormal));
    float3x3 instanceModelMatrix3x3 = (float3x3)instanceModelMatrix;
    // Tangents should be transformed by the normal matrix (inverse-transpose),
    // not the raw model matrix, to keep TBN correct under non-uniform scaling.
    float3 instTangent = mul(instanceNormalMatrix3x3, input.Tangent.xyz);
    float3 worldTangent = normalize(mul(model3x3, instTangent));
    output.UV = input.UV;
    output.Tangent = float4(worldTangent, input.Tangent.w);
    return output;
}

namespace Hable_Filmic_Tonemapping {
    static const float A = 0.15; static const float B = 0.50;
    static const float C = 0.10; static const float D = 0.20;
    static const float E = 0.02; static const float F = 0.30;
    static const float W = 11.2;
    float3 Uncharted2Tonemap(float3 x) {
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    }
}

// Fragment shader entry point
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    // --- 1. Material Properties ---
    float2 uv = float2(input.UV.x, 1.0 - input.UV.y);
    float4 baseColor = (material.baseColorTextureSet < 0) ? material.baseColorFactor : baseColorMap.Sample(uv) * material.baseColorFactor;
    float4 mrOrSpecGloss = (material.physicalDescriptorTextureSet < 0) ? float4(1.0, 1.0, 1.0, 1.0) : metallicRoughnessMap.Sample(uv);
    float metallic = 0.0, roughness = 1.0;
    float3 F0, albedo;

    if (material.useSpecGlossWorkflow != 0) {
        float3 specColorSG = mrOrSpecGloss.rgb * material.specularFactor;
        float gloss = clamp(mrOrSpecGloss.a * material.glossinessFactor, 0.0, 1.0);
        roughness = clamp(1.0 - gloss, 0.0, 1.0);
        F0 = specColorSG;
        albedo = baseColor.rgb * (1.0 - max(F0.r, max(F0.g, F0.b)));
    } else {
        // glTF metallic-roughness texture packs metallic in B, roughness in G (linear space)
        float metallicTex = mrOrSpecGloss.b;
        float roughnessTex = mrOrSpecGloss.g;
        metallic = clamp(metallicTex * material.metallicFactor, 0.0, 1.0);
        roughness = clamp(roughnessTex * material.roughnessFactor, 0.0, 1.0);
        F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);
        albedo = baseColor.rgb * (1.0 - metallic);
    }

    float ao = (material.occlusionTextureSet < 0) ? 1.0 : occlusionMap.Sample(uv).r;
    // Emissive: default to constant white when no emissive texture so authored emissiveFactor works per glTF spec.
    // If a texture is present but factor is zero, assume (1,1,1) to preserve emissive textures by default.
    float3 emissiveTex = (material.emissiveTextureSet < 0) ? float3(1.0, 1.0, 1.0) : emissiveMap.Sample(uv).rgb;
    float3 emissiveFactor = material.emissiveFactor;
    float3 emissive = emissiveTex * emissiveFactor;
    if (material.hasEmissiveStrengthExt)
      emissive *= material.emissiveStrength;

    if (material.alphaMask > 0.5 && baseColor.a < material.alphaMaskCutoff) {
        discard;
    }

    // --- 2. Normal Calculation ---
    float3 N = normalize(input.Normal);
    if (material.normalTextureSet >= 0) {
        float3 tangentNormal = normalMap.Sample(uv).xyz * 2.0 - 1.0;
        float3 T = normalize(input.Tangent.xyz);
        float handedness = input.Tangent.w;
        float3 B = normalize(cross(N, T)) * handedness;
        float3x3 TBN = float3x3(T, B, N);
        N = normalize(mul(TBN, tangentNormal));
    }

    float3 V = normalize(ubo.camPos.xyz - input.WorldPos);

    // --- 3. Opaque Lighting Calculation ---
    float3 diffuseLighting = float3(0.0,0.0,0.0), specularLighting = float3(0.0,0.0,0.0);
    for (int i = 0; i < ubo.lightCount; i++) {
        LightData light = lightBuffer[i];
        float3 L, radiance;
        if (light.lightType == 1) { L = normalize(-light.position.xyz); radiance = light.color.rgb; }
        else { L = normalize(light.position.xyz - input.WorldPos); float d = length(light.position.xyz - input.WorldPos); radiance = light.color.rgb / max(d * d, 0.001); }
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float3 H = normalize(V + L);
            float NdotV = max(dot(N, V), 0.0);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float D = DistributionGGX(NdotH, roughness);
            float G = GeometrySmith(NdotV, NdotL, roughness);
            float3 F = FresnelSchlick(HdotV, F0);
            float3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
            float3 kD = (1.0 - F) * (1.0 - metallic);
            specularLighting += specular * radiance * NdotL;
            diffuseLighting += (kD * albedo / PI) * radiance * NdotL;
        }
    }
    // specularLighting = min(specularLighting, 50.0);
    float3 ambient = albedo * ao * (0.03 * ubo.scaleIBLAmbient);
    float3 opaqueLit = diffuseLighting + specularLighting + ambient + emissive;

    // --- 4. Final Color Assembly with Refraction ---
    float3 color = opaqueLit;
    float T = clamp(material.transmissionFactor, 0.0, 1.0);
    float alphaOut = baseColor.a;

    if (T > 0.0) {
        // 1. FAKE REFRACTION (Screen-Space UV Distortion)
        // Compute UV from clip-space with perspective divide -> NDC [-1,1] -> [0,1]
        float2 ndc = input.Position.xy / max(input.Position.w, 1e-6);
        float2 baseUV = saturate(ndc * 0.5 + 0.5);
        float NdotV = max(dot(N, V), 0.0);
        float refractionStrength = (1.0 - (1.0 / material.ior)) * 0.15;
        float2 distortion = N.xy * Fresnel_Dielectric(NdotV, material.ior).r * refractionStrength;
        float border = 1.0 / max(ubo.screenDimensions.x, ubo.screenDimensions.y);
        float2 minUV = float2(border, border);
        float2 maxUV = float2(1.0 - border, 1.0 - border);
        float2 refrUV = clamp(baseUV + distortion, minUV, maxUV);
        float3 refractedColor = opaqueSceneColor.Sample(refrUV).rgb;

        // Effective transmission depends solely on the transmission factor (KHR_materials_transmission)
        // Do not multiply by baseColor.a, as many GLTF glass materials keep albedo alpha at 0
        float T_eff = saturate(T);

        refractedColor *= baseColor.rgb;

        // 3. Fresnel weighting for transmission: reduce diffuse with transmission; keep specular; add refraction
        float3 F_view = FresnelSchlick(NdotV, F0);
        float F_avg = (F_view.r + F_view.g + F_view.b) / 3.0;

        // Screen-space reflection (cheap): sample scene color using reflected view vector
        float3 R = reflect(-V, N);
        float2 reflOffset = R.xy * (0.08 * (1.0 - roughness));
        float2 reflUV = clamp(baseUV + reflOffset, minUV, maxUV);
        float3 reflectedColor = opaqueSceneColor.Sample(reflUV).rgb;
        float3 reflectWeight = F_view * (1.0 - roughness);
        float metallicBoost = lerp(0.5, 1.2, metallic);
        float3 reflectTerm = reflectedColor * reflectWeight * metallicBoost;

        float3 surfaceTerm = diffuseLighting * (1.0 - T_eff) + specularLighting + ambient + emissive + reflectTerm;
        float3 refractedTerm = refractedColor * T_eff * (1.0 - F_avg);
        color = surfaceTerm + refractedTerm;

        // 4. ALPHA: Correct glass transparency - higher transmission = lower alpha (more transparent)
        float specAlpha = saturate(((F_view.r + F_view.g + F_view.b) / 3.0) * (1.0 - 0.2 * roughness));
        // For glass: base transparency from transmission, with specular reflection preserving some opacity
        float baseTransparency = T_eff * (1.0 - F_avg * 0.5);
        alphaOut = clamp(max(baseTransparency + specAlpha * (1.0 - T_eff), specAlpha * 0.1), 0.005, 1.0);
    }

    // --- 5. Post-Processing ---
    // Apply exposure and filmic tonemapping; apply gamma only if the swapchain is NOT sRGB
    color *= ubo.exposure;

    // Uncharted2 / Hable filmic tonemap
    float3 t = Hable_Filmic_Tonemapping::Uncharted2Tonemap(color * 1.2);
    float3 w = Hable_Filmic_Tonemapping::Uncharted2Tonemap(float3(1,1,1) * (Hable_Filmic_Tonemapping::W / 1.2));
    color = t / max(w, float3(1e-6, 1e-6, 1e-6));

    // ubo.padding0 repurposed as: outputIsSRGB (1 = swapchain is sRGB, 0 = UNORM)
    if (ubo.padding0 == 0) {
        color = pow(saturate(color), float3(1.0 / ubo.gamma));
    } else {
        color = saturate(color);
    }

    return float4(color, alphaOut);
}
