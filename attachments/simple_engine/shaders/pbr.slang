// Input from vertex buffer
struct VSInput {
    [[vk::location(0)]] float3 Position;
    [[vk::location(1)]] float3 Normal;
    [[vk::location(2)]] float2 UV;
    [[vk::location(3)]] float4 Tangent;

    // Per-instance data. Model matrix is a true 4x4, while the normal
    // matrix is provided as three float4 columns that match the CPU
    // layout (glm::mat3x4: 3 columns of vec4, xyz used, w unused).
    [[vk::location(4)]] column_major float4x4 InstanceModelMatrix;  // binding 1 (uses 4 locations)
    [[vk::location(8)]] float4 InstanceNormal0;                     // normal matrix column 0
    [[vk::location(9)]] float4 InstanceNormal1;                     // normal matrix column 1
    [[vk::location(10)]] float4 InstanceNormal2;                    // normal matrix column 2
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
    float4 worldPos = mul(ubo.model, mul(instanceModelMatrix, float4(input.Position, 1.0)));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));
    output.WorldPos = worldPos.xyz;

    // DEBUG EXPERIMENT:
    // Ignore the per-instance normal matrix columns (InstanceNormal0/1/2)
    // and derive normals purely from the combined entity * instance
    // transform. This helps isolate whether InstanceNormal is the
    // source of the discrepancy between nearby ground patches.
    float3x3 combined3x3 = (float3x3)mul(ubo.model, instanceModelMatrix);

    float3 worldNormal = normalize(mul(combined3x3, input.Normal));
    output.Normal = worldNormal;

    // Geometric normal (pre-normal-map) uses the same combined transform.
    float3 geomNormal = worldNormal;
    output.GeometricNormal = geomNormal;

    // For this test, transform tangents with the same combined3x3 so
    // the TBN basis is consistent with the position transform, while
    // still bypassing InstanceNormal{0,1,2}.
    float3 worldTangent = normalize(mul(combined3x3, input.Tangent.xyz));
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

// Fragment shader entry point for generic PBR materials
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
        // We flip the V coordinate for all textures (uv.y -> 1-uv.y). In
        // tangent space, this corresponds to inverting the bitangent.
        // glTF's tangent.w encodes the sign of the bitangent relative to
        // cross(N,T) in the *unflipped* UV space, so we must negate it here
        // to keep the normal map oriented correctly after the V flip.
        float handedness = -input.Tangent.w;
        float3 B = normalize(cross(N, T)) * handedness;
        float3x3 TBN = float3x3(T, B, N);
        N = normalize(mul(TBN, tangentNormal));
    }

    // Geometric (non-normal-mapped) normal for large-scale effects like Fresnel,
    // refraction and screen-space reflections. This stays stable across pixels
    // on a flat pane and helps avoid flickering when rotating the camera.
    float3 G = normalize(input.GeometricNormal);

    // --- 3. Opaque Lighting Calculation ---
    float3 V = normalize(ubo.camPos.xyz - input.WorldPos);

    float3 diffuseLighting  = float3(0.0, 0.0, 0.0);
    float3 specularLighting = float3(0.0, 0.0, 0.0);

    // Accumulate per-light diffuse and specular terms using GGX microfacet BRDF.
    for (int i = 0; i < ubo.lightCount; i++) {
        LightData light = lightBuffer[i];
        float3 L, radiance;
        if (light.lightType == 1) {
            L = normalize(-light.position.xyz);
            radiance = light.color.rgb;
        } else {
            // Point/spot/emissive: position.xyz is light position in world space
            L = normalize(light.position.xyz - input.WorldPos);
            float d = length(light.position.xyz - input.WorldPos);
            radiance = light.color.rgb / max(d * d, 0.0001);

            // For emissive-derived lights (lightType == 3), skip lights whose
            // contribution has attenuated to a numerically negligible level at
            // this fragment. This avoids paying full BRDF cost for thousands
            // of far-away emissive sources while preserving the visible ones.
            if (light.lightType == 3) {
                float radianceLuma = dot(radiance, float3(0.299, 0.587, 0.114));
                if (radianceLuma < 1e-4) {
                    continue;
                }
            }
        }
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float3 H = normalize(V + L);
            float NdotV = max(dot(N, V), 0.0);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);

            float D      = DistributionGGX(NdotH, roughness);
            float G  = GeometrySmith(NdotV, NdotL, roughness);
            float3 F     = FresnelSchlick(HdotV, F0);
            float3 spec  = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
            float3 kD    = (1.0 - F) * (1.0 - metallic);

            specularLighting += spec * radiance * NdotL;
            diffuseLighting  += (kD * albedo / PI) * radiance * NdotL;
        }
    }
    // specularLighting = min(specularLighting, 50.0);
    float3 ambient = albedo * ao * (0.03 * ubo.scaleIBLAmbient);
    float3 opaqueLit = diffuseLighting + specularLighting + ambient + emissive;

    // --- 4. Final Color Assembly (opaque only; transmission handled in GlassPSMain) ---
    float3 color    = opaqueLit;
    float  alphaOut = baseColor.a;

    // --- 5. Post-Processing ---
    // Apply exposure and filmic tonemapping; apply gamma only if the swapchain is NOT sRGB.
    color *= ubo.exposure;

    // Uncharted2 / Hable filmic tonemap, canonical form without the 1.2 pre-scale
    // so that midtones and shadows are not over-compressed relative to highlights.
    float3 t = Hable_Filmic_Tonemapping::Uncharted2Tonemap(color);
    float3 w = Hable_Filmic_Tonemapping::Uncharted2Tonemap(float3(1,1,1) * Hable_Filmic_Tonemapping::W);
    color = t / max(w, float3(1e-6, 1e-6, 1e-6));

    if (ubo.padding0 == 0) {
        color = pow(saturate(color), float3(1.0 / ubo.gamma));
    } else {
        color = saturate(color);
    }

    return float4(color, alphaOut);
}

// Fragment shader entry point specialized for architectural glass.
// Shares the same inputs and bindings as PSMain, but uses a much simpler
// and more stable shading model: primarily refraction of the opaque scene
// with a small ambient/emissive surface term. Direct diffuse/specular
// lighting and screen-space reflections are omitted to avoid global
// bright/dark flashes across large glass surfaces.
[[shader("fragment")]]
float4 GlassPSMain(VSOutput input) : SV_TARGET
{
    // --- 1. Material / texture sampling (minimal subset) ---
    float2 uv = float2(input.UV.x, 1.0 - input.UV.y);

    float4 baseColor = (material.baseColorTextureSet < 0)
        ? material.baseColorFactor
        : baseColorMap.Sample(uv) * material.baseColorFactor;

    // Ambient occlusion
    float ao = (material.occlusionTextureSet < 0)
        ? 1.0
        : occlusionMap.Sample(uv).r;

    // Emissive (same logic as PSMain)
    float3 emissiveTex = (material.emissiveTextureSet < 0)
        ? float3(1.0, 1.0, 1.0)
        : emissiveMap.Sample(uv).rgb;
    float3 emissiveFactor = material.emissiveFactor;
    float3 emissive = emissiveTex * emissiveFactor;
    if (material.hasEmissiveStrengthExt)
        emissive *= material.emissiveStrength;

    // Alpha mask discard as in PSMain
    if (material.alphaMask > 0.5 && baseColor.a < material.alphaMaskCutoff) {
        discard;
    }

    // Geometric normal for view-angle dependence and refraction
    float3 G = normalize(input.GeometricNormal);
    float3 V = normalize(ubo.camPos.xyz - input.WorldPos);

    // Base albedo used only for ambient tint and transmission tint
    float3 albedo = baseColor.rgb;

    // Simple ambient term so glass has some base brightness even when the
    // background is dark. For thin architectural glass we keep this very
    // low so the glass itself does not look "filled in" or frosted; most
    // of the perceived brightness should come from what you see *through*
    // the glass rather than from the glass surface.
    float3 ambient = albedo * (0.5 * ubo.scaleIBLAmbient);

    // Transmission factor from push constants
    float T = clamp(material.transmissionFactor, 0.0, 1.0);
    float T_eff = T;

    float3 color;
    float alphaOut = baseColor.a;

    if (T_eff > 0.0) {
        // Stylized, environment-independent glass:
        // - Do NOT sample opaqueSceneColor.
        // - Treat glass as a tinted, view-dependent surface with a subtle
        //   rim highlight. This keeps behavior stable while still giving
        //   users a sense of thickness and angle.

        // Use symmetric |N·V| so that front/back views of thin glass walls
        // behave consistently (important when looking down into glasses).
        float NdotV = abs(dot(G, V));

        // Base clear color from albedo, slightly dimmed so glass does not
        // appear self-emissive.
        float3 clearColor = albedo * 0.6;

        // Rim term stronger at grazing angles (1 - NdotV)^3, but keep it subtle
        float edge = pow(1.0 - NdotV, 3.0);
        float3 rimColor = lerp(clearColor, float3(1.0, 1.0, 1.0), 0.25);

        // Blend clear glass and rim, scaled by transmission. Both are attenuated
        // so that backgrounds remain visible through the glass.
        float3 glassBody = clearColor * (T_eff * 0.5);
        float3 rim       = rimColor * (edge * T_eff * 0.4);

        // Surface term: ambient + emissive, reduced as transmission grows and
        // globally attenuated so it does not wash out the view through glass.
        // Keep this subtle so glass does not appear frosted; for highly
        // transmissive glass (T_eff close to 1) this term becomes very small.
        float3 surfaceBase = ambient + emissive;
        float3 surfaceTerm = surfaceBase * (1.0 - T_eff) * 0.15;

        color = glassBody + rim + surfaceTerm;

        // Fresnel only influences alpha (how opaque the glass appears), not color.
        float3 F_view = FresnelSchlick(NdotV, float3(0.04, 0.04, 0.04));
        float F_avg = (F_view.r + F_view.g + F_view.b) / 3.0;

        // Opacity model for architectural glass: mostly transparent at
        // normal incidence, with a gentle Fresnel-driven increase in
        // opacity toward grazing angles. TransmissionFactor controls how
        // much of the underlying scene shows through overall.

        // Base opacity driven primarily by transmission: highly
        // transmissive glass stays very transparent.
        float baseAlpha = lerp(0.08, 0.25, 1.0 - T_eff);

        // Edge boost: more opaque at grazing angles, scaled by Fresnel.
        float edgeFactor = pow(1.0 - NdotV, 2.0);
        float edgeAlpha  = F_avg * edgeFactor * 0.8;

        alphaOut = saturate(baseAlpha + edgeAlpha);
        alphaOut = max(alphaOut, 0.02);
    } else {
        // Non-transmissive fallback: just ambient + emissive.
        color = ambient + emissive;
    }

    // --- 3. Post-processing (same as PSMain) ---
    color *= ubo.exposure;

    // Uncharted2 / Hable filmic tonemap. Use the canonical form without
    // the extra 1.2 pre-scale so that midtones and shadows are not
    // over-compressed relative to highlights.
    float3 t = Hable_Filmic_Tonemapping::Uncharted2Tonemap(color);
    float3 w = Hable_Filmic_Tonemapping::Uncharted2Tonemap(float3(1,1,1) * Hable_Filmic_Tonemapping::W);
    color = t / max(w, float3(1e-6, 1e-6, 1e-6));

    if (ubo.padding0 == 0) {
        color = pow(saturate(color), float3(1.0 / ubo.gamma));
    } else {
        color = saturate(color);
    }

    return float4(color, alphaOut);
}
