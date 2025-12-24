/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Ray query compute shader for ray-traced rendering
// Uses Slang's ray query extension as an alternative to rasterization

import common_types;
import pbr_utils;
import lighting_utils;
import tonemapping_utils;

// GPU data structures for proper geometry and material access
struct GeometryInfo {
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
    uint vertexCount;
    uint materialIndex;
    uint indexCount;   // number of indices in the index buffer
    uint _pad0;
    // Instance -> world normal transform (3 columns; xyz used, w unused)
    float4 normalMatrix0;
    float4 normalMatrix1;
    float4 normalMatrix2;
};

struct MaterialData {
    float3 albedo;
    float metallic;
    float3 emissive;
    float roughness;
    float ao;
    float ior;
    float emissiveStrength;
    float alpha;
    float transmissionFactor;
    float alphaCutoff;
    int alphaMode; // 0=OPAQUE, 1=MASK, 2=BLEND (matches glTF)
    uint isGlass;
    uint isLiquid;

    // Thick-glass parameters (RQ-only)
    float3 absorptionColor;      // Color after traveling absorptionDistance in the medium (1=none)
    float absorptionDistance;    // Distance at which absorptionColor applies (meters)
    uint thinWalled;             // 1 = thin surface (no thickness), 0 = thick volume

    // Raster parity: texture-set flags (-1 = no texture; 0 = sample from texture table)
    int baseColorTextureSet;
    int physicalDescriptorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;

    // Ray Query texture table indices (binding 6)
    int baseColorTexIndex;
    int normalTexIndex;
    int physicalTexIndex;
    int occlusionTexIndex;
    int emissiveTexIndex;

    // Specular-glossiness workflow support
    int useSpecGlossWorkflow;
    float glossinessFactor;
    float3 specularFactor;
    int hasEmissiveStrengthExt;
    uint _padMat[3];
};

// C++ Vertex structure layout (tightly packed, 48 bytes total):
// - position: vec3 at offset 0 (12 bytes)
// - normal: vec3 at offset 12 (12 bytes)  
// - texCoord: vec2 at offset 24 (8 bytes)
// - tangent: vec4 at offset 32 (16 bytes)
// We'll read normals directly as floats to avoid PhysicalStorageBuffer alignment issues

// Ray Query uses a dedicated uniform layout to avoid CPU↔shader drift.
// IMPORTANT: This must match `RayQueryUniformBufferObject` in `renderer.h`.
struct RayQueryUniforms {
    [[vk::offset(0)]]   float4x4 model;
    [[vk::offset(64)]]  float4x4 view;
    [[vk::offset(128)]] float4x4 proj;
    [[vk::offset(192)]] float4 camPos;

    [[vk::offset(208)]] float exposure;
    [[vk::offset(212)]] float gamma;
    [[vk::offset(216)]] float scaleIBLAmbient;
    [[vk::offset(220)]] int lightCount;
    [[vk::offset(224)]] int enableRayQueryReflections;
    [[vk::offset(228)]] int enableRayQueryTransparency;

    [[vk::offset(232)]] float2 screenDimensions;
    [[vk::offset(240)]] int geometryInfoCount;
    [[vk::offset(244)]] int materialCount;
    [[vk::offset(248)]] int _pad0;
    [[vk::offset(252)]] int enableThickGlass;     // 0/1 toggle for thick-glass attenuation
    [[vk::offset(256)]] float thicknessClamp;     // max thickness in meters (safety clamp)
    [[vk::offset(260)]] float absorptionScale;    // scales sigma_a (1=as-is)
    // Ray Query shadows: 0/1 enable (wired from C++ as `enableRayQueryShadows`)
    [[vk::offset(264)]] int _pad1;
    // Ray Query soft shadows (area-light approximation)
    [[vk::offset(268)]] int shadowSampleCount;    // 1 = hard shadows (single shadow ray)
    [[vk::offset(272)]] float shadowSoftness;     // 0 = hard; otherwise scales effective light radius (see shader)
    [[vk::offset(276)]] float reflectionIntensity; // user control
    [[vk::offset(280)]] float _padShadow1;
    [[vk::offset(284)]] float _padShadow2;
};

// Compute shader descriptor bindings
[[vk::binding(0, 0)]] ConstantBuffer<RayQueryUniforms> ubo;
[[vk::binding(1, 0)]] RaytracingAccelerationStructure tlas;  // Top-level acceleration structure
[[vk::binding(2, 0)]] RWTexture2D<float4> outputImage;       // Output render target
[[vk::binding(3, 0)]] StructuredBuffer<LightData> lightBuffer;
[[vk::binding(4, 0)]] StructuredBuffer<GeometryInfo> geometryInfoBuffer;  // Geometry vertex/index addresses
[[vk::binding(5, 0)]] StructuredBuffer<MaterialData> materialBuffer;      // Material properties
// Fixed-size Ray Query texture table (combined image samplers)
// Must match Renderer::RQ_MAX_TEX in C++ (currently 2048)
static const uint RQ_MAX_TEX = 2048;
[[vk::binding(6, 0)]] Sampler2D baseColorTex[RQ_MAX_TEX];

// (No debug buffer in production layout)

// (No debug toggles in production)

// NOTE: Ray Query debugPrintf/printf diagnostics are intentionally removed.
// Keep Ray Query VVL-clean and log-noise-free by default.

float3 materialDebugColor(uint materialIndex) {
    float u = float(materialIndex + 1);
    float3 h = frac(float3(0.06711056, 0.00583715, 0.2065) * u);
    return frac(h * 2.6180339);
}

// Simple ray-scene intersection result
struct HitInfo {
    bool hit;
    float t;
    float3 worldPos;
    float3 normal;
    float3 baseColor;    // base color (factor * texture), not lit
    float3 color;
    float3 F0;
    float roughness;
    float metallic;
    float transmission;  // Added for glass/transparency support
    float ior;
    bool isGlass;        // Material is glass (should always have reflections/transparency)
    bool isLiquid;
    bool thinWalled;     // true = thin surface (no thickness), false = thick volume
    int alphaMode;       // 0=OPAQUE, 1=MASK, 2=BLEND
    float opacity;       // 0..1, derived from baseColor alpha (factor * texture)
    uint instanceId;     // Committed instance ID (for debug coloring)
    float2 uv;           // Interpolated UV at the hit (glTF V flip already applied)
    uint texIndex;       // Resolved baseColor texture index used for sampling (for stats/debug)
    uint materialIndex;  // Resolved/clamped material index used
};

static const float RQ_RAY_EPS = 0.002;

uint rqHash(uint v)
{
    // Thomas Wang 32-bit integer hash
    v = (v ^ 61u) ^ (v >> 16u);
    v *= 9u;
    v = v ^ (v >> 4u);
    v *= 0x27d4eb2du;
    v = v ^ (v >> 15u);
    return v;
}

float rqRand01(inout uint state)
{
    state = rqHash(state);
    // 24-bit mantissa-ish
    return float(state & 0x00FFFFFFu) * (1.0 / 16777216.0);
}

float2 rqSampleDisk(inout uint state)
{
    // Uniform disk sampling
    float u1 = rqRand01(state);
    float u2 = rqRand01(state);
    float r = sqrt(max(u1, 0.0));
    float a = 6.28318530718 * u2;
    return float2(cos(a), sin(a)) * r;
}

float3 skyColor(float3 dir) {
    float t = saturate(0.5 * (dir.y + 1.0));
    float3 a = float3(0.06, 0.06, 0.09);
    float3 b = float3(0.20, 0.25, 0.32);
    return lerp(a, b, t);
}

// Heuristic explicit-LOD for compute sampling (no implicit derivatives available)
float computeTextureLOD(float3 worldPos, float roughness) {
    // Approximate screen-space footprint from view distance and modulate by roughness
    float d = length(ubo.camPos.xyz - worldPos);
    // Tune 0.25 scale empirically for this scene; avoids over-sharp minification
    float lod = log2(max(d * 0.25, 1.0));
    lod *= lerp(0.6, 1.4, saturate(roughness));
    return max(lod, 0.0);
}

float3 shadeWithSecondaryRays(float3 rayOrigin, float3 rayDir, HitInfo hit, inout uint rngState) {
    float3 base = max(hit.color, 0.0);
    int maxBounces = clamp(ubo._pad0, 0, 10);

    float3 N = hit.normal;
    float3 V = normalize(-rayDir);
    float NdotV = abs(dot(N, V));

    // Fresnel for reflection weighting
    float3 F = FresnelSchlick(NdotV, hit.F0);
    if (hit.transmission > 0.01 || hit.isGlass) {
        F = Fresnel_Dielectric(NdotV, hit.ior);
    }
    float Fr = saturate((F.r + F.g + F.b) * (1.0 / 3.0));

    // Transmission gate
    float opacity = clamp(hit.opacity, 0.0, 1.0);
    float blendTransmission = (hit.alphaMode == 2) ? (1.0 - opacity) : 0.0;
    float physicalTransmission = clamp(hit.transmission, 0.0, 1.0);

    // Many scenes tag architectural glass via a material hint (`hit.isGlass`) even when
    // `KHR_materials_transmission` is not authored. Provide a robust fallback so glass
    // does not turn black and so interior lighting remains visible through windows.
    float glassTransmission = 0.0;
    if (hit.isGlass) {
        glassTransmission = max(physicalTransmission, (1.0 - opacity));
        if (glassTransmission < 0.01) {
            glassTransmission = 0.90;
        }
    }

    float T = 0.0;
    if (ubo.enableRayQueryTransparency != 0 && !hit.isLiquid) {
        T = max(physicalTransmission, max(blendTransmission, glassTransmission));
    }

    // Reflection ray (chain up to maxBounces)
    float3 reflCol = float3(0.0, 0.0, 0.0);
    bool doRefl = (ubo.enableRayQueryReflections != 0) && (Fr > 1e-4);
    if (doRefl) {
        float3 ro = hit.worldPos + hit.normal * RQ_RAY_EPS;
        float3 rd = normalize(reflect(rayDir, N));

        float3 last = skyColor(rd);
        for (int b = 0; b < maxBounces; ++b) {
            HitInfo rh = traceRay(ro, rd, RQ_RAY_EPS, 10000.0, rngState);
            last = rh.hit ? max(rh.color, 0.0) : skyColor(rd);
            if (!rh.hit) {
                break;
            }
            // Next bounce
            float3 Nr = normalize(rh.normal);
            rd = normalize(reflect(rd, Nr));
            ro = rh.worldPos + Nr * RQ_RAY_EPS;
        }
        reflCol = last * ubo.reflectionIntensity;
    }

    // Transmission ray (iterative loop to support multiple layers)
    float3 thruCol = skyColor(rayDir);
    bool doThru = (T > 1e-4);
    if (doThru) {
        float3 curRd = rayDir;
        float3 curRo = hit.worldPos;
        float3 accumTint = float3(1.0);
        HitInfo curHit = hit;

        for (int bounce = 0; bounce < maxBounces + 1; ++bounce) {
            float curPhysT = clamp(curHit.transmission, 0.0, 1.0);
            float curOpacity = clamp(curHit.opacity, 0.0, 1.0);
            float curGlassT = curHit.isGlass ? max(curPhysT, 1.0 - curOpacity) : 0.0;
            if (curHit.isGlass && curGlassT < 0.01) curGlassT = 0.9;
            float curT = max(curPhysT, max((curHit.alphaMode == 2 ? 1.0 - curOpacity : 0.0), curGlassT));

            if (curT < 1e-4) {
                thruCol = accumTint * curHit.color;
                break;
            }

            float3 nextRd = curRd;
            bool refracts = (curPhysT > 1e-4 || curHit.isGlass);
            if (refracts) {
                float3 Nn = curHit.normal;
                float eta = 1.0 / max(curHit.ior, 1.0);
                // Determine if we're entering or exiting based on ray direction vs normal
                if (dot(curRd, curHit.normal) > 0.0) {
                    Nn = -curHit.normal;
                    eta = max(curHit.ior, 1.0);
                }
                float3 refrDir;
                if (refract(curRd, Nn, eta, refrDir)) {
                    nextRd = normalize(refrDir);
                } else {
                    // Total internal reflection
                    thruCol = float3(0, 0, 0);
                    break;
                }
            }

            // Accumulate tint for refractive surfaces
            if (refracts) {
                float3 tint = max(clamp(curHit.baseColor, 0.0, 1.0), float3(0.5, 0.5, 0.5));
                accumTint *= tint;
            }

            // Trace from just inside the entry point to reduce self-hits.
            float3 startPos = curHit.worldPos + nextRd * (4.0 * RQ_RAY_EPS);

            // Volumetric absorption for THICK glass (skip for thin-walled)
            if (ubo.enableThickGlass != 0 && !curHit.thinWalled && refracts) {
                HitInfo exitHit = traceRay(startPos, nextRd, (4.0 * RQ_RAY_EPS), 10000.0, rngState);
                bool haveExitSameSurface = exitHit.hit &&
                                          (exitHit.instanceId == curHit.instanceId) &&
                                          (exitHit.materialIndex == curHit.materialIndex);
                if (haveExitSameSurface) {
                    float thickness = min(distance(exitHit.worldPos, curHit.worldPos), max(0.0, ubo.thicknessClamp));
                    if (thickness > 1e-6) {
                        uint mi = min(curHit.materialIndex, (uint)max(0, ubo.materialCount-1));
                        MaterialData m = materialBuffer[mi];
                        float3 C = saturate(m.absorptionColor);
                        float D = max(m.absorptionDistance, 1e-4);
                        float3 sigma_a = -log(max(C, 1e-3)) / D;
                        sigma_a *= max(ubo.absorptionScale, 0.0);
                        accumTint *= saturate(exp(-sigma_a * thickness));
                    }
                    startPos = exitHit.worldPos + nextRd * (4.0 * RQ_RAY_EPS);
                }
            }

            // Trace the scene beyond this layer
            HitInfo nextHit = traceRay(startPos, nextRd, RQ_RAY_EPS, 10000.0, rngState);
            if (!nextHit.hit) {
                thruCol = accumTint * skyColor(nextRd);
                break;
            }

            // Assume opaque for now; check if we should continue looping
            thruCol = accumTint * nextHit.color;

            float nPhysT = clamp(nextHit.transmission, 0.0, 1.0);
            float nOpacity = clamp(nextHit.opacity, 0.0, 1.0);
            float nGlassT = nextHit.isGlass ? max(nPhysT, 1.0 - nOpacity) : 0.0;
            if (nextHit.isGlass && nGlassT < 0.01) nGlassT = 0.9;
            float nT = max(nPhysT, max((nextHit.alphaMode == 2 ? 1.0 - nOpacity : 0.0), nGlassT));

            // Stop if the next hit is opaque or we reached max bounces
            if (nT < 1e-4 || bounce == maxBounces) {
                break;
            }

            // Refractive radiance compensation (mitigate amplification)
            if (refracts) {
                bool entering = (dot(curRd, curHit.normal) < 0.0);
                float eta_ratio = entering ? (1.0 / max(curHit.ior, 1.0)) : max(curHit.ior, 1.0);
                if (entering) {
                    float invEta2 = 1.0 / max(eta_ratio * eta_ratio, 1e-4);
                    accumTint *= clamp(invEta2, 0.0, 1.0);
                }
            }

            // Move to the next transmissive layer
            curHit = nextHit;
            curRd = nextRd;
        }
    }

    if (T > 1e-4) {
        // Transmissive: energy-conserving mix using Fresnel reflectance Fr and
        // transmission factor T (authored/heuristic). Transmission weight Ft = (1-Fr)*T.
        float Fr_s = saturate(Fr);
        float Ft = saturate((1.0 - Fr_s) * T);
        float sum = Fr_s + Ft;
        if (sum > 1.0) {
            // Normalize to avoid accidental gain (should rarely trigger)
            Fr_s /= sum;
            Ft   /= sum;
        }

        float3 mixed = Fr_s * reflCol + Ft * thruCol;

        // For alpha BLEND without physical transmission, composite with base using coverage.
        if (hit.alphaMode == 2 && physicalTransmission <= 1e-4 && !hit.isGlass) {
            return base * opacity + mixed * (1.0 - opacity);
        }

        // Glass/transmission path: return the energy-conserving mixture directly,
        // plus direct surface highlights (specular + emissive) from the glass surface.
        return mixed + base;
    }

    // Opaque: add a controlled reflection contribution (avoids double-counting too much)
    float reflWeight = doRefl ? (Fr * (1.0 - clamp(hit.roughness, 0.0, 1.0))) : 0.0;
    return lerp(base, reflCol, reflWeight);
}

float2 computeTriangleUV(uint instIndex, uint primitiveIndex, float2 bary) {
    float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    if (instIndex >= uint(max(0, ubo.geometryInfoCount))) return float2(0.0, 0.0);
    GeometryInfo geoInfo = geometryInfoBuffer[instIndex];
    if (geoInfo.vertexBufferAddress == 0 || geoInfo.indexBufferAddress == 0) return float2(0.0, 0.0);

    uint triCount = geoInfo.indexCount / 3u;
    if (primitiveIndex >= triCount) return float2(0.0, 0.0);

    uint* indexBuffer = (uint*)geoInfo.indexBufferAddress;
    float* vertexBuffer = (float*)geoInfo.vertexBufferAddress;
    uint idxBase = primitiveIndex * 3u;
    uint i0 = indexBuffer[idxBase + 0u];
    uint i1 = indexBuffer[idxBase + 1u];
    uint i2 = indexBuffer[idxBase + 2u];
    if (i0 >= geoInfo.vertexCount || i1 >= geoInfo.vertexCount || i2 >= geoInfo.vertexCount) return float2(0.0, 0.0);

    uint vertexStride = 12; // floats
    float2 uv0 = float2(vertexBuffer[i0 * vertexStride + 6], vertexBuffer[i0 * vertexStride + 7]);
    float2 uv1 = float2(vertexBuffer[i1 * vertexStride + 6], vertexBuffer[i1 * vertexStride + 7]);
    float2 uv2 = float2(vertexBuffer[i2 * vertexStride + 6], vertexBuffer[i2 * vertexStride + 7]);
    float2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
    uv.y = 1.0 - uv.y; // flip V for glTF
    return uv;
}

float computeBaseColorAlpha(MaterialData material, uint instIndex, uint primitiveIndex, float2 bary) {
    float alpha = material.alpha;
    if (material.baseColorTextureSet >= 0) {
        float2 uv = computeTriangleUV(instIndex, primitiveIndex, bary);
        uint tiBase = (uint)min(max(material.baseColorTexIndex, 0), int(RQ_MAX_TEX - 1));
        float4 baseColor = baseColorTex[NonUniformResourceIndex(tiBase)].SampleLevel(uv, 0.0);
        baseColor *= float4(material.albedo, material.alpha);
        alpha = baseColor.a;
    }
    return alpha;
}

// Shadow query: returns true when there is an occluder between origin and tMax along direction.
// - MASK materials are alpha-tested against alphaCutoff.
// - BLEND materials are treated as non-occluding for shadows.
bool traceShadowOccluded(float3 origin, float3 direction, float tMin, float tMax) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;

    RayQuery<RAY_FLAG_NONE> q;
    uint mask = 0xFF;
    // Force non-opaque so we can decide in-shader whether a candidate occludes.
    // This is required so transmissive / glass materials do not fully block shadow rays.
    q.TraceRayInline(
        tlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_NON_OPAQUE,
        mask,
        ray
    );

    int maxIterations = 1000;
    int iteration = 0;
    while (q.Proceed() && iteration < maxIterations) {
        iteration++;

        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            uint instIndex = q.CandidateInstanceID();
            if (instIndex < uint(max(0, ubo.geometryInfoCount))) {
                GeometryInfo geoInfoC = geometryInfoBuffer[instIndex];
                uint materialIndexC = 0u;
                if (ubo.materialCount > 0) {
                    materialIndexC = min(geoInfoC.materialIndex, (uint)(ubo.materialCount - 1));
                }
                MaterialData matC = materialBuffer[materialIndexC];

                bool accept = true;

                // Treat transmissive surfaces as non-occluding for shadows.
                // NOTE: This includes "opaque-but-glass" materials (alphaMode==OPAQUE with isGlass hint).
                bool transmissive = (matC.isGlass != 0) || (matC.transmissionFactor > 0.01);
                if (transmissive || matC.alphaMode == 2) {
                    accept = false;
                } else if (matC.alphaMode == 1) {
                    float alpha = computeBaseColorAlpha(matC, instIndex, q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics());
                    accept = (alpha >= matC.alphaCutoff);
                }

                if (accept) {
                    q.CommitNonOpaqueTriangleHit();
                    break;
                }
            }
        }
    }

    return (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
}

// Calculate refraction direction using Snell's law
// Returns true if refraction occurs, false if total internal reflection
bool refract(float3 I, float3 N, float eta, out float3 refracted) {
    float cosi = -dot(N, I);
    float cost2 = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if (cost2 > 0.0) {
        refracted = eta * I + (eta * cosi - sqrt(cost2)) * N;
        return true;
    }
    refracted = float3(0, 0, 0); // Initialize even on failure
    return false; // Total internal reflection
}

// Perform ray query and return hit information with proper vertex normals and material properties
HitInfo traceRay(float3 origin, float3 direction, float tMin, float tMax, inout uint rngState) {
    HitInfo result;
    result.hit = false;
    result.t = tMax;
    result.uv = float2(0.0, 0.0);
    result.texIndex = 0u;
    result.materialIndex = 0u;
    result.baseColor = float3(1.0, 1.0, 1.0);
    result.color = float3(0.0, 0.0, 0.0);
    result.alphaMode = 0;
    result.opacity = 1.0;
    
    // Create ray query object
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    
    // Initialize ray query.
    // We do NOT force opaque because we need to alpha-test MASK materials in-shader
    // (RayQuery inline traversal has no any-hit shader).
    RayQuery<RAY_FLAG_NONE> q;
    uint primaryMask = 0xFF;
    q.TraceRayInline(
        tlas,
        RAY_FLAG_NONE,
        primaryMask,
        ray
    );
    
    // Process ray query - loop until Proceed() returns false
    // For opaque geometry, this finds the closest hit automatically
    // Add safety limit to prevent infinite loops
    int maxIterations = 1000;
    int iteration = 0;
    while (q.Proceed() && iteration < maxIterations) {
        iteration++;

        // Alpha-mask handling: emulate any-hit by inspecting candidate triangle alpha
        // and committing only when the candidate passes alpha test.
        // Slang/HLSL ray query candidate type for triangle intersections.
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            uint instIndex = q.CandidateInstanceID();
            if (instIndex < uint(max(0, ubo.geometryInfoCount))) {
                GeometryInfo geoInfoC = geometryInfoBuffer[instIndex];
                uint materialIndexC = 0u;
                if (ubo.materialCount > 0) {
                    materialIndexC = min(geoInfoC.materialIndex, (uint)(ubo.materialCount - 1));
                }
                MaterialData matC = materialBuffer[materialIndexC];

                bool accept = true;
                if (matC.alphaMode == 1) {
                    float alpha = computeBaseColorAlpha(matC, instIndex, q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics());
                    accept = (alpha >= matC.alphaCutoff);
                }

                if (accept) {
                    q.CommitNonOpaqueTriangleHit();
                }
            }
        }
    }
    
    // Check if we hit anything
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        result.hit = true;
        result.t = q.CommittedRayT();
        result.worldPos = origin + direction * result.t;
        // Use CommittedInstanceID() which returns the instance custom index we set on CPU
        // (our per-instance sequential index that matches geometryInfoBuffer order).
        uint instIndex = q.CommittedInstanceID();
        result.instanceId = instIndex;
        
        // Get barycentric coordinates
        float2 bary = q.CommittedTriangleBarycentrics();
        float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
        
        // PROPER GEOMETRY DATA FETCHING WITH SAFETY CHECKS
        // Per-instance geometry info index equals the instance custom index we assigned on CPU.
        uint blasIndex = instIndex;
        uint primitiveIndex = q.CommittedPrimitiveIndex();

        // Validate instance index is in bounds of geometry info buffer
        if (blasIndex >= uint(max(0, ubo.geometryInfoCount))) {
            // Invalid BLAS index, use default values
            result.normal = float3(0, 1, 0);
            result.baseColor = float3(0.8, 0.8, 0.8);
            result.color = float3(0.8, 0.8, 0.8);
            result.metallic = 0.0;
            result.roughness = 0.5;
            result.transmission = 0.0;
            result.isGlass = false;
            result.alphaMode = 0;
            result.opacity = 1.0;
            result.thinWalled = true;
            return result;
        }
        
        // Get geometry info for this BLAS (unique mesh)
        GeometryInfo geoInfo = geometryInfoBuffer[blasIndex];
        
        // Fetch material first so that even if geometry fetch fails we can still show a material color
        // Material property fetch with bounds clamp
        // Clamp material index to valid range to prevent out-of-bounds access
        uint materialIndex = 0u;
        if (ubo.materialCount > 0) {
            materialIndex = min(geoInfo.materialIndex, (uint) (ubo.materialCount - 1));
        }
        result.materialIndex = materialIndex;
        MaterialData material = materialBuffer[materialIndex];

        // Validate buffer addresses are non-zero (geometry may still be streaming)
        if (geoInfo.vertexBufferAddress == 0 || geoInfo.indexBufferAddress == 0) {
            // Geometry not ready: show a stable material-derived color so the frame isn't flat gray
            result.normal = float3(0, 1, 0);
            float3 albedoBase = float3(material.albedo);
            result.baseColor = albedoBase;
            float u0 = float(materialIndex + 1);
            float3 h0 = frac(float3(0.06711056, 0.00583715, 0.2065) * u0);
            float3 hashColor0 = frac(h0 * 2.6180339);
            result.color = saturate(0.7 * hashColor0 + 0.3 * albedoBase);
            result.metallic = material.metallic;
            result.roughness = material.roughness;
            result.transmission = material.transmissionFactor;
            result.isGlass = (material.isGlass != 0);
            result.alphaMode = material.alphaMode;
            result.opacity = clamp(material.alpha, 0.0, 1.0);
            result.thinWalled = (material.thinWalled != 0);
            return result;
        }
        
        // Cast device addresses to typed pointers for index buffer
        uint* indexBuffer = (uint*)geoInfo.indexBufferAddress;
        float* vertexBuffer = (float*)geoInfo.vertexBufferAddress;

        // Validate primitive index is within range of available triangles
        uint triCount = geoInfo.indexCount / 3u;
        if (primitiveIndex >= triCount) {
            // Out of bounds primitive; show material-derived color
            result.normal = float3(0, 1, 0);
            float3 albedoBase = float3(material.albedo);
            result.baseColor = albedoBase;
            float u1 = float(materialIndex + 1);
            float3 h1 = frac(float3(0.06711056, 0.00583715, 0.2065) * u1);
            float3 hashColor1 = frac(h1 * 2.6180339);
            result.color = saturate(0.7 * hashColor1 + 0.3 * albedoBase);
            result.metallic = material.metallic;
            result.roughness = material.roughness;
            result.transmission = material.transmissionFactor;
            result.isGlass = (material.isGlass != 0);
            result.alphaMode = material.alphaMode;
            result.opacity = clamp(material.alpha, 0.0, 1.0);
            result.thinWalled = (material.thinWalled != 0);
            return result;
        }

        // Fetch triangle indices
        uint idxBase = primitiveIndex * 3u;
        uint i0 = indexBuffer[idxBase + 0u];
        uint i1 = indexBuffer[idxBase + 1u];
        uint i2 = indexBuffer[idxBase + 2u];
        
        if (i0 >= geoInfo.vertexCount || i1 >= geoInfo.vertexCount || i2 >= geoInfo.vertexCount) {
            // Out of bounds vertex indices - still present material-derived color
            result.normal = float3(0, 1, 0);
            float3 albedoBase = float3(material.albedo);
            result.baseColor = albedoBase;
            float u2 = float(materialIndex + 1);
            float3 h2 = frac(float3(0.06711056, 0.00583715, 0.2065) * u2);
            float3 hashColor2 = frac(h2 * 2.6180339);
            result.color = saturate(0.7 * hashColor2 + 0.3 * albedoBase);
            result.metallic = material.metallic;
            result.roughness = material.roughness;
            result.transmission = material.transmissionFactor;
            result.isGlass = (material.isGlass != 0);
            result.alphaMode = material.alphaMode;
            result.opacity = clamp(material.alpha, 0.0, 1.0);
            result.thinWalled = (material.thinWalled != 0);
            return result;
        }
        
        // Vertex layout: pos(3) + normal(3) + texCoord(2) + tangent(4) = 12 floats per vertex
        uint vertexStride = 12;  // floats per vertex
        
        // Read object-space normals directly (offset 3 floats for position, then 3 floats for normal)
        float3 n0 = float3(vertexBuffer[i0 * vertexStride + 3],
                           vertexBuffer[i0 * vertexStride + 4],
                           vertexBuffer[i0 * vertexStride + 5]);
        float3 n1 = float3(vertexBuffer[i1 * vertexStride + 3],
                           vertexBuffer[i1 * vertexStride + 4],
                           vertexBuffer[i1 * vertexStride + 5]);
        float3 n2 = float3(vertexBuffer[i2 * vertexStride + 3],
                           vertexBuffer[i2 * vertexStride + 4],
                           vertexBuffer[i2 * vertexStride + 5]);
        
        // Interpolate normal using barycentric coordinates
        float3 interpolatedNormal = n0 * barycentrics.x +
                                     n1 * barycentrics.y +
                                     n2 * barycentrics.z;
        
        // Transform to world space using per-instance normal matrix
        float3x3 nrmMat = float3x3(geoInfo.normalMatrix0.xyz, geoInfo.normalMatrix1.xyz, geoInfo.normalMatrix2.xyz);
        float3 N = normalize(mul(nrmMat, interpolatedNormal));
        result.normal = N;

        // Read UVs and sample baseColor texture if available
        float2 uv0 = float2(vertexBuffer[i0 * vertexStride + 6], vertexBuffer[i0 * vertexStride + 7]);
        float2 uv1 = float2(vertexBuffer[i1 * vertexStride + 6], vertexBuffer[i1 * vertexStride + 7]);
        float2 uv2 = float2(vertexBuffer[i2 * vertexStride + 6], vertexBuffer[i2 * vertexStride + 7]);
        float2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
        uv.y = 1.0 - uv.y; // flip V for glTF
        result.uv = uv;
        
        // --- PBR texture sampling (explicit LOD 0 for compute) ---
        float2 uvSample = uv;
        float4 baseColor = float4(material.albedo, material.alpha);
        float lodHint = computeTextureLOD(result.worldPos, material.roughness);
        if (material.baseColorTextureSet >= 0) {
            uint tiBase = (uint)min(max(material.baseColorTexIndex, 0), int(RQ_MAX_TEX - 1));
            baseColor = baseColorTex[NonUniformResourceIndex(tiBase)].SampleLevel(uvSample, lodHint);
            baseColor *= float4(material.albedo, material.alpha);
            result.texIndex = tiBase;
        } else {
            result.texIndex = 0u;
        }

        result.baseColor = saturate(baseColor.rgb);

        result.alphaMode = material.alphaMode;
        result.opacity = clamp(baseColor.a, 0.0, 1.0);

        // Physical descriptor: metallic-roughness (default) or spec-gloss
        float4 mrOrSpecGloss = float4(1.0, 1.0, 1.0, 1.0);
        if (material.physicalDescriptorTextureSet >= 0) {
            uint tiPhys = (uint)min(max(material.physicalTexIndex, 0), int(RQ_MAX_TEX - 1));
            mrOrSpecGloss = baseColorTex[NonUniformResourceIndex(tiPhys)].SampleLevel(uvSample, lodHint);
        }

        float metallic = 0.0;
        float roughness = 1.0;
        float3 F0 = float3(0.04, 0.04, 0.04);
        float3 albedo = baseColor.rgb;

        if (material.useSpecGlossWorkflow != 0) {
            float3 specColorSG = mrOrSpecGloss.rgb * material.specularFactor;
            float gloss = clamp(mrOrSpecGloss.a * material.glossinessFactor, 0.0, 1.0);
            roughness = clamp(1.0 - gloss, 0.0, 1.0);
            F0 = specColorSG;
            float maxF0 = max(F0.r, max(F0.g, F0.b));
            albedo = baseColor.rgb * (1.0 - maxF0);
        } else {
            float metallicTex = mrOrSpecGloss.b;
            float roughnessTex = mrOrSpecGloss.g;
            metallic = clamp(metallicTex * material.metallic, 0.0, 1.0);
            roughness = clamp(roughnessTex * material.roughness, 0.0, 1.0);
            F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);
            albedo = baseColor.rgb * (1.0 - metallic);
        }

        // Ambient occlusion
        float ao = material.ao;
        if (material.occlusionTextureSet >= 0) {
            uint tiAO = (uint)min(max(material.occlusionTexIndex, 0), int(RQ_MAX_TEX - 1));
            ao *= baseColorTex[NonUniformResourceIndex(tiAO)].SampleLevel(uvSample, lodHint).r;
        }

        // Emissive
        float3 emissiveTex = float3(1.0, 1.0, 1.0);
        if (material.emissiveTextureSet >= 0) {
            uint tiE = (uint)min(max(material.emissiveTexIndex, 0), int(RQ_MAX_TEX - 1));
            emissiveTex = baseColorTex[NonUniformResourceIndex(tiE)].SampleLevel(uvSample, lodHint).rgb;
        }
        float3 emissive = emissiveTex * material.emissive;
        if (material.hasEmissiveStrengthExt != 0) {
            emissive *= material.emissiveStrength;
        }

        // Store F0 so the caller can compute Fresnel for reflection/transmission.
        result.F0 = F0;
        result.ior = max(material.ior, 1.0);
        result.isLiquid = (material.isLiquid != 0);

        // Normal mapping (tangent space)
        if (material.normalTextureSet >= 0) {
            uint tiN = (uint)min(max(material.normalTexIndex, 0), int(RQ_MAX_TEX - 1));
            float3 tangentNormal = baseColorTex[NonUniformResourceIndex(tiN)].SampleLevel(uvSample, lodHint).xyz * 2.0 - 1.0;
            
            // Read and interpolate tangent (object-space) from vertex buffer
            float4 t0 = float4(vertexBuffer[i0 * vertexStride + 8],
                               vertexBuffer[i0 * vertexStride + 9],
                               vertexBuffer[i0 * vertexStride + 10],
                               vertexBuffer[i0 * vertexStride + 11]);
            float4 t1 = float4(vertexBuffer[i1 * vertexStride + 8],
                               vertexBuffer[i1 * vertexStride + 9],
                               vertexBuffer[i1 * vertexStride + 10],
                               vertexBuffer[i1 * vertexStride + 11]);
            float4 t2 = float4(vertexBuffer[i2 * vertexStride + 8],
                               vertexBuffer[i2 * vertexStride + 9],
                               vertexBuffer[i2 * vertexStride + 10],
                               vertexBuffer[i2 * vertexStride + 11]);
            float4 tan4 = t0 * barycentrics.x + t1 * barycentrics.y + t2 * barycentrics.z;
            float3 T = normalize(mul(nrmMat, tan4.xyz));

            // We flip V for glTF (uv.y = 1-uv.y). In tangent space this inverts bitangent.
            // glTF tangent.w encodes bitangent sign in unflipped UV space, so negate it.
            float handedness = -tan4.w;
            float3 B = normalize(cross(N, T)) * handedness;
            float3x3 TBN = float3x3(T, B, N);
            N = normalize(mul(TBN, tangentNormal));
            result.normal = N;
        }

        // --- Direct lighting (GGX) ---
        float3 V = normalize(-direction);
        N = result.normal;
        // Flip normal for backfaces so lighting works inside single-sided rooms
        if (dot(N, V) < 0.0) {
            N = -N;
            result.normal = N;
        }

        // Effective transmission for diffuse scaling (to avoid over-brightening transmissive surfaces)
        float T_diff = clamp(material.transmissionFactor, 0.0, 1.0);
        if (material.isGlass != 0) T_diff = max(T_diff, 0.9);
        if (material.alphaMode == 2) T_diff = max(T_diff, 1.0 - clamp(baseColor.a, 0.0, 1.0));

        float3 diffuseLighting = float3(0.0, 0.0, 0.0);
        float3 specularLighting = float3(0.0, 0.0, 0.0);

        const bool shadowsEnabled = (ubo._pad1 != 0);
        int lc = max(ubo.lightCount, 0);
        for (int li = 0; li < lc; ++li) {
            LightData light = lightBuffer[li];

            // Determine whether to use multi-sample soft shadows for this light.
            // Directional shadows stay hard for now.
            int samples = 1;
            float softness = max(ubo.shadowSoftness, 0.0);
            int reqSamples = max(ubo.shadowSampleCount, 1);
            if (shadowsEnabled && reqSamples > 1 && softness > 0.0 && light.lightType != 1) {
                samples = min(reqSamples, 32);
            }

            // Build a stable-ish RNG for this light
            uint lightRng = rqHash(rngState ^ (uint(li) * 747796405u) ^ rqHash(asuint(result.worldPos.x) + 31u * asuint(result.worldPos.y)));

            float3 diffAcc = float3(0.0, 0.0, 0.0);
            float3 specAcc = float3(0.0, 0.0, 0.0);

            for (int si = 0; si < samples; ++si) {
                float3 L;
                float3 radiance;
                float tMaxShadow = 10000.0;

                if (light.lightType == 1) {
                    // Directional
                    L = normalize(-light.position.xyz);
                    radiance = light.color.rgb;
                } else {
                    // Point/spot/emissive
                    float3 lightPos = light.position.xyz;

                    float3 toCenter = lightPos - result.worldPos;
                    float dCenter = length(toCenter);
                    float3 Lcenter = (dCenter > 1e-5) ? (toCenter / dCenter) : float3(0, 0, 1);

                    // Effective area-light radius (in meters) as a function of range.
                    // `shadowSoftness` is authored as a fraction of `light.range`.
                    float lightRadius = softness * max(light.range, 0.0);
                    lightRadius = clamp(lightRadius, 0.0, 2.0);

                    float3 samplePos = lightPos;
                    if (samples > 1 && lightRadius > 0.0) {
                        float3 up = (abs(Lcenter.y) < 0.999) ? float3(0, 1, 0) : float3(1, 0, 0);
                        float3 T = normalize(cross(up, Lcenter));
                        float3 B = cross(Lcenter, T);
                        float2 d = rqSampleDisk(lightRng);
                        samplePos = lightPos + (T * d.x + B * d.y) * lightRadius;
                    }

                    float3 toLight = samplePos - result.worldPos;
                    float d = length(toLight);
                    L = (d > 1e-5) ? (toLight / d) : float3(0, 0, 1);
                    tMaxShadow = max(d - RQ_RAY_EPS, RQ_RAY_EPS);

                    float attenuation = 1.0;
                    if (light.lightType == 3) {
                        float r = max(light.range, 0.001);
                        attenuation = 1.0 / (1.0 + (d / r) * (d / r));
                    } else {
                        attenuation = 1.0 / max(d * d, 0.0001);
                        // GLTF style range attenuation
                        if (light.range > 0.0) {
                            attenuation *= pow(saturate(1.0 - pow(d / light.range, 4.0)), 2.0);
                        }
                    }
                    radiance = light.color.rgb * attenuation;

                    if (light.lightType == 2) {
                        // Spot light cone attenuation
                        float3 D = normalize(light.direction.xyz);
                        float cd = dot(D, -L);
                        float cosInner = cos(light.innerConeAngle);
                        float cosOuter = cos(light.outerConeAngle);
                        float spotAttenuation = saturate((cd - cosOuter) / max(cosInner - cosOuter, 0.0001));
                        spotAttenuation *= spotAttenuation;
                        radiance *= spotAttenuation;
                    }
                }

                float rawDot = dot(N, L);
                float NdotL = (light.lightType == 3) ? abs(rawDot) : max(rawDot, 0.0);
                if (NdotL <= 0.0) {
                    continue;
                }

                float visibility = 1.0;
                if (shadowsEnabled) {
                    float3 shadowOrigin = result.worldPos + N * RQ_RAY_EPS;
                    bool occluded = traceShadowOccluded(shadowOrigin, L, RQ_RAY_EPS, tMaxShadow);
                    visibility = occluded ? 0.0 : 1.0;
                }

                float3 H = normalize(V + L);
                float NdotV = max(dot(N, V), 0.0);
                float NdotH = max(dot(N, H), 0.0);
                float HdotV = max(dot(H, V), 0.0);
                float D = DistributionGGX(NdotH, roughness);
                float G = GeometrySmith(NdotV, NdotL, roughness);
                float3 F = FresnelSchlick(HdotV, F0);
                float3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.0001);
                float3 kD = (1.0 - F) * (1.0 - metallic) * (1.0 - T_diff);

                specAcc += spec * radiance * NdotL * visibility;
                diffAcc += (kD * albedo / PI) * radiance * NdotL * visibility;
            }

            float inv = (samples > 1) ? (1.0 / float(samples)) : 1.0;
            specularLighting += specAcc * inv;
            diffuseLighting += diffAcc * inv;
        }

        // Avoid ambient "fill" on glass/transmissive surfaces; it can make them look opaque/frosted.
        // Keep ambient for opaque materials to retain basic IBL/ambient parity.
        bool treatAsTransmissive = (material.isGlass != 0) || (material.transmissionFactor > 0.01) || (material.alphaMode == 2);
        float3 ambient = treatAsTransmissive ? float3(0.0, 0.0, 0.0) : (albedo * (0.1 * ubo.scaleIBLAmbient) * ao);
        float3 color = ambient + diffuseLighting + specularLighting + emissive;

        result.color = color;
        result.metallic = metallic;
        result.roughness = roughness;
        result.transmission = material.transmissionFactor;
        result.isGlass = (material.isGlass != 0);
        result.alphaMode = material.alphaMode;
        // Keep texture-derived alpha/opacity (baseColor factor * baseColor texture).
        result.opacity = clamp(baseColor.a, 0.0, 1.0);
        result.thinWalled = (material.thinWalled != 0);
    }
    
    return result;
}

// Compute shader entry point
[[shader("compute")]]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;
    uint2 imageDim = uint2(ubo.screenDimensions);
    
    // Bounds check
    if (pixelCoord.x >= imageDim.x || pixelCoord.y >= imageDim.y) {
        return;
    }
    
    // Generate primary ray
    float2 uv = (float2(pixelCoord) + 0.5) / float2(imageDim);
    float2 ndc = uv * 2.0 - 1.0;
    
    // Compute ray direction using inverse view-projection
    // Unproject a point on the near plane (z=0 in Vulkan NDC) to get direction
    float4x4 invProj = inverse(ubo.proj);
    float4x4 invView = inverse(ubo.view);
    
    // Unproject near plane point (z=0) and far plane point (z=1) to get ray direction
    // Near plane in clip space: z=0, w=1
    float4 nearClip = float4(ndc, 0.0, 1.0);
    float4 farClip = float4(ndc, 1.0, 1.0);
    
    // Transform to view space
    float4 nearView = mul(invProj, nearClip);
    float4 farView = mul(invProj, farClip);
    nearView /= nearView.w;
    farView /= farView.w;
    
    // Transform to world space
    float3 nearWorld = mul(invView, nearView).xyz;
    float3 farWorld = mul(invView, farView).xyz;
    
    // Primary ray from camera position through the pixel
    float3 rayOrigin = ubo.camPos.xyz;
    float3 rayDir = normalize(farWorld - nearWorld);
    
    uint rngState = rqHash(pixelCoord.x + 4099u * pixelCoord.y + 131071u);
    HitInfo hit = traceRay(rayOrigin, rayDir, 0.0001, 10000.0, rngState);
    
    if (hit.hit) {
        float3 c = shadeWithSecondaryRays(rayOrigin, rayDir, hit, rngState);
        // Output linear HDR-ish color; composite pass will apply exposure/gamma.
        outputImage[pixelCoord] = float4(c, 1.0);

    } else {
        // Sky/background color
        outputImage[pixelCoord] = float4(skyColor(rayDir), 1.0);
    }
}
