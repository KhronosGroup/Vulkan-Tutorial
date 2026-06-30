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
import common_types;
import pbr_utils;
import lighting_utils;
import tonemapping_utils;

struct VSInput {
    [[vk::location(0)]] float3 Position;
    [[vk::location(1)]] float3 Normal;
    [[vk::location(2)]] float2 UV;
    [[vk::location(3)]] float4 Tangent;
    [[vk::location(4)]] column_major float4x4 InstanceModelMatrix;
    [[vk::location(8)]] float4 InstanceNormal0;
    [[vk::location(9)]] float4 InstanceNormal1;
    [[vk::location(10)]] float4 InstanceNormal2;
};

struct VSOutput {
    float4 Position : SV_POSITION;
    float3 WorldPos;
    float3 Normal : NORMAL;
    float3 GeometricNormal : NORMAL1;
    float2 UV : TEXCOORD0;
    float4 Tangent : TANGENT;
};

[[vk::binding(0, 1)]] Sampler2D opaqueSceneColor;

[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] Sampler2D baseColorMap;
[[vk::binding(2, 0)]] Sampler2D metallicRoughnessMap;
[[vk::binding(3, 0)]] Sampler2D normalMap;
[[vk::binding(4, 0)]] Sampler2D occlusionMap;
[[vk::binding(5, 0)]] Sampler2D emissiveMap;
// Bindings 6, 7, 8 (SSBOs) REMOVED
[[vk::binding(10, 0)]] Sampler2D reflectionMap;

[[vk::push_constant]] PushConstants material;

[[shader("vertex")]]
VSOutput VSMain(VSInput input)
{
    VSOutput output;
    float4x4 instanceModelMatrix = input.InstanceModelMatrix;
    float4 worldPos = mul(ubo.model, mul(instanceModelMatrix, float4(input.Position, 1.0)));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));
    output.WorldPos = worldPos.xyz;

    float3x3 instNormal = float3x3(input.InstanceNormal0.xyz, input.InstanceNormal1.xyz, input.InstanceNormal2.xyz);
    float3x3 model3x3 = (float3x3)ubo.model;
    float3 worldNormal = normalize(mul(model3x3, mul(instNormal, input.Normal)));
    output.Normal = worldNormal;
    output.GeometricNormal = worldNormal;

    float3 worldTangent = normalize(mul(model3x3, mul(instNormal, input.Tangent.xyz)));
    output.UV = input.UV;
    output.Tangent = float4(worldTangent, input.Tangent.w);
    return output;
}

[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    float2 uv = float2(input.UV.x, 1.0 - input.UV.y);
    float4 baseColor = (material.baseColorTextureSet < 0) ? material.baseColorFactor : baseColorMap.Sample(uv) * material.baseColorFactor;

    float metallic = material.metallicFactor;
    float roughness = material.roughnessFactor;
    float3 albedo = baseColor.rgb * (1.0 - metallic);

    if (material.alphaMask > 0.5 && baseColor.a < material.alphaMaskCutoff) { discard; }

    float3 N = normalize(input.Normal);
    float3 V = normalize(ubo.camPos.xyz - input.WorldPos);
    float3 L = normalize(float3(1.0, 1.0, 1.0));

    float NdotL = max(dot(N, L), 0.0);
    float3 ambient = albedo * 0.1 * ubo.scaleIBLAmbient;
    float3 diffuse = albedo * NdotL;

    return float4(diffuse + ambient, baseColor.a);
}

[[shader("fragment")]]
float4 GlassPSMain(VSOutput input) : SV_TARGET
{
    float2 uv = float2(input.UV.x, 1.0 - input.UV.y);
    float4 baseColor = (material.baseColorTextureSet < 0) ? material.baseColorFactor : baseColorMap.Sample(uv) * material.baseColorFactor;

    float2 uvR = input.Position.xy / ubo.screenDimensions;
    float3 bg = opaqueSceneColor.Sample(uvR).rgb;

    return float4(lerp(baseColor.rgb, bg, 0.9), 1.0);
}
