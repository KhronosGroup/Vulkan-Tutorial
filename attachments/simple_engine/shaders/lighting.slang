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
// Combined vertex and fragment shader for basic/legacy lighting
// This shader implements the Phong lighting model as a fallback when BRDF/PBR is disabled
// Note: BRDF/PBR is now the default lighting model - this is used only when explicitly requested

// Import shared utility modules
import common_types;

// Input from vertex buffer
struct VSInput {
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 Tangent : TANGENT; // Added to match vertex layout (unused in basic lighting)
};

// Output from vertex shader / Input to fragment shader
struct VSOutput {
    float4 Position : SV_POSITION;
    float3 WorldPos : POSITION;
    float3 Normal : NORMAL;
    float2 TexCoord : TEXCOORD0;
    float4 Tangent : TANGENT; // Pass through tangent (unused in basic lighting)
};

// Bindings
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] Sampler2D texSampler;

// Push constants for material properties
struct LegacyPushConstants {
    float4 ambientColor;
    float4 diffuseColor;
    float4 specularColor;
    float shininess;
};

// Push constants
[[vk::push_constant]] LegacyPushConstants material;

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
    output.TexCoord = input.TexCoord;

    // Pass tangent (unused in basic lighting but required for vertex layout compatibility)
    output.Tangent = input.Tangent;

    return output;
}

// Fragment shader entry point
[[shader("fragment")]]
float4 PSMain(VSOutput input) : SV_TARGET
{
    // Sample texture
    float4 texColor = texSampler.Sample(input.TexCoord);

    // Normalize vectors
    float3 normal = normalize(input.Normal);
    // Use first light for legacy fallback (matches Renderer::updateUniformBufferInternal logic if it was using this UBO)
    // Actually, common_types.slang UBO doesn't have lightPos.
    // We should probably just use a default light or ignore it for now as this is a fallback.
    float3 lightDir = normalize(float3(1.0, 1.0, 1.0));
    float3 viewDir = normalize(ubo.camPos.xyz - input.WorldPos);
    float3 reflectDir = reflect(-lightDir, normal);

    // Ambient
    float3 ambient = material.ambientColor.rgb * float3(0.1, 0.1, 0.1);

    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    float3 diffuse = diff * material.diffuseColor.rgb;

    // Specular
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    float3 specular = spec * material.specularColor.rgb;

    // Combine components
    float3 result = (ambient + diffuse + specular) * texColor.rgb;

    return float4(result, texColor.a);
}
