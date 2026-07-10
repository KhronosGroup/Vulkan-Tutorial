/* Copyright (c) 2026, Khronos Group and contributors
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
struct CameraData {
    float4x4 viewProjection;
    float4 cameraPosition;
};

struct LightData {
    float3 pointPosition;
    float pointIntensity;
    float3 ambientColor;
    float ambientIntensity;
};

// Shared binding table:
// 0 camera.
// 1 object data.
// 2 point light.
// 3 albedo combined image sampler.
[[vk::binding(0, 0)]]
ConstantBuffer<CameraData> g_camera;

[[vk::binding(2, 0)]]
ConstantBuffer<LightData> g_pointLight;

[[vk::binding(3, 0)]]
Sampler2D g_albedoTexture;

// In Slang we can use HLSL annotations to communicate attributes between vertex and fragment shaders.
struct FragmentInput {
    float3 worldPosition;
    float3 normal;
    float3 tangent;
    float tangentSign;
    float2 uv;
};

// Simple illumination function
float3 illuminate(float3 albedo, float3 worldPosition, float3 normal, LightData pointLight)
{
    // Basic Blinn-Phong lighting with distance attenuation.
    float3 lightDelta = pointLight.pointPosition - worldPosition;
    float lightDistance = length(lightDelta);
    float3 lightDirection = lightDelta / max(lightDistance, 0.001);
    float attenuation = 1.0 / (1.0 + 0.09 * lightDistance + 0.032 * lightDistance * lightDistance);

    float diffuseAmount = max(dot(normal, lightDirection), 0.0);
    float3 viewDirection = normalize(g_camera.cameraPosition.xyz - worldPosition);
    float3 halfVector = normalize(lightDirection + viewDirection);
    float specularAmount = pow(max(dot(normal, halfVector), 0.0), 32.0) * 0.2;

    float3 ambient = albedo * pointLight.ambientColor * pointLight.ambientIntensity;
    float3 direct =
        (albedo * diffuseAmount + specularAmount) * pointLight.ambientColor * pointLight.pointIntensity * attenuation;

    return ambient + direct;
}

[shader("fragment")]
float4 fragmentMain(FragmentInput input) {
    // The albedo combined sampled image is selected by HEAP_WITH_PUSH_INDEX.
    float4 texel = g_albedoTexture.Sample(input.uv);
    if (texel.a < 0.5) {
        // Discard transparent fragments.
        discard;
    }

    float3 normal = normalize(input.normal);
    if (texel.a < 0.0) {
        // HACK: Disable the performance warning by faking the use of all inputs.
        // If we remove this code, we get this warning when running with validation layers:
        // [VK_SHADER_STAGE_VERTEX_BIT] has an Output value declared but there is no corresponding Input declared in
        // [VK_SHADER_STAGE_FRAGMENT_BIT].
        //
        // This is not invalid; it is just a performance warning telling us that the write to the unused output is
        // discarded. This can cause a performance loss on some GPUs, especially when using unlinked shader objects,
        // since they might disable cross-stage shader optimizations.
        //
        // We should aim for a proper solution. For example, we could have a custom vertex shader that skips writing the
        // unused inputs, but this is intentionally avoided to keep the tutorial simple.
        normal += input.tangent * 0.001 + input.tangentSign * 0.001;
    }
    return float4(illuminate(texel.rgb, input.worldPosition, normalize(normal), g_pointLight), 1.0);
}
