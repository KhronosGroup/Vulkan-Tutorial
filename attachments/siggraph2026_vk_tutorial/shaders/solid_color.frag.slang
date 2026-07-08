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
struct SolidColorData {
    float4 color;
};

// This shader intentionally reuses binding 2.
// Its shader-object mapping points that binding at a different descriptor heap offset
// than the textured shader's point light.
[[vk::binding(2, 0)]]
ConstantBuffer<SolidColorData> g_solidColor;

struct FragmentInput {
    float3 worldPosition : TEXCOORD2;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float tangentSign : TEXCOORD1;
    float2 uv : TEXCOORD0;
};

[shader("fragment")]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    float4 color = g_solidColor.color;

    // HACK: Disable performance warning:
    // [VK_SHADER_STAGE_VERTEX_BIT] has an Output value declared but there is no corresponding Input declared in
    // [VK_SHADER_STAGE_FRAGMENT_BIT]. This is not invalid, but the write to the unused Output is discarded, causing a
    // performance validation message. Proper solution is to have a custom vertex shader, but this is intentionally
    // avoided to keep the tutorial simple.
    if (color.a < 0.0f) {
        color.xy += input.uv * 0.001;
        color.w += input.tangentSign * 0.001;
        color.xyz += input.tangent * 0.001;
        color.xyz += input.normal * 0.001;
        color.xyz += input.worldPosition * 0.001;
    }

    return color;
}
