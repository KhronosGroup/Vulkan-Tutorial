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
    float3 worldPosition;
    float3 normal;
    float3 tangent;
    float tangentSign;
    float2 uv;
};

[shader("fragment")] float4 fragmentMain(FragmentInput input) : SV_Target
{
    float4 color = g_solidColor.color;

    if (color.a < 0.0f) {
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
        color.xy += input.uv * 0.001;
        color.w += input.tangentSign * 0.001;
        color.xyz += input.tangent * 0.001;
        color.xyz += input.normal * 0.001;
        color.xyz += input.worldPosition * 0.001;
    }

    return color;
}
