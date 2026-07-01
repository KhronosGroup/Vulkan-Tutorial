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

// Fullscreen composite pass: samples the off-screen opaque color and writes to swapchain

import tonemapping_utils;

struct VSOut {
    float4 Position : SV_POSITION;
    float2 UV       : TEXCOORD0;
};

// Export entrypoint for vertex stage
[shader("vertex")] VSOut VSMain(uint vid : SV_VertexID)
{
    // Fullscreen triangle (no vertex buffer)
    float2 pos = float2( (vid == 2) ? 3.0 : -1.0,
                         (vid == 1) ? 3.0 : -1.0 );
    float2 uv  = float2( (vid == 2) ? 2.0 : 0.0,
                         (vid == 1) ? 2.0 : 0.0 );
    VSOut o;
    o.Position = float4(pos, 0.0, 1.0);
    o.UV = uv;
    return o;
}

// Set 0, binding 0: combined image sampler for the off-screen scene color
[[vk::binding(0, 0)]] Sampler2D sceneColor;

struct Push {
    float exposure;
    float gamma;
    int   outputIsSRGB; // 1 when the color attachment is SRGB; 0 otherwise
    float _pad;         // pad to 16 bytes for push constant layout
};
[[vk::push_constant]] Push pushConsts;

// Export entrypoint for fragment stage
[shader("fragment")] float4 PSMain(VSOut i) : SV_TARGET
{
    float4 c = sceneColor.Sample(i.UV);
    float3 color = c.rgb;

    // Apply exposure and filmic tonemapping
    color *= pushConsts.exposure;

    // Uncharted2 / Hable filmic tonemap, canonical form
    float3 t = Hable_Filmic_Tonemapping::Uncharted2Tonemap(color);
    float3 w = Hable_Filmic_Tonemapping::Uncharted2Tonemap(float3(1,1,1) * Hable_Filmic_Tonemapping::W);
    color = t / max(w, float3(1e-6, 1e-6, 1e-6));

    // If the attachment is NOT SRGB, encode gamma here. When it is SRGB,
    // the hardware will encode at store so we keep color in linear space.
    if (pushConsts.outputIsSRGB == 0) {
        color = pow(max(color, 0.0), float3(1.0 / pushConsts.gamma));
    } else {
        color = saturate(color);
    }
    return float4(color, 1.0);
}
