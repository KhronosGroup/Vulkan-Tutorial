/* Copyright (c) 2026 Holochip Corporation
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
// Morph target accumulation + skeletal skinning — Chapter 5 (Morph Targets).
//
// REQUIREMENTS:
//   Vulkan device feature: descriptorBindingVariableDescriptorCount (VK_EXT_descriptor_indexing)
//   Vulkan device feature: runtimeDescriptorArray
//   The descriptor set layout for set=1 must be created with VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT.
//
// Two-pass operation per vertex:
//   Pass 1: accumulate weighted morph target position deltas onto the base position.
//   Pass 2: apply skeletal skinning via the blended joint matrix.

struct InputVertex {
    float p[3];
    float n[3];
    float uv[2];
    float t[4];
};

struct OutputVertex {
    float p[3];
    float n[3];
    float uv[2];
    float t[4];
};

// Set 0: per-mesh buffers
[[vk::binding(0, 0)]] StructuredBuffer<InputVertex>    base_vertices;
[[vk::binding(1, 0)]] RWStructuredBuffer<OutputVertex> output_vertices;
[[vk::binding(2, 0)]] StructuredBuffer<float4x4>       joint_matrices;
[[vk::binding(3, 0)]] StructuredBuffer<uint4>          joint_indices;
[[vk::binding(4, 0)]] StructuredBuffer<float4>         joint_weights;

// Set 1: runtime-length array of morph delta buffers (one StructuredBuffer per target).
// Requires VK_EXT_descriptor_indexing + runtimeDescriptorArray.
struct MorphDelta {
    float d[3];
};
[[vk::binding(0, 1)]] StructuredBuffer<MorphDelta>         morph_targets[];

struct MorphWeightBlock {
    float    weights[24];
    uint32_t active_count;
    uint32_t apply_skinning; // 1 = apply skeletal skinning to the position, 0 = morph only
    uint32_t pad[2];
};

struct PushConstants {
    uint32_t       vertex_count;
    uint32_t       morph_indices[24]; // Which descriptor slot each active target lives in
    MorphWeightBlock morph_weights;
};
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint vertex_idx = dispatch_id.x;
    if (vertex_idx >= pc.vertex_count) return;

    InputVertex base = base_vertices[vertex_idx];
    float3 base_pos = float3(base.p[0], base.p[1], base.p[2]);
    float3 base_nrm = float3(base.n[0], base.n[1], base.n[2]);
    float4 base_tan = float4(base.t[0], base.t[1], base.t[2], base.t[3]);

    // --- Pass 1: morph accumulation ---
    float3 morphed_position = base_pos;
    float3 morphed_normal   = base_nrm;

    for (uint m = 0; m < pc.morph_weights.active_count; ++m) {
        float weight = pc.morph_weights.weights[m];
        if (abs(weight) < 1e-5f) continue;

        uint slot = pc.morph_indices[m];
        MorphDelta delta_raw = morph_targets[NonUniformResourceIndex(slot)][vertex_idx];
        float3 delta = float3(delta_raw.d[0], delta_raw.d[1], delta_raw.d[2]);
        morphed_position += weight * delta;
        // Normal deltas are stored in a separate buffer in full implementations;
        // here we approximate by reusing the position delta direction.
    }

    // --- Pass 2: skeletal skinning ---
    uint4 j_idx = joint_indices[vertex_idx];
    float4 j_w = joint_weights[vertex_idx];

    float4x4 skin_matrix =
        j_w.x * joint_matrices[j_idx.x] +
        j_w.y * joint_matrices[j_idx.y] +
        j_w.z * joint_matrices[j_idx.z] +
        j_w.w * joint_matrices[j_idx.w];

    // Only apply skinning when the mesh actually has joint data. Morph-only meshes (no
    // skin) leave the joint buffers unbound/zero, so blindly applying skin_matrix would
    // collapse them to the origin; the flag lets those meshes pass the morphed position
    // through unchanged.
    float3x3 skin_rot = float3x3(
        skin_matrix[0].xyz,
        skin_matrix[1].xyz,
        skin_matrix[2].xyz);
    float3 final_position = morphed_position;
    float3 final_normal   = morphed_normal;
    float3 final_tangent  = base_tan.xyz;
    if (pc.morph_weights.apply_skinning != 0u) {
        final_position = mul(skin_matrix, float4(morphed_position, 1.0)).xyz;
        final_normal   = normalize(mul(skin_rot, morphed_normal));
        final_tangent  = normalize(mul(skin_rot, base_tan.xyz));
    }

    OutputVertex out;
    out.p[0] = final_position.x;
    out.p[1] = final_position.y;
    out.p[2] = final_position.z;

    out.n[0] = final_normal.x;
    out.n[1] = final_normal.y;
    out.n[2] = final_normal.z;

    out.t[0] = final_tangent.x;
    out.t[1] = final_tangent.y;
    out.t[2] = final_tangent.z;
    out.t[3] = base_tan.w;
    
    out.uv[0] = base.uv[0];
    out.uv[1] = base.uv[1];
    
    output_vertices[vertex_idx] = out;
}
