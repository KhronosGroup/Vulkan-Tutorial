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
// GPU compute skinning — Chapter 3: Compute Skinning.
// Reads rest-pose vertices and joint matrices, writes animated vertices.
// Dispatched once per skinned mesh per frame, after joint matrices are uploaded.

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

[[vk::binding(0, 0)]] StructuredBuffer<InputVertex>    input_vertices;
[[vk::binding(1, 0)]] RWStructuredBuffer<OutputVertex> output_vertices;
[[vk::binding(2, 0)]] StructuredBuffer<float4x4>       joint_matrices;
[[vk::binding(3, 0)]] StructuredBuffer<uint4>          joint_indices;
[[vk::binding(4, 0)]] StructuredBuffer<float4>         joint_weights;

struct SkinPushConstants {
    uint vertex_count;
};
[[vk::push_constant]] SkinPushConstants push_constants;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint vertex_id = dispatchThreadID.x;
    if (vertex_id >= push_constants.vertex_count) return;

    InputVertex v = input_vertices[vertex_id];
    uint4 j_idx = joint_indices[vertex_id];
    float4 j_w = joint_weights[vertex_id];

    // Weighted sum of per-joint matrices (Linear Blend Skinning).
    float4x4 skin_matrix =
        j_w.x * joint_matrices[j_idx.x] +
        j_w.y * joint_matrices[j_idx.y] +
        j_w.z * joint_matrices[j_idx.z] +
        j_w.w * joint_matrices[j_idx.w];

    float3 v_pos = float3(v.p[0], v.p[1], v.p[2]);
    float3 v_nrm = float3(v.n[0], v.n[1], v.n[2]);
    float4 v_tan = float4(v.t[0], v.t[1], v.t[2], v.t[3]);

    float4 animated_pos = mul(skin_matrix, float4(v_pos, 1.0));

    // Transform direction vectors by the rotation-scale part only (no translation).
    float3x3 skin_rot = float3x3(
        skin_matrix[0].xyz,
        skin_matrix[1].xyz,
        skin_matrix[2].xyz);

    float3 animated_normal  = normalize(mul(skin_rot, v_nrm));
    float3 animated_tangent = normalize(mul(skin_rot, v_tan.xyz));

    OutputVertex out_v;
    out_v.p[0] = animated_pos.x;
    out_v.p[1] = animated_pos.y;
    out_v.p[2] = animated_pos.z;
    
    out_v.n[0] = animated_normal.x;
    out_v.n[1] = animated_normal.y;
    out_v.n[2] = animated_normal.z;
    
    out_v.t[0] = animated_tangent.x;
    out_v.t[1] = animated_tangent.y;
    out_v.t[2] = animated_tangent.z;
    out_v.t[3] = v_tan.w; // preserve handedness
    
    out_v.uv[0] = v.uv[0];
    out_v.uv[1] = v.uv[1];

    output_vertices[vertex_id] = out_v;
}
