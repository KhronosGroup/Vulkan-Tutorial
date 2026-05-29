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
// Skinning heatmap diagnostic shader — Chapter 6 (Debugging & Visual Auditing).
//
// Two rendering modes controlled by pipeline selection (not specialization constants,
// for clarity — see Chapter 6 for the specialization-constant approach):
//
//   DOMINANT BONE:   Colors each pixel by which joint has the largest weight.
//                    Exposed via fragment_dominant_bone().
//
//   WEIGHT DISTRIBUTION: Colors by how many joints have non-trivial influence
//                    (blue = 1 joint, red = 4 joints). Exposed via fragment_weight_distribution().
//
// Both fragment shaders share the same vertex shader (vertex_main).

// Per-camera data passed via a uniform/push constant in the real pipeline.
// Declared here as a push constant for simplicity.
struct CameraPushConstants {
    float4x4 view_proj;
};
[[vk::push_constant]] CameraPushConstants camera;

// Vertex input mirrors the skinned OutputVertex layout + joint data from parallel buffers.
struct Vertex {
    float3 position;
    float3 normal;
    float4 tangent;
    float2 texcoord;
};

[[vk::binding(0, 0)]] StructuredBuffer<Vertex>    vertices;
[[vk::binding(1, 0)]] StructuredBuffer<float4x4>  joint_matrices;  // For world position
[[vk::binding(2, 0)]] StructuredBuffer<uint4>     joint_indices;
[[vk::binding(3, 0)]] StructuredBuffer<float4>    joint_weights;
[[vk::binding(4, 0)]] StructuredBuffer<float4>    joint_colors;    // One RGBA per joint

struct VertexOut {
    float4 position : SV_Position;
    float4 weights  : TEXCOORD0;
    uint4  joints   : TEXCOORD1;
};

[shader("vertex")]
VertexOut vertex_main(uint vertex_id : SV_VertexID)
{
    Vertex v = vertices[vertex_id];
    uint4 j_idx = joint_indices[vertex_id];
    float4 j_w = joint_weights[vertex_id];

    float4x4 skin_matrix =
        j_w.x * joint_matrices[j_idx.x] +
        j_w.y * joint_matrices[j_idx.y] +
        j_w.z * joint_matrices[j_idx.z] +
        j_w.w * joint_matrices[j_idx.w];

    VertexOut out;
    out.position = mul(camera.view_proj, mul(skin_matrix, float4(v.position, 1.0)));
    out.weights  = j_w;
    out.joints   = j_idx;
    return out;
}

// Colors each pixel by the dominant (highest-weight) joint using a per-joint color table.
// Immediately reveals stray vertex assignments and weight-painting territory errors.
[shader("fragment")]
float4 fragment_dominant_bone(VertexOut input) : SV_Target
{
    uint  dominant = 0;
    float max_w    = input.weights.x;

    if (input.weights.y > max_w) { max_w = input.weights.y; dominant = 1; }
    if (input.weights.z > max_w) { max_w = input.weights.z; dominant = 2; }
    if (input.weights.w > max_w) { max_w = input.weights.w; dominant = 3; }

    uint actual_joint = input.joints[dominant];
    return joint_colors[actual_joint];
}

// Colors by number of joints with weight > 5% — blue (1 bone) through red (4 bones).
// Helps identify regions where weight painting is unnecessarily noisy.
[shader("fragment")]
float4 fragment_weight_distribution(VertexOut input) : SV_Target
{
    float complexity = 0.0f;
    float4 w = input.weights;
    if (w.x > 0.05f) complexity += 1.0f;
    if (w.y > 0.05f) complexity += 1.0f;
    if (w.z > 0.05f) complexity += 1.0f;
    if (w.w > 0.05f) complexity += 1.0f;

    float t = (complexity - 1.0f) / 3.0f; // [0..1]: 1 bone → 4 bones

    float3 cool = float3(0, 0, 1);
    float3 mid  = float3(0, 1, 0);
    float3 warm = float3(1, 0, 0);

    float3 color;
    if (t < 0.5f) color = lerp(cool, mid, t * 2.0f);
    else          color = lerp(mid, warm, (t - 0.5f) * 2.0f);

    return float4(color, 1.0f);
}
