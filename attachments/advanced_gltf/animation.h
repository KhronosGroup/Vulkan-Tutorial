#pragma once
#include "node.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Canonical from appendix_types.adoc
enum InterpolationMode { STEP, LINEAR, CUBICSPLINE };

struct AnimationSampler {
    InterpolationMode    interpolation = LINEAR;
    std::vector<float>   inputs;          // Timestamps in seconds
    std::vector<glm::vec4> outputs_raw;   // Packed: for CUBICSPLINE stores in_tan/value/out_tan triples

    // Unpacked for CUBICSPLINE — filled at load time from outputs_raw
    std::vector<glm::vec4> in_tangents;
    std::vector<glm::vec4> values;
    std::vector<glm::vec4> out_tangents;
};

struct AnimationChannel {
    enum PathType { TRANSLATION, ROTATION, SCALE, WEIGHTS };
    PathType path;
    uint32_t node_index;
    uint32_t sampler_index;
};

struct Pose {
    std::vector<glm::vec3> translations;
    std::vector<glm::quat> rotations;
    std::vector<glm::vec3> scales;
};

// Matches the glTF skin object.
struct Skin {
    std::string              name;
    std::vector<uint32_t>    joints;                // Node indices
    std::vector<glm::mat4>   inverse_bind_matrices; // One per joint
    uint32_t                 skeleton_root = INVALID_NODE_INDEX;
};

// Binary search for the largest keyframe index whose timestamp <= time.
inline uint32_t find_keyframe(const AnimationSampler& sampler, float time) {
    if (sampler.inputs.size() < 2) return 0;
    auto it  = std::lower_bound(sampler.inputs.begin(), sampler.inputs.end(), time);
    uint32_t idx = static_cast<uint32_t>(std::distance(sampler.inputs.begin(), it));
    return (idx > 0) ? idx - 1 : 0;
}

// Hermite cubic spline interpolation for vec3 (glTF CUBICSPLINE mode).
// dt is the duration of the keyframe interval (t1 - t0), needed to scale tangents correctly.
inline glm::vec3 cubic_spline_interpolate_vec3(
    float t, float dt,
    glm::vec3 p0, glm::vec3 out_tan0,
    glm::vec3 p1, glm::vec3 in_tan1)
{
    float t2 = t * t;
    float t3 = t2 * t;
    float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 =         t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 =         t3 -        t2;
    return h00 * p0 + h10 * dt * out_tan0
         + h01 * p1 + h11 * dt * in_tan1;
}

// Hermite cubic spline for quaternion. The glTF spec requires normalization after the blend.
inline glm::quat cubic_spline_interpolate_quat(
    float t, float dt,
    glm::quat p0, glm::quat out_tan0,
    glm::quat p1, glm::quat in_tan1)
{
    float t2 = t * t;
    float t3 = t2 * t;
    float h00 =  2.0f * t3 - 3.0f * t2 + 1.0f;
    float h10 =         t3 - 2.0f * t2 + t;
    float h01 = -2.0f * t3 + 3.0f * t2;
    float h11 =         t3 -        t2;
    glm::vec4 blended =
          h00 * glm::vec4(p0.x,       p0.y,       p0.z,       p0.w)
        + h10 * dt * glm::vec4(out_tan0.x, out_tan0.y, out_tan0.z, out_tan0.w)
        + h01 * glm::vec4(p1.x,       p1.y,       p1.z,       p1.w)
        + h11 * dt * glm::vec4(in_tan1.x,  in_tan1.y,  in_tan1.z,  in_tan1.w);
    return glm::normalize(glm::quat(blended.w, blended.x, blended.y, blended.z));
}

// Canonical signature from appendix_types.adoc.
// Writes Pose transforms back to their corresponding scene graph nodes and marks them dirty.
inline void apply_pose_to_scene_graph(
    std::vector<Node>&           nodes,
    const Pose&                  pose,
    const std::vector<uint32_t>& joint_indices)
{
    for (size_t i = 0; i < joint_indices.size(); ++i) {
        Node& node      = nodes[joint_indices[i]];
        node.translation    = pose.translations[i];
        node.local_rotation = pose.rotations[i];
        node.scale          = pose.scales[i];
        node.mark_dirty();
    }
}

// Pre-computes joint matrices (J * IB) on the CPU before uploading to the GPU skinning shader.
// Call after the animation update and scene graph update, once per frame.
inline void compute_joint_matrices(
    const Skin&              skin,
    const std::vector<Node>& nodes,
    std::vector<glm::mat4>&  joint_matrices_out)
{
    joint_matrices_out.resize(skin.joints.size());
    for (size_t i = 0; i < skin.joints.size(); ++i) {
        const Node& joint_node   = nodes[skin.joints[i]];
        joint_matrices_out[i]    = joint_node.world_matrix * skin.inverse_bind_matrices[i];
    }
}
