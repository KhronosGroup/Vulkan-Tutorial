#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

// Canonical from appendix_types.adoc
static constexpr uint32_t INVALID_NODE_INDEX = 0xFFFFFFFF;

enum TransformStatus : uint8_t {
    Clean      = 0,
    LocalDirty = 1 << 0,
    WorldDirty = 1 << 1,
};

// Physics collider metadata extracted from glTF node extras
struct ColliderDef {
    enum class Shape { CAPSULE, BOX, NONE };
    Shape     shape           = Shape::NONE;
    float     radius          = 0.0f;
    float     half_height     = 0.0f;
    glm::vec3 box_half_extents = {0, 0, 0};
    float     mass            = 1.0f;
    std::string collision_group;
    std::string collision_mask;
};

// Joint constraint metadata extracted from glTF node extras
struct ConstraintDef {
    enum class Type { NONE, BALL_SOCKET, HINGE };
    Type      type            = Type::NONE;
    float     swing_limit_deg = 180.0f;
    float     twist_limit_deg = 180.0f;
    float     hinge_min_deg   = -180.0f;
    float     hinge_max_deg   =  180.0f;
    glm::vec3 hinge_axis      = {0, 0, 1};
    std::string parent_bone;
};

struct Node {
    uint32_t              node_index;
    uint32_t              parent_index = INVALID_NODE_INDEX;
    std::vector<uint32_t> child_indices;
    std::string           name;

    // Local SRT components
    glm::vec3 translation    = {0, 0, 0};
    glm::quat local_rotation = glm::identity<glm::quat>();
    glm::vec3 scale          = {1, 1, 1};

    glm::mat4 world_matrix = glm::mat4(1.0f);
    uint8_t   status       = TransformStatus::Clean;
    bool      is_joint     = false;

    ColliderDef  collider_def;
    ConstraintDef constraint_def;

    void mark_dirty() {
        status |= TransformStatus::LocalDirty | TransformStatus::WorldDirty;
    }

    glm::mat4 get_local_matrix() const {
        return glm::translate(glm::mat4(1.0f), translation) *
               glm::mat4_cast(local_rotation) *
               glm::scale(glm::mat4(1.0f), scale);
    }

    // Extracts rotation from world_matrix, stripping scale.
    glm::quat get_world_rotation() const {
        glm::mat3 rs = glm::mat3(world_matrix);
        glm::mat3 r;
        r[0] = glm::normalize(rs[0]);
        r[1] = glm::normalize(rs[1]);
        r[2] = glm::normalize(rs[2]);
        return glm::quat_cast(r);
    }
};

class SceneGraph {
public:
    std::vector<Node> nodes;

    // Linear pass — only correct when nodes are topologically sorted (parent before child).
    void update_transforms() {
        for (auto& node : nodes) {
            if (node.status & TransformStatus::WorldDirty) {
                if (node.parent_index != INVALID_NODE_INDEX) {
                    node.world_matrix =
                        nodes[node.parent_index].world_matrix * node.get_local_matrix();
                } else {
                    node.world_matrix = node.get_local_matrix();
                }
                for (uint32_t child : node.child_indices)
                    nodes[child].status |= TransformStatus::WorldDirty;
                node.status = TransformStatus::Clean;
            }
        }
    }

    // Recursive subtree update — correct regardless of node order; used after IK/physics writes.
    void update_world_matrices_subtree(uint32_t index) {
        Node& node = nodes[index];
        if (node.parent_index != INVALID_NODE_INDEX) {
            node.world_matrix =
                nodes[node.parent_index].world_matrix * node.get_local_matrix();
        } else {
            node.world_matrix = node.get_local_matrix();
        }
        for (uint32_t child : node.child_indices) {
            nodes[child].status |= TransformStatus::WorldDirty;
            update_world_matrices_subtree(child);
        }
        node.status = TransformStatus::Clean;
    }
};

// Free-function overload used by IK solvers (wraps the SceneGraph member).
// IK chapters call update_world_matrices_subtree(nodes, idx) — this adapts that call site.
inline void update_world_matrices_subtree(std::vector<Node>& nodes, uint32_t index) {
    Node& node = nodes[index];
    if (node.parent_index != INVALID_NODE_INDEX) {
        node.world_matrix =
            nodes[node.parent_index].world_matrix * node.get_local_matrix();
    } else {
        node.world_matrix = node.get_local_matrix();
    }
    for (uint32_t child : node.child_indices) {
        nodes[child].status |= TransformStatus::WorldDirty;
        update_world_matrices_subtree(nodes, child);
    }
    node.status = TransformStatus::Clean;
}
