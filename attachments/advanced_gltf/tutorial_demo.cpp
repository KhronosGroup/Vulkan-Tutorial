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
#include "tutorial_demo.h"
#include "scene_loading.h"
#include "renderer_advanced_types.h"

#include "animation_component.h"
#include "camera_component.h"
#include "engine.h"
#include "entity.h"
#include "imgui/imgui.h"
#include "physics_system.h"
#include "transform_component.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <string>

TutorialDemoComponent::TutorialDemoComponent(Engine *engine)
    : Component("TutorialDemoComponent"), m_engine(engine)
{}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

Entity *TutorialDemoComponent::FindFoxMesh() const
{
    for (const auto &e : m_engine->GetEntities())
    {
        const std::string &n = e->GetName();
        if (n.rfind("Fox_", 0) == 0 && n.find("AnimController") == std::string::npos)
            return e.get();
    }
    return nullptr;
}

glm::vec3 TutorialDemoComponent::MouseToWorld(float depth) const
{
    // GetViewMatrix/GetProjectionMatrix are not const-qualified; const_cast is required.
    auto *cam = const_cast<CameraComponent *>(m_engine->GetActiveCamera());
    if (!cam)
        return {};

    const ImGuiIO &io = ImGui::GetIO();
    float          w  = io.DisplaySize.x;
    float          h  = io.DisplaySize.y;
    if (w <= 0.0f || h <= 0.0f)
        return {};

    float ndcX = (io.MousePos.x / w) * 2.0f - 1.0f;
    float ndcY = 1.0f - (io.MousePos.y / h) * 2.0f;

    glm::mat4 invVP = glm::inverse(cam->GetProjectionMatrix() * cam->GetViewMatrix());

    glm::vec4 n4{ndcX, ndcY, 0.0f, 1.0f};
    glm::vec4 f4{ndcX, ndcY, 1.0f, 1.0f};

    glm::vec4 wn = invVP * n4;
    wn /= wn.w;
    glm::vec4 wf = invVP * f4;
    wf /= wf.w;

    glm::vec3 ray = glm::normalize(glm::vec3(wf) - glm::vec3(wn));
    return glm::vec3(wn) + ray * depth;
}

// ---------------------------------------------------------------------------
// Chapter 1 — Scene Graph
// ---------------------------------------------------------------------------

void TutorialDemoComponent::DrawSceneGraphPanel()
{
    if (!ImGui::CollapsingHeader("Chapter 1 — Scene Graph"))
        return;

    auto entities = SnapshotEntities(m_engine);
    ImGui::Text("Total entities: %zu", entities.size());
    ImGui::Separator();

    static bool bistroVisible = true;
    if (ImGui::Checkbox("Toggle All Bistro Models", &bistroVisible))
    {
        for (auto *e : entities)
        {
            if (e->GetName().find("bistro_") == 0)
                e->SetActive(bistroVisible);
        }
    }
    ImGui::Separator();

    ImGui::BeginChild("EntityList", ImVec2(0.0f, 200.0f), true);
    for (auto *e : entities)
    {
        bool active = e->IsActive();
        ImGui::PushID(e);
        if (ImGui::Checkbox("##active", &active))
            e->SetActive(active);
        ImGui::SameLine();
        ImGui::Text("%s", e->GetName().c_str());
        if (auto *t = e->GetComponent<TransformComponent>())
        {
            const glm::vec3 &p = t->GetPosition();
            ImGui::SameLine();
            ImGui::TextDisabled("(%.1f, %.1f, %.1f)", p.x, p.y, p.z);
        }
        ImGui::PopID();
    }
    ImGui::EndChild();

    ImGui::TextWrapped("Tip: uncheck the box next to an entity to hide it. "
                       "Each entity is a node in the scene graph; TransformComponent "
                       "stores local SRT and recomputes the world matrix on demand, "
                       "using the same dirty-flag pattern as Node::mark_dirty() in node.h.");
}

// ---------------------------------------------------------------------------
// Chapter 2 — Skeletal Animation
// ---------------------------------------------------------------------------

void TutorialDemoComponent::DrawAnimationPanel()
{
    if (!ImGui::CollapsingHeader("Chapter 2 — Skeletal Animation", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    Entity *animEntity = m_engine->GetEntity("Fox_AnimController");
    AnimationComponent *anim = animEntity ? animEntity->GetComponent<AnimationComponent>() : nullptr;

    if (!anim)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                           "Fox model not loaded.\n"
                           "Run:  assets/download_samples.sh\n"
                           "then rebuild to enable animation controls.");
        return;
    }

    const auto &clips = anim->GetAnimations();
    if (clips.empty())
    {
        ImGui::Text("Model has no animation clips.");
        return;
    }

    // Clip selector
    std::vector<const char *> names;
    names.reserve(clips.size());
    for (const auto &c : clips)
        names.push_back(c.name.c_str());

    if (ImGui::Combo("Clip", &m_selectedAnim, names.data(), static_cast<int>(names.size())))
        anim->Play(static_cast<size_t>(m_selectedAnim), true);

    // Transport controls
    if (anim->IsPlaying())
    {
        if (ImGui::Button("Pause"))
            anim->Pause();
    }
    else
    {
        if (ImGui::Button("Play"))
            anim->Resume();
    }
    ImGui::SameLine();
    if (ImGui::Button("Restart"))
        anim->Play(static_cast<size_t>(m_selectedAnim), true);

    // Speed
    if (ImGui::SliderFloat("Speed", &m_animSpeed, 0.0f, 3.0f, "%.2f x"))
        anim->SetSpeed(m_animSpeed);

    // Timeline scrubber (read-only progress)
    float dur = anim->GetCurrentDuration();
    float t   = (dur > 0.0f) ? (anim->GetCurrentTime() / dur) : 0.0f;
    char  overlay[32];
    std::snprintf(overlay, sizeof(overlay), "%.2f / %.2f s",
                  anim->GetCurrentTime(), dur);
    ImGui::ProgressBar(t, ImVec2(-1.0f, 0.0f), overlay);

    ImGui::Spacing();
    ImGui::TextWrapped("AnimationComponent samples each channel with STEP, LINEAR, or "
                       "CUBICSPLINE interpolation matching the glTF spec.  The helper "
                       "functions in animation.h (find_keyframe, cubic_spline_interpolate_vec3, "
                       "apply_pose_to_scene_graph) implement these algorithms directly.");
}

// ---------------------------------------------------------------------------
// Chapter 3 — Rigid-Body Physics
// ---------------------------------------------------------------------------

void TutorialDemoComponent::DrawPhysicsPanel()
{
    if (!ImGui::CollapsingHeader("Chapter 3 — Rigid-Body Physics"))
        return;

    auto *physics = m_engine->GetPhysicsSystem();

    if (ImGui::SliderFloat("Gravity Scale", &m_gravityScale, 0.0f, 3.0f, "%.2f"))
        physics->SetGravity(glm::vec3(0.0f, -9.81f * m_gravityScale, 0.0f));

    ImGui::SliderFloat("Throw Force", &m_throwForce, 1.0f, 30.0f, "%.1f");

    ImGui::Separator();
    ImGui::Text("Fox rigid body:");

    Entity *foxMesh = FindFoxMesh();
    if (!foxMesh)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                           "Fox mesh not found — load Fox.gltf first.");
        return;
    }

    if (!m_activeBody)
    {
        if (ImGui::Button("Add Rigid Body"))
        {
            m_activeEntity = foxMesh;
            m_activeBody   = physics->CreateRigidBody(
                foxMesh, CollisionShape::Box, 1.0f);
            if (m_activeBody)
                m_activeBody->SetKinematic(true);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(hold G to grab while kinematic, release to throw)");
    }
    else
    {
        glm::vec3 vel = m_activeBody->GetLinearVelocity();
        ImGui::Text("Velocity: (%.2f, %.2f, %.2f) m/s", vel.x, vel.y, vel.z);

        if (ImGui::Button("Throw!"))
        {
            m_activeBody->SetKinematic(false);
            m_activeBody->SetLinearVelocity(
                glm::vec3(0.0f, m_throwForce, -m_throwForce * 0.5f));
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
        {
            physics->DestroyRigidBody(m_activeBody);
            m_activeBody   = nullptr;
            m_activeEntity = nullptr;
            m_grabMode     = false;
            if (auto *t = foxMesh->GetComponent<TransformComponent>())
                t->SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
        }
    }

    ImGui::Spacing();
    ImGui::TextWrapped("PhysicsSystem drives Jolt Physics for collision and rigid body simulation. "
                       "Static environment meshes are streamed in and out based on camera distance "
                       "to maintain high performance in large scenes.");
}

// ---------------------------------------------------------------------------
// Chapter 4 — Inverse Kinematics (FABRIK)
// ---------------------------------------------------------------------------

void TutorialDemoComponent::DrawIKPanel()
{
    if (!ImGui::CollapsingHeader("Chapter 4 — Inverse Kinematics (FABRIK)"))
        return;

    ImGui::TextWrapped("Hold [G] in the viewport (not over this panel) to grab the "
                       "Fox mesh and drag it.  Release G to throw it using the drag "
                       "velocity — demonstrating the IK-target-follow → physics-handoff "
                       "pattern from Chapter 4.");

    ImGui::SliderFloat("Grab Depth", &m_grabDepth, 1.0f, 20.0f, "%.1f");

    if (m_grabMode)
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
                           " GRAB ACTIVE  — release G to throw");

    ImGui::Separator();
    ImGui::Text("FABRIK solver visualisation (3-bone chain, animated target):");

    // --- FABRIK canvas ---
    const float cw = ImGui::GetContentRegionAvail().x;
    const float ch = 160.0f;
    ImVec2      cp = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##fab", ImVec2(cw, ch));
    ImDrawList *dl = ImGui::GetWindowDrawList();

    dl->AddRectFilled(cp, ImVec2(cp.x + cw, cp.y + ch),
                      IM_COL32(18, 18, 30, 220), 4.0f);

    // Animated IK target orbits inside the canvas
    float cx = cp.x + cw * 0.5f;
    float cy = cp.y + ch * 0.5f;
    float tx = cx + std::cos(m_fabrikPhase) * cw * 0.38f;
    float ty = cy + std::sin(m_fabrikPhase * 0.7f) * ch * 0.38f;

    // Root is at the bottom-centre
    const float  boneLen = ch * 0.27f;
    glm::vec2    root    = {cx, cp.y + ch - 12.0f};
    glm::vec2    target  = {tx, ty};

    // Initialise all joints at the root so the forward pass has valid data
    // even on the very first frame before any previous values exist.
    glm::vec2 joints[4];
    for (auto &j : joints)
        j = root;

    // Forward pass — pull end-effector toward target
    joints[3] = target;
    for (int i = 2; i >= 0; --i)
    {
        glm::vec2 d = joints[i + 1] - joints[i];
        float     l = glm::length(d);
        if (l > 0.001f)
            joints[i] = joints[i + 1] - (d / l) * boneLen;
    }
    // Backward pass — re-anchor root
    joints[0] = root;
    for (int i = 1; i <= 3; ++i)
    {
        glm::vec2 d = joints[i] - joints[i - 1];
        float     l = glm::length(d);
        if (l > 0.001f)
            joints[i] = joints[i - 1] + (d / l) * boneLen;
    }

    // Draw bones and joints
    for (int i = 0; i < 3; ++i)
    {
        dl->AddLine(ImVec2(joints[i].x, joints[i].y),
                    ImVec2(joints[i + 1].x, joints[i + 1].y),
                    IM_COL32(90, 190, 255, 255), 3.0f);
        dl->AddCircleFilled(ImVec2(joints[i].x, joints[i].y), 5.0f,
                            IM_COL32(255, 200, 70, 255));
    }
    dl->AddCircleFilled(ImVec2(joints[3].x, joints[3].y), 5.0f,
                        IM_COL32(255, 200, 70, 255));

    // Draw IK target
    dl->AddCircle(ImVec2(tx, ty), 9.0f, IM_COL32(255, 70, 70, 255), 0, 2.0f);
    dl->AddText(ImVec2(tx + 12.0f, ty - 8.0f), IM_COL32(255, 100, 100, 255), "target");

    // Root label
    dl->AddText(ImVec2(root.x + 6.0f, root.y - 8.0f), IM_COL32(180, 180, 180, 200), "root");

    ImGui::Spacing();
    ImGui::TextWrapped("update_world_matrices_subtree() in node.h propagates the dirty flag "
                       "upward after each FABRIK pass, so only the affected subtree is "
                       "recomputed — matching the pattern shown in Chapter 4.");
}

// ---------------------------------------------------------------------------
// Chapter 5 — Morph Targets
// ---------------------------------------------------------------------------

void TutorialDemoComponent::DrawMorphPanel()
{
    if (!ImGui::CollapsingHeader("Chapter 5 — Morph Targets"))
        return;

    bool hasMorphModel = false;
    for (const auto &e : m_engine->GetEntities())
    {
        const auto &n = e->GetName();
        if (n.rfind("AnimatedMorphCube_", 0) == 0 ||
            n.rfind("MorphPrimitivesTest_", 0) == 0)
        {
            hasMorphModel = true;
            break;
        }
    }

    if (!hasMorphModel)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f),
                           "No morph-target model loaded.\n"
                           "Run:  assets/download_samples.sh\n"
                           "then call LoadGLTFModel for AnimatedMorphCube.gltf.");
    }

    ImGui::Text("Morph target weights (for reference; engine applies via compute):");
    ImGui::SliderFloat("Weight 0", &m_morphWeights[0], 0.0f, 1.0f);
    ImGui::SliderFloat("Weight 1", &m_morphWeights[1], 0.0f, 1.0f);
    ImGui::SliderFloat("Weight 2", &m_morphWeights[2], 0.0f, 1.0f);
    ImGui::SliderFloat("Weight 3", &m_morphWeights[3], 0.0f, 1.0f);

    ImGui::Spacing();
    ImGui::TextWrapped("morph_accumulate.slang accumulates weighted position/normal/tangent "
                       "deltas in a single compute dispatch.  Weight animation uses the glTF "
                       "WEIGHTS AnimationPath, sampled by the same keyframe pipeline as "
                       "translation and rotation channels.");
}

// ---------------------------------------------------------------------------
// Chapter 6 — Debug Heatmap
// ---------------------------------------------------------------------------

void TutorialDemoComponent::DrawDebugPanel()
{
    if (!ImGui::CollapsingHeader("Chapter 6 — Debug & Skinning Heatmap"))
        return;

    ImGui::TextWrapped("pbr_heatmap.slang provides two fragment entry points that share "
                       "one vertex shader: fragment_dominant_bone colours each pixel by "
                       "the bone index with highest weight; fragment_weight_distribution "
                       "goes green for balanced weights and red for a single dominant bone.");

    ImGui::Spacing();
    ImGui::Text("Scene summary:");

    int withAnim = 0, withTransform = 0;
    auto snapshot = SnapshotEntities(m_engine);
    for (auto *e : snapshot)
    {
        if (e->GetComponent<AnimationComponent>())
            ++withAnim;
        if (e->GetComponent<TransformComponent>())
            ++withTransform;
    }
    ImGui::BulletText("Entities total        : %zu",
                      snapshot.size());
    ImGui::BulletText("With AnimationComponent: %d", withAnim);
    ImGui::BulletText("With TransformComponent: %d", withTransform);

    if (auto *physics = m_engine->GetPhysicsSystem())
    {
        glm::vec3 g = physics->GetGravity();
        ImGui::BulletText("Gravity               : (%.2f, %.2f, %.2f)", g.x, g.y, g.z);
    }
}

// ---------------------------------------------------------------------------
// Update — called every frame after ImGui::NewFrame()
// ---------------------------------------------------------------------------

void TutorialDemoComponent::Update(std::chrono::milliseconds deltaTime)
{
    // Advance FABRIK canvas animation
    m_fabrikPhase += static_cast<float>(deltaTime.count()) * 0.001f;

    // ---- G-key grab & throw ----
    // ImGui::GetIO().WantCaptureKeyboard is true when the user is typing into an
    // ImGui input field; skip grab input in that case so text entry still works.
    const ImGuiIO &io      = ImGui::GetIO();
    // GLFW stores raw key codes in KeysDown; 'G' == GLFW_KEY_G == 71 (ASCII).
    bool           grabKey = ImGui::IsKeyDown(ImGuiKey_G) && !io.WantCaptureKeyboard;

    if (grabKey)
    {
        glm::vec3 worldPt = MouseToWorld(m_grabDepth);

        if (!m_grabMode)
        {
            // First frame of grab — initialise state
            m_grabMode      = true;
            m_grabCurrent   = worldPt;
            m_throwVelocity = {};
            m_grabOffset    = {};

            // Attach a kinematic rigid body to the first Fox mesh entity found.
            // If a body already exists (e.g. from the physics panel), reuse it.
            if (!m_activeBody)
            {
                Entity *fox = FindFoxMesh();
                if (fox)
                {
                    m_activeEntity = fox;
                    m_activeBody   = m_engine->GetPhysicsSystem()->CreateRigidBody(
                        fox, CollisionShape::Box, 1.0f);
                }
            }
            if (m_activeBody)
                m_activeBody->SetKinematic(true);

            // Capture the offset between the object's current position and the cursor-ray
            // point so the object is grabbed in place (no teleport to the cursor) and then
            // moves relative to cursor motion.
            if (m_activeEntity)
            {
                if (auto* t = m_activeEntity->GetComponent<TransformComponent>())
                    m_grabOffset = t->GetPosition() - worldPt;
            }
        }

        // Compute instantaneous velocity from position delta
        float dt = static_cast<float>(deltaTime.count()) * 0.001f;
        if (dt > 0.0f)
            m_throwVelocity = (worldPt - m_grabCurrent) / dt;
        m_grabCurrent = worldPt;

        // Drag the rigid body to follow the cursor, preserving the initial grab offset.
        if (m_activeBody)
            m_activeBody->SetPosition(worldPt + m_grabOffset);
    }
    else if (m_grabMode)
    {
        // G released — switch to dynamic and apply throw velocity
        m_grabMode = false;
        if (m_activeBody)
        {
            m_activeBody->SetKinematic(false);
            m_activeBody->SetLinearVelocity(m_throwVelocity);
            // The physics system now owns this entity's transform. Tell the animation
            // system to stop driving it, otherwise the animation resets the transform to
            // the animated pose every frame and the object oscillates with the physics pose.
            SetEntityPhysicsOwned(m_activeEntity, true);
            // Body is now owned by the physics system; we stop tracking it here
            // so the throw arc plays out without further interference.
            m_activeBody   = nullptr;
            m_activeEntity = nullptr;
        }
    }

    // ---- Right-click to spawn balls ----
    // Right-click in the viewport to spawn a new "ball" (using procedural sphere)
    // with a dynamic rigid body (mass=1.0).
    if (ImGui::IsMouseClicked(1) && !io.WantCaptureMouse)
    {
        glm::vec3 spawnPos = MouseToWorld(2.0f); // Spawn 2m in front of camera
        std::string ballName = "Ball_" + std::to_string(m_ballCounter++);
        if (Entity* ball = m_engine->CreateEntity(ballName)) {
            // Update the engine's last spawned ball global for optimized camera tracking
            extern Entity* g_lastSpawnedBall;
            g_lastSpawnedBall = ball;

            ball->AddComponent<TransformComponent>()->SetPosition(spawnPos);
            auto* mesh = ball->AddComponent<MeshComponent>();
            mesh->CreateSphere(0.2f, glm::vec3(1.0f, 0.2f, 0.2f), 16);
            m_engine->GetPhysicsSystem()->CreateRigidBody(ball, CollisionShape::Sphere, 1.0f);
            m_engine->GetRenderer()->preAllocateEntityResources(ball);
        }
    }

    // ---- Tutorial ImGui window ----
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 385.0f, 10.0f),
                            ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(375.0f, io.DisplaySize.y - 20.0f),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.88f);

    ImGui::Begin("Advanced glTF Tutorial", nullptr, ImGuiWindowFlags_NoCollapse);

    ImGui::Text("%.0f FPS  |  [G] Grab & Throw  |  [WASD] Camera", io.Framerate);
    if (m_grabMode)
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "  GRAB MODE ACTIVE");
    ImGui::Separator();

    DrawSceneGraphPanel();
    DrawAnimationPanel();
    DrawPhysicsPanel();
    DrawIKPanel();
    DrawMorphPanel();
    DrawDebugPanel();

    ImGui::End();
}
