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
#pragma once

#include "component.h"

#include <chrono>
#include <glm/glm.hpp>

class Engine;
class RigidBody;
class Entity;

/**
 * Component that renders the Advanced glTF tutorial ImGui panels and handles
 * interactive grab-and-throw via the G key.  One instance lives on a dedicated
 * "TutorialDemo" entity and its Update() is called every frame after
 * ImGui::NewFrame(), so all ImGui calls inside are safe.
 */
class TutorialDemoComponent final : public Component
{
  public:
    explicit TutorialDemoComponent(Engine *engine);
    void Update(std::chrono::milliseconds deltaTime) override;

  private:
    Engine *m_engine;

    // Chapter 2 — Animation
    int   m_selectedAnim = 0;
    float m_animSpeed    = 1.0f;

    // Chapter 3 — Physics
    float      m_gravityScale  = 1.0f;
    float      m_throwForce    = 8.0f;
    RigidBody *m_activeBody    = nullptr;
    Entity    *m_activeEntity  = nullptr;

    // Chapter 4 — IK / Grab & Throw
    bool      m_grabMode      = false;
    glm::vec3 m_grabCurrent   = {};
    glm::vec3 m_throwVelocity = {};
    glm::vec3 m_grabOffset    = {}; // offset from cursor-ray point to the grabbed object, captured on grab
    float     m_grabDepth     = 4.0f;
    float     m_fabrikPhase   = 0.0f;   // drives the animated FABRIK canvas target

    // Chapter 5 — Morph
    float m_morphWeights[4] = {};

    // helpers
    Entity    *FindFoxMesh() const;
    glm::vec3  MouseToWorld(float depth) const;

    int m_ballCounter = 0;

    void DrawSceneGraphPanel();
    void DrawAnimationPanel();
    void DrawPhysicsPanel();
    void DrawIKPanel();
    void DrawMorphPanel();
    void DrawDebugPanel();
};
