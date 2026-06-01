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
#pragma once

// =============================================================================
// OmmIntegration — wires OpacityMicromapBuilder into the engine startup
// =============================================================================
//
// This thin wrapper class is the single "seam" between the course module and
// the engine.  It:
//
//   1. Builds micromaps for every alpha-masked mesh after textures are loaded.
//   2. Patches the geometry pNext chain just before each BLAS is constructed.
//   3. Registers the ImGui panel so the engine window shows OMM controls.
//   4. Cleans everything up when the scene is replaced or the engine shuts down.
//
// Usage in main.cpp (or engine initialisation code):
//
//   OmmIntegration omm;
//   omm.init(engine, renderer, modelLoader);
//   // ... load scene ...
//   omm.buildMicromaps(ommConfig);   // call once after textures are ready
//
// From this point on, whenever the engine calls buildAccelerationStructures(),
// it will find the micromap pNext chains already attached to the geometry.
//
// See: en/Building_a_Simple_Engine/Courses/Opacity_Micromaps/05_implementation_overview.adoc
// =============================================================================

#include "opacity_micromap_builder.h"
#include <functional>
#include <vector>
#include <atomic>

// Forward declarations
class Renderer;
class Engine;
class ModelLoader;
class MeshComponent;

// ---------------------------------------------------------------------------
// OmmSceneStats — reported to the ImGui panel
// ---------------------------------------------------------------------------
struct OmmSceneStats {
  uint32_t meshesConsidered  = 0; // alpha-masked meshes examined
  uint32_t micromapsBuilt    = 0; // VkAccelerationStructureKHR micromap objects created
  uint64_t totalGpuBytes     = 0;
  float    avgPctOpaque      = 0.f;
  float    avgPctTransparent = 0.f;
  float    avgPctUnknown     = 0.f;
};

// ---------------------------------------------------------------------------
// OmmIntegration
// ---------------------------------------------------------------------------
class OmmIntegration {
public:
  OmmIntegration()  = default;
  ~OmmIntegration() { shutdown(); }

  // Non-copyable
  OmmIntegration(const OmmIntegration&)            = delete;
  OmmIntegration& operator=(const OmmIntegration&) = delete;

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Initialise with the engine subsystems.  Call once after the renderer
  /// and model loader are ready, before loading any scene.
  void init(Renderer& renderer, ModelLoader& modelLoader);

  /// Build micromaps for all currently-loaded alpha-masked meshes and register
  /// the BLAS pNext hooks.  Call after all mesh textures are uploaded.
  /// Safe to call multiple times (resets and rebuilds on each call).
  void buildMicromaps(const OmmConfig& config = {});

  /// Release all GPU resources and unregister the ImGui panel.
  void shutdown();

  // ── Accessors ─────────────────────────────────────────────────────────────

  [[nodiscard]] bool            isSupported()  const { return m_builder.isSupported(); }
  [[nodiscard]] const OmmSceneStats& stats()   const { return m_stats; }
  [[nodiscard]] OpacityMicromapBuilder& builder()    { return m_builder; }

  /// Returns the pNext chain to attach to a geometry triangles struct for the
  /// given MeshComponent, or nullptr when no micromap was built for it.
  [[nodiscard]] void* getPNextForMesh(const MeshComponent* mesh) const;

  [[nodiscard]] bool isBuildInProgress() const { return m_buildInProgress.load(); }

private:
  OpacityMicromapBuilder m_builder;
  OmmSceneStats          m_stats{};

  Renderer*    m_renderer    = nullptr;
  ModelLoader* m_modelLoader = nullptr;

  bool m_initialised = false;
  std::atomic<bool> m_buildInProgress{false};
};
