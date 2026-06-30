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
// OmmImGuiPanel — in-engine UI for the Opacity Micromaps course module
// =============================================================================
//
// This panel is injected into the engine's "Renderer" ImGui window via
// Renderer::RegisterImGuiPanel().  It shows:
//
//   ┌─────────────────────────────────────────────────────┐
//   │  Opacity Micromaps (Course Module)                  │
//   │  Hardware support:  ✓ / ✗                           │
//   │  Micromaps built:  12 / 15 alpha-masked meshes      │
//   │  GPU memory:  384 KiB                               │
//   │                                                     │
//   │  [Progress bars]  Opaque / Transparent / Unknown    │
//   │                                                     │
//   │  Subdivision level  [ 0 ][ 1 ][●2][ 3 ][ 4 ]       │
//   │  ☑ Allow Unknown state                              │
//   │  Opaque threshold   ───●──── 0.95                   │
//   │  Transparent thresh ─●────── 0.05                   │
//   │                                                     │
//   │  [ Rebuild Micromaps ]                              │
//   └─────────────────────────────────────────────────────┘
//
// The panel is entirely self-contained and does not modify any engine source.
// =============================================================================

#include "omm_integration.h"

class Renderer;

namespace OmmImGuiPanel {

  /// Draw the panel into the currently-open ImGui window.
  /// Called automatically from the Renderer::RegisterImGuiPanel() callback.
  void draw(OmmIntegration& omm, Renderer* renderer);

  /// The config that the "Rebuild" button will use.  Modified in-place by the
  /// panel controls.  External code can read it back after calling draw().
  OmmConfig& mutableConfig();

} // namespace OmmImGuiPanel
