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

// =============================================================================
// OmmImGuiPanel — implementation
// =============================================================================
//
// This panel is registered via Renderer::RegisterImGuiPanel() and injected
// directly into the engine's "Renderer" ImGui window.  It is completely
// self-contained and only reads public accessors on OmmIntegration and
// Renderer.
//
// LAYOUT
// ------
// The panel opens a collapsing header "Opacity Micromaps (Course Module)"
// inside the existing "Renderer" ImGui window — no extra window needed.
// Inside it shows:
//   • Hardware support status (green/red)
//   • Build summary (meshes built, GPU memory)
//   • Coloured progress bars: opaque / transparent / unknown percentages
//   • Live config controls: subdivision level, thresholds, unknown-state toggle
//   • A "Rebuild Micromaps" button that calls OmmIntegration::buildMicromaps()
// =============================================================================

#include "omm_imgui_panel.h"
#include "../imgui/imgui.h"
#include "../renderer.h"

namespace {

// Static config instance — survives across frames.
OmmConfig s_config{};

// Colourful helpers matching the engine ImGui style (dark theme).
const ImVec4 kGreen  { 0.30f, 0.85f, 0.40f, 1.f };
const ImVec4 kRed    { 0.95f, 0.30f, 0.30f, 1.f };
const ImVec4 kYellow { 0.98f, 0.82f, 0.25f, 1.f };
const ImVec4 kBlue   { 0.40f, 0.65f, 1.00f, 1.f };
const ImVec4 kGrey   { 0.60f, 0.60f, 0.60f, 1.f };
const ImVec4 kWhite  { 1.00f, 1.00f, 1.00f, 1.f };

void coloredBar(float fraction, ImVec4 colour, const char* label) {
  const float barW = ImGui::GetContentRegionAvail().x * 0.60f;
  ImDrawList* dl   = ImGui::GetWindowDrawList();
  ImVec2 pos       = ImGui::GetCursorScreenPos();

  // Background
  dl->AddRectFilled(pos,
                    ImVec2(pos.x + barW, pos.y + 12.f),
                    IM_COL32(50, 50, 50, 200), 3.f);
  // Fill
  const float fillW = barW * std::max(0.f, std::min(fraction, 1.f));
  if (fillW > 0.f)
    dl->AddRectFilled(pos,
                      ImVec2(pos.x + fillW, pos.y + 12.f),
                      ImGui::ColorConvertFloat4ToU32(colour), 3.f);
  ImGui::Dummy(ImVec2(barW, 14.f));
  ImGui::SameLine();
  ImGui::TextColored(colour, "%s  %.1f%%", label, fraction * 100.f);
}

} // namespace


// =============================================================================
// OmmImGuiPanel
// =============================================================================

OmmConfig& OmmImGuiPanel::mutableConfig() {
  return s_config;
}

void OmmImGuiPanel::draw(OmmIntegration& omm, Renderer* /*renderer*/) {
  ImGui::Spacing();
  ImGui::Separator();

  // ── Collapsing header ─────────────────────────────────────────────────────
  const bool open = ImGui::CollapsingHeader("Opacity Micromaps  (Course Module)");
  if (!open) return;

  ImGui::Spacing();

  // ── Hardware support ──────────────────────────────────────────────────────
  if (omm.isSupported()) {
    ImGui::TextColored(kGreen, "[OK]  VK_KHR_opacity_micromap:  ENABLED");
  } else {
    ImGui::TextColored(kRed,   "[--]  VK_KHR_opacity_micromap:  NOT SUPPORTED");
    ImGui::TextWrapped("This GPU does not support opacity micromaps.  "
                       "Shadow rays will continue to use the any-hit shader path "
                       "for alpha-masked geometry, which is correct but slower.  "
                       "A GPU with NVIDIA Ada Lovelace or newer architecture is required.");
    ImGui::Spacing();
    return;
  }

  // ── Build summary ─────────────────────────────────────────────────────────
  const OmmSceneStats& st = omm.stats();

  ImGui::TextColored(kBlue, "Build summary");
  ImGui::Indent();
    ImGui::Text("Alpha-masked meshes examined : %u", st.meshesConsidered);
    ImGui::Text("Micromaps built              : %u", st.micromapsBuilt);
    if (st.totalGpuBytes > 0) {
      if (st.totalGpuBytes < 1024 * 1024)
        ImGui::Text("GPU memory                   : %u KiB",
                    static_cast<uint32_t>(st.totalGpuBytes / 1024));
      else
        ImGui::Text("GPU memory                   : %.2f MiB",
                    static_cast<double>(st.totalGpuBytes) / (1024.0 * 1024.0));
    } else {
      ImGui::TextColored(kGrey, "GPU memory                   : (none built yet)");
    }
  ImGui::Unindent();

  // ── Average micro-triangle classification breakdown ────────────────────────
  if (st.micromapsBuilt > 0) {
    ImGui::Spacing();
    ImGui::TextColored(kBlue, "Average micro-triangle breakdown  (across %u meshes)", st.micromapsBuilt);
    ImGui::Indent();
      coloredBar(st.avgPctOpaque,      kGreen,  "Opaque       (hardware blocks — no shader)");
      coloredBar(st.avgPctTransparent, kBlue,   "Transparent  (hardware passes — no shader)");
      coloredBar(st.avgPctUnknown,     kYellow, "Unknown      (any-hit shader fires)");

      const float shaderSaved = st.avgPctOpaque + st.avgPctTransparent;
      ImGui::Spacing();
      ImGui::TextColored(shaderSaved > 0.8f ? kGreen : kYellow,
                         "Shader invocations avoided:  ~%.0f%%", shaderSaved * 100.f);
    ImGui::Unindent();
  } else if (st.meshesConsidered > 0) {
    ImGui::TextColored(kYellow, "No micromaps were built yet — click 'Build Micromaps' below.");
  } else {
    ImGui::TextColored(kGrey, "No alpha-masked meshes found in the current scene.");
  }

  // ── Configuration controls ────────────────────────────────────────────────
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::TextColored(kBlue, "Configuration");
  ImGui::Indent();

    // Subdivision level selector (0-4 as radio-style buttons)
    ImGui::Text("Subdivision level:");
    ImGui::SameLine();
    ImGui::TextColored(kGrey, " (higher = more accurate, more GPU memory)");
    for (int lvl = 0; lvl <= 4; ++lvl) {
      if (lvl > 0) ImGui::SameLine();
      char lbl[12];
      std::snprintf(lbl, sizeof(lbl), " %d ", lvl);
      const bool active = (static_cast<int>(s_config.subdivisionLevel) == lvl);
      if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.55f, 0.90f, 1.f));
      if (ImGui::SmallButton(lbl))
        s_config.subdivisionLevel = static_cast<uint32_t>(lvl);
      if (active) ImGui::PopStyleColor();
    }
    // Show micro-tri count for the chosen level
    uint32_t microCount = 1;
    for (uint32_t i = 0; i < s_config.subdivisionLevel; ++i) microCount *= 4;
    ImGui::SameLine();
    ImGui::TextColored(kGrey, " = %u micro-tris/triangle", microCount);

    ImGui::Spacing();
    ImGui::Checkbox("Allow Unknown state (edge micro-tris → any-hit shader)",
                    &s_config.allowUnknownState);
    ImGui::SliderFloat("Opaque threshold",      &s_config.opaqueThreshold,
                       0.50f, 1.00f, "%.2f");
    ImGui::SliderFloat("Transparent threshold", &s_config.transparentThreshold,
                       0.00f, 0.49f, "%.2f");
    int spp = static_cast<int>(s_config.samplesPerMicroTriangle);
    if (ImGui::SliderInt("Samples per micro-tri (build quality)", &spp, 1, 8))
      s_config.samplesPerMicroTriangle = static_cast<uint32_t>(std::max(1, spp));

  ImGui::Unindent();

  // ── Rebuild button ────────────────────────────────────────────────────────
  ImGui::Spacing();
  if (omm.isBuildInProgress()) {
    ImGui::TextColored(kYellow, "Build in progress... please wait.");
  } else {
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.20f, 0.55f, 0.20f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.75f, 0.25f, 1.f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.15f, 0.40f, 0.15f, 1.f));
    if (ImGui::Button("  Rebuild Micromaps  ", ImVec2(-1.f, 0.f))) {
      omm.buildMicromaps(s_config);
    }
    ImGui::PopStyleColor(3);
  }

  // ── Conceptual explanation (collapsed by default) ─────────────────────────
  ImGui::Spacing();
  if (ImGui::TreeNode("What are Opacity Micromaps?")) {
    ImGui::TextWrapped(
      "Every triangle in an alpha-masked mesh (foliage, fences, curtains) "
      "is subdivided into a grid of micro-triangles.  Each micro-triangle is "
      "pre-classified as Opaque, Transparent, or Unknown before any frame is drawn.");
    ImGui::Spacing();
    ImGui::TextWrapped(
      "During shadow-ray traversal the GPU hardware reads this classification "
      "table directly — no shader code runs.  Only the narrow Unknown band at "
      "alpha-gradient edges still invokes the any-hit shader.  For a typical "
      "tree canopy this eliminates over 90%% of shadow-ray shader invocations.");
    ImGui::Spacing();
    ImGui::TextColored(kBlue,
      "Course reference:  "
      "en/Building_a_Simple_Engine/Courses/Opacity_Micromaps/");
    ImGui::TreePop();
  }

  ImGui::Spacing();
}
