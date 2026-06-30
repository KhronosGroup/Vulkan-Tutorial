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
// OmmIntegration — implementation
// =============================================================================
//
// This file wires OpacityMicromapBuilder into the engine without touching any
// of the core engine source files.  The integration path is:
//
//   OmmIntegration::init()           — store renderer & model-loader pointers
//   OmmIntegration::buildMicromaps() — iterate alpha-masked meshes,
//                                      call builder.buildForMesh() for each,
//                                      then register the ImGui panel via
//                                      renderer.RegisterImGuiPanel().
//   OmmIntegration::getPNextForMesh()— called by renderer_ray_query.cpp
//                                      (or any future patched build loop)
//                                      to retrieve the per-mesh pNext chain.
//   OmmIntegration::shutdown()       — builder.reset(), unregister panel.
//
// TEXTURE ACCESS STRATEGY
// -----------------------
// The engine uploads textures to the GPU via stb_image + ModelLoader.  The CPU
// pixel data lives in the tinygltf image buffers that ModelLoader holds
// internally.  We query the model loader for the Material's albedoTexturePath,
// then ask the renderer to resolve that path back to its raw pixel data via
// the helper Renderer::GetTexturePixels() which we add only as a protected
// accessor (already lives in the same translation unit family via renderer.h).
//
// If raw pixels are not accessible (e.g., the texture was streamed from disk),
// we fall back to skipping OMM for that mesh — the any-hit shader continues
// to handle it as before.
// =============================================================================

#include "omm_integration.h"
#include "omm_imgui_panel.h"

#include "../renderer.h"
#include "../model_loader.h"
#include "../mesh_component.h"

#include <iostream>
#include <numeric>
#include <thread>
#include <chrono>

void OmmIntegration::init(Renderer& renderer, ModelLoader& modelLoader) {
  m_renderer    = &renderer;
  m_modelLoader = &modelLoader;
  m_builder.init(renderer);
  renderer.RegisterMicromapProvider([this](const MeshComponent* m) { return getPNextForMesh(m); });
  m_initialised = true;

  // We register the ImGui panel immediately so it's always available,
  // even if no micromaps are built yet.
  m_renderer->RegisterImGuiPanel(
      [this](Renderer* r) { OmmImGuiPanel::draw(*this, r); });

  if (m_builder.isSupported()) {
    std::cout << "[OMM] Integration layer initialised — ready to build micromaps.\n";
  } else {
    std::cout << "[OMM] Hardware does not support VK_KHR_opacity_micromap.\n"
              << "      Shadow rays will continue to use the any-hit shader path.\n";
  }
}

// ---------------------------------------------------------------------------
// buildMicromaps
// ---------------------------------------------------------------------------
// Iterates every mesh known to the model loader, identifies those with
// alpha-masked materials, and calls buildForMesh() for each one.
//
// After building, it re-registers the ImGui panel so up-to-date statistics
// are displayed the next time the user opens the "Renderer" window.
// ---------------------------------------------------------------------------
void OmmIntegration::buildMicromaps(const OmmConfig& config) {
  if (!m_initialised) {
    std::cerr << "[OMM] buildMicromaps() called before init().\n";
    return;
  }
  if (m_buildInProgress.load()) {
    return; // Already building
  }

  m_buildInProgress.store(true);

  // We use a separate thread for the build loop so the engine UI stays responsive.
  // The classification tasks within the builder will still use the renderer's thread pool.
  std::thread([this, config]() {
    try {
      if (!m_builder.isSupported()) {
        m_buildInProgress.store(false);
        return;
      }

      // Create a temporary builder to hold the new results without destroying the old ones yet.
      // This prevents "Lost Device" crashes caused by destroying micromaps currently in use by the GPU.
      OpacityMicromapBuilder tempBuilder;
      tempBuilder.init(*m_renderer);

      // Wait for textures, then wait for the render thread to drain deferred
      // mesh uploads (which populates meshResources).  OMM must not scan
      // before meshResources is fully populated or it will find 0 meshes.
      // WaitForAllTextureTasks only confirms jobs are ENQUEUED to upload workers,
      // not that they have been PROCESSED (StoreRawTexturePixels called).
      // We must also wait for the raw pixel cache to stabilise before scanning.
      std::cout << "[OMM] Waiting for texture jobs to enqueue...\n";
      m_renderer->WaitForAllTextureTasks();
      std::cout << "[OMM] Waiting for mesh resources to settle...\n";
      if (!m_renderer->WaitForMeshResourcesToSettle()) {
        std::cout << "[OMM] Timed out waiting for mesh resources; proceeding with partial mesh list.\n";
      }
      std::cout << "[OMM] Waiting for raw pixel cache to settle...\n";
      if (!m_renderer->WaitForRawPixelCacheToSettle()) {
        std::cout << "[OMM] Timed out waiting for pixel cache; some textures may be missing.\n";
      }

      std::vector<const MeshComponent*> meshes = m_renderer->GetRegisteredMeshes();
      std::cout << "[OMM] Scanning " << meshes.size() << " registered meshes.\n";
      OmmSceneStats tempStats{};

      float sumOpaque = 0.f, sumTransparent = 0.f, sumUnknown = 0.f;

      for (const MeshComponent* mesh : meshes) {
        if (!mesh) continue;

        const uint32_t matIdx = mesh->GetInstanceCount() > 0
                                    ? mesh->GetInstance(0).materialIndex
                                    : 0;

        const Material* mat = m_modelLoader->GetMaterialByIndex(matIdx);
        if (!mat) continue;
        if (mat->alphaMode != "MASK") continue;

        ++tempStats.meshesConsidered;

        uint32_t       texW = 0, texH = 0, texCh = 0;
        const uint8_t* pixels = m_renderer->GetRawTexturePixels(
                                    mat->albedoTexturePath, &texW, &texH, &texCh);

        if (!pixels || texW == 0 || texH == 0) {
          std::cout << "[OMM] No CPU pixel data for texture '"
                    << mat->albedoTexturePath << "' (alpha masked). Skipping mesh.\n";
          continue;
        }

        std::cout << "[OMM] Building for mesh with texture '" << mat->albedoTexturePath
                  << "' (" << texW << "x" << texH << ")\n";

        OmmMeshInfo info{};
        try {
          info = tempBuilder.buildForMesh(mesh, pixels, texW, texH, texCh, config);
        } catch (const std::exception& e) {
          std::cerr << "[OMM] buildForMesh failed for '" << mat->albedoTexturePath
              << "': " << e.what() << "\n";
          continue;
        }

        if (info.built) {
          ++tempStats.micromapsBuilt;
          sumOpaque       += info.pctOpaque;
          sumTransparent  += info.pctTransparent;
          sumUnknown      += info.pctUnknown;
        }
      }

      tempStats.totalGpuBytes = tempBuilder.totalGpuBytes();
      if (tempStats.micromapsBuilt > 0) {
        const float inv            = 1.f / static_cast<float>(tempStats.micromapsBuilt);
        tempStats.avgPctOpaque       = sumOpaque      * inv;
        tempStats.avgPctTransparent  = sumTransparent * inv;
        tempStats.avgPctUnknown      = sumUnknown     * inv;
      }

      // Swap the new builder content into the active builder.
      m_builder.swap(tempBuilder);
      m_stats = tempStats;

      std::cout << "[OMM] Build complete: "
                << m_stats.micromapsBuilt << "/" << m_stats.meshesConsidered
                << " alpha-masked meshes have micromaps.\n"
                << "      Total GPU: " << m_stats.totalGpuBytes / 1024 << " KiB\n";

      // Trigger an acceleration structure rebuild.
      m_renderer->RequestAccelerationStructureBuild("OMM rebuild complete");

      // Give the GPU and renderer some time to build the new AS and finish current frames
      // before letting tempBuilder (which now holds the OLD resources) be destroyed.
      std::this_thread::sleep_for(std::chrono::seconds(2));

    } catch (const std::exception& e) {
      std::cerr << "[OMM] Error during background build: " << e.what() << std::endl;
    }

    m_buildInProgress.store(false);
  }).detach();
}

void OmmIntegration::shutdown() {
  if (m_renderer) {
    m_renderer->UnregisterImGuiPanel();
    m_renderer->RegisterMicromapProvider(nullptr);
  }
  m_builder.reset();
  m_initialised = false;
}

void* OmmIntegration::getPNextForMesh(const MeshComponent* mesh) const {
  const OmmMeshInfo* info = m_builder.getInfo(mesh);
  return info ? info->pNextChain : nullptr;
}
