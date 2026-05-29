#!/usr/bin/env bash
# Copyright (c) 2026 Holochip Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Downloads the subset of Khronos glTF Sample Assets needed for the Advanced glTF tutorial.
# All models are from https://github.com/KhronosGroup/glTF-Sample-Assets
# and are licensed CC0-1.0 or CC-BY-4.0 as noted.
#
# Pin: commit 2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf (2026-04-27)
# Update this hash after auditing the CHANGELOG for breaking directory renames.

set -euo pipefail

REPO="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets"
COMMIT="2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf"
BASE="${REPO}/${COMMIT}/Models"
OUT="$(dirname "$0")"

download() {
    local dir="$1"; shift
    mkdir -p "${OUT}/${dir}"
    for file in "$@"; do
        local dest="${OUT}/${dir}/${file##*/}"
        if [ ! -f "${dest}" ]; then
            echo "  ↓  ${dir}/${file##*/}"
            curl -fsSL "${BASE}/${dir}/${file}" -o "${dest}"
        else
            echo "  ✓  ${dir}/${file##*/} (cached)"
        fi
    done
}

echo "=== Downloading glTF Sample Assets ==="

# --- Chapter 1: Scene Graph Hierarchy ---
# BoxAnimated  — CC0   — simple node hierarchy with animated transforms
echo "→ BoxAnimated"
download "BoxAnimated/glTF" \
    "BoxAnimated.gltf" \
    "BoxAnimated0.bin"

# RiggedSimple — CC0 — minimal 2-joint skinned mesh
echo "→ RiggedSimple"
download "RiggedSimple/glTF" \
    "RiggedSimple.gltf" \
    "RiggedSimple0.bin"

# --- Chapter 2: Skeletal Compute Skinning ---
# SimpleSkin   — CC0 — the canonical 2-joint skinning tutorial model
echo "→ SimpleSkin"
download "SimpleSkin/glTF" \
    "SimpleSkin.gltf" \
    "SimpleSkin_animation.bin" \
    "SimpleSkin_geometry.bin" \
    "SimpleSkin_inverseBindMatrices.bin" \
    "SimpleSkin_skinningData.bin"

# Fox          — CC-BY-4.0 (Sketchfab)
echo "→ Fox (CC-BY-4.0)"
download "Fox/glTF" \
    "Fox.gltf" \
    "Fox.bin" \
    "Texture.png"

# --- Chapter 3: Interpolation ---
# InterpolationTest — CC0 — exercises STEP / LINEAR / CUBICSPLINE
echo "→ InterpolationTest"
download "InterpolationTest/glTF" \
    "InterpolationTest.gltf" \
    "InterpolationTest_data.bin" \
    "InterpolationTest_img0.png"

# --- Chapter 5: Morph Targets ---
# AnimatedMorphCube — CC0 — simplest morph target demo
echo "→ AnimatedMorphCube"
download "AnimatedMorphCube/glTF" \
    "AnimatedMorphCube.gltf" \
    "AnimatedMorphCube.bin"

# MorphPrimitivesTest — CC0 — multiple target/primitive combinations
echo "→ MorphPrimitivesTest"
download "MorphPrimitivesTest/glTF" \
    "MorphPrimitivesTest.gltf" \
    "MorphPrimitivesTest.bin" \
    "uv_texture.jpg"

echo ""
echo "=== Done ==="
echo ""
echo "NOTE: Physics ragdoll metadata (glTF extras.collider / extras.constraint)"
echo "      is a tutorial-specific schema — no standard Khronos sample includes it."
echo "      Use the add_physics_extras.py script to annotate RiggedSimple or Fox"
echo "      before testing the Physics Integration chapter."
