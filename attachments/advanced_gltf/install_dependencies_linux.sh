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

# Installs all dependencies for the Advanced glTF tutorial on Linux.
# Delegates to the simple_engine install script (which handles glm, GLFW,
# OpenAL, tinygltf, KTX, etc.) then notes that JoltPhysics is fetched
# automatically by CMake via FetchContent — no manual installation needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SE_SCRIPT="${SCRIPT_DIR}/../simple_engine/install_dependencies_linux.sh"

if [[ ! -f "${SE_SCRIPT}" ]]; then
    echo "Error: simple_engine install script not found at ${SE_SCRIPT}" >&2
    exit 1
fi

echo "=== Installing simple_engine dependencies ==="
bash "${SE_SCRIPT}"

echo ""
echo "=== Advanced glTF tutorial additional dependencies ==="
echo "JoltPhysics v5.2.0 is fetched automatically by CMake (FetchContent)."
echo "No additional manual installation is required."
echo ""
echo "Build instructions:"
echo "  cd attachments/advanced_gltf"
echo "  mkdir build && cd build"
echo "  cmake .."
echo "  cmake --build . --parallel"
