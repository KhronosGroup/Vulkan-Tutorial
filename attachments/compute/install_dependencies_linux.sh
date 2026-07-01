#!/bin/bash
# Install dependencies for the Advanced Vulkan Compute tutorial demos.
# The compute demos require: GLFW (windowed demos), GLM, stb.
# They do NOT require tinyobjloader, tinygltf, or KTX.
# slangc must be provided by the Vulkan SDK (1.4.335+).
#
# Chapter 05 (OpenCL on Vulkan) requires the OpenCL ICD loader + headers and the
# clspv compiler + clvk runtime to RUN. By default this script ALWAYS builds
# them: the Chapter 05 sample cannot run without clspv. Building clspv pulls in
# LLVM, so the first run can take 20-40 minutes.
#
# Pass --skip-clspv-clvk (or set SKIP_CLSPV_CLVK=1) to skip that source build.
# CMake's find_program/find_library calls degrade gracefully without them:
# the Chapter 05 binary still builds (with its clspv AOT path and clvk runtime
# path disabled), which is all a build-only smoke test needs. Use this only
# for CI build verification — it does not produce a runnable Chapter 05 demo.
# clspv/clvk are built into ~/opencl-on-vulkan.

set -e

SKIP_CLSPV_CLVK="${SKIP_CLSPV_CLVK:-0}"
for arg in "$@"; do
    case "$arg" in
        --skip-clspv-clvk) SKIP_CLSPV_CLVK=1 ;;
    esac
done

TOOLS_DIR="${HOME}/opencl-on-vulkan"

echo "Installing dependencies for Advanced Vulkan Compute demos..."

detect_package_manager() {
    if command -v apt-get &> /dev/null; then echo "apt"
    elif command -v dnf &> /dev/null; then echo "dnf"
    elif command -v pacman &> /dev/null; then echo "pacman"
    else echo "unknown"
    fi
}

PM=$(detect_package_manager)

case $PM in
    apt)
        echo "Detected Ubuntu/Debian"
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            ninja-build \
            clang \
            libglfw3-dev \
            libglm-dev \
            libstb-dev \
            libxxf86vm-dev \
            libxi-dev \
            opencl-headers \
            ocl-icd-opencl-dev \
            clinfo
        ;;
    dnf)
        echo "Detected Fedora/RHEL"
        sudo dnf install -y \
            gcc-c++ \
            cmake \
            ninja-build \
            clang \
            glfw-devel \
            glm-devel \
            libXxf86vm-devel \
            libXi-devel \
            opencl-headers \
            ocl-icd-devel \
            clinfo
        # stb is header-only; install manually if not available
        if ! rpm -q stb-devel &>/dev/null; then
            echo "stb not in dnf — installing headers manually..."
            sudo mkdir -p /usr/local/include
            curl -fsSL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h \
                -o /tmp/stb_image.h
            sudo cp /tmp/stb_image.h /usr/local/include/
        fi
        ;;
    pacman)
        echo "Detected Arch Linux"
        sudo pacman -S --needed --noconfirm \
            base-devel \
            cmake \
            ninja \
            clang \
            glfw-x11 \
            glm \
            stb \
            opencl-headers \
            ocl-icd \
            clinfo
        ;;
    *)
        echo "Unsupported package manager. Install manually:"
        echo "  cmake >= 3.29, ninja, clang, libglfw3-dev, libglm-dev, libstb-dev"
        exit 1
        ;;
esac

if [ "$SKIP_CLSPV_CLVK" = "1" ]; then
    echo ""
    echo "======================================================="
    echo "  Skipping clspv + clvk build (--skip-clspv-clvk)"
    echo "  Chapter 05 will build with its AOT/runtime paths disabled."
    echo "======================================================="
else
# ---------------------------------------------------------------------------
# Chapter 05 (REQUIRED to run): build clspv (OpenCL C -> SPIR-V) and clvk
# (OpenCL 3.0 on Vulkan). The Chapter 05 sample cannot run without these.
# ---------------------------------------------------------------------------
echo ""
echo "======================================================="
echo "  Building clspv + clvk into $TOOLS_DIR  (REQUIRED)"
echo "  clspv pulls in LLVM/Clang — first build can take 20-40 min"
echo "======================================================="
mkdir -p "$TOOLS_DIR"
NPROC=$(nproc 2>/dev/null || echo 4)

# --- clspv -----------------------------------------------------------------
if [ ! -x "$TOOLS_DIR/clspv/build/bin/clspv" ]; then
    if [ ! -d "$TOOLS_DIR/clspv" ]; then
        git clone --depth 1 https://github.com/google/clspv.git "$TOOLS_DIR/clspv"
    fi
    # clspv fetches its own pinned LLVM/Clang/SPIRV deps.
    python3 "$TOOLS_DIR/clspv/utils/fetch_sources.py" --shallow || \
        python3 "$TOOLS_DIR/clspv/utils/fetch_sources.py"
    cmake -S "$TOOLS_DIR/clspv" -B "$TOOLS_DIR/clspv/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    ninja -C "$TOOLS_DIR/clspv/build" -j "$NPROC" clspv
fi

# --- clvk ------------------------------------------------------------------
if [ ! -f "$TOOLS_DIR/clvk/build/libOpenCL.so" ]; then
    if [ ! -d "$TOOLS_DIR/clvk" ]; then
        git clone https://github.com/kpet/clvk.git "$TOOLS_DIR/clvk"
    fi
    git -C "$TOOLS_DIR/clvk" submodule update --init --recursive
    "$TOOLS_DIR/clvk/external/clspv/utils/fetch_sources.py" --shallow || \
        "$TOOLS_DIR/clvk/external/clspv/utils/fetch_sources.py" || true
    # CMAKE_CXX_EXTENSIONS=OFF avoids a PCH gnu++17-vs-c++17 mismatch in the
    # bundled SPIRV-LLVM-Translator (otherwise the build fails to compile).
    cmake -S "$TOOLS_DIR/clvk" -B "$TOOLS_DIR/clvk/build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_EXTENSIONS=OFF
    ninja -C "$TOOLS_DIR/clvk/build" -j "$NPROC"
fi

# --- register clvk as an OpenCL ICD so clGetPlatformIDs finds it -----------
CLVK_LIB=$(find "$TOOLS_DIR/clvk/build" -name 'libOpenCL.so*' | head -1)
if [ -n "$CLVK_LIB" ]; then
    sudo mkdir -p /etc/OpenCL/vendors
    echo "$CLVK_LIB" | sudo tee /etc/OpenCL/vendors/clvk.icd > /dev/null
    echo "Registered clvk ICD: $CLVK_LIB"
fi

echo ""
echo "clspv: $TOOLS_DIR/clspv/build/bin/clspv"
echo "clvk : $CLVK_LIB"
echo ""
echo "Add clspv to PATH so CMake finds it (add this to your shell profile):"
echo "  export PATH=\"$TOOLS_DIR/clspv/build/bin:\$PATH\""
echo "Verify the layered OpenCL platform is visible:"
echo "  clinfo -l        # should list a 'clvk' platform"
echo ""
fi
echo "======================================================="
echo "  Vulkan SDK (with slangc) must be installed separately"
echo "======================================================="
echo "Download from: https://vulkan.lunarg.com/"
echo ""
echo "Quick install:"
echo "  VULKAN_VERSION=\$(curl -s https://vulkan.lunarg.com/sdk/latest/linux.txt)"
echo "  curl -O https://sdk.lunarg.com/sdk/download/\$VULKAN_VERSION/linux/vulkansdk-linux-x86_64-\$VULKAN_VERSION.tar.xz"
echo "  mkdir -p ~/vulkansdk && tar -xJf vulkansdk-linux-x86_64-\$VULKAN_VERSION.tar.xz -C ~/vulkansdk"
echo "  source ~/vulkansdk/\$VULKAN_VERSION/setup-env.sh"
echo ""
echo "Verify slangc is available after SDK install:"
echo "  slangc --version"
echo ""
echo "Build the compute demos:"
echo "  cd attachments/compute"
echo "  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build build --parallel"
echo ""
echo "Dependencies installed successfully."
