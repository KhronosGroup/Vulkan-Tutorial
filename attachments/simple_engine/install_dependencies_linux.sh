#!/bin/bash

# Install script for Simple Game Engine dependencies on Linux
# This script installs all required dependencies for building the Simple Game Engine

set -e  # Exit on any error

echo "Installing Simple Game Engine dependencies for Linux..."

# Detect the Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
    OS=Debian
    VER=$(cat /etc/debian_version)
else
    OS=$(uname -s)
    VER=$(uname -r)
fi

echo "Detected OS: $OS $VER"

# Function to install dependencies on Ubuntu/Debian
require_vulkan_headers() {
	if [ -n "${VULKAN_SDK:-}" ] && [ -f "${VULKAN_SDK}/include/vulkan/vulkan.h" ]; then
		return 0
	fi
	if [ -f "/usr/include/vulkan/vulkan.h" ]; then
		return 0
	fi
	echo ""
	echo "Vulkan SDK (or Vulkan development headers) not found."
	echo "Install the Vulkan SDK from LunarG, then re-run this script."
	echo "https://vulkan.lunarg.com/"
	exit 1
}

build_and_install_tinygltf() {
	if [ -f "/usr/local/lib/cmake/tinygltf/tinygltfConfig.cmake" ] || [ -f "/usr/lib/cmake/tinygltf/tinygltfConfig.cmake" ]; then
		return 0
	fi
	local workRoot="$1"
	echo "Installing tinygltf (from source)..."
	rm -rf "${workRoot}/tinygltf" "${workRoot}/tinygltf-build"
	git clone --depth 1 https://github.com/syoyo/tinygltf.git "${workRoot}/tinygltf"
	cmake -S "${workRoot}/tinygltf" -B "${workRoot}/tinygltf-build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DTINYGLTF_BUILD_LOADER_EXAMPLE=OFF \
		-DTINYGLTF_BUILD_GL_EXAMPLES=OFF \
		-DTINYGLTF_BUILD_STB_IMAGE=ON
	cmake --build "${workRoot}/tinygltf-build" --parallel
	sudo cmake --install "${workRoot}/tinygltf-build"
}

build_and_install_ktx() {
	if [ -f "/usr/local/lib/cmake/KTX/KTXConfig.cmake" ] || [ -f "/usr/lib/cmake/KTX/KTXConfig.cmake" ]; then
		return 0
	fi
	require_vulkan_headers
	local workRoot="$1"
	echo "Installing KTX-Software (from source)..."
	rm -rf "${workRoot}/KTX-Software" "${workRoot}/KTX-Software-build"
	git clone --depth 1 --branch v4.3.2 https://github.com/KhronosGroup/KTX-Software.git "${workRoot}/KTX-Software"
	cmake -S "${workRoot}/KTX-Software" -B "${workRoot}/KTX-Software-build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DKTX_FEATURE_TESTS=OFF \
		-DKTX_FEATURE_TOOLS=OFF \
		-DKTX_FEATURE_VULKAN=ON
	cmake --build "${workRoot}/KTX-Software-build" --parallel
	sudo cmake --install "${workRoot}/KTX-Software-build"
}

install_ubuntu_debian() {
	echo "Installing dependencies for Ubuntu/Debian..."

	sudo apt-get update

	sudo apt-get install -y \
		build-essential \
		cmake \
		git \
		ninja-build \
		pkg-config \
		ca-certificates \
		curl \
		zip \
		unzip \
		tar \
		libglfw3-dev \
		libglm-dev \
		libopenal-dev \
		nlohmann-json3-dev \
		libx11-dev \
		libxrandr-dev \
		libxinerama-dev \
		libxcursor-dev \
		libxi-dev \
		zlib1g-dev \
		libpng-dev \
		libzstd-dev

	local workRoot
	workRoot="${HOME}/.cache/simple_engine_deps"
	mkdir -p "${workRoot}"

	build_and_install_tinygltf "${workRoot}"
	build_and_install_ktx "${workRoot}"

	echo ""
	echo "Note: This script does not install Vulkan or slangc."
	echo "Install the Vulkan SDK from LunarG (it includes slangc) if you need shader compilation."
}

# Function to install dependencies on Fedora/RHEL/CentOS
install_fedora_rhel() {
    echo "Installing dependencies for Fedora/RHEL/CentOS..."

	sudo dnf install -y \
		gcc \
		gcc-c++ \
		cmake \
		git \
		ninja-build \
		pkgconf-pkg-config \
		glfw-devel \
		glm-devel \
		openal-soft-devel \
		nlohmann-json-devel \
		zlib-devel \
		libpng-devel \
		zstd-devel

	local workRoot
	workRoot="${HOME}/.cache/simple_engine_deps"
	mkdir -p "${workRoot}"

	build_and_install_tinygltf "${workRoot}"
	build_and_install_ktx "${workRoot}"

	echo ""
	echo "Note: This script does not install Vulkan or slangc."
}

# Function to install dependencies on Arch Linux
install_arch() {
    echo "Installing dependencies for Arch Linux..."

	sudo pacman -Sy


	sudo pacman -S --noconfirm \
		base-devel \
		cmake \
		git \
		ninja \
		pkgconf \
		glfw \
		glm \
		openal \
		nlohmann-json \
		zlib \
		libpng \
		zstd

	local workRoot
	workRoot="${HOME}/.cache/simple_engine_deps"
	mkdir -p "${workRoot}"

	build_and_install_tinygltf "${workRoot}"
	build_and_install_ktx "${workRoot}"

	echo ""
	echo "Note: This script does not install Vulkan or slangc."
}

# Install dependencies based on detected OS
case "$OS" in
    "Ubuntu"* | "Debian"* | "Linux Mint"*)
        install_ubuntu_debian
        ;;
    "Fedora"* | "Red Hat"* | "CentOS"* | "Rocky"*)
        install_fedora_rhel
        ;;
    "Arch"* | "Manjaro"*)
        install_arch
        ;;
    *)
        echo "Unsupported Linux distribution: $OS"
        echo "Please install the following dependencies manually:"
        echo "- CMake (3.29 or later)"
        echo "- Vulkan SDK"
        echo "- GLFW3 development libraries"
        echo "- GLM (OpenGL Mathematics) library"
        echo "- OpenAL development libraries"
        echo "- KTX library"
        echo "- STB library"
        echo "- tinygltf library"
        echo "- Slang compiler"
        exit 1
        ;;
esac

echo ""
echo "Dependencies installation completed!"
echo ""
echo "To build the Simple Game Engine:"
echo "1. cd to the simple_engine directory"
echo "2. cmake -S . -B build -G Ninja"
echo "3. cmake --build build --target SimpleEngine --parallel $(nproc)"
echo ""
echo "If you have Vulkan SDK installed, shader compilation uses slangc from the SDK automatically."
