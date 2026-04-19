#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SIMPLE_ENGINE_DIR="$SCRIPT_DIR/../simple_engine"

if [ ! -d "$SIMPLE_ENGINE_DIR" ]; then
    echo "Error: simple_engine directory not found at $SIMPLE_ENGINE_DIR"
    exit 1
fi

# Run simple engine dependencies first
# We pass the work root to avoid conflicts if needed, but the script uses a fixed one anyway.
bash "$SIMPLE_ENGINE_DIR/install_dependencies_linux.sh"

# Install OpenXR
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
fi

echo "Installing OpenXR dependencies for $OS..."

case "$OS" in
    "Ubuntu"* | "Debian"* | "Linux Mint"*)
        sudo apt-get update
        sudo apt-get install -y libopenxr-dev
        ;;
    "Fedora"* | "Red Hat"* | "CentOS"* | "Rocky"*)
        sudo dnf install -y openxr-devel
        ;;
    "Arch"* | "Manjaro"*)
        sudo pacman -S --noconfirm openxr
        ;;
    *)
        echo "Warning: Unsupported OS for automatic OpenXR installation. Please install OpenXR SDK manually."
        ;;
esac

echo "OpenXR dependencies installation completed!"
