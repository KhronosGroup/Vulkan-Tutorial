#!/bin/bash
# Raspberry Pi Setup Script for ML Inference Wildlife Cam
# This script installs system dependencies and triggers the main project setup.

set -e

echo "============================================================"
echo "Raspberry Pi ML Inference - Environment Setup"
echo "============================================================"

# Check if running on Raspberry Pi or ARM
IS_RPI=false
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    IS_RPI=true
    echo "Confirmed: Running on Raspberry Pi."
elif uname -m | grep -q "aarch64\|arm"; then
    echo "Running on ARM architecture (compatible environment)."
else
    echo "Warning: This does not look like a Raspberry Pi or ARM environment."
    echo "If you are on a desktop and want to simulate rPi, use -DFORCE_RPI_BUILD=ON during CMake."
fi

# Install system dependencies
echo "Installing system dependencies via apt..."
sudo apt-get update
sudo apt-get install -y \
    cmake \
    g++ \
    libvulkan-dev \
    vulkan-tools \
    libglfw3-dev \
    libopencv-dev \
    python3-pip \
    python3-venv \
    wget \
    tar \
    xz-utils

# Verify Vulkan support
if [ "$IS_RPI" = true ]; then
    echo "Checking for Raspberry Pi Vulkan drivers (Mesa)..."
    sudo apt-get install -y mesa-vulkan-drivers
fi

# Go to parent directory to run main setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo ""
echo "Triggering main project setup (will download models and runtimes)..."
./setup_env.sh --with-optional

echo ""
echo "============================================================"
echo "Raspberry Pi Setup Complete!"
echo "============================================================"
echo ""
echo "To build the wildlife cam app:"
echo "  1. mkdir build && cd build"
echo "  2. cmake -DML_BACKEND=NNEF ..  (or -DML_BACKEND=IREE, -DML_BACKEND=TFLITE)"
echo "  3. cmake --build . --target embedded_wildlife_cam_app"
echo ""
echo "Run the application (NNEF example):"
echo "  ./rPi/embedded_wildlife_cam_app --model models/mobilenetv2_nnef_optimized --labels data/imagenet_classes.txt"
