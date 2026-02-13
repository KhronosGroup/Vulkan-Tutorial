#!/bin/bash
# One-time setup script for ML Inference examples
# Run this once: ./setup_env.sh
# Optional: ./setup_env.sh --with-optional

set -e

echo "============================================================"
echo "ML Inference Examples - One-Time Setup"
echo "============================================================"
echo ""

# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    echo "Please install Python 3.8 or later"
    exit 1
fi

echo "Found Python: $(python3 --version)"
echo ""

# Run the Python setup script
if [ "$1" == "--with-optional" ]; then
    echo "Installing with optional dependencies (ONNX Runtime, IREE, TensorFlow Lite)..."
    python3 setup_python_env.py --with-optional
else
    echo "Installing core dependencies only..."
    echo "To include optional dependencies (ONNX Runtime, IREE), run: ./setup_env.sh --with-optional"
    echo ""
    python3 setup_python_env.py
fi

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. mkdir build && cd build"
echo "  2. cmake .."
echo "  3. cmake --build ."
echo ""
echo "Built executables:"
echo "  ./mnist         - MNIST digit recognition (drawable canvas)"
echo "  ./classification - Image classification with MobileNetV2"
echo ""
echo "CMake auto-detects ONNX Runtime and all dependencies!"
echo ""
