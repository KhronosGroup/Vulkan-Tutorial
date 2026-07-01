#!/bin/bash
set -e

# Default values
OUTPUT="model.vmfb"
TARGET_BACKEND="vulkan-spirv"

# Usage
usage() {
    echo "Usage: $0 <input_model> [-o <output_vmfb>] [--target <backend>]"
    echo "Input model can be .onnx, .mlir, or .stablehlo"
    echo "Backends: vulkan-spirv (default), llvm-cpu"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

INPUT_MODEL=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            OUTPUT="$2"
            shift 2
            ;;
        --target)
            TARGET_BACKEND="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
VENV_BIN="$PROJECT_ROOT/venv/bin"
IREE_COMPILE="$VENV_BIN/iree-compile"
IREE_IMPORT_ONNX="$VENV_BIN/iree-import-onnx"

if [ ! -f "$IREE_COMPILE" ]; then
    echo "Error: iree-compile not found at $IREE_COMPILE"
    echo "Please run ./setup_env.sh --with-optional first."
    exit 1
fi

TEMP_MLIR=$(mktemp /tmp/iree_compile_XXXXXX.mlir)

echo "Compiling $INPUT_MODEL to $OUTPUT for $TARGET_BACKEND..."

if [[ "$INPUT_MODEL" == *.onnx ]]; then
    if [ ! -f "$IREE_IMPORT_ONNX" ]; then
        echo "Error: iree-import-onnx not found. Ensure iree-compiler is installed with ONNX support."
        exit 1
    fi
    echo "Importing ONNX to MLIR..."
    "$IREE_IMPORT_ONNX" "$INPUT_MODEL" -o "$TEMP_MLIR"
    INPUT_TO_COMPILE="$TEMP_MLIR"
else
    INPUT_TO_COMPILE="$INPUT_MODEL"
fi

"$IREE_COMPILE" \
    --iree-hal-target-backends=$TARGET_BACKEND \
    "$INPUT_TO_COMPILE" \
    -o "$OUTPUT"

# Cleanup
if [ -f "$TEMP_MLIR" ]; then
    rm "$TEMP_MLIR"
fi

echo "Done: $OUTPUT"
