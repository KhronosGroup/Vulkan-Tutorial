#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import platform
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to NNEF")
    parser.add_argument("--input", "-i", required=True, help="Input ONNX model file")
    parser.add_argument("--output", "-o", required=True, help="Output NNEF model directory")
    
    args = parser.parse_args()
    
    # Determine the project root to find the venv
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if platform.system() == "Windows":
        python_exe = project_root / "venv" / "Scripts" / "python.exe"
    else:
        python_exe = project_root / "venv" / "bin" / "python3"

    if not python_exe.exists():
        print(f"Error: Python executable not found at {python_exe}")
        print("Please run ./setup_env.sh first.")
        sys.exit(1)

    print(f"Converting ONNX model from {args.input} to {args.output}...")
    
    # Use nnef_tools.convert module
    # We often need to specify input shapes for NNEF conversion
    cmd = [
        str(python_exe), "-m", "nnef_tools.convert",
        "--input-model", args.input,
        "--output-model", args.output,
        "--input-format", "onnx",
        "--output-format", "nnef"
    ]
    
    # Special case for MNIST/MobileNet model if we know it has dynamic axes
    if "mnist" in args.input.lower():
        cmd.extend(["--input-shapes", "{'input': (1, 1, 28, 28)}"])
    elif "mobilenet" in args.input.lower():
        cmd.extend(["--input-shapes", "{'input': (1, 3, 224, 224)}", "--fold-constants"])

    try:
        subprocess.run(cmd, check=True)
        print("✓ Conversion completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
