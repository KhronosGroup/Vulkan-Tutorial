#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert NNEF model to ONNX")
    parser.add_argument("--input", "-i", required=True, help="Input NNEF model directory")
    parser.add_argument("--output", "-o", required=True, help="Output ONNX model file")
    
    args = parser.parse_args()
    
    # Determine the project root to find the venv
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if platform_system() == "Windows":
        python_exe = project_root / "venv" / "Scripts" / "python.exe"
    else:
        python_exe = project_root / "venv" / "bin" / "python3"

    if not python_exe.exists():
        print(f"Error: Python executable not found at {python_exe}")
        print("Please run ./setup_env.sh first.")
        sys.exit(1)

    print(f"Converting NNEF model from {args.input} to {args.output}...")
    
    # Use nnef_tools.convert module
    cmd = [
        str(python_exe), "-m", "nnef_tools.convert",
        "--input-model", args.input,
        "--output-model", args.output,
        "--input-format", "nnef",
        "--output-format", "onnx"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Conversion completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed with error: {e}")
        sys.exit(1)

def platform_system():
    import platform
    return platform.system()

if __name__ == "__main__":
    main()
