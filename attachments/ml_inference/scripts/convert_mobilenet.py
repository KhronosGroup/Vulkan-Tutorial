#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import platform
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert and Optimize MobileNetV2 from ONNX to NNEF and TFLite")
    parser.add_argument("--input", "-i", default="models/mobilenetv2.onnx", help="Input ONNX model file")
    parser.add_argument("--output-nnef", default="models/mobilenetv2_nnef_optimized", help="Output NNEF model directory")
    parser.add_argument("--output-tflite", default="models/mobilenetv2.tflite", help="Output TFLite model file")
    
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

    input_onnx = Path(args.input)
    if not input_onnx.exists():
        # Try parent dir
        input_onnx = project_root / args.input
        if not input_onnx.exists():
            print(f"Error: Input model {args.input} not found.")
            sys.exit(1)

    # 1. Convert to Optimized NNEF
    print(f"\n[1/2] Converting to Optimized NNEF: {args.output_nnef}...")
    nnef_cmd = [
        str(python_exe), "-m", "nnef_tools.convert",
        "--input-model", str(input_onnx),
        "--output-model", args.output_nnef,
        "--input-format", "onnx",
        "--output-format", "nnef",
        "--input-shapes", "{'input': (1, 3, 224, 224)}",
        "--fold-constants",
        "--optimize"
    ]
    
    try:
        subprocess.run(nnef_cmd, check=True)
        print("✓ NNEF conversion and optimization completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ NNEF conversion failed: {e}")

    # 2. Convert to TFLite
    print(f"\n[2/2] Converting to TFLite: {args.output_tflite}...")
    # use our patched version to handle newer ONNX
    tflite_cmd = [
        str(python_exe), "scripts/onnx2tf_patched.py",
        "-i", str(input_onnx),
        "-o", "tflite_temp",
        "--non_verbose"
    ]
    
    try:
        if os.path.exists("tflite_temp"):
            import shutil
            shutil.rmtree("tflite_temp")
            
        subprocess.run(tflite_cmd, check=True)
        
        # find the generated tflite
        tflite_files = list(Path("tflite_temp").glob("*.tflite"))
        if not tflite_files:
            # check subdirectories
            tflite_files = list(Path("tflite_temp").rglob("*.tflite"))
            
        if tflite_files:
            import shutil
            shutil.copy(str(tflite_files[0]), args.output_tflite)
            print(f"✓ TFLite conversion completed successfully: {args.output_tflite}")
            # cleanup
            shutil.rmtree("tflite_temp")
        else:
            print("! TFLite file not found in 'tflite_temp' output directory.")
                
    except subprocess.CalledProcessError as e:
        print(f"✗ TFLite conversion failed: {e}")

if __name__ == "__main__":
    main()
