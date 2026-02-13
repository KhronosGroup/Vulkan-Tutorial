import numpy as np
import onnxruntime as ort
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Compare ONNX Runtime and IREE outputs")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Path to input .npz or .npy")
    parser.add_argument("--iree-out", required=True, help="Path to IREE output .npy")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for comparison")
    args = parser.parse_args()

    if not os.path.exists(args.iree_out):
        print(f"Error: IREE output file not found: {args.iree_out}")
        sys.exit(1)

    # Load input
    try:
        if args.input.endswith(".npz"):
            data = np.load(args.input)
            if 'input' in data:
                x = data['input']
            else:
                x = data[data.files[0]]
        else:
            x = np.load(args.input)
    except Exception as e:
        print(f"Error loading input {args.input}: {e}")
        sys.exit(1)

    # Run ONNX Runtime
    try:
        session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        ort_outs = session.run(None, {input_name: x})
        ort_out = ort_outs[0]
    except Exception as e:
        print(f"Error running ONNX Runtime: {e}")
        sys.exit(1)

    # Load IREE output
    try:
        iree_out = np.load(args.iree_out)
    except Exception as e:
        print(f"Error loading IREE output {args.iree_out}: {e}")
        sys.exit(1)

    # Shape check
    if ort_out.shape != iree_out.shape:
        print(f"✗ Shape mismatch: ORT {ort_out.shape} vs IREE {iree_out.shape}")
        # Try to reshape if possible (sometimes IREE flattens or has extra dims)
        if ort_out.size == iree_out.size:
            print(f"Attempting to reshape IREE output to {ort_out.shape}")
            iree_out = iree_out.reshape(ort_out.shape)
        else:
            sys.exit(1)

    # Compare
    print(f"Comparing outputs (tolerance={args.tolerance})...")
    mae = np.mean(np.abs(ort_out - iree_out))
    max_diff = np.max(np.abs(ort_out - iree_out))
    
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Max Difference: {max_diff:.6f}")

    if max_diff < args.tolerance:
        print("✓ SUCCESS: Outputs match within tolerance")
        sys.exit(0)
    else:
        print("✗ FAILURE: Outputs differ too much")
        sys.exit(1)

if __name__ == "__main__":
    main()
