#!/usr/bin/env python3
import torch
import torchvision
from pathlib import Path
import os

def export_mobilenet_v2():
    print("Exporting MobileNetV2 to IREE-compatible ONNX (Opset 18)...")
    
    # Load model
    model = torchvision.models.mobilenet_v2(weights='DEFAULT')
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Path
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / "mobilenetv2_iree.onnx"
    
    # Export
    # Opset 18 is known to work with the current IREE ONNX importer
    torch.onnx.export(
        model, 
        dummy_input, 
        str(output_path), 
        opset_version=18, 
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✓ Exported to {output_path}")
    return output_path

if __name__ == "__main__":
    export_mobilenet_v2()
