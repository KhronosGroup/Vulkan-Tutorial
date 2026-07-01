#!/usr/bin/env python3
"""
Download MobileNetV2 model from ONNX Model Zoo for image classification demo.
"""

import urllib.request
import os
import sys

def download_mobilenetv2():
    """Download MobileNetV2 from ONNX Model Zoo"""

    model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    model_path = "models/mobilenetv2.onnx"

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Check if model already exists
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model already exists at {model_path} ({size_mb:.2f} MB)")
        return True

    print(f"Downloading MobileNetV2 from ONNX Model Zoo...")
    print(f"URL: {model_url}")
    print(f"Destination: {model_path}")

    try:
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(model_url, model_path, reporthook=reporthook)
        print()  # New line after progress

        # Verify download
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model downloaded successfully!")
        print(f"  Size: {size_mb:.2f} MB")

        # Expected size is around 13-14 MB
        if size_mb < 10 or size_mb > 20:
            print(f"⚠ Warning: Model size {size_mb:.2f} MB seems unusual (expected ~14 MB)")
            return False

        return True

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return False

def download_imagenet_labels():
    """Download ImageNet class labels"""

    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels_path = "data/imagenet_classes.txt"

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Check if labels already exist
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            num_lines = len(f.readlines())
        print(f"✓ ImageNet labels already exist at {labels_path} ({num_lines} classes)")
        return True

    print(f"Downloading ImageNet class labels...")
    print(f"URL: {labels_url}")
    print(f"Destination: {labels_path}")

    try:
        urllib.request.urlretrieve(labels_url, labels_path)

        # Verify download
        with open(labels_path, 'r') as f:
            lines = f.readlines()
            num_classes = len(lines)

        print(f"✓ Labels downloaded successfully!")
        print(f"  Classes: {num_classes}")

        if num_classes != 1000:
            print(f"⚠ Warning: Expected 1000 classes, got {num_classes}")
            return False

        return True

    except Exception as e:
        print(f"\n✗ Error downloading labels: {e}")
        if os.path.exists(labels_path):
            os.remove(labels_path)
        return False

def main():
    print("=" * 60)
    print("MobileNetV2 Image Classification Demo - Model Download")
    print("=" * 60)
    print()

    success = True

    # Download model
    if not download_mobilenetv2():
        success = False
    print()

    # Download labels
    if not download_imagenet_labels():
        success = False
    print()

    if success:
        print("=" * 60)
        print("✓ All downloads completed successfully!")
        print("=" * 60)
        print()
        print("You can now build and run the image classification demo:")
        print("  mkdir build && cd build")
        print("  cmake ..")
        print("  cmake --build .")
        print("  ./image_classifier")
        return 0
    else:
        print("=" * 60)
        print("✗ Some downloads failed. Please check the errors above.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
