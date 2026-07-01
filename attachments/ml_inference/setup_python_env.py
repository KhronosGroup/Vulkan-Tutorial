#!/usr/bin/env python3
"""
Setup Python environment for ML Inference examples
Creates a virtual environment and installs all required dependencies
Works on both Linux and Windows

Usage:
    python3 setup_python_env.py                    # Install core dependencies only
    python3 setup_python_env.py --with-optional    # Install with ONNX Runtime and TensorFlow Lite
"""

import os
import sys
import subprocess
import platform
import argparse
import urllib.request
import tarfile
from pathlib import Path

def print_header(msg):
    """Print a formatted header message"""
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60)

def check_python_version():
    """Ensure Python version is 3.8 or later"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Error: Python 3.8+ required, but found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")

def is_venv_valid(venv_path):
    """Check if virtual environment is valid (has required executables)"""
    if platform.system() == "Windows":
        pip_exe = venv_path / "Scripts" / "pip.exe"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        pip_exe = venv_path / "bin" / "pip"
        python_exe = venv_path / "bin" / "python"

    return pip_exe.exists() and python_exe.exists()

def create_venv(venv_path):
    """Create a virtual environment"""
    print_header("Python Virtual Environment")

    if venv_path.exists():
        # Check if venv is valid
        if is_venv_valid(venv_path):
            print(f"✓ Virtual environment already exists at: {venv_path}")
            print("Will verify and update packages...")
            return True
        else:
            print(f"✗ Virtual environment at {venv_path} is corrupted")
            print("Removing and recreating...")
            import shutil
            shutil.rmtree(venv_path)

    print(f"Creating virtual environment at: {venv_path}")

    # Try creating venv with pip
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        capture_output=True,
        text=True
    )

    # If it failed, try without pip (common on some Debian/Ubuntu systems without python3-venv)
    if result.returncode != 0:
        print("Standard venv creation failed, trying without pip...")
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(venv_path)],
            check=True
        )

        # Install pip manually using get-pip.py
        print("Installing pip manually...")
        import urllib.request
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_path = venv_path / "get-pip.py"

        try:
            urllib.request.urlretrieve(get_pip_url, get_pip_path)
            python_exe = get_python_executable(venv_path)
            subprocess.run([str(python_exe), str(get_pip_path)], check=True)
        finally:
            if get_pip_path.exists():
                get_pip_path.unlink()  # Clean up

    print("✓ Virtual environment created")
    return False

def get_pip_executable(venv_path):
    """Get the path to pip in the virtual environment"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"

def get_python_executable(venv_path):
    """Get the path to python in the virtual environment"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def upgrade_pip(pip_exe):
    """Upgrade pip to the latest version"""
    print("\nUpgrading pip...")
    subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
    print("✓ pip upgraded")

def check_missing_packages(python_exe, install_optional=False):
    """Check which packages are missing or need installation"""
    core_packages = ["torch", "torchvision", "onnx", "onnxscript", "numpy", "opencv-python", "lpips"]
    optional_packages = ["onnxruntime", "tensorflow", "iree-compiler", "iree-runtime", "nnef", "nnef-tools"] if install_optional else []

    missing = []
    for pkg in core_packages + optional_packages:
        # Special case for packages with different import names
        import_name = pkg
        if pkg == "iree-compiler": import_name = "iree.compiler"
        if pkg == "iree-runtime": import_name = "iree.runtime"
        
        try:
            subprocess.run(
                [str(python_exe), "-c", f"import {import_name}"],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            missing.append(pkg)

    return missing

def install_dependencies(pip_exe, python_exe, install_optional=False, venv_existed=False):
    """Install required Python packages"""
    print_header("Installing/Updating Python Dependencies")

    # Core dependencies for MNIST training
    core_packages = [
        "torch",
        "torchvision",
        "onnx",
        "onnxscript",  # Required for torch.onnx.export
        "numpy",
        "opencv-python",
        "lpips",
    ]

    # Optional dependencies for advanced ML integration
    optional_packages = [
        "onnxruntime",
        "tensorflow-lite",
        "iree-compiler",
        "iree-runtime",
        "nnef",
        "nnef-tools",
    ]

    # Check what's missing if venv existed
    if venv_existed:
        print("\nChecking existing environment...")
        missing = check_missing_packages(python_exe, install_optional)

        if not missing:
            print("✓ All required packages already installed")
            return

        print(f"Missing packages: {', '.join(missing)}")
        print("Installing missing packages only...")

    print("\nCore packages:")
    for pkg in core_packages:
        print(f"  - {pkg}")

    if install_optional:
        print("\nOptional packages:")
        for pkg in optional_packages:
            print(f"  - {pkg}")

    # Install PyTorch with CPU support (works everywhere)
    # For GPU support, users can reinstall manually
    # Only install if missing when venv existed
    if not venv_existed or "torch" in missing or "torchvision" in missing:
        print("\nInstalling PyTorch (CPU version)...")
        subprocess.run([
            str(pip_exe), "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ], check=True)

    # Install other core dependencies only if missing
    core_to_install = []
    if not venv_existed:
        core_to_install = ["onnx", "onnxscript", "numpy"]
    else:
        for pkg in ["onnx", "onnxscript", "numpy"]:
            if pkg in missing:
                core_to_install.append(pkg)

    if core_to_install:
        print(f"\nInstalling {', '.join(core_to_install)}...")
        subprocess.run([
            str(pip_exe), "install"
        ] + core_to_install, check=True)

    if install_optional:
        print("\nInstalling optional dependencies...")

        if not venv_existed or "onnxruntime" in missing:
            try:
                subprocess.run([
                    str(pip_exe), "install",
                    "onnxruntime"
                ], check=True)
                print("✓ ONNX Runtime installed")
            except subprocess.CalledProcessError:
                print("✗ Warning: ONNX Runtime installation failed (non-fatal)")

        if not venv_existed or "tensorflow" in missing:
            try:
                # TensorFlow Lite might have different package names on different platforms
                subprocess.run([
                    str(pip_exe), "install",
                    "tensorflow"  # Includes TFLite
                ], check=True)
                print("✓ TensorFlow (with TFLite) installed")
            except subprocess.CalledProcessError:
                print("✗ Warning: TensorFlow installation failed (non-fatal)")

        if not venv_existed or "iree-compiler" in missing or "iree-runtime" in missing:
            try:
                subprocess.run([
                    str(pip_exe), "install",
                    "iree-compiler", "iree-runtime"
                ], check=True)
                print("✓ IREE Compiler and Runtime (Python) installed")
            except subprocess.CalledProcessError:
                print("✗ Warning: IREE installation failed (non-fatal)")

        if not venv_existed or "nnef" in missing or "nnef-tools" in missing:
            try:
                # NNEF requires stdint.h on some modern compilers
                env = os.environ.copy()
                env["CFLAGS"] = env.get("CFLAGS", "") + " -include stdint.h"
                env["CXXFLAGS"] = env.get("CXXFLAGS", "") + " -include stdint.h"
                
                print("\nInstalling NNEF and NNEF-tools...")
                subprocess.run([
                    str(pip_exe), "install",
                    "nnef", "nnef-tools"
                ], env=env, check=True)
                print("✓ NNEF and NNEF-tools installed")
            except subprocess.CalledProcessError:
                print("✗ Warning: NNEF installation failed (non-fatal)")

    print("\n✓ All dependencies installed")

def verify_installation(python_exe, check_optional=False):
    """Verify that all packages are importable"""
    print_header("Verifying Installation")

    core_packages = ["torch", "torchvision", "onnx", "onnxscript", "numpy"]
    optional_packages = ["onnxruntime", "tensorflow"]

    # Check core packages
    print("Core packages:")
    for pkg in core_packages:
        try:
            result = subprocess.run(
                [str(python_exe), "-c", f"import {pkg}; print({pkg}.__version__)"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.strip()
            print(f"  ✓ {pkg:15s} {version}")
        except subprocess.CalledProcessError:
            print(f"  ✗ {pkg:15s} FAILED TO IMPORT")
            return False

    # Check optional packages
    if check_optional:
        print("\nOptional packages:")
        optional_packages = [
            ("onnxruntime", "onnxruntime"),
            ("tensorflow", "tensorflow"),
            ("iree-compiler", "iree.compiler"),
            ("iree-runtime", "iree.runtime")
        ]
        for pkg, import_name in optional_packages:
            try:
                result = subprocess.run(
                    [str(python_exe), "-c", f"import {import_name}; print('installed')"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"  ✓ {pkg:15s} installed")
            except subprocess.CalledProcessError:
                print(f"  ✗ {pkg:15s} not installed (optional)")

    return True

def download_onnx_runtime(script_dir):
    """Download ONNX Runtime for C++ inference"""
    print_header("ONNX Runtime C++ Library")

    version = "1.23.2"  # Latest stable release

    # Create third_party directory
    third_party_dir = script_dir / "third_party"
    third_party_dir.mkdir(exist_ok=True)

    # Cleanup old versions
    for old_dir in third_party_dir.glob("onnxruntime-*"):
        if version not in old_dir.name:
            print(f"Cleaning up old ONNX Runtime: {old_dir.name}")
            import shutil
            shutil.rmtree(old_dir)

    sys_platform = platform.system()
    arch = platform.machine()
    
    if sys_platform == "Linux":
        if arch == "aarch64":
            filename = f"onnxruntime-linux-aarch64-{version}.tgz"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/{filename}"
            extract_dir = third_party_dir / f"onnxruntime-linux-aarch64-{version}"
        else:
            # Using the GPU version as suggested by the user
            filename = f"onnxruntime-linux-x64-gpu-{version}.tgz"
            url = f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/{filename}"
            extract_dir = third_party_dir / f"onnxruntime-linux-x64-gpu-{version}"
    elif sys_platform == "Windows":
        # Using the GPU version for Windows as well to be consistent
        filename = f"onnxruntime-win-x64-gpu-{version}.zip"
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/{filename}"
        extract_dir = third_party_dir / f"onnxruntime-win-x64-gpu-{version}"
    elif sys_platform == "Darwin":
        filename = f"onnxruntime-osx-x86_64-{version}.tgz"
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/{filename}"
        extract_dir = third_party_dir / f"onnxruntime-osx-x86_64-{version}"
    else:
        print(f"✗ Unsupported platform: {sys_platform}")
        return False

    # Check if already downloaded
    if extract_dir.exists():
        print(f"✓ ONNX Runtime {version} already downloaded at {extract_dir}")
        return True

    print(f"Downloading ONNX Runtime {version} for {sys_platform}...")
    print(f"URL: {url}")

    download_path = third_party_dir / filename

    try:
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
            sys.stdout.write(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, download_path, reporthook=reporthook)
        print()  # New line after progress

        print("Extracting...")
        if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
            with tarfile.open(download_path, 'r:gz') as tar:
                tar.extractall(third_party_dir)
        elif filename.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(third_party_dir)

        # Clean up download
        download_path.unlink()

        print(f"✓ ONNX Runtime {version} extracted to {extract_dir}")
        return True

    except Exception as e:
        print(f"\n✗ Error downloading ONNX Runtime: {e}")
        if download_path.exists():
            download_path.unlink()
        return False

def download_iree_runtime(script_dir):
    """Download IREE Runtime for C++ inference"""
    print_header("IREE Runtime C++ Library")

    # Use a recent stable-ish release candidate
    version = "3.11.0rc20260211"
    
    # Create third_party directory
    third_party_dir = script_dir / "third_party"
    third_party_dir.mkdir(exist_ok=True)

    sys_platform = platform.system()
    arch = platform.machine()
    
    if sys_platform == "Linux":
        if arch == "aarch64":
            filename = f"iree-dist-{version}-linux-aarch64.tar.xz"
            url = f"https://github.com/iree-org/iree/releases/download/iree-{version}/{filename}"
            extract_dir = third_party_dir / f"iree-dist-linux-aarch64-{version}"
        else:
            filename = f"iree-dist-{version}-linux-x86_64.tar.xz"
            url = f"https://github.com/iree-org/iree/releases/download/iree-{version}/{filename}"
            extract_dir = third_party_dir / f"iree-dist-linux-x86_64-{version}"
    else:
        # Skip IREE on non-Linux for now as per plan focus on Desktop/Embedded Linux
        print(f"! IREE C++ runtime download skipped for {sys_platform} (Linux focused)")
        return True

    # Check if already downloaded
    if extract_dir.exists():
        print(f"✓ IREE Runtime {version} already downloaded at {extract_dir}")
        return True

    print(f"Downloading IREE Runtime {version} for {sys_platform}...")
    print(f"URL: {url}")

    download_path = third_party_dir / filename

    try:
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
            sys.stdout.write(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, download_path, reporthook=reporthook)
        print()  # New line after progress

        print("Extracting...")
        # Note: tarfile supports .xz if lzma is available in Python
        with tarfile.open(download_path, 'r:xz') as tar:
            # We want to extract it into extract_dir
            extract_dir.mkdir(exist_ok=True)
            tar.extractall(extract_dir)

        # Clean up download
        download_path.unlink()

        print(f"✓ IREE Runtime {version} extracted to {extract_dir}")
        
        # Patch missing headers (common in some rc distributions)
        patch_iree_headers(extract_dir)
        
        return True

    except Exception as e:
        print(f"\n✗ Error downloading IREE Runtime: {e}")
        if download_path.exists():
            download_path.unlink()
        return False

def patch_iree_headers(iree_dist_dir):
    """Fetch missing headers directly from GitHub if they are not in the distribution"""
    print("Checking for missing IREE headers...")
    
    base_include = iree_dist_dir / "include" / "iree" / "base"
    hal_include = iree_dist_dir / "include" / "iree" / "hal"
    
    missing_files = {
        base_include: [
            "allocator.h", "status.h", "status_cc.h", "assert.h", "attributes.h", 
            "config.h", "target_platform.h", "wait_handle.h"
        ],
        hal_include: [
            "allocator.h", "buffer.h", "resource_set.h", "driver.h", "semaphore.h",
            "command_buffer.h", "event.h", "executable.h", "executable_layout.h",
            "executable_loader.h", "queue.h", "fence.h", "descriptor_set_layout.h"
        ]
    }
    
    repo_url = "https://raw.githubusercontent.com/iree-org/iree/main/runtime/src/iree"
    
    for folder, files in missing_files.items():
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = folder / f
            if not target.exists():
                # We need to determine the relative path for the URL
                rel_path = folder.relative_to(iree_dist_dir / "include" / "iree")
                url = f"{repo_url}/{rel_path}/{f}"
                print(f"  Downloading missing header: {f}...")
                try:
                    urllib.request.urlretrieve(url, target)
                except Exception as e:
                    print(f"  ! Failed to download {f}: {e}")

    # Special case: Vulkan headers might be expected in a specific subfolder
    vulkan_internal = hal_include / "drivers" / "vulkan"
    if vulkan_internal.exists():
        vulkan_headers_path = iree_dist_dir / "include" / "third_party" / "vulkan_headers" / "include" / "vulkan"
        vulkan_headers_path.mkdir(parents=True, exist_ok=True)
        # Link or copy system vulkan headers if possible, or just download a few key ones
        # For this sample, we'll try to download vulkan.h and relevant ones if missing
        v_headers = ["vulkan.h", "vulkan_core.h", "vk_platform.h"]
        v_url = "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Headers/main/include/vulkan"
        for vh in v_headers:
            v_target = vulkan_headers_path / vh
            if not v_target.exists():
                print(f"  Downloading missing Vulkan header: {vh}...")
                try:
                    urllib.request.urlretrieve(f"{v_url}/{vh}", v_target)
                except:
                    pass

def setup_webgpu_environment(script_dir):
    """Attempt to set up or verify WebGPU (Dawn) executable environment"""
    print_header("WebGPU (Dawn) Executable Environment")
    print("Attempting to verify WebGPU (Dawn) executable environment...")

    # 1. Check for Vulkan (often a prerequisite for WebGPU)
    vulkan_ok = False
    try:
        if platform.system() == "Linux":
            res = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
            if res.returncode == 0:
                print("✓ Vulkan SDK/Runtime detected")
                vulkan_ok = True
        elif platform.system() == "Windows":
            res = subprocess.run(["where", "vulkan-1.dll"], capture_output=True, text=True)
            if res.returncode == 0:
                print("✓ Vulkan Runtime detected")
                vulkan_ok = True
    except:
        pass

    if not vulkan_ok:
        print("! Warning: Vulkan not detected. WebGPU requires Vulkan 1.3+ on most desktop platforms.")

    # 2. Check for Slang compiler (can emit WebGPU/WGSL)
    try:
        res = subprocess.run(["slangc", "-version"], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"✓ Slang compiler detected: {res.stdout.strip()}")
    except:
        print("! Warning: slangc not found. Shader compilation might fail.")

    # 3. WebGPU specific note
    print("\nNote: Native WebGPU support in ONNX Runtime is enabled in this build.")
    print("If execution fails, ensure you have the latest GPU drivers and 'Dawn' libraries.")
    print("WebGPU requires Vulkan 1.3+ on Linux and Windows.")
    
    # 3. TensorFlow Lite C API
    print_header("TensorFlow Lite C API")
    
    tflite_version = "2.14.0"
    tflite_dir = script_dir / "third_party" / f"tensorflow-lite-{tflite_version}"
    
    if tflite_dir.exists():
        print(f"✓ TFLite C API {tflite_version} already setup")
    else:
        print(f"Setting up TFLite C API {tflite_version}...")
        tflite_dir.mkdir(parents=True, exist_ok=True)
        
        # We download headers from LiteRT (the successor) or TF repo
        headers = [
            "c_api.h", "c_api_opaque.h", "c_api_types.h", "common.h"
        ]
        base_url = "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.14.0/tensorflow/lite/c"
        
        include_dir = tflite_dir / "include" / "tensorflow" / "lite" / "c"
        include_dir.mkdir(parents=True, exist_ok=True)
        
        for h in headers:
            try:
                urllib.request.urlretrieve(f"{base_url}/{h}", include_dir / h)
                print(f"  Downloaded header: {h}")
            except:
                print(f"  ! Failed to download {h}")
        
        # Link or copy the .so from our ai-edge-litert package if available
        # OR just use a placeholder for now if it's too complex to find
        print("  Looking for TFLite/LiteRT library in venv...")
        lib_found = False
        if platform.system() == "Linux":
            # Check LiteRT
            litert_lib = script_dir / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "ai_edge_litert" / "libLiteRt.so"
            if litert_lib.exists():
                lib_dir = tflite_dir / "lib"
                lib_dir.mkdir(exist_ok=True)
                import shutil
                shutil.copy(litert_lib, lib_dir / "libtensorflowlite_c.so") # Rename for consistency
                print(f"  ✓ Found LiteRT library and staged as libtensorflowlite_c.so")
                lib_found = True
        
        if not lib_found:
            print("  ! TFLite library not found in venv. TFLite backend might be disabled in CMake.")

    return True

def download_classification_model(script_dir):
    """Download MobileNetV2 model and ImageNet labels"""
    print_header("Image Classification Model")

    # Download model
    model_dir = script_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "mobilenetv2-12.onnx"

    if model_path.exists():
        print(f"✓ MobileNetV2 model already downloaded")
    else:
        print("Downloading MobileNetV2 model from ONNX Model Zoo...")
        model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"

        try:
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(model_url, model_path, reporthook=reporthook)
            print()

            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ Model downloaded ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"\n✗ Error downloading model: {e}")
            if model_path.exists():
                model_path.unlink()
            return False

    # Download labels
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)
    labels_path = data_dir / "imagenet_classes.txt"

    if labels_path.exists():
        print(f"✓ ImageNet labels already downloaded")
    else:
        print("Downloading ImageNet class labels...")
        labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

        try:
            urllib.request.urlretrieve(labels_url, labels_path)
            with open(labels_path, 'r') as f:
                num_classes = len(f.readlines())
            print(f"✓ Labels downloaded ({num_classes} classes)")
        except Exception as e:
            print(f"\n✗ Error downloading labels: {e}")
            if labels_path.exists():
                labels_path.unlink()
            return False

    return True

def create_activation_scripts(venv_path, script_dir):
    """Note about CMake usage (no activation scripts needed)"""
    print_header("Environment Ready for CMake")

    print("\nNOTE: You do NOT need to activate the virtual environment manually.")
    print("CMake will automatically use the correct Python interpreter.")
    print(f"\nVirtual environment location: {venv_path.absolute()}")

    if platform.system() == "Windows":
        venv_python = venv_path / "Scripts" / "python.exe"
    else:
        venv_python = venv_path / "bin" / "python"

    print(f"Python executable: {venv_python.absolute()}")
    print("\nCMake targets like 'train_mnist' will use this Python automatically.")

def print_next_steps(venv_path, has_optional=False):
    """Print instructions for next steps"""
    print_header("Setup Complete!")

    print("\nYour Python environment is ready for ML Inference examples.")

    print("\n" + "=" * 60)
    print("NEXT STEPS - Work from your IDE:")
    print("=" * 60)

    print("\n1. Open the project in your IDE:")
    print("   - CLion: File → Open → Select ml_inference folder")
    print("   - VS Code: File → Open Folder → Select ml_inference folder")
    print("   - Visual Studio: Open the CMakeLists.txt")

    print("\n2. Configure and build (IDE does this automatically)")

    print("\n3. Train the MNIST model:")
    print("   - In IDE: Build the 'train_mnist' target")
    print("   - Or from command line: make train_mnist")

    print("\n4. Run the MNIST application:")
    print("   - In IDE: Run the 'mnist_inference' target")
    print("   - Or from command line: ./build/mnist_inference")

    if has_optional:
        print("\n" + "=" * 60)
        print("Optional dependencies installed:")
        print("  ✓ ONNX Runtime - For ONNX model integration examples")
        print("  ✓ TensorFlow Lite - For TFLite integration examples")

    print("\n" + "=" * 60)
    print("IMPORTANT: You do NOT need to activate any virtual environment!")
    print("CMake automatically uses the correct Python interpreter.")
    print("=" * 60)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Setup Python environment for ML Inference examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 setup_python_env.py                    # Install core dependencies only
  python3 setup_python_env.py --with-optional    # Install with ONNX Runtime and TensorFlow Lite
        """
    )
    parser.add_argument(
        "--with-optional",
        action="store_true",
        help="Install optional dependencies (ONNX Runtime, TensorFlow Lite)"
    )
    return parser.parse_args()

def main():
    """Main setup function"""
    args = parse_args()

    print_header("ML Inference Python Environment Setup")
    print(f"Platform: {platform.system()} {platform.machine()}")
    if args.with_optional:
        print("Mode: Installing with optional dependencies")
    else:
        print("Mode: Installing core dependencies only")

    # Check Python version
    check_python_version()

    # Determine paths
    script_dir = Path(__file__).parent
    venv_path = script_dir / "venv"

    try:
        # Create virtual environment (returns True if already existed)
        venv_existed = create_venv(venv_path)

        # Get executables
        pip_exe = get_pip_executable(venv_path)
        python_exe = get_python_executable(venv_path)

        # Upgrade pip
        upgrade_pip(pip_exe)

        # Install dependencies (will check and only install missing if venv existed)
        install_dependencies(pip_exe, python_exe, install_optional=args.with_optional, venv_existed=venv_existed)

        # Verify installation
        if not verify_installation(python_exe, check_optional=args.with_optional):
            print("\n✗ Installation verification failed!")
            sys.exit(1)

        # Download ONNX Runtime C++ library
        if not download_onnx_runtime(script_dir):
            print("\n✗ Failed to download ONNX Runtime")
            sys.exit(1)

        # Download IREE Runtime C++ library (if optional enabled)
        if args.with_optional:
            if not download_iree_runtime(script_dir):
                print("\n✗ Failed to download IREE Runtime")
                # Non-fatal for now as it's a new feature
        
        # Download classification model and labels
        if not download_classification_model(script_dir):
            print("\n✗ Failed to download classification model")
            sys.exit(1)

        # Setup WebGPU environment
        setup_webgpu_environment(script_dir)

        # Create activation scripts
        create_activation_scripts(venv_path, script_dir)

        # Print next steps
        print_next_steps(venv_path, has_optional=args.with_optional)

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during setup: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
