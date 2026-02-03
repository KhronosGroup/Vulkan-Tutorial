@echo off
REM One-time setup script for ML Inference examples
REM Run this once: setup_env.bat
REM Optional: setup_env.bat --with-optional

setlocal enabledelayedexpansion

echo ============================================================
echo ML Inference Examples - One-Time Setup
echo ============================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3 is required but not found
    echo Please install Python 3.8 or later from python.org
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Found Python: %PYTHON_VERSION%
echo.

REM Run the Python setup script
if "%1"=="--with-optional" (
    echo Installing with optional dependencies ^(ONNX Runtime, TensorFlow Lite^)...
    python setup_python_env.py --with-optional
) else (
    echo Installing core dependencies only...
    echo To include optional dependencies, run: setup_env.bat --with-optional
    echo.
    python setup_python_env.py
)

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Open the project in your IDE ^(Visual Studio, CLion, VS Code, etc.^)
echo   2. Configure CMake ^(your IDE will do this automatically^)
echo   3. Build the project
echo   4. Run the train_mnist target to train the model
echo.
echo No further setup needed - CMake handles everything!
echo.

pause
