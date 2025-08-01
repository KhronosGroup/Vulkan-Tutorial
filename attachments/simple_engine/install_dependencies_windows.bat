@echo off
REM Install script for Simple Game Engine dependencies on Windows
REM This script installs all required dependencies for building the Simple Game Engine

echo Installing Simple Game Engine dependencies for Windows...

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator - good!
) else (
    echo This script requires administrator privileges.
    echo Please run as administrator.
    pause
    exit /b 1
)

REM Check if vcpkg is installed
where vcpkg >nul 2>&1
if %errorLevel% == 0 (
    echo vcpkg found in PATH
) else (
    echo vcpkg not found in PATH. Installing vcpkg...

    REM Install vcpkg
    if not exist "C:\vcpkg" (
        echo Cloning vcpkg to C:\vcpkg...
        git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
        if %errorLevel% neq 0 (
            echo Failed to clone vcpkg. Please install Git first.
            pause
            exit /b 1
        )
    )

    REM Bootstrap vcpkg
    echo Bootstrapping vcpkg...
    cd /d C:\vcpkg
    call bootstrap-vcpkg.bat
    if %errorLevel% neq 0 (
        echo Failed to bootstrap vcpkg.
        pause
        exit /b 1
    )

    REM Add vcpkg to PATH for this session
    set PATH=%PATH%;C:\vcpkg

    REM Integrate vcpkg with Visual Studio
    echo Integrating vcpkg with Visual Studio...
    vcpkg integrate install
)

REM Check if Chocolatey is installed for additional packages
where choco >nul 2>&1
if %errorLevel% neq 0 (
    echo Installing Chocolatey package manager...
    powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
    if %errorLevel% neq 0 (
        echo Failed to install Chocolatey. Some dependencies may need manual installation.
    )
)

REM Install CMake if not present
where cmake >nul 2>&1
if %errorLevel% neq 0 (
    echo Installing CMake...
    choco install cmake -y
    if %errorLevel% neq 0 (
        echo Failed to install CMake via Chocolatey. Please install manually from https://cmake.org/download/
    )
)

REM Install Git if not present
where git >nul 2>&1
if %errorLevel% neq 0 (
    echo Installing Git...
    choco install git -y
    if %errorLevel% neq 0 (
        echo Failed to install Git via Chocolatey. Please install manually from https://git-scm.com/download/win
    )
)

REM Install Vulkan SDK
echo Installing Vulkan SDK...
if not exist "C:\VulkanSDK" (
    echo Downloading and installing Vulkan SDK...
    choco install vulkan-sdk -y
    if %errorLevel% neq 0 (
        echo Failed to install Vulkan SDK via Chocolatey.
        echo Please download and install manually from https://vulkan.lunarg.com/sdk/home#windows
        echo Make sure to set the VULKAN_SDK environment variable.
    )
) else (
    echo Vulkan SDK appears to be already installed.
)

REM Install vcpkg packages
echo Installing dependencies via vcpkg...

REM Set vcpkg triplet for x64 Windows
set VCPKG_DEFAULT_TRIPLET=x64-windows

REM Install GLFW
echo Installing GLFW...
vcpkg install glfw3:x64-windows
if %errorLevel% neq 0 (
    echo Warning: Failed to install GLFW via vcpkg
)

REM Install GLM
echo Installing GLM...
vcpkg install glm:x64-windows
if %errorLevel% neq 0 (
    echo Warning: Failed to install GLM via vcpkg
)

REM Install OpenAL
echo Installing OpenAL...
vcpkg install openal-soft:x64-windows
if %errorLevel% neq 0 (
    echo Warning: Failed to install OpenAL via vcpkg
)

REM Install KTX
echo Installing KTX...
vcpkg install ktx:x64-windows
if %errorLevel% neq 0 (
    echo Warning: Failed to install KTX via vcpkg
)

REM Install STB
echo Installing STB...
vcpkg install stb:x64-windows
if %errorLevel% neq 0 (
    echo Warning: Failed to install STB via vcpkg
)

REM Install tinygltf
echo Installing tinygltf...
vcpkg install tinygltf:x64-windows
if %errorLevel% neq 0 (
    echo Warning: Failed to install tinygltf via vcpkg
)

REM Install Slang compiler
echo Installing Slang compiler...
if not exist "C:\Program Files\Slang" (
    echo Downloading Slang compiler...
    set SLANG_VERSION=2024.1.21
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/shader-slang/slang/releases/download/v%SLANG_VERSION%/slang-%SLANG_VERSION%-win64.zip' -OutFile 'slang-win64.zip'"
    if %errorLevel% == 0 (
        echo Extracting Slang compiler...
        powershell -Command "Expand-Archive -Path 'slang-win64.zip' -DestinationPath 'C:\Program Files\Slang' -Force"
        del slang-win64.zip

        REM Add Slang to PATH (requires restart or new command prompt)
        echo Adding Slang to system PATH...
        for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH') do set "CURRENT_PATH=%%B"
        setx PATH "%CURRENT_PATH%;C:\Program Files\Slang\bin" /M
        echo Note: You may need to restart your command prompt for Slang to be available in PATH
    ) else (
        echo Failed to download Slang compiler. Please install manually from:
        echo https://github.com/shader-slang/slang/releases
    )
) else (
    echo Slang compiler appears to be already installed.
)

REM Set environment variables for CMake to find vcpkg
echo Setting up CMake integration...
setx CMAKE_TOOLCHAIN_FILE "C:\vcpkg\scripts\buildsystems\vcpkg.cmake" /M
setx VCPKG_TARGET_TRIPLET "x64-windows" /M

echo.
echo Dependencies installation completed!
echo.
echo To build the Simple Game Engine:
echo 1. Open a new command prompt (to get updated PATH)
echo 2. cd to the simple_engine directory
echo 3. mkdir build ^&^& cd build
echo 4. cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
echo 5. cmake --build . --config Release
echo.
echo Or use Visual Studio:
echo 1. Open the CMakeLists.txt file in Visual Studio
echo 2. Visual Studio should automatically detect vcpkg integration
echo 3. Build the project using Ctrl+Shift+B
echo.
echo Note: You may need to restart your command prompt or IDE for environment variables to take effect.

pause
