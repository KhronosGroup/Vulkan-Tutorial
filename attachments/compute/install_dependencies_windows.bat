@echo off
REM Install dependencies for the Advanced Vulkan Compute tutorial demos (Windows).
REM Prerequisites:
REM   - vcpkg installed and VCPKG_INSTALLATION_ROOT set (https://github.com/microsoft/vcpkg)
REM   - Vulkan SDK (with slangc) from https://vulkan.lunarg.com/
REM
REM The compute demos require: GLFW (windowed demos), GLM, stb.
REM They do NOT require tinyobjloader, tinygltf, or KTX.

setlocal

echo Installing dependencies for Advanced Vulkan Compute demos...
echo.

REM -------------------------------------------------------------------
REM Check for vcpkg
REM -------------------------------------------------------------------
if "%VCPKG_INSTALLATION_ROOT%"=="" (
    echo ERROR: VCPKG_INSTALLATION_ROOT is not set.
    echo Please install vcpkg from https://github.com/microsoft/vcpkg and set:
    echo   set VCPKG_INSTALLATION_ROOT=C:\path\to\vcpkg
    echo.
    exit /b 1
)

if not exist "%VCPKG_INSTALLATION_ROOT%\vcpkg.exe" (
    echo ERROR: vcpkg.exe not found at %VCPKG_INSTALLATION_ROOT%\vcpkg.exe
    exit /b 1
)

echo Using vcpkg at: %VCPKG_INSTALLATION_ROOT%
echo.

REM -------------------------------------------------------------------
REM Configure binary caching (speeds up CI and repeat installs)
REM -------------------------------------------------------------------
if "%VCPKG_BINARY_SOURCES%"=="" (
    set VCPKG_BINARY_SOURCES=clear;files,%TEMP%\vcpkg-cache,readwrite
)

REM -------------------------------------------------------------------
REM Install packages
REM -------------------------------------------------------------------
echo Installing packages via vcpkg (x64-windows)...
REM 'opencl' provides the OpenCL ICD loader + headers used by Chapter 05.
"%VCPKG_INSTALLATION_ROOT%\vcpkg.exe" install ^
    glfw3 ^
    glm ^
    stb ^
    opencl ^
    --triplet=x64-windows

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: vcpkg install failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================================
echo  Vulkan SDK (with slangc) must be installed separately
echo ================================================================
echo Download the Vulkan SDK installer from: https://vulkan.lunarg.com/
echo Install it, then make sure VULKAN_SDK is set in your environment.
echo.
echo Verify slangc is available after SDK install:
echo   slangc --version
echo.
echo ================================================================
echo  Chapter 05 (OpenCL on Vulkan): clspv + clvk  (REQUIRED)
echo ================================================================
echo The Chapter 05 sample cannot run without clspv. clspv and clvk are not on
echo vcpkg, so build them from source in a Visual Studio "x64 Native Tools"
echo command prompt (clspv pulls in LLVM; the first build takes a while):
echo.
echo   git clone --depth 1 https://github.com/google/clspv.git
echo   cd clspv ^&^& python utils\fetch_sources.py --shallow
echo   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^&^& ninja -C build clspv
echo   REM add %CD%\build\bin to PATH so CMake finds clspv.exe
echo.
echo   git clone https://github.com/kpet/clvk.git
echo   cd clvk ^&^& git submodule update --init --recursive
echo   external\clspv\utils\fetch_sources.py --shallow
echo   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_EXTENSIONS=OFF ^&^& ninja -C build
echo   REM register OpenCL.dll from clvk\build as an ICD, or place it next to the exe
echo.
echo Verify the layered platform is visible with: clinfo -l   (look for 'clvk')
echo.
echo Build the compute demos:
echo   cd attachments\compute
echo   cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="%VCPKG_INSTALLATION_ROOT%\scripts\buildsystems\vcpkg.cmake"
echo   cmake --build build --config Release --parallel
echo.
echo Dependencies installed successfully.

endlocal
