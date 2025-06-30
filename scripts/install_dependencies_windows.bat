@echo off
echo Installing dependencies for Vulkan Tutorial...

:: Check if vcpkg is installed
where vcpkg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo vcpkg not found. Please install vcpkg first.
    echo Visit https://github.com/microsoft/vcpkg for installation instructions.
    echo Typically, you would:
    echo 1. git clone https://github.com/Microsoft/vcpkg.git
    echo 2. cd vcpkg
    echo 3. .\bootstrap-vcpkg.bat
    echo 4. Add vcpkg to your PATH
    exit /b 1
)

:: Install dependencies using vcpkg
echo Installing GLFW...
vcpkg install glfw3:x64-windows

echo Installing GLM...
vcpkg install glm:x64-windows

echo Installing tinyobjloader...
vcpkg install tinyobjloader:x64-windows

echo Installing stb...
vcpkg install stb:x64-windows

:: Remind about Vulkan SDK
echo.
echo Don't forget to install the Vulkan SDK from https://vulkan.lunarg.com/
echo.

echo All dependencies have been installed successfully!
echo You can now use CMake to build your Vulkan project.
echo.
echo Example CMake command:
echo cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path\to\vcpkg]\scripts\buildsystems\vcpkg.cmake
echo cmake --build build

exit /b 0
