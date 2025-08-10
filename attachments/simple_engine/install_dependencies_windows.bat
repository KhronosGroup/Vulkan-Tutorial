@echo off
REM Install script for Simple Game Engine dependencies on Windows
REM This script installs all required dependencies for building the Simple Game Engine

echo Installing Simple Game Engine dependencies for Windows...

REM Check if running as administrator
REM Administrator privileges are not required. Proceeding without elevation.
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator (optional).
) else (
    echo Running without administrator privileges.
)

REM vcpkg detection and optional local install
set "VCPKG_EXE="
where vcpkg >nul 2>&1
if %errorlevel%==0 (
    echo vcpkg found in PATH
    set "VCPKG_EXE=vcpkg"
) else (
    echo vcpkg not found in PATH.
    set "VCPKG_HOME=%USERPROFILE%\vcpkg"
    set /p INSTALL_VCPKG="Install vcpkg locally to %VCPKG_HOME%? (Y/N): "
    if /I "%INSTALL_VCPKG%"=="Y" (
        if not exist "%VCPKG_HOME%" (
            echo Cloning vcpkg into %VCPKG_HOME% ...
            git clone https://github.com/Microsoft/vcpkg.git "%VCPKG_HOME%"
            if %errorlevel% neq 0 (
                echo Failed to clone vcpkg. Ensure Git is installed and try again.
                goto AFTER_VCPKG
            )
        ) else (
            echo vcpkg directory already exists at %VCPKG_HOME%
        )
        pushd "%VCPKG_HOME%"
        call bootstrap-vcpkg.bat
        if %errorlevel% neq 0 (
            echo Failed to bootstrap vcpkg.
            popd
            goto AFTER_VCPKG
        )
        popd
        set "VCPKG_EXE=%VCPKG_HOME%\vcpkg.exe"
        set /p ADD_VCPKG_PATH="Add vcpkg to PATH for this session? (Y/N): "
        if /I "%ADD_VCPKG_PATH%"=="Y" set "PATH=%PATH%;%VCPKG_HOME%"
    ) else (
        echo Skipping vcpkg installation.
    )
)
:AFTER_VCPKG

REM Tool checks (no forced install)
where cmake >nul 2>&1
if %errorlevel%==0 (
    echo CMake found in PATH
) else (
    echo CMake not found in PATH.
    set /p OPEN_CMAKE="Open CMake download page in browser? (Y/N): "
    if /I "%OPEN_CMAKE%"=="Y" start "" "https://cmake.org/download/"
)

where git >nul 2>&1
if %errorlevel%==0 (
    echo Git found in PATH
) else (
    echo Git not found in PATH.
    set /p OPEN_GIT="Open Git for Windows download page? (Y/N): "
    if /I "%OPEN_GIT%"=="Y" start "" "https://git-scm.com/download/win"
)

REM Vulkan SDK detection (no forced install)
set "HAVE_VULKAN_SDK="
if defined VULKAN_SDK set "HAVE_VULKAN_SDK=1"
where vulkaninfo >nul 2>&1
if %errorlevel%==0 set "HAVE_VULKAN_SDK=1"
if defined HAVE_VULKAN_SDK (
    echo Vulkan SDK detected.
) else (
    echo Vulkan SDK not detected.
    set /p OPEN_VULKAN="Open Vulkan SDK download page (LunarG) in browser? (Y/N): "
    if /I "%OPEN_VULKAN%"=="Y" start "" "https://vulkan.lunarg.com/sdk/home#windows"
)

REM Optional vcpkg package installation
if defined VCPKG_EXE (
    set /p INSTALL_VCPKG_PKGS="Install common dependencies via vcpkg (glfw3, glm, openal-soft, ktx, tinygltf)? (Y/N): "
    if /I "%INSTALL_VCPKG_PKGS%"=="Y" (
        set "VCPKG_DEFAULT_TRIPLET=x64-windows"
        echo Installing packages with %VCPKG_EXE% (triplet %VCPKG_DEFAULT_TRIPLET%) ...
        "%VCPKG_EXE%" install glfw3:%VCPKG_DEFAULT_TRIPLET% glm:%VCPKG_DEFAULT_TRIPLET% openal-soft:%VCPKG_DEFAULT_TRIPLET% ktx:%VCPKG_DEFAULT_TRIPLET% tinygltf:%VCPKG_DEFAULT_TRIPLET%
        if %errorlevel% neq 0 (
            echo Warning: Some vcpkg installations may have failed. Please review output.
        )
    ) else (
        echo Skipping vcpkg package installation.
    )
) else (
    echo vcpkg not available; skipping vcpkg package installation.
)

REM Slang compiler detection and optional install
set "SLANGC_EXE="
where slangc >nul 2>&1
if %errorlevel%==0 (
    echo Slang compiler found in PATH
    set "SLANGC_EXE=slangc"
) else (
    if defined VULKAN_SDK (
        if exist "%VULKAN_SDK%\Bin\slangc.exe" set "SLANGC_EXE=%VULKAN_SDK%\Bin\slangc.exe"
        if not defined SLANGC_EXE if exist "%VULKAN_SDK%\Bin64\slangc.exe" set "SLANGC_EXE=%VULKAN_SDK%\Bin64\slangc.exe"
    )
)

if defined SLANGC_EXE (
    echo Using Slang at %SLANGC_EXE%
) else (
    echo Slang compiler (slangc) not found.
    set /p INSTALL_SLANG="Download and install latest Slang locally (no admin)? (Y/N): "
    if /I "%INSTALL_SLANG%"=="Y" (
        set "SLANG_ROOT=%LOCALAPPDATA%\slang"
        if not exist "%SLANG_ROOT%" mkdir "%SLANG_ROOT%"
        echo Downloading latest Slang release...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "\
$ErrorActionPreference='Stop'; \
$r=Invoke-RestMethod 'https://api.github.com/repos/shader-slang/slang/releases/latest'; \
$asset=$r.assets | Where-Object { $_.name -match 'win64.*\\.zip$' } | Select-Object -First 1; \
if(-not $asset){ throw 'No win64 asset found'; } \
$out=Join-Path $env:TEMP $asset.name; \
Invoke-WebRequest $asset.browser_download_url -OutFile $out; \
Expand-Archive -Path $out -DestinationPath $env:LOCALAPPDATA\slang -Force; \
Write-Host ('Downloaded Slang ' + $r.tag_name) \
"
        echo Locating slangc.exe...
        set "SLANGC_PATH="
        for /f "delims=" %%F in ('dir /b /s "%LOCALAPPDATA%\slang\slangc.exe" 2^>nul') do (
            set "SLANGC_PATH=%%F"
            goto FOUND_SLANG
        )
        :FOUND_SLANG
        if defined SLANGC_PATH (
            echo Found slangc at "%SLANGC_PATH%"
            for %%D in ("%SLANGC_PATH%") do set "SLANG_BIN=%%~dpD"
            set /p ADD_SLANG_PATH="Add Slang to PATH for this session? (Y/N): "
            if /I "%ADD_SLANG_PATH%"=="Y" set "PATH=%SLANG_BIN%;%PATH%"
            set "SLANGC_EXE=%SLANGC_PATH%"
        ) else (
            echo Failed to locate slangc after extraction. Please install manually if needed: https://github.com/shader-slang/slang/releases
        )
    ) else (
        echo Skipping Slang installation.
    )
)

REM Final guidance (no machine-wide env changes)
echo.
echo Dependencies check completed!
echo.
echo Build instructions:
echo 1. Open a new command prompt (if you added tools to PATH for this session).
echo 2. cd to attachments\simple_engine
echo 3. mkdir build ^&^& cd build
echo 4. If using vcpkg toolchain, run:
echo    cmake .. -DCMAKE_TOOLCHAIN_FILE=%VCPKG_HOME%\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
echo    (adjust path if you installed vcpkg elsewhere; or omit this flag if not using vcpkg)
echo 5. cmake --build . --config Release
echo.
echo Alternatively open CMakeLists.txt in Visual Studio and configure normally.
echo.

pause
