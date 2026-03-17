@echo off
set SCRIPT_DIR=%~dp0
set SIMPLE_ENGINE_DIR=%SCRIPT_DIR%..\simple_engine

if not exist "%SIMPLE_ENGINE_DIR%" (
    echo Error: simple_engine directory not found at %SIMPLE_ENGINE_DIR%
    exit /b 1
)

echo Calling simple_engine dependencies installer...
call "%SIMPLE_ENGINE_DIR%\install_dependencies_windows.bat"

echo.
echo Installing OpenXR loader...
vcpkg install openxr-loader --triplet=x64-windows

echo.
echo OpenXR dependencies installation completed!
exit /b 0
