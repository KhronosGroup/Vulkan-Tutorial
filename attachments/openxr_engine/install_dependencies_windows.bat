@echo off
set SCRIPT_DIR=%~dp0
set SIMPLE_ENGINE_DIR=%SCRIPT_DIR%..\simple_engine

if not exist "%SIMPLE_ENGINE_DIR%" (
    echo Error: simple_engine directory not found at %SIMPLE_ENGINE_DIR%
    exit /b 1
)

:: Ensure vcpkg is accessible
where vcpkg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    if defined VCPKG_INSTALLATION_ROOT (
        if exist "%VCPKG_INSTALLATION_ROOT%\vcpkg.exe" (
            set "PATH=%VCPKG_INSTALLATION_ROOT%;%PATH%"
        )
    )
)

echo Calling simple_engine dependencies installer...
call "%SIMPLE_ENGINE_DIR%\install_dependencies_windows.bat"
if %ERRORLEVEL% neq 0 (
    echo Error: simple_engine dependencies installation failed.
    exit /b %ERRORLEVEL%
)

echo.
echo Installing OpenXR loader...
vcpkg install openxr-loader --triplet=x64-windows
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install openxr-loader.
    exit /b %ERRORLEVEL%
)

echo.
echo OpenXR dependencies installation completed!
exit /b 0
