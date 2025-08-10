@echo off
setlocal enabledelayedexpansion

REM Fetch the Bistro example assets into the desired assets directory.
REM Default target: assets\bistro at the repository root.
REM Usage:
REM   fetch_bistro_assets.bat [target-dir]
REM Examples:
REM   fetch_bistro_assets.bat
REM   fetch_bistro_assets.bat attachments\simple_engine\Assets\bistro

set REPO_SSH=git@github.com:gpx1000/bistro.git
set REPO_HTTPS=https://github.com/gpx1000/bistro.git

if "%~1"=="" (
    set TARGET_DIR=assets\bistro
) else (
    set TARGET_DIR=%~1
)

REM Ensure parent directory exists
for %%I in ("%TARGET_DIR%") do set PARENT=%%~dpI
if not exist "%PARENT%" mkdir "%PARENT%"

REM If directory exists and is a git repo, update it; otherwise clone it
if exist "%TARGET_DIR%\.git" (
    echo Updating existing bistro assets in %TARGET_DIR%
    pushd "%TARGET_DIR%"
    git pull --ff-only
    popd
) else (
    echo Cloning bistro assets into %TARGET_DIR%
    REM Try SSH first; fall back to HTTPS on failure
    git clone --depth 1 "%REPO_SSH%" "%TARGET_DIR%" 2>nul
    if %ERRORLEVEL% neq 0 (
        echo SSH clone failed, trying HTTPS
        git clone --depth 1 "%REPO_HTTPS%" "%TARGET_DIR%"
    )
)

REM If git-lfs is available, ensure LFS content is pulled
where git >nul 2>nul
if %ERRORLEVEL%==0 (
    pushd "%TARGET_DIR%"
    git lfs version >nul 2>nul
    if %ERRORLEVEL%==0 (
        git lfs install --local >nul 2>nul
        git lfs pull
    )
    popd
)

echo Bistro assets ready at: %TARGET_DIR%
endlocal
exit /b 0
