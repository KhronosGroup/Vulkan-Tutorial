@echo off
rem Copyright (c) 2026 Holochip Corporation
rem
rem SPDX-License-Identifier: Apache-2.0
rem
rem Licensed under the Apache License, Version 2.0 the "License";
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.

rem Installs all dependencies for the Advanced glTF tutorial on Windows.
rem Delegates to the simple_engine install script (which uses vcpkg for glm,
rem GLFW, OpenAL, tinygltf, KTX, etc.) then notes that JoltPhysics is fetched
rem automatically by CMake via FetchContent.
setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set SE_SCRIPT=%SCRIPT_DIR%..\simple_engine\install_dependencies_windows.bat

if not exist "%SE_SCRIPT%" (
    echo Error: simple_engine install script not found at %SE_SCRIPT% >&2
    exit /b 1
)

echo === Installing simple_engine dependencies ===
call "%SE_SCRIPT%"
if errorlevel 1 (
    echo Error: simple_engine dependency installation failed. >&2
    exit /b 1
)

echo.
echo === Advanced glTF tutorial additional dependencies ===
echo JoltPhysics v5.2.0 is fetched automatically by CMake (FetchContent^).
echo No additional manual installation is required.
echo.
echo Build instructions:
echo   cd attachments\advanced_gltf
echo   mkdir build ^&^& cd build
echo   cmake ..
echo   cmake --build . --parallel
