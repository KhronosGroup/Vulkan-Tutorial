# Copyright (c) 2026, Khronos Group and contributors
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.26)

# This tutorial builds one executable that renders a glTF scene.
# The main objective is to showcase the use of Vulkan.
project(vulkan_siggraph LANGUAGES CXX)

# Export compile_commands.json.
# This allows editors and language servers to understand the C++ build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json for language servers")

# Vulkan provides headers and the loader library.
find_package(Vulkan REQUIRED)
# Git is used to populate pinned dependencies.
find_package(Git REQUIRED)

# Configure third party libraries
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
set(JSON_SystemInclude ON CACHE BOOL "" FORCE)

set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_WERROR OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_INSTALL OFF CACHE BOOL "" FORCE)

set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_GL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_VALIDATOR_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_BUILDER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)

set(SIGGRAPH_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(SIGGRAPH_DATA_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/external")
set(SIGGRAPH_FETCH_GIT_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/cmake/fetch_git.cmake")

# Ensure one pinned Git checkout exists in the source tree.
function(siggraph_checkout_git name repository tag source_dir sparse_path)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}"
      "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
      "-DGIT_REPOSITORY=${repository}"
      "-DGIT_TAG=${tag}"
      "-DSOURCE_DIR=${source_dir}"
      "-DSPARSE_PATH=${sparse_path}" -P
      "${SIGGRAPH_FETCH_GIT_SCRIPT}"
    COMMAND_ERROR_IS_FATAL ANY)

  set("${name}_SOURCE_DIR" "${source_dir}" PARENT_SCOPE)
endfunction()

# Dependencies are pinned here. The last argument is normally is used for sparse
# checkouts.
siggraph_checkout_git(
  glfw
  https://github.com/glfw/glfw.git
  b00e6a8a88ad1b60c0a045e696301deb92c9a13e
  "${SIGGRAPH_EXTERNAL_DIR}/glfw"
  "")

siggraph_checkout_git(
  glm
  https://github.com/g-truc/glm.git
  6f14f4792a0cde5d0cf2c910506724d61cb95834
  "${SIGGRAPH_EXTERNAL_DIR}/glm"
  "")

siggraph_checkout_git(
  vk_bootstrap
  https://github.com/charles-lunarg/vk-bootstrap.git
  5ca6780498864ae4c12f3a594ee6a6c5133d4ce0
  "${SIGGRAPH_EXTERNAL_DIR}/vk-bootstrap"
  "")

siggraph_checkout_git(
  nlohmann_json
  https://github.com/nlohmann/json.git
  4fad4468974a7b1b26d374b1c5955d2ac7d449b0
  "${SIGGRAPH_EXTERNAL_DIR}/nlohmann"
  "")

siggraph_checkout_git(
  stb
  https://github.com/nothings/stb.git
  31c1ad37456438565541f4919958214b6e762fb4
  "${SIGGRAPH_EXTERNAL_DIR}/stb"
  "")

siggraph_checkout_git(
  tinygltf
  https://github.com/syoyo/tinygltf.git
  d31c16e333a6c8d593cad43f325f4e1825dd4776
  "${SIGGRAPH_EXTERNAL_DIR}/tinygltf"
  "")

siggraph_checkout_git(
  gltf_sample_assets
  https://github.com/KhronosGroup/glTF-Sample-Assets.git
  2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf
  "${SIGGRAPH_DATA_EXTERNAL_DIR}/sponza"
  Models/Sponza # Sparse checkout. Only checkouts Sponza
)

# Add CMake-aware dependencies from the pinned checkouts.
add_subdirectory("${glfw_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/deps/glfw")
add_subdirectory("${glm_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/deps/glm")
add_subdirectory("${vk_bootstrap_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/deps/vk-bootstrap")
add_subdirectory("${nlohmann_json_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/deps/nlohmann")
add_subdirectory("${tinygltf_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/deps/tinygltf")

# stb has no CMake project, expose its fetched include directory as an interface
# target for normal target_link_libraries usage.
add_library(stb INTERFACE)
add_library(stb::stb ALIAS stb)
target_include_directories(stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")

# tinygltf provides a target in header - only mode, but not a namespaced alias.
if(NOT TARGET tinygltf::tinygltf)
  add_library(tinygltf::tinygltf ALIAS tinygltf)
endif()
target_include_directories(tinygltf SYSTEM INTERFACE "${tinygltf_SOURCE_DIR}")

# Slang compiles the small tutorial shaders to SPIR - V.
find_program(
  SLANGC_EXECUTABLE
  NAMES slangc
  HINTS "$ENV{VULKAN_SDK}/Bin"
  DOC "Path to the Slang compiler executable")
if(NOT SLANGC_EXECUTABLE)
  message(
    FATAL_ERROR
      "slangc was not found. Configure with -DSLANGC_EXECUTABLE=<path-to-slangc> or add slangc to PATH."
  )
endif()

# CMake writes compiled shader outputs into the build tree, not the source tree.
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

# Helper for declaring one shader compile step. The executable depends on the
# custom targets created here, so shaders are compiled before the app is linked.
function(add_slang_shader target_name source_file entry_point stage output_file)
  set(reflection_json "${output_file}.json")
  get_filename_component(output_dir "${output_file}" DIRECTORY)

  # Store shader source files so they can be added to the project.
  set_source_files_properties("${source_file}" PROPERTIES HEADER_FILE_ONLY TRUE)
  set_property(GLOBAL APPEND PROPERTY SIGGRAPH_SHADER_SOURCES "${source_file}")

  add_custom_command(
    OUTPUT "${output_file}" "${reflection_json}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
    COMMAND
      "${SLANGC_EXECUTABLE}" "${source_file}" -entry
      "${entry_point}" # Main function, for example vertexMain or fragmentMain.
      -stage "${stage}" # vertex, fragment, etc.
      -target spirv -profile spirv_1_5
      # Slang uses row major layout by default, but we want to use column major.
      -matrix-layout-column-major -reflection-json
      "${reflection_json}" # Generate reflection data as JSON.
      -o "${output_file}"
    DEPENDS "${source_file}"
    VERBATIM USES_TERMINAL
    COMMENT "Compiling ${source_file}")
  add_custom_target(
    "${target_name}"
    DEPENDS "${output_file}" "${reflection_json}"
    SOURCES "${source_file}")
endfunction()

# Output filenames used by the C++ code through SIGGRAPH_SHADER_DIR.
set(BASIC_VERT_SPV "${SHADER_OUTPUT_DIR}/basic.vert.spv")
set(BASIC_FRAG_SPV "${SHADER_OUTPUT_DIR}/basic.frag.spv")
set(ALBEDO_FRAG_SPV "${SHADER_OUTPUT_DIR}/albedo.frag.spv")
set(SOLID_COLOR_FRAG_SPV "${SHADER_OUTPUT_DIR}/solid_color.frag.spv")

# Compile the vertex shader. It transforms scene vertices and reads per object
# transforms through descriptor heap mappings.
add_slang_shader(
  basic_vertex_shader "${CMAKE_CURRENT_SOURCE_DIR}/shaders/basic.vert.slang"
  vertexMain
  vertex
  "${BASIC_VERT_SPV}")

# Compile the fragment shader. It samples textures selected through a push -
# index heap mapping.
add_slang_shader(
  basic_fragment_shader
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/basic.frag.slang"
  fragmentMain
  fragment
  "${BASIC_FRAG_SPV}")

# Compile the albedo - only fragment shader.It samples albedo but not the normal texture.
add_slang_shader(
  albedo_fragment_shader
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/albedo.frag.slang"
  fragmentMain
  fragment
  "${ALBEDO_FRAG_SPV}")

# Compile the solid - color fragment shader.It reads a constant - offset uniform
# buffer descriptor.
add_slang_shader(
  solid_color_fragment_shader
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/solid_color.frag.slang"
  fragmentMain
  fragment
  "${SOLID_COLOR_FRAG_SPV}")

# Main tutorial executable sources.
add_executable(vulkan_siggraph src/main.cpp src/main.h src/util.cpp src/util.h)

# Save shaders in their own source group called Shaders.
get_property(SIGGRAPH_SHADER_SOURCES GLOBAL PROPERTY SIGGRAPH_SHADER_SOURCES)
target_sources(vulkan_siggraph PRIVATE ${SIGGRAPH_SHADER_SOURCES})
source_group("Shaders" FILES ${SIGGRAPH_SHADER_SOURCES})

# Make shader compilation part of the normal application build.
add_dependencies(
  vulkan_siggraph
  basic_vertex_shader
  basic_fragment_shader
  albedo_fragment_shader
  solid_color_fragment_shader)

target_compile_definitions(
  vulkan_siggraph
  PRIVATE GLFW_INCLUDE_NONE
          # Points at the sample assets tutorial.
          SIGGRAPH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
          # tells the app from where to read SPIR - V files.
          SIGGRAPH_SHADER_DIR="${SHADER_OUTPUT_DIR}"
          # The Vulkan - Hpp definitions: enable dynamic dispatch loading and
          # designated initializers.
          VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
          VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)

# Allow source files to include local headers with #include "util.h".
target_include_directories(vulkan_siggraph PRIVATE src)

# The sample uses C++ 23.
target_compile_features(vulkan_siggraph PRIVATE cxx_std_23)

# Link Vulkan, the window library, math helpers, and bootstrap helpers.
target_link_libraries(
  vulkan_siggraph
  PRIVATE Vulkan::Vulkan
          glfw
          glm::glm
          vk-bootstrap::vk-bootstrap
          nlohmann_json::nlohmann_json
          stb::stb
          tinygltf::tinygltf)

# Use strict warnings so tutorial code stays clear and portable across compilers.
if(MSVC)
  target_compile_options(vulkan_siggraph PRIVATE /W4 /permissive-)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options( vulkan_siggraph PRIVATE /clang:-Wno-missing-designated-field-initializers)
  endif()
else()
  target_compile_options(vulkan_siggraph PRIVATE -Wall -Wextra -Wpedantic)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(vulkan_siggraph PRIVATE -Wno-missing-designated-field-initializers)
  endif()
endif()

# In Visual Studio, make the tutorial executable the default startup project.
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT vulkan_siggraph)
