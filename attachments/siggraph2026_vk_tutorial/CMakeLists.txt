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

# This tutorial builds one executable that renders a descriptor-heap glTF scene with Vulkan shader objects.
project(vulkan_siggraph LANGUAGES CXX)

# Export compile_commands.json so editors and language servers can understand the C++ build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json for language servers")

# The Vulkan SDK provides headers, the loader library, and the Slang compiler used below.
find_package(Vulkan REQUIRED)

# Keep third-party libraries small: this sample only needs their runtime/library targets.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_WERROR OFF CACHE BOOL "" FORCE)

# Add vendored dependencies. vk-bootstrap removes a lot of boilerplate from instance,
# device, queue, and swapchain creation, which keeps the tutorial focused on Vulkan flow.
add_subdirectory(external/glfw)
add_subdirectory(external/glm)
add_subdirectory(external/vk-bootstrap)

# Slang compiles the small tutorial shaders to SPIR-V, the binary shader format Vulkan consumes.
find_program(SLANGC_EXECUTABLE
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/Bin"
    DOC "Path to the Slang compiler executable"
)
if(NOT SLANGC_EXECUTABLE)
  message(FATAL_ERROR "slangc was not found. Configure with -DSLANGC_EXECUTABLE=<path-to-slangc> or add slangc to PATH.")
endif()

# CMake writes compiled shader outputs into the build tree, not the source tree.
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

# Helper for declaring one shader compile step. The executable depends on the custom
# targets created here, so shaders are compiled before the app is linked/run.
function(add_slang_shader target_name source_file entry_point stage output_file)
  set(reflection_json "${output_file}.json")

  # Store shader source files so they can be added to the project
  set_source_files_properties("${source_file}" PROPERTIES HEADER_FILE_ONLY TRUE)
  set_property(GLOBAL APPEND PROPERTY SIGGRAPH_SHADER_SOURCES "${source_file}")

  add_custom_command(
        OUTPUT "${output_file}" "${reflection_json}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND "${SLANGC_EXECUTABLE}"
                "${source_file}"
                -entry "${entry_point}" # Main function, for example vertexMain or fragmentMain.
                -stage "${stage}" # vertex, fragment, etc.
                -target spirv
                -profile spirv_1_5
                -matrix-layout-column-major # Keep Slang's matrix layout explicit.
                -reflection-json "${reflection_json}" # Generate reflection data as JSON.
                -o "${output_file}"
        DEPENDS "${source_file}"
        VERBATIM
        COMMENT "Compiling ${source_file}"
    )
  add_custom_target("${target_name}"
      DEPENDS "${output_file}" "${reflection_json}"
      SOURCES "${source_file}"
  )
endfunction()

# Output filenames used by the C++ code through SIGGRAPH_SHADER_DIR.
set(BASIC_VERT_SPV "${SHADER_OUTPUT_DIR}/basic.vert.spv")
set(BASIC_FRAG_SPV "${SHADER_OUTPUT_DIR}/basic.frag.spv")
set(ALBEDO_FRAG_SPV "${SHADER_OUTPUT_DIR}/albedo.frag.spv")
set(SOLID_COLOR_FRAG_SPV "${SHADER_OUTPUT_DIR}/solid_color.frag.spv")

# Compile the vertex shader. It transforms scene vertices and reads per-object transforms through descriptor heap mappings.
add_slang_shader(
    basic_vertex_shader
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/basic.vert.slang"
    vertexMain
    vertex
    "${BASIC_VERT_SPV}"
)

# Compile the fragment shader. It samples textures selected through a push-index heap mapping.
add_slang_shader(
    basic_fragment_shader
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/basic.frag.slang"
    fragmentMain
    fragment
    "${BASIC_FRAG_SPV}"
)

# Compile the albedo-only fragment shader. It samples albedo and lights with interpolated geometry normals.
add_slang_shader(
    albedo_fragment_shader
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/albedo.frag.slang"
    fragmentMain
    fragment
    "${ALBEDO_FRAG_SPV}"
)

# Compile the solid-color fragment shader. It reads a constant-offset uniform buffer descriptor.
add_slang_shader(
    solid_color_fragment_shader
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/solid_color.frag.slang"
    fragmentMain
    fragment
    "${SOLID_COLOR_FRAG_SPV}"
)

# Main tutorial executable sources.
add_executable(vulkan_siggraph
    src/main.cpp
    src/main.h
    src/util.cpp
    src/util.h
)

# Save shaders in their own source group called shaders
# Visual Studio will display this as a folder: vulkan_siggraph/shaders
get_property(SIGGRAPH_SHADER_SOURCES GLOBAL PROPERTY SIGGRAPH_SHADER_SOURCES)
target_sources(vulkan_siggraph PRIVATE ${SIGGRAPH_SHADER_SOURCES})
source_group("Shaders" FILES ${SIGGRAPH_SHADER_SOURCES})

# Make shader compilation part of the normal application build.
add_dependencies(vulkan_siggraph basic_vertex_shader basic_fragment_shader albedo_fragment_shader solid_color_fragment_shader)

# SIGGRAPH_SHADER_DIR tells the app where CMake put the SPIR-V files.
# SIGGRAPH_DATA_DIR points at the sample assets used by the descriptor heap tutorial.
# The Vulkan-Hpp definitions enable dynamic dispatch loading and designated initializers.
target_compile_definitions(vulkan_siggraph
    PRIVATE
        GLFW_INCLUDE_NONE
        SIGGRAPH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
        SIGGRAPH_SHADER_DIR="${SHADER_OUTPUT_DIR}"
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
)

# Allow source files to include local headers with #include "util.h".
target_include_directories(vulkan_siggraph PRIVATE src)
target_include_directories(vulkan_siggraph SYSTEM PRIVATE external/nlohmann/single_include)
target_include_directories(vulkan_siggraph SYSTEM PRIVATE external/stb)
target_include_directories(vulkan_siggraph SYSTEM PRIVATE external/tinygltf)

# The sample uses C++23 library/features and Vulkan-Hpp designated initialization.
target_compile_features(vulkan_siggraph PRIVATE cxx_std_23)

# Link Vulkan, the window library, math helpers and bootstrap helpers
target_link_libraries(vulkan_siggraph
    PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
        vk-bootstrap::vk-bootstrap
)

# Use strict warnings so tutorial code stays clear and portable across compilers.
if(MSVC)
  target_compile_options(vulkan_siggraph PRIVATE /std:c++latest /W4 /permissive-)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(vulkan_siggraph PRIVATE /clang:-Wno-missing-designated-field-initializers)
  endif()
else()
  target_compile_options(vulkan_siggraph PRIVATE -std=c++2c -Wall -Wextra -Wpedantic -Wno-missing-designated-field-initializers)
endif()

# In Visual Studio, make the tutorial executable the default startup project.
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT vulkan_siggraph)
