# Copyright (c) 2026 Holochip Corporation
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.29)
project(AdvancedGLTF VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Simple Engine — bootstrap all engine dependencies.
#
# IMPORTED targets created inside add_subdirectory() are local to that
# subdirectory's scope and are NOT visible back in this parent scope.  We
# therefore call find_package() ourselves (using simple_engine's custom Find
# modules) so that all targets (glm::glm, tinygltf::tinygltf, etc.) are
# created in THIS scope before SimpleEngineLib is defined.
#
# Cache variables set here are warm when add_subdirectory runs, so the
# find_package() calls inside simple_engine are no-ops.  Parent-scope targets
# ARE visible in child scopes, so the if(NOT TARGET ...) guards inside each
# Find module skip duplicate creation.
#
# EXCLUDE_FROM_ALL prevents the SimpleEngine executable from building by
# default; this project builds AdvancedGLTF instead.
# ---------------------------------------------------------------------------
set(SE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../simple_engine")
if(NOT IS_DIRECTORY "${SE_DIR}")
    message(FATAL_ERROR
        "simple_engine not found at: ${SE_DIR}\n"
        "This project must be located at attachments/advanced_gltf/ alongside attachments/simple_engine/.")
endif()

# simple_engine's custom Find modules create INTERFACE/UNKNOWN IMPORTED targets
# in the subdirectory scope, which are NOT visible back in this parent scope.
# We must therefore run every find_package() in our own scope, using the same
# custom Find modules, so the resulting targets exist here.  All cache
# variables set by these calls (KTX_LIBRARY, tinygltf_INCLUDE_DIR, etc.) are
# visible everywhere, so when add_subdirectory later re-runs the same
# find_package() calls they become no-ops: the cache is warm and the
# if(NOT TARGET ...) guards inside each Find module see the targets we just
# created (parent-scope targets ARE visible in child scopes).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake" "${SE_DIR}/CMake")

find_package(glm       REQUIRED)
find_package(Vulkan    REQUIRED)
find_package(tinygltf  REQUIRED)
find_package(KTX       REQUIRED)
find_package(VulkanHpp REQUIRED)
if(NOT ANDROID)
    find_package(glfw3  REQUIRED)
    find_package(OpenAL REQUIRED)
endif()

add_subdirectory("${SE_DIR}" simple_engine_build EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# JoltPhysics v5.2.0 — physics engine used in chapters 3 and 4.
# NOTE: The Jolt API is not stable across major versions. Check CHANGELOG.md
#       for BodyInterface and constraint API changes before updating the tag.
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG        v5.2.0
    SOURCE_SUBDIR  Build)
set(DOUBLE_PRECISION                    OFF CACHE BOOL "" FORCE)
set(GENERATE_DEBUG_SYMBOLS              ON  CACHE BOOL "" FORCE)
set(PROFILER_IN_DEBUG_AND_RELEASE       ON  CACHE BOOL "" FORCE)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE ON  CACHE BOOL "" FORCE)
set(ENABLE_OBJECT_STREAM                ON  CACHE BOOL "" FORCE)
set(FLOATING_POINT_EXCEPTIONS_ENABLED   ON  CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS                 OFF CACHE BOOL "" FORCE)
if(MSVC)
    set(USE_STATIC_MSVC_RUNTIME_LIBRARY OFF CACHE BOOL "" FORCE)
    set(INTERPROCEDURAL_OPTIMIZATION ON  CACHE BOOL "" FORCE)
else()
    set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
endif()
FetchContent_MakeAvailable(JoltPhysics)

# ---------------------------------------------------------------------------
# JoltPhysics MSVC Fix
# Suppress C4865: "underlying type of enum will change when /Zc:enumTypes is specified"
# This warning is triggered in Windows SDK headers when Jolt is built with /Wall.
# We also disable /WX (Warnings as Errors) for Jolt via ENABLE_ALL_WARNINGS=OFF above.
# ---------------------------------------------------------------------------
if(MSVC)
    if(TARGET Jolt)
        target_compile_options(Jolt PRIVATE /wd4865)
    endif()
endif()

# ---------------------------------------------------------------------------
# Advanced glTF shader compilation.
# Uses SLANGC_EXECUTABLE set by simple_engine's find_program() call above.
# Target names are prefixed 'agltf_' to avoid colliding with the 'shaders'
# target already created inside simple_engine_build.
# ---------------------------------------------------------------------------
set(AGLTF_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(AGLTF_SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${AGLTF_SHADER_OUT}")

function(agltf_compile_shader target_name slang_file entry_point out_stem)
    set(spv_out "${AGLTF_SHADER_OUT}/${out_stem}.spv")
    set(entry_args "")
    if(NOT "${entry_point}" STREQUAL "")
        set(entry_args -entry "${entry_point}")
    endif()

    if(SLANGC_EXECUTABLE)
        add_custom_command(
            OUTPUT  "${spv_out}"
            COMMAND "${SLANGC_EXECUTABLE}"
                    "${slang_file}"
                    -target spirv
                    -profile spirv_1_5
                    -emit-spirv-directly
                    ${entry_args}
                    -o      "${spv_out}"
                    -I      "${AGLTF_SHADER_DIR}"
                    -I      "${SE_DIR}/shaders"
            DEPENDS "${slang_file}"
            COMMENT "slangc → ${out_stem}.spv"
            VERBATIM)
        add_custom_target("${target_name}" DEPENDS "${spv_out}")
    else()
        add_custom_target("${target_name}")
        message(STATUS "slangc not found — skipping ${out_stem}.spv")
    endif()
endfunction()

# Compile simple_engine shaders directly into our output directory
file(GLOB SE_SHADERS "${SE_DIR}/shaders/*.slang")
list(FILTER SE_SHADERS EXCLUDE REGEX ".*/(common_types|pbr_utils|lighting_utils|tonemapping_utils)\\.slang$")
set(SE_SHADER_TARGETS "")
foreach(shader_path ${SE_SHADERS})
    get_filename_component(shader_name_we ${shader_path} NAME_WE)
    agltf_compile_shader("se_shader_${shader_name_we}" "${shader_path}" "" "${shader_name_we}")
    list(APPEND SE_SHADER_TARGETS "se_shader_${shader_name_we}")
endforeach()

agltf_compile_shader(agltf_shader_skinning
    "${AGLTF_SHADER_DIR}/skinning.slang"
    main "skinning")

agltf_compile_shader(agltf_shader_morph
    "${AGLTF_SHADER_DIR}/morph_accumulate.slang"
    main "morph_accumulate")

agltf_compile_shader(agltf_shader_heatmap_vert
    "${AGLTF_SHADER_DIR}/pbr_heatmap.slang"
    vertex_main "pbr_heatmap_vertex")

agltf_compile_shader(agltf_shader_heatmap_dominant
    "${AGLTF_SHADER_DIR}/pbr_heatmap.slang"
    fragment_dominant_bone "pbr_heatmap_fragment_dominant_bone")

agltf_compile_shader(agltf_shader_heatmap_weights
    "${AGLTF_SHADER_DIR}/pbr_heatmap.slang"
    fragment_weight_distribution "pbr_heatmap_fragment_weight_distribution")

add_custom_target(advanced_gltf_shaders
    DEPENDS
        agltf_shader_skinning
        agltf_shader_morph
        agltf_shader_heatmap_vert
        agltf_shader_heatmap_dominant
        agltf_shader_heatmap_weights
        ${SE_SHADER_TARGETS})

# ---------------------------------------------------------------------------
# SimpleEngineLib — static library compiled from simple_engine's SOURCES_COMMON.
#
# simple_engine builds a desktop executable, not a linkable library. We
# replicate the explicit SOURCES_COMMON list here so AdvancedGLTF can link
# against engine code without modifying simple_engine's build system.
#
# This list MUST stay in sync with SOURCES_COMMON in simple_engine/CMakeLists.txt.
# Do NOT replace it with file(GLOB): vulkan_dispatch.cpp lives in the same
# directory but is intentionally absent from SOURCES_COMMON and must remain
# excluded.
# ---------------------------------------------------------------------------
set(SE_SOURCES_COMMON
    ${CMAKE_CURRENT_SOURCE_DIR}/engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scene_loading.cpp
    ${SE_DIR}/platform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer_rendering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer_pipelines.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer_resources.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/renderer_ray_query.cpp
    ${SE_DIR}/memory_pool.cpp
    ${SE_DIR}/resource_manager.cpp
    ${SE_DIR}/entity.cpp
    ${SE_DIR}/component.cpp
    ${SE_DIR}/transform_component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_component.cpp
    ${SE_DIR}/camera_component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/animation_component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/model_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/audio_system.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/physics_system.cpp
    ${SE_DIR}/imgui/imgui.cpp
    ${SE_DIR}/imgui/imgui_draw.cpp
    ${SE_DIR}/imgui/imgui_widgets.cpp
    ${SE_DIR}/imgui/imgui_tables.cpp
    ${SE_DIR}/vulkan_device.cpp
    ${SE_DIR}/pipeline.cpp
    ${SE_DIR}/descriptor_manager.cpp
    ${SE_DIR}/renderdoc_debug_system.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui_system.cpp
    ${SE_DIR}/mikktspace.c
)

# Prepend ${SE_DIR}/ to files that are NOT local to advanced_gltf
set(SE_SOURCES_ABSOLUTE "")
foreach(src ${SE_SOURCES_COMMON})
    if(IS_ABSOLUTE "${src}")
        list(APPEND SE_SOURCES_ABSOLUTE "${src}")
    else()
        list(APPEND SE_SOURCES_ABSOLUTE "${SE_DIR}/${src}")
    endif()
endforeach()
set(SE_SOURCES_COMMON ${SE_SOURCES_ABSOLUTE})

add_library(SimpleEngineLib STATIC ${SE_SOURCES_COMMON})
add_dependencies(SimpleEngineLib advanced_gltf_shaders)
set_target_properties(SimpleEngineLib PROPERTIES CXX_STANDARD 20)

target_include_directories(SimpleEngineLib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${SE_DIR}")

target_compile_definitions(SimpleEngineLib PUBLIC
    GLM_ENABLE_EXPERIMENTAL
    _USE_MATH_DEFINES
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    PLATFORM_DESKTOP)

target_link_libraries(SimpleEngineLib PUBLIC
    Vulkan::Vulkan
    VulkanHpp::VulkanHpp
    glm::glm
    tinygltf::tinygltf
    KTX::ktx
    Jolt)

target_link_libraries(SimpleEngineLib PRIVATE
    glfw
    OpenAL::OpenAL)

if(MSVC)
    # NOMINMAX needs to inherit so MSVC doesn't have problems with min / max being a MSVC macro.
    target_compile_definitions(SimpleEngineLib PUBLIC
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_options(SimpleEngineLib PUBLIC
        /permissive- /Zc:__cplusplus /EHsc /W3 /MP /bigobj /wd4714 /wd4723)

    target_link_libraries(SimpleEngineLib PUBLIC Dbghelp)

    # Enable Link Time Code Generation to match Jolt and resolve potential LTO mismatches
    set_target_properties(SimpleEngineLib PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(WIN32)
    target_compile_definitions(SimpleEngineLib PUBLIC
        NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

# ---------------------------------------------------------------------------
# AdvancedGLTF — tutorial demo executable
# ---------------------------------------------------------------------------
add_executable(AdvancedGLTF
    main.cpp
    tutorial_demo.cpp
    physics_world_jolt.cpp)

target_include_directories(AdvancedGLTF PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_definitions(AdvancedGLTF PRIVATE
    ADVANCED_GLTF_REQUIRES_DESCRIPTOR_INDEXING=1)

target_link_libraries(AdvancedGLTF PRIVATE
    SimpleEngineLib
    Jolt
    glm::glm)

if(MSVC)
    set_target_properties(AdvancedGLTF PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Ensure all shaders (from simple_engine and local) are in the executable's shader directory.
add_dependencies(AdvancedGLTF advanced_gltf_shaders)

# Copy downloaded sample assets if the assets directory has been populated.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/assets")
    add_custom_command(TARGET AdvancedGLTF POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/assets"
                "$<TARGET_FILE_DIR:AdvancedGLTF>/assets"
        COMMENT "Copying assets alongside executable"
        VERBATIM)
endif()
