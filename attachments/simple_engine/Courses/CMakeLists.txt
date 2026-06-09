# =============================================================================
# Courses/CMakeLists.txt
# =============================================================================
#
# This subdirectory contains self-contained course modules that extend the
# Simple Engine with new rendering features.  Each module is optional — if
# the parent CMakeLists.txt includes this directory with:
#
#   option(ENABLE_COURSES "Build the Simple Engine course modules" ON)
#   if(ENABLE_COURSES)
#       add_subdirectory(Courses)
#   endif()
#
# then all enabled courses are compiled as a STATIC library that is linked
# into the SimpleEngine target.  No existing engine source files are modified
# by this build; the modules attach to the engine via the public extension
# points added to renderer.h.
#
# ADDING A NEW COURSE
# -------------------
# 1. Create a subdirectory under Courses/ (e.g. Courses/My_New_Topic/).
# 2. Add its source files to the COURSE_SOURCES list below.
# 3. If the course has a Slang shader, add it to COURSE_SLANG_SHADERS below.
# 4. Document the course in en/Building_a_Simple_Engine/Courses/.
#
# The course modules in this file:
#   * Opacity Micromaps  — VK_KHR_opacity_micromap hardware shadow acceleration
# =============================================================================

cmake_minimum_required(VERSION 3.29)

# ─────────────────────────────────────────────────────────────────────────────
# Per-course opt-in options
# ─────────────────────────────────────────────────────────────────────────────
option(ENABLE_COURSE_OPACITY_MICROMAPS
       "Build the Opacity Micromaps course module (VK_KHR_opacity_micromap)"
       ON)

# ─────────────────────────────────────────────────────────────────────────────
# Collect sources from enabled course modules
# ─────────────────────────────────────────────────────────────────────────────
set(COURSE_SOURCES "")
set(COURSE_SLANG_SHADERS "")

if(ENABLE_COURSE_OPACITY_MICROMAPS)
    list(APPEND COURSE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/opacity_micromap_builder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/omm_integration.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/omm_imgui_panel.cpp
    )
    # No standalone shader needed: VK_KHR_opacity_micromap is transparent to
    # the existing ray_query.slang shader — the hardware handles everything.
    message(STATUS "[Courses] Opacity Micromaps module ENABLED")
else()
    message(STATUS "[Courses] Opacity Micromaps module disabled")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Build the course modules library (only if there are sources)
# ─────────────────────────────────────────────────────────────────────────────
if(COURSE_SOURCES)
    add_library(SimpleEngineCourses STATIC ${COURSE_SOURCES})

    set_target_properties(SimpleEngineCourses PROPERTIES CXX_STANDARD 20)

    if(ENABLE_COURSE_OPACITY_MICROMAPS)
        target_compile_definitions(SimpleEngineCourses PUBLIC ENABLE_COURSE_OPACITY_MICROMAPS)
    endif()

    # The course modules include engine headers directly (../renderer.h etc.)
    target_include_directories(SimpleEngineCourses PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..        # engine root (renderer.h, etc.)
    )

    # Inherit the same compile definitions as the main engine target so that
    # the Vulkan-Hpp dispatch loader and GLM settings are consistent.
    target_compile_definitions(SimpleEngineCourses PRIVATE
        GLM_ENABLE_EXPERIMENTAL
        _USE_MATH_DEFINES
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    )

    # MSVC portability
    if(MSVC)
        target_compile_definitions(SimpleEngineCourses PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            _CRT_SECURE_NO_WARNINGS
        )
        target_compile_options(SimpleEngineCourses PRIVATE
            /permissive-
            /Zc:__cplusplus
            /EHsc
            /W3
            /MP
        )
    endif()

    # Link against Vulkan (same path as the engine).
    if(TARGET Vulkan::cppm)
        target_link_libraries(SimpleEngineCourses PUBLIC Vulkan::cppm)
    else()
        target_link_libraries(SimpleEngineCourses PUBLIC Vulkan::Vulkan)
    endif()

    target_link_libraries(SimpleEngineCourses PUBLIC glm::glm)

    if (ANDROID)
        target_link_libraries(SimpleEngineCourses PUBLIC game-activity::game-activity)
    endif ()

    # Export the library name to the parent scope so CMakeLists.txt can link it.
    set(SIMPLE_ENGINE_COURSES_LIB SimpleEngineCourses PARENT_SCOPE)

    # ── Shader compilation (future courses that DO add shaders) ──────────────
    if(COURSE_SLANG_SHADERS AND SLANGC_EXECUTABLE)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)
        foreach(SHADER ${COURSE_SLANG_SHADERS})
            get_filename_component(SHADER_NAME ${SHADER} NAME)
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv
                COMMAND ${SLANGC_EXECUTABLE} ${SHADER}
                        -target spirv -profile spirv_1_4 -emit-spirv-directly
                        -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv
                DEPENDS ${SHADER}
                COMMENT "Compiling course shader ${SHADER_NAME}"
            )
            list(APPEND COURSE_SHADER_SPVS
                ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv)
        endforeach()

        if(COURSE_SHADER_SPVS)
            add_custom_target(CourseShaders DEPENDS ${COURSE_SHADER_SPVS})
            add_dependencies(SimpleEngineCourses CourseShaders)
        endif()
    endif()

else()
    # No course sources enabled — export an empty library name.
    set(SIMPLE_ENGINE_COURSES_LIB "" PARENT_SCOPE)
    message(STATUS "[Courses] No course modules enabled; skipping library build.")
endif()
