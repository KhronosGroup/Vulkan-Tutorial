cmake_minimum_required(VERSION 3.29)

project(OpenXREngine VERSION 1.0.0 LANGUAGES CXX C)

# Option to enable/disable Vulkan C++20 module support for this standalone project
option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan in OpenXREngine" OFF)

# Enable C++ module dependency scanning only when modules are enabled
if(ENABLE_CPP20_MODULE)
    set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

# Path to the base Simple Engine
set(SIMPLE_ENGINE_DIR "${CMAKE_CURRENT_LIST_DIR}/../simple_engine")

# Add CMake module path for custom find modules
list(APPEND CMAKE_MODULE_PATH "${SIMPLE_ENGINE_DIR}/CMake")

# Find required packages
find_package (glm REQUIRED)
find_package (Vulkan REQUIRED)
find_package (tinygltf REQUIRED)
find_package (KTX REQUIRED)

# Find or download Vulkan-Hpp headers matching the Vulkan SDK/NDK version
find_package(VulkanHpp REQUIRED)

# OpenXR is REQUIRED for this engine variant
find_package(OpenXR REQUIRED)

if(ENABLE_CPP20_MODULE)
    # Set up Vulkan C++ module
    add_library(VulkanCppModule)
    add_library(Vulkan::cppm ALIAS VulkanCppModule)

    target_compile_definitions(VulkanCppModule
            PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )
    target_include_directories(VulkanCppModule
            PUBLIC
            "${Vulkan_INCLUDE_DIR}"
            "${VulkanHpp_INCLUDE_DIRS}"
    )
    target_link_libraries(VulkanCppModule
            PUBLIC
            Vulkan::Vulkan
    )

    set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

    target_sources(VulkanCppModule
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES
            BASE_DIRS
            "${VulkanHpp_CPPM_DIR}"
            FILES
            "${VulkanHpp_CPPM_DIR}/vulkan/vulkan.cppm"
    )

    # MSVC-specific options to improve module support
    if(MSVC)
        target_compile_options(VulkanCppModule PRIVATE
            /std:c++latest
            /permissive-
            /Zc:__cplusplus
            /EHsc
            /Zc:preprocessor
        )
    endif()
else()
    add_library(VulkanCppModule INTERFACE)
    add_library(Vulkan::cppm ALIAS VulkanCppModule)
    target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
    target_compile_definitions(VulkanCppModule
            INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )
    target_include_directories(VulkanCppModule INTERFACE "${VulkanHpp_INCLUDE_DIRS}")
endif()

# Platform-specific settings
if(ANDROID)
    # Android-specific settings
    add_definitions(-DPLATFORM_ANDROID)
    find_package(game-activity REQUIRED CONFIG)
else()
    # Desktop-specific settings
    add_definitions(-DPLATFORM_DESKTOP)
    find_package(glfw3 REQUIRED)
    find_package(OpenAL REQUIRED)
endif()

# Shader compilation (using local shaders in openxr_engine)
file(GLOB SLANG_SHADER_SOURCES ${CMAKE_CURRENT_LIST_DIR}/shaders/*.slang)
list(FILTER SLANG_SHADER_SOURCES EXCLUDE REGEX ".*/(common_types|pbr_utils|lighting_utils|tonemapping_utils)\\.slang$")

# Find slangc executable
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin)

if(SLANGC_EXECUTABLE)
    # Ensure the output directory for compiled shaders exists
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

    # Compile Slang shaders using slangc
    foreach(SHADER ${SLANG_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        string(REGEX REPLACE "\.slang$" "" OUTPUT_NAME ${SHADER_NAME})
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_NAME}.spv
            COMMAND ${SLANGC_EXECUTABLE} ${SHADER} -target spirv -profile spirv_1_4 -emit-spirv-directly -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_NAME}.spv
            DEPENDS ${SHADER}
            COMMENT "Compiling Slang shader ${SHADER_NAME} with slangc"
        )
        list(APPEND SHADER_SPVS ${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_NAME}.spv)
    endforeach()

    add_custom_target(shaders DEPENDS ${SHADER_SPVS})
else()
    add_custom_target(shaders)
endif()

# Source files
# We reference most files from Simple Engine via SIMPLE_ENGINE_DIR.
# Local files are those modified for OpenXR support.
set(SOURCES_COMMON
    engine.cpp                       # OpenXR version
    renderer_core.cpp                # OpenXR version
    renderer_rendering.cpp           # OpenXR version
    renderer_resources.cpp           # OpenXR version
    camera_component.cpp             # OpenXR version
    xr_context.cpp                   # OpenXR only

    ${SIMPLE_ENGINE_DIR}/scene_loading.cpp
    ${SIMPLE_ENGINE_DIR}/platform.cpp
    renderer_pipelines.cpp
    ${SIMPLE_ENGINE_DIR}/renderer_compute.cpp
    ${SIMPLE_ENGINE_DIR}/renderer_utils.cpp
    ${SIMPLE_ENGINE_DIR}/renderer_ray_query.cpp
    ${SIMPLE_ENGINE_DIR}/memory_pool.cpp
    ${SIMPLE_ENGINE_DIR}/resource_manager.cpp
    ${SIMPLE_ENGINE_DIR}/entity.cpp
    ${SIMPLE_ENGINE_DIR}/component.cpp
    ${SIMPLE_ENGINE_DIR}/transform_component.cpp
    ${SIMPLE_ENGINE_DIR}/mesh_component.cpp
    ${SIMPLE_ENGINE_DIR}/animation_component.cpp
    ${SIMPLE_ENGINE_DIR}/model_loader.cpp
    ${SIMPLE_ENGINE_DIR}/audio_system.cpp
    ${SIMPLE_ENGINE_DIR}/physics_system.cpp
    ${SIMPLE_ENGINE_DIR}/imgui_system.cpp
    ${SIMPLE_ENGINE_DIR}/imgui/imgui.cpp
    ${SIMPLE_ENGINE_DIR}/imgui/imgui_draw.cpp
    ${SIMPLE_ENGINE_DIR}/imgui/imgui_tables.cpp
    ${SIMPLE_ENGINE_DIR}/imgui/imgui_widgets.cpp
    ${SIMPLE_ENGINE_DIR}/vulkan_device.cpp
    ${SIMPLE_ENGINE_DIR}/pipeline.cpp
    ${SIMPLE_ENGINE_DIR}/descriptor_manager.cpp
    ${SIMPLE_ENGINE_DIR}/renderdoc_debug_system.cpp
    ${SIMPLE_ENGINE_DIR}/mikktspace.c
)

set(SOURCES_DESKTOP
    main.cpp
)

# Create target
if (ANDROID)
    add_library(OpenXREngine STATIC ${SOURCES_COMMON})
else ()
    add_executable(OpenXREngine ${SOURCES_COMMON} ${SOURCES_DESKTOP})
endif ()

add_dependencies(OpenXREngine shaders)
set_target_properties (OpenXREngine PROPERTIES CXX_STANDARD 20)

# Include directories
target_include_directories(OpenXREngine PRIVATE
    .
    ${SIMPLE_ENGINE_DIR}
    ${SIMPLE_ENGINE_DIR}/imgui
)

# Enable required defines
target_compile_definitions(OpenXREngine PRIVATE
    GLM_ENABLE_EXPERIMENTAL
    _USE_MATH_DEFINES
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
)

# Link libraries
target_link_libraries(OpenXREngine PUBLIC
    Vulkan::cppm
    glm::glm
    tinygltf::tinygltf
    KTX::ktx
    OpenXR::OpenXR
)

if (ANDROID)
    target_link_libraries(OpenXREngine PUBLIC game-activity::game-activity OpenSLES android log)
else ()
    target_link_libraries(OpenXREngine PRIVATE glfw OpenAL::OpenAL)
endif()

# Windows/MSVC portability and build settings
if(MSVC)
    target_compile_definitions(OpenXREngine PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
    target_compile_options(OpenXREngine PRIVATE
        /permissive-
        /Zc:__cplusplus
        /EHsc
        /W3
        /MP
        /bigobj
    )
    target_link_libraries(OpenXREngine PRIVATE Dbghelp)
elseif(WIN32)
    target_compile_definitions(OpenXREngine PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
endif()

# Copy Assets from Simple Engine
if(EXISTS ${SIMPLE_ENGINE_DIR}/Assets)
    if (NOT ANDROID)
    add_custom_command(TARGET OpenXREngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${SIMPLE_ENGINE_DIR}/Assets ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Assets
        COMMENT "Copying Assets from Simple Engine to output directory"
    )
endif()
endif ()

# Packaging configuration
include(CPack)
set(CPACK_PACKAGE_NAME "OpenXREngine")
set(CPACK_PACKAGE_VENDOR "OpenXREngine Team")
set(CPACK_PACKAGE_VERSION "1.0.0")

if (NOT ANDROID)
    install(TARGETS OpenXREngine DESTINATION bin)
    if(SLANGC_EXECUTABLE)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders DESTINATION share/OpenXREngine)
    endif()
    # Assets are copied from Simple Engine
    if(EXISTS ${SIMPLE_ENGINE_DIR}/Assets)
        install(DIRECTORY ${SIMPLE_ENGINE_DIR}/Assets DESTINATION share/OpenXREngine)
    endif()
endif ()
