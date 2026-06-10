cmake_minimum_required(VERSION 3.29)

project(SimpleEngine VERSION 1.0.0 LANGUAGES CXX C)

# ─────────────────────────────────────────────────────────────────────────────
# Course modules — optional self-contained feature extensions
# Each course lives entirely in Courses/ and is linked into the engine target.
# Disable globally with:  cmake -DENABLE_COURSES=OFF ..
# ─────────────────────────────────────────────────────────────────────────────
option(ENABLE_COURSES "Build the Simple Engine course modules" OFF)

# Option to enable/disable Vulkan C++20 module support for this standalone project
option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan in SimpleEngine" OFF)

# Enable C++ module dependency scanning only when modules are enabled
if(ENABLE_CPP20_MODULE)
    set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

# Add CMake module path for custom find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

# Find required packages
find_package (glm REQUIRED)
find_package (Vulkan REQUIRED)
find_package (tinygltf REQUIRED)
find_package (KTX REQUIRED)

# Find or download Vulkan-Hpp headers matching the Vulkan SDK/NDK version
find_package(VulkanHpp REQUIRED)

if(ENABLE_CPP20_MODULE)
    # Set up Vulkan C++ module for this standalone project
    add_library(VulkanCppModule)
    add_library(Vulkan::cppm ALIAS VulkanCppModule)

    target_compile_definitions(VulkanCppModule
            PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    )
    # Build include list — fetched Vulkan-Headers must come before the SDK
    # include dir when the installed SDK is too old for VK_KHR_opacity_micromap.
    # This should be reverted when SDK 351 is released.
    set(_vkmod_incdirs)
    if (DEFINED VULKAN_FETCHED_HEADERS_INCLUDE AND EXISTS "${VULKAN_FETCHED_HEADERS_INCLUDE}")
        list(APPEND _vkmod_incdirs "${VULKAN_FETCHED_HEADERS_INCLUDE}")
    endif ()
    list(APPEND _vkmod_incdirs "${Vulkan_INCLUDE_DIR}" "${VulkanHpp_INCLUDE_DIRS}")
    target_include_directories(VulkanCppModule PUBLIC ${_vkmod_incdirs})
    unset(_vkmod_incdirs)
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
    # Same header-ordering logic as the module path: fetched Vulkan-Headers must
    # come before VulkanHpp_INCLUDE_DIRS so the newer vulkan_core.h is found
    # first when the installed SDK is too old for VK_KHR_opacity_micromap.
    # This section should be reverted when SDK 351 is released.
    set(_vkmod_incdirs)
    if (DEFINED VULKAN_FETCHED_HEADERS_INCLUDE AND EXISTS "${VULKAN_FETCHED_HEADERS_INCLUDE}")
        list(APPEND _vkmod_incdirs "${VULKAN_FETCHED_HEADERS_INCLUDE}")
    endif ()
    list(APPEND _vkmod_incdirs "${VulkanHpp_INCLUDE_DIRS}")
    target_include_directories(VulkanCppModule INTERFACE ${_vkmod_incdirs})
    unset(_vkmod_incdirs)
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

# Shader compilation
# Find Slang shaders (exclude utility modules that are imported, not compiled standalone)
file(GLOB SLANG_SHADER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang)
list(FILTER SLANG_SHADER_SOURCES EXCLUDE REGEX ".*/(common_types|pbr_utils|lighting_utils|tonemapping_utils)\\.slang$")

# Find slangc executable (optional)
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin)
find_program(SPIRV_OPT_EXECUTABLE spirv-opt HINTS $ENV{VULKAN_SDK}/bin)

if(SLANGC_EXECUTABLE)
    # Ensure the output directory for compiled shaders exists
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

    # Compile Slang shaders using slangc
    foreach(SHADER ${SLANG_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        get_filename_component(SHADER_NAME_WE ${SHADER_NAME} NAME_WE)
        string(REGEX REPLACE "\.slang$" "" OUTPUT_NAME ${SHADER_NAME})
        set(OUTPUT_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/${OUTPUT_NAME}.spv)

        set(SLANGC_FLAGS -target spirv -profile spirv_1_3 -g0 -O3)
        if (ANDROID)
            list(APPEND SLANGC_FLAGS -DPLATFORM_ANDROID=1)
        endif ()

        if (SPIRV_OPT_EXECUTABLE)
            add_custom_command(
                    OUTPUT ${OUTPUT_SPV}
                    COMMAND ${SLANGC_EXECUTABLE} ${SHADER} ${SLANGC_FLAGS} -o ${OUTPUT_SPV}.tmp
                    COMMAND ${SPIRV_OPT_EXECUTABLE} --strip-debug --strip-nonsemantic -Os ${OUTPUT_SPV}.tmp -o ${OUTPUT_SPV}
                    COMMAND ${CMAKE_COMMAND} -E remove ${OUTPUT_SPV}.tmp
                    DEPENDS ${SHADER}
                    COMMENT "Compiling and optimizing Slang shader ${SHADER_NAME}"
            )
        else ()
            add_custom_command(
                    OUTPUT ${OUTPUT_SPV}
                    COMMAND ${SLANGC_EXECUTABLE} ${SHADER} ${SLANGC_FLAGS} -o ${OUTPUT_SPV}
                    DEPENDS ${SHADER}
                    COMMENT "Compiling Slang shader ${SHADER_NAME} with slangc"
            )
        endif ()
        list(APPEND SHADER_SPVS ${OUTPUT_SPV})
    endforeach()

    add_custom_target(shaders DEPENDS ${SHADER_SPVS})

    # Copy shaders to the source directory's shaders folder so the app can find them when run from root
    add_custom_command(TARGET shaders POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_SPVS} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/
            COMMENT "Copying compiled shaders to source shaders directory"
    )

    if (ANDROID)
        # On Android, copy shaders to the Assets directory so Gradle can package them
        add_custom_command(TARGET shaders POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/Assets/shaders
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_SPVS} ${CMAKE_CURRENT_SOURCE_DIR}/Assets/shaders/
                COMMENT "Copying compiled shaders to Assets/shaders for Android packaging"
        )
    endif ()
else()
    message(STATUS "slangc not found. Skipping shader compilation step.")
    add_custom_target(shaders)
endif()

# Source files
# NOTE: Android builds include this project via `add_subdirectory(...)` from
# `android/app/src/main/cpp/CMakeLists.txt`, so we must not require a desktop `main()`.
set(SOURCES_COMMON
    engine.cpp
    scene_loading.cpp
    platform.cpp
    renderer_core.cpp
    renderer_rendering.cpp
    renderer_pipelines.cpp
    renderer_compute.cpp
    renderer_utils.cpp
    renderer_resources.cpp
    renderer_ray_query.cpp
    memory_pool.cpp
    resource_manager.cpp
    entity.cpp
    component.cpp
    transform_component.cpp
    mesh_component.cpp
    camera_component.cpp
    animation_component.cpp
    model_loader.cpp
    audio_system.cpp
    physics_system.cpp
    imgui_system.cpp
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    vulkan_device.cpp
    pipeline.cpp
    descriptor_manager.cpp
    renderdoc_debug_system.cpp
    mikktspace.c
)

set(SOURCES_DESKTOP
        main.cpp
)

# Create target
if (ANDROID)
    # Android: build the engine as a library to be linked into the app's `simple_engine_android` SHARED library.
    add_library(SimpleEngine STATIC ${SOURCES_COMMON} ${SOURCES_DESKTOP})
else ()
    # Desktop: build the runnable executable (unchanged behavior vs `HEAD`).
    add_executable(SimpleEngine ${SOURCES_COMMON} ${SOURCES_DESKTOP})
endif ()

add_dependencies(SimpleEngine shaders)
set_target_properties (SimpleEngine PROPERTIES CXX_STANDARD 20)

# ── Course modules ────────────────────────────────────────────────────────────
# Each course is a static library defined in Courses/CMakeLists.txt.
# The SIMPLE_ENGINE_COURSES_LIB variable is set in that file; it is empty when
# all course modules are disabled.
if(ENABLE_COURSES)
    add_subdirectory(Courses)
    if(SIMPLE_ENGINE_COURSES_LIB)
        target_link_libraries(SimpleEngine PUBLIC ${SIMPLE_ENGINE_COURSES_LIB})
        target_include_directories(SimpleEngine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Courses)
        message(STATUS "[SimpleEngine] Course modules linked: ${SIMPLE_ENGINE_COURSES_LIB}")
    endif()
endif()

# Enable required defines for GLM experimental extensions and MSVC math constants
target_compile_definitions(SimpleEngine PRIVATE
    GLM_ENABLE_EXPERIMENTAL
    _USE_MATH_DEFINES
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
)

# Link libraries
# Prefer the Vulkan C++ module target when available (configured at the parent level),
# but fall back to the standard Vulkan library otherwise.
if(TARGET Vulkan::cppm)
    target_link_libraries(SimpleEngine PUBLIC Vulkan::cppm)
else()
    target_link_libraries(SimpleEngine PUBLIC Vulkan::Vulkan)
endif()

target_link_libraries(SimpleEngine PUBLIC
    glm::glm
    tinygltf::tinygltf
    KTX::ktx
)

if (ANDROID)
    target_link_libraries(SimpleEngine PUBLIC game-activity::game-activity OpenSLES android log)
else ()
    target_link_libraries(SimpleEngine PRIVATE glfw OpenAL::OpenAL)
endif()

# Windows/MSVC portability and build settings
if(MSVC)
    # Avoid Windows.h macro pollution and CRT warnings; improve conformance and build perf
    target_compile_definitions(SimpleEngine PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
    target_compile_options(SimpleEngine PRIVATE
        /permissive-
        /Zc:__cplusplus
        /EHsc
        /W3
        /MP
        /bigobj
    )
    # Crash reporter uses Dbghelp; pragma should suffice, but make it explicit for clarity
    target_link_libraries(SimpleEngine PRIVATE Dbghelp)
elseif(WIN32)
    # Non-MSVC Windows toolchains (e.g., MinGW)
    target_compile_definitions(SimpleEngine PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
    )
endif()

# Copy model and texture files if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
    if (NOT ANDROID)
    add_custom_command(TARGET SimpleEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models
        COMMENT "Copying models to output directory"
    )
endif()
endif ()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/textures)
    if (NOT ANDROID)
    add_custom_command(TARGET SimpleEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/textures ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/textures
        COMMENT "Copying textures to output directory"
    )
endif()
endif ()

# Add packaging configuration
include(CPack)

# Set package properties
set(CPACK_PACKAGE_NAME "SimpleEngine")
set(CPACK_PACKAGE_VENDOR "SimpleEngine Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A simple game engine built with Vulkan")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "SimpleEngine")

# Set platform-specific package generators
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_PACKAGE_NAME "SimpleEngine")
    set(CPACK_NSIS_DISPLAY_NAME "SimpleEngine")
    set(CPACK_NSIS_HELP_LINK "https://github.com/yourusername/SimpleEngine")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/yourusername/SimpleEngine")
    set(CPACK_NSIS_CONTACT "your.email@example.com")
    set(CPACK_NSIS_MODIFY_PATH ON)
elseif(APPLE)
    set(CPACK_GENERATOR "ZIP;DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "SimpleEngine")
    set(CPACK_DMG_FORMAT "UDBZ")
else()
    set(CPACK_GENERATOR "ZIP;DEB")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Your Name <your.email@example.com>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "games")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libvulkan1, libglfw3, libglm-dev, libktx-dev")
endif()

# Include binary and resource directories in the package
if (NOT ANDROID)
install(TARGETS SimpleEngine DESTINATION bin)
if(SLANGC_EXECUTABLE)
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders DESTINATION share/SimpleEngine)
endif()

# Install models and textures if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models DESTINATION share/SimpleEngine)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/textures)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/textures DESTINATION share/SimpleEngine)
endif()

# Install README if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README.md DESTINATION share/SimpleEngine)
endif()
endif ()
