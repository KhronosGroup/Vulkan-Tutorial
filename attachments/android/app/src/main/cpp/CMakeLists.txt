cmake_minimum_required(VERSION 3.22.1)

project(vulkan_tutorial_android)

# Set the path to the main CMakeLists.txt relative to this file
set(MAIN_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../CMakeLists.txt")

# Find the Vulkan package
find_package(Vulkan REQUIRED)

# Find the tinyobjloader package
find_package(tinyobjloader REQUIRED)

# Find the glm package
find_package(glm REQUIRED)

# Find the tinygltf package (needed for 35_gltf_ktx)
find_package(tinygltf REQUIRED)

# Find the KTX package (needed for 35_gltf_ktx)
find_package(KTX REQUIRED)

# Find the stb_image.h header
find_path(STB_INCLUDEDIR stb_image.h PATH_SUFFIXES stb)
if (NOT STB_INCLUDEDIR)
  message(FATAL_ERROR "stb_image.h not found")
endif()

# Set up shader compilation tools
add_executable(glslang::validator IMPORTED)
find_program(GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
set_property(TARGET glslang::validator PROPERTY IMPORTED_LOCATION "${GLSLANG_VALIDATOR}")

# Define shader building function
function(add_shaders_target TARGET)
  cmake_parse_arguments("SHADER" "" "CHAPTER_NAME" "SOURCES" ${ARGN})
  set(SHADERS_DIR ${SHADER_CHAPTER_NAME}/shaders)
  add_custom_command(
    OUTPUT ${SHADERS_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )
  add_custom_command(
    OUTPUT ${SHADERS_DIR}/frag.spv ${SHADERS_DIR}/vert.spv
    COMMAND glslang::validator
    ARGS --target-env vulkan1.0 ${SHADER_SOURCES} --quiet
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Shaders"
    VERBATIM
  )
  add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/frag.spv ${SHADERS_DIR}/vert.spv)
endfunction()

# Include the game-activity library
find_package(game-activity REQUIRED CONFIG)
include_directories(${ANDROID_NDK}/sources/android/game-activity/include)

# Set C++ standard to match the main project
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the Vulkan C++ module
add_library(VulkanCppModule SHARED)
target_compile_definitions(VulkanCppModule
    PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)
target_include_directories(VulkanCppModule
    PRIVATE
    "${Vulkan_INCLUDE_DIR}"
)
target_link_libraries(VulkanCppModule
    PUBLIC
    ${Vulkan_LIBRARIES}
)
set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

# Set up the C++ module file set
target_sources(VulkanCppModule
    PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS
    "${Vulkan_INCLUDE_DIR}"
    FILES
    "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
)

# Set up shader compilation for all chapters
set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# Copy shader source files to the build directory
configure_file(
    "${SHADER_SOURCE_DIR}/27_shader_depth.frag"
    "${SHADER_OUTPUT_DIR}/27_shader_depth.frag"
    COPYONLY
)
configure_file(
    "${SHADER_SOURCE_DIR}/27_shader_depth.vert"
    "${SHADER_OUTPUT_DIR}/27_shader_depth.vert"
    COPYONLY
)

# Compile shaders
set(SHADER_SOURCES "${SHADER_OUTPUT_DIR}/27_shader_depth.frag" "${SHADER_OUTPUT_DIR}/27_shader_depth.vert")
add_shaders_target(vulkan_tutorial_shaders CHAPTER_NAME "${SHADER_OUTPUT_DIR}" SOURCES ${SHADER_SOURCES})

# Set default chapter if not provided
if(NOT DEFINED CHAPTER)
    set(CHAPTER "34_android")
endif()

# Define the list of supported chapters
set(SUPPORTED_CHAPTERS
    "34_android"
    "35_gltf_ktx"
)

# Validate the chapter
list(FIND SUPPORTED_CHAPTERS ${CHAPTER} CHAPTER_INDEX)
if(CHAPTER_INDEX EQUAL -1)
    message(FATAL_ERROR "Invalid chapter: ${CHAPTER}. Supported chapters are: ${SUPPORTED_CHAPTERS}")
endif()

message(STATUS "Building chapter: ${CHAPTER}")

# Add the main native library
add_library(vulkan_tutorial_android SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/${CHAPTER}.cpp
    game_activity_bridge.cpp
)

# Add dependency on shader compilation
add_dependencies(vulkan_tutorial_android vulkan_tutorial_shaders)

# Set include directories
target_include_directories(vulkan_tutorial_android PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Vulkan_INCLUDE_DIR}
    ${ANDROID_NDK}/sources/android/game-activity/include
    ${STB_INCLUDEDIR}
)

# Link against libraries
target_link_libraries(vulkan_tutorial_android
    VulkanCppModule
    game-activity::game-activity
    android
    log
    ${Vulkan_LIBRARIES}
    glm::glm
)

# Add chapter-specific libraries and definitions
if(CHAPTER STREQUAL "34_android")
    target_link_libraries(vulkan_tutorial_android
        tinyobjloader::tinyobjloader
    )
elseif(CHAPTER STREQUAL "35_gltf_ktx")
    target_link_libraries(vulkan_tutorial_android
        tinygltf::tinygltf
        KTX::ktx
    )

    # Add necessary definitions for tinygltf and KTX
    target_compile_definitions(vulkan_tutorial_android PRIVATE
        TINYGLTF_IMPLEMENTATION
        TINYGLTF_NO_EXTERNAL_IMAGE
        TINYGLTF_NO_STB_IMAGE
        TINYGLTF_NO_STB_IMAGE_WRITE
        KTX_FEATURE_KTX1
        KTX_FEATURE_KTX2
        KTX_FEATURE_WRITE
    )
endif()
