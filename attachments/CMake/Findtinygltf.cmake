# FindTinyGLTF.cmake
#
# Finds the TinyGLTF library
#
# This will define the following variables
#
#    TinyGLTF_FOUND
#    TinyGLTF_INCLUDE_DIRS
#
# and the following imported targets
#
#    tinygltf::tinygltf
#

# First, try to find nlohmann_json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  message(STATUS "nlohmann_json not found, fetching from GitHub...")
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2  # Use a specific tag for stability
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# Try to find TinyGLTF using standard find_package
find_path(TinyGLTF_INCLUDE_DIR
  NAMES tiny_gltf.h
  PATH_SUFFIXES include tinygltf
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tinygltf
  REQUIRED_VARS TinyGLTF_INCLUDE_DIR
  FAIL_MESSAGE ""  # Suppress the error message to allow our fallback
)

if(TinyGLTF_FOUND)
  set(TinyGLTF_INCLUDE_DIRS ${TinyGLTF_INCLUDE_DIR})

  if(NOT TARGET tinygltf::tinygltf)
    add_library(tinygltf::tinygltf INTERFACE IMPORTED)
    set_target_properties(tinygltf::tinygltf PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TinyGLTF_INCLUDE_DIRS}"
      INTERFACE_COMPILE_DEFINITIONS "TINYGLTF_IMPLEMENTATION;TINYGLTF_NO_EXTERNAL_IMAGE;TINYGLTF_NO_STB_IMAGE;TINYGLTF_NO_STB_IMAGE_WRITE"
    )
    if(TARGET nlohmann_json::nlohmann_json)
      target_link_libraries(tinygltf::tinygltf INTERFACE nlohmann_json::nlohmann_json)
    endif()
  endif()
else()
  # If not found, create a custom tinygltf implementation
  message(STATUS "TinyGLTF not found, creating a custom implementation...")

  # Create a directory for our custom tinygltf implementation
  set(TINYGLTF_DIR "${CMAKE_CURRENT_BINARY_DIR}/tinygltf")
  file(REMOVE_RECURSE "${TINYGLTF_DIR}")
  file(MAKE_DIRECTORY "${TINYGLTF_DIR}")

  # Download the necessary files directly
  file(DOWNLOAD
    "https://raw.githubusercontent.com/syoyo/tinygltf/v2.8.18/tiny_gltf.h"
    "${TINYGLTF_DIR}/tiny_gltf.h"
    SHOW_PROGRESS
  )

  file(DOWNLOAD
    "https://raw.githubusercontent.com/syoyo/tinygltf/v2.8.18/json.hpp"
    "${TINYGLTF_DIR}/json.hpp"
    SHOW_PROGRESS
  )

  file(DOWNLOAD
    "https://raw.githubusercontent.com/syoyo/tinygltf/v2.8.18/stb_image.h"
    "${TINYGLTF_DIR}/stb_image.h"
    SHOW_PROGRESS
  )

  file(DOWNLOAD
    "https://raw.githubusercontent.com/syoyo/tinygltf/v2.8.18/stb_image_write.h"
    "${TINYGLTF_DIR}/stb_image_write.h"
    SHOW_PROGRESS
  )

  # Create a symbolic link to make nlohmann/json.hpp available
  file(MAKE_DIRECTORY "${TINYGLTF_DIR}/nlohmann")
  file(CREATE_LINK "${TINYGLTF_DIR}/json.hpp" "${TINYGLTF_DIR}/nlohmann/json.hpp" SYMBOLIC)

  # Create a simple CMakeLists.txt file
  file(WRITE "${TINYGLTF_DIR}/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.10...3.29)
project(tinygltf)

if(NOT TARGET tinygltf)
  add_library(tinygltf INTERFACE)
  target_include_directories(tinygltf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(tinygltf INTERFACE
    TINYGLTF_IMPLEMENTATION
    TINYGLTF_NO_EXTERNAL_IMAGE
    TINYGLTF_NO_STB_IMAGE
    TINYGLTF_NO_STB_IMAGE_WRITE
  )
endif()
")

  # Add the subdirectory
  add_subdirectory(${TINYGLTF_DIR} ${CMAKE_CURRENT_BINARY_DIR}/tinygltf-build)

  # Create an alias for the tinygltf target
  if(NOT TARGET tinygltf::tinygltf)
    add_library(tinygltf_wrapper INTERFACE)
    target_link_libraries(tinygltf_wrapper INTERFACE tinygltf)
    target_compile_definitions(tinygltf_wrapper INTERFACE
      TINYGLTF_IMPLEMENTATION
      TINYGLTF_NO_EXTERNAL_IMAGE
      TINYGLTF_NO_STB_IMAGE
      TINYGLTF_NO_STB_IMAGE_WRITE
    )
    if(TARGET nlohmann_json::nlohmann_json)
      target_link_libraries(tinygltf_wrapper INTERFACE nlohmann_json::nlohmann_json)
    endif()
    add_library(tinygltf::tinygltf ALIAS tinygltf_wrapper)
  endif()

  set(TinyGLTF_FOUND TRUE)
endif()
