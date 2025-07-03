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
find_package_handle_standard_args(TinyGLTF
  REQUIRED_VARS TinyGLTF_INCLUDE_DIR
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
  # If not found, use FetchContent to download and build
  include(FetchContent)

  message(STATUS "TinyGLTF not found, fetching from GitHub...")
  FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.8.18  # Use a specific tag for stability
  )

  # Configure tinygltf before making it available
  FetchContent_GetProperties(tinygltf)
  if(NOT tinygltf_POPULATED)
    FetchContent_Populate(tinygltf)

    # Update the minimum required CMake version to avoid deprecation warning
    file(READ "${tinygltf_SOURCE_DIR}/CMakeLists.txt" TINYGLTF_CMAKE_CONTENT)
    string(REPLACE "cmake_minimum_required(VERSION 3.6)"
                   "cmake_minimum_required(VERSION 3.10)"
                   TINYGLTF_CMAKE_CONTENT "${TINYGLTF_CMAKE_CONTENT}")
    file(WRITE "${tinygltf_SOURCE_DIR}/CMakeLists.txt" "${TINYGLTF_CMAKE_CONTENT}")

    # Create a symbolic link to make nlohmann/json.hpp available
    if(EXISTS "${tinygltf_SOURCE_DIR}/json.hpp")
      file(MAKE_DIRECTORY "${tinygltf_SOURCE_DIR}/nlohmann")
      file(CREATE_LINK "${tinygltf_SOURCE_DIR}/json.hpp" "${tinygltf_SOURCE_DIR}/nlohmann/json.hpp" SYMBOLIC)
    endif()

    # Set tinygltf to header-only mode
    set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "Use header only version" FORCE)
    set(TINYGLTF_INSTALL OFF CACHE BOOL "Do not install tinygltf" FORCE)

    # Add the subdirectory after modifying the CMakeLists.txt
    add_subdirectory(${tinygltf_SOURCE_DIR} ${tinygltf_BINARY_DIR})
  endif()

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
