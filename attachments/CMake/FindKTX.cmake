# FindKTX.cmake
# Finds the KTX library
#
# This module defines the following variables:
#
#    KTX_FOUND        - True if KTX was found
#    KTX_INCLUDE_DIRS - Include directories for KTX
#    KTX_LIBRARIES    - Libraries to link against KTX
#
# It also defines the following imported targets:
#    KTX::ktx
#

# Try to find the package using pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_KTX QUIET ktx libktx ktx2 libktx2)
endif()

# Find the include directory
find_path(KTX_INCLUDE_DIR
  NAMES ktx.h
  PATH_SUFFIXES include ktx KTX ktx2 KTX2
  HINTS
    ${PC_KTX_INCLUDEDIR}
    /usr/include
    /usr/local/include
    $ENV{KTX_DIR}/include
    $ENV{VULKAN_SDK}/include
    ${CMAKE_SOURCE_DIR}/external/ktx/include
)

# Find the library
find_library(KTX_LIBRARY
  NAMES ktx ktx2 libktx libktx2
  PATH_SUFFIXES lib lib64
  HINTS
    ${PC_KTX_LIBDIR}
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    $ENV{KTX_DIR}/lib
    $ENV{VULKAN_SDK}/lib
    ${CMAKE_SOURCE_DIR}/external/ktx/lib
)

if (NOT KTX_INCLUDE_DIR OR NOT KTX_LIBRARY)
  # If not found, use FetchContent to download and build
  include(FetchContent)

  message(STATUS "KTX not found, fetching from GitHub...")
  FetchContent_Declare(
    ktx
    GIT_REPOSITORY https://github.com/KhronosGroup/KTX-Software.git
    GIT_TAG v4.4.2  # Use a specific tag for stability
  )

  # Set options to minimize build time and dependencies
  set(KTX_FEATURE_TOOLS OFF CACHE BOOL "Build KTX tools" FORCE)
  set(KTX_FEATURE_DOC OFF CACHE BOOL "Build KTX documentation" FORCE)
  set(KTX_FEATURE_TESTS OFF CACHE BOOL "Build KTX tests" FORCE)

  FetchContent_MakeAvailable(ktx)
endif()

# Set the variables
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KTX
  REQUIRED_VARS KTX_INCLUDE_DIR KTX_LIBRARY
)

if(KTX_FOUND)
  set(KTX_INCLUDE_DIRS ${KTX_INCLUDE_DIR})
  set(KTX_LIBRARIES ${KTX_LIBRARY})

  if(NOT TARGET KTX::ktx)
    add_library(KTX::ktx UNKNOWN IMPORTED)
    set_target_properties(KTX::ktx PROPERTIES
      IMPORTED_LOCATION "${KTX_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${KTX_INCLUDE_DIRS}"
    )
  endif()
else()
  # Debug output if KTX is not found (fail-safe, should not be needed)
  message(STATUS "KTX include directory search paths: ${PC_KTX_INCLUDEDIR}, /usr/include, /usr/local/include, $ENV{KTX_DIR}/include, $ENV{VULKAN_SDK}/include, ${CMAKE_SOURCE_DIR}/external/ktx/include")
  message(STATUS "KTX library search paths: ${PC_KTX_LIBDIR}, /usr/lib, /usr/lib64, /usr/local/lib, /usr/local/lib64, $ENV{KTX_DIR}/lib, $ENV{VULKAN_SDK}/lib, ${CMAKE_SOURCE_DIR}/external/ktx/lib")
endif()

mark_as_advanced(KTX_INCLUDE_DIR KTX_LIBRARY)
