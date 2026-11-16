# FindKTX.cmake
#
# Finds the KTX library
#
# This will define the following variables
#
#    KTX_FOUND
#    KTX_INCLUDE_DIRS
#    KTX_LIBRARIES
#
# and the following imported targets
#
#    KTX::ktx
#

# Try to find an already-installed KTX using pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_KTX QUIET ktx libktx ktx2 libktx2)
endif()

# Try to find KTX using standard find_package
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

include(FindPackageHandleStandardArgs)
# Run the standard args check as non-REQUIRED/QUIET so a missing KTX does not
# abort configuration here - we still want to fall back to FetchContent below.
set(_ktx_saved_find_required ${KTX_FIND_REQUIRED})
set(_ktx_saved_find_quietly ${KTX_FIND_QUIETLY})
set(KTX_FIND_REQUIRED FALSE)
set(KTX_FIND_QUIETLY TRUE)
find_package_handle_standard_args(KTX
  REQUIRED_VARS KTX_INCLUDE_DIR KTX_LIBRARY
)
set(KTX_FIND_REQUIRED ${_ktx_saved_find_required})
set(KTX_FIND_QUIETLY ${_ktx_saved_find_quietly})
unset(_ktx_saved_find_required)
unset(_ktx_saved_find_quietly)

if(NOT KTX_FOUND)
  message(STATUS "KTX include directory search paths: ${PC_KTX_INCLUDEDIR}, /usr/include, /usr/local/include, $ENV{KTX_DIR}/include, $ENV{VULKAN_SDK}/include, ${CMAKE_SOURCE_DIR}/external/ktx/include")
  message(STATUS "KTX library search paths: ${PC_KTX_LIBDIR}, /usr/lib, /usr/lib64, /usr/local/lib, /usr/local/lib64, $ENV{KTX_DIR}/lib, $ENV{VULKAN_SDK}/lib, ${CMAKE_SOURCE_DIR}/external/ktx/lib")
endif()

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

  # Create an alias to match the expected target name
  if(NOT TARGET KTX::ktx)
    add_library(KTX::ktx ALIAS ktx)
  endif()

  set(KTX_FOUND TRUE)
endif()
