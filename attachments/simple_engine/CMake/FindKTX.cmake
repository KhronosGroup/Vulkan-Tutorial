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

# Try to find the package using CONFIG mode first (e.g., from vcpkg)
if(COMMAND _find_package)
  _find_package(ktx CONFIG QUIET)
else()
  find_package(ktx CONFIG QUIET)
endif()

if(ktx_FOUND)
  if(NOT TARGET KTX::ktx AND TARGET ktx::ktx)
    add_library(KTX::ktx ALIAS ktx::ktx)
  endif()
  set(KTX_FOUND TRUE)
  return()
endif()

# Check if we're on Linux - if so, we'll skip the search and directly use FetchContent
if(UNIX AND NOT APPLE)
  # On Linux, we assume KTX is not installed and proceed directly to fetching it
  set(KTX_FOUND FALSE)
else()
  # On non-Linux platforms, try to find KTX using pkg-config first
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_KTX QUIET ktx libktx ktx2 libktx2)
  endif()

  # Try to find KTX using standard find_path/find_library
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

  if(KTX_INCLUDE_DIR AND KTX_LIBRARY)
    set(KTX_FOUND TRUE)
  else()
    set(KTX_FOUND FALSE)
  endif()
endif()

# If not found, use FetchContent to download and build
if(NOT KTX_FOUND)
  include(FetchContent)

  # Only show the message on non-Linux platforms (on Linux we expect to fetch)
  if(NOT (UNIX AND NOT APPLE))
    message(STATUS "KTX not found, fetching from GitHub...")
  endif()

  FetchContent_Declare(
    ktx
    GIT_REPOSITORY https://github.com/KhronosGroup/KTX-Software.git
    GIT_TAG v4.3.1  # Use a specific tag for stability
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

# Finalize the variables and targets
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KTX
  REQUIRED_VARS KTX_FOUND
)

if(KTX_FOUND)
  if(KTX_INCLUDE_DIR AND KTX_LIBRARY)
    set(KTX_INCLUDE_DIRS ${KTX_INCLUDE_DIR})
    set(KTX_LIBRARIES ${KTX_LIBRARY})

    if(NOT TARGET KTX::ktx)
      add_library(KTX::ktx UNKNOWN IMPORTED)
      set_target_properties(KTX::ktx PROPERTIES
        IMPORTED_LOCATION "${KTX_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${KTX_INCLUDE_DIRS}"
      )
    endif()
  endif()
endif()
