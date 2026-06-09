# FindVulkanHpp.cmake
#
# Finds or downloads the Vulkan-Hpp headers and Vulkan Profiles headers
#
# This will define the following variables
#
#    VulkanHpp_FOUND
#    VulkanHpp_INCLUDE_DIRS
#
# and the following imported targets
#
#    VulkanHpp::VulkanHpp
#

# Try to find the package using standard find_path
find_path(VulkanHpp_INCLUDE_DIR
  NAMES vulkan/vulkan.hpp
  PATHS
    ${Vulkan_INCLUDE_DIR}
    /usr/include
    /usr/local/include
    $ENV{VULKAN_SDK}/include
    ${ANDROID_NDK}/sources/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../include
)

# Also try to find vulkan.cppm
find_path(VulkanHpp_CPPM_DIR
  NAMES vulkan/vulkan.cppm
  PATHS
    ${Vulkan_INCLUDE_DIR}
    /usr/include
    /usr/local/include
    $ENV{VULKAN_SDK}/include
    ${ANDROID_NDK}/sources/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../include
)

# Try to find vulkan_profiles.hpp
find_path(VulkanProfiles_INCLUDE_DIR
  NAMES vulkan/vulkan_profiles.hpp
  PATHS
    ${Vulkan_INCLUDE_DIR}
    /usr/include
    /usr/local/include
    $ENV{VULKAN_SDK}/include
    ${ANDROID_NDK}/sources/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../external
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../include
)

# Function to extract Vulkan version from vulkan_core.h
function(extract_vulkan_version VULKAN_CORE_H_PATH OUTPUT_VERSION_TAG)
    if (NOT EXISTS ${VULKAN_CORE_H_PATH})
        set(${OUTPUT_VERSION_TAG} "v1.3.275" PARENT_SCOPE)
        return()
    endif ()
  # Extract the version information from vulkan_core.h
  file(STRINGS ${VULKAN_CORE_H_PATH} VULKAN_VERSION_MAJOR_LINE REGEX "^#define VK_VERSION_MAJOR")
  file(STRINGS ${VULKAN_CORE_H_PATH} VULKAN_VERSION_MINOR_LINE REGEX "^#define VK_VERSION_MINOR")
  file(STRINGS ${VULKAN_CORE_H_PATH} VULKAN_HEADER_VERSION_LINE REGEX "^#define VK_HEADER_VERSION")

  set(VERSION_TAG "v1.3.275") # Default fallback

  if(VULKAN_VERSION_MAJOR_LINE AND VULKAN_VERSION_MINOR_LINE AND VULKAN_HEADER_VERSION_LINE)
    string(REGEX REPLACE "^#define VK_VERSION_MAJOR[ \t]+([0-9]+).*$" "\\1" VULKAN_VERSION_MAJOR "${VULKAN_VERSION_MAJOR_LINE}")
    string(REGEX REPLACE "^#define VK_VERSION_MINOR[ \t]+([0-9]+).*$" "\\1" VULKAN_VERSION_MINOR "${VULKAN_VERSION_MINOR_LINE}")
    string(REGEX REPLACE "^#define VK_HEADER_VERSION[ \t]+([0-9]+).*$" "\\1" VULKAN_HEADER_VERSION "${VULKAN_HEADER_VERSION_LINE}")

    # Construct the version tag
    set(VERSION_TAG "v${VULKAN_VERSION_MAJOR}.${VULKAN_VERSION_MINOR}.${VULKAN_HEADER_VERSION}")
  else()
    # Alternative approach: look for VK_HEADER_VERSION_COMPLETE
    file(STRINGS ${VULKAN_CORE_H_PATH} VULKAN_HEADER_VERSION_COMPLETE_LINE REGEX "^#define VK_HEADER_VERSION_COMPLETE")
    file(STRINGS ${VULKAN_CORE_H_PATH} VULKAN_HEADER_VERSION_LINE REGEX "^#define VK_HEADER_VERSION")

    if(VULKAN_HEADER_VERSION_COMPLETE_LINE AND VULKAN_HEADER_VERSION_LINE)
      # Extract the header version
      string(REGEX REPLACE "^#define VK_HEADER_VERSION[ \t]+([0-9]+).*$" "\\1" VULKAN_HEADER_VERSION "${VULKAN_HEADER_VERSION_LINE}")

      # Check if the complete version line contains the major and minor versions
      if(VULKAN_HEADER_VERSION_COMPLETE_LINE MATCHES "VK_MAKE_API_VERSION\\(.*,[ \t]*([0-9]+),[ \t]*([0-9]+),[ \t]*VK_HEADER_VERSION\\)")
        set(VULKAN_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(VULKAN_VERSION_MINOR "${CMAKE_MATCH_2}")
        set(VERSION_TAG "v${VULKAN_VERSION_MAJOR}.${VULKAN_VERSION_MINOR}.${VULKAN_HEADER_VERSION}")
      endif()
    endif()
  endif()

  # Return the version tag
  set(${OUTPUT_VERSION_TAG} ${VERSION_TAG} PARENT_SCOPE)
endfunction()

# Determine the Vulkan version to use for Vulkan-Hpp and Vulkan-Profiles
set(VULKAN_VERSION_TAG "v1.3.275") # Default version

# Try to detect the Vulkan version
set(VULKAN_CORE_H "")

# If we're building for Android, try to detect the NDK's Vulkan version
if(DEFINED ANDROID_NDK)
  # Find the vulkan_core.h file in the NDK
  find_file(VULKAN_CORE_H vulkan_core.h
    PATHS
      ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/vulkan
      ${ANDROID_NDK}/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include/vulkan
      ${ANDROID_NDK}/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include/vulkan
      ${ANDROID_NDK}/toolchains/llvm/prebuilt/windows/sysroot/usr/include/vulkan
    NO_DEFAULT_PATH
  )

  if(VULKAN_CORE_H)
    extract_vulkan_version(${VULKAN_CORE_H} VULKAN_VERSION_TAG)
    message(STATUS "Detected NDK Vulkan version: ${VULKAN_VERSION_TAG}")
  else()
    message(STATUS "Could not find vulkan_core.h in NDK, using default version: ${VULKAN_VERSION_TAG}")
  endif()
# For desktop builds, try to detect the Vulkan SDK version
elseif(DEFINED ENV{VULKAN_SDK})
  # Find the vulkan_core.h file in the Vulkan SDK
  find_file(VULKAN_CORE_H vulkan_core.h
    PATHS
      $ENV{VULKAN_SDK}/include/vulkan
          $ENV{VULKAN_SDK}/x86_64/include/vulkan
    NO_DEFAULT_PATH
  )

  if(VULKAN_CORE_H)
    extract_vulkan_version(${VULKAN_CORE_H} VULKAN_VERSION_TAG)
    message(STATUS "Detected Vulkan SDK version: ${VULKAN_VERSION_TAG}")
  else()
    message(STATUS "Could not find vulkan_core.h in Vulkan SDK, using default version: ${VULKAN_VERSION_TAG}")
  endif()
# If Vulkan package was already found, try to use its include directory
elseif(DEFINED Vulkan_INCLUDE_DIR)
  # Find the vulkan_core.h file in the Vulkan include directory
  find_file(VULKAN_CORE_H vulkan_core.h
    PATHS
      ${Vulkan_INCLUDE_DIR}/vulkan
    NO_DEFAULT_PATH
  )

  if(VULKAN_CORE_H)
    extract_vulkan_version(${VULKAN_CORE_H} VULKAN_VERSION_TAG)
    message(STATUS "Detected Vulkan version from include directory: ${VULKAN_VERSION_TAG}")
  else()
    message(STATUS "Could not find vulkan_core.h in Vulkan include directory, using default version: ${VULKAN_VERSION_TAG}")
  endif()
else()
  # Try to find vulkan_core.h in system paths
  find_file(VULKAN_CORE_H vulkan_core.h
    PATHS
      /usr/include/vulkan
      /usr/local/include/vulkan
  )

  if(VULKAN_CORE_H)
    extract_vulkan_version(${VULKAN_CORE_H} VULKAN_VERSION_TAG)
    message(STATUS "Detected system Vulkan version: ${VULKAN_VERSION_TAG}")
  else()
    message(STATUS "Could not find vulkan_core.h in system paths, using default version: ${VULKAN_VERSION_TAG}")
  endif()
endif()

# Check if the detected version is less than 1.4.351
string(REGEX REPLACE "^v" "" VULKAN_VERSION_NUM "${VULKAN_VERSION_TAG}")
if (VULKAN_VERSION_NUM VERSION_LESS "1.4.351")
    message(STATUS "Vulkan version ${VULKAN_VERSION_NUM} is less than 1.4.351. Fetching latest Vulkan-Headers from git...")
    include(FetchContent)
    FetchContent_Declare(
            VulkanHeaders
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG main
    )
    FetchContent_Populate(VulkanHeaders)

    # Override Vulkan_INCLUDE_DIR to use the git headers
    set(Vulkan_INCLUDE_DIR "${vulkanheaders_SOURCE_DIR}/include" CACHE PATH "Path to Vulkan headers" FORCE)
    set(VULKAN_VERSION_TAG "main")

    # Force fetching of Vulkan-Hpp and Vulkan-Profiles
    set(VulkanHpp_INCLUDE_DIR "VulkanHpp_INCLUDE_DIR-NOTFOUND" CACHE PATH "" FORCE)
    set(VulkanHpp_CPPM_DIR "VulkanHpp_CPPM_DIR-NOTFOUND" CACHE PATH "" FORCE)
    set(VulkanProfiles_INCLUDE_DIR "VulkanProfiles_INCLUDE_DIR-NOTFOUND" CACHE PATH "" FORCE)

    message(STATUS "Using Vulkan-Headers from git: ${Vulkan_INCLUDE_DIR}")

    # Update the existing Vulkan::Headers target if it exists
    if (TARGET Vulkan::Headers)
        set_target_properties(Vulkan::Headers PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
        )
    endif ()
    if (TARGET Vulkan::Vulkan)
        set_target_properties(Vulkan::Vulkan PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
        )
    endif ()
endif ()

# ── Minimum version check for VK_KHR_opacity_micromap ─────────────────────────
# VkAccelerationStructureTrianglesOpacityMicromapKHR first appeared in Vulkan
# headers 1.4.351 (VK_KHR_opacity_micromap promotion from EXT to KHR).
# If the installed SDK/system headers are older, we fetch both Vulkan-Headers
# (C structs) and Vulkan-Hpp (C++ wrappers) from the GitHub main branch so the
# project can compile.  The fetched C headers are prepended to all include
# paths at the end of this file so they shadow the too-old SDK headers.
# This block of code should be removed once SDK 351 is released.
set(VULKAN_KHR_OMM_MIN_HEADER_VERSION 351)
set(VULKAN_HEADERS_SUFFICIENT FALSE)

if (VULKAN_CORE_H AND EXISTS "${VULKAN_CORE_H}")
    file(STRINGS "${VULKAN_CORE_H}" _vk_hdr_line REGEX "^#define VK_HEADER_VERSION ")
    file(STRINGS "${VULKAN_CORE_H}" _vk_cmp_line REGEX "^#define VK_HEADER_VERSION_COMPLETE")
    string(REGEX MATCH "[0-9]+" _vk_patch "${_vk_hdr_line}")
    if (_vk_cmp_line MATCHES "VK_MAKE_API_VERSION\\([^,]+,[ \t]*([0-9]+),[ \t]*([0-9]+),")
        set(_vk_major "${CMAKE_MATCH_1}")
        set(_vk_minor "${CMAKE_MATCH_2}")
    else ()
        set(_vk_major 0)
        set(_vk_minor 0)
    endif ()
    if ((_vk_major GREATER 1) OR
    (_vk_major EQUAL 1 AND _vk_minor GREATER 4) OR
    (_vk_major EQUAL 1 AND _vk_minor EQUAL 4 AND
            NOT _vk_patch LESS VULKAN_KHR_OMM_MIN_HEADER_VERSION))
        set(VULKAN_HEADERS_SUFFICIENT TRUE)
    endif ()
    message(STATUS "Installed Vulkan headers: ${_vk_major}.${_vk_minor}.${_vk_patch} — need >= 1.4.${VULKAN_KHR_OMM_MIN_HEADER_VERSION} for VK_KHR_opacity_micromap")
    unset(_vk_hdr_line)
    unset(_vk_cmp_line)
    unset(_vk_patch)
    unset(_vk_major)
    unset(_vk_minor)
else ()
    message(STATUS "Could not verify Vulkan header version — will fetch latest for VK_KHR_opacity_micromap safety")
endif ()

if (NOT VULKAN_HEADERS_SUFFICIENT)
    message(STATUS "Installed Vulkan headers too old for VK_KHR_opacity_micromap — fetching from GitHub...")

    include(FetchContent)
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif ()

    # ── Step 1: Fetch Vulkan-Hpp at main ──────────────────────────────────────
    # Use a distinct content name (VulkanHppMain, not VulkanHpp) so a stale
    # FetchContent cache entry from a prior run at an older versioned tag is
    # never silently reused here.
    FetchContent_Declare(
            VulkanHppMain
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Hpp.git
            GIT_TAG main
    )
    FetchContent_GetProperties(VulkanHppMain SOURCE_DIR VulkanHppMain_SOURCE_DIR)
    if (NOT VulkanHppMain_POPULATED)
        FetchContent_Populate(VulkanHppMain)
        FetchContent_GetProperties(VulkanHppMain SOURCE_DIR VulkanHppMain_SOURCE_DIR)
    endif ()
    message(STATUS "Fetched Vulkan-Hpp (main): ${VulkanHppMain_SOURCE_DIR}")

    # ── Step 2: Determine the exact VK_HEADER_VERSION Vulkan-Hpp expects ─────
    # vulkan.hpp contains a static_assert that pins the required C header
    # version, e.g.:
    #   static_assert( VK_HEADER_VERSION == 352, "Wrong VK_HEADER_VERSION!" );
    # Fetching both Vulkan-Hpp and Vulkan-Headers independently at 'main' races:
    # they advance separately and can diverge by one revision, triggering that
    # assertion at compile time.  Parse the expected version now so Vulkan-Headers
    # can be fetched at the matching versioned tag rather than at 'main'.
    set(_vk_headers_tag "main")
    set(_vk_headers_content_name "VulkanHeadersC_main")
    if (EXISTS "${VulkanHppMain_SOURCE_DIR}/vulkan/vulkan.hpp")
        file(STRINGS "${VulkanHppMain_SOURCE_DIR}/vulkan/vulkan.hpp" _assert_line
                REGEX "ASSERT.*VK_HEADER_VERSION ==")
        if (_assert_line MATCHES "==[ \t]*([0-9]+)")
            set(_vk_expected_patch "${CMAKE_MATCH_1}")
            set(_vk_headers_tag "v1.4.${_vk_expected_patch}")
            # Encode the patch version in the content name so FetchContent's
            # cache is invalidated automatically when Vulkan-Hpp advances to a
            # new header version — no manual cache wipe needed.
            set(_vk_headers_content_name "VulkanHeadersC_v1_4_${_vk_expected_patch}")
            message(STATUS "Vulkan-Hpp (main) asserts VK_HEADER_VERSION == ${_vk_expected_patch} — fetching Vulkan-Headers at ${_vk_headers_tag}")
            unset(_vk_expected_patch)
        else ()
            message(STATUS "Could not parse VK_HEADER_VERSION assertion from vulkan.hpp — falling back to Vulkan-Headers main")
        endif ()
        unset(_assert_line)
    endif ()

    # ── Step 3: Fetch Vulkan-Headers at the matched versioned tag ─────────────
    FetchContent_Declare(
            ${_vk_headers_content_name}
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG "${_vk_headers_tag}"
    )
    FetchContent_GetProperties(${_vk_headers_content_name} SOURCE_DIR _VulkanHeadersC_srcdir)
    if (NOT ${_vk_headers_content_name}_POPULATED)
        FetchContent_Populate(${_vk_headers_content_name})
        FetchContent_GetProperties(${_vk_headers_content_name} SOURCE_DIR _VulkanHeadersC_srcdir)
    endif ()
    # Vulkan-Headers lays out its headers under include/vulkan/
    set(VULKAN_FETCHED_HEADERS_INCLUDE "${_VulkanHeadersC_srcdir}/include")
    message(STATUS "Fetched Vulkan-Headers (${_vk_headers_tag}): ${VULKAN_FETCHED_HEADERS_INCLUDE}")
    unset(_vk_headers_tag)
    unset(_vk_headers_content_name)
    unset(_VulkanHeadersC_srcdir)

    # Set VulkanHpp_INCLUDE_DIR (FORCE cache) so the existing versioned-fetch
    # block below sees the directory as already satisfied and does not re-fetch
    # at the old SDK-matched tag.
    set(VulkanHpp_INCLUDE_DIR "${VulkanHppMain_SOURCE_DIR}" CACHE PATH
            "Vulkan-Hpp include directory (fetched at main for VK_KHR_opacity_micromap)" FORCE)
    if (EXISTS "${VulkanHppMain_SOURCE_DIR}/vulkan/vulkan.cppm")
        set(VulkanHpp_CPPM_DIR "${VulkanHppMain_SOURCE_DIR}" CACHE PATH
                "Vulkan-Hpp cppm directory (fetched at main)" FORCE)
    endif ()
endif ()
# ── End of minimum version check ──────────────────────────────────────────────

# If the include directory wasn't found, use FetchContent to download and build
if(NOT VulkanHpp_INCLUDE_DIR OR NOT VulkanHpp_CPPM_DIR)
  # If not found, use FetchContent to download
  include(FetchContent)

  message(STATUS "Vulkan-Hpp not found, fetching from GitHub with version ${VULKAN_VERSION_TAG}...")
  FetchContent_Declare(
    VulkanHpp
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Hpp.git
    GIT_TAG ${VULKAN_VERSION_TAG}  # Use the detected or default version
  )

  # Set policy to suppress the deprecation warning
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()

  # Make sure FetchContent is available
  include(FetchContent)

  # Populate the content
  FetchContent_GetProperties(VulkanHpp SOURCE_DIR VulkanHpp_SOURCE_DIR)
  if(NOT VulkanHpp_POPULATED)
    FetchContent_Populate(VulkanHpp)
    # Get the source directory after populating
    FetchContent_GetProperties(VulkanHpp SOURCE_DIR VulkanHpp_SOURCE_DIR)
  endif()

  # Set the include directory to the source directory
  set(VulkanHpp_INCLUDE_DIR ${VulkanHpp_SOURCE_DIR})
  # Ensure we also include the Vulkan-Headers if we fetched them
  if (vulkanheaders_SOURCE_DIR)
      list(APPEND VulkanHpp_INCLUDE_DIR "${vulkanheaders_SOURCE_DIR}/include")
  endif ()
  message(STATUS "VulkanHpp_SOURCE_DIR: ${VulkanHpp_SOURCE_DIR}")
  message(STATUS "VulkanHpp_INCLUDE_DIR: ${VulkanHpp_INCLUDE_DIR}")

  # Check if vulkan.cppm exists in the downloaded repository
  if(EXISTS "${VulkanHpp_SOURCE_DIR}/vulkan/vulkan.cppm")
    set(VulkanHpp_CPPM_DIR ${VulkanHpp_SOURCE_DIR})
  else()
    # If vulkan.cppm doesn't exist, we need to create it
      set(VulkanHpp_CPPM_DIR ${CMAKE_CURRENT_BINARY_DIR}/VulkanHpp)
      file(MAKE_DIRECTORY ${VulkanHpp_CPPM_DIR}/vulkan)

      # Create vulkan.cppm file
      file(WRITE "${VulkanHpp_CPPM_DIR}/vulkan/vulkan.cppm"
"// Auto-generated vulkan.cppm file
module;
#include <vulkan/vulkan.hpp>
export module vulkan;
export namespace vk {
  using namespace VULKAN_HPP_NAMESPACE;
}
")
  endif()
endif()

# If the Vulkan Profiles include directory wasn't found, use FetchContent to download
if(NOT VulkanProfiles_INCLUDE_DIR)
  # If not found, use FetchContent to download
  include(FetchContent)

  message(STATUS "Vulkan-Profiles not found, fetching from GitHub main branch...")
  FetchContent_Declare(
    VulkanProfiles
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Profiles.git
    GIT_TAG main  # Use main branch instead of a specific tag
  )

  # Set policy to suppress the deprecation warning
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()

  # Populate the content
  FetchContent_GetProperties(VulkanProfiles SOURCE_DIR VulkanProfiles_SOURCE_DIR)
  if(NOT VulkanProfiles_POPULATED)
    FetchContent_Populate(VulkanProfiles)
    # Get the source directory after populating
    FetchContent_GetProperties(VulkanProfiles SOURCE_DIR VulkanProfiles_SOURCE_DIR)
  endif()

  # Create the include directory structure if it doesn't exist
  set(VulkanProfiles_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/VulkanProfiles/include)
  file(MAKE_DIRECTORY ${VulkanProfiles_INCLUDE_DIR}/vulkan)

  # Create a stub vulkan_profiles.hpp file if it doesn't exist
  if(NOT EXISTS "${VulkanProfiles_INCLUDE_DIR}/vulkan/vulkan_profiles.hpp")
    file(WRITE "${VulkanProfiles_INCLUDE_DIR}/vulkan/vulkan_profiles.hpp"
"// Auto-generated vulkan_profiles.hpp stub file
#pragma once
#include <vulkan/vulkan.hpp>

namespace vp {
    // Stub implementation for Vulkan Profiles
    struct ProfileDesc {
        const char* name;
        uint32_t specVersion;
    };

    inline bool GetProfileSupport(VkPhysicalDevice physicalDevice, const ProfileDesc* pProfile, VkBool32* pSupported) {
        *pSupported = VK_TRUE;
        return true;
    }
}
")
  endif()

  message(STATUS "VulkanProfiles_SOURCE_DIR: ${VulkanProfiles_SOURCE_DIR}")
  message(STATUS "VulkanProfiles_INCLUDE_DIR: ${VulkanProfiles_INCLUDE_DIR}")
endif()

# Set the variables
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VulkanHpp
  REQUIRED_VARS VulkanHpp_INCLUDE_DIR
  FAIL_MESSAGE "Could NOT find VulkanHpp. Install it or set VulkanHpp_INCLUDE_DIR to the directory containing vulkan/vulkan.hpp"
)

# Debug output
message(STATUS "VulkanHpp_FOUND: ${VulkanHpp_FOUND}")
message(STATUS "VULKANHPP_FOUND: ${VULKANHPP_FOUND}")

if(VulkanHpp_FOUND)
  set(VulkanHpp_INCLUDE_DIRS ${VulkanHpp_INCLUDE_DIR})

  # Force git headers to the front to avoid conflicts with system headers
  if (vulkanheaders_SOURCE_DIR)
      include_directories(BEFORE SYSTEM "${vulkanheaders_SOURCE_DIR}/include")
  endif ()
  include_directories(BEFORE SYSTEM "${VulkanHpp_INCLUDE_DIR}")
  # The pinned C headers (VulkanHeadersC_v1_4_XXX) must shadow any unpinned
  # main-branch headers added above.  Add them last so BEFORE prepends them
  # to the front of the global include path, ahead of vulkanheaders_SOURCE_DIR.
  if (DEFINED VULKAN_FETCHED_HEADERS_INCLUDE AND EXISTS "${VULKAN_FETCHED_HEADERS_INCLUDE}")
      include_directories(BEFORE SYSTEM "${VULKAN_FETCHED_HEADERS_INCLUDE}")
  endif ()

  # Make sure VulkanHpp_CPPM_DIR is set
  if(NOT DEFINED VulkanHpp_CPPM_DIR)
    # Check if vulkan.cppm exists in the include directory
    if(EXISTS "${VulkanHpp_INCLUDE_DIR}/vulkan/vulkan.cppm")
      set(VulkanHpp_CPPM_DIR ${VulkanHpp_INCLUDE_DIR})
      message(STATUS "Found vulkan.cppm in VulkanHpp_INCLUDE_DIR: ${VulkanHpp_CPPM_DIR}")
    elseif(DEFINED VulkanHpp_SOURCE_DIR AND EXISTS "${VulkanHpp_SOURCE_DIR}/vulkan/vulkan.cppm")
      set(VulkanHpp_CPPM_DIR ${VulkanHpp_SOURCE_DIR})
      message(STATUS "Found vulkan.cppm in VulkanHpp_SOURCE_DIR: ${VulkanHpp_CPPM_DIR}")
    elseif(DEFINED vulkanhpp_SOURCE_DIR AND EXISTS "${vulkanhpp_SOURCE_DIR}/vulkan/vulkan.cppm")
      set(VulkanHpp_CPPM_DIR ${vulkanhpp_SOURCE_DIR})
      message(STATUS "Found vulkan.cppm in vulkanhpp_SOURCE_DIR: ${VulkanHpp_CPPM_DIR}")
    else()
      # If vulkan.cppm doesn't exist, we need to create it
      set(VulkanHpp_CPPM_DIR ${CMAKE_CURRENT_BINARY_DIR}/VulkanHpp)
      file(MAKE_DIRECTORY ${VulkanHpp_CPPM_DIR}/vulkan)
      message(STATUS "Creating vulkan.cppm in ${VulkanHpp_CPPM_DIR}")

      # Create vulkan.cppm file
      file(WRITE "${VulkanHpp_CPPM_DIR}/vulkan/vulkan.cppm"
"// Auto-generated vulkan.cppm file
module;
#include <vulkan/vulkan.hpp>
export module vulkan;
export namespace vk {
  using namespace VULKAN_HPP_NAMESPACE;
}
")
    endif()
  endif()

  message(STATUS "Final VulkanHpp_CPPM_DIR: ${VulkanHpp_CPPM_DIR}")

  # Add Vulkan Profiles include directory if found
  if(VulkanProfiles_INCLUDE_DIR AND EXISTS "${VulkanProfiles_INCLUDE_DIR}/vulkan/vulkan_profiles.hpp")
    list(APPEND VulkanHpp_INCLUDE_DIRS ${VulkanProfiles_INCLUDE_DIR})
    message(STATUS "Added Vulkan Profiles include directory: ${VulkanProfiles_INCLUDE_DIR}")
  endif()

  # Create an imported target
  if(NOT TARGET VulkanHpp::VulkanHpp)
    add_library(VulkanHpp::VulkanHpp INTERFACE IMPORTED)
    set_target_properties(VulkanHpp::VulkanHpp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${VulkanHpp_INCLUDE_DIRS}"
    )
  endif()
elseif(DEFINED VulkanHpp_SOURCE_DIR OR DEFINED vulkanhpp_SOURCE_DIR)
  # If find_package_handle_standard_args failed but we have a VulkanHpp source directory from FetchContent
  # Create an imported target
  if(NOT TARGET VulkanHpp::VulkanHpp)
    add_library(VulkanHpp::VulkanHpp INTERFACE IMPORTED)

    # Determine the source directory
    if(DEFINED VulkanHpp_SOURCE_DIR)
      set(_vulkanhpp_source_dir ${VulkanHpp_SOURCE_DIR})
    elseif(DEFINED vulkanhpp_SOURCE_DIR)
      set(_vulkanhpp_source_dir ${vulkanhpp_SOURCE_DIR})
    endif()

    message(STATUS "Using fallback VulkanHpp source directory: ${_vulkanhpp_source_dir}")

    set_target_properties(VulkanHpp::VulkanHpp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_vulkanhpp_source_dir}"
    )
  endif()

  # Set variables to indicate that VulkanHpp was found
  set(VulkanHpp_FOUND TRUE)
  set(VULKANHPP_FOUND TRUE)

  # Set include directories
  if(DEFINED _vulkanhpp_source_dir)
    set(VulkanHpp_INCLUDE_DIR ${_vulkanhpp_source_dir})
  elseif(DEFINED VulkanHpp_SOURCE_DIR)
    set(VulkanHpp_INCLUDE_DIR ${VulkanHpp_SOURCE_DIR})
  elseif(DEFINED vulkanhpp_SOURCE_DIR)
    set(VulkanHpp_INCLUDE_DIR ${vulkanhpp_SOURCE_DIR})
  endif()
  set(VulkanHpp_INCLUDE_DIRS ${VulkanHpp_INCLUDE_DIR})

  # Add Vulkan Profiles include directory if found
  if(VulkanProfiles_INCLUDE_DIR AND EXISTS "${VulkanProfiles_INCLUDE_DIR}/vulkan/vulkan_profiles.hpp")
    list(APPEND VulkanHpp_INCLUDE_DIRS ${VulkanProfiles_INCLUDE_DIR})
    message(STATUS "Added Vulkan Profiles include directory to fallback: ${VulkanProfiles_INCLUDE_DIR}")
  endif()

  # Make sure VulkanHpp_CPPM_DIR is set
  if(NOT DEFINED VulkanHpp_CPPM_DIR)
    # Check if vulkan.cppm exists in the downloaded repository
    if(DEFINED VulkanHpp_INCLUDE_DIR AND EXISTS "${VulkanHpp_INCLUDE_DIR}/vulkan/vulkan.cppm")
      set(VulkanHpp_CPPM_DIR ${VulkanHpp_INCLUDE_DIR})
      message(STATUS "Found vulkan.cppm in VulkanHpp_INCLUDE_DIR: ${VulkanHpp_CPPM_DIR}")
    elseif(DEFINED _vulkanhpp_source_dir AND EXISTS "${_vulkanhpp_source_dir}/vulkan/vulkan.cppm")
      set(VulkanHpp_CPPM_DIR ${_vulkanhpp_source_dir})
      message(STATUS "Found vulkan.cppm in _vulkanhpp_source_dir: ${VulkanHpp_CPPM_DIR}")
    elseif(DEFINED VulkanHpp_SOURCE_DIR AND EXISTS "${VulkanHpp_SOURCE_DIR}/vulkan/vulkan.cppm")
      set(VulkanHpp_CPPM_DIR ${VulkanHpp_SOURCE_DIR})
      message(STATUS "Found vulkan.cppm in VulkanHpp_SOURCE_DIR: ${VulkanHpp_CPPM_DIR}")
    elseif(DEFINED vulkanhpp_SOURCE_DIR AND EXISTS "${vulkanhpp_SOURCE_DIR}/vulkan/vulkan.cppm")
      set(VulkanHpp_CPPM_DIR ${vulkanhpp_SOURCE_DIR})
      message(STATUS "Found vulkan.cppm in vulkanhpp_SOURCE_DIR: ${VulkanHpp_CPPM_DIR}")
    else()
      # If vulkan.cppm doesn't exist, we need to create it
      set(VulkanHpp_CPPM_DIR ${CMAKE_CURRENT_BINARY_DIR}/VulkanHpp)
      file(MAKE_DIRECTORY ${VulkanHpp_CPPM_DIR}/vulkan)
      message(STATUS "Creating vulkan.cppm in ${VulkanHpp_CPPM_DIR}")

      # Create vulkan.cppm file
      file(WRITE "${VulkanHpp_CPPM_DIR}/vulkan/vulkan.cppm"
"// Auto-generated vulkan.cppm file
module;
#include <vulkan/vulkan.hpp>
export module vulkan;
export namespace vk {
  using namespace VULKAN_HPP_NAMESPACE;
}
")
    endif()
  endif()

  message(STATUS "Final VulkanHpp_CPPM_DIR: ${VulkanHpp_CPPM_DIR}")
endif()

# ── Prepend fetched C headers so they shadow the too-old SDK headers ──────────
# This must run after all the VulkanHpp_INCLUDE_DIRS / target setup above so
# that we can insert at position 0 regardless of which code path ran.
# This section should be removed when SDK 351 is released.
if (DEFINED VULKAN_FETCHED_HEADERS_INCLUDE AND EXISTS "${VULKAN_FETCHED_HEADERS_INCLUDE}")
    if (DEFINED VulkanHpp_INCLUDE_DIRS)
        list(INSERT VulkanHpp_INCLUDE_DIRS 0 "${VULKAN_FETCHED_HEADERS_INCLUDE}")
    else ()
        set(VulkanHpp_INCLUDE_DIRS "${VULKAN_FETCHED_HEADERS_INCLUDE}")
    endif ()
    # Update the imported target so any consumer that links VulkanHpp::VulkanHpp
    # also gets the fetched headers first in its include path.
    if (TARGET VulkanHpp::VulkanHpp)
        set_target_properties(VulkanHpp::VulkanHpp PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${VulkanHpp_INCLUDE_DIRS}")
    endif ()
    # Patch the Vulkan::Headers / Vulkan::Vulkan targets that FindVulkan.cmake
    # created — they point to the old SDK, so prepend the fetched headers there
    # too so that direct users of those targets also see the newer definitions.
    foreach (_vk_tgt Vulkan::Headers Vulkan::Vulkan)
        if (TARGET "${_vk_tgt}")
            get_target_property(_vk_incdirs "${_vk_tgt}" INTERFACE_INCLUDE_DIRECTORIES)
            if (_vk_incdirs)
                list(INSERT _vk_incdirs 0 "${VULKAN_FETCHED_HEADERS_INCLUDE}")
            else ()
                set(_vk_incdirs "${VULKAN_FETCHED_HEADERS_INCLUDE}")
            endif ()
            set_target_properties("${_vk_tgt}" PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${_vk_incdirs}")
        endif ()
    endforeach ()
    unset(_vk_tgt)
    unset(_vk_incdirs)
    message(STATUS "Prepended fetched Vulkan-Headers to all include paths for VK_KHR_opacity_micromap")
endif ()
# ── End of fetched-headers prepend ────────────────────────────────────────────

mark_as_advanced(VulkanHpp_INCLUDE_DIR VulkanHpp_CPPM_DIR)
