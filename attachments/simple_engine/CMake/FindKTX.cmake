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

# Try to find KTX using pkg-config first
find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(PC_KTX QUIET ktx libktx ktx2 libktx2)
endif ()

# Try to find KTX using standard find_path and find_library
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

# If not found in the system, use FetchContent to download and build
if (NOT KTX_INCLUDE_DIR OR NOT KTX_LIBRARY)
    include(FetchContent)

    message(STATUS "KTX not found, fetching from GitHub...")

    FetchContent_Declare(
            ktx
            GIT_REPOSITORY https://github.com/KhronosGroup/KTX-Software.git
            GIT_TAG v4.4.2
    )

    # Minimize build time and dependencies
    set(KTX_FEATURE_TOOLS OFF CACHE BOOL "Build KTX tools" FORCE)
    set(KTX_FEATURE_DOC OFF CACHE BOOL "Build KTX documentation" FORCE)
    set(KTX_FEATURE_TESTS OFF CACHE BOOL "Build KTX tests" FORCE)
    set(KTX_FEATURE_STATIC_LIBRARY ON CACHE BOOL "Build KTX as static library" FORCE)

    FetchContent_MakeAvailable(ktx)

    if (TARGET ktx)
        get_target_property(KTX_TARGET_INCLUDE_DIR ktx INTERFACE_INCLUDE_DIRECTORIES)
        if (KTX_TARGET_INCLUDE_DIR)
            set(KTX_INCLUDE_DIR ${KTX_TARGET_INCLUDE_DIR})
        else ()
            FetchContent_GetProperties(ktx SOURCE_DIR ktx_SOURCE_DIR)
            set(KTX_INCLUDE_DIR ${ktx_SOURCE_DIR}/include)
        endif ()
        set(KTX_LIBRARY ktx)
    endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KTX
        REQUIRED_VARS KTX_INCLUDE_DIR KTX_LIBRARY
)

if (KTX_FOUND)
    set(KTX_INCLUDE_DIRS ${KTX_INCLUDE_DIR})
    set(KTX_LIBRARIES ${KTX_LIBRARY})

    if (NOT TARGET KTX::ktx)
        if (TARGET ktx)
            # FetchContent target already exists — alias it
            add_library(KTX::ktx ALIAS ktx)
        else ()
            add_library(KTX::ktx UNKNOWN IMPORTED)
            set_target_properties(KTX::ktx PROPERTIES
                    IMPORTED_LOCATION "${KTX_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${KTX_INCLUDE_DIR}"
            )
        endif ()
    endif ()
endif ()

mark_as_advanced(KTX_INCLUDE_DIR KTX_LIBRARY)
