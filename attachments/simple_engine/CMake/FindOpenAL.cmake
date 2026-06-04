# FindOpenAL.cmake
#
# Finds the OpenAL library
#
# This will define the following variables:
#
#    OpenAL_FOUND
#    OPENAL_INCLUDE_DIR
#    OPENAL_LIBRARY
#
# and the following imported targets:
#
#    OpenAL::OpenAL
#

# Try to find OpenAL using CONFIG mode first
find_package(OpenAL CONFIG QUIET)

if (OpenAL_FOUND)
    if (NOT TARGET OpenAL::OpenAL)
        if (TARGET OpenAL)
            add_library(OpenAL::OpenAL ALIAS OpenAL)
        endif ()
    endif ()

    # Set variables for find_package_handle_standard_args
    if (NOT OPENAL_INCLUDE_DIR)
        get_target_property(OPENAL_INCLUDE_DIR OpenAL::OpenAL INTERFACE_INCLUDE_DIRECTORIES)
    endif ()
    if (NOT OPENAL_LIBRARY)
        set(OPENAL_LIBRARY OpenAL)
    endif ()
endif ()

if (NOT OpenAL_FOUND)
    # Try to find using standard find_path/find_library
    find_path(OPENAL_INCLUDE_DIR NAMES AL/al.h OpenAL/al.h)
    find_library(OPENAL_LIBRARY NAMES OpenAL al openal OpenAL32)

    if (OPENAL_INCLUDE_DIR AND OPENAL_LIBRARY)
        set(OpenAL_FOUND TRUE)
        if (NOT TARGET OpenAL::OpenAL)
            add_library(OpenAL::OpenAL UNKNOWN IMPORTED)
            set_target_properties(OpenAL::OpenAL PROPERTIES
                    IMPORTED_LOCATION "${OPENAL_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${OPENAL_INCLUDE_DIR}"
            )
        endif ()
    endif ()
endif ()

if (NOT OpenAL_FOUND)
    # If not found, use FetchContent to download and build openal-soft
    include(FetchContent)

    message(STATUS "OpenAL not found, fetching openal-soft from GitHub...")

    FetchContent_Declare(
            openal-soft
            GIT_REPOSITORY https://github.com/kcat/openal-soft.git
            GIT_TAG 1.23.1
    )

    # Set options to minimize OpenAL build
    set(ALSOFT_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ALSOFT_TESTS OFF CACHE BOOL "" FORCE)
    set(ALSOFT_UTILS OFF CACHE BOOL "" FORCE)
    set(ALSOFT_NO_CONFIG_UTIL ON CACHE BOOL "" FORCE)
    set(ALSOFT_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(openal-soft)

    # openal-soft defines a target named 'OpenAL'
    if (TARGET OpenAL)
        set(OpenAL_FOUND TRUE)
        if (NOT TARGET OpenAL::OpenAL)
            add_library(OpenAL::OpenAL ALIAS OpenAL)
        endif ()
        # Satisfy find_package_handle_standard_args
        FetchContent_GetProperties(openal-soft SOURCE_DIR openal_SOURCE_DIR)
        set(OPENAL_INCLUDE_DIR "${openal_SOURCE_DIR}/include")
        set(OPENAL_LIBRARY OpenAL)
    endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenAL
        REQUIRED_VARS OPENAL_INCLUDE_DIR OPENAL_LIBRARY
)
