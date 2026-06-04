# Findglfw3.cmake
#
# Finds the GLFW3 library
#
# This will define the following variables:
#
#    glfw3_FOUND
#    GLFW3_INCLUDE_DIR
#    GLFW3_LIBRARY
#
# and the following imported targets:
#
#    glfw
#

# Try to find GLFW3 using CONFIG mode first (common for vcpkg)
find_package(glfw3 CONFIG QUIET)

if (glfw3_FOUND)
    if (NOT TARGET glfw)
        if (TARGET glfw3::glfw)
            add_library(glfw ALIAS glfw3::glfw)
        endif ()
    endif ()

    # Set variables for find_package_handle_standard_args
    if (NOT GLFW3_INCLUDE_DIR)
        get_target_property(GLFW3_INCLUDE_DIR glfw INTERFACE_INCLUDE_DIRECTORIES)
    endif ()
    if (NOT GLFW3_LIBRARY)
        set(GLFW3_LIBRARY glfw)
    endif ()
endif ()

if (NOT glfw3_FOUND)
    # Try to find using standard find_path/find_library
    find_path(GLFW3_INCLUDE_DIR NAMES GLFW/glfw3.h)
    find_library(GLFW3_LIBRARY NAMES glfw glfw3)

    if (GLFW3_INCLUDE_DIR AND GLFW3_LIBRARY)
        set(glfw3_FOUND TRUE)
        if (NOT TARGET glfw)
            add_library(glfw UNKNOWN IMPORTED)
            set_target_properties(glfw PROPERTIES
                    IMPORTED_LOCATION "${GLFW3_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GLFW3_INCLUDE_DIR}"
            )
        endif ()
    endif ()
endif ()

if (NOT glfw3_FOUND)
    # If not found, use FetchContent to download and build
    include(FetchContent)

    message(STATUS "glfw3 not found, fetching from GitHub...")

    FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.4
    )

    # Set options to minimize GLFW build
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

    # Set policy to suppress the deprecation warning
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif ()

    FetchContent_GetProperties(glfw)
    if (NOT glfw_POPULATED)
        FetchContent_Populate(glfw)

        # Update the minimum required CMake version to avoid errors on new CMake versions
        file(READ "${glfw_SOURCE_DIR}/CMakeLists.txt" GLFW_CMAKE_CONTENT)
        string(REPLACE "cmake_minimum_required(VERSION 3.0)"
                "cmake_minimum_required(VERSION 3.5)"
                GLFW_CMAKE_CONTENT "${GLFW_CMAKE_CONTENT}")
        file(WRITE "${glfw_SOURCE_DIR}/CMakeLists.txt" "${GLFW_CMAKE_CONTENT}")

        add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})
    endif ()

    # GLFW's CMakeLists.txt defines a target named 'glfw'
    if (TARGET glfw)
        set(glfw3_FOUND TRUE)
        # Satisfy find_package_handle_standard_args
        set(GLFW3_INCLUDE_DIR "${glfw_SOURCE_DIR}/include")
        set(GLFW3_LIBRARY glfw)
    endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(glfw3
        REQUIRED_VARS GLFW3_INCLUDE_DIR GLFW3_LIBRARY
)
