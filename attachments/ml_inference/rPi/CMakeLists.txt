cmake_minimum_required(VERSION 3.20)
project(rpi_wildlife_cam LANGUAGES CXX)

# Detection logic for Raspberry Pi
set(IS_RPI FALSE)
if(EXISTS "/proc/device-tree/model")
    file(READ "/proc/device-tree/model" DEVICE_MODEL)
    if(DEVICE_MODEL MATCHES "Raspberry Pi")
        set(IS_RPI TRUE)
    endif()
endif()

# Support for emulated rPi via flag or if we are already on ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    set(IS_RPI TRUE)
endif()

# Support for forcing the build (e.g. for CI or desktop testing)
option(FORCE_RPI_BUILD "Force building rPi project even if not on rPi" ON)

if(IS_RPI)
    message(STATUS "Raspberry Pi environment detected. Building rPi wildlife cam.")
else()
    message(STATUS "Not a Raspberry Pi environment. Building rPi wildlife cam in DESKTOP SIMULATION mode.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(OpenCV QUIET)

# Backend selection
set(ML_BACKEND "AUTO" CACHE STRING "ML Backend to use (IREE, NNEF, TFLITE, or AUTO)")

# Detection logic for TFLite
if(NOT TFLITE_DIR)
    file(GLOB TFLITE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tensorflow-lite-*")
    if(TFLITE_DIRS)
        list(GET TFLITE_DIRS 0 TFLITE_DIR)
    endif()
endif()

if(TFLITE_DIR AND EXISTS "${TFLITE_DIR}/lib/libtensorflowlite_c.so")
    set(HAS_TFLITE TRUE)
    message(STATUS "Using TFLite from: ${TFLITE_DIR}")
else()
    set(HAS_TFLITE FALSE)
endif()

# NNEF is built-in to ml_common/vulkan_nnef_inference.cpp
set(HAS_NNEF TRUE)

# Include parent directory for common headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/renderer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/preprocessing)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nnef-parser/include)

# Define the executable
add_executable(embedded_wildlife_cam_app
    embedded_wildlife_cam.cpp
)

# Vulkan and OpenCV
target_link_libraries(embedded_wildlife_cam_app PRIVATE
    ml_common
    Vulkan::Vulkan
)
if(OpenCV_FOUND)
    target_link_libraries(embedded_wildlife_cam_app PRIVATE ${OpenCV_LIBRARIES})
    target_compile_definitions(embedded_wildlife_cam_app PRIVATE HAVE_OPENCV)
else()
    message(STATUS "OpenCV not found. rPi wildlife cam will run in simulation mode (no camera input).")
endif()

if(TARGET ml_shaders)
    add_dependencies(embedded_wildlife_cam_app ml_shaders)
endif()

target_compile_definitions(embedded_wildlife_cam_app PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
)

# Backend-specific setup
function(enable_iree_backend)
    if(HAS_IREE)
        target_include_directories(embedded_wildlife_cam_app PRIVATE ${IREE_RUNTIME_DIR}/include)
        target_link_libraries(embedded_wildlife_cam_app PRIVATE 
            iree_runtime_runtime
            iree_hal_drivers_vulkan_registration_registration
            iree_tooling_numpy_io
            iree_io_stream
            iree_base_base
        )
        target_compile_definitions(embedded_wildlife_cam_app PRIVATE USE_IREE)
        message(STATUS "IREE backend enabled")
    else()
        message(WARNING "IREE runtime not found, IREE backend disabled.")
    endif()
endfunction()

function(enable_tflite_backend)
    if(HAS_TFLITE)
        target_include_directories(embedded_wildlife_cam_app PRIVATE 
            ${TFLITE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/litert/include
        )
        target_link_libraries(embedded_wildlife_cam_app PRIVATE 
            ${TFLITE_DIR}/lib/libtensorflowlite_c.so
        )
        # Check if the library is actually LiteRT
        # For this desktop simulation, we know it is.
        target_compile_definitions(embedded_wildlife_cam_app PRIVATE USE_LITERT LITERT_DISABLE_GPU)
        message(STATUS "TFLite backend enabled (via LiteRT)")
    else()
        message(WARNING "TFLite runtime not found, TFLite backend disabled.")
    endif()
endfunction()

function(enable_nnef_backend)
    target_compile_definitions(embedded_wildlife_cam_app PRIVATE USE_NNEF)
    message(STATUS "NNEF backend enabled (using custom Vulkan engine)")
endfunction()

if(ML_BACKEND STREQUAL "AUTO")
    message(STATUS "ML_BACKEND=AUTO: Enabling all available backends")
    enable_nnef_backend()
    if(HAS_IREE)
        enable_iree_backend()
    endif()
    if(HAS_TFLITE)
        enable_tflite_backend()
    endif()
elseif(ML_BACKEND STREQUAL "IREE")
    enable_iree_backend()
    if(NOT HAS_IREE)
        message(FATAL_ERROR "IREE backend selected but IREE runtime not found.")
    endif()
elseif(ML_BACKEND STREQUAL "NNEF")
    enable_nnef_backend()
elseif(ML_BACKEND STREQUAL "TFLITE")
    enable_tflite_backend()
    if(NOT HAS_TFLITE)
        message(FATAL_ERROR "TFLite backend selected but TFLite runtime not found.")
    endif()
else()
    message(FATAL_ERROR "Unknown ML_BACKEND: ${ML_BACKEND}. Use IREE, NNEF, TFLITE, or AUTO.")
endif()

# GLFW dependency for Renderer (needed for headers/linking even if headless)
if(TARGET glfw)
    target_link_libraries(embedded_wildlife_cam_app PRIVATE glfw)
elseif(glfw3_FOUND)
    target_link_libraries(embedded_wildlife_cam_app PRIVATE glfw)
elseif(GLFW_FOUND)
    target_include_directories(embedded_wildlife_cam_app PRIVATE ${GLFW_INCLUDE_DIRS})
    target_link_libraries(embedded_wildlife_cam_app PRIVATE ${GLFW_LIBRARIES})
endif()

# Copy shaders, models, and data
add_custom_command(TARGET embedded_wildlife_cam_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_BINARY_DIR}/../shaders
        $<TARGET_FILE_DIR:embedded_wildlife_cam_app>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../models
        $<TARGET_FILE_DIR:embedded_wildlife_cam_app>/models
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../models/mobilenetv2_nnef_optimized
        $<TARGET_FILE_DIR:embedded_wildlife_cam_app>/models/mobilenetv2_nnef_optimized
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../data
        $<TARGET_FILE_DIR:embedded_wildlife_cam_app>/data
    COMMENT "Copying assets for rPi wildlife cam"
)
