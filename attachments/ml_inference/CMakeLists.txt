cmake_minimum_required(VERSION 3.20)
project(ml_inference_examples VERSION 1.0.0 LANGUAGES CXX)

# Support for standalone build (finding custom modules in parent directory)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMake")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../CMake")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ML_TRAIN_MODELS "Automatically train ML models during build" OFF)
option(ML_SETUP_PYTHON_ENV "Setup Python environment for training" ON)
option(ML_INSTALL_OPTIONAL_DEPS "Install optional dependencies (ONNX Runtime, TensorFlow Lite)" OFF)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Vulkan include directory support
if(NOT Vulkan_INCLUDE_DIR AND Vulkan_INCLUDE_DIRS)
    set(Vulkan_INCLUDE_DIR ${Vulkan_INCLUDE_DIRS})
endif()

# Find GLFW for GUI application
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "GLFW not found via find_package, trying pkg-config")
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(GLFW glfw3)
    endif()
endif()

# Add Vulkan include path
if(Vulkan_INCLUDE_DIR)
    include_directories(${Vulkan_INCLUDE_DIR})
endif()

# Common compile options
if(MSVC)
    add_compile_options(/W4 /WX-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# shader compilation setup
file(GLOB SLANG_SHADER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.slang)

# Add GTest via FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add ImGui via FetchContent
FetchContent_Declare(
    imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.1.zip
)
FetchContent_MakeAvailable(imgui)
set(IMGUI_DIR ${imgui_SOURCE_DIR})

# Find slangc executable
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin)

if(SLANGC_EXECUTABLE)
    # Ensure the output directory for compiled shaders exists
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

    # Compile Slang shaders using slangc
    foreach(SHADER ${SLANG_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)

        # Special handling for imgui.slang (vertex + fragment shader)
        if(SHADER_NAME STREQUAL "imgui")
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/imgui.vert.spv
                COMMAND ${SLANGC_EXECUTABLE} ${SHADER} -target spirv -profile spirv_1_4 -emit-spirv-directly -stage vertex -entry VSMain -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/imgui.vert.spv
                DEPENDS ${SHADER}
                COMMENT "Compiling Slang shader imgui.slang (vertex stage)"
            )
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/imgui.frag.spv
                COMMAND ${SLANGC_EXECUTABLE} ${SHADER} -target spirv -profile spirv_1_4 -emit-spirv-directly -stage fragment -entry PSMain -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/imgui.frag.spv
                DEPENDS ${SHADER}
                COMMENT "Compiling Slang shader imgui.slang (fragment stage)"
            )
            list(APPEND SHADER_SPVS ${CMAKE_CURRENT_BINARY_DIR}/shaders/imgui.vert.spv)
            list(APPEND SHADER_SPVS ${CMAKE_CURRENT_BINARY_DIR}/shaders/imgui.frag.spv)
        else()
            # All other shaders are compute shaders
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.comp.spv
                COMMAND ${SLANGC_EXECUTABLE} ${SHADER} -target spirv -profile spirv_1_4 -emit-spirv-directly -stage compute -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.comp.spv
                DEPENDS ${SHADER}
                COMMENT "Compiling Slang shader ${SHADER_NAME}.slang with slangc"
            )
            list(APPEND SHADER_SPVS ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.comp.spv)
        endif()
    endforeach()

    add_custom_target(ml_shaders DEPENDS ${SHADER_SPVS})
    message(STATUS "Shader compilation enabled: ${SLANGC_EXECUTABLE}")
else()
    message(STATUS "slangc not found. Shader compilation will be skipped.")
    add_custom_target(ml_shaders)
endif()

# Find OpenCV for camera integration
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found. Camera inference example will be skipped or run in simulation mode.")
endif()

# Auto-detect ONNX Runtime from third_party directory
if(NOT ONNXRUNTIME_DIR)
    # Prefer GPU/WebGPU builds if available
    file(GLOB ONNX_RUNTIME_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-*-gpu-*")
    if(NOT ONNX_RUNTIME_DIRS)
        file(GLOB ONNX_RUNTIME_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-*-webgpu")
    endif()
    if(NOT ONNX_RUNTIME_DIRS)
        file(GLOB ONNX_RUNTIME_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-*")
    endif()
    
    if(ONNX_RUNTIME_DIRS)
        list(GET ONNX_RUNTIME_DIRS 0 ONNXRUNTIME_DIR)
    endif()
endif()

if(ONNXRUNTIME_DIR AND EXISTS "${ONNXRUNTIME_DIR}")
    set(HAS_ONNX_RUNTIME TRUE)
    message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")
else()
    set(HAS_ONNX_RUNTIME FALSE)
    message(STATUS "ONNX Runtime not found. Some examples will be limited or skipped.")
endif()

# Auto-detect IREE Runtime from third_party directory
if(NOT IREE_RUNTIME_DIR)
    file(GLOB IREE_DIST_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/iree-dist-linux-x86_64-*")
    if(IREE_DIST_DIRS)
        list(GET IREE_DIST_DIRS 0 IREE_RUNTIME_DIR)
    endif()
endif()

if(IREE_RUNTIME_DIR AND EXISTS "${IREE_RUNTIME_DIR}")
    set(HAS_IREE TRUE)
    message(STATUS "Using IREE Runtime from: ${IREE_RUNTIME_DIR}")
    
    # Add IREE to find_package path
    set(IREERuntime_DIR "${IREE_RUNTIME_DIR}/lib/cmake/IREE")
    # The distribution uses IREERuntime for the runtime package name
    find_package(IREERuntime CONFIG REQUIRED)
else()
    set(HAS_IREE FALSE)
    message(STATUS "IREE Runtime not found. IREE examples will be skipped.")
endif()

# Common library for shared components
if(glfw3_FOUND OR GLFW_FOUND)
    add_library(ml_common STATIC
        common/renderer/renderer.cpp
        common/preprocessing/vulkan_preprocessing.cpp
        common/imgui_system.cpp
        common/vulkan_nnef_inference.cpp
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        third_party/nnef-parser/src/nnef.cpp
    )

    target_include_directories(ml_common PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_CURRENT_SOURCE_DIR}/common/renderer
        ${CMAKE_CURRENT_SOURCE_DIR}/common/preprocessing
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nnef-parser/include
        ${IMGUI_DIR}
    )

    target_compile_definitions(ml_common PUBLIC
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
    )

    target_link_libraries(ml_common PUBLIC Vulkan::Vulkan)

    if(glfw3_FOUND)
        target_link_libraries(ml_common PUBLIC glfw)
    elseif(GLFW_FOUND)
        target_include_directories(ml_common PUBLIC ${GLFW_INCLUDE_DIRS})
        target_link_libraries(ml_common PUBLIC ${GLFW_LIBRARIES})
    endif()

    add_dependencies(ml_common ml_shaders)

    # Helper function to add ML inference examples
    function(add_ml_example TARGET_NAME SOURCE_FILES)
        add_executable(${TARGET_NAME} ${SOURCE_FILES})
        target_link_libraries(${TARGET_NAME} PRIVATE ml_common)

        # Copy compiled shaders, models, and data to executable directory at build time
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_BINARY_DIR}/shaders
                $<TARGET_FILE_DIR:${TARGET_NAME}>/shaders
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/models
                $<TARGET_FILE_DIR:${TARGET_NAME}>/models
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/data
                $<TARGET_FILE_DIR:${TARGET_NAME}>/data
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/sample_images
                $<TARGET_FILE_DIR:${TARGET_NAME}>/sample_images
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_weights.bin
                $<TARGET_FILE_DIR:${TARGET_NAME}>/mnist_weights.bin
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_model.onnx
                $<TARGET_FILE_DIR:${TARGET_NAME}>/mnist_model.onnx
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_model.onnx.data
                $<TARGET_FILE_DIR:${TARGET_NAME}>/mnist_model.onnx.data
            COMMENT "Copying resources for ${TARGET_NAME}"
        )
    endfunction()

    # MNIST interactive inference application (drawable canvas with real-time recognition)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mnist_inference.cpp")
        set(MNIST_SOURCES "mnist_inference.cpp;vulkan_mnist_inference.cpp")
        if(HAS_ONNX_RUNTIME)
            list(APPEND MNIST_SOURCES "onnx_inference.cpp")
        endif()
        if(HAS_IREE)
            list(APPEND MNIST_SOURCES "iree_mnist_inference.cpp")
        endif()
        
        add_ml_example(mnist "${MNIST_SOURCES}")
        
        if(HAS_ONNX_RUNTIME)
            target_include_directories(mnist PRIVATE ${ONNXRUNTIME_DIR}/include)
            if(WIN32)
                target_link_libraries(mnist PRIVATE ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
            elseif(APPLE)
                target_link_libraries(mnist PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)
            else()
                target_link_libraries(mnist PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
            endif()
            target_compile_definitions(mnist PRIVATE HAS_ONNX_RUNTIME)
        endif()
        
        if(HAS_IREE)
            target_include_directories(mnist PRIVATE ${IREE_RUNTIME_DIR}/include)
            target_link_libraries(mnist PRIVATE 
                iree_runtime_runtime
                iree_hal_drivers_vulkan_registration_registration
                iree_tooling_numpy_io
                iree_io_stream
                iree_base_base
            )
            target_compile_definitions(mnist PRIVATE HAS_IREE)
            
            # Ensure VMFB is copied
            add_custom_command(TARGET mnist POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${CMAKE_CURRENT_SOURCE_DIR}/mnist_model.vmfb
                    $<TARGET_FILE_DIR:mnist>/mnist_model.vmfb
                COMMENT "Copying IREE VMFB for mnist"
            )
        endif()
        
        message(STATUS "Building MNIST interactive inference application")
    endif()

    # Image classification with ONNX Runtime (MobileNetV2)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/image_classifier.cpp")
        if(HAS_ONNX_RUNTIME)
            add_ml_example(classification
                "image_classifier.cpp;onnx_inference.cpp;common/preprocessing/vulkan_preprocessing.cpp"
            )

            target_include_directories(classification PRIVATE
                ${ONNXRUNTIME_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}
            )

            # Platform-specific library linking
            if(WIN32)
                target_link_libraries(classification PRIVATE
                    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
                )
            elseif(APPLE)
                target_link_libraries(classification PRIVATE
                    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib
                )
            else()
                target_link_libraries(classification PRIVATE
                    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
                )
            endif()

            target_compile_definitions(classification PRIVATE HAS_ONNX_RUNTIME)

            message(STATUS "Building image classification example")
        else()
            message(STATUS "Image classification requires ONNX Runtime.")
            message(STATUS "  Run the setup script to download it automatically:")
            message(STATUS "  ./setup_env.sh")
        endif()
    endif()

    # Real-time Camera Inference (OpenCV + ONNX Runtime)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/camera_inference.cpp" AND OpenCV_FOUND AND ONNXRUNTIME_DIR)
        add_ml_example(camera_inference 
            "camera_inference.cpp;onnx_inference.cpp;common/preprocessing/vulkan_preprocessing.cpp"
        )
        target_link_libraries(camera_inference PRIVATE ${OpenCV_LIBRARIES})
        target_include_directories(camera_inference PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_DIR}/include)

        if(WIN32)
            target_link_libraries(camera_inference PRIVATE ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
        elseif(APPLE)
            target_link_libraries(camera_inference PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)
        else()
            target_link_libraries(camera_inference PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
        endif()

        message(STATUS "Building real-time camera inference example")
    endif()

    # IREE Vulkan Inference Example
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/iree_vulkan_inference.cpp" AND HAS_IREE)
        add_ml_example(iree_vulkan_inference "iree_vulkan_inference.cpp")
        
        target_include_directories(iree_vulkan_inference PRIVATE
            ${IREE_RUNTIME_DIR}/include
        )

        target_link_libraries(iree_vulkan_inference PRIVATE 
            iree_runtime_runtime
            iree_hal_drivers_vulkan_registration_registration
            iree_tooling_numpy_io
            iree_io_stream
            iree_base_base
            Vulkan::Vulkan
        )
        
        target_compile_definitions(iree_vulkan_inference PRIVATE HAS_IREE)
        message(STATUS "Building IREE Vulkan inference example")
    endif()

    # Latency and Power Sampler Tool
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/latency_power_sampler.cpp")
        add_executable(latency_power_sampler tools/latency_power_sampler.cpp)
        # No heavy dependencies, just standard library
        message(STATUS "Building latency and power sampler tool")
    endif()

else()
    message(STATUS "GLFW not found - skipping GUI examples")
endif()

    # CI Render Validation (Headless Vulkan example with GTest)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ci_render_validation.cpp")
        set(CI_SOURCES "ci_render_validation.cpp;vulkan_mnist_inference.cpp")
        if(HAS_ONNX_RUNTIME)
            list(APPEND CI_SOURCES "onnx_inference.cpp")
        endif()
        
        add_executable(ci_validation "${CI_SOURCES}")
        target_link_libraries(ci_validation PRIVATE ml_common GTest::gtest_main)
        
        # Copy compiled shaders, models, and data to executable directory at build time
        add_custom_command(TARGET ci_validation POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_BINARY_DIR}/shaders
                $<TARGET_FILE_DIR:ci_validation>/shaders
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/models
                $<TARGET_FILE_DIR:ci_validation>/models
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/data
                $<TARGET_FILE_DIR:ci_validation>/data
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/sample_images
                $<TARGET_FILE_DIR:ci_validation>/sample_images
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_weights.bin
                $<TARGET_FILE_DIR:ci_validation>/mnist_weights.bin
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_model.onnx
                $<TARGET_FILE_DIR:ci_validation>/mnist_model.onnx
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_model.onnx.data
                $<TARGET_FILE_DIR:ci_validation>/mnist_model.onnx.data
            COMMENT "Copying resources for ci_validation"
        )
        
        if(HAS_ONNX_RUNTIME)
            target_include_directories(ci_validation PRIVATE ${ONNXRUNTIME_DIR}/include)
            if(WIN32)
                target_link_libraries(ci_validation PRIVATE ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
            elseif(APPLE)
                target_link_libraries(ci_validation PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)
            else()
                target_link_libraries(ci_validation PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
            endif()
            target_compile_definitions(ci_validation PRIVATE HAS_ONNX_RUNTIME)
        endif()
        
        message(STATUS "Building CI render validation example (GTest)")
    endif()

    # MNIST Integration Test
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/mnist_integration_test.cpp")
        add_executable(mnist_integration_test 
            tests/mnist_integration_test.cpp 
            vulkan_mnist_inference.cpp
            common/renderer/renderer.cpp
        )
        target_link_libraries(mnist_integration_test PRIVATE ml_common)
        
        # Copy resources for the test
        add_custom_command(TARGET mnist_integration_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_BINARY_DIR}/shaders
                $<TARGET_FILE_DIR:mnist_integration_test>/shaders
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/data
                $<TARGET_FILE_DIR:mnist_integration_test>/data
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_weights.bin
                $<TARGET_FILE_DIR:mnist_integration_test>/mnist_weights.bin
        )
        
        if(TARGET train_mnist)
            add_dependencies(mnist_integration_test train_mnist)
        endif()
    endif()

    # Desktop NNEF Engine
    add_executable(nnef_inference nnef_inference.cpp)
    target_link_libraries(nnef_inference PRIVATE ml_common Vulkan::Vulkan)
    target_include_directories(nnef_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common)
    target_compile_definitions(nnef_inference PRIVATE 
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
        USE_NNEF
    )
    add_custom_command(TARGET nnef_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_BINARY_DIR}/shaders
            $<TARGET_FILE_DIR:nnef_inference>/shaders
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/models/mobilenetv2_nnef_optimized
            $<TARGET_FILE_DIR:nnef_inference>/models/mobilenetv2_nnef_optimized
    )

    # Desktop IREE Engine (Sample 8)
    if(HAS_IREE)
        add_executable(iree_desktop_inference iree_vulkan_inference.cpp)
        target_link_libraries(iree_desktop_inference PRIVATE 
            iree_runtime_runtime
            iree_hal_drivers_vulkan_registration_registration
            iree_tooling_numpy_io
            iree_io_stream
            iree_base_base
        )
        target_include_directories(iree_desktop_inference PRIVATE ${IREE_RUNTIME_DIR}/include)
    endif()

    # MNIST Robustness Test
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/mnist_robustness_test.cpp")
        add_executable(mnist_robustness_test 
            tests/mnist_robustness_test.cpp 
            vulkan_mnist_inference.cpp
            common/renderer/renderer.cpp
        )
        target_link_libraries(mnist_robustness_test PRIVATE ml_common)
        
        # Copy resources for the test
        add_custom_command(TARGET mnist_robustness_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_BINARY_DIR}/shaders
                $<TARGET_FILE_DIR:mnist_robustness_test>/shaders
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/data
                $<TARGET_FILE_DIR:mnist_robustness_test>/data
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_weights.bin
                $<TARGET_FILE_DIR:mnist_robustness_test>/mnist_weights.bin
        )
    endif()

    # MNIST Comparison Test (Vulkan vs ONNX)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/mnist_comparison_test.cpp")
        set(COMP_SOURCES tests/mnist_comparison_test.cpp vulkan_mnist_inference.cpp common/renderer/renderer.cpp)
        if(HAS_ONNX_RUNTIME)
            list(APPEND COMP_SOURCES onnx_inference.cpp)
        endif()

        add_executable(mnist_comparison_test ${COMP_SOURCES})
        target_link_libraries(mnist_comparison_test PRIVATE ml_common)

        if(HAS_ONNX_RUNTIME)
            target_include_directories(mnist_comparison_test PRIVATE ${ONNXRUNTIME_DIR}/include)
            if(WIN32)
                target_link_libraries(mnist_comparison_test PRIVATE ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
            elseif(APPLE)
                target_link_libraries(mnist_comparison_test PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)
            else()
                target_link_libraries(mnist_comparison_test PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
            endif()
            target_compile_definitions(mnist_comparison_test PRIVATE HAS_ONNX_RUNTIME)
        endif()

        add_custom_command(TARGET mnist_comparison_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_BINARY_DIR}/shaders
                $<TARGET_FILE_DIR:mnist_comparison_test>/shaders
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/data
                $<TARGET_FILE_DIR:mnist_comparison_test>/data
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_weights.bin
                $<TARGET_FILE_DIR:mnist_comparison_test>/mnist_weights.bin
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_model.onnx
                $<TARGET_FILE_DIR:mnist_comparison_test>/mnist_model.onnx
        )
    endif()

    # Unit Tests
    enable_testing()
    add_test(NAME MNIST_Integration_Test COMMAND mnist_integration_test)
    add_test(NAME MNIST_Robustness_Test COMMAND mnist_robustness_test)
    if(TARGET mnist_comparison_test)
        add_test(NAME MNIST_Comparison_Test COMMAND mnist_comparison_test)
    endif()

    # Weight Loader Unit Test
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/weight_loader_test.cpp")
        add_executable(weight_loader_test tests/weight_loader_test.cpp)
        target_include_directories(weight_loader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        
        add_custom_command(TARGET weight_loader_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/mnist_weights.bin
                $<TARGET_FILE_DIR:weight_loader_test>/mnist_weights.bin
        )
        
        add_test(NAME WeightLoader_Unit_Test COMMAND weight_loader_test)
    endif()

    # Wildlife Cam GTest
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/wildlife_cam_test.cpp" AND OpenCV_FOUND)
        add_executable(wildlife_cam_test tests/wildlife_cam_test.cpp)
        target_link_libraries(wildlife_cam_test PRIVATE 
            ml_common
            GTest::gtest_main
            GTest::gmock
            ${OpenCV_LIBRARIES}
            Vulkan::Vulkan
        )
        target_include_directories(wildlife_cam_test PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/common
        )
        target_compile_definitions(wildlife_cam_test PRIVATE 
            VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
            VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
            USE_NNEF
        )
        
        add_custom_command(TARGET wildlife_cam_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_BINARY_DIR}/shaders
                $<TARGET_FILE_DIR:wildlife_cam_test>/shaders
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/models/mobilenetv2_nnef_optimized
                $<TARGET_FILE_DIR:wildlife_cam_test>/models/mobilenetv2_nnef_optimized
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/sample_images
                $<TARGET_FILE_DIR:wildlife_cam_test>/sample_images
        )
        
        add_test(NAME WildlifeCam_GTest COMMAND wildlife_cam_test)
    endif()

# RL Automated Exploration (Curiosity-driven RND example)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rl_exploration.cpp")
    add_ml_example(rl_exploration rl_exploration.cpp)
    target_link_libraries(rl_exploration PRIVATE GTest::gtest_main)
    add_test(NAME RL_Exploration_Test COMMAND rl_exploration)
    message(STATUS "Building RL automated exploration example (GTest)")
endif()

# Ray Tracing Optimization Demo
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rt_optimization.cpp")
    set(RT_SOURCES "rt_optimization.cpp")
    if(HAS_ONNX_RUNTIME)
        list(APPEND RT_SOURCES "onnx_inference.cpp")
    endif()
    
    add_ml_example(rt_optimization "${RT_SOURCES}")
    
    if(HAS_ONNX_RUNTIME)
        target_include_directories(rt_optimization PRIVATE ${ONNXRUNTIME_DIR}/include)
        if(WIN32)
            target_link_libraries(rt_optimization PRIVATE ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)
        elseif(APPLE)
            target_link_libraries(rt_optimization PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib)
        else()
            target_link_libraries(rt_optimization PRIVATE ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
        endif()
        target_compile_definitions(rt_optimization PRIVATE HAS_ONNX_RUNTIME)
    endif()
    
    message(STATUS "Building RT optimization example")
endif()

# Python environment setup for ML training
find_package(Python3 COMPONENTS Interpreter)

option(ML_AUTO_SETUP_VENV "Automatically setup Python virtual environment during configuration" OFF)

if(Python3_FOUND AND ML_SETUP_PYTHON_ENV)
    set(VENV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/venv")

    # Check if virtual environment exists
    if(WIN32)
        set(VENV_PYTHON "${VENV_DIR}/Scripts/python.exe")
    else()
        set(VENV_PYTHON "${VENV_DIR}/bin/python")
    endif()

    if(NOT EXISTS "${VENV_PYTHON}")
        if(ML_AUTO_SETUP_VENV)
            message(STATUS "Setting up Python environment for ML training...")

            # Build setup command with optional dependencies flag
            set(SETUP_CMD ${Python3_EXECUTABLE} setup_python_env.py)
            if(ML_INSTALL_OPTIONAL_DEPS)
                list(APPEND SETUP_CMD --with-optional)
            endif()

            # Run setup script
            execute_process(
                COMMAND ${SETUP_CMD}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE SETUP_RESULT
            )

            if(SETUP_RESULT EQUAL 0)
                message(STATUS "✓ Python environment setup complete")
            else()
                message(WARNING "Python environment setup failed.")
            endif()
        else()
            message(STATUS "Python virtual environment not found. You can set it up by running: ${Python3_EXECUTABLE} setup_python_env.py")
            message(STATUS "Or reconfigure with -DML_AUTO_SETUP_VENV=ON")
        endif()
    else()
        message(STATUS "Python virtual environment found at: ${VENV_DIR}")
    endif()

    # Create a target for setting up the Python environment
    add_custom_target(setup_venv
        COMMAND ${Python3_EXECUTABLE} setup_python_env.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Setting up Python virtual environment..."
    )

    # Optional: Train models during build
    if(ML_TRAIN_MODELS AND EXISTS "${VENV_PYTHON}")
        set(WEIGHTS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/mnist_weights.bin")

        # Add custom target to train MNIST model
        add_custom_command(
            OUTPUT ${WEIGHTS_FILE}
            COMMAND ${VENV_PYTHON} train_mnist.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Training MNIST model (this may take several minutes)..."
            DEPENDS train_mnist.py
        )

        add_custom_target(train_mnist_model ALL DEPENDS ${WEIGHTS_FILE})
        message(STATUS "MNIST model will be trained during build (use -DML_TRAIN_MODELS=OFF to disable)")
    else()
        # Add manual target for training with proper dependency tracking
        if(EXISTS "${VENV_PYTHON}")
            set(WEIGHTS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/mnist_weights.bin")
            set(ONNX_FILE "${CMAKE_CURRENT_SOURCE_DIR}/mnist_model.onnx")

            # Custom command that only runs when outputs are missing or inputs changed
            add_custom_command(
                OUTPUT ${WEIGHTS_FILE} ${ONNX_FILE}
                COMMAND ${VENV_PYTHON} train_mnist.py
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                DEPENDS train_mnist.py
                COMMENT "Training MNIST model..."
            )

            # Target that depends on the outputs
            add_custom_target(train_mnist DEPENDS ${WEIGHTS_FILE} ${ONNX_FILE})

            # Make mnist depend on training artifacts so they're built automatically
            if(TARGET mnist)
                add_dependencies(mnist train_mnist)
                message(STATUS "mnist will automatically train model if needed")
            endif()

            message(STATUS "To train the MNIST model manually, run: make train_mnist")
        endif()
    endif()
endif()

# Print configuration summary
message(STATUS "ML Inference Examples Configuration:")
message(STATUS "  Vulkan SDK: ${Vulkan_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")

# Ensure all source files are visible in the IDE
file(GLOB_RECURSE IDE_SOURCES *.cpp *.h *.hpp *.slang)
add_custom_target(all_project_files SOURCES ${IDE_SOURCES})

# Raspberry Pi wildlife cam (Sample 5)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rPi/CMakeLists.txt")
    add_subdirectory(rPi)
endif()

message(STATUS "Python Environment:")
if(Python3_FOUND)
    message(STATUS "  Python: ${Python3_VERSION}")
    if(EXISTS "${VENV_PYTHON}")
        message(STATUS "  Virtual environment: Found")
    elseif(ML_AUTO_SETUP_VENV)
        message(STATUS "  Virtual environment: Will be created on next configure")
    else()
        message(STATUS "  Virtual environment: Not found (run 'setup_venv' target to create)")
    endif()
    message(STATUS "  Optional deps (ONNX Runtime, TFLite): ${ML_INSTALL_OPTIONAL_DEPS}")
    if(NOT ML_INSTALL_OPTIONAL_DEPS)
        message(STATUS "    To enable: cmake -DML_INSTALL_OPTIONAL_DEPS=ON ..")
    endif()
else()
    message(STATUS "  Python: Not found (optional, for model training)")
endif()
message(STATUS "")
