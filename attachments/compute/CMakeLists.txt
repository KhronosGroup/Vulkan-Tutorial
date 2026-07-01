cmake_minimum_required(VERSION 3.29)

project(VulkanAdvancedCompute)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../CMake")

option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan" OFF)

if(ENABLE_CPP20_MODULE)
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan 1.4.335 REQUIRED)
find_package(stb REQUIRED)
set(STB_INCLUDEDIR ${stb_INCLUDE_DIRS})

if(ENABLE_CPP20_MODULE)
  add_library(VulkanCppModule)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)
  target_compile_definitions(VulkanCppModule
    PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)
  target_include_directories(VulkanCppModule PUBLIC "${Vulkan_INCLUDE_DIR}")
  target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)
  set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)
  if(MSVC)
    target_compile_options(VulkanCppModule PRIVATE
      /std:c++latest /permissive- /Zc:__cplusplus /EHsc /Zc:preprocessor /translateInclude)
  endif()
  target_sources(VulkanCppModule PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS "${Vulkan_INCLUDE_DIR}"
    FILES "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm")
  target_sources(VulkanCppModule PRIVATE "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm")
else()
  add_library(VulkanCppModule INTERFACE)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)
  target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
  target_compile_definitions(VulkanCppModule
    INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)
endif()

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

# ---------------------------------------------------------------------------
# add_compute_chapter(NAME [WINDOWED] [ENTRIES e1 e2 ...] [LIBS lib1 ...]
#                          [SLANG_DEFINES D1 D2 ...])
#   NAME          — chapter directory prefix (also the .cpp and .slang filename)
#   WINDOWED      — link GLFW and define DEMO_WINDOWED
#   ENTRIES       — list of slangc -entry names; defaults to "compMain"
#   LIBS          — extra link libraries
#   SLANG_DEFINES — preprocessor defines passed to slangc as -D<define>
# ---------------------------------------------------------------------------
function(add_compute_chapter CHAPTER_NAME)
  cmake_parse_arguments(ARG "WINDOWED" "" "ENTRIES;LIBS;SLANG_DEFINES" ${ARGN})

  add_executable(${CHAPTER_NAME} ${CHAPTER_NAME}.cpp)
  set_target_properties(${CHAPTER_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CHAPTER_NAME}
    CXX_STANDARD 20)

  target_link_libraries(${CHAPTER_NAME} Vulkan::cppm glm::glm)
  target_include_directories(${CHAPTER_NAME} PRIVATE
    ${STB_INCLUDEDIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

  if(ENABLE_CPP20_MODULE)
    target_compile_definitions(${CHAPTER_NAME} PRIVATE USE_CPP20_MODULES=1)
  endif()
  target_compile_definitions(${CHAPTER_NAME} PRIVATE VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS)

  if(ARG_WINDOWED)
    target_link_libraries(${CHAPTER_NAME} glfw)
    target_compile_definitions(${CHAPTER_NAME} PRIVATE DEMO_WINDOWED=1)
  endif()

  if(DEFINED ARG_LIBS)
    target_link_libraries(${CHAPTER_NAME} ${ARG_LIBS})
  endif()

  if(WIN32 AND ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set_target_properties(${CHAPTER_NAME} PROPERTIES
      VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${CHAPTER_NAME}")
  endif()

  # Build the -entry flags list (default: just compMain)
  if(DEFINED ARG_ENTRIES)
    set(ENTRY_FLAGS "")
    foreach(E IN LISTS ARG_ENTRIES)
      list(APPEND ENTRY_FLAGS -entry ${E})
    endforeach()
  else()
    set(ENTRY_FLAGS -entry compMain)
  endif()

  # Build the -D flags list for slangc preprocessor defines
  set(DEFINE_FLAGS "")
  if(DEFINED ARG_SLANG_DEFINES)
    foreach(D IN LISTS ARG_SLANG_DEFINES)
      list(APPEND DEFINE_FLAGS -D${D})
    endforeach()
  endif()

  # Compile the Slang shader if a .slang file exists alongside the .cpp
  file(GLOB SLANG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${CHAPTER_NAME}.slang")
  if(SLANG_SRC)
    set(SHADERS_DIR ${CHAPTER_NAME}/shaders)
    set(SPV_OUT ${SHADERS_DIR}/slang.spv)

    add_custom_command(
      OUTPUT ${SHADERS_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR})

    add_custom_command(
      OUTPUT ${SPV_OUT}
      COMMAND ${SLANGC_EXECUTABLE}
        ${SLANG_SRC}
        -target spirv
        -profile spirv_1_4
        -emit-spirv-directly
        -fvk-use-entrypoint-name
        ${ENTRY_FLAGS}
        ${DEFINE_FLAGS}
        -o slang.spv
      WORKING_DIRECTORY ${SHADERS_DIR}
      DEPENDS ${SHADERS_DIR} ${SLANG_SRC}
      COMMENT "Compiling ${CHAPTER_NAME}.slang"
      VERBATIM)

    add_custom_target(${CHAPTER_NAME}_shader DEPENDS ${SPV_OUT})
    add_dependencies(${CHAPTER_NAME} ${CHAPTER_NAME}_shader)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Demos — one per tutorial chapter, with explicit entry point lists
# ---------------------------------------------------------------------------
add_compute_chapter(02_compute_architecture
  WINDOWED
  ENTRIES compMain)

add_compute_chapter(03_memory_models
  WINDOWED
  ENTRIES splatPass advectVelPass divergencePass jacobiPass gradientPass advectDyePass renderPass)

add_compute_chapter(04_subgroup_operations
  WINDOWED
  ENTRIES physicsMain renderMain)

add_compute_chapter(06_advanced_data_structures
  WINDOWED
  ENTRIES primaryRayMain shadowQueueMain)

add_compute_chapter(07_gpu_driven_pipelines
  WINDOWED
  ENTRIES cullMain vertMain fragMain)

add_compute_chapter(08_async_compute
  WINDOWED
  ENTRIES constraintPass vertMain fragMain)

add_compute_chapter(09_specialized_math
  WINDOWED
  ENTRIES noiseMain denoiseMain)

add_compute_chapter(10_performance_optimization
  WINDOWED
  ENTRIES heatmapMain)

# Chapter 12 — Embedded Compute: parallel FABRIK IK solver.
# Headless (no WINDOWED): pure compute dispatch for Raspberry Pi, NVIDIA Jetson,
# and other embedded Linux targets. NOT intended for Android — use the windowed
# chapter samples (02–10) via the Android launcher instead.
add_compute_chapter(12_embedded
  ENTRIES ikMain)

# ---------------------------------------------------------------------------
# Chapter 05 — OpenCL on Vulkan (special: NOT a Slang demo)
#
# This chapter demonstrates the OpenCL→Vulkan toolchain by rendering ONE OpenCL
# C kernel (05_opencl_on_vulkan.cl) two ways:
#   * clspv (AOT)   — compiled to julia.spv here, loaded into a raw Vulkan pipeline
#   * clvk (runtime)— driven through the OpenCL 3.0 API at run time
# Both paths are optional; the demo degrades gracefully when a tool is absent so
# the rest of the project always builds. Build clspv/clvk with the
# install_dependencies_* scripts in this directory.
# ---------------------------------------------------------------------------
find_program(CLSPV_EXECUTABLE clspv)
find_path(OpenCL_INCLUDE_DIR CL/cl.h)
find_library(OpenCL_LIBRARY NAMES OpenCL)

add_executable(05_opencl_on_vulkan 05_opencl_on_vulkan.cpp)
set_target_properties(05_opencl_on_vulkan PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/05_opencl_on_vulkan
  CXX_STANDARD 20)
# Windowed app: it owns a Vulkan swapchain and presents, so it needs GLFW.
target_link_libraries(05_opencl_on_vulkan Vulkan::cppm glfw)
target_include_directories(05_opencl_on_vulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(05_opencl_on_vulkan PRIVATE
  VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS)
if(ENABLE_CPP20_MODULE)
  target_compile_definitions(05_opencl_on_vulkan PRIVATE USE_CPP20_MODULES=1)
endif()

set(CH5_OUT ${CMAKE_BINARY_DIR}/05_opencl_on_vulkan)

# The clvk path reads the kernel SOURCE at run time — copy it next to the binary
# whenever the .cl changes (keyed on the file, so edits re-copy without a relink).
add_custom_command(
  OUTPUT ${CH5_OUT}/05_opencl_on_vulkan.cl
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CH5_OUT}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/05_opencl_on_vulkan.cl
    ${CH5_OUT}/05_opencl_on_vulkan.cl
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/05_opencl_on_vulkan.cl
  COMMENT "Copying 05_opencl_on_vulkan.cl next to the binary (clvk reads it at runtime)"
  VERBATIM)
add_custom_target(05_opencl_on_vulkan_cl DEPENDS ${CH5_OUT}/05_opencl_on_vulkan.cl)
add_dependencies(05_opencl_on_vulkan 05_opencl_on_vulkan_cl)

# clvk runtime layering path: OpenCL 3.0 host API.
if(OpenCL_INCLUDE_DIR AND OpenCL_LIBRARY)
  target_compile_definitions(05_opencl_on_vulkan PRIVATE HAVE_OPENCL=1)
  target_include_directories(05_opencl_on_vulkan PRIVATE ${OpenCL_INCLUDE_DIR})
  target_link_libraries(05_opencl_on_vulkan ${OpenCL_LIBRARY})
  message(STATUS "Chapter 05: OpenCL found (${OpenCL_LIBRARY}) — clvk runtime path enabled")
else()
  message(STATUS "Chapter 05: OpenCL headers/loader not found — clvk runtime path disabled")
endif()

# clspv AOT path: compile the OpenCL C kernel to Vulkan-flavour SPIR-V.
if(CLSPV_EXECUTABLE)
  set(CH5_SPV ${CH5_OUT}/shaders/forest.spv)
  add_custom_command(
    OUTPUT ${CH5_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CH5_OUT}/shaders
    COMMAND ${CLSPV_EXECUTABLE} -cl-std=CL1.2 --inline-entry-points
            ${CMAKE_CURRENT_SOURCE_DIR}/05_opencl_on_vulkan.cl -o ${CH5_SPV}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/05_opencl_on_vulkan.cl
    COMMENT "clspv: compiling 05_opencl_on_vulkan.cl -> forest.spv"
    VERBATIM)
  add_custom_target(05_opencl_on_vulkan_spv DEPENDS ${CH5_SPV})
  add_dependencies(05_opencl_on_vulkan 05_opencl_on_vulkan_spv)
  target_compile_definitions(05_opencl_on_vulkan PRIVATE HAVE_CLSPV_AOT=1)
  message(STATUS "Chapter 05: clspv found (${CLSPV_EXECUTABLE}) — AOT path enabled")
else()
  message(STATUS "Chapter 05: clspv not found — AOT path disabled (run install_dependencies_*)")
endif()
