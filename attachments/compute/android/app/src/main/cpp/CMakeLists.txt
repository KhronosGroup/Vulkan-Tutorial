cmake_minimum_required(VERSION 3.22)
project(vulkan_compute_chapters)

# Windowed chapter sources (02–10) + Android host glue.
# Chapter 12 (12_embedded.cpp) is a headless embedded sample; it is NOT included
# here because it requires no display surface and is not suitable for Android.
# ANDROID_BUILD=1 activates:
#   - glfw_android_shim.h in place of GLFW/glfw3.h
#   - AAssetManager-based shader loading
#   - __android_log_print logging
#   - chapterXX_run() entry points (instead of int main())
#
# Source paths navigate from android/app/src/main/cpp/ back to attachments/compute/:
#   ../../../../../../compute/<file>

set(CHAPTER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/02_compute_architecture.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/03_memory_models.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/04_subgroup_operations.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/05_opencl_on_vulkan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/06_advanced_data_structures.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/07_gpu_driven_pipelines.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/08_async_compute.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/09_specialized_math.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute/10_performance_optimization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/android_host.cpp"
)

add_library(vulkan_compute_chapters SHARED ${CHAPTER_SOURCES})

target_compile_features(vulkan_compute_chapters PRIVATE cxx_std_20)

target_compile_definitions(vulkan_compute_chapters PRIVATE
    ANDROID_BUILD=1
    DEMO_WINDOWED=1
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)

find_library(vulkan-lib vulkan)
find_library(android-lib android)
find_library(log-lib log)

target_link_libraries(vulkan_compute_chapters
    ${vulkan-lib}
    ${android-lib}
    ${log-lib})

# glfw_android_shim.h is next to android_host.cpp.
# compute_common.h is in the compute chapter directory.
target_include_directories(vulkan_compute_chapters PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../compute")
