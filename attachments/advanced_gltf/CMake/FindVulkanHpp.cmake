# FindVulkanHpp.cmake — re-entrancy shim
#
# advanced_gltf/CMakeLists.txt calls find_package(VulkanHpp) before
# add_subdirectory(simple_engine), and simple_engine/CMakeLists.txt calls it
# again.  simple_engine's implementation uses FetchContent_Populate, which is
# fatal when the content is already populated (cmake 3.x).  This shim is
# found first (advanced_gltf/CMake precedes simple_engine/CMake in
# CMAKE_MODULE_PATH) and short-circuits the second invocation.
if(TARGET VulkanHpp::VulkanHpp)
    set(VulkanHpp_FOUND TRUE)
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../simple_engine/CMake/FindVulkanHpp.cmake")
