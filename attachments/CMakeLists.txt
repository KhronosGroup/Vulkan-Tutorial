cmake_minimum_required (VERSION 3.29)

# Enable C++ module dependency scanning
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Check if VULKAN_SDK environment variable is set
if(NOT DEFINED ENV{VULKAN_SDK})
    message(WARNING "VULKAN_SDK environment variable is not set. This may cause issues with finding Vulkan SDK.")
endif()

project (VulkanTutorial)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

find_package (glfw3 REQUIRED)
find_package (glm REQUIRED)

# Try to find Vulkan
find_package (Vulkan REQUIRED)

# Check if Vulkan was found correctly
if(NOT Vulkan_FOUND OR NOT EXISTS "${Vulkan_INCLUDE_DIR}")
    message(WARNING "Vulkan SDK not found in expected location: ${Vulkan_INCLUDE_DIR}")

    # Try to find Vulkan in alternative locations
    if(DEFINED ENV{VULKAN_SDK})
        message(STATUS "Trying to find Vulkan SDK in VULKAN_SDK environment variable: $ENV{VULKAN_SDK}")
        set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include")
        if(EXISTS "${Vulkan_INCLUDE_DIR}")
            message(STATUS "Found Vulkan include directory at: ${Vulkan_INCLUDE_DIR}")

            # Update Vulkan library path if needed
            if(NOT EXISTS "${Vulkan_LIBRARY}")
                if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
                    set(Vulkan_LIBRARY "$ENV{VULKAN_SDK}/lib/vulkan-1.lib")
                else()
                    set(Vulkan_LIBRARY "$ENV{VULKAN_SDK}/lib/libvulkan.so")
                endif()
                message(STATUS "Setting Vulkan library to: ${Vulkan_LIBRARY}")
            endif()

            # If the Vulkan::Vulkan target already exists, update its properties
            if(TARGET Vulkan::Vulkan)
                message(STATUS "Updating Vulkan::Vulkan target properties")
                set_target_properties(Vulkan::Vulkan PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
                    IMPORTED_LOCATION "${Vulkan_LIBRARY}"
                )
            else()
                # Create the Vulkan::Vulkan target if it doesn't exist
                message(STATUS "Creating Vulkan::Vulkan target")
                add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
                set_target_properties(Vulkan::Vulkan PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}"
                    IMPORTED_LOCATION "${Vulkan_LIBRARY}"
                )
            endif()
        else()
            message(FATAL_ERROR "Vulkan include directory not found at: ${Vulkan_INCLUDE_DIR}")
        endif()
    else()
        message(FATAL_ERROR "Vulkan SDK not found. Please install Vulkan SDK or set VULKAN_SDK environment variable.")
    endif()
endif()

find_package (VulkanHpp REQUIRED)

# Check if VulkanHpp_CPPM_DIR is set and exists
if(NOT DEFINED VulkanHpp_CPPM_DIR OR NOT EXISTS "${VulkanHpp_CPPM_DIR}")
    message(WARNING "VulkanHpp CPPM directory not found: ${VulkanHpp_CPPM_DIR}")

    # Try to find VulkanHpp in VULKAN_SDK
    if(DEFINED ENV{VULKAN_SDK})
        message(STATUS "Trying to find VulkanHpp in VULKAN_SDK environment variable: $ENV{VULKAN_SDK}")
        set(VulkanHpp_CPPM_DIR "$ENV{VULKAN_SDK}/include")
        if(EXISTS "${VulkanHpp_CPPM_DIR}")
            message(STATUS "Found VulkanHpp CPPM directory at: ${VulkanHpp_CPPM_DIR}")

            # Check if vulkan.cppm exists
            if(NOT EXISTS "${VulkanHpp_CPPM_DIR}/vulkan/vulkan.cppm")
                message(STATUS "vulkan.cppm not found, creating it")
                file(MAKE_DIRECTORY "${VulkanHpp_CPPM_DIR}/vulkan")
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
        else()
            message(FATAL_ERROR "VulkanHpp CPPM directory not found at: ${VulkanHpp_CPPM_DIR}")
        endif()
    else()
        message(FATAL_ERROR "VulkanHpp CPPM directory not found and VULKAN_SDK environment variable is not set.")
    endif()
endif()

find_package (tinyobjloader REQUIRED)
find_package (TinyGLTF REQUIRED)
find_package (KTX REQUIRED)

# set up Vulkan C++ module
add_library(VulkanCppModule)
add_library(Vulkan::cppm ALIAS VulkanCppModule)

target_compile_definitions(VulkanCppModule
        PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)
target_include_directories(VulkanCppModule
        PRIVATE
        "${Vulkan_INCLUDE_DIR}"
)

# Check if Vulkan include directory exists
if(NOT EXISTS "${Vulkan_INCLUDE_DIR}")
    message(FATAL_ERROR "Vulkan include directory not found: ${Vulkan_INCLUDE_DIR}. Please install Vulkan SDK or set VULKAN_SDK environment variable.")
endif()
target_link_libraries(VulkanCppModule
        PUBLIC
        Vulkan::Vulkan
)

set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

# VulkanHpp_CPPM_DIR is already checked above

target_sources(VulkanCppModule
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS
        "${VulkanHpp_CPPM_DIR}"
        FILES
        "${VulkanHpp_CPPM_DIR}/vulkan/vulkan.cppm"
)


# Add the vulkan.cppm file directly as a source file
# (Checks for existence are already done above)
target_sources(VulkanCppModule
        PRIVATE
        "${VulkanHpp_CPPM_DIR}/vulkan/vulkan.cppm"
)

find_package(stb REQUIRED)
set(STB_INCLUDEDIR ${stb_INCLUDE_DIRS})

add_executable (glslang::validator IMPORTED)
find_program (GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
set_property (TARGET glslang::validator PROPERTY IMPORTED_LOCATION "${GLSLANG_VALIDATOR}")
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

function (add_shaders_target TARGET)
  cmake_parse_arguments ("SHADER" "" "CHAPTER_NAME" "SOURCES" ${ARGN})
  set (SHADERS_DIR ${SHADER_CHAPTER_NAME}/shaders)
  add_custom_command (
    OUTPUT ${SHADERS_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )
  add_custom_command (
    OUTPUT ${SHADERS_DIR}/frag.spv ${SHADERS_DIR}/vert.spv
    COMMAND glslang::validator
    ARGS --target-env vulkan1.0 ${SHADER_SOURCES} --quiet
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Shaders"
    VERBATIM
    )
  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/frag.spv ${SHADERS_DIR}/vert.spv)
endfunction ()

function (add_slang_shader_target TARGET)
  cmake_parse_arguments ("SHADER" "" "CHAPTER_NAME" "SOURCES" ${ARGN})
  set (SHADERS_DIR ${SHADER_CHAPTER_NAME}/shaders)
  file(GLOB HAS_COMPUTE ${CHAPTER_SHADER}.comp)
  set (ENTRY_POINTS -entry vertMain -entry fragMain)
  if(HAS_COMPUTE)
    list(APPEND ENTRY_POINTS -entry compMain)
  endif()
  add_custom_command (
          OUTPUT ${SHADERS_DIR}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )
  add_custom_command (
          OUTPUT  ${SHADERS_DIR}/slang.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
          WORKING_DIRECTORY ${SHADERS_DIR}
          DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )
  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

function (add_chapter CHAPTER_NAME)
  cmake_parse_arguments (CHAPTER "" "SHADER" "LIBS;TEXTURES;MODELS" ${ARGN})
  add_executable (${CHAPTER_NAME} ${CHAPTER_NAME}.cpp)
  set_target_properties (${CHAPTER_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CHAPTER_NAME})
  set_target_properties (${CHAPTER_NAME} PROPERTIES CXX_STANDARD 20)
  target_link_libraries (${CHAPTER_NAME} Vulkan::cppm glfw)
  target_include_directories (${CHAPTER_NAME} PRIVATE ${STB_INCLUDEDIR})

  if(WIN32)
    if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
      set_target_properties(${CHAPTER_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/${CHAPTER_NAME}")
    endif()
  endif()

  if (DEFINED CHAPTER_SHADER)
    set (CHAPTER_SHADER_TARGET ${CHAPTER_NAME}_shader)
    file (GLOB SHADER_SOURCES ${CHAPTER_SHADER}.frag ${CHAPTER_SHADER}.vert ${CHAPTER_SHADER}.comp)
    add_shaders_target (${CHAPTER_SHADER_TARGET} CHAPTER_NAME ${CHAPTER_NAME} SOURCES ${SHADER_SOURCES})
    add_dependencies (${CHAPTER_NAME} ${CHAPTER_SHADER_TARGET})

    set (CHAPTER_SHADER_SLANG_TARGET ${CHAPTER_NAME}_slang_shader)
    file (GLOB SHADER_SLANG_SOURCES ${CHAPTER_SHADER}.slang)
    if(SHADER_SLANG_SOURCES)
      add_slang_shader_target( ${CHAPTER_SHADER_SLANG_TARGET} CHAPTER_NAME ${CHAPTER_NAME} SOURCES ${SHADER_SLANG_SOURCES})
      add_dependencies(${CHAPTER_NAME} ${CHAPTER_SHADER_SLANG_TARGET})
    endif()
  endif ()
  if (DEFINED CHAPTER_LIBS)
    target_link_libraries (${CHAPTER_NAME} ${CHAPTER_LIBS})
  endif ()
  if (DEFINED CHAPTER_MODELS)
    file (COPY assets/${CHAPTER_MODELS} DESTINATION ${CMAKE_BINARY_DIR}/${CHAPTER_NAME}/models)
  endif ()
  if (DEFINED CHAPTER_TEXTURES)
    file (COPY assets/${CHAPTER_TEXTURES} DESTINATION ${CMAKE_BINARY_DIR}/${CHAPTER_NAME}/textures)
  endif ()
endfunction ()

add_chapter (00_base_code)

add_chapter (01_instance_creation)

add_chapter (02_validation_layers)

add_chapter (03_physical_device_selection)

add_chapter (04_logical_device)

add_chapter (05_window_surface)

add_chapter (06_swap_chain_creation)

add_chapter (07_image_views)

add_chapter (08_graphics_pipeline)

add_chapter (09_shader_modules
  SHADER 09_shader_base)

add_chapter (10_fixed_functions
  SHADER 09_shader_base)

add_chapter (12_graphics_pipeline_complete
  SHADER 09_shader_base)

add_chapter (14_command_buffers
  SHADER 09_shader_base)

add_chapter (15_hello_triangle
  SHADER 09_shader_base)

add_chapter (16_frames_in_flight
  SHADER 09_shader_base)

add_chapter (17_swap_chain_recreation
  SHADER 09_shader_base)

add_chapter (18_vertex_input
  SHADER 18_shader_vertexbuffer
  LIBS glm::glm)

add_chapter (19_vertex_buffer
  SHADER 18_shader_vertexbuffer
  LIBS glm::glm)

add_chapter (20_staging_buffer
  SHADER 18_shader_vertexbuffer
  LIBS glm::glm)

add_chapter (21_index_buffer
  SHADER 18_shader_vertexbuffer
  LIBS glm::glm)

add_chapter (22_descriptor_layout
  SHADER 22_shader_ubo
  LIBS glm::glm)

add_chapter (23_descriptor_sets
  SHADER 22_shader_ubo
  LIBS glm::glm)

add_chapter (24_texture_image
  SHADER 22_shader_ubo
  TEXTURES ../../images/texture.jpg
  LIBS glm::glm)

add_chapter (25_sampler
  SHADER 22_shader_ubo
  TEXTURES ../../images/texture.jpg
  LIBS glm::glm)

add_chapter (26_texture_mapping
  SHADER 26_shader_textures
  TEXTURES ../../images/texture.jpg
  LIBS glm::glm)

add_chapter (27_depth_buffering
  SHADER 27_shader_depth
  TEXTURES ../../images/texture.jpg
  LIBS glm::glm)

add_chapter (28_model_loading
  SHADER 27_shader_depth
  MODELS viking_room.obj
  TEXTURES viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (29_mipmapping
  SHADER 27_shader_depth
  MODELS viking_room.obj
  TEXTURES viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (30_multisampling
  SHADER 27_shader_depth
  MODELS viking_room.obj
  TEXTURES viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (31_compute_shader
  SHADER 31_shader_compute
  LIBS glm::glm)

add_chapter (32_ecosystem_utilities
  SHADER 27_shader_depth
  MODELS viking_room.obj
  TEXTURES viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (33_vulkan_profiles
  SHADER 27_shader_depth
  MODELS viking_room.obj
  TEXTURES viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (34_android
  SHADER 27_shader_depth
  MODELS viking_room.obj
  TEXTURES viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (35_gltf_ktx
  SHADER 27_shader_depth
  MODELS viking_room.glb
  TEXTURES viking_room.ktx2
  LIBS glm::glm tinygltf::tinygltf KTX::ktx)

add_chapter (36_multiple_objects
  SHADER 27_shader_depth
  MODELS viking_room.glb
  TEXTURES viking_room.ktx2
  LIBS glm::glm tinygltf::tinygltf KTX::ktx)

add_chapter (37_multithreading
  SHADER 37_shader_compute
  LIBS glm::glm)
