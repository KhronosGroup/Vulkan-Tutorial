cmake_minimum_required (VERSION 3.8)

project (VulkanTutorial)

find_package (Vulkan REQUIRED)

# stb
find_path(STB_INCLUDE_DIR NAMES "stb_image.h" PATHS "/usr/include/stb" "C:/Vulkan/stb")
if (NOT STB_INCLUDE_DIR)
  message (FATAL_ERROR "stb_image.h not found")
else()
  message(STATUS "stb found in ${STB_INCLUDE_DIR}")
endif()

# glfw
find_path(GLFW_INCLUDE_DIR NAMES "GLFW/glfw3.h" PATHS "/usr/include" "C:/Vulkan/GLFW/include")
if (NOT GLFW_INCLUDE_DIR)
  message (FATAL_ERROR "glfw3.h not found")
else()
  message(STATUS "GLFW found in ${GLFW_INCLUDE_DIR}")
endif()

if (UNIX)
  find_package(glfw3 3.3 REQUIRED)
  MESSAGE(STATUS "GLFW: ${GLFW_LIBRARIES}")
else ()
  find_library(GLFW_LIBRARIES NAMES glfw3 glfw)
endif ()
if (NOT GLFW_LIBRARIES)
  message (FATAL_ERROR "glfw3 library not found")
endif ()


# glm
find_path(GLM_INCLUDE_DIR NAMES "glm/glm.hpp" PATHS "/usr/include" "C:/Vulkan/glm")
if (NOT GLM_INCLUDE_DIR)
  message (FATAL_ERROR "glm.hpp not found")
else()
  message(STATUS "glm found in ${GLM_INCLUDE_DIR}")
endif()

# Tiny obj loader
find_path(TINYOBJ_INCLUDE_DIR NAMES "tiny_obj_loader.h" PATHS "/usr/include" "C:/Vulkan/tinyobjloader")
if (NOT TINYOBJ_INCLUDE_DIR)
  message (FATAL_ERROR "tiny_obj_loader.h not found")
else()
  message(STATUS "Tiny obj loader found in ${TINYOBJ_INCLUDE_DIR}")
endif()

add_executable (glslang::validator IMPORTED)
find_program (GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
set_property (TARGET glslang::validator PROPERTY IMPORTED_LOCATION "${GLSLANG_VALIDATOR}")

function (add_shaders_target TARGET)
  cmake_parse_arguments ("SHADER" "" "CHAPTER_NAME" "SOURCES" ${ARGN})
  set (SHADERS_DIR ${SHADER_CHAPTER_NAME}/shaders)
  add_custom_command (
    OUTPUT ${SHADERS_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    )
  set (SHADERS ${SHADERS_DIR}/frag.spv ${SHADERS_DIR}/vert.spv)
  # Some chapters may have compute shaders in addition to vertex and fragment shaders,
  # so we conditionally check this and add them to the target
  string(FIND "${SHADER_SOURCES}" "${CHAPTER_SHADER}.comp" COMPUTE_SHADER_INDEX)
  if (${COMPUTE_SHADER_INDEX} GREATER -1)  
    set (SHADERS ${SHADERS} ${SHADERS_DIR}/comp.spv)
  endif()    
  add_custom_command (
    OUTPUT ${SHADERS}
    COMMAND glslang::validator
    ARGS --target-env vulkan1.0 ${SHADER_SOURCES} --quiet
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
    COMMENT "Compiling Shaders"
    VERBATIM
    )
  add_custom_target (${TARGET} DEPENDS ${SHADERS})
endfunction ()

function (add_chapter CHAPTER_NAME)
  cmake_parse_arguments (CHAPTER "" "SHADER" "LIBS;TEXTURES;MODELS" ${ARGN})

  add_executable (${CHAPTER_NAME} ${CHAPTER_NAME}.cpp)
  set_target_properties (${CHAPTER_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_BINARY_DIR}/${CHAPTER_NAME}>)
  set_target_properties (${CHAPTER_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<1:${CMAKE_BINARY_DIR}/${CHAPTER_NAME}>)
  set_target_properties (${CHAPTER_NAME} PROPERTIES CXX_STANDARD 17)
  target_link_libraries (${CHAPTER_NAME} Vulkan::Vulkan ${GLFW_LIBRARIES})
  target_include_directories (${CHAPTER_NAME} PRIVATE ${STB_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${GLM_INCLUDE_DIR} ${TINYOBJ_INCLUDE_DIR})

  if (DEFINED CHAPTER_SHADER)
    set (CHAPTER_SHADER_TARGET ${CHAPTER_NAME}_shader)
    file (GLOB SHADER_SOURCES ${CHAPTER_SHADER}.frag ${CHAPTER_SHADER}.vert ${CHAPTER_SHADER}.comp)
    add_shaders_target (${CHAPTER_SHADER_TARGET} CHAPTER_NAME ${CHAPTER_NAME} SOURCES ${SHADER_SOURCES})
    add_dependencies (${CHAPTER_NAME} ${CHAPTER_SHADER_TARGET})
  endif ()
  if (DEFINED CHAPTER_LIBS)
    target_link_libraries (${CHAPTER_NAME} ${CHAPTER_LIBS})
  endif ()
  if (DEFINED CHAPTER_MODELS)
    file (COPY ${CHAPTER_MODELS} DESTINATION ${CMAKE_BINARY_DIR}/${CHAPTER_NAME}/models)
  endif ()
  if (DEFINED CHAPTER_TEXTURES)
    file (COPY ${CHAPTER_TEXTURES} DESTINATION ${CMAKE_BINARY_DIR}/${CHAPTER_NAME}/textures)
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

add_chapter (11_render_passes
  SHADER 09_shader_base)

add_chapter (12_graphics_pipeline_complete
  SHADER 09_shader_base)

add_chapter (13_framebuffers
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
  SHADER 18_shader_vertexbuffer)

add_chapter (19_vertex_buffer
  SHADER 18_shader_vertexbuffer)

add_chapter (20_staging_buffer
  SHADER 18_shader_vertexbuffer)

add_chapter (21_index_buffer
  SHADER 18_shader_vertexbuffer)

add_chapter (22_descriptor_set_layout
  SHADER 22_shader_ubo)

add_chapter (23_descriptor_sets
  SHADER 22_shader_ubo)

add_chapter (24_texture_image
  SHADER 22_shader_ubo
  TEXTURES ../images/texture.jpg)

add_chapter (25_sampler
  SHADER 22_shader_ubo
  TEXTURES ../images/texture.jpg)

add_chapter (26_texture_mapping
  SHADER 26_shader_textures
  TEXTURES ../images/texture.jpg)

add_chapter (27_depth_buffering
  SHADER 27_shader_depth
  TEXTURES ../images/texture.jpg)

add_chapter (28_model_loading
  SHADER 27_shader_depth
  MODELS ../attachments/viking_room.obj
  TEXTURES ../attachments/viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (29_mipmapping
  SHADER 27_shader_depth
  MODELS ../attachments/viking_room.obj
  TEXTURES ../attachments/viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (30_multisampling
  SHADER 27_shader_depth
  MODELS ../attachments/viking_room.obj
  TEXTURES ../attachments/viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (31_compute_shader
  SHADER 31_shader_compute)
