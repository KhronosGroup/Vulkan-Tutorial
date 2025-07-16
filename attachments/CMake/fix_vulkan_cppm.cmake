# This script checks for vulkan.cppm files and fixes them if they have the old format
# without direct imports from the detail namespace

# Function to fix a vulkan.cppm file
function(fix_vulkan_cppm_file file_path)
  # Read the content of the file
  file(READ "${file_path}" VULKAN_CPPM_CONTENT)

  # Check if it's using the old format (without direct detail imports)
  if(VULKAN_CPPM_CONTENT MATCHES "using VULKAN_HPP_NAMESPACE::DispatchLoaderBase" AND
     NOT VULKAN_CPPM_CONTENT MATCHES "using VULKAN_HPP_NAMESPACE::detail::DispatchLoaderBase")
    message(STATUS "Fixing vulkan.cppm file at: ${file_path}")

    # Create a modified version that directly imports the detail symbols
    file(WRITE "${file_path}"
"// Modified vulkan.cppm file
module;
#include <vulkan/vulkan.hpp>
export module vulkan;
export namespace vk {
  // Import symbols from the main namespace
  using namespace VULKAN_HPP_NAMESPACE;

  // Import symbols from the detail namespace
  using VULKAN_HPP_NAMESPACE::detail::DispatchLoaderBase;
  using VULKAN_HPP_NAMESPACE::detail::DispatchLoaderDynamic;
  using VULKAN_HPP_NAMESPACE::detail::DispatchLoaderStatic;
  using VULKAN_HPP_NAMESPACE::detail::ObjectDestroy;
  using VULKAN_HPP_NAMESPACE::detail::ObjectDestroyShared;
  using VULKAN_HPP_NAMESPACE::detail::ObjectFree;
  using VULKAN_HPP_NAMESPACE::detail::ObjectFreeShared;
  using VULKAN_HPP_NAMESPACE::detail::ObjectRelease;
  using VULKAN_HPP_NAMESPACE::detail::ObjectReleaseShared;
  using VULKAN_HPP_NAMESPACE::detail::PoolFree;
  using VULKAN_HPP_NAMESPACE::detail::PoolFreeShared;
  using VULKAN_HPP_NAMESPACE::detail::createResultValueType;
  using VULKAN_HPP_NAMESPACE::detail::resultCheck;
  using VULKAN_HPP_NAMESPACE::detail::DynamicLoader;

  // Export detail namespace for other symbols
  namespace detail {
    using namespace VULKAN_HPP_NAMESPACE::detail;
  }

  // Export raii namespace
  namespace raii {
    using namespace VULKAN_HPP_RAII_NAMESPACE;

    // Import symbols from the detail namespace
    using VULKAN_HPP_NAMESPACE::detail::ContextDispatcher;
    using VULKAN_HPP_NAMESPACE::detail::DeviceDispatcher;
  }
}
")
    message(STATUS "Fixed vulkan.cppm file at: ${file_path}")
  else()
    message(STATUS "vulkan.cppm file at ${file_path} already has the correct format or is not a standard vulkan.cppm file")
  endif()
endfunction()

# Check if vulkan.cppm exists in /usr/include/vulkan
if(EXISTS "/usr/include/vulkan/vulkan.cppm")
  message(STATUS "Found vulkan.cppm in /usr/include/vulkan")

  # Try to fix the file directly
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E echo "Checking write permission for /usr/include/vulkan/vulkan.cppm"
    COMMAND_ECHO STDOUT
  )

  # Try to create a temporary file to check write permission
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E touch /usr/include/vulkan/vulkan.cppm.tmp
    RESULT_VARIABLE WRITE_RESULT
    ERROR_QUIET
  )

  if(WRITE_RESULT EQUAL 0)
    # We have write permission, remove the temporary file
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E remove /usr/include/vulkan/vulkan.cppm.tmp
    )

    # Fix the file directly
    fix_vulkan_cppm_file("/usr/include/vulkan/vulkan.cppm")
  else()
    message(STATUS "No write permission for /usr/include/vulkan/vulkan.cppm, creating a copy in the build directory")

    # Create a copy in the build directory that we can modify
    set(VULKAN_CPPM_COPY "${CMAKE_BINARY_DIR}/include/vulkan/vulkan.cppm")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/vulkan")
    file(COPY "/usr/include/vulkan/vulkan.cppm" DESTINATION "${CMAKE_BINARY_DIR}/include/vulkan")

    # Fix the copy
    fix_vulkan_cppm_file("${VULKAN_CPPM_COPY}")

    # Add the include directory to the include path
    include_directories(BEFORE "${CMAKE_BINARY_DIR}/include")

    # Set the VulkanHpp_CPPM_DIR variable
    set(VulkanHpp_CPPM_DIR "${CMAKE_BINARY_DIR}/include" CACHE PATH "Path to the directory containing vulkan.cppm" FORCE)
  endif()
endif()

# Check if vulkan.cppm exists in the Vulkan SDK
if(DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/include/vulkan/vulkan.cppm")
  message(STATUS "Found vulkan.cppm in Vulkan SDK")

  # Try to fix the file directly
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E echo "Checking write permission for $ENV{VULKAN_SDK}/include/vulkan/vulkan.cppm"
    COMMAND_ECHO STDOUT
  )

  # Try to create a temporary file to check write permission
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E touch "$ENV{VULKAN_SDK}/include/vulkan/vulkan.cppm.tmp"
    RESULT_VARIABLE WRITE_RESULT
    ERROR_QUIET
  )

  if(WRITE_RESULT EQUAL 0)
    # We have write permission, remove the temporary file
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E remove "$ENV{VULKAN_SDK}/include/vulkan/vulkan.cppm.tmp"
    )

    # Fix the file directly
    fix_vulkan_cppm_file("$ENV{VULKAN_SDK}/include/vulkan/vulkan.cppm")
  else()
    message(STATUS "No write permission for $ENV{VULKAN_SDK}/include/vulkan/vulkan.cppm, creating a copy in the build directory")

    # Create a copy in the build directory that we can modify
    set(VULKAN_CPPM_COPY "${CMAKE_BINARY_DIR}/include/vulkan/vulkan.cppm")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/vulkan")
    file(COPY "$ENV{VULKAN_SDK}/include/vulkan/vulkan.cppm" DESTINATION "${CMAKE_BINARY_DIR}/include/vulkan")

    # Fix the copy
    fix_vulkan_cppm_file("${VULKAN_CPPM_COPY}")

    # Add the include directory to the include path
    include_directories(BEFORE "${CMAKE_BINARY_DIR}/include")

    # Set the VulkanHpp_CPPM_DIR variable
    set(VulkanHpp_CPPM_DIR "${CMAKE_BINARY_DIR}/include" CACHE PATH "Path to the directory containing vulkan.cppm" FORCE)
  endif()
endif()
