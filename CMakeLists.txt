cmake_minimum_required (VERSION 3.29)

project (VulkanTutorialRoot)
# This is to allow the CMakeLists.txt files in attachments/ to be accessed from root level
# Use command: cmake -S . -B build to configure from root level
# Then use command: cmake --build build to build from root level
# This will generate the build files in the build/ directory
# The build files will reference the source files in the attachments/ directory
# No need to change or delete any lines in attachments/ CMakeLists.txt files
# You can still build from attachments/ directory if needed
# Just navigate to attachments/ and run cmake and build commands as usual
add_subdirectory(attachments)

# Any other subdirectories can be added here
# Need to be carefull using CMAKE_SOURCE_DIR is points to the root level CMakeLists.txt
# Use CMAKE_CURRENT_SOURCE_DIR to point to the current directory instead
# For Simple Vulkan Engine project please uncomment the following line
# add_subdirectory(attachments/simple_engine)