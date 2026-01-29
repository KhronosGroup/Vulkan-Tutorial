cmake_minimum_required (VERSION 3.29)

project (VulkanTutorialRoot)
# This is to allow the CMakeLists.txt files in attachments/ to be accessed from root level
# Use command: cmake -S . -B build to configure from root level
add_subdirectory(attachments)

# Any other subdirectories can be added here