cmake_minimum_required(VERSION 3.22.1)

project(simple_engine_android)

# Add the parent project's cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../attachments/CMake")

# Include the game-activity library
find_package(game-activity REQUIRED CONFIG)

# Set C++ standard to match the main project
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the simple_engine project as a subdirectory
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." simple_engine_build)

# Add the main native library
add_library(simple_engine_android SHARED
        game_activity_bridge.cpp
)

# Link against libraries
target_link_libraries(simple_engine_android
        SimpleEngine
        game-activity::game-activity
        android
        log
)
