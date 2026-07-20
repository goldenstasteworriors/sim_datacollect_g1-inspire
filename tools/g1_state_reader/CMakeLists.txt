cmake_minimum_required(VERSION 3.16)
project(g1_read_lowstate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED SONIC_ROOT)
  message(FATAL_ERROR "Pass -DSONIC_ROOT=/path/to/GR00T-WholeBodyControl")
endif()

set(UNITREE_SDK_ROOT "${SONIC_ROOT}/gear_sonic_deploy/thirdparty/unitree_sdk2")
if(NOT EXISTS "${UNITREE_SDK_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR "Unitree SDK2 not found under ${UNITREE_SDK_ROOT}")
endif()

set(BUILD_EXAMPLES OFF CACHE BOOL "Do not build Unitree SDK examples" FORCE)
add_subdirectory("${UNITREE_SDK_ROOT}" unitree_sdk2_build EXCLUDE_FROM_ALL)

add_executable(g1_read_lowstate main.cpp)
target_link_libraries(g1_read_lowstate PRIVATE unitree_sdk2)
