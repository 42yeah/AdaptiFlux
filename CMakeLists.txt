cmake_minimum_required(VERSION 3.24.0)

project(AdaptiFlux CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(deps)
add_subdirectory(src)
