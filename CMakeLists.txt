cmake_minimum_required(VERSION 3.6)
project(ParallelProgramming)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_CXX_FLAGS "-framework OpenCL")

set(SOURCE_FILES main.cpp)
add_executable(ParallelProgramming ${SOURCE_FILES})