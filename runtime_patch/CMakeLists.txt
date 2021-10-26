cmake_minimum_required(VERSION 3.4)
project(runtime_patch CXX C)

set(pybind11_DIR "/home/mlevental/miniconda3/envs/pytorch_shape_inference_memory_planning/share/cmake/pybind11")
find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`

add_library(runtime_patch SHARED runtime_patch.cpp)
target_link_libraries(runtime_patch PRIVATE pybind11::embed dl)

add_executable(demo demo.cpp)