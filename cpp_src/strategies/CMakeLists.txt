cmake_minimum_required(VERSION 3.4)
project(runtime_patch CXX C)

set(pybind11_DIR "$ENV{CMAKE_PREFIX_PATH}/share/cmake/pybind11")
find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`

#add_library(memory_planning SHARED memory_planning.cpp greedy_by_size.cpp)
#target_link_libraries(memory_planning PRIVATE pybind11::embed dl)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(memory_planning memory_planning.cpp greedy_by_size.cpp)
