cmake_minimum_required(VERSION 3.4...3.18)
project(jemalloc_bindings)

set(pybind11_DIR "/home/mlevental/miniconda3/envs/pytorch_shape_inference_memory_planning/share/cmake/pybind11")
find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`
pybind11_add_module(jemalloc_bindings main.cpp)

# only for the VERSION_INFO define
target_compile_definitions(jemalloc_bindings
        PRIVATE VERSION_INFO=${VERSION_INFO})

# find jemalloc
find_package(PkgConfig REQUIRED)
pkg_check_modules(JEMALLOC REQUIRED jemalloc)
pkg_search_module(JEMALLOC REQUIRED jemalloc)

# make all connections for jemallc
include_directories(${JEMALLOC_INCLUDE_DIRS})
target_link_directories(jemalloc_bindings PRIVATE ${JEMALLOC_LIBRARY_DIRS})
target_link_libraries(jemalloc_bindings PRIVATE ${JEMALLOC_LIBRARIES})
