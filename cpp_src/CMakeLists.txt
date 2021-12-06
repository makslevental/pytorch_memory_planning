cmake_minimum_required(VERSION 3.4...3.18)
project(pytorch_memory_planning)

# find jemalloc
find_package(PkgConfig REQUIRED)
pkg_check_modules(JEMALLOC REQUIRED jemalloc)
pkg_search_module(JEMALLOC REQUIRED jemalloc)
include_directories(${JEMALLOC_INCLUDE_DIRS})

add_subdirectory(jemalloc_bindings)
add_subdirectory(runtime_patch)
add_subdirectory(strategies)
