cmake_minimum_required(VERSION 3.15)
project(libcupp_proposal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/cuda_benchmark)

add_executable(libcupp_proposal main.cu cuda/numeric cuda/algorithm cuda/detail/__reduce)
target_include_directories(libcupp_proposal PRIVATE ${CUDA_INCLUDE_DIRS} external/cub)
target_compile_options(libcupp_proposal PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_link_libraries(libcupp_proposal PRIVATE cuda_benchmark)