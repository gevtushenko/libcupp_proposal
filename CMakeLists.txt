cmake_minimum_required(VERSION 3.15)
project(libcupp_proposal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(libcupp_proposal main.cu cuda/numeric cuda/detail/__reduce)
target_include_directories(libcupp_proposal PRIVATE ${CUDA_INCLUDE_DIRS})
target_compile_options(libcupp_proposal PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -g>)