cmake_minimum_required(VERSION 3.20)
project(galaxy_collision_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

add_executable(galaxy_sim
  src/main.cu
)

target_include_directories(galaxy_sim PRIVATE src)
set_target_properties(galaxy_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
