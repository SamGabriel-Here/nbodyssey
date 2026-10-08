#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Abort with file/line on any failed CUDA call. Wrap every runtime/API call.
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                   __LINE__, cudaGetErrorString(err_));                       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                         \
  } while (0)
