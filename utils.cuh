#pragma once

#include <assert.h>
#include <omp.h>
#include <stdio.h>

// #define OLD_DISP
// #define OLD_JULIA
#define REAL double
// #define USE_CONST_MEM

// For the CUDA runtime routines (prefixed with "cuda_")
#include "helper_cuda.h"
#include <cuda_runtime.h>

#define HANDLE_ERROR checkCudaErrors

#define START_GPU                                                              \
  {                                                                            \
    cudaEvent_t start, stop;                                                   \
    float elapsedTime;                                                         \
    checkCudaErrors(cudaEventCreate(&start));                                  \
    checkCudaErrors(cudaEventCreate(&stop));                                   \
    checkCudaErrors(cudaEventRecord(start, 0));

#define END_GPU                                                                \
  checkCudaErrors(cudaEventRecord(stop, 0));                                   \
  checkCudaErrors(cudaEventSynchronize(stop));                                 \
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));            \
  printf("GPU Time used:  %3.1f ms\n", elapsedTime);                           \
  checkCudaErrors(cudaEventDestroy(start));                                    \
  checkCudaErrors(cudaEventDestroy(stop));                                     \
  }
