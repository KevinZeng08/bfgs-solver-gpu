// CUDA operations for BFGS
#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 256
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

static __global__ void _DotProductKernel(const double *yTH, const double *y,
                                         double *dot, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // dot product yTH @ y
  // dot by block
  __shared__ double cache[BLOCK_SIZE];
  double sum_temp = 0.0;
  while (id < n) {
    sum_temp += yTH[id] * y[id];
    id += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = sum_temp;

  __syncthreads();

  // reduce
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      cache[threadIdx.x] += cache[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(dot, cache[0]);
  }
}

static __global__ void _UpdateHKernel(double *H, const double *dot,
                                      const double *yTH, double sy,
                                      const double *s, const double *Hy,
                                      int n) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;

  double tmp = (1.0 + dot[0] / sy);
  if (x_id < n && y_id < n) {
    H[x_id * n + y_id] += (((tmp * s[x_id] * s[y_id]) - Hy[x_id] * s[y_id] -
                            s[x_id] * yTH[y_id]) /
                           sy);
  }
}

enum class GPULayout { ROW_MAJOR, COL_MAJOR };

/**
 * mat = (n, m), vec = (m, 1), out = (n, 1)
 * COL_MAJOR: 等价于mat = (m, n), vec = (n, 1), out = (m, 1)
 */
template <typename T>
static __global__ void _GEMVKernel(const T *mat, GPULayout layout, const T *vec,
                                   T *out, int n, int m) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (layout == GPULayout::ROW_MAJOR) {
    if (id < n) {
      out[id] = 0.0;
      for (int col = 0; col < m; col++) {
        out[id] += mat[id * m + col] * vec[col];
      }
    }
  } else if (layout == GPULayout::COL_MAJOR) {
    if (id < m) {
      out[id] = 0.0;
      for (int row = 0; row < n; row++) {
        out[id] += mat[row * m + id] * vec[row];
      }
    }
  }
}