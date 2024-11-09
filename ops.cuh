// CUDA operations for BFGS
#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemv.h"
#include "cutlass/gemm/kernel/gemv.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/tensor_ref.h"

#define BLOCK_SIZE 256
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
enum class GPULayout { ROW_MAJOR, COL_MAJOR };

// Cutlass GEMV code reference: https://github.com/NVIDIA/cutlass/issues/909
#define CUTLASS_CHECK(status)                                                  \
  {                                                                            \
    cutlass::Status error = status;                                            \
    if (error != cutlass::Status::kSuccess) {                                  \
      auto msg = std::string("[") + __FILE__ +                                 \
                 "] Got cutlass error: " + cutlassGetStatusString(error) +     \
                 " at: " + std::to_string(__LINE__);                           \
      std::cerr << msg << std::endl;                                           \
      throw std::runtime_error(msg);                                           \
    }                                                                          \
  }

template <typename T>
void _GEMVCutlass(const T *mat, GPULayout layout, const T *vec, T *out, int n,
                  int m) {
  using ElementAccumulator = T;
  using ElementA = T;
  using ElementB = T;
  using ElementC = T;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1, ElementAccumulator, ElementAccumulator>;

  if (layout == GPULayout::ROW_MAJOR) {
    using LayoutA = cutlass::layout::RowMajor;
    using GemvKernel =
        cutlass::gemm::kernel::Gemv<ElementA, LayoutA, ElementB, ElementC,
                                    ElementAccumulator, EpilogueOp>;
    using DeviceGemvInstance = cutlass::gemm::device::Gemv<GemvKernel>;

    T alpha = 1.0;
    T beta = 0.0;

    typename DeviceGemvInstance::Arguments args(
        {n, m}, {alpha, beta},
        cutlass::TensorRef<T, LayoutA>(const_cast<T *>(mat), LayoutA(m)), vec,
        out, out);
    DeviceGemvInstance gemv_op;
    cutlass::Status status;
    status = gemv_op.can_implement(args);
    CUTLASS_CHECK(status);
    status = gemv_op.initialize(args);
    CUTLASS_CHECK(status);

    status = gemv_op();
    CUTLASS_CHECK(status);
  } else if (layout == GPULayout::COL_MAJOR) { // transpose mat
    // TODO: For cutlass GEMV, RowMajor is much faster than ColumnMajor
    using LayoutA = cutlass::layout::RowMajor;
    using GemvKernel =
        cutlass::gemm::kernel::Gemv<ElementA, LayoutA, ElementB, ElementC,
                                    ElementAccumulator, EpilogueOp>;
    using DeviceGemvInstance = cutlass::gemm::device::Gemv<GemvKernel>;

    T alpha = 1.0;
    T beta = 0.0;

    typename DeviceGemvInstance::Arguments args(
        {m, n}, {alpha, beta},
        cutlass::TensorRef<T, LayoutA>(const_cast<T *>(mat), LayoutA(n)), vec,
        out, out);
    DeviceGemvInstance gemv_op;
    cutlass::Status status;
    status = gemv_op.can_implement(args);
    CUTLASS_CHECK(status);
    status = gemv_op.initialize(args);
    CUTLASS_CHECK(status);

    status = gemv_op();
    CUTLASS_CHECK(status);
  }
}

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

template <typename T>
static __global__ void _UpdateHKernel(T *H, const T *dot,
                                      const T *yTH, T sy,
                                      const T *s, const T *Hy,
                                      int n) {
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;
  int y_id = blockIdx.y * blockDim.y + threadIdx.y;
  T tmp = (1.0 + dot[0] / sy);

  while (x_id < n && y_id < n) {
    H[x_id * n + y_id] += (((tmp * s[x_id] * s[y_id]) - Hy[x_id] * s[y_id] -
                            s[x_id] * yTH[y_id]) /
                           sy);
    x_id += blockDim.x * gridDim.x;
    y_id += blockDim.y * gridDim.y;
  }
}
