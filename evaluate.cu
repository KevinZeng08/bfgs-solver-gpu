// evaluate kernel correctness on different devices

#include <random>
#include <thrust/device_vector.h>
#include <vector>

#include "ops.cuh"
#include "ops.h"
#include "utils.cuh"

void EvaluateGENVCutlassAndCublas() {
  int n = 1024, m = 512;
  std::vector<double> mat(n * m), vec_row(m), vec_col(n);
  for (int i = 0; i < n * m; i++)
    mat[i] = i + 1;
  for (int i = 0; i < m; i++)
    vec_row[i] = i + 1;
  for (int i = 0; i < n; i++)
    vec_col[i] = i + 1;

  // cpu version
  std::vector<double> out_row(n), out_col(m);
  _GEMVCpu<double>(const_cast<double *>(mat.data()), CPULayout::ROW_MAJOR,
                   const_cast<double *>(vec_row.data()), out_row.data(), n, m);
  _GEMVCpu<double>(const_cast<double *>(mat.data()), CPULayout::COL_MAJOR,
                   const_cast<double *>(vec_col.data()), out_col.data(), n, m);

  // cutlass version
  thrust::device_vector<double> d_mat_cutlass(mat), d_vec_row_cutlass(vec_row);
  thrust::device_vector<double> d_out_row_cutlass(n);
  _GEMVCutlass<double>(
      thrust::raw_pointer_cast(d_mat_cutlass.data()), GPULayout::ROW_MAJOR,
      thrust::raw_pointer_cast(d_vec_row_cutlass.data()),
      thrust::raw_pointer_cast(d_out_row_cutlass.data()), n, m);
  // column major
  thrust::device_vector<double> d_mat_cutlass_col(mat),
      d_vec_col_cutlass(vec_col);
  thrust::device_vector<double> d_out_col_cutlass(m);
  _GEMVCutlass<double>(
      thrust::raw_pointer_cast(d_mat_cutlass_col.data()), GPULayout::COL_MAJOR,
      thrust::raw_pointer_cast(d_vec_col_cutlass.data()),
      thrust::raw_pointer_cast(d_out_col_cutlass.data()), n, m);

  HANDLE_ERROR(cudaGetLastError());

  // cublas version
  cublasHandle_t handle;
  cublasCreate(&handle);
  thrust::device_vector<double> d_mat_cublas(mat),
      d_vec_row_cublas(vec_row), d_vec_col_cublas(vec_col);
  thrust::device_vector<double> d_out_row_cublas(n), d_out_col_cublas(m);
  _GEMVCublas<double>(thrust::raw_pointer_cast(d_mat_cublas.data()),
                      GPULayout::ROW_MAJOR,
                      thrust::raw_pointer_cast(d_vec_row_cublas.data()),
                      thrust::raw_pointer_cast(d_out_row_cublas.data()), n, m,
                      handle);
  _GEMVCublas<double>(thrust::raw_pointer_cast(d_mat_cublas.data()),
                      GPULayout::COL_MAJOR,
                      thrust::raw_pointer_cast(d_vec_col_cublas.data()),
                      thrust::raw_pointer_cast(d_out_col_cublas.data()), n, m,
                      handle);
  cublasDestroy(handle);

  // check
  std::vector<double> out_row_cutlass(n);
  std::vector<double> out_col_cutlass(m);
  std::vector<double> out_col_cublas(m), out_row_cublas(n);
  thrust::copy(d_out_row_cutlass.begin(), d_out_row_cutlass.end(),
               out_row_cutlass.begin());
  thrust::copy(d_out_row_cublas.begin(), d_out_row_cublas.end(),
               out_row_cublas.begin());
  thrust::copy(d_out_col_cutlass.begin(), d_out_col_cutlass.end(),
               out_col_cutlass.begin());
  thrust::copy(d_out_col_cublas.begin(), d_out_col_cublas.end(),
               out_col_cublas.begin());
  for (int i = 0; i < n; i++) {
    assert(fabs(out_row[i] - out_row_cutlass[i]) < 1e-6);
    assert(fabs(out_row[i] - out_row_cublas[i]) < 1e-6);
  }
  for (int i = 0; i < m; i++) {
    // printf("%f %f\n", out_col[i], out_col_cublas[i]);
    // printf("%f %f\n", out_col[i], out_col_cutlass[i]);
    assert(fabs(out_col[i] - out_col_cublas[i]) < 1e-6);
    assert(fabs(out_col[i] - out_col_cutlass[i]) < 1e-6);
  }
}

void EvaluateGEMV() {
  int n = 1024, m = 512;
  // cpu version
  std::vector<double> mat(n * m), vec_row(m), vec_col(n);
  // random init
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  for (int i = 0; i < n * m; i++)
    mat[i] = dis(gen);
  for (int i = 0; i < m; i++)
    vec_row[i] = dis(gen);
  for (int i = 0; i < n; i++)
    vec_col[i] = dis(gen);
  std::vector<double> out_row(n), out_col(m);
  _GEMVCpu<double>(const_cast<double *>(mat.data()), CPULayout::ROW_MAJOR,
                   const_cast<double *>(vec_row.data()), out_row.data(), n, m);
  _GEMVCpu<double>(const_cast<double *>(mat.data()), CPULayout::COL_MAJOR,
                   const_cast<double *>(vec_col.data()), out_col.data(), n, m);

  // gpu version
  thrust::device_vector<double> d_mat(mat), d_vec_row(vec_row),
      d_vec_col(vec_col);
  thrust::device_vector<double> d_out_row(n), d_out_col(m);
  _GEMVKernel<double><<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      thrust::raw_pointer_cast(d_mat.data()), GPULayout::ROW_MAJOR,
      thrust::raw_pointer_cast(d_vec_row.data()),
      thrust::raw_pointer_cast(d_out_row.data()), n, m);
  _GEMVKernel<double><<<(m + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      thrust::raw_pointer_cast(d_mat.data()), GPULayout::COL_MAJOR,
      thrust::raw_pointer_cast(d_vec_col.data()),
      thrust::raw_pointer_cast(d_out_col.data()), n, m);

  HANDLE_ERROR(cudaGetLastError());

  // cutlass version
  thrust::device_vector<double> d_mat_cutlass(mat), d_vec_row_cutlass(vec_row),
      d_vec_col_cutlass(vec_col);
  thrust::device_vector<double> d_out_row_cutlass(n), d_out_col_cutlass(m);
  _GEMVCutlass<double>(
      thrust::raw_pointer_cast(d_mat_cutlass.data()), GPULayout::ROW_MAJOR,
      thrust::raw_pointer_cast(d_vec_row_cutlass.data()),
      thrust::raw_pointer_cast(d_out_row_cutlass.data()), n, m);
  _GEMVCutlass<double>(
      thrust::raw_pointer_cast(d_mat_cutlass.data()), GPULayout::COL_MAJOR,
      thrust::raw_pointer_cast(d_vec_col_cutlass.data()),
      thrust::raw_pointer_cast(d_out_col_cutlass.data()), n, m);

  HANDLE_ERROR(cudaGetLastError());

  // check
  std::vector<double> out_row2(n), out_col2(m), out_row_cutlass(n),
      out_col_cutlass(m);
  thrust::copy(d_out_row.begin(), d_out_row.end(), out_row2.begin());
  thrust::copy(d_out_col.begin(), d_out_col.end(), out_col2.begin());
  thrust::copy(d_out_row_cutlass.begin(), d_out_row_cutlass.end(),
               out_row_cutlass.begin());
  thrust::copy(d_out_col_cutlass.begin(), d_out_col_cutlass.end(),
               out_col_cutlass.begin());

  for (int i = 0; i < n; i++) {
    assert(fabs(out_row[i] - out_row2[i]) < 1e-6);
    assert(fabs(out_row[i] - out_row_cutlass[i]) < 1e-6);
  }
  for (int i = 0; i < m; i++) {
    assert(fabs(out_col[i] - out_col2[i]) < 1e-6);
    assert(fabs(out_col[i] - out_col_cutlass[i]) < 1e-6);
  }
}

// void EvaluateUpdateH() {
//   int n = 3490;
//   std::vector<double> y(n), yTH(n), s(n), Hy(n);
//   std::vector<double> H(n * n);
//   double sy = 1.0;
//   // random init
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_real_distribution<> dis(0, 1);
//   for (int i = 0; i < n; i++)
//     y[i] = dis(gen);
//   for (int i = 0; i < n; i++)
//     yTH[i] = dis(gen);
//   for (int i = 0; i < n; i++)
//     s[i] = dis(gen);
//   for (int i = 0; i < n; i++)
//     Hy[i] = dis(gen);
//   for (int i = 0; i < n; i++)
//     H[i * n + i] = 1.0;

//   thrust::device_vector<double> d_H(H);

//   // cpu version
//   _UpdateHCpu(H.data(), y.data(), yTH.data(), sy, s.data(), Hy.data(), n);

//   // gpu version
//   thrust::device_vector<double> d_y(y), d_yTH(yTH), d_s(s), d_Hy(Hy);
//   thrust::device_vector<double> d_dot(1);
//   _DotProductKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
//       thrust::raw_pointer_cast(d_yTH.data()),
//       thrust::raw_pointer_cast(d_y.data()),
//       thrust::raw_pointer_cast(d_dot.data()), n);
//   dim3 gridSize((n + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
//                 (n + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
//   dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//   _UpdateHKernel<double><<<gridSize, blockSize>>>(
//       thrust::raw_pointer_cast(d_H.data()),
//       d_dot[0],
//       thrust::raw_pointer_cast(d_yTH.data()), sy,
//       thrust::raw_pointer_cast(d_s.data()),
//       thrust::raw_pointer_cast(d_Hy.data()), n);

//   HANDLE_ERROR(cudaGetLastError());
//   // check
//   std::vector<double> H2(n * n);
//   thrust::copy(d_H.begin(), d_H.end(), H2.begin());
//   for (int i = 0; i < n * n; i++) {
//     assert(fabs(H[i] - H2[i]) < 1e-6);
//   }
// }

void EvaluateDotProduct() {
  int n = 3490;
  std::vector<double> y(n), yTH(n);
  // random init
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  for (int i = 0; i < n; i++)
    y[i] = dis(gen);
  for (int i = 0; i < n; i++)
    yTH[i] = dis(gen);

  // cpu version
  double dot_cpu = 0.0;
  for (int i = 0; i < n; i++)
    dot_cpu += yTH[i] * y[i];

  // gpu version
  thrust::device_vector<double> d_y(y), d_yTH(yTH), d_dot(1);
  _DotProductKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      thrust::raw_pointer_cast(d_yTH.data()),
      thrust::raw_pointer_cast(d_y.data()),
      thrust::raw_pointer_cast(d_dot.data()), n);

  HANDLE_ERROR(cudaGetLastError());

  double dot_gpu = d_dot[0];
  assert(fabs(dot_cpu - dot_gpu) < 1e-6);
}

int main() {
  // EvaluateDotProduct();
  // EvaluateGEMV();
  // EvaluateUpdateH();
  EvaluateGENVCutlassAndCublas();
  return 0;
}
