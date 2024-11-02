// CPU operations for BFGS
#pragma once

enum class CPULayout { ROW_MAJOR, COL_MAJOR };

/**
 * mat = (n, m), vec = (m, 1), out = (n, 1)
 * COL_MAJOR: 等价于 mat = (m, n), vec = (n, 1), out = (m, 1)
 */
template <typename T>
static void _GEMVCpu(const T *mat, CPULayout layout, const T *vec, T *out,
                     int n, int m) {
  if (layout == CPULayout::ROW_MAJOR) {
    for (int i = 0; i < n; i++) {
      out[i] = 0.0;
      for (int j = 0; j < m; j++) {
        out[i] += mat[i * m + j] * vec[j];
      }
    }
  } else if (layout == CPULayout::COL_MAJOR) {
    memset(out, 0, m * sizeof(T));
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        out[i] += mat[j * m + i] * vec[j];
      }
    }
  }
}

static void _UpdateHCpu(double *H, const double *y, const double *yTH,
                        double sy, const double *s, const double *Hy, int n) {
  double dot = 0.0;
  for (int i = 0; i < n; i++)
    dot += yTH[i] * y[i];
  double tmp = (1.0 + dot / sy);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      H[i * n + j] +=
          (((tmp * s[i] * s[j]) - Hy[i] * s[j] - s[i] * yTH[j]) / sy);
}
