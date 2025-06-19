#include <cmath>

#define MIN(A, B) (((A) < (B)) ? (A) : (B))
#define ABS_UINT(A, N) (((A) > (N)) ? ((A)-(N)) : ((N)-(A)))
#define MAX(A, B) (((A) > (B)) ? (A) : (B))

int sign(unsigned int index, unsigned int n) {
  if (index == n) {
    return 0;
  }
  return index > n ? 1 : -1;
}

unsigned int minabs_index(unsigned int i, unsigned int j, unsigned int n) {
  int sign_i = sign(i, n);
  int sign_j = sign(j, n);

  if ((sign_i == 0) || (sign_j == 0)) {
    return n;
  }
  unsigned int minabs = MIN(ABS_UINT(i, n), ABS_UINT(j, n));

  if (sign_i * sign_j > 0) {
    return minabs + n;
  } else {
    return n - minabs;
  }
}

unsigned int sum_index(unsigned int i, unsigned int j, unsigned int n) {
  if (i + j < n) {
    return 0;
  }

  if (i + j > 3 * n) {
    return 2 * n;
  }
  return i + j - n;
}

/// Min-sum activation function
extern "C"
void minsum_abs(const double *pmf1,
                const double *pmf2,
                double       *pmf_out,
                unsigned int  n_points) {
  unsigned int n_total = 2 * n_points + 1;

  for (unsigned int i = 0; i < n_total; i++) {
    for (unsigned int j = 0; j < n_total; j++) {
      double probability = pmf1[i] * pmf2[j];
      unsigned int point = minabs_index(i, j, n_points);
      pmf_out[point] += probability;
    }
  }
}

/// Convolution with saturation
extern "C"
void convolve_thr(const double *pmf1,
                  const double *pmf2,
                  double       *pmf_out,
                  unsigned int  n_points) {
  unsigned int n_total = 2 * n_points + 1;

  for (unsigned int i = 0; i < n_total; i++) {
    for (unsigned int j = 0; j < n_total; j++) {
      double probability = pmf1[i] * pmf2[j];
      unsigned int point = sum_index(i, j, n_points);
      pmf_out[point] += probability;
    }
  }
}

/// Apply scale to LLR distribution
extern "C"
void scale(const double *pmf_in,
           double       *pmf_out,
           unsigned int  n_points,
           double        llr_scale) {
  unsigned int max_index = 2 * n_points;

  for (unsigned int i = 0; i < 2 * n_points + 1; i++) {
    // Use 1st order interpolation
    double new_point       = ((double)i - n_points) * llr_scale + n_points;
    unsigned int new_index = std::floor(new_point);

    double alpha = 1.0 - new_point + std::floor(new_point);

    pmf_out[new_index]                     += pmf_in[i] * alpha;
    pmf_out[MIN(new_index + 1, max_index)] += pmf_in[i] * (1.0 - alpha);
  }
}

extern "C"
void gldpc_cn_op(const double *pmf_a,
                 const double *pmf_b,
                 double       *pmf_out,
                 unsigned int  n_points) {
  // MAX(a + b, 0) - MAX(a, b)
  unsigned int n_total = 2 * n_points + 1;

  for (unsigned int i = 0; i < n_total; i++) {
    for (unsigned int j = 0; j < n_total; j++) {
      unsigned int point = MAX(i + j, 2 * n_points) - MAX(i, j);
      pmf_out[point] += pmf_a[i] * pmf_b[j];
    }
  }
}
