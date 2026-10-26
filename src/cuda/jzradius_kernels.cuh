// Shared math for jzradius forward/backward radius traversal.
//
// The same code is compiled twice:
//   * with nvcc for the CUDA FFI handlers (`__device__` kernels), and
//   * with g++ for the CPU mirror used by the test suite (`__host__` loops).
//
// Keeping one implementation guarantees the CPU tests exercise exactly the
// arithmetic that the GPU kernels run (up to reduction order / atomics).
//
// Conventions (PLAN.md 6, 11):
//   * Positions are in tree-sorted order; work items index into those arrays.
//   * Self-interaction work items skip i == j, and skip j <= i when
//     `unique_pairs != 0` (pairs are then counted once in sorted-index space).
//   * `pair_scale` is 1 for unique/cross, 0.5 for directed_half autocorrelation.
//   * Distances use the componentwise minimum-image convention with a scalar
//     cubic boxsize; `boxsize <= 0` disables the periodic wrap.
//
// No per-bin scratch arrays are used: softmax statistics (max, denominator,
// A, B, C) are recomputed in streaming passes, so any bin count works.
#ifndef JZRADIUS_KERNELS_CUH_
#define JZRADIUS_KERNELS_CUH_

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define JZ_HD __host__ __device__
#else
#define JZ_HD
#endif

namespace jzradius {

enum : int32_t {
  kModeSoftmax = 0,
  kModeGaussian = 1,
};

struct Params {
  int64_t n_q;
  int64_t n_s;
  int32_t n_bins;
  int32_t n_work;
  int32_t self_query;    // 1: query and source are the same particle array
  int32_t unique_pairs;  // 1: count each unordered pair once (skip j <= i)
  int32_t mode;          // kModeSoftmax / kModeGaussian
  double r_max;
  double boxsize;        // 0 disables the wrap
  double temperature;
  double eps;
  double r_safe;
  double pair_scale;     // 1.0 or 0.5
};

JZ_HD inline double mic_component(double d, double boxsize) {
  if (boxsize <= 0.0) return d;
  return d - boxsize * round(d / boxsize);
}

JZ_HD inline double logit(double r, double center, double alpha) {
  const double diff = r - center;
  return -alpha * diff * diff;
}

JZ_HD inline double logit_deriv(double r, double center, double alpha) {
  return -2.0 * alpha * (r - center);
}

// Normalization state for one pair: max logit and softmax denominator.
// Gaussian mode is *not* normalized (PLAN.md 6.2); since q_b <= 0 its weights
// are plain exp(q_b), which cannot overflow.
struct PairNorm {
  double m;
  double denom;
  bool softmax;
};

JZ_HD inline PairNorm pair_norm(double r, const double* centers, int n_bins,
                                double temperature, int32_t mode) {
  const double alpha = 1.0 / (temperature * temperature);
  PairNorm out;
  out.softmax = (mode == kModeSoftmax);
  out.m = 0.0;
  out.denom = 1.0;
  if (out.softmax) {
    out.m = -INFINITY;
    for (int b = 0; b < n_bins; ++b) {
      const double q = logit(r, centers[b], alpha);
      if (q > out.m) out.m = q;
    }
    out.denom = 0.0;
    for (int b = 0; b < n_bins; ++b) {
      out.denom += exp(logit(r, centers[b], alpha) - out.m);
    }
  }
  return out;
}

JZ_HD inline double bin_weight(double r, double center, double temperature,
                               const PairNorm& n) {
  const double alpha = 1.0 / (temperature * temperature);
  const double q = logit(r, center, alpha);
  if (!n.softmax) return exp(q);
  return exp(q - n.m) / n.denom;
}

// Streaming softmax/Gaussian statistics for one pair.  Returns A, B, C:
//   A = sum_b g_b W_b, B = sum_b W_b q'_b, C = sum_b g_b W_b q'_b.
template <typename T>
JZ_HD inline void pair_statistics(double r, const double* centers, int n_bins,
                                  double temperature, int32_t mode,
                                  const T* g, double* A, double* B,
                                  double* C, double* weight_sum,
                                  double* max_logit) {
  const double alpha = 1.0 / (temperature * temperature);
  const PairNorm n = pair_norm(r, centers, n_bins, temperature, mode);
  double a = 0.0, bb = 0.0, c = 0.0;
  for (int b = 0; b < n_bins; ++b) {
    const double w = bin_weight(r, centers[b], temperature, n);
    const double qp = logit_deriv(r, centers[b], alpha);
    const double gb = static_cast<double>(g[b]);
    a += gb * w;
    bb += w * qp;
    c += gb * w * qp;
  }
  *A = a;
  *B = bb;
  *C = c;
  if (weight_sum != nullptr) *weight_sum = n.denom;
  if (max_logit != nullptr) *max_logit = n.m;
}

// Forward contribution of one accepted pair.
template <typename T>
JZ_HD inline void accumulate_pair_forward(double r, const double* centers,
                                          int n_bins, double temperature,
                                          int32_t mode, double pair_weight,
                                          double pair_scale, T* hist) {
  const PairNorm n = pair_norm(r, centers, n_bins, temperature, mode);
  const double scale = pair_weight * pair_scale;
  for (int b = 0; b < n_bins; ++b) {
    hist[b] += static_cast<T>(scale * bin_weight(r, centers[b], temperature, n));
  }
}

// Backward contribution of one accepted pair: position and weight cotangents.
template <typename T>
JZ_HD inline void accumulate_pair_backward(double r, const double d[3],
                                           const double* centers, int n_bins,
                                           double temperature, int32_t mode,
                                           double wq, double ws,
                                           double pair_scale,
                                           const T* g, T* gpos_q,
                                           T* gpos_s, T* gwq, T* gws) {
  double A = 0.0, B = 0.0, C = 0.0;
  pair_statistics(r, centers, n_bins, temperature, mode, g, &A, &B, &C,
                  nullptr, nullptr);
  const double dL_dr = (mode == kModeSoftmax) ? (C - A * B) : C;
  // The pair weight a_ij = wq * ws (times pair_scale) multiplies every
  // derivative of the pair: dL/dr_pair = a_ij * dL/dr.
  const double s = (pair_scale * wq * ws) * dL_dr / r;

  for (int k = 0; k < 3; ++k) {
    const T inc = static_cast<T>(s * d[k]);
    gpos_q[k] += inc;
    gpos_s[k] -= inc;
  }
  *gwq += static_cast<T>(pair_scale * ws * A);
  *gws += static_cast<T>(pair_scale * wq * A);
}

// Periodic lower bound between two axis-aligned boxes (PLAN.md 10.3).
JZ_HD inline double interval_gap(double a0, double a1, double b0, double b1,
                                 double boxsize) {
  if (boxsize <= 0.0) {
    if (a1 < b0) return b0 - a1;
    if (b1 < a0) return a0 - b1;
    return 0.0;
  }
  double best = INFINITY;
  for (int s = -1; s <= 1; ++s) {
    const double shift = s * boxsize;
    double gap = 0.0;
    if (a1 < b0 + shift) {
      gap = (b0 + shift) - a1;
    } else if (b1 + shift < a0) {
      gap = a0 - (b1 + shift);
    }
    if (gap < best) best = gap;
  }
  return best > 0.0 ? best : 0.0;
}

JZ_HD inline double box_lower_bound(const double* qlo, const double* qhi,
                                    const double* slo, const double* shi,
                                    double boxsize) {
  double r2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double g = interval_gap(qlo[k], qhi[k], slo[k], shi[k], boxsize);
    r2 += g * g;
  }
  return sqrt(r2);
}

}  // namespace jzradius

#endif  // JZRADIUS_KERNELS_CUH_
