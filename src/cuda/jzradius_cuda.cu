// CUDA FFI handlers for jzradius (forward radius traversal, backward VJP,
// node lower bounds).  Registered from Python via `jax.ffi.register_ffi_target`
// with `platform="CUDA"` and `api_version=1`.
//
// Work items index leaf pairs in tree-sorted space:
//   work[w] = (q_start, q_end, s_start, s_end)  (exclusive ends, int32)
// Each block owns one work item; threads stride over the query particles and
// loop over the source particles.  Accumulation uses atomics, which is why the
// deterministic=True path is not implemented for the GPU kernels (see
// docs/limitations.md and the D-series deviation log).
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"
#include "jzradius_kernels.cuh"

namespace ffi = xla::ffi;

// The shared arithmetic header lives in `namespace jzradius` so the same code
// can be compiled by g++ for the CPU test mirror.
using jzradius::Params;
using jzradius::accumulate_pair_backward;
using jzradius::accumulate_pair_forward;
using jzradius::bin_weight;
using jzradius::box_lower_bound;
using jzradius::pair_norm;
using jzradius::pair_statistics;

namespace {

struct WorkItem {
  int32_t q0, q1, s0, s1;
};

inline Params decode_params(int64_t n_q, int64_t n_s, int32_t n_bins,
                            int32_t n_work, int32_t self_query,
                            int32_t unique_pairs, int32_t mode, double r_max,
                            double boxsize, double temperature, double eps,
                            double r_safe, double pair_scale) {
  Params p;
  p.n_q = n_q;
  p.n_s = n_s;
  p.n_bins = n_bins;
  p.n_work = n_work;
  p.self_query = self_query;
  p.unique_pairs = unique_pairs;
  p.mode = mode;
  p.r_max = r_max;
  p.boxsize = boxsize;
  p.temperature = temperature;
  p.eps = eps;
  p.r_safe = r_safe;
  p.pair_scale = pair_scale;
  return p;
}

// Distance for one pair.  Returns false when the pair must be skipped.
template <typename T>
__device__ inline bool pair_distance(const T* q, const T* s, const Params& p,
                                     double d[3], double* r) {
  double r2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    d[k] = jzradius::mic_component(static_cast<double>(q[k] - s[k]), p.boxsize);
    r2 += d[k] * d[k];
  }
  if (r2 > p.r_max * p.r_max) return false;
  const double dist = sqrt(r2 + p.eps);
  if (dist < p.r_safe) return false;
  *r = dist;
  return true;
}

// Forward bin loop: one thread computes the full histogram contribution.
template <typename T>
__global__ void forward_kernel_bins(const T* __restrict__ qpos,
                                    const T* __restrict__ spos,
                                    const double* __restrict__ centers,
                                    const WorkItem* __restrict__ work,
                                    const T* __restrict__ wq,
                                    const T* __restrict__ ws,
                                    const int32_t* __restrict__ orig, Params p,
                                    T* __restrict__ hist) {
  const WorkItem item = work[blockIdx.x];
  for (int64_t i = item.q0 + threadIdx.x; i < item.q1; i += blockDim.x) {
    const T* qi = qpos + 3 * i;
    const double a_i = static_cast<double>(wq[i]);
    const int64_t j_start =
          (p.self_query && p.unique_pairs) ? (i > item.s0 ? i : item.s0) : item.s0;
    for (int64_t j = j_start; j < item.s1; ++j) {
      if (p.self_query && (j == i || (p.unique_pairs && j <= i))) continue;
      double d[3], r;
      if (!pair_distance(qi, spos + 3 * j, p, d, &r)) continue;
      // Orient unordered pairs by *input* index (PLAN.md 6.5 / D010):
      // the reference convention is i < j in the caller's particle order.
      double wq_sel = a_i, ws_sel = static_cast<double>(ws[j]);
      if (p.self_query && p.unique_pairs && orig[i] > orig[j]) {
        wq_sel = static_cast<double>(wq[j]);
        ws_sel = static_cast<double>(ws[i]);
      }
      const double a = wq_sel * ws_sel;
      const jzradius::PairNorm norm =
          pair_norm(r, centers, p.n_bins, p.temperature, p.mode);
      for (int b = 0; b < p.n_bins; ++b) {
        const double w =
            bin_weight(r, centers[b], p.temperature, norm) * p.pair_scale;
        atomicAdd(&hist[b], static_cast<T>(a * w));
      }
    }
  }
}

template <typename T>
__global__ void backward_kernel(const T* __restrict__ qpos,
                                const T* __restrict__ spos,
                                const double* __restrict__ centers,
                                const WorkItem* __restrict__ work,
                                const T* __restrict__ wq,
                                const T* __restrict__ ws,
                                const int32_t* __restrict__ orig,
                                const T* __restrict__ g, Params p,
                                T* __restrict__ gpos_q,
                                T* __restrict__ gpos_s,
                                T* __restrict__ gwq, T* __restrict__ gws) {
  const WorkItem item = work[blockIdx.x];
  for (int64_t i = item.q0 + threadIdx.x; i < item.q1; i += blockDim.x) {
    const T* qi = qpos + 3 * i;
    const double wq_i = static_cast<double>(wq[i]);
    const int64_t j_start =
          (p.self_query && p.unique_pairs) ? (i > item.s0 ? i : item.s0) : item.s0;
    for (int64_t j = j_start; j < item.s1; ++j) {
      if (p.self_query && (j == i || (p.unique_pairs && j <= i))) continue;
      double d[3], r;
      if (!pair_distance(qi, spos + 3 * j, p, d, &r)) continue;
      double A = 0.0, B = 0.0, C = 0.0;
      pair_statistics(r, centers, p.n_bins, p.temperature, p.mode, g, &A, &B,
                      &C, nullptr, nullptr);
      double wq_sel = wq_i, ws_sel = static_cast<double>(ws[j]);
      int64_t wi = i, wj = j;
      if (p.self_query && p.unique_pairs && orig[i] > orig[j]) {
        wq_sel = static_cast<double>(wq[j]);
        ws_sel = static_cast<double>(ws[i]);
        wi = j;
        wj = i;
        // Roles are now (query = j, source = i); flip d to stay q_wi - s_wj.
        for (int k = 0; k < 3; ++k) d[k] = -d[k];
      }
      const double dL_dr =
          (p.mode == jzradius::kModeSoftmax) ? (C - A * B) : C;
      const double s = p.pair_scale * wq_sel * ws_sel * dL_dr / r;
      for (int k = 0; k < 3; ++k) {
        const T inc = static_cast<T>(s * d[k]);
        atomicAdd(&gpos_q[3 * wi + k], inc);
        atomicAdd(&gpos_s[3 * wj + k], -inc);
      }
      atomicAdd(&gwq[wi], static_cast<T>(p.pair_scale * ws_sel * A));
      atomicAdd(&gws[wj], static_cast<T>(p.pair_scale * wq_sel * A));
    }
  }
}

template <typename T>
__global__ void node_bounds_kernel(const T* __restrict__ qlo,
                                   const T* __restrict__ qhi,
                                   const T* __restrict__ slo,
                                   const T* __restrict__ shi,
                                   int64_t n_pairs, double boxsize,
                                   double* __restrict__ out) {
  const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (idx >= n_pairs) return;
  double q0[3], q1[3], s0[3], s1[3];
  for (int k = 0; k < 3; ++k) {
    q0[k] = static_cast<double>(qlo[3 * idx + k]);
    q1[k] = static_cast<double>(qhi[3 * idx + k]);
    s0[k] = static_cast<double>(slo[3 * idx + k]);
    s1[k] = static_cast<double>(shi[3 * idx + k]);
  }
  out[idx] = jzradius::box_lower_bound(q0, q1, s0, s1, boxsize);
}

struct Buffers {
  const void* qpos = nullptr;
  const void* spos = nullptr;
  const void* centers = nullptr;
  const void* work = nullptr;
  const void* wq = nullptr;
  const void* ws = nullptr;
};

inline bool check_common(const ffi::AnyBuffer& qpos, const ffi::AnyBuffer& spos,
                         const ffi::AnyBuffer& centers,
                         const ffi::AnyBuffer& work, const ffi::AnyBuffer& wq,
                         const ffi::AnyBuffer& ws, const Params& p) {
  if (qpos.element_count() != 3 * p.n_q || spos.element_count() != 3 * p.n_s) {
    return false;
  }
  if (centers.element_count() != p.n_bins) return false;
  if (work.element_count() != 4 * static_cast<int64_t>(p.n_work)) return false;
  if (wq.element_count() != p.n_q || ws.element_count() != p.n_s) return false;
  return qpos.element_type() == spos.element_type() &&
         qpos.element_type() == wq.element_type() &&
         qpos.element_type() == ws.element_type() &&
         (qpos.element_type() == ffi::DataType::F32 ||
          qpos.element_type() == ffi::DataType::F64);
}

}  // namespace

/* ------------------------------- forward -------------------------------- */
static ffi::Error ForwardImpl(cudaStream_t stream, ffi::AnyBuffer qpos,
                              ffi::AnyBuffer spos, ffi::AnyBuffer centers,
                              ffi::AnyBuffer work, ffi::AnyBuffer wq,
                              ffi::AnyBuffer ws, ffi::AnyBuffer orig,
                              int32_t self_query,
                              int32_t unique_pairs, int32_t mode, double r_max,
                              double boxsize, double temperature, double eps,
                              double r_safe, double pair_scale,
                              ffi::Result<ffi::AnyBuffer> hist) {
  const Params p = decode_params(
      qpos.dimensions()[0], spos.dimensions()[0],
      static_cast<int32_t>(centers.element_count()),
      static_cast<int32_t>(work.element_count() / 4), self_query, unique_pairs,
      mode, r_max, boxsize, temperature, eps, r_safe, pair_scale);
  if (!check_common(qpos, spos, centers, work, wq, ws, p) ||
      orig.element_count() != p.n_q ||
      orig.element_type() != ffi::DataType::S32 ||
      hist->element_count() != p.n_bins) {
    return ffi::Error::InvalidArgument("jzradius: forward buffer mismatch");
  }
  const size_t scalar = (qpos.element_type() == ffi::DataType::F64)
                            ? sizeof(double)
                            : sizeof(float);
  cudaMemsetAsync(hist->untyped_data(), 0, p.n_bins * scalar, stream);
  if (p.n_work == 0) return ffi::Error::Success();
  const WorkItem* items = static_cast<const WorkItem*>(work.untyped_data());
  const int threads = 128;
  const dim3 grid(p.n_work);
  if (qpos.element_type() == ffi::DataType::F64) {
    auto* h = static_cast<double*>(hist->untyped_data());
    forward_kernel_bins<double><<<grid, threads, 0, stream>>>(
        static_cast<const double*>(qpos.untyped_data()),
        static_cast<const double*>(spos.untyped_data()),
        static_cast<const double*>(centers.untyped_data()), items,
        static_cast<const double*>(wq.untyped_data()),
        static_cast<const double*>(ws.untyped_data()),
        static_cast<const int32_t*>(orig.untyped_data()), p, h);
  } else {
    auto* h = static_cast<float*>(hist->untyped_data());
    forward_kernel_bins<float><<<grid, threads, 0, stream>>>(
        static_cast<const float*>(qpos.untyped_data()),
        static_cast<const float*>(spos.untyped_data()),
        static_cast<const double*>(centers.untyped_data()), items,
        static_cast<const float*>(wq.untyped_data()),
        static_cast<const float*>(ws.untyped_data()),
        static_cast<const int32_t*>(orig.untyped_data()), p, h);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusForward, ForwardImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()  // qpos
        .Arg<ffi::AnyBuffer>()  // spos
        .Arg<ffi::AnyBuffer>()  // centers
        .Arg<ffi::AnyBuffer>()  // work
        .Arg<ffi::AnyBuffer>()  // wq
        .Arg<ffi::AnyBuffer>()  // ws
        .Arg<ffi::AnyBuffer>()  // orig (sorted -> original index)
        .Attr<int32_t>("self_query")
        .Attr<int32_t>("unique_pairs")
        .Attr<int32_t>("mode")
        .Attr<double>("r_max")
        .Attr<double>("boxsize")
        .Attr<double>("temperature")
        .Attr<double>("eps")
        .Attr<double>("r_safe")
        .Attr<double>("pair_scale")
        .Ret<ffi::AnyBuffer>(),
    {xla::ffi::Traits::kCmdBufferCompatible});

/* ------------------------------- backward ------------------------------- */
static ffi::Error BackwardImpl(cudaStream_t stream, ffi::AnyBuffer qpos,
                               ffi::AnyBuffer spos, ffi::AnyBuffer centers,
                               ffi::AnyBuffer work, ffi::AnyBuffer wq,
                               ffi::AnyBuffer ws, ffi::AnyBuffer orig,
                               ffi::AnyBuffer g,
                               int32_t self_query, int32_t unique_pairs,
                               int32_t mode, double r_max, double boxsize,
                               double temperature, double eps, double r_safe,
                               double pair_scale,
                               ffi::Result<ffi::AnyBuffer> gpos_q,
                               ffi::Result<ffi::AnyBuffer> gpos_s,
                               ffi::Result<ffi::AnyBuffer> gwq,
                               ffi::Result<ffi::AnyBuffer> gws) {
  const Params p = decode_params(
      qpos.dimensions()[0], spos.dimensions()[0],
      static_cast<int32_t>(centers.element_count()),
      static_cast<int32_t>(work.element_count() / 4), self_query, unique_pairs,
      mode, r_max, boxsize, temperature, eps, r_safe, pair_scale);
  if (!check_common(qpos, spos, centers, work, wq, ws, p) ||
      orig.element_count() != p.n_q ||
      orig.element_type() != ffi::DataType::S32 ||
      g.element_count() != p.n_bins) {
    return ffi::Error::InvalidArgument("jzradius: backward buffer mismatch");
  }
  if (p.n_work == 0) return ffi::Error::Success();
  const size_t scalar = (qpos.element_type() == ffi::DataType::F64)
                            ? sizeof(double)
                            : sizeof(float);
  cudaMemsetAsync(gpos_q->untyped_data(), 0, 3 * p.n_q * scalar, stream);
  cudaMemsetAsync(gpos_s->untyped_data(), 0, 3 * p.n_s * scalar, stream);
  cudaMemsetAsync(gwq->untyped_data(), 0, p.n_q * scalar, stream);
  cudaMemsetAsync(gws->untyped_data(), 0, p.n_s * scalar, stream);
  if (p.n_work == 0) return ffi::Error::Success();
  const WorkItem* items = static_cast<const WorkItem*>(work.untyped_data());
  const int threads = 128;
  const dim3 grid(p.n_work);
  if (qpos.element_type() == ffi::DataType::F64) {
    backward_kernel<double><<<grid, threads, 0, stream>>>(
        static_cast<const double*>(qpos.untyped_data()),
        static_cast<const double*>(spos.untyped_data()),
        static_cast<const double*>(centers.untyped_data()), items,
        static_cast<const double*>(wq.untyped_data()),
        static_cast<const double*>(ws.untyped_data()),
        static_cast<const int32_t*>(orig.untyped_data()),
        static_cast<const double*>(g.untyped_data()), p,
        static_cast<double*>(gpos_q->untyped_data()),
        static_cast<double*>(gpos_s->untyped_data()),
        static_cast<double*>(gwq->untyped_data()),
        static_cast<double*>(gws->untyped_data()));
  } else {
    backward_kernel<float><<<grid, threads, 0, stream>>>(
        static_cast<const float*>(qpos.untyped_data()),
        static_cast<const float*>(spos.untyped_data()),
        static_cast<const double*>(centers.untyped_data()), items,
        static_cast<const float*>(wq.untyped_data()),
        static_cast<const float*>(ws.untyped_data()),
        static_cast<const int32_t*>(orig.untyped_data()),
        static_cast<const float*>(g.untyped_data()), p,
        static_cast<float*>(gpos_q->untyped_data()),
        static_cast<float*>(gpos_s->untyped_data()),
        static_cast<float*>(gwq->untyped_data()),
        static_cast<float*>(gws->untyped_data()));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusBackward, BackwardImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()  // qpos
        .Arg<ffi::AnyBuffer>()  // spos
        .Arg<ffi::AnyBuffer>()  // centers
        .Arg<ffi::AnyBuffer>()  // work
        .Arg<ffi::AnyBuffer>()  // wq
        .Arg<ffi::AnyBuffer>()  // ws
        .Arg<ffi::AnyBuffer>()  // orig (sorted -> original index)
        .Arg<ffi::AnyBuffer>()  // g
        .Attr<int32_t>("self_query")
        .Attr<int32_t>("unique_pairs")
        .Attr<int32_t>("mode")
        .Attr<double>("r_max")
        .Attr<double>("boxsize")
        .Attr<double>("temperature")
        .Attr<double>("eps")
        .Attr<double>("r_safe")
        .Attr<double>("pair_scale")
        .Ret<ffi::AnyBuffer>()  // gpos_q
        .Ret<ffi::AnyBuffer>()  // gpos_s
        .Ret<ffi::AnyBuffer>()  // gwq
        .Ret<ffi::AnyBuffer>(),  // gws
    {xla::ffi::Traits::kCmdBufferCompatible});

/* ------------------------------ node bounds ----------------------------- */
static ffi::Error NodeBoundsImpl(cudaStream_t stream, ffi::AnyBuffer qlo,
                                 ffi::AnyBuffer qhi, ffi::AnyBuffer slo,
                                 ffi::AnyBuffer shi, double boxsize,
                                 ffi::Result<ffi::AnyBuffer> out) {
  const int64_t n_pairs = qlo.element_count() / 3;
  if (qlo.element_count() != 3 * n_pairs || out->element_count() != n_pairs ||
      out->element_type() != ffi::DataType::F64) {
    return ffi::Error::InvalidArgument("jzradius: node bounds shape mismatch");
  }
  const int threads = 128;
  const int blocks = static_cast<int>((n_pairs + threads - 1) / threads);
  if (qlo.element_type() == ffi::DataType::F32) {
    node_bounds_kernel<float><<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(qlo.untyped_data()),
        static_cast<const float*>(qhi.untyped_data()),
        static_cast<const float*>(slo.untyped_data()),
        static_cast<const float*>(shi.untyped_data()), n_pairs, boxsize,
        static_cast<double*>(out->untyped_data()));
  } else {
    node_bounds_kernel<double><<<blocks, threads, 0, stream>>>(
        static_cast<const double*>(qlo.untyped_data()),
        static_cast<const double*>(qhi.untyped_data()),
        static_cast<const double*>(slo.untyped_data()),
        static_cast<const double*>(shi.untyped_data()), n_pairs, boxsize,
        static_cast<double*>(out->untyped_data()));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusNodeBounds, NodeBoundsImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()  // qlo
        .Arg<ffi::AnyBuffer>()  // qhi
        .Arg<ffi::AnyBuffer>()  // slo
        .Arg<ffi::AnyBuffer>()  // shi
        .Attr<double>("boxsize")
        .Ret<ffi::AnyBuffer>(),
    {xla::ffi::Traits::kCmdBufferCompatible});

/* ------------------------------ work queue ------------------------------ */
// One BFS level of the dual-tree descent, entirely on the device (D012/D013).
// Each thread owns one frontier item; children are appended with atomics.  The
// caller loops over levels and re-runs a level with larger buffers whenever
// `counts[2]` (overflow) is set, so nothing is ever silently dropped.
__global__ void work_items_kernel(jzradius::NodeArrays q, jzradius::NodeArrays s,
                                  const int32_t* __restrict__ frontier,
                                  int32_t n_in, int32_t next_cap,
                                  int32_t work_cap, int32_t self_query,
                                  int32_t unique_pairs, double r_max,
                                  double prune_tol, double boxsize,
                                  int32_t* __restrict__ next_out,
                                  int32_t* __restrict__ work_out,
                                  int32_t* __restrict__ counts) {
  const int32_t stride = blockDim.x * gridDim.x;
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_in; i += stride) {
    const int32_t nq = frontier[2 * i];
    const int32_t ns = frontier[2 * i + 1];
    const jzradius::PairPlan p = jzradius::plan_pair(
        q, s, nq, ns, r_max, prune_tol, boxsize, self_query);
    if (p.action == jzradius::kPairEmit) {
      const int32_t idx = atomicAdd(&counts[1], 1);
      if (idx < work_cap) {
        int32_t* w = work_out + 4 * idx;
        w[0] = q.start[nq];
        w[1] = q.start[nq] + q.count[nq];
        w[2] = s.start[ns];
        w[3] = s.start[ns] + s.count[ns];
      } else {
        atomicExch(&counts[2], 1);
      }
      continue;
    }
    if (p.action == jzradius::kPairExpandQ ||
        p.action == jzradius::kPairExpandS) {
      const bool on_q = p.action == jzradius::kPairExpandQ;
      for (int32_t c = p.c0; c < p.c1; ++c) {
        const int32_t idx = atomicAdd(&counts[0], 1);
        if (idx < next_cap) {
          next_out[2 * idx] = on_q ? c : nq;
          next_out[2 * idx + 1] = on_q ? ns : c;
        } else {
          atomicExch(&counts[2], 1);
        }
      }
      continue;
    }
    if (p.action == jzradius::kPairExpandDiag) {
      for (int32_t ia = p.c0; ia < p.c1; ++ia) {
        for (int32_t ib = ia; ib < p.c1; ++ib) {
          int32_t idx = atomicAdd(&counts[0], 1);
          if (idx < next_cap) {
            next_out[2 * idx] = ia;
            next_out[2 * idx + 1] = ib;
          } else {
            atomicExch(&counts[2], 1);
          }
          if (!unique_pairs && ia != ib) {
            idx = atomicAdd(&counts[0], 1);
            if (idx < next_cap) {
              next_out[2 * idx] = ib;
              next_out[2 * idx + 1] = ia;
            } else {
              atomicExch(&counts[2], 1);
            }
          }
        }
      }
    }
  }
}

static ffi::Error WorkItemsImpl(
    cudaStream_t stream, ffi::AnyBuffer qstart, ffi::AnyBuffer qcount,
    ffi::AnyBuffer qleft, ffi::AnyBuffer qright, ffi::AnyBuffer qlo,
    ffi::AnyBuffer qhi, ffi::AnyBuffer sstart, ffi::AnyBuffer scount,
    ffi::AnyBuffer sleft, ffi::AnyBuffer sright, ffi::AnyBuffer slo,
    ffi::AnyBuffer shi, ffi::AnyBuffer frontier, int32_t self_query,
    int32_t unique_pairs, double r_max, double prune_tol, double boxsize,
    ffi::Result<ffi::AnyBuffer> next_out, ffi::Result<ffi::AnyBuffer> work_out,
    ffi::Result<ffi::AnyBuffer> counts) {
  const int64_t n_q = qstart.element_count();
  const int64_t n_s = sstart.element_count();
  const int64_t n_in = frontier.element_count() / 2;
  const int64_t next_cap = next_out->element_count() / 2;
  const int64_t work_cap = work_out->element_count() / 4;
  const bool shapes_ok =
      qcount.element_count() == n_q && qleft.element_count() == n_q &&
      qright.element_count() == n_q && qlo.element_count() == 3 * n_q &&
      qhi.element_count() == 3 * n_q && scount.element_count() == n_s &&
      sleft.element_count() == n_s && sright.element_count() == n_s &&
      slo.element_count() == 3 * n_s && shi.element_count() == 3 * n_s &&
      frontier.element_count() == 2 * n_in &&
      next_out->element_count() == 2 * next_cap &&
      work_out->element_count() == 4 * work_cap &&
      counts->element_count() == 3;
  const bool dtypes_ok =
      qstart.element_type() == ffi::DataType::S32 &&
      qcount.element_type() == ffi::DataType::S32 &&
      qleft.element_type() == ffi::DataType::S32 &&
      qright.element_type() == ffi::DataType::S32 &&
      sstart.element_type() == ffi::DataType::S32 &&
      scount.element_type() == ffi::DataType::S32 &&
      sleft.element_type() == ffi::DataType::S32 &&
      sright.element_type() == ffi::DataType::S32 &&
      frontier.element_type() == ffi::DataType::S32 &&
      next_out->element_type() == ffi::DataType::S32 &&
      work_out->element_type() == ffi::DataType::S32 &&
      counts->element_type() == ffi::DataType::S32 &&
      qlo.element_type() == ffi::DataType::F64 &&
      qhi.element_type() == ffi::DataType::F64 &&
      slo.element_type() == ffi::DataType::F64 &&
      shi.element_type() == ffi::DataType::F64;
  if (!shapes_ok || !dtypes_ok) {
    return ffi::Error::InvalidArgument("jzradius: work-items buffer mismatch");
  }
  jzradius::NodeArrays q{
      static_cast<const int32_t*>(qstart.untyped_data()),
      static_cast<const int32_t*>(qcount.untyped_data()),
      static_cast<const int32_t*>(qleft.untyped_data()),
      static_cast<const int32_t*>(qright.untyped_data()),
      static_cast<const double*>(qlo.untyped_data()),
      static_cast<const double*>(qhi.untyped_data())};
  jzradius::NodeArrays s{
      static_cast<const int32_t*>(sstart.untyped_data()),
      static_cast<const int32_t*>(scount.untyped_data()),
      static_cast<const int32_t*>(sleft.untyped_data()),
      static_cast<const int32_t*>(sright.untyped_data()),
      static_cast<const double*>(slo.untyped_data()),
      static_cast<const double*>(shi.untyped_data())};
  // Output buffers are not guaranteed to be zero-initialised by XLA.
  cudaMemsetAsync(counts->untyped_data(), 0, 3 * sizeof(int32_t), stream);
  if (n_in <= 0) return ffi::Error::Success();
  const int threads = 128;
  int blocks = static_cast<int>((n_in + threads - 1) / threads);
  if (blocks > 65535) blocks = 65535;
  work_items_kernel<<<blocks, threads, 0, stream>>>(
      q, s, static_cast<const int32_t*>(frontier.untyped_data()),
      static_cast<int32_t>(n_in), static_cast<int32_t>(next_cap),
      static_cast<int32_t>(work_cap), self_query, unique_pairs, r_max,
      prune_tol, boxsize, static_cast<int32_t*>(next_out->untyped_data()),
      static_cast<int32_t*>(work_out->untyped_data()),
      static_cast<int32_t*>(counts->untyped_data()));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusWork, WorkItemsImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()  // qstart
        .Arg<ffi::AnyBuffer>()  // qcount
        .Arg<ffi::AnyBuffer>()  // qleft
        .Arg<ffi::AnyBuffer>()  // qright
        .Arg<ffi::AnyBuffer>()  // qlo
        .Arg<ffi::AnyBuffer>()  // qhi
        .Arg<ffi::AnyBuffer>()  // sstart
        .Arg<ffi::AnyBuffer>()  // scount
        .Arg<ffi::AnyBuffer>()  // sleft
        .Arg<ffi::AnyBuffer>()  // sright
        .Arg<ffi::AnyBuffer>()  // slo
        .Arg<ffi::AnyBuffer>()  // shi
        .Arg<ffi::AnyBuffer>()  // frontier (n_in, 2)
        .Attr<int32_t>("self_query")
        .Attr<int32_t>("unique_pairs")
        .Attr<double>("r_max")
        .Attr<double>("prune_tol")
        .Attr<double>("boxsize")
        .Ret<ffi::AnyBuffer>()  // next frontier (cap, 2)
        .Ret<ffi::AnyBuffer>()  // work items (cap, 4)
        .Ret<ffi::AnyBuffer>()  // counts (3,)
    ,
    {xla::ffi::Traits::kCmdBufferCompatible});
