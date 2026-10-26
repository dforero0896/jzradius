// CPU mirror of the jzradius CUDA kernels.
//
// This is a *bit-for-bit faithful* port of `src/cuda/jzradius_cuda.cu` built on
// the same math header (`src/cuda/jzradius_kernels.cuh`).  It exists so the
// forward/backward FFI path can be exercised on a machine without a GPU
// (this cluster's compute nodes are currently unusable); once GPUs are
// available the identical Python code path runs the CUDA target.  It is not a
// different algorithm and it is not a fallback to the pure-JAX reference.
#include <cmath>
#include <cstdint>
#include <cstring>

#include "xla/ffi/api/ffi.h"
#include "../cuda/jzradius_kernels.cuh"

namespace ffi = xla::ffi;

namespace {

struct WorkItem {
  int32_t q0, q1, s0, s1;
};

inline jzradius::Params decode_params(
    int64_t n_q, int64_t n_s, int32_t n_bins, int32_t n_work, int32_t self_query,
    int32_t unique_pairs, int32_t mode, double r_max, double boxsize,
    double temperature, double eps, double r_safe, double pair_scale) {
  jzradius::Params p;
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

template <typename T>
inline bool pair_distance(const T* q, const T* s, const jzradius::Params& p,
                          double d[3], double* r) {
  double r2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    d[k] = jzradius::mic_component(static_cast<double>(q[k]) -
                                       static_cast<double>(s[k]),
                                   p.boxsize);
    r2 += d[k] * d[k];
  }
  if (r2 > p.r_max * p.r_max) return false;
  const double dist = std::sqrt(r2 + p.eps);
  if (dist < p.r_safe) return false;
  *r = dist;
  return true;
}

template <typename T>
void forward_cpu(const T* qpos, const T* spos, const double* centers,
                 const WorkItem* work, const T* wq, const T* ws,
                 const int32_t* orig, const jzradius::Params& p, T* hist,
                 int64_t n_pairs_expected) {
  (void)n_pairs_expected;
  for (int32_t w = 0; w < p.n_work; ++w) {
    const WorkItem item = work[w];
    for (int64_t i = item.q0; i < item.q1; ++i) {
      const T* qi = qpos + 3 * i;
      const double wq_i = static_cast<double>(wq[i]);
      const int64_t j_start =
          (p.self_query && p.unique_pairs) ? (i > item.s0 ? i : item.s0) : item.s0;
      for (int64_t j = j_start; j < item.s1; ++j) {
        if (p.self_query && (j == i || (p.unique_pairs && j <= i))) continue;
        double d[3], r;
        if (!pair_distance(qi, spos + 3 * j, p, d, &r)) continue;
        // Orientation: `unique` counts each unordered pair once.  For a
        // self-query the reference convention is i < j in *input* order, so
        // pick the roles from `orig` (sorted -> original index) rather than
        // from the tree order.  Without this, asymmetric weights
        // (wq != ws) would depend on the tree permutation (D010).
        double wq_sel = wq_i, ws_sel = static_cast<double>(ws[j]);
        if (p.self_query && p.unique_pairs && orig != nullptr &&
            orig[i] > orig[j]) {
          wq_sel = static_cast<double>(wq[j]);
          ws_sel = static_cast<double>(ws[i]);
        }
        const double a = wq_sel * ws_sel;
        jzradius::accumulate_pair_forward(r, centers, p.n_bins, p.temperature,
                                          p.mode, a, p.pair_scale, hist);
      }
    }
  }
}

template <typename T>
void backward_cpu(const T* qpos, const T* spos, const double* centers,
                  const WorkItem* work, const T* wq, const T* ws,
                  const int32_t* orig, const T* g, const jzradius::Params& p,
                  T* gpos_q, T* gpos_s, T* gwq, T* gws) {
  for (int32_t w = 0; w < p.n_work; ++w) {
    const WorkItem item = work[w];
    for (int64_t i = item.q0; i < item.q1; ++i) {
      const T* qi = qpos + 3 * i;
      const double wq_i = static_cast<double>(wq[i]);
      const int64_t j_start =
          (p.self_query && p.unique_pairs) ? (i > item.s0 ? i : item.s0) : item.s0;
      for (int64_t j = j_start; j < item.s1; ++j) {
        if (p.self_query && (j == i || (p.unique_pairs && j <= i))) continue;
        double d[3], r;
        if (!pair_distance(qi, spos + 3 * j, p, d, &r)) continue;
        double wq_sel = wq_i, ws_sel = static_cast<double>(ws[j]);
        int64_t wi = i, wj = j;
        if (p.self_query && p.unique_pairs && orig != nullptr &&
            orig[i] > orig[j]) {
          wq_sel = static_cast<double>(wq[j]);
          ws_sel = static_cast<double>(ws[i]);
          wi = j;
          wj = i;
          // The roles are now (query = j, source = i), so the separation
          // d = q_i - s_j must be flipped to stay q_wi - s_wj.
          for (int k = 0; k < 3; ++k) d[k] = -d[k];
        }
        jzradius::accumulate_pair_backward(
            r, d, centers, p.n_bins, p.temperature, p.mode, wq_sel, ws_sel,
            p.pair_scale, g, gpos_q + 3 * wi, gpos_s + 3 * wj, gwq + wi,
            gws + wj);
      }
    }
  }
}

inline bool check_common(const ffi::AnyBuffer& qpos, const ffi::AnyBuffer& spos,
                         const ffi::AnyBuffer& centers,
                         const ffi::AnyBuffer& work, const ffi::AnyBuffer& wq,
                         const ffi::AnyBuffer& ws,
                         const jzradius::Params& p) {
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

static ffi::Error ForwardCpu(ffi::AnyBuffer qpos, ffi::AnyBuffer spos,
                             ffi::AnyBuffer centers, ffi::AnyBuffer work,
                             ffi::AnyBuffer wq, ffi::AnyBuffer ws,
                             ffi::AnyBuffer orig,
                             int32_t self_query, int32_t unique_pairs,
                             int32_t mode, double r_max, double boxsize,
                             double temperature, double eps, double r_safe,
                             double pair_scale,
                             ffi::Result<ffi::AnyBuffer> hist) {
  const jzradius::Params p = decode_params(
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
  const WorkItem* items = static_cast<const WorkItem*>(work.untyped_data());
  // Output buffers are not guaranteed to be zero-initialised by XLA.
  std::memset(hist->untyped_data(), 0,
              p.n_bins * (qpos.element_type() == ffi::DataType::F64
                              ? sizeof(double)
                              : sizeof(float)));
  if (qpos.element_type() == ffi::DataType::F64) {
    forward_cpu<double>(static_cast<const double*>(qpos.untyped_data()),
                        static_cast<const double*>(spos.untyped_data()),
                        static_cast<const double*>(centers.untyped_data()),
                        items, static_cast<const double*>(wq.untyped_data()),
                        static_cast<const double*>(ws.untyped_data()),
                        static_cast<const int32_t*>(orig.untyped_data()), p,
                        static_cast<double*>(hist->untyped_data()), 0);
  } else {
    forward_cpu<float>(static_cast<const float*>(qpos.untyped_data()),
                       static_cast<const float*>(spos.untyped_data()),
                       static_cast<const double*>(centers.untyped_data()),
                       items, static_cast<const float*>(wq.untyped_data()),
                       static_cast<const float*>(ws.untyped_data()),
                       static_cast<const int32_t*>(orig.untyped_data()), p,
                       static_cast<float*>(hist->untyped_data()), 0);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusForwardCpu, ForwardCpu,
    ffi::Ffi::Bind()
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
        .Ret<ffi::AnyBuffer>());

static ffi::Error BackwardCpu(ffi::AnyBuffer qpos, ffi::AnyBuffer spos,
                              ffi::AnyBuffer centers, ffi::AnyBuffer work,
                              ffi::AnyBuffer wq, ffi::AnyBuffer ws,
                              ffi::AnyBuffer orig, ffi::AnyBuffer g, int32_t self_query,
                              int32_t unique_pairs, int32_t mode, double r_max,
                              double boxsize, double temperature, double eps,
                              double r_safe, double pair_scale,
                              ffi::Result<ffi::AnyBuffer> gpos_q,
                              ffi::Result<ffi::AnyBuffer> gpos_s,
                              ffi::Result<ffi::AnyBuffer> gwq,
                              ffi::Result<ffi::AnyBuffer> gws) {
  const jzradius::Params p = decode_params(
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
  const WorkItem* items = static_cast<const WorkItem*>(work.untyped_data());
  const size_t scalar = (qpos.element_type() == ffi::DataType::F64)
                            ? sizeof(double)
                            : sizeof(float);
  std::memset(gpos_q->untyped_data(), 0, 3 * p.n_q * scalar);
  std::memset(gpos_s->untyped_data(), 0, 3 * p.n_s * scalar);
  std::memset(gwq->untyped_data(), 0, p.n_q * scalar);
  std::memset(gws->untyped_data(), 0, p.n_s * scalar);
  if (qpos.element_type() == ffi::DataType::F64) {
    backward_cpu<double>(
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
    backward_cpu<float>(
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
    JzRadiusBackwardCpu, BackwardCpu,
    ffi::Ffi::Bind()
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
        .Ret<ffi::AnyBuffer>());

static ffi::Error NodeBoundsCpu(ffi::AnyBuffer qlo, ffi::AnyBuffer qhi,
                                ffi::AnyBuffer slo, ffi::AnyBuffer shi,
                                double boxsize,
                                ffi::Result<ffi::AnyBuffer> out) {
  const int64_t n_pairs = qlo.element_count() / 3;
  if (qlo.element_count() != 3 * n_pairs || out->element_count() != n_pairs ||
      out->element_type() != ffi::DataType::F64) {
    return ffi::Error::InvalidArgument("jzradius: node bounds shape mismatch");
  }
  const int is_f64 = qlo.element_type() == ffi::DataType::F64;
  for (int64_t idx = 0; idx < n_pairs; ++idx) {
    double q0[3], q1[3], s0[3], s1[3];
    for (int k = 0; k < 3; ++k) {
      if (is_f64) {
        q0[k] = static_cast<const double*>(qlo.untyped_data())[3 * idx + k];
        q1[k] = static_cast<const double*>(qhi.untyped_data())[3 * idx + k];
        s0[k] = static_cast<const double*>(slo.untyped_data())[3 * idx + k];
        s1[k] = static_cast<const double*>(shi.untyped_data())[3 * idx + k];
      } else {
        q0[k] = static_cast<const float*>(qlo.untyped_data())[3 * idx + k];
        q1[k] = static_cast<const float*>(qhi.untyped_data())[3 * idx + k];
        s0[k] = static_cast<const float*>(slo.untyped_data())[3 * idx + k];
        s1[k] = static_cast<const float*>(shi.untyped_data())[3 * idx + k];
      }
    }
    static_cast<double*>(out->untyped_data())[idx] =
        jzradius::box_lower_bound(q0, q1, s0, s1, boxsize);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusNodeBoundsCpu, NodeBoundsCpu,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()  // qlo
        .Arg<ffi::AnyBuffer>()  // qhi
        .Arg<ffi::AnyBuffer>()  // slo
        .Arg<ffi::AnyBuffer>()  // shi
        .Attr<double>("boxsize")
        .Ret<ffi::AnyBuffer>());

// ---------------------------------------------------------------------------
// One level of the dual-tree descent (CPU mirror of the CUDA kernel).
//
// Reads the current frontier of node pairs and writes the next frontier plus
// the leaf pairs discovered on this level.  `counts` returns
// ``[n_next, n_work, overflow]``; `overflow` is set when either output buffer
// was too small, in which case the caller re-runs the level with larger
// buffers.
// ---------------------------------------------------------------------------

static ffi::Error WorkItemsCpu(
    ffi::AnyBuffer qstart, ffi::AnyBuffer qcount, ffi::AnyBuffer qleft,
    ffi::AnyBuffer qright, ffi::AnyBuffer qlo, ffi::AnyBuffer qhi,
    ffi::AnyBuffer sstart, ffi::AnyBuffer scount, ffi::AnyBuffer sleft,
    ffi::AnyBuffer sright, ffi::AnyBuffer slo, ffi::AnyBuffer shi,
    ffi::AnyBuffer frontier, int32_t self_query, int32_t unique_pairs,
    double r_max, double prune_tol, double boxsize,
    ffi::Result<ffi::AnyBuffer> next_out, ffi::Result<ffi::AnyBuffer> work_out,
    ffi::Result<ffi::AnyBuffer> counts) {
  const int64_t n_q = qstart.element_count();
  const int64_t n_s = sstart.element_count();
  const int64_t n_in = frontier.element_count() / 2;
  const int64_t next_cap = next_out->element_count() / 2;
  const int64_t work_cap = work_out->element_count() / 4;
  if (qcount.element_count() != n_q || qleft.element_count() != n_q ||
      qright.element_count() != n_q || qlo.element_count() != 3 * n_q ||
      qhi.element_count() != 3 * n_q || scount.element_count() != n_s ||
      sleft.element_count() != n_s || sright.element_count() != n_s ||
      slo.element_count() != 3 * n_s || shi.element_count() != 3 * n_s ||
      frontier.element_count() != 2 * n_in ||
      next_out->element_count() != 2 * next_cap ||
      work_out->element_count() != 4 * work_cap ||
      counts->element_count() != 3) {
    return ffi::Error::InvalidArgument("jzradius: work-items shape mismatch");
  }
  const bool ints_ok = qstart.element_type() == ffi::DataType::S32 &&
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
                       counts->element_type() == ffi::DataType::S32;
  const bool boxes_ok = qlo.element_type() == ffi::DataType::F64 &&
                        qhi.element_type() == ffi::DataType::F64 &&
                        slo.element_type() == ffi::DataType::F64 &&
                        shi.element_type() == ffi::DataType::F64;
  if (!ints_ok || !boxes_ok) {
    return ffi::Error::InvalidArgument("jzradius: work-items dtype mismatch");
  }

  const int32_t* qs = static_cast<const int32_t*>(qstart.untyped_data());
  const int32_t* qc = static_cast<const int32_t*>(qcount.untyped_data());
  const int32_t* ql = static_cast<const int32_t*>(qleft.untyped_data());
  const int32_t* qr = static_cast<const int32_t*>(qright.untyped_data());
  const int32_t* ss = static_cast<const int32_t*>(sstart.untyped_data());
  const int32_t* sc = static_cast<const int32_t*>(scount.untyped_data());
  const int32_t* sl = static_cast<const int32_t*>(sleft.untyped_data());
  const int32_t* sr = static_cast<const int32_t*>(sright.untyped_data());
  const int32_t* fr = static_cast<const int32_t*>(frontier.untyped_data());
  const jzradius::NodeArrays q{qs,  qc,  ql,  qr,
                               static_cast<const double*>(qlo.untyped_data()),
                               static_cast<const double*>(qhi.untyped_data())};
  const jzradius::NodeArrays s{ss,  sc,  sl,  sr,
                               static_cast<const double*>(slo.untyped_data()),
                               static_cast<const double*>(shi.untyped_data())};

  int32_t* nxt = static_cast<int32_t*>(next_out->untyped_data());
  int32_t* wrk = static_cast<int32_t*>(work_out->untyped_data());
  int32_t* cnt = static_cast<int32_t*>(counts->untyped_data());
  int64_t n_next = 0, n_work = 0;
  int32_t overflow = 0;

  for (int64_t i = 0; i < n_in; ++i) {
    const int32_t nq = fr[2 * i];
    const int32_t ns = fr[2 * i + 1];
    if (nq < 0 || nq >= n_q || ns < 0 || ns >= n_s) {
      return ffi::Error::InvalidArgument(
          "jzradius: frontier node id out of range");
    }
    const jzradius::PairPlan p =
        jzradius::plan_pair(q, s, nq, ns, r_max, prune_tol, boxsize, self_query);
    if (p.action == jzradius::kPairEmit) {
      if (n_work < work_cap) {
        int32_t* w = wrk + 4 * n_work;
        w[0] = qs[nq];
        w[1] = qs[nq] + qc[nq];
        w[2] = ss[ns];
        w[3] = ss[ns] + sc[ns];
      } else {
        overflow = 1;
      }
      ++n_work;
      continue;
    }
    if (p.action == jzradius::kPairExpandQ ||
        p.action == jzradius::kPairExpandS) {
      const bool on_q = p.action == jzradius::kPairExpandQ;
      for (int32_t c = p.c0; c < p.c1; ++c) {
        if (n_next < next_cap) {
          nxt[2 * n_next] = on_q ? c : nq;
          nxt[2 * n_next + 1] = on_q ? ns : c;
        } else {
          overflow = 1;
        }
        ++n_next;
      }
      continue;
    }
    if (p.action == jzradius::kPairExpandDiag) {
      for (int32_t ia = p.c0; ia < p.c1; ++ia) {
        for (int32_t ib = ia; ib < p.c1; ++ib) {
          if (n_next < next_cap) {
            nxt[2 * n_next] = ia;
            nxt[2 * n_next + 1] = ib;
          } else {
            overflow = 1;
          }
          ++n_next;
          if (!unique_pairs && ia != ib) {
            if (n_next < next_cap) {
              nxt[2 * n_next] = ib;
              nxt[2 * n_next + 1] = ia;
            } else {
              overflow = 1;
            }
            ++n_next;
          }
        }
      }
    }
  }
  cnt[0] = static_cast<int32_t>(n_next);
  cnt[1] = static_cast<int32_t>(n_work);
  cnt[2] = overflow;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusWorkCpu, WorkItemsCpu,
    ffi::Ffi::Bind()
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
);
