// Single-op plane-native dual-tree traversal (`jzradius_traverse`).
//
// One typed-XLA-FFI call performs the complete walk:
//   1. materialise canonical node arrays (start/count/left/right/lo/hi) from
//      jztree's packed planes, on the device;
//   2. bottom-up conservative AABB pass (leaves from particles, parents from
//      children);
//   3. BFS over planes with prefix-sum compaction of the frontier and of the
//      interaction list (deterministic order, no atomic appends);
//   4. report `n_work` on the device so the histogram kernels can skip the
//      unused tail of the padded interaction list.
//
// The plane iteration lives here in C++, not in Python (PLAN_ADDENDUM Rev 3,
// G1).  The only host synchronisation is one pinned read of the per-level
// counters; `counts[2]` reports how many of those reads were performed.
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"
#include "jzradius_traverse.cuh"

namespace ffi = xla::ffi;

using jzradius::NodeArrays;
using jzradius::kPairEmit;
namespace trav = jzradius::trav;

namespace {

constexpr int kThreads = 128;
constexpr int kMaxLevels = 32;  // packed planes we accept (jztree builds < 10)

// ---------------------------------------------------------------------------
// 1. canonical node arrays from the packed planes
// ---------------------------------------------------------------------------

__global__ void structure_kernel(trav::PlaneView pv, int32_t n_nodes,
                                 int32_t n_particles, int32_t* start,
                                 int32_t* count, int32_t* left, int32_t* right) {
  const int32_t stride = blockDim.x * gridDim.x;
  for (int32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n_nodes;
       k += stride) {
    bool leaf = false;
    int32_t c0 = -1, c1 = -1, st = 0, ct = 0;
    trav::node_children(pv, k, &c0, &c1, &leaf);
    trav::node_particle_range(pv, k, &st, &ct);
    if (st < 0) st = 0;
    if (ct < 0) ct = 0;
    if (st + ct > n_particles) ct = n_particles - st;
    start[k] = st;
    count[k] = ct;
    left[k] = leaf ? -1 : c0;
    right[k] = leaf ? -1 : c1;
  }
}

template <typename T>
__global__ void leaf_boxes_kernel(const T* pos, const int32_t* start,
                                  const int32_t* count, int32_t n_leaves,
                                  double* lo, double* hi) {
  const int32_t stride = blockDim.x * gridDim.x;
  for (int32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n_leaves;
       k += stride) {
    if (count[k] <= 0) {
      for (int c = 0; c < 3; ++c) {
        lo[3 * k + c] = 1e300;
        hi[3 * k + c] = -1e300;
      }
      continue;
    }
    trav::box_from_particles(pos, start[k], count[k], lo + 3 * k, hi + 3 * k);
  }
}

// One launch per plane, driven by the C++ loop: the children (plane p-1) are
// already boxed, so the union is exact.
__global__ void internal_boxes_kernel(const int32_t* left, const int32_t* right,
                                      int32_t id_base, int32_t n_plane,
                                      const double* clo, const double* chi,
                                      double* lo, double* hi) {
  // `id_base` is the plane's first *dense* node id; the plane spans n_plane ids.
  const int32_t stride = blockDim.x * gridDim.x;
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_plane;
       i += stride) {
    const int32_t k = id_base + i;
    trav::box_union_children(clo, chi, left[k], right[k], lo + 3 * k,
                             hi + 3 * k);
  }
}

// ---------------------------------------------------------------------------
// 2. frontier initialisation (top plane: diagonals for self, product for cross)
// ---------------------------------------------------------------------------

__global__ void init_frontier_kernel(const int32_t* n2n_ispl, int32_t top_base,
                                     int32_t n_top_q, int32_t n_top_s,
                                     int32_t self_query, int32_t* frontier) {
  const int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t n = self_query ? n_top_q : n_top_q * n_top_s;
  if (t >= n) return;
  int32_t i, j;
  if (self_query) {
    i = t;
    j = t;
  } else {
    i = t / n_top_s;
    j = t % n_top_s;
  }
  // Node ids are packed across planes: plane `top` starts at top_base.
  frontier[2 * t] = top_base + i;
  frontier[2 * t + 1] = top_base + j;
}

// ---------------------------------------------------------------------------
// 3. refine / scan / scatter
// ---------------------------------------------------------------------------

__global__ void refine_kernel(const NodeArrays qa, const NodeArrays sa,
                              trav::TravParams tp, const int32_t* frontier,
                              int32_t n_front, int32_t* child_cnt,
                              int32_t* work_cnt) {
  const int32_t stride = blockDim.x * gridDim.x;
  for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n_front;
       t += stride) {
    const trav::PairCounts pc =
        trav::refine_counts(qa, sa, tp, frontier[2 * t], frontier[2 * t + 1]);
    child_cnt[t] = pc.n_child;
    work_cnt[t] = pc.n_work;
  }
}

// Block scan of two count streams at once: writes inclusive scans and the
// per-block sums.  kThreads must be a power of two.
__global__ void scan_local_kernel(const int32_t* ca, const int32_t* cb,
                                  int32_t n, int32_t* ia, int32_t* ib,
                                  int32_t* blk_a, int32_t* blk_b) {
  __shared__ int32_t sa[kThreads];
  __shared__ int32_t sb[kThreads];
  const int32_t b = blockIdx.x;
  const int32_t t = threadIdx.x;
  const int32_t i = b * kThreads + t;
  sa[t] = (i < n) ? ca[i] : 0;
  sb[t] = (i < n) ? cb[i] : 0;
  __syncthreads();
  for (int32_t off = 1; off < kThreads; off <<= 1) {
    const int32_t pa = (t >= off) ? sa[t - off] : 0;
    const int32_t pb = (t >= off) ? sb[t - off] : 0;
    __syncthreads();
    sa[t] += pa;
    sb[t] += pb;
    __syncthreads();
  }
  if (i < n) {
    ia[i] = sa[t];
    ib[i] = sb[t];
  }
  if (t == kThreads - 1) {
    blk_a[b] = sa[t];
    blk_b[b] = sb[t];
  }
}

// Single thread: exclusive scan over the per-block sums.
__global__ void scan_blocks_kernel(const int32_t* blk_a, const int32_t* blk_b,
                                   int32_t nblk, int32_t* base_a,
                                   int32_t* base_b) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int32_t ra = 0, rb = 0;
  for (int32_t b = 0; b < nblk; ++b) {
    base_a[b] = ra;
    base_b[b] = rb;
    ra += blk_a[b];
    rb += blk_b[b];
  }
}

// off[i] = block_base + inclusive[i] - count[i]; totals[0] = grand total.
__global__ void scan_finalize_kernel(const int32_t* cnt, const int32_t* incl,
                                     const int32_t* base, int32_t n,
                                     int32_t* off, int32_t* total) {
  const int32_t stride = blockDim.x * gridDim.x;
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    off[i] = base[i / kThreads] + incl[i] - cnt[i];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0 && n > 0) {
    *total = base[(n - 1) / kThreads] + incl[n - 1];
  }
}

__global__ void scatter_kernel(const NodeArrays qa, const NodeArrays sa,
                               trav::TravParams tp, const int32_t* frontier,
                               int32_t n_front, const int32_t* child_off,
                               const int32_t* work_off, int32_t work_base,
                               int32_t* next_frontier, int32_t* work) {
  const int32_t stride = blockDim.x * gridDim.x;
  for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n_front;
       t += stride) {
    const int32_t nq = frontier[2 * t];
    const int32_t ns = frontier[2 * t + 1];
    const trav::PairCounts pc = trav::refine_counts(qa, sa, tp, nq, ns);
    int32_t base = child_off[t];
    for (int32_t k = 0; k < pc.n_child; ++k) {
      int32_t cq, cs;
      if (!trav::child_pair_at(pc, tp, k, nq, ns, &cq, &cs)) break;
      next_frontier[2 * (base + k)] = cq;
      next_frontier[2 * (base + k) + 1] = cs;
    }
    if (pc.n_work > 0) {
      const int32_t w = work_base + work_off[t];
      work[4 * w + 0] = qa.start[nq];
      work[4 * w + 1] = qa.start[nq] + qa.count[nq];
      work[4 * w + 2] = sa.start[ns];
      work[4 * w + 3] = sa.start[ns] + sa.count[ns];
      if (pc.n_work == 2) {  // mirrored orientation for ordered walks
        work[4 * (w + 1) + 0] = sa.start[ns];
        work[4 * (w + 1) + 1] = sa.start[ns] + sa.count[ns];
        work[4 * (w + 1) + 2] = qa.start[nq];
        work[4 * (w + 1) + 3] = qa.start[nq] + qa.count[nq];
      }
    }
  }
}

// Sentinel-fill the unused tail of the interaction list: the histogram kernels
// launch one block per row of the static capacity and skip negative rows.
__global__ void pad_kernel(int32_t* work, int32_t from, int32_t cap) {
  const int32_t stride = blockDim.x * gridDim.x;
  for (int32_t w = from + blockIdx.x * blockDim.x + threadIdx.x; w < cap;
       w += stride) {
    work[4 * w + 0] = -1;
    work[4 * w + 1] = -1;
    work[4 * w + 2] = -1;
    work[4 * w + 3] = -1;
  }
}

// ---------------------------------------------------------------------------
// 4. host driver
// ---------------------------------------------------------------------------

struct Scratch {
  int32_t* start = nullptr;
  int32_t* count = nullptr;
  int32_t* left = nullptr;
  int32_t* right = nullptr;
  double* lo = nullptr;
  double* hi = nullptr;
  int32_t* fa = nullptr;
  int32_t* fb = nullptr;
  int32_t* child_cnt = nullptr;
  int32_t* work_cnt = nullptr;
  int32_t* child_off = nullptr;
  int32_t* work_off = nullptr;
  int32_t* incl_a = nullptr;
  int32_t* incl_b = nullptr;
  int32_t* blk_a = nullptr;
  int32_t* blk_b = nullptr;
  int32_t* base_a = nullptr;
  int32_t* base_b = nullptr;
  int32_t* totals = nullptr;
};

ffi::Error alloc_scratch(Scratch* s, int32_t n_nodes, int32_t fcap,
                         int32_t nblk, cudaStream_t stream) {
  cudaError_t e = cudaSuccess;
  auto A = [&](auto** p, size_t n) {
    const cudaError_t r =
        cudaMallocAsync(reinterpret_cast<void**>(p), n * sizeof(**p), stream);
    if (r != cudaSuccess && e == cudaSuccess) e = r;
  };
  A(&s->start, n_nodes);
  A(&s->count, n_nodes);
  A(&s->left, n_nodes);
  A(&s->right, n_nodes);
  A(&s->lo, size_t(3) * n_nodes);
  A(&s->hi, size_t(3) * n_nodes);
  A(&s->fa, size_t(2) * fcap);
  A(&s->fb, size_t(2) * fcap);
  A(&s->child_cnt, fcap);
  A(&s->work_cnt, fcap);
  A(&s->child_off, fcap);
  A(&s->work_off, fcap);
  A(&s->incl_a, fcap);
  A(&s->incl_b, fcap);
  A(&s->blk_a, nblk);
  A(&s->blk_b, nblk);
  A(&s->base_a, nblk);
  A(&s->base_b, nblk);
  A(&s->totals, 4);
  if (e != cudaSuccess) {
    return ffi::Error::Internal("jzradius: traverse scratch allocation failed");
  }
  return ffi::Error::Success();
}

void free_scratch(Scratch* s, cudaStream_t stream) {
  void* ptrs[] = {s->start,    s->count,    s->left,    s->right,
                  s->lo,       s->hi,       s->fa,      s->fb,
                  s->child_cnt, s->work_cnt, s->child_off, s->work_off,
                  s->incl_a,   s->incl_b,   s->blk_a,   s->blk_b,
                  s->base_a,   s->base_b,   s->totals};
  for (void* p : ptrs) {
    if (p != nullptr) cudaFreeAsync(p, stream);
  }
  *s = Scratch{};
}

// Pinned staging (8 bytes per level read).
int32_t* pinned_counters(size_t n) {
  static int32_t* buf = nullptr;
  static size_t cap = 0;
  if (buf == nullptr || cap < n) {
    if (buf != nullptr) cudaFreeHost(buf);
    cap = n < 64 ? 64 : n;
    if (cudaHostAlloc(reinterpret_cast<void**>(&buf), cap * sizeof(int32_t),
                      cudaHostAllocDefault) != cudaSuccess) {
      buf = nullptr;
      cap = 0;
      return nullptr;
    }
  }
  return buf;
}

template <typename T>
ffi::Error traverse_typed(cudaStream_t stream, trav::PlaneView pv,
                          const T* pos, int32_t n_particles, int32_t self_query,
                          int32_t unique_pairs, double r_max, double prune_tol,
                          double boxsize, int32_t work_cap, int32_t* work,
                          int32_t* counts) {
  const int32_t nlevels = pv.nlevels;
  const int32_t n_nodes = trav::total_nodes(pv.n2n_ispl, nlevels);
  const int32_t n_leaves = trav::plane_nodes(pv, 0);
  if (n_nodes <= 0 || n_leaves <= 0) {
    const int32_t zero[3] = {0, 0, 0};
    cudaMemcpyAsync(counts, zero, sizeof(zero), cudaMemcpyHostToDevice, stream);
    return ffi::Error::Success();
  }

  trav::TravParams tp{};
  tp.n_particles = n_particles;
  tp.self_query = self_query;
  tp.unique_pairs = unique_pairs;
  tp.r_max = r_max;
  tp.prune_tol = prune_tol;
  tp.boxsize = boxsize;
  tp.work_cap = work_cap;

  // A level's frontier is bounded by the number of nodes on that plane; start
  // generously and grow on overflow (never truncate).
  int32_t fcap = n_nodes < 1024 ? 1024 : 4 * n_nodes;
  if (fcap > (1 << 24)) fcap = 1 << 24;

  int32_t* hostcnt = pinned_counters(8);
  if (hostcnt == nullptr) {
    return ffi::Error::Internal("jzradius: pinned host allocation failed");
  }

  const int32_t top = nlevels - 1;
  const int32_t n_top = trav::plane_nodes(pv, top);
  const int32_t n_init = self_query ? n_top : n_top * n_top;

  Scratch s;
  int32_t final_counts[3] = {0, 0, 0};
  bool done = false;

  for (int attempt = 0; attempt < 3 && !done; ++attempt) {
    if (n_init > fcap) {
      fcap = n_init;
      continue;
    }
    const int32_t nblk = (fcap + kThreads - 1) / kThreads;
    ffi::Error err = alloc_scratch(&s, n_nodes, fcap, nblk, stream);
    if (err.failure()) {
      free_scratch(&s, stream);
      return err;
    }
    tp.frontier_cap = fcap;
    const NodeArrays qa{s.start, s.count, s.left, s.right, s.lo, s.hi};
    const NodeArrays sa = qa;

    int grid = (n_nodes + kThreads - 1) / kThreads;
    structure_kernel<<<grid, kThreads, 0, stream>>>(pv, n_nodes, n_particles,
                                                    s.start, s.count, s.left,
                                                    s.right);
    grid = (n_leaves + kThreads - 1) / kThreads;
    leaf_boxes_kernel<T><<<grid, kThreads, 0, stream>>>(
        pos, s.start, s.count, n_leaves, s.lo, s.hi);
    for (int32_t p = 1; p < nlevels; ++p) {
      const int32_t n_plane = trav::plane_nodes(pv, p);
      if (n_plane <= 0) continue;
      grid = (n_plane + kThreads - 1) / kThreads;
      internal_boxes_kernel<<<grid, kThreads, 0, stream>>>(
          s.left, s.right, trav::plane_id_base(pv, p), n_plane, s.lo, s.hi,
          s.lo, s.hi);
    }

    grid = (n_init + kThreads - 1) / kThreads;
    init_frontier_kernel<<<grid, kThreads, 0, stream>>>(
        pv.n2n_ispl, trav::plane_id_base(pv, top), n_top, n_top, self_query,
        s.fa);

    int32_t n_front = n_init;
    int32_t work_base = 0;
    int32_t* cur = s.fa;
    int32_t* nxt = s.fb;
    bool overflow = false;
    int32_t steps = 0;

    for (int32_t step = 0; step <= nlevels && n_front > 0; ++step) {
      const int32_t curblk = (n_front + kThreads - 1) / kThreads;
      grid = curblk;
      refine_kernel<<<grid, kThreads, 0, stream>>>(qa, sa, tp, cur, n_front,
                                                   s.child_cnt, s.work_cnt);
      scan_local_kernel<<<curblk, kThreads, 0, stream>>>(
          s.child_cnt, s.work_cnt, n_front, s.incl_a, s.incl_b, s.blk_a,
          s.blk_b);
      scan_blocks_kernel<<<1, 1, 0, stream>>>(s.blk_a, s.blk_b, curblk,
                                              s.base_a, s.base_b);
      scan_finalize_kernel<<<curblk, kThreads, 0, stream>>>(
          s.child_cnt, s.incl_a, s.base_a, n_front, s.child_off, s.totals);
      scan_finalize_kernel<<<curblk, kThreads, 0, stream>>>(
          s.work_cnt, s.incl_b, s.base_b, n_front, s.work_off, s.totals + 1);
      if (cudaMemcpyAsync(hostcnt, s.totals, 2 * sizeof(int32_t),
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        free_scratch(&s, stream);
        return ffi::Error::Internal("jzradius: traverse counter copy failed");
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        free_scratch(&s, stream);
        return ffi::Error::Internal("jzradius: traverse level failed");
      }
      ++steps;
      const int32_t n_child = hostcnt[0];
      const int32_t n_emit = hostcnt[1];
      if (n_child > fcap || work_base + n_emit > work_cap) {
        overflow = true;
        break;
      }
      scatter_kernel<<<grid, kThreads, 0, stream>>>(
          qa, sa, tp, cur, n_front, s.child_off, s.work_off, work_base, nxt,
          work);
      work_base += n_emit;
      int32_t* tmp = cur;
      cur = nxt;
      nxt = tmp;
      n_front = n_child;
    }

    free_scratch(&s, stream);
    final_counts[0] = work_base;
    final_counts[1] = overflow ? 1 : 0;
    final_counts[2] = steps;
    done = !overflow;
    if (!done) fcap *= 2;
  }

  if (final_counts[1] == 0) {
    const int32_t tail = work_cap - final_counts[0];
    if (tail > 0) {
      const int grid = (tail + kThreads - 1) / kThreads;
      pad_kernel<<<grid, kThreads, 0, stream>>>(work, final_counts[0], work_cap);
    }
  }
  cudaMemcpyAsync(counts, final_counts, sizeof(final_counts),
                  cudaMemcpyHostToDevice, stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return ffi::Error::Internal("jzradius: traverse finalize failed");
  }
  // Overflow is reported through `counts[1]` (the Python side raises
  // WorkQueueOverflow); the op itself succeeds so the flag is observable.
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// 5. FFI handler
// ---------------------------------------------------------------------------

ffi::Error TraverseImpl(cudaStream_t stream, ffi::AnyBuffer n2n,
                        ffi::AnyBuffer n2n_ispl, ffi::AnyBuffer n2l,
                        ffi::AnyBuffer n2l_ispl, ffi::AnyBuffer pos,
                        ffi::Result<ffi::AnyBuffer> work,
                        ffi::Result<ffi::AnyBuffer> counts, int32_t self_query,
                        int32_t unique_pairs, double r_max, double prune_tol,
                        double boxsize) {
  if (n2n.element_type() != ffi::DataType::S32 ||
      n2n_ispl.element_type() != ffi::DataType::S32 ||
      n2l.element_type() != ffi::DataType::S32 ||
      n2l_ispl.element_type() != ffi::DataType::S32) {
    return ffi::Error::InvalidArgument(
        "jzradius: tree planes must be int32 buffers");
  }
  if (counts->element_type() != ffi::DataType::S32 ||
      counts->dimensions().size() != 1 || counts->dimensions()[0] != 3) {
    return ffi::Error::InvalidArgument(
        "jzradius: counts must be an int32 buffer of length 3");
  }
  if (work->element_type() != ffi::DataType::S32 ||
      work->dimensions().size() != 2 || work->dimensions()[1] != 4) {
    return ffi::Error::InvalidArgument(
        "jzradius: work must be an int32 (cap, 4) buffer");
  }
  const auto ptype = pos.element_type();
  if (ptype != ffi::DataType::F32 && ptype != ffi::DataType::F64) {
    return ffi::Error::InvalidArgument(
        "jzradius: positions must be float32 or float64");
  }
  const int64_t ispl_len = n2n_ispl.element_count();
  if (ispl_len < 2 || ispl_len > 64) {
    return ffi::Error::InvalidArgument(
        "jzradius: n2n_ispl must have 2..64 entries (levels + 1)");
  }
  if (n2l_ispl.element_count() != ispl_len) {
    return ffi::Error::InvalidArgument(
        "jzradius: n2n_ispl and n2l_ispl must have the same length");
  }
  if (pos.element_count() != 3 * int64_t(pos.dimensions()[0])) {
    return ffi::Error::InvalidArgument("jzradius: positions must be (N, 3)");
  }
  const int32_t nlevels = int32_t(ispl_len - 1);
  if (n2n.element_count() != n2l.element_count()) {
    return ffi::Error::InvalidArgument(
        "jzradius: n2n and n2l must be packed over the same planes");
  }

  int32_t* host = pinned_counters(64);
  if (host == nullptr) {
    return ffi::Error::Internal("jzradius: pinned host allocation failed");
  }
  if (cudaMemcpyAsync(host, n2n_ispl.untyped_data(),
                      size_t(ispl_len) * sizeof(int32_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return ffi::Error::Internal("jzradius: reading n2n_ispl failed");
  }
  int32_t base[kMaxLevels + 1];
  for (int32_t p = 0; p <= nlevels; ++p) base[p] = host[p];
  if (cudaMemcpyAsync(host, n2l_ispl.untyped_data(),
                      size_t(ispl_len) * sizeof(int32_t),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return ffi::Error::Internal("jzradius: reading n2l_ispl failed");
  }
  int32_t leaf_base[kMaxLevels + 1];
  for (int32_t p = 0; p <= nlevels; ++p) leaf_base[p] = host[p];

  const int32_t n_nodes = base[nlevels] - nlevels;
  if (n_nodes < 0 || int64_t(n_nodes) != n2n.element_count() - nlevels) {
    return ffi::Error::InvalidArgument(
        "jzradius: packed planes are inconsistent with n2n_ispl");
  }
  if (leaf_base[nlevels] - nlevels != n_nodes) {
    return ffi::Error::InvalidArgument(
        "jzradius: n2l planes do not match n2n planes");
  }

  trav::PlaneView pv{static_cast<const int32_t*>(n2n.untyped_data()), base,
                     static_cast<const int32_t*>(n2l.untyped_data()), leaf_base,
                     nlevels};

  const int32_t work_cap = int32_t(work->dimensions()[0]);
  int32_t* work_ptr = static_cast<int32_t*>(work->untyped_data());
  int32_t* counts_ptr = static_cast<int32_t*>(counts->untyped_data());
  const int64_t n_part = pos.dimensions()[0];

  if (ptype == ffi::DataType::F64) {
    return traverse_typed<double>(stream, pv,
                                  static_cast<const double*>(pos.untyped_data()),
                                  int32_t(n_part), self_query, unique_pairs,
                                  r_max, prune_tol, boxsize, work_cap, work_ptr,
                                  counts_ptr);
  }
  return traverse_typed<float>(stream, pv,
                               static_cast<const float*>(pos.untyped_data()),
                               int32_t(n_part), self_query, unique_pairs, r_max,
                               prune_tol, boxsize, work_cap, work_ptr,
                               counts_ptr);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusTraverse, TraverseImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()  // n2n
        .Arg<ffi::AnyBuffer>()  // n2n_ispl
        .Arg<ffi::AnyBuffer>()  // n2l
        .Arg<ffi::AnyBuffer>()  // n2l_ispl
        .Arg<ffi::AnyBuffer>()  // positions (Morton sorted)
        .Ret<ffi::AnyBuffer>()  // work items (cap, 4)
        .Ret<ffi::AnyBuffer>()  // counts (n_work, overflow, level reads)
        .Attr<int32_t>("self_query")
        .Attr<int32_t>("unique_pairs")
        .Attr<double>("r_max")
        .Attr<double>("prune_tol")
        .Attr<double>("boxsize"),
    {xla::ffi::Traits::kCmdBufferCompatible});
