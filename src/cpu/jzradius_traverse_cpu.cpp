// CPU mirror of `jzradius_traverse`: the same plane-native walk, executed as
// plain loops.  It exists so the production traversal logic (node
// materialisation, box pass, refine/scan/scatter) can be exercised on a
// machine without a GPU, and so the GPU kernels have a byte-for-byte reference
// for the interaction-list set.
//
// No device is involved: `PlatformStream` is absent from the binding and the
// buffers are host memory.
#include <cstring>

#include <vector>

#include "xla/ffi/api/ffi.h"
#include "jzradius_traverse.cuh"

namespace ffi = xla::ffi;

using jzradius::NodeArrays;
namespace trav = jzradius::trav;

namespace {

ffi::Error TraverseCpu(ffi::AnyBuffer n2n, ffi::AnyBuffer n2n_ispl,
                       ffi::AnyBuffer n2l, ffi::AnyBuffer n2l_ispl,
                       ffi::AnyBuffer pos, ffi::Result<ffi::AnyBuffer> work,
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
  const auto ptype = pos.element_type();
  if (ptype != ffi::DataType::F32 && ptype != ffi::DataType::F64) {
    return ffi::Error::InvalidArgument(
        "jzradius: positions must be float32 or float64");
  }
  if (counts->element_type() != ffi::DataType::S32 ||
      counts->element_count() != 3) {
    return ffi::Error::InvalidArgument(
        "jzradius: counts must be an int32 buffer of length 3");
  }
  if (work->element_type() != ffi::DataType::S32 ||
      work->dimensions().size() != 2 || work->dimensions()[1] != 4) {
    return ffi::Error::InvalidArgument(
        "jzradius: work must be an int32 (cap, 4) buffer");
  }

  int32_t* out_counts = static_cast<int32_t*>(counts->untyped_data());
  out_counts[0] = 0;
  out_counts[1] = 0;
  out_counts[2] = 0;

  const int32_t* n2n_p = static_cast<const int32_t*>(n2n.untyped_data());
  const int32_t* n2n_ispl_p =
      static_cast<const int32_t*>(n2n_ispl.untyped_data());
  const int32_t* n2l_p = static_cast<const int32_t*>(n2l.untyped_data());
  const int32_t* n2l_ispl_p =
      static_cast<const int32_t*>(n2l_ispl.untyped_data());
  const int32_t nlevels = int32_t(n2n_ispl.element_count()) - 1;
  if (nlevels < 1 || nlevels > 64) {
    return ffi::Error::InvalidArgument(
        "jzradius: n2n_ispl must have 2..65 entries (levels + 1)");
  }

  const int32_t n_nodes = trav::total_nodes(n2n_ispl_p, nlevels);
  const int32_t n_leaves = trav::plane_nodes(
      trav::PlaneView{n2n_p, n2n_ispl_p, n2l_p, n2l_ispl_p, nlevels}, 0);
  if (n_nodes <= 0 || n_leaves <= 0) {
    return ffi::Error::Success();
  }
  if (int64_t(n_nodes) != n2n.element_count() - nlevels) {
    return ffi::Error::InvalidArgument(
        "jzradius: packed planes are inconsistent with n2n_ispl");
  }

  const trav::PlaneView pv{n2n_p, n2n_ispl_p, n2l_p, n2l_ispl_p, nlevels};
  const int64_t n_part = pos.dimensions()[0];
  trav::TravParams tp{};
  tp.n_particles = int32_t(n_part);
  tp.self_query = self_query;
  tp.unique_pairs = unique_pairs;
  tp.r_max = r_max;
  tp.prune_tol = prune_tol;
  tp.boxsize = boxsize;
  const int32_t work_cap = int32_t(work->dimensions()[0]);
  tp.work_cap = work_cap;
  int32_t* work_p = static_cast<int32_t*>(work->untyped_data());

  std::vector<int32_t> start(n_nodes), count(n_nodes), left(n_nodes),
      right(n_nodes);
  std::vector<double> lo(size_t(3) * n_nodes), hi(size_t(3) * n_nodes);

  for (int32_t k = 0; k < n_nodes; ++k) {
    bool leaf = false;
    int32_t c0 = -1, c1 = -1, st = 0, ct = 0;
    trav::node_children(pv, k, &c0, &c1, &leaf);
    trav::node_particle_range(pv, k, &st, &ct);
    if (st < 0) st = 0;
    if (ct < 0) ct = 0;
    if (st + ct > tp.n_particles) ct = tp.n_particles - st;
    start[k] = st;
    count[k] = ct;
    left[k] = leaf ? -1 : c0;
    right[k] = leaf ? -1 : c1;
  }

  auto box_from_pos = [&](int32_t k) {
    if (count[k] <= 0) {
      for (int c = 0; c < 3; ++c) {
        lo[3 * k + c] = 1e300;
        hi[3 * k + c] = -1e300;
      }
      return;
    }
    if (ptype == ffi::DataType::F64) {
      trav::box_from_particles(static_cast<const double*>(pos.untyped_data()),
                               start[k], count[k], lo.data() + 3 * k,
                               hi.data() + 3 * k);
    } else {
      trav::box_from_particles(static_cast<const float*>(pos.untyped_data()),
                               start[k], count[k], lo.data() + 3 * k,
                               hi.data() + 3 * k);
    }
  };
  for (int32_t k = 0; k < n_leaves; ++k) box_from_pos(k);
  for (int32_t p = 1; p < nlevels; ++p) {
    const int32_t base = n2n_ispl_p[p] - p;  // dense id base of this plane
    const int32_t n_plane = n2n_ispl_p[p + 1] - n2n_ispl_p[p] - 1;
    for (int32_t k = base; k < base + n_plane; ++k) {
      trav::box_union_children(lo.data(), hi.data(), left[k], right[k],
                               lo.data() + 3 * k, hi.data() + 3 * k);
    }
  }

  const NodeArrays qa{start.data(), count.data(), left.data(), right.data(),
                      lo.data(), hi.data()};
  const NodeArrays sa = qa;

  const int32_t top = nlevels - 1;
  const int32_t n_top =
      trav::plane_nodes(pv, top);
  std::vector<int32_t> cur, nxt;
  // Node ids are packed across planes: plane `top` starts at its ispl entry.
  const int32_t top_base = n2n_ispl_p[top] - top;  // dense packed ids
  if (self_query) {
    for (int32_t i = 0; i < n_top; ++i) {
      cur.push_back(top_base + i);
      cur.push_back(top_base + i);
    }
  } else {
    for (int32_t i = 0; i < n_top; ++i) {
      for (int32_t j = 0; j < n_top; ++j) {
        cur.push_back(top_base + i);
        cur.push_back(top_base + j);
      }
    }
  }

  int32_t work_base = 0;
  int32_t steps = 0;
  int32_t overflow = 0;
  const int32_t n_front = int32_t(cur.size() / 2);

  std::vector<int32_t> child_cnt(n_front), work_cnt(n_front);
  std::vector<int32_t> child_off(n_front), work_off(n_front);
  int32_t p_n_front = n_front;
  while (p_n_front > 0) {
    ++steps;
    int32_t child_total = 0;
    int32_t emit_total = 0;
    for (int32_t t = 0; t < p_n_front; ++t) {
      const trav::PairCounts pc = trav::refine_counts(
          qa, sa, tp, cur[2 * t], cur[2 * t + 1]);
      child_cnt[t] = pc.n_child;
      work_cnt[t] = pc.n_work;
      child_off[t] = child_total;
      work_off[t] = emit_total;
      child_total += pc.n_child;
      emit_total += pc.n_work;
    }
    if (work_base + emit_total > work_cap) {
      overflow = 1;
      break;
    }
    nxt.assign(size_t(2) * (child_total > 0 ? child_total : 0), 0);
    for (int32_t t = 0; t < p_n_front; ++t) {
      const int32_t nq = cur[2 * t];
      const int32_t ns = cur[2 * t + 1];
      const trav::PairCounts pc = trav::refine_counts(qa, sa, tp, nq, ns);
      for (int32_t k = 0; k < pc.n_child; ++k) {
        int32_t cq, cs;
        if (!trav::child_pair_at(pc, tp, k, nq, ns, &cq, &cs)) break;
        nxt[2 * (child_off[t] + k)] = cq;
        nxt[2 * (child_off[t] + k) + 1] = cs;
      }
      if (pc.n_work > 0) {
        const int32_t w = work_base + work_off[t];
        work_p[4 * w + 0] = start[nq];
        work_p[4 * w + 1] = start[nq] + count[nq];
        work_p[4 * w + 2] = start[ns];
        work_p[4 * w + 3] = start[ns] + count[ns];
        if (pc.n_work == 2) {  // mirrored orientation for ordered walks
          work_p[4 * (w + 1) + 0] = start[ns];
          work_p[4 * (w + 1) + 1] = start[ns] + count[ns];
          work_p[4 * (w + 1) + 2] = start[nq];
          work_p[4 * (w + 1) + 3] = start[nq] + count[nq];
        }
      }
    }
    work_base += emit_total;
    cur.swap(nxt);
    p_n_front = child_total;
    if (p_n_front > int32_t(child_cnt.size())) {
      child_cnt.resize(p_n_front);
      work_cnt.resize(p_n_front);
      child_off.resize(p_n_front);
      work_off.resize(p_n_front);
    }
  }

  if (overflow == 0) {
    for (int32_t w = work_base; w < work_cap; ++w) {
      work_p[4 * w + 0] = -1;
      work_p[4 * w + 1] = -1;
      work_p[4 * w + 2] = -1;
      work_p[4 * w + 3] = -1;
    }
  }
  out_counts[0] = work_base;
  out_counts[1] = overflow;
  out_counts[2] = steps;
  // Overflow is reported through counts[1]; see jzradius_traverse.cu.
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JzRadiusTraverseCpu, TraverseCpu,
    ffi::Ffi::Bind()
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
        .Attr<double>("boxsize"));
