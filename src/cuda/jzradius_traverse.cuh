// Plane-native dual-tree traversal, shared by the CUDA build (kernels) and the
// CPU mirror (plain loops).
//
// The production traversal consumes jztree's *packed* tree planes directly:
//
//   n2n       packed node->node split points, plane p occupies
//             [n2n_ispl[p], n2n_ispl[p+1]) and holds (nodes_p + 1) entries
//   n2l       packed node->leaf split points, same layout
//   plane 0   leaves; plane p > 0 has node i whose children are plane p-1
//             indices [n2n[p][i], n2n[p][i+1])
//
// Node ids in this file are *packed ids*: plane p's node i has packed id
// `n2n_ispl[p] + i`.  That makes the packed buffer itself the flat node array,
// so no host-side flattening is needed: the kernels materialise
// start/count/left/right/lo/hi on the device and the walk runs entirely in
// C++/CUDA (see docs/algorithm.md and PLAN_ADDENDUM.md Revision 3).
#pragma once

#include "jzradius_kernels.cuh"

namespace jzradius {
namespace trav {

// ---------------------------------------------------------------------------
// Packed planes
// ---------------------------------------------------------------------------

struct PlaneView {
  const int32_t* n2n;       // packed node->node splits
  const int32_t* n2n_ispl;  // (nlevels + 1) offsets into n2n
  const int32_t* n2l;       // packed node->leaf splits
  const int32_t* n2l_ispl;  // (nlevels + 1) offsets into n2l
  int32_t nlevels;
};

struct TravParams {
  int32_t n_particles;
  int32_t self_query;
  int32_t unique_pairs;
  int32_t frontier_cap;
  int32_t work_cap;
  double r_max;
  double prune_tol;
  double boxsize;
};

// Number of nodes on plane p.
JZ_HD inline int32_t plane_nodes(const PlaneView& pv, int32_t p) {
  return pv.n2n_ispl[p + 1] - pv.n2n_ispl[p] - 1;
}

// First packed node id of plane p.  Each plane stores (nodes_p + 1) split
// entries, so the raw ispl offsets leave one hole per plane; subtracting the
// plane index compacts the id space to the dense node numbering
// `total_nodes()` reports.
JZ_HD inline int32_t plane_id_base(const PlaneView& pv, int32_t p) {
  return pv.n2n_ispl[p] - p;
}

// Plane that holds packed node id k.
JZ_HD inline int32_t level_of(const PlaneView& pv, int32_t k) {
  int32_t p = 0;
  while (p + 1 < pv.nlevels && k >= plane_id_base(pv, p + 1)) ++p;
  return p;
}

// Children of packed node id k (packed ids, half open).  Leaves get c0 == c1 == -1.
JZ_HD inline void node_children(const PlaneView& pv, int32_t k, int32_t* c0,
                                int32_t* c1, bool* leaf) {
  const int32_t p = level_of(pv, k);
  if (p == 0) {
    *c0 = -1;
    *c1 = -1;
    *leaf = true;
    return;
  }
  const int32_t* s = pv.n2n + pv.n2n_ispl[p];
  const int32_t i = k - plane_id_base(pv, p);
  *c0 = plane_id_base(pv, p - 1) + s[i];
  *c1 = plane_id_base(pv, p - 1) + s[i + 1];
  *leaf = false;
}

// Particle range [start, start + count) of packed node id k in Morton order.
JZ_HD inline void node_particle_range(const PlaneView& pv, int32_t k,
                                      int32_t* start, int32_t* count) {
  const int32_t p = level_of(pv, k);
  const int32_t* s0 = pv.n2n + pv.n2n_ispl[0];
  if (p == 0) {
    *start = s0[k];
    *count = s0[k + 1] - s0[k];
    return;
  }
  const int32_t* l = pv.n2l + pv.n2l_ispl[p];
  const int32_t i = k - plane_id_base(pv, p);
  const int32_t l0 = l[i];
  const int32_t l1 = l[i + 1];
  *start = s0[l0];
  *count = s0[l1] - s0[l0];
}

// ---------------------------------------------------------------------------
// Bottom-up conservative boxes (jztree stores centres only, so we build them)
// ---------------------------------------------------------------------------

template <typename T>
JZ_HD inline void box_from_particles(const T* pos, int32_t start, int32_t count,
                                     double* lo, double* hi) {
  double l[3];
  double h[3];
  for (int k = 0; k < 3; ++k) {
    l[k] = static_cast<double>(pos[3 * start + k]);
    h[k] = l[k];
  }
  for (int32_t i = start + 1; i < start + count; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double x = static_cast<double>(pos[3 * i + k]);
      if (x < l[k]) l[k] = x;
      if (x > h[k]) h[k] = x;
    }
  }
  for (int k = 0; k < 3; ++k) {
    lo[k] = l[k];
    hi[k] = h[k];
  }
}

JZ_HD inline void box_union_children(const double* clo, const double* chi,
                                     int32_t c0, int32_t c1, double* lo,
                                     double* hi) {
  for (int k = 0; k < 3; ++k) {
    lo[k] = clo[3 * c0 + k];
    hi[k] = chi[3 * c0 + k];
  }
  for (int32_t c = c0 + 1; c < c1; ++c) {
    for (int k = 0; k < 3; ++k) {
      const double a = clo[3 * c + k];
      const double b = chi[3 * c + k];
      if (a < lo[k]) lo[k] = a;
      if (b > hi[k]) hi[k] = b;
    }
  }
}

// ---------------------------------------------------------------------------
// Refinement: same decision as the level-by-level queue (plan_pair), plus the
// number of objects each action will write at the next scatter step.
// ---------------------------------------------------------------------------

// Diagonal expansion writes the upper triangle of child pairs (ia <= ib).
// Ordered (directed_half) walks get their second orientation from
// `needs_mirror` on the non-diagonal expansions, which keeps every rectangle
// unique -- enumerating the lower triangle here as well would duplicate work.
JZ_HD inline int32_t diag_child_count(int32_t nc) { return nc * (nc + 1) / 2; }

// Enumerate the t-th diagonal child pair (upper triangle only).
JZ_HD inline bool diag_child_at(int32_t nc, int32_t t, int32_t* ia,
                                int32_t* ib) {
  int32_t c = 0;
  for (int32_t i = 0; i < nc; ++i) {
    for (int32_t j = i; j < nc; ++j, ++c) {
      if (c == t) {
        *ia = i;
        *ib = j;
        return true;
      }
    }
  }
  return false;
}

struct PairCounts {
  int32_t action;   // PairAction
  int32_t c0;       // first child packed id (expand actions)
  int32_t c1;       // one past the last child packed id
  int32_t n_child;  // objects written to the next frontier
  int32_t n_work;   // objects written to the interaction list
};

// Decide what one frontier pair does.  `q`/`s` are the flat node arrays that
// the caller keeps up to date on the device; `plan_pair` is the shared
// arithmetic header's decision function so the two queues cannot drift.
JZ_HD inline PairCounts refine_counts(const NodeArrays& q, const NodeArrays& s,
                                      const TravParams& tp, int32_t nq,
                                      int32_t ns) {
  PairCounts out{kPairPrune, 0, 0, 0, 0};
  if (q.count[nq] <= 0 || s.count[ns] <= 0) return out;
  const PairPlan plan =
      plan_pair(q, s, nq, ns, tp.r_max, tp.prune_tol, tp.boxsize, tp.self_query);
  out.action = plan.action;
  out.c0 = plan.c0;
  out.c1 = plan.c1;
  const int32_t nc = plan.c1 - plan.c0;
  switch (plan.action) {
    case kPairEmit:
      // Unordered walks emit one rectangle per leaf pair; ordered
      // (directed_half) walks need the mirrored rectangle too, since the
      // histogram kernel only enumerates one orientation per rectangle.
      out.n_work = (tp.unique_pairs || nq == ns) ? 1 : 2;
      break;
    case kPairExpandQ:
    case kPairExpandS:
      out.n_child = nc;
      break;
    case kPairExpandDiag:
      out.n_child = diag_child_count(nc);
      break;
    default:
      break;
  }
  return out;
}

// Write the t-th object of an expansion action as a frontier pair.  The node
// that is *not* expanded keeps the id of the current frontier item.
JZ_HD inline bool child_pair_at(const PairCounts& pc, const TravParams& tp,
                                int32_t t, int32_t nq, int32_t ns, int32_t* cq,
                                int32_t* cs) {
  const int32_t nc = pc.c1 - pc.c0;
  if (pc.action == kPairExpandQ) {
    *cq = pc.c0 + t;
    *cs = ns;
    return true;
  }
  if (pc.action == kPairExpandS) {
    *cq = nq;
    *cs = pc.c0 + t;
    return true;
  }
  if (pc.action == kPairExpandDiag) {
    int32_t ia, ib;
    if (!diag_child_at(nc, t, &ia, &ib)) return false;
    *cq = pc.c0 + ia;
    *cs = pc.c0 + ib;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Host-side layout checks (no device work)
// ---------------------------------------------------------------------------

// Total packed node count implied by the planes.
JZ_HOST inline int32_t total_nodes(const int32_t* n2n_ispl, int32_t nlevels) {
  return n2n_ispl[nlevels] - nlevels;
}

}  // namespace trav
}  // namespace jzradius
