"""CPU dual-tree traversal (M4 gate).

This is the reference implementation of the traversal *structure* that the
CUDA kernels will follow (PLAN.md 11 and Appendix B).  It is intentionally
written with an explicit stack and NumPy float64 so that

* the exact set of accepted ``(i, j)`` pairs can be compared with the
  brute-force oracle, and
* pruning/expansion decisions are auditable before they are ported to CUDA.

It is **not** the deliverable: passing these tests does not satisfy the
completion criterion (PLAN.md 1.5), which requires compiled CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bounds import interval_gap_reference
from .config import r_safe
from .errors import RadiusConfigError
from .tree_adapter import RadiusTree

__all__ = ["TraversalStats", "TraversalResult", "traverse_tree_pairs"]


@dataclass
class TraversalStats:
    n_node_pairs_visited: int = 0
    n_node_pairs_pruned: int = 0
    n_leaf_pairs: int = 0
    n_candidate_particle_pairs: int = 0
    n_accepted_pairs: int = 0
    max_queue_size: int = 0


@dataclass
class TraversalResult:
    pairs: np.ndarray          # (P, 2) original indices, i <= j for "unique"
    r: np.ndarray              # (P,) minimum-image distances
    histogram: np.ndarray      # (n_bins,)
    stats: TraversalStats = field(default_factory=TraversalStats)


def _logits(r: np.ndarray, centers: np.ndarray, temperature: float):
    return -(r[:, None] - centers[None, :]) ** 2 / float(temperature) ** 2


def _weights(r: np.ndarray, centers: np.ndarray, temperature: float, mode: str):
    q = _logits(r, centers, temperature)
    if mode == "gaussian":
        return np.exp(q)
    if mode == "softmax":
        q = q - q.max(axis=1, keepdims=True)
        e = np.exp(q)
        return e / e.sum(axis=1, keepdims=True)
    raise RadiusConfigError(f"unknown weight mode {mode!r}")


def _periodic_distance(pos_i: np.ndarray, pos_j: np.ndarray, boxsize: float) -> np.ndarray:
    d = pos_i - pos_j
    if boxsize and np.isfinite(boxsize) and boxsize > 0:
        d = d - boxsize * np.round(d / boxsize)
    return d


def _node_lower_bound(tree_q: RadiusTree, nq: int,
                      tree_s: RadiusTree, ns: int,
                      boxsize: float) -> float:
    """Lower bound on the minimum-image distance between two node boxes."""
    dim = tree_q.node_lo.shape[1]
    total = 0.0
    for ax in range(dim):
        gap = interval_gap_reference(
            tree_q.node_lo[nq, ax], tree_q.node_hi[nq, ax],
            tree_s.node_lo[ns, ax], tree_s.node_hi[ns, ax],
            boxsize,
        )
        total += gap * gap
    return float(np.sqrt(total))


def traverse_tree_pairs(
    tree_q: RadiusTree,
    *,
    r_max: float,
    bin_centers: Optional[np.ndarray] = None,
    temperature: float = 0.05,
    mode: str = "softmax",
    weights_query: Optional[np.ndarray] = None,
    weights_source: Optional[np.ndarray] = None,
    self_query: bool = True,
    pair_convention: str = "unique",
    boxsize: Optional[float] = None,
    eps: Optional[float] = None,
    tree_s: Optional[RadiusTree] = None,
) -> TraversalResult:
    """Traverse one or two trees and collect accepted pairs.

    ``self_query=True`` requires ``tree_s is None`` (or the same tree) and
    applies the uniqueness rule inside the traversal.  For ``self_query=False``
    pass a separate ``tree_s``; all query/source pairs are enumerated.
    """
    if r_max <= 0 or not np.isfinite(r_max):
        raise RadiusConfigError(f"r_max must be positive and finite, got {r_max}")
    if pair_convention not in ("unique", "directed_half"):
        raise RadiusConfigError(f"unknown pair_convention {pair_convention!r}")
    if self_query and tree_s is not None and tree_s is not tree_q:
        raise RadiusConfigError("self_query traversal needs a single tree")

    tree_s = tree_q if tree_s is None else tree_s
    if tree_q.sorted_pos.shape[1] != tree_s.sorted_pos.shape[1]:
        raise RadiusConfigError("trees have different dimensions")

    L = float(tree_q.boxsize if boxsize is None else boxsize)
    periodic = np.isfinite(L) and L > 0
    if eps is None:
        eps = 1e-60                      # PLAN.md 6.1 float64 under-sqrt epsilon
    prune_tol = 8.0 * np.finfo(np.float64).eps * (L if periodic else 1.0)
    r2_max = r_max * r_max

    centers = None
    if bin_centers is not None:
        centers = np.asarray(bin_centers, dtype=np.float64)
        if centers.ndim != 1 or centers.size == 0:
            raise RadiusConfigError("bin_centers must be a non-empty 1-D array")
        if np.any(np.diff(centers) <= 0) or np.any(centers < 0):
            raise RadiusConfigError(
                "bin_centers must be non-negative and strictly increasing"
            )
    if weights_query is not None:
        weights_query = np.asarray(weights_query, dtype=np.float64)
        if weights_query.shape != (tree_q.n_particles,):
            raise RadiusConfigError("weights_query has wrong shape")
    if weights_source is not None:
        weights_source = np.asarray(weights_source, dtype=np.float64)
        if weights_source.shape != (tree_s.n_particles,):
            raise RadiusConfigError("weights_source has wrong shape")

    p_sorted_q = tree_q.sorted_pos
    p_sorted_s = tree_s.sorted_pos
    to_orig_q = tree_q.sorted_to_orig
    to_orig_s = tree_s.sorted_to_orig
    r_safe_val = r_safe(L if periodic else 0.0)

    stats = TraversalStats()
    out_i, out_j, out_r = [], [], []
    hist = np.zeros(centers.size, dtype=np.float64) if centers is not None else None

    stack = [(tree_q.root, tree_s.root)] if self_query else [(tree_q.root, tree_s.root)]
    stats.max_queue_size = 1
    while stack:
        nq, ns = stack.pop()
        stats.n_node_pairs_visited += 1
        if stats.n_node_pairs_visited > 50_000_000:
            raise RuntimeError("traversal did not terminate; aborting")

        lb = _node_lower_bound(tree_q, nq, tree_s, ns, L)
        if lb > r_max + prune_tol:
            stats.n_node_pairs_pruned += 1
            continue

        leaf_q = tree_q.is_leaf[nq]
        leaf_s = tree_s.is_leaf[ns]
        if leaf_q and leaf_s:
            stats.n_leaf_pairs += 1
            s0 = int(tree_q.node_start[nq])
            s1 = s0 + int(tree_q.node_count[nq])
            t0 = int(tree_s.node_start[ns])
            t1 = t0 + int(tree_s.node_count[ns])
            for a in range(s0, s1):
                i = int(to_orig_q[a])
                for b in range(t0, t1):
                    j = int(to_orig_s[b])
                    if self_query:
                        if i == j:
                            continue
                        if pair_convention == "unique" and j <= i:
                            continue
                    stats.n_candidate_particle_pairs += 1
                    d = _periodic_distance(p_sorted_q[a], p_sorted_s[b], L)
                    r2 = float(d @ d)
                    if r2 > r2_max:
                        continue
                    r = float(np.sqrt(r2 + eps))
                    if r < r_safe_val:
                        continue
                    stats.n_accepted_pairs += 1
                    if hist is not None:
                        if pair_convention == "directed_half" and self_query:
                            scale = 0.5
                        else:
                            scale = 1.0
                        a_w = 1.0
                        if weights_query is not None:
                            a_w *= weights_query[i]
                        if weights_source is not None:
                            a_w *= weights_source[j]
                        hist += scale * a_w * _weights(
                            np.array([r]), centers, temperature, mode
                        )[0]
                    else:
                        out_i.append(i)
                        out_j.append(j)
                        out_r.append(r)
            continue

        # expand the larger side
        expand_q = (not leaf_q) and (
            leaf_s or int(tree_q.node_count[nq]) >= int(tree_s.node_count[ns])
        )
        if expand_q:
            for c in tree_q.children_of(nq):
                stack.append((c, ns))
        else:
            for c in tree_s.children_of(ns):
                stack.append((nq, c))
        stats.max_queue_size = max(stats.max_queue_size, len(stack))

    if hist is not None:
        pairs = np.empty((0, 2), dtype=np.int64)
        dist = np.empty((0,), dtype=np.float64)
    else:
        pairs = np.array(list(zip(out_i, out_j)), dtype=np.int64).reshape(-1, 2)
        dist = np.asarray(out_r, dtype=np.float64)
        if pairs.size:
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            pairs, dist = pairs[order], dist[order]
    return TraversalResult(pairs=pairs, r=dist, histogram=hist, stats=stats)


def build_work_items(
    tree_q: RadiusTree,
    tree_s: Optional[RadiusTree] = None,
    *,
    r_max: float,
    boxsize: Optional[float] = None,
    max_work_items: int = 1 << 22,
    prune_tol: Optional[float] = None,
    stats: Optional[TraversalStats] = None,
    unique_pairs: bool = True,
) -> np.ndarray:
    """Descend the dual tree and return leaf-pair work items for the kernels.

    The result is an ``(W, 4)`` int32 array of ``(q0, q1, s0, s1)`` half-open
    particle ranges **in tree-sorted space**; the FFI kernels apply the
    self/uniqueness rules to sorted indices.  Items depend only on the tree
    geometry and ``r_max`` (never on the bin centres), so they are reused by the
    forward and backward kernels.

    ``unique_pairs=True`` (the default) descends only the upper triangle of
    self-pairs, matching ``pair_convention="unique"``.  Pass
    ``unique_pairs=False`` for ``"directed_half"`` so both orientations of
    off-diagonal leaf pairs are enqueued; the kernel then halves the result.

    Raises :class:`~jzradius.errors.WorkQueueOverflow` when the queue would grow
    beyond ``max_work_items`` (PLAN.md 7.3: never silently drop interactions).
    """
    from .errors import WorkQueueOverflow

    if r_max <= 0 or not np.isfinite(r_max):
        raise RadiusConfigError(f"r_max must be positive and finite, got {r_max}")
    self_query = tree_s is None or tree_s is tree_q
    tree_s = tree_q if tree_s is None else tree_s

    L = float(tree_q.boxsize if boxsize is None else boxsize)
    periodic = np.isfinite(L) and L > 0
    if prune_tol is None:
        prune_tol = 8.0 * np.finfo(np.float64).eps * (L if periodic else 1.0)

    def _push(work: list, a: int, b: int) -> None:
        s0 = int(tree_q.node_start[a])
        s1 = s0 + int(tree_q.node_count[a])
        t0 = int(tree_s.node_start[b])
        t1 = t0 + int(tree_s.node_count[b])
        if s1 > s0 and t1 > t0:
            work.append((s0, s1, t0, t1))

    work: list = []
    stack = [(int(tree_q.root), int(tree_s.root))]
    n_visited = n_pruned = n_leaf_pairs = 0
    while stack:
        nq, ns = stack.pop()
        n_visited += 1
        if n_visited > 50_000_000:
            raise RadiusConfigError("work-item descent did not terminate")
        lb = _node_lower_bound(tree_q, nq, tree_s, ns, L)
        if lb > r_max + prune_tol:
            n_pruned += 1
            continue
        leaf_q = bool(tree_q.is_leaf[nq])
        leaf_s = bool(tree_s.is_leaf[ns])
        if leaf_q and leaf_s:
            n_leaf_pairs += 1
            _push(work, nq, ns)
        elif self_query and nq == ns:
            kids = list(tree_q.children_of(nq))
            for ia in range(len(kids)):
                for ib in range(ia, len(kids)):
                    stack.append((kids[ia], kids[ib]))
                    if not unique_pairs and ia != ib:
                        # Ordered enumeration: the mirrored leaf pair carries
                        # the (j, i) orientations the kernel will visit.
                        stack.append((kids[ib], kids[ia]))
        else:
            expand_q = (not leaf_q) and (
                leaf_s or int(tree_q.node_count[nq]) >= int(tree_s.node_count[ns])
            )
            if expand_q:
                for c in tree_q.children_of(nq):
                    stack.append((c, ns))
            else:
                for c in tree_s.children_of(ns):
                    stack.append((nq, c))
        if len(work) > max_work_items:
            raise WorkQueueOverflow(
                f"work queue exceeded max_work_items={max_work_items} "
                f"({len(work)} leaf pairs so far); increase the limit or reduce r_max"
            )

    if stats is not None:
        stats.n_node_pairs_visited = n_visited
        stats.n_node_pairs_pruned = n_pruned
        stats.n_leaf_pairs = n_leaf_pairs

    if not work:
        return np.zeros((0, 4), dtype=np.int32)
    if tree_q.n_particles >= 2**31 or tree_s.n_particles >= 2**31:
        raise RadiusConfigError("particle counts must fit in int32")
    return np.asarray(work, dtype=np.int32)
