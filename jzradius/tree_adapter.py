"""Adapter from jztree's level-packed tree representation to a flat ``RadiusTree``.

jztree stores the hierarchy as *tree planes* (:class:`jztree.data.PackedArray`):
level 0 is the leaf level, each level holds its nodes densely in Morton order,
and two packed split arrays describe the relations

* ``ispl_n2l[level]``: for node ``i`` on ``level``, the covered leaves are
  ``[ispl_n2l[i], ispl_n2l[i+1])``,
* ``ispl_n2n[level]``: for node ``i`` on ``level``, the children on level
  ``level - 1`` are ``[ispl_n2n[i], ispl_n2n[i+1])``.  On level 0 the split
  indexes particles (i.e. the leaf-to-particle relation).

The flat representation here gives every node a stable integer id, a parent,
children, a particle range and a conservative axis-aligned bounding box, so
the traversal code can treat the hierarchy as a plain array of nodes.

Bounding boxes come from particle min/max (PLAN.md 9.5 priority 3): leaves use
their particles, internal nodes the union of their children.  This is
conservative by construction.  Note that ``get_node_geometry(...)["ext"]`` is
the *full width*, so boxes are ``cent +/- ext/2``; we do not use it here, but
any future code that does must halve it (see docs/upstream_api_notes.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .errors import TreeAdapterUnavailable

__all__ = [
    "LevelData",
    "RadiusTree",
    "build_radius_tree_from_levels",
    "build_radius_tree_from_jztree",
    "build_naive_radius_tree",
    "check_tree_invariants",
]


@dataclass
class LevelData:
    """One tree plane, zero-padded to its true node count.

    Attributes
    ----------
    center:
        ``(n, dim)`` node centers.
    level:
        ``(n,)`` Morton level (larger = coarser).
    leaf_splits:
        ``(n + 1,)`` leaf indices covered by each node.
    child_splits:
        ``(n + 1,)`` child node indices in the *next lower* level; for level 0
        these are particle indices.
    """

    center: np.ndarray
    level: np.ndarray
    leaf_splits: np.ndarray
    child_splits: np.ndarray


@dataclass
class RadiusTree:
    """Flat, canonical tree representation (PLAN.md 9.1)."""

    sorted_pos: np.ndarray            # (N, dim) z-ordered positions
    sorted_to_orig: np.ndarray        # (N,) sorted index -> original index
    orig_to_sorted: np.ndarray        # (N,) original index -> sorted index
    node_start: np.ndarray            # (M,) first particle covered
    node_count: np.ndarray            # (M,) number of particles covered
    node_left: np.ndarray             # (M,) first child, -1 for leaves
    node_right: np.ndarray            # (M,) one past last child, -1 for leaves
    node_parent: np.ndarray           # (M,) parent, -1 for root
    node_level: np.ndarray            # (M,) Morton level, root has max+1
    node_lo: np.ndarray               # (M, dim) bounding box lower corner
    node_hi: np.ndarray               # (M, dim) bounding box upper corner
    node_center: np.ndarray           # (M, dim) center of the bounding box
    node_radius: np.ndarray           # (M,) radius of the enclosing sphere
    root: int
    n_particles: int
    n_nodes: int
    leaf_size: int
    boxsize: float = 0.0
    is_leaf: np.ndarray = field(default=None)  # (M,) bool convenience mask

    def __post_init__(self):
        if self.is_leaf is None:
            self.is_leaf = np.asarray(self.node_left < 0)

    def particles_of(self, node: int) -> np.ndarray:
        """Original particle indices covered by ``node``."""
        lo = int(self.node_start[node])
        hi = lo + int(self.node_count[node])
        return self.sorted_to_orig[lo:hi]

    def children_of(self, node: int) -> range:
        if self.is_leaf[node]:
            return range(0)
        return range(int(self.node_left[node]), int(self.node_right[node]))


def _tree_error(msg: str) -> TreeAdapterUnavailable:
    return TreeAdapterUnavailable(msg)


def _as_2d(a, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise _tree_error(f"{name} must be 2-D, got shape {a.shape}")
    return a


def _fill_boxes(tree: RadiusTree) -> None:
    """Fill ``node_lo/hi/center/radius`` from leaves and child unions."""
    pos = tree.sorted_pos
    for k in range(tree.n_nodes):
        if tree.is_leaf[k]:
            s = int(tree.node_start[k])
            e = s + int(tree.node_count[k])
            if e > s:
                tree.node_lo[k] = pos[s:e].min(axis=0)
                tree.node_hi[k] = pos[s:e].max(axis=0)
            else:
                tree.node_lo[k] = tree.node_hi[k] = tree.node_center[k]
    # parents always have a strictly larger level than their children, so
    # processing in increasing level order visits children first regardless of
    # how node ids were assigned (level planes give smaller ids, the naive
    # pre-order builder gives larger ones).
    order = np.argsort(tree.node_level, kind="stable")
    for k in order:
        k = int(k)
        if tree.is_leaf[k]:
            continue
        c0, c1 = int(tree.node_left[k]), int(tree.node_right[k])
        if c1 <= c0:
            raise _tree_error(f"internal node {k} has no children")
        tree.node_lo[k] = tree.node_lo[c0:c1].min(axis=0)
        tree.node_hi[k] = tree.node_hi[c0:c1].max(axis=0)
    tree.node_center[:] = 0.5 * (tree.node_lo + tree.node_hi)
    tree.node_radius = 0.5 * np.linalg.norm(tree.node_hi - tree.node_lo, axis=1)


def build_radius_tree_from_levels(
    levels: Sequence[LevelData],
    sorted_pos: np.ndarray,
    sorted_to_orig: np.ndarray,
    *,
    boxsize: float,
    leaf_size: int,
) -> RadiusTree:
    """Flatten level planes into a :class:`RadiusTree`.

    ``levels[0]`` is the leaf plane; ``levels[-1]`` is the coarsest real plane.
    Node ids are assigned level by level, so children always have smaller ids
    than their parents.  A virtual root covering the periodic box is appended
    (``root == n_nodes - 1``).
    """
    if len(levels) == 0:
        raise _tree_error("need at least one tree level")

    pos = _as_2d(sorted_pos, "sorted_pos").astype(np.float64, copy=False)
    n_particles = pos.shape[0]
    if len(sorted_to_orig) != n_particles:
        raise _tree_error("sorted_to_orig length does not match positions")
    if leaf_size < 1:
        raise _tree_error(f"invalid leaf_size {leaf_size}")

    per_level_centers: List[np.ndarray] = []
    per_level_nodes: List[int] = []
    for li, lv in enumerate(levels):
        c = _as_2d(lv.center, f"levels[{li}].center")
        n = c.shape[0]
        if n == 0:
            raise _tree_error(f"levels[{li}] has no nodes")
        if np.shape(lv.leaf_splits) != (n + 1,):
            raise _tree_error(
                f"levels[{li}].leaf_splits must have shape {(n + 1,)}, "
                f"got {np.shape(lv.leaf_splits)}"
            )
        if np.shape(lv.child_splits) != (n + 1,):
            raise _tree_error(
                f"levels[{li}].child_splits must have shape {(n + 1,)}, "
                f"got {np.shape(lv.child_splits)}"
            )
        per_level_centers.append(c)
        per_level_nodes.append(n)

    leaf_splits0 = np.asarray(levels[0].leaf_splits, dtype=np.int64)
    if leaf_splits0[0] != 0 or leaf_splits0[-1] != n_particles:
        raise _tree_error(
            "leaf level does not partition the particle array: "
            f"[{leaf_splits0[0]}, {leaf_splits0[-1]}) vs N={n_particles}"
        )
    if np.any(np.diff(leaf_splits0) < 0):
        raise _tree_error("leaf_splits must be non-decreasing")

    n_leaves = per_level_nodes[0]
    if len(leaf_splits0) != n_leaves + 1:
        raise _tree_error("leaf count disagrees with leaf_splits length")

    offsets = np.zeros(len(levels), dtype=np.int64)
    for li in range(1, len(levels)):
        offsets[li] = offsets[li - 1] + per_level_nodes[li - 1]
    n_real = int(offsets[-1] + per_level_nodes[-1])
    root = n_real
    n_nodes = n_real + 1

    node_level = np.empty(n_nodes, dtype=np.int64)
    node_left = np.full(n_nodes, -1, dtype=np.int64)
    node_right = np.full(n_nodes, -1, dtype=np.int64)
    node_parent = np.full(n_nodes, -1, dtype=np.int64)
    node_start = np.zeros(n_nodes, dtype=np.int64)
    node_count = np.zeros(n_nodes, dtype=np.int64)
    node_center = np.zeros((n_nodes, pos.shape[1]), dtype=np.float64)
    node_lo = np.zeros((n_nodes, pos.shape[1]), dtype=np.float64)
    node_hi = np.zeros((n_nodes, pos.shape[1]), dtype=np.float64)

    # leaves: particle ranges
    node_start[:n_leaves] = leaf_splits0[:-1]
    node_count[:n_leaves] = np.diff(leaf_splits0)
    node_level[:n_leaves] = np.asarray(levels[0].level, dtype=np.int64)
    node_center[:n_leaves] = per_level_centers[0]

    # internal nodes
    for li in range(1, len(levels)):
        off = int(offsets[li])
        n = per_level_nodes[li]
        ls = np.asarray(levels[li].leaf_splits, dtype=np.int64)
        if np.any(np.diff(ls) < 0):
            raise _tree_error(f"levels[{li}].leaf_splits not monotone")
        if ls[0] != leaf_splits0[0] or ls[-1] > leaf_splits0[-1]:
            raise _tree_error(f"levels[{li}].leaf_splits out of range")
        # leaf index -> particle index
        node_start[off:off + n] = leaf_splits0[ls[:-1]]
        node_count[off:off + n] = leaf_splits0[ls[1:]] - leaf_splits0[ls[:-1]]
        node_level[off:off + n] = np.asarray(levels[li].level, dtype=np.int64)
        node_center[off:off + n] = per_level_centers[li]

        cs = np.asarray(levels[li].child_splits, dtype=np.int64)
        child_off = int(offsets[li - 1])
        if np.any(np.diff(cs) < 0):
            raise _tree_error(f"levels[{li}].child_splits not monotone")
        if cs[0] != 0 or cs[-1] != per_level_nodes[li - 1]:
            raise _tree_error(
                f"levels[{li}].child_splits do not cover level {li - 1}"
            )
        node_left[off:off + n] = child_off + cs[:-1]
        node_right[off:off + n] = child_off + cs[1:]
        for k in range(n):
            node_parent[child_off + cs[k]:child_off + cs[k + 1]] = off + k

    # virtual root over the top level
    top = len(levels) - 1
    top_off = int(offsets[top])
    node_left[root] = top_off
    node_right[root] = top_off + per_level_nodes[top]
    node_parent[top_off:top_off + per_level_nodes[top]] = root
    node_level[root] = int(np.max(node_level[:n_real])) + 1
    node_start[root] = 0
    node_count[root] = n_particles
    if boxsize is None or not np.isfinite(boxsize) or boxsize <= 0:
        node_lo[root] = pos.min(axis=0)
        node_hi[root] = pos.max(axis=0)
    else:
        node_lo[root] = 0.0
        node_hi[root] = float(boxsize)
    node_center[root] = 0.5 * (node_lo[root] + node_hi[root])

    tree = RadiusTree(
        sorted_pos=np.ascontiguousarray(pos),
        sorted_to_orig=np.asarray(sorted_to_orig, dtype=np.int64),
        orig_to_sorted=np.argsort(np.asarray(sorted_to_orig, dtype=np.int64)),
        node_start=node_start,
        node_count=node_count,
        node_left=node_left,
        node_right=node_right,
        node_parent=node_parent,
        node_level=node_level,
        node_lo=node_lo,
        node_hi=node_hi,
        node_center=node_center,
        node_radius=np.zeros(n_nodes),
        root=int(root),
        n_particles=int(n_particles),
        n_nodes=int(n_nodes),
        leaf_size=int(leaf_size),
        boxsize=float(boxsize) if boxsize is not None else 0.0,
    )
    _fill_boxes(tree)
    check_tree_invariants(tree)
    return tree


def build_radius_tree_from_jztree(
    th,
    sorted_pos: np.ndarray,
    sorted_to_orig: np.ndarray,
    *,
    boxsize: float,
    leaf_size: int,
) -> RadiusTree:
    """Adapt a :class:`jztree.data.TreeHierarchy` (after ``zsort_and_tree``).

    ``sorted_pos``/``sorted_to_orig`` must be the z-ordered arrays returned by
    ``jztree.tree.zsort``.
    """
    nlevels = int(th.ispl_n2n.nlevels())
    pos = np.asarray(sorted_pos, dtype=np.float64)
    if nlevels == 1 and int(th.ispl_n2n.num(0)) == 0:
        # jztree stores no split at all when the whole particle set fits into a
        # single leaf (N <= max_leaf_size).  Materialise that implicit leaf.
        n_particles = pos.shape[0]
        if n_particles == 0:
            raise _tree_error("cannot build a tree for zero particles")
        single = np.arange(n_particles + 1, dtype=np.int64)
        levels = [
            LevelData(
                center=pos.mean(axis=0, keepdims=True),
                level=np.zeros(1, dtype=np.int64),
                leaf_splits=single,
                child_splits=single,
            )
        ]
        return build_radius_tree_from_levels(
            levels,
            sorted_pos=pos,
            sorted_to_orig=sorted_to_orig,
            boxsize=boxsize,
            leaf_size=leaf_size,
        )
    levels = []
    for li in range(nlevels):
        # ``num(level)`` counts the stored split entries, i.e. nodes + 1.
        n = int(th.ispl_n2n.num(li)) - 1
        n2n = np.asarray(th.ispl_n2n.get(li, size=n + 1))
        center = np.asarray(th.geom_cent.get(li, size=n))
        if n2n.shape[0] != n + 1 or center.shape[0] != n:
            raise _tree_error(
                f"level {li}: packed arrays have unexpected length"
            )
        if li == 0:
            # On the leaf level n2n *is* the leaf -> particle relation
            # (jztree up_calls insert `ispl` there); n2l on this level is the
            # trivial node -> leaf identity (see tree.py `_define_split_hierarchy`).
            leaf_splits = n2n
        else:
            leaf_splits = np.asarray(th.ispl_n2l.get(li, size=n + 1))
            if leaf_splits.shape[0] != n + 1:
                raise _tree_error(
                    f"level {li}: n2l has unexpected length"
                )
        levels.append(
            LevelData(
                center=np.asarray(center, dtype=np.float64),
                # Plane index is the hierarchy depth; th.lvl is the Morton
                # resolution level and is not monotone with the hierarchy.
                level=np.full(n, li, dtype=np.int64),
                leaf_splits=np.asarray(leaf_splits, dtype=np.int64),
                child_splits=n2n,
            )
        )
    return build_radius_tree_from_levels(
        levels,
        sorted_pos,
        sorted_to_orig,
        boxsize=boxsize,
        leaf_size=leaf_size,
    )


def build_naive_radius_tree(
    positions: np.ndarray,
    *,
    boxsize: float,
    leaf_size: int = 32,
) -> RadiusTree:
    """Median-split tree used to exercise traversal on CPU (test scaffold).

    Not the product tree (that one adapts jztree), but it yields the same
    canonical structure so the traversal code is identical.  Median splits are
    performed on the longest axis, and nodes are created post-order so children
    always have smaller ids than their parents.
    """
    pos = np.array(_as_2d(positions, "positions"), dtype=np.float64, copy=True)
    orig = np.arange(pos.shape[0], dtype=np.int64)
    n = pos.shape[0]
    if n == 0:
        raise _tree_error("cannot build a tree for zero particles")
    if leaf_size < 1:
        raise _tree_error(f"invalid leaf_size {leaf_size}")

    dim = pos.shape[1]
    node_level: List[int] = []
    node_left: List[int] = []
    node_right: List[int] = []
    node_parent: List[int] = []
    node_start: List[int] = []
    node_count: List[int] = []
    node_center: List[np.ndarray] = []
    is_leaf: List[bool] = []

    def alloc(parent, start, count, center):
        idx = len(node_level)
        node_level.append(-1)
        node_left.append(-1)
        node_right.append(-1)
        node_parent.append(parent)
        node_start.append(start)
        node_count.append(count)
        node_center.append(np.asarray(center, dtype=np.float64))
        is_leaf.append(True)
        return idx

    def fill(idx: int, lo: int, hi: int) -> None:
        cnt = hi - lo
        if cnt <= leaf_size:
            node_center[idx] = pos[lo:hi].mean(axis=0)
            node_level[idx] = 0
            return
        sub = pos[lo:hi]
        axis = int(np.argmax(sub.max(axis=0) - sub.min(axis=0)))
        order = np.argsort(sub[:, axis], kind="stable")
        pos[lo:hi] = sub[order]
        orig[lo:hi] = orig[lo:hi][order]
        mid = lo + cnt // 2
        # allocate both children consecutively so the child range is contiguous
        left = alloc(idx, lo, mid - lo, 0.0)
        right = alloc(idx, mid, hi - mid, 0.0)
        is_leaf[idx] = False
        node_left[idx], node_right[idx] = left, right + 1
        fill(left, lo, mid)
        fill(right, mid, hi)
        node_level[idx] = max(node_level[left], node_level[right]) + 1
        node_center[idx] = 0.5 * (node_center[left] + node_center[right])

    root_real = alloc(-1, 0, n, 0.0)
    fill(root_real, 0, n)
    root = alloc(-1, 0, n, 0.5 * (pos.min(axis=0) + pos.max(axis=0)))
    is_leaf[root] = False
    node_left[root], node_right[root] = root_real, root_real + 1
    node_parent[root_real] = root
    node_level[root] = node_level[root_real] + 1

    tree = RadiusTree(
        sorted_pos=np.ascontiguousarray(pos),
        sorted_to_orig=orig,
        orig_to_sorted=np.argsort(orig),
        node_start=np.asarray(node_start, dtype=np.int64),
        node_count=np.asarray(node_count, dtype=np.int64),
        node_left=np.asarray(node_left, dtype=np.int64),
        node_right=np.asarray(node_right, dtype=np.int64),
        node_parent=np.asarray(node_parent, dtype=np.int64),
        node_level=np.asarray(node_level, dtype=np.int64),
        node_lo=np.zeros((len(node_level), dim)),
        node_hi=np.zeros((len(node_level), dim)),
        node_center=np.asarray(node_center),
        node_radius=np.zeros(len(node_level)),
        root=int(root),
        n_particles=n,
        n_nodes=len(node_level),
        leaf_size=int(leaf_size),
        boxsize=float(boxsize) if boxsize is not None else 0.0,
        is_leaf=np.asarray(is_leaf, dtype=bool),
    )
    if tree.boxsize > 0:
        tree.node_lo[root] = 0.0
        tree.node_hi[root] = tree.boxsize
    _fill_boxes(tree)
    check_tree_invariants(tree)
    return tree


def check_tree_invariants(tree: RadiusTree) -> None:
    """Assert the adapter invariants of PLAN.md 9.3 (raises on violation)."""
    n, m = tree.n_particles, tree.n_nodes
    if tree.sorted_pos.shape[0] != n:
        raise _tree_error("sorted_pos length does not match n_particles")
    if tree.sorted_to_orig.shape != (n,) or tree.orig_to_sorted.shape != (n,):
        raise _tree_error("index mapping arrays have wrong shape")
    if not np.array_equal(tree.orig_to_sorted[tree.sorted_to_orig],
                          np.arange(n)):
        raise _tree_error("sorted_to_orig / orig_to_sorted are not inverses")
    if tree.root < 0 or tree.root >= m:
        raise _tree_error("root index out of range")
    if tree.node_parent[tree.root] != -1:
        raise _tree_error("root has a parent")

    for k in range(m):
        s, c = int(tree.node_start[k]), int(tree.node_count[k])
        if s < 0 or c < 0 or s + c > n:
            raise _tree_error(f"node {k} particle range out of bounds")
        if tree.is_leaf[k]:
            if c > tree.leaf_size:
                raise _tree_error(
                    f"leaf {k} holds {c} > leaf_size {tree.leaf_size}"
                )
        else:
            c0, c1 = int(tree.node_left[k]), int(tree.node_right[k])
            if not (0 <= c0 < c1 <= m):
                raise _tree_error(f"node {k} child range invalid")
            child_counts = int(tree.node_count[c0:c1].sum())
            if child_counts != c:
                raise _tree_error(
                    f"node {k} count {c} != children count {child_counts}"
                )
            if not np.all(tree.node_parent[c0:c1] == k):
                raise _tree_error(f"node {k} children have wrong parent")
            if np.any(tree.node_lo[c0:c1] < tree.node_lo[k] - 1e-9) or np.any(
                tree.node_hi[c0:c1] > tree.node_hi[k] + 1e-9
            ):
                raise _tree_error(f"node {k} box does not contain its children")

    covered = np.zeros(n, dtype=bool)
    for k in range(m):
        if tree.is_leaf[k]:
            s, c = int(tree.node_start[k]), int(tree.node_count[k])
            covered[s:s + c] = True
    if not covered.all():
        raise _tree_error("leaves do not cover all particles")
    if tree.node_count[tree.root] != n:
        raise _tree_error("root does not cover all particles")
