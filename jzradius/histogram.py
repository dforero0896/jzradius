"""Public differentiable radius-histogram API (PLAN.md 7, 11, M6/M7).

``radius_softmax_histogram`` is the user-facing entry point.  It

1. validates the configuration (``RadiusHistogramConfig``),
2. builds periodic Morton/median-split trees for query and source particles,
3. descends the dual tree with the verified lower bounds to build leaf-pair work
   items (the same descent as ``traversal.build_work_items``),
4. launches the typed-FFI forward kernel, and
5. wraps the whole thing in ``jax.custom_vjp`` whose backward rule launches the
   analytic VJP kernel (positions *and* pair weights).

Tree construction currently happens on the host (jztree when running on GPU,
the median-split scaffold on CPU).  Consequently the function runs eagerly and
is not ``jax.jit``-able in v1; tracing raises a clear error instead of silently
producing wrong results.  See docs/limitations.md and the deviation log.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import ffi
from .config import (
    RadiusHistogramConfig,
    default_r_max,
    r_safe as _r_safe,
    validate_bin_centers,
    validate_config,
)
from .errors import JzRadiusError, RadiusConfigError
from .tree_adapter import RadiusTree, build_naive_radius_tree, build_radius_tree_from_jztree
from .traversal import build_work_items
from .work_queue import build_work_items_device

__all__ = ["radius_softmax_histogram", "build_radius_trees"]

_MODES = ("softmax", "gaussian")


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)


def _median_spacing(pos: np.ndarray, boxsize: float, max_samples: int = 256) -> float:
    """Median nearest-neighbour spacing, estimated on a deterministic subsample."""
    n = pos.shape[0]
    if n < 2:
        return float(boxsize)
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    if n > max_samples:
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
    sample = pos[idx]
    best = np.full(sample.shape[0], np.inf)
    chunk = 2048
    for start in range(0, n, chunk):
        block = pos[start:start + chunk]
        d = sample[:, None, :] - block[None, :, :]
        if boxsize and np.isfinite(boxsize) and boxsize > 0:
            d = d - boxsize * np.round(d / boxsize)
        r = np.sqrt(np.sum(d * d, axis=-1))
        if block.shape[0] == sample.shape[0] and start == 0:
            np.fill_diagonal(r, np.inf)
        best = np.minimum(best, r.min(axis=1))
    finite = best[np.isfinite(best)]
    if finite.size == 0:
        return float(boxsize)
    return float(np.median(finite))


def build_radius_trees(
    pos_query: np.ndarray,
    pos_source: Optional[np.ndarray],
    *,
    boxsize: float,
    leaf_size: int,
    prefer_jztree: bool = True,
) -> Tuple[RadiusTree, RadiusTree, bool]:
    """Build the query/source trees in sorted space.

    Returns ``(tree_q, tree_s, self_query)``.  On a GPU backend the trees come
    from jztree (``zsort`` + ``build_tree_hierarchy``); otherwise the
    median-split scaffold is used so the whole pipeline is testable on CPU.
    """
    self_query = pos_source is None
    if self_query:
        pos_source = pos_query

    use_jztree = prefer_jztree and jax.default_backend() == "gpu"
    if use_jztree:
        return _build_jztree_trees(pos_query, pos_source, boxsize, leaf_size)
    tree_q = build_naive_radius_tree(pos_query, boxsize=boxsize, leaf_size=leaf_size)
    tree_s = tree_q if self_query else build_naive_radius_tree(
        pos_source, boxsize=boxsize, leaf_size=leaf_size
    )
    return tree_q, tree_s, self_query


_TREE_FALLBACKS: list = []


def _build_jztree_trees(pos_query, pos_source, boxsize, leaf_size):
    import os

    os.environ.setdefault("JZTREE_SKIP_JAX_CUDA_CHECK", "1")
    import jztree
    from jztree.tree import TreeConfig, zsort, build_tree_hierarchy

    def one(pos):
        pos_j = jnp.asarray(pos)
        posz, idz = zsort(pos_j)
        th = None
        # jztree sizes its leaf buffer as ``alloc_fac_nodes * npart`` and can
        # come up a few entries short for small/irregular inputs
        # ("Tree-Leaf allocation too small", see D011). Retry with a larger
        # allocation factor before giving up.
        for fac in (1.0, 1.5, 2.5):
            try:
                th = build_tree_hierarchy(
                    posz, TreeConfig(max_leaf_size=leaf_size, alloc_fac_nodes=fac)
                )
                break
            except RuntimeError:
                if fac == 2.5:
                    raise
        tree = build_radius_tree_from_jztree(
            th,
            np.asarray(posz),
            np.asarray(idz),
            boxsize=boxsize,
            leaf_size=leaf_size,
        )
        return tree

    def one_or_scaffold(pos):
        # jztree's leaf-buffer estimate can be a few entries short for very
        # small or oddly shaped inputs ("Tree-Leaf allocation too small").
        # The in-house median-split tree is exact as well (same traversal and
        # same kernels), only slower to build, so use it as a fallback (D011).
        try:
            return one(pos)
        except Exception:
            _TREE_FALLBACKS.append(1)
            return build_naive_radius_tree(pos, boxsize=boxsize, leaf_size=leaf_size)

    tree_q = one_or_scaffold(pos_query)
    tree_s = tree_q if pos_source is pos_query else None
    if tree_s is None:
        tree_s = one_or_scaffold(pos_source)
    return tree_q, tree_s, pos_source is pos_query


def _prepare_centers_and_rmax(pos_np, centers, r_max, temperature, boxsize):
    if r_max is None:
        if centers is None:
            raise RadiusConfigError("provide bin_centers or r_max")
        r_max = default_r_max(
            np.asarray(centers, dtype=np.float64),
            temperature,
            _median_spacing(pos_np, float(boxsize)),
        )
    r_max = float(r_max)
    if centers is None:
        centers_np = np.linspace(0.0, r_max, 32, dtype=np.float64)
    else:
        centers_np = validate_bin_centers(centers)
    if centers_np[-1] > r_max + 1e-12 * max(1.0, abs(r_max)):
        raise RadiusConfigError(
            f"largest bin center {centers_np[-1]} exceeds r_max={r_max}"
        )
    return centers_np, r_max


def radius_softmax_histogram(
    pos_query,
    pos_source=None,
    bin_centers=None,
    *,
    temperature: float = 0.05,
    boxsize: float,
    r_max: Optional[float] = None,
    weights_query=None,
    weights_source=None,
    self_query: Optional[bool] = None,
    pair_convention: str = "unique",
    leaf_size: int = 32,
    deterministic: bool = False,
    mode: str = "softmax",
    max_work_items: int = 1 << 22,
):
    """Differentiable soft histogram of periodic two-point distances.

    Parameters follow PLAN.md 7.2.  ``bin_centers`` defaults to 32 bins spanning
    ``(0, r_max)``.  ``self_query`` defaults to ``pos_source is None``; with
    ``self_query=True`` and ``pair_convention="unique"`` each unordered pair is
    counted once (orientation fixed in tree-sorted index space, PLAN.md D006).

    The function is ``jax.grad``/``value_and_grad``-able through the
    ``custom_vjp`` below.  It is not ``jax.jit``-able in v1 because the Morton /
    median-split tree and the dual-tree work items are built on the host; see
    docs/limitations.md.
    """
    if mode not in _MODES:
        raise RadiusConfigError(f"mode must be one of {_MODES}, got {mode!r}")
    if pair_convention not in ("unique", "directed_half"):
        raise RadiusConfigError(f"unknown pair_convention {pair_convention!r}")

    q = jnp.asarray(pos_query)
    if q.ndim != 2 or q.shape[1] != 3:
        raise RadiusConfigError(f"pos_query must have shape (N, 3), got {q.shape}")
    if q.dtype not in (jnp.float32, jnp.float64):
        raise RadiusConfigError(f"only float32/float64 are supported, got {q.dtype}")
    if q.shape[0] == 0:
        raise RadiusConfigError("pos_query must contain at least one particle")

    inferred_self = pos_source is None
    if self_query is None:
        self_query = inferred_self
    self_query = bool(self_query)
    if self_query:
        if pos_source is not None and jnp.asarray(pos_source).shape != q.shape:
            raise RadiusConfigError(
                "self_query=True requires pos_source to match pos_query"
            )
        s = q
    else:
        if pos_source is None:
            raise RadiusConfigError(
                "pos_source is required unless self_query is True / implied"
            )
        s = jnp.asarray(pos_source)
        if s.ndim != 2 or s.shape[1] != 3:
            raise RadiusConfigError(f"pos_source must have shape (N, 3), got {s.shape}")
        if s.dtype != q.dtype:
            raise RadiusConfigError("query and source positions must share a dtype")
        if s.shape[0] == 0:
            raise RadiusConfigError("pos_source must contain at least one particle")

    if pair_convention == "directed_half" and not self_query:
        raise RadiusConfigError("directed_half is only defined for self_query=True")
    if deterministic:
        raise JzRadiusError(
            "deterministic=True is not implemented for the FFI path yet: the "
            "kernels accumulate with atomics (PLAN.md 6.7 allows this; see "
            "docs/limitations.md and deviation D007)"
        )

    has_wq = weights_query is not None
    has_ws = weights_source is not None
    nq, ns = int(q.shape[0]), int(s.shape[0])
    wq = (
        jnp.asarray(weights_query, dtype=q.dtype)
        if has_wq
        else jnp.ones((nq,), dtype=q.dtype)
    )
    if has_wq and wq.shape != (nq,):
        raise RadiusConfigError("weights_query must have shape (N_query,)")
    ws = (
        jnp.asarray(weights_source, dtype=s.dtype)
        if has_ws
        else jnp.ones((ns,), dtype=s.dtype)
    )
    if has_ws and ws.shape != (ns,):
        raise RadiusConfigError("weights_source must have shape (N_source,)")

    # Self-query: a single weight array plays both roles.  When only one of the
    # two weight arguments is supplied the trailing weight gradient has to be
    # folded into it, because the other cotangent is None externally.
    # Self-query: the pair weight is a_ij = wq_i * ws_j (PLAN.md 6.5), so a
    # missing weight array contributes ones -- it must NOT be replaced by the
    # other array (that would silently square the weights and change the
    # value).  The fold flags below only decide which weight cotangent is
    # returned; the unused one is dropped in the backward pass.
    fold_ws_into_wq = self_query and has_wq and not has_ws
    fold_wq_into_ws = self_query and has_ws and not has_wq

    centers = (
        None if bin_centers is None else jnp.asarray(bin_centers, dtype=jnp.float64)
    )
    return _hist_core(
        q,
        s,
        wq,
        ws,
        centers,
        self_query,
        pair_convention,
        float(temperature),
        float(boxsize),
        None if r_max is None else float(r_max),
        int(leaf_size),
        mode,
        int(max_work_items),
        has_wq,
        has_ws,
        fold_ws_into_wq,
        fold_wq_into_ws,
    )


@partial(
    jax.custom_vjp,
    nondiff_argnums=tuple(range(5, 17)),
)
def _hist_core(
    pos_q,
    pos_s,
    wq,
    ws,
    centers,
    self_query,
    pair_convention,
    temperature,
    boxsize,
    r_max,
    leaf_size,
    mode,
    max_work_items,
    has_wq,
    has_ws,
    fold_ws_into_wq,
    fold_wq_into_ws,
):
    return _hist_core_fwd(
        pos_q, pos_s, wq, ws, centers, self_query, pair_convention, temperature,
        boxsize, r_max, leaf_size, mode, max_work_items, has_wq, has_ws,
        fold_ws_into_wq, fold_wq_into_ws,
    )[0]


def _hist_core_fwd(
    pos_q,
    pos_s,
    wq,
    ws,
    centers,
    self_query,
    pair_convention,
    temperature,
    boxsize,
    r_max,
    leaf_size,
    mode,
    max_work_items,
    has_wq,
    has_ws,
    fold_ws_into_wq,
    fold_wq_into_ws,
):
    if _is_tracer(pos_q) or _is_tracer(pos_s):
        raise JzRadiusError(
            "radius_softmax_histogram builds host-side trees and is not jit-able "
            "in v1; call it eagerly (jax.grad/value_and_grad work eagerly)"
        )
    q_np = np.asarray(pos_q, dtype=np.float64)
    s_np = q_np if self_query else np.asarray(pos_s, dtype=np.float64)
    centers_np, r_max = _prepare_centers_and_rmax(
        q_np, None if centers is None else np.asarray(centers), r_max,
        temperature, boxsize,
    )
    cfg = RadiusHistogramConfig(
        boxsize=float(boxsize),
        r_max=float(r_max),
        temperature=float(temperature),
        mode=mode,
        self_query=bool(self_query),
        pair_convention=pair_convention,
        leaf_size=int(leaf_size),
        dtype=np.dtype(q_np.dtype).name,
        deterministic=False,
        max_work_items=int(max_work_items),
    )
    validate_config(cfg)

    tree_q, tree_s, derived_self = build_radius_trees(
        q_np, None if self_query else s_np, boxsize=float(boxsize),
        leaf_size=int(leaf_size),
    )
    if derived_self != bool(self_query):
        raise JzRadiusError("internal error: tree self-query mismatch")

    # Only `unique` enumerates unordered pairs once; `directed_half` walks all
    # ordered i != j pairs and halves the contribution (PLAN.md 6.5).
    triangular = pair_convention == "unique"
    # On the GPU the descent runs as device kernels (D012/D013); the CPU path
    # keeps the interpreted descent, which is what the tests cross-check.
    if jax.default_backend() == "gpu":
        work = build_work_items_device(
            tree_q,
            tree_s,
            r_max=float(r_max),
            boxsize=float(boxsize),
            max_work_items=int(max_work_items),
            unique_pairs=triangular,
        )
    else:
        work = build_work_items(
            tree_q,
            tree_s,
            r_max=float(r_max),
            boxsize=float(boxsize),
            max_work_items=int(max_work_items),
            unique_pairs=triangular,
        )
    work_j = jnp.asarray(work, dtype=jnp.int32)
    n_work = int(work_j.shape[0])
    centers_j = jnp.asarray(centers_np, dtype=jnp.float64)

    eps = 1e-30 if q_np.dtype == np.float32 else 1e-60
    r_safe_val = _r_safe(float(boxsize), np.dtype(q_np.dtype).name)
    pair_scale = 0.5 if pair_convention == "directed_half" else 1.0
    params = ffi.build_params(
        n_q=int(q_np.shape[0]),
        n_s=int(s_np.shape[0]),
        n_bins=int(centers_np.size),
        n_work=n_work,
        self_query=bool(self_query),
        triangular=triangular,
        mode=mode,
        r_max=float(r_max),
        boxsize=float(boxsize),
        temperature=float(temperature),
        eps=eps,
        r_safe=r_safe_val,
        pair_scale=pair_scale,
    )

    perm_q = jnp.asarray(tree_q.sorted_to_orig, dtype=jnp.int32)
    perm_s = jnp.asarray(tree_s.sorted_to_orig, dtype=jnp.int32)
    q_sorted = jnp.take(pos_q, perm_q, axis=0)
    s_sorted = jnp.take(pos_s, perm_s, axis=0)
    wq_sorted = jnp.take(wq, perm_q, axis=0)
    ws_sorted = jnp.take(ws, perm_s, axis=0)

    # `perm_q` orients unordered self-pairs by *input* index inside the
    # kernel (PLAN.md 6.5 / D010); it is unused for cross-correlation.
    hist = ffi.forward(
        q_sorted, s_sorted, centers_j, work_j, wq_sorted, ws_sorted, perm_q,
        params,
    )
    res = (
        perm_q,
        perm_s,
        centers_j,
        work_j,
        params,
        q_sorted,
        s_sorted,
        wq_sorted,
        ws_sorted,
    )
    return hist, res


def _hist_core_bwd(
    self_query,
    pair_convention,
    temperature,
    boxsize,
    r_max,
    leaf_size,
    mode,
    max_work_items,
    has_wq,
    has_ws,
    fold_ws_into_wq,
    fold_wq_into_ws,
    res,
    g,
):
    perm_q, perm_s, centers_j, work_j, params, q_sorted, s_sorted, wq_sorted, ws_sorted = res
    g = jnp.asarray(g, dtype=q_sorted.dtype)
    gq_s, gs_s, gwq_s, gws_s = ffi.backward(
        q_sorted, s_sorted, centers_j, work_j, wq_sorted, ws_sorted, perm_q, g,
        params,
    )
    gq = jnp.zeros_like(q_sorted).at[perm_q].add(gq_s).astype(q_sorted.dtype)
    gs = jnp.zeros_like(s_sorted).at[perm_s].add(gs_s).astype(s_sorted.dtype)
    gwq = jnp.zeros_like(wq_sorted).at[perm_q].add(gwq_s)
    gws = jnp.zeros_like(ws_sorted).at[perm_s].add(gws_s)

    if fold_ws_into_wq:
        # ws was ones, so gws is the cotangent wrt a constant: drop it.
        gws = None
    elif fold_wq_into_ws:
        gwq = None
    else:
        if not has_wq:
            gwq = None
        if not has_ws:
            gws = None
    return gq, gs, gwq, gws, None


_hist_core.defvjp(_hist_core_fwd, _hist_core_bwd)
