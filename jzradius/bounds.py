"""Periodic node-node lower bounds (PLAN.md section 10).

The one invariant that matters:

    lower_bound(A, B) <= min_{i in A, j in B} d_periodic(x_i, x_j) + tol

Over-estimating a bound silently drops pairs, so every function here is
conservative by construction (gaps are clamped at zero, intervals are shifted
by whole box lengths only).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

PRUNE_TOL_FACTOR = 8.0


def _as_float_dtype(*values):
    dtypes = [jnp.asarray(v).dtype for v in values]
    return jnp.result_type(*dtypes)


def periodic_interval_gap(a0, a1, b0, b1, boxsize):
    """1-D minimum-image gap between intervals ``[a0, a1]`` and ``[b0, b1]``.

    PLAN.md 10.2.  Works for arrays (broadcasting) and for intervals that
    extend outside ``[0, L)`` as long as their width is below ``L/2``.
    """
    dtype = _as_float_dtype(a0, a1, b0, b1, boxsize)
    a0, a1, b0, b1, boxsize = (
        jnp.asarray(v, dtype=dtype) for v in (a0, a1, b0, b1, boxsize)
    )
    shape = jnp.broadcast_shapes(a0.shape, a1.shape, b0.shape, b1.shape)
    best = jnp.full(shape, jnp.inf, dtype=dtype)
    for shift in (-boxsize, jnp.zeros_like(boxsize), boxsize):
        s0 = b0 + shift
        s1 = b1 + shift
        gap = jnp.where(
            a1 < s0,
            s0 - a1,
            jnp.where(s1 < a0, a0 - s1, jnp.zeros_like(a0)),
        )
        best = jnp.minimum(best, gap)
    return jnp.maximum(best, jnp.zeros_like(best))


def lower_bound_boxes(lo_a, hi_a, lo_b, hi_b, boxsize):
    """3-D axis-aligned periodic lower bound, PLAN.md 10.3.

    ``lo_*`` / ``hi_*`` have shape ``(..., 3)``.  Returns shape ``(...)``.
    """
    dtype = _as_float_dtype(lo_a, hi_a, lo_b, hi_b, boxsize)
    lo_a, hi_a, lo_b, hi_b = (
        jnp.asarray(v, dtype=dtype) for v in (lo_a, hi_a, lo_b, hi_b)
    )
    if lo_a.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) bounds, got {lo_a.shape}")
    gaps = [
        periodic_interval_gap(
            lo_a[..., k], hi_a[..., k], lo_b[..., k], hi_b[..., k], boxsize
        )
        for k in range(3)
    ]
    return jnp.sqrt(sum(g * g for g in gaps))


def lower_bound_spheres(center_a, radius_a, center_b, radius_b, boxsize):
    """Cheaper bounding-sphere variant, PLAN.md 10.4."""
    from .math import periodic_distance

    d = periodic_distance(center_a - center_b, boxsize)
    return jnp.maximum(d - radius_a - radius_b, 0.0)


def prune_tolerance(boxsize, dtype) -> float:
    """PLAN.md 10.1: ``prune_tol = 8 * eps_dtype * L``."""
    eps = float(jnp.finfo(dtype).eps)
    return PRUNE_TOL_FACTOR * eps * float(boxsize)


def boxes_from_particles(pos, dtype=None):
    """Tight axis-aligned bounds of a particle set: ``(lo, hi)``."""
    pos = jnp.asarray(pos)
    if dtype is not None:
        pos = pos.astype(dtype)
    if pos.shape[0] == 0:
        raise ValueError("cannot bound an empty particle set")
    return jnp.min(pos, axis=0), jnp.max(pos, axis=0)


def interval_gap_reference(a0, a1, b0, b1, boxsize):
    """Scalar NumPy transcription of PLAN.md 10.2 (test oracle)."""
    best = np.inf
    for shift in (-boxsize, 0.0, boxsize):
        s0, s1 = b0 + shift, b1 + shift
        if a1 < s0:
            gap = s0 - a1
        elif s1 < a0:
            gap = a0 - s1
        else:
            return 0.0
        best = min(best, gap)
    return max(best, 0.0)
