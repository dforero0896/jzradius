"""Brute-force JAX reference for periodic radius histograms (PLAN.md section 8).

This is the *golden oracle* for every GPU test.  It is O(Nq * Ns) and refuses
large inputs on purpose.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .config import default_r_max, validate_bin_centers
from .errors import RadiusConfigError
from .math import (
    dtype_name,
    periodic_delta,
    periodic_distance,
    soft_weights,
)

MAX_REFERENCE_N = 20_000
MAX_REFERENCE_PAIRS = 4_000_000


def _as_positions(pos, name):
    pos = jnp.asarray(pos)
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise RadiusConfigError(f"{name} must have shape (N, 3), got {pos.shape}")
    return pos


def _pair_indices(pos_q, pos_s, self_query, pair_convention):
    """Return ``(i, j)`` index arrays for the requested pair convention."""
    nq, ns = pos_q.shape[0], pos_s.shape[0]
    if not self_query:
        # Cross-correlation: every directed (query, source) pair.
        i = jnp.arange(nq)[:, None]
        j = jnp.arange(ns)[None, :]
        return jnp.broadcast_to(i, (nq, ns)), jnp.broadcast_to(j, (nq, ns))

    if nq != ns:
        raise RadiusConfigError(
            "self_query=True requires pos_query and pos_source to describe the "
            f"same catalog; got {nq} and {ns} particles"
        )
    if pair_convention == "unique":
        i, j = jnp.triu_indices(nq, k=1)
        return i, j
    if pair_convention == "directed_half":
        i = jnp.arange(nq)[:, None]
        j = jnp.arange(ns)[None, :]
        i, j = jnp.broadcast_to(i, (nq, ns)), jnp.broadcast_to(j, (nq, ns))
        keep = i != j
        return i[keep], j[keep]
    raise RadiusConfigError(f"unknown pair_convention {pair_convention!r}")


def reference_pairs(
    pos_query,
    pos_source,
    boxsize,
    r_max,
    *,
    self_query=False,
    pair_convention="unique",
):
    """Return ``(i, j, r)`` for every accepted pair (brute force)."""
    pos_q = _as_positions(pos_query, "pos_query")
    pos_s = _as_positions(pos_source, "pos_source")
    if pos_q.dtype != pos_s.dtype:
        raise RadiusConfigError(
            f"positions must share a dtype, got {pos_q.dtype} and {pos_s.dtype}"
        )
    dtype = dtype_name(pos_q.dtype)
    n = int(pos_q.shape[0]) * int(pos_s.shape[0])
    if n > MAX_REFERENCE_PAIRS:
        raise RadiusConfigError(
            f"brute-force reference refuses {n} candidate pairs "
            f"(limit {MAX_REFERENCE_PAIRS}); use the tree traversal instead"
        )
    if self_query and int(pos_q.shape[0]) > MAX_REFERENCE_N:
        raise RadiusConfigError(
            f"brute-force reference refuses N={int(pos_q.shape[0])} "
            f"(limit {MAX_REFERENCE_N})"
        )

    i, j = _pair_indices(pos_q, pos_s, self_query, pair_convention)
    delta = pos_q[i] - pos_s[j]
    r = periodic_distance(delta, boxsize)
    keep = r <= r_max
    return i[keep], j[keep], r[keep]


def pair_weights(weights_query, weights_source, i, j, dtype):
    """``a_ij`` per PLAN.md 6.5."""
    a = jnp.ones(i.shape, dtype=dtype)
    if weights_query is not None:
        a = a * jnp.asarray(weights_query, dtype=dtype)[i]
    if weights_source is not None:
        a = a * jnp.asarray(weights_source, dtype=dtype)[j]
    return a


def reference_histogram(
    pos_query,
    pos_source,
    bin_centers,
    temperature,
    boxsize,
    r_max=None,
    weights_query=None,
    weights_source=None,
    self_query=False,
    pair_convention="unique",
    mode="softmax",
):
    """Brute-force histogram ``H_b`` (PLAN.md 6.5 / 8.1).

    Auto-correlation (``self_query=True``) uses the same catalog passed twice.
    ``pair_convention="unique"`` enumerates unordered pairs once; the
    ``"directed_half"`` variant enumerates both directions and multiplies by
    1/2 *inside* the differentiable graph (PLAN.md 6.5).
    """
    pos_q = _as_positions(pos_query, "pos_query")
    pos_s = _as_positions(pos_source, "pos_source")
    centers = jnp.asarray(validate_bin_centers(bin_centers), dtype=pos_q.dtype)
    if r_max is None:
        r_max = default_r_max(centers, temperature)
    r_max = float(r_max)
    if not (0.0 < r_max < float(boxsize) / 2.0):
        raise RadiusConfigError(
            f"r_max={r_max} must satisfy 0 < r_max < boxsize/2={float(boxsize) / 2.0}"
        )

    nq, ns = int(pos_q.shape[0]), int(pos_s.shape[0])
    if weights_query is not None:
        weights_query = jnp.asarray(weights_query, dtype=pos_q.dtype)
        if weights_query.shape != (nq,):
            raise RadiusConfigError(
                f"weights_query must have shape ({nq},), got {weights_query.shape}"
            )
    if weights_source is not None:
        weights_source = jnp.asarray(weights_source, dtype=pos_q.dtype)
        if weights_source.shape != (ns,):
            raise RadiusConfigError(
                f"weights_source must have shape ({ns},), got {weights_source.shape}"
            )

    i, j, r = reference_pairs(
        pos_q,
        pos_s,
        boxsize,
        r_max,
        self_query=self_query,
        pair_convention=pair_convention,
    )
    w = soft_weights(r, centers, temperature, mode=mode)
    a = pair_weights(weights_query, weights_source, i, j, pos_q.dtype)
    hist = jnp.sum(a[:, None] * w, axis=0)
    if self_query and pair_convention == "directed_half":
        hist = 0.5 * hist
    return hist


def reference_pair_set(pos_query, pos_source, boxsize, r_max, **kwargs):
    """Deterministic pair-set representation for equality tests.

    Returns a sorted list of ``(orig_i, orig_j, r)`` tuples.  Use this (not just
    the pair count) to detect silently wrong pair sets.
    """
    i, j, r = reference_pairs(pos_query, pos_source, boxsize, r_max, **kwargs)
    i = np.asarray(i)
    j = np.asarray(j)
    r = np.asarray(r)
    order = np.lexsort((j, i))
    return [(int(ii), int(jj), float(rr)) for ii, jj, rr in zip(i[order], j[order], r[order])]
