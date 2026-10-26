"""Device-side dual-tree work queue (PLAN.md 11, deviations D012/D013).

The reference descent in :func:`jzradius.traversal.build_work_items` walks the
dual tree in interpreted Python, which dominated every GPU call: at ``N=1e5`` it
took 2.6 s out of 2.9 s end to end (see ``docs/benchmarks.md``).  This module
runs the *same* descent as a sequence of device kernels: one breadth-first level
per launch, with the frontier and the emitted leaf pairs living on the device.

Only the pruning/expansion decision is shared logic, and it lives in the shared
arithmetic header (``src/cuda/jzradius_kernels.cuh::plan_pair``) so the CUDA
kernel and its CPU mirror cannot drift apart, exactly like the pair arithmetic.
The resulting work items are identical (as sets) to the host descent; that is
what ``tests/test_work_queue.py`` asserts.

The buffers are pre-allocated and re-run with larger capacity whenever a level
reports an overflow, so no interaction is ever silently dropped.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from . import ffi
from .errors import RadiusConfigError, WorkQueueOverflow

_MAX_LEVELS = 4096
_MAX_CAP = 1 << 26


def _node_arrays(tree) -> dict:
    """Pack a :class:`~jzradius.tree_adapter.RadiusTree` into contiguous arrays."""
    return {
        "start": jnp.asarray(np.ascontiguousarray(tree.node_start, dtype=np.int32)),
        "count": jnp.asarray(np.ascontiguousarray(tree.node_count, dtype=np.int32)),
        "left": jnp.asarray(np.ascontiguousarray(tree.node_left, dtype=np.int32)),
        "right": jnp.asarray(np.ascontiguousarray(tree.node_right, dtype=np.int32)),
        "lo": jnp.asarray(np.ascontiguousarray(tree.node_lo, dtype=np.float64)),
        "hi": jnp.asarray(np.ascontiguousarray(tree.node_hi, dtype=np.float64)),
    }


def build_work_items_device(
    tree_q,
    tree_s=None,
    *,
    r_max: float,
    boxsize: Optional[float] = None,
    max_work_items: int = 1 << 22,
    prune_tol: Optional[float] = None,
    unique_pairs: bool = True,
    platform: Optional[str] = None,
):
    """Return the ``(W, 4)`` int32 leaf-pair work items, built on the device.

    Same contract as :func:`jzradius.traversal.build_work_items`: half-open
    particle ranges ``(q0, q1, s0, s1)`` in tree-sorted space, ``unique_pairs``
    descending only the upper triangle of self-pairs.
    """
    if r_max <= 0 or not np.isfinite(r_max):
        raise RadiusConfigError(f"r_max must be positive and finite, got {r_max}")
    self_query = tree_s is None or tree_s is tree_q
    tree_s = tree_q if tree_s is None else tree_s

    L = float(tree_q.boxsize if boxsize is None else boxsize)
    periodic = bool(np.isfinite(L) and L > 0)
    if prune_tol is None:
        prune_tol = 8.0 * np.finfo(np.float64).eps * (L if periodic else 1.0)

    qa = _node_arrays(tree_q)
    sa = qa if self_query else _node_arrays(tree_s)

    frontier = jnp.asarray([[tree_q.root, tree_s.root]], dtype=jnp.int32)
    next_cap = min(max(16, 8 * tree_q.n_nodes), _MAX_CAP)
    work_cap = min(max(1024, 16 * tree_q.n_nodes), max_work_items)
    chunks = []

    for _level in range(_MAX_LEVELS):
        n_in = int(frontier.shape[0])
        if n_in == 0:
            break

        for _attempt in range(12):
            nxt, work, counts = ffi.work_items(
                qa["start"], qa["count"], qa["left"], qa["right"], qa["lo"],
                qa["hi"], sa["start"], sa["count"], sa["left"], sa["right"],
                sa["lo"], sa["hi"], frontier,
                next_cap=next_cap,
                work_cap=work_cap,
                self_query=self_query,
                unique_pairs=bool(unique_pairs),
                r_max=float(r_max),
                prune_tol=float(prune_tol),
                boxsize=L if periodic else 0.0,
                platform=platform,
            )
            counts_host = np.asarray(counts)
            n_next = int(counts_host[0])
            n_work = int(counts_host[1])
            overflow = int(counts_host[2])
            if not overflow:
                break
            grew = False
            if n_next > next_cap:
                next_cap = min(4 * n_next, _MAX_CAP)
                grew = True
            if n_work > work_cap:
                if n_work > max_work_items:
                    raise WorkQueueOverflow(
                        f"work queue exceeded max_work_items={max_work_items} "
                        f"({n_work} leaf pairs on a single level); "
                        "increase the limit or reduce r_max"
                    )
                work_cap = min(4 * n_work, max_work_items)
                grew = True
            if n_next > _MAX_CAP:
                raise WorkQueueOverflow(
                    f"dual-tree frontier exceeded {_MAX_CAP} node pairs; "
                    "increase the limit or reduce r_max"
                )
            if not grew:
                raise RadiusConfigError(
                    "device work-queue level did not fit its buffers"
                )
        else:
            raise RadiusConfigError("device work-queue level overflowed repeatedly")

        total = sum(int(c.shape[0]) for c in chunks) + n_work
        if total > max_work_items:
            raise WorkQueueOverflow(
                f"work queue exceeded max_work_items={max_work_items} "
                f"({total} leaf pairs so far); increase the limit or reduce r_max"
            )
        if n_work:
            chunks.append(work[:n_work])
        frontier = nxt[:n_next]

    if not chunks:
        return jnp.zeros((0, 4), dtype=jnp.int32)
    return jnp.concatenate(chunks, axis=0)
