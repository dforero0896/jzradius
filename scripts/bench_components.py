"""Component-level benchmark for jzradius (PLAN_ADDENDUM.md M9.0, M9.4).

Every M9 before/after claim must cite ``benchmarks/components.json``; this
script produces it.  Per configuration it reports, over ``--repeats`` runs
(median/min/max):

    tree_build_s        jztree zsort + build_tree_hierarchy (device FFI)
    flatten_s           packed levels -> canonical flat node arrays (host)
    workitems_host_s    interpreted dual-tree descent (v1 path)
    workitems_device_s  device BFS work queue (M9.1 path, None on v1)
    forward_kernel_s    the forward FFI call alone, prebuilt buffers
    backward_kernel_s   the backward FFI call alone, prebuilt buffers
    python_overhead_s   wall_fwd - the sum of the four stages above
    wall_fwd_s          public forward call, end to end
    wall_fwd_bwd_s      public value_and_grad call, end to end

Timing method: host segments use ``time.perf_counter``; device segments are
bounded by an explicit ``block_until_ready`` (jaxlib exposes no CUDA event API
to Python, so kernel times include launch overhead).  Run from node-local
scratch on the cluster (D003): every result records ``cwd`` so this is
auditable.

Usage
-----
    python scripts/bench_components.py --n 10000 100000 --out benchmarks/components.json
    python scripts/bench_components.py --n 100000 --repeats 5 --leaf-size 32

Refuses ``N > 1e5`` unless ``--allow-large`` is passed (PLAN.md 2.6 / 5.1).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (after the x64 flag)

from jzradius import ffi  # noqa: E402
from jzradius.histogram import (  # noqa: E402
    _build_jztree_trees,
    build_radius_trees,
    radius_softmax_histogram,
)
from jzradius.traversal import build_work_items  # noqa: E402

try:  # the M9.1 device queue; absent on the v1 baseline
    from jzradius.work_queue import build_work_items_device
except Exception:  # pragma: no cover - v1 worktrees
    build_work_items_device = None


def _sync():
    jax.block_until_ready(jnp.zeros(1, dtype=jnp.float32))


def _median_run(fn, repeats: int) -> Dict[str, float]:
    """Warm up once, then time ``repeats`` runs; return median/min/max."""
    fn()
    _sync()
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, jax.Array):
            jax.block_until_ready(out)
        _sync()
        samples.append(time.perf_counter() - t0)
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
    }


def _stages(n: int, leaf_size: int, boxsize: float, r_max: float, centers):
    """Build every input the forward path needs, on the host, once."""
    rng = np.random.default_rng(0)
    pos = rng.random((n, 3))
    pos_j = jnp.asarray(pos)

    def tree_only():
        return _build_jztree_trees(pos_j, pos_j, boxsize, leaf_size)

    def full():
        return build_radius_trees(pos, None, boxsize=boxsize, leaf_size=leaf_size)

    return pos, pos_j, tree_only, full


def bench_case(
    n: int,
    *,
    repeats: int,
    leaf_size: int,
    boxsize: float,
    r_max: float,
    centers_np: np.ndarray,
    temperature: float,
) -> Dict[str, object]:
    pos, pos_j, tree_only, full = _stages(n, leaf_size, boxsize, r_max, centers_np)

    # --- tree build vs flatten -------------------------------------------
    # `tree_only` returns a flat RadiusTree, so we time the pair once and
    # subtract the host flatten measured separately on the same hierarchy.
    import jztree.tree as jt

    cfg_tree = jt.TreeConfig(max_leaf_size=leaf_size)

    def build_hier():
        posz, _ = jt.zsort(pos_j)
        return jt.build_tree_hierarchy(posz, cfg_tree)

    from jzradius.tree_adapter import build_radius_tree_from_jztree

    posz, idz = jt.zsort(pos_j)
    th = jt.build_tree_hierarchy(posz, cfg_tree)
    posz_h, idz_h = np.asarray(posz), np.asarray(idz)

    def flatten():
        return build_radius_tree_from_jztree(
            th, posz_h, idz_h, boxsize=boxsize, leaf_size=leaf_size
        )

    tree_build = _median_run(build_hier, repeats)
    flatten_t = _median_run(flatten, repeats)

    tree_q, tree_s, _ = full()
    unique = True

    work_host = build_work_items(
        tree_q, tree_s, r_max=r_max, boxsize=boxsize, max_work_items=1 << 22,
        unique_pairs=unique,
    )
    workitems_host = _median_run(
        lambda: build_work_items(
            tree_q, tree_s, r_max=r_max, boxsize=boxsize,
            max_work_items=1 << 22, unique_pairs=unique,
        ),
        repeats,
    )
    if build_work_items_device is not None:
        work_dev = np.asarray(build_work_items_device(
            tree_q, tree_s, r_max=r_max, boxsize=boxsize,
            max_work_items=1 << 22, unique_pairs=unique,
        ))
        workitems_device = _median_run(
            lambda: build_work_items_device(
                tree_q, tree_s, r_max=r_max, boxsize=boxsize,
                max_work_items=1 << 22, unique_pairs=unique,
            ),
            repeats,
        )
        identical = sorted(map(tuple, work_host.tolist())) == sorted(
            map(tuple, work_dev.tolist())
        )
    else:
        workitems_device = None
        identical = None

    # --- kernels alone ----------------------------------------------------
    perm = jnp.asarray(tree_q.sorted_to_orig, dtype=jnp.int32)
    qpos = jnp.take(tree_q.sorted_pos, jnp.asarray(tree_q.sorted_to_orig), axis=0)
    w = jnp.ones((n,), dtype=jnp.float64)
    work_j = jnp.asarray(work_host, dtype=jnp.int32)
    centers_j = jnp.asarray(centers_np, dtype=jnp.float64)
    params = ffi.build_params(
        n_q=n, n_s=n, n_bins=int(centers_np.size), n_work=int(work_j.shape[0]),
        self_query=True, triangular=unique, mode="softmax", r_max=r_max,
        boxsize=boxsize, temperature=temperature, eps=1e-60,
        r_safe=1e-12 * boxsize, pair_scale=1.0,
    )

    def fwd_kernel():
        return ffi.forward(qpos, qpos, centers_j, work_j, w, w, perm, params)

    hist = fwd_kernel()
    hist = jax.block_until_ready(hist)
    fwd_k = _median_run(fwd_kernel, repeats)

    cot = jnp.asarray(np.full(centers_np.size, 0.1))

    def bwd_kernel():
        return ffi.backward(
            qpos, qpos, centers_j, work_j, w, w, perm, cot, params
        )

    bwd_k = _median_run(bwd_kernel, repeats)

    # --- end to end -------------------------------------------------------
    def public_fwd():
        return radius_softmax_histogram(
            pos, bin_centers=centers_j, temperature=temperature,
            boxsize=boxsize, r_max=r_max, self_query=True, leaf_size=leaf_size,
            max_work_items=1 << 22,
        )

    def public_grad():
        g = jax.grad(lambda p: jnp.sum(public_fwd() ** 2))
        return g(pos)

    wall_fwd = _median_run(public_fwd, repeats)
    wall_grad = _median_run(public_grad, repeats)

    # Host fraction == everything that is not a device kernel in forward:
    # the flatten (host numpy), the work-item stage when it is the host
    # descent, and all Python glue.  Device stages are the jztree build, the
    # device work queue (if present) and the histogram kernel.
    device = {
        k: tree_build[k] + fwd_k[k]
        + (workitems_device[k] if workitems_device else 0.0)
        for k in ("median", "min", "max")
    }
    out: Dict[str, object] = {
        "N": n,
        "leaf_size": leaf_size,
        "dtype": "float64",
        "r_max": r_max,
        "n_work_items": int(work_host.shape[0]),
        "n_nodes": int(tree_q.n_nodes),
        "backend": jax.default_backend(),
        "device": jax.devices()[0].device_kind,
        "work_items_device_matches_host": identical,
        "tree_build_gpu_s": tree_build,
        "flatten_s": flatten_t,
        "workitems_host_s": workitems_host,
        "workitems_device_s": workitems_device,
        "forward_kernel_s": fwd_k,
        "backward_kernel_s": bwd_k,
        "python_overhead_s": {
            k: wall_fwd[k] - device[k] for k in ("median", "min", "max")
        },
        "wall_fwd_s": wall_fwd,
        "wall_fwd_bwd_s": wall_grad,
    }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[10000, 100000])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--leaf-size", type=int, default=32)
    ap.add_argument("--boxsize", type=float, default=1.0)
    ap.add_argument("--r-max", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/components.json"))
    ap.add_argument("--allow-large", action="store_true",
                    help="allow N > 1e5 (requires PLAN.md 2.6 approval)")
    args = ap.parse_args(argv)

    too_large = [n for n in args.n if n > 100000]
    if too_large and not args.allow_large:
        raise SystemExit(
            f"refusing N={too_large} without --allow-large (PLAN.md 2.6)"
        )

    centers = np.linspace(1e-3, args.r_max * 0.9, args.bins)
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
        "git_describe": os.popen(
            "git describe --tags --always --dirty 2>/dev/null").read().strip(),
        "host": platform.node(),
        "cwd": os.getcwd(),
        "repeats": args.repeats,
        "temperature": args.temperature,
        "bins": args.bins,
        "cases": [],
    }
    for n in args.n:
        case = bench_case(
            n, repeats=args.repeats, leaf_size=args.leaf_size,
            boxsize=args.boxsize, r_max=args.r_max, centers_np=centers,
            temperature=args.temperature,
        )
        payload["cases"].append(case)
        m = case["wall_fwd_s"]["median"]
        host = case["python_overhead_s"]["median"]
        print(
            f"N={n:>8}  tree={1e3*case['tree_build_gpu_s']['median']:7.1f}ms "
            f"flatten={1e3*case['flatten_s']['median']:6.1f}ms "
            f"work_host={1e3*case['workitems_host_s']['median']:7.1f}ms "
            f"work_dev="
            f"{(1e3*case['workitems_device_s']['median'] if case['workitems_device_s'] else float('nan')):6.1f}ms "
            f"fwd_kernel={1e3*case['forward_kernel_s']['median']:6.1f}ms "
            f"wall_fwd={m:6.3f}s wall_grad={case['wall_fwd_bwd_s']['median']:6.3f}s "
            f"host_frac={100.0*host/m:5.1f}%"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
