# Testing

Test policy follows PLAN.md §18: tiered by size, fixed seeds, three point
distributions, and exact pair-set comparisons instead of count-only checks.

## Tiers and commands

| Tier | Marker | Size cap | Wall cap | Script |
|---|---|---|---|---|
| smoke | `smoke` | N ≤ 32 | 60 s | `scripts/run_smoke.py` |
| unit | `unit` | N ≤ 256 | 300 s | `scripts/run_unit.py` |
| integration | `integration` | N ≤ 20 000 | 900 s | `scripts/run_integration.py` (GPU) |
| scale | — | N ≥ 1e6 | approval | `scripts/run_benchmark.py` (not written) |

```bash
# while .venv lives on NFS, the -S/PYTHONPATH shim avoids the site stall (D003)
V=.venv/lib/python3.11/site-packages
W=/home/dforero/miniconda3/envs/webjax/lib/python3.11/site-packages
env PYTHONPATH="$PWD:$V:$W" JAX_PLATFORMS=cpu .venv/bin/python -S -m pytest tests/ -q
env PYTHONPATH="$PWD:$V:$W" JAX_PLATFORMS=cpu .venv/bin/python -S scripts/run_unit.py
env PYTHONPATH="$PWD:$V:$W" JAX_PLATFORMS=cpu .venv/bin/python -S scripts/run_smoke.py
```

Current totals: 237 passed, 3 skipped on CPU, and 239 passed, 0 skipped on a
GPU (one A30, ~190 s). The CPU skips are the `@pytest.mark.gpu` jztree adapter
test, the jztree-dependent FFI test when `jztree` is not on `PYTHONPATH`, and
`tests/test_external_pyfcfc.py`, which needs the optional pyfcfc build (see
`docs/examples.md`); all of them clear on a GPU node or with those two
dependencies present. The GPU run needs both platforms registered:

```bash
srun -p gpu --gres=gpu:1 -n1 -t 0:20:00 /bin/bash -c '
  cd /home/dforero/coder/jzradius_dev && V=$PWD/.venv/lib/python3.11/site-packages
  W=/home/dforero/miniconda3/envs/webjax/lib/python3.11/site-packages
  env PYTHONPATH="$PWD:$PWD/jztree/src:$V:$W" JAX_PLATFORMS=cuda,cpu \
      JZTREE_SKIP_JAX_CUDA_CHECK=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      .venv/bin/python -S -m pytest tests -p no:randomly'
```

`JAX_PLATFORMS=cuda,cpu` (not `cuda` alone) is required: several tests use
`jax.io_callback`, which needs a CPU device and otherwise fails with
"failed to find a local CPU device". `$PWD/jztree/src` keeps the editable
upstream clone importable inside the allocation (D005).

The optional external parity test against pyfcfc's independent soft-histogram
adjoint lives in `tests/test_external_pyfcfc.py` (marked `slow`, module-level
`importorskip`), so it never blocks the default tiers:

```bash
V=.venv/lib/python3.11/site-packages
W=/home/dforero/miniconda3/envs/webjax/lib/python3.11/site-packages
env PYTHONPATH="$PWD:$V:$W:$PWD/.pyfcfc_site" JAX_PLATFORMS=cpu \
    .venv/bin/python -S -m pytest tests/test_external_pyfcfc.py -q
```

`scripts/_runner.py` enforces the wall cap with a `timeout`, runs
`--maxfail=3`, and prints only the tail (context hygiene, PLAN §2.4).

## Fixtures

`tests/conftest.py` enables x64 **before** importing `jax.numpy`, defines
`SEEDS = [0, 1, 7, 42, 1234]`, and three distributions:

- `uniform_points` — iid uniform in `[0, L)^3`
- `clustered_points` — 5 lognormal-scale clusters (σ = 0.01 L), wrapped
- `boundary_points` — near the 0/L faces, including seam-crossing samples

Clustered data is mandatory for pruning tests (PLAN §18.4).

## Test inventory

| File | Covers |
|---|---|
| `test_math.py` | softmax normalization/stability, analytic `dW/dr`, VJP identities, single-bin limit, config validation |
| `test_periodic.py` | MIC wraparound table, half-box ties, zero separation, MIC gradient, NumPy oracle agreement |
| `test_reference.py` | pair conventions, weights, periodic distances, finite-difference gradient, pair-set ordering |
| `test_bounds.py` | `periodic_interval_gap` vs NumPy oracle, seam boxes, **mandatory lower-bound ≤ true min distance property test** (PLAN §10.5), sphere bounds, prune tolerance |
| `test_tree_adapter.py` | mapping invertibility, leaf partition, conservative boxes, radius = ½ box diagonal, synthetic level planes, invariant checker, GPU-gated real `TreeHierarchy` |
| `test_traversal_cpu.py` | stats consistency, pruning effectiveness, tiny-radius/no-pair edge cases, determinism, leaf-size independence, validation |
| `test_pair_set_equivalence.py` | **exact pair set** vs brute force: N ∈ {2,3,17,64,257}, 3 distributions, L ∈ {1,100}, all seeds, `unique`/`directed_half`/cross, weighted/unweighted histograms |
| `test_ffi_forward.py` | typed-FFI forward through the real `custom_vjp` API vs `reference_histogram`: softmax/Gaussian, f32/f64, L ∈ {1,100}, 3 distributions; raw kernel vs CPU traversal; `directed_half == unique`; empty interaction list; GPU smoke (gated) |
| `test_ffi_backward.py` | typed-FFI backward vs `jax.grad(reference_histogram)`: positions and weights, self and cross correlation, f32/f64, `grad_wq`/`grad_ws` independence, unused-weight zero, explicit `jit`-unsupported error |

## Tolerances (PLAN §18.3)

| Quantity | Tolerance |
|---|---|
| float64 forward, deterministic reduction | atol 1e-9, rtol 1e-8 |
| float64 forward, atomics | atol 1e-7, rtol 1e-6 |
| float64 gradient | atol/rtol 1e-5 |
| float32 gradient | atol/rtol 1e-3 |

CPU traversal tests compare pair sets by canonical `(min(i,j), max(i,j))`
because permuting back from tree-sorted order may flip `i`/`j`. Distances are
compared at 1e-12, histograms at 1e-9 absolute.

## Index-space rules (learned the hard way)

- `traverse_tree_pairs` returns **original input indices**; do not apply
  `sorted_to_orig` a second time.
- `reference_histogram` / `reference_pair_set` operate in the index space of
  the arrays passed in. If you hand them `tree.sorted_pos`, per-particle
  weight arrays must be permuted with `tree.sorted_to_orig`.
- For `self_query=True` with asymmetric query/source weights the orientation
  of an unordered pair is index-space dependent (D006); use symmetric weights
  or cross-correlation for weighted histogram comparisons.
- FFI outputs come back in original input order; do not unsort them again.
- `build_work_items(unique_pairs=...)` controls whether both orientations are
  emitted for self-query: `True` for `unique`, `False` for `directed_half`.
- The gradients must always be checked against
  `jax.grad(reference_histogram)` with the same conventions, not against a
  hand-rolled formula.

## Skips

GPU markers are skipped explicitly on CPU hosts. A skipped GPU test is never
counted as a pass. On a GPU host nothing is skipped any more; the CPU suite's
skips are the `@pytest.mark.gpu` tree-adapter test, whose upstream tree
construction is
itself an FFI kernel.

The CPU FFI tests need `jzradius/_libs/libjzradius_cpu.so`; the `ffi_cpu`
session fixture builds it via `scripts/build_ext.py --no-cuda` when absent and
skips (with the build error) if `g++`/the headers are unavailable. The CUDA
artifact is built by the same script but only loaded when
`jax.default_backend() == "gpu"`, so a stale CUDA `.so` can never mask a CPU
run.

## Backend parity

`src/cuda/jzradius_kernels.cuh` is compiled twice — by `nvcc` into
`libjzradius_cuda.so` and by `g++` into `libjzradius_cpu.so`. Tests drive the
CPU artifact through the same `jax.ffi.register_ffi_target` +
`jax.ffi.ffi_call` + `custom_vjp` path used on GPU, so a passing CPU suite
checks tree construction, work-item generation, FFI plumbing, bin math and the
VJP. The CUDA artifact is additionally exercised on-device: the whole suite
(including `test_forward_on_cuda_backend` and weight/position gradient parity
against `jax.grad(reference_histogram)`) passes on an A30 with
`JAX_PLATFORMS=cuda,cpu`.

`self_query=True, pair_convention="unique"` orients each unordered pair by
**input** index (`orig` buffer, D010); with asymmetric query/source weights
this is what makes the histogram permutation invariant, so weighted self tests
must pass `weights_query` only (or symmetric arrays).
