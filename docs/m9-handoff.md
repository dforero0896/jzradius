# M9 handoff — state of play and next GPU tasks

Written 2026-10-26 while the GPU partition was unresponsive (jobs sat in `R` with
no output for 6+ minutes; see "Cluster notes"). Nothing below was measured on a
healthy GPU, so no performance claim should be quoted from this page.

## Where the code stands

* `HEAD = 92fa78d` — device dual-tree work queue (D013) and `add_balancing_levels`
  (D014) are committed; `v1.0.0` tag = `228f41c` is the frozen correctness baseline.
* Working tree (uncommitted) contains the plane-native M9 work: the single
  `jzradius_traverse` op (`src/cuda/jzradius_traverse.{cuh,cu}`,
  `src/cpu/jzradius_traverse_cpu.cpp`), the `ffi.traverse` binding, the production
  switch in `histogram.py::_hist_core_fwd` (GPU + `self_query` uses
  `jztree_planes` with `coarse_fac=4, stop_coarsen=1`), sentinel padding of the
  interaction list, and `tests/test_traverse_planes.py` (13 cases, green on CPU).
* CPU suite last known green: `303 passed, 4 skipped` including
  `test_traverse_planes.py`.
* **Not yet validated on GPU**: the plane path's numerics (histogram + gradients)
  and its timing. The previous attempt died in the cluster, not in the code.
* Draft, deliberately not wired into the build: `src/cuda/jzradius_cuda_shared.cu`
  and `src/cuda/jzradius_kernels_shared.cuh` (shared-memory G4 optimisation).
  Feasibility review is in the conversation; the fix list is below.

## Task 1 — measure the kernel (decides when G4 happens)

`benchmarks/components.json` says `forward_kernel_s = 2.7 ms` at N=1e5, which
cannot be right for ~6e7 candidate pairs and ~2.6e6 accepted pairs × 2·`n_bins`
`exp` calls. Suspect that field timed the asynchronous launch, not the kernel.

`scratch/kernel_timer.py` (outside the repo) isolates it: the `ffi` call is
wrapped in `jax.jit`, 20 calls are queued and one `block_until_ready()` amortises
launch overhead, a single-sync median is reported beside it, and candidate counts
are summed from the interaction list so the arithmetic floor is visible.

```
cd /home/dforero/coder/jzradius_dev
NOBUILD=1 srun -p gpu --gres=gpu:1 -n1 -t 0:20:00 \
  scratch/run_gpu.sh /home/dforero/coder/jzradius_dev/jzradius timer
```

The runner copies the repo to node-local `/tmp` (D003), rebuilds only when
`NOBUILD` is unset, and mirrors stdout to `jzradius/gpulog.txt`. Both `.so`
artifacts were rebuilt on the login node at 16:44 (sm_80/86/89/90), so
`NOBUILD=1` is correct as long as the sources do not change.

Decision rule — G4 is worth doing **now** only if the forward+backward kernels
are a large share of the ~0.31 s wall time at N=1e5. If they come out at a few
tens of ms (the expected answer), G4 stays queued behind Task 3, because the
remaining wall time is host-side jztree tree construction plus Python dispatch.

## Task 2 — validate the plane path on GPU

```
NOBUILD=1 srun -p gpu --gres=gpu:1 -n1 -t 0:25:00 \
  scratch/run_gpu.sh /home/dforero/coder/jzradius_dev/jzradius validate
```

`scratch/plane_validate.py` compares the public API against
`reference_histogram` (value, position gradients, weight gradients) at N=500/2000,
then times N=1e4/1e5 and reports `len(_TREE_FALLBACKS)` so a silent fall back to
the naive tree cannot be mistaken for the plane path. Then the full suite:

```
NOBUILD=1 srun -p gpu --gres=gpu:1 -n1 -t 0:40:00 \
  scratch/run_gpu.sh /home/dforero/coder/jzradius_dev/jzradius pytest
```

Expect `JAX_PLATFORMS=cuda,cpu` inside the runner (part of the suite uses
`jax.io_callback`, which needs a CPU platform registered). GPU totals last
measured on a healthy A30: 293 passed, 1 skipped; they must not regress.

## Task 3 — host path first (highest value)

`build_tree_hierarchy` alone was 120–160 ms at N=1e5, i.e. about half the wall
clock, and no kernel work touches it. Addendum T1/T4 (plane-native traversal
already done) plus a `histogram_from_tree` entry point that accepts a
pre-built plane set would remove it. Do this before G4 unless Task 1 says the
kernels are dominant.

## Task 4 — the shared-memory draft (fix list before any test)

Read `src/cuda/jzradius_kernels_shared.cuh` and `src/cuda/jzradius_cuda_shared.cu`
first; the design is sound but as delivered it cannot compile or run.

1. `jzradius_cuda_shared.cu:14` includes `jzradius_kernels.cuh`, yet
   `Params::max_leaf_size` exists only in the `_shared` header → compile error.
   Keep **one** arithmetic header (add the field to `jzradius_kernels.cuh` and
   delete the duplicate); two copies break the single-source-of-truth promise in
   `docs/algorithm.md`.
2. `ffi.py:148 _attrs()` has no `max_leaf_size`, so the new `.Attr<int32_t>`
   (`:368`, `:467`) is unbound and every call would fail. Append it after
   `pair_scale`, add the kwarg to `forward()`/`backward()`, and pass it from
   `histogram.py` (the leaf size actually used to build the tree, including the
   naive fallback).
3. `scripts/build_ext.py:37` does not compile the `_shared` translation unit and
   the CPU mirror still includes the old header → the draft is dead code and the
   CPU/GPU parity guarantee silently lapses.
4. No bound check on `s_count = item.s1 - item.s0` (`:102`, `:191`) before
   writing `shared_spos[3*t]`. Leaves are bounded by `max_leaf_size` today
   (jztree splits at `npart > max_leaf_size`), but a leaf-size mismatch would
   corrupt shared memory. Add `if (s_count > L) __trap();` and validate
   `max_leaf_size >= 1` in the launcher.
5. The launchers never check `cudaGetLastError()`; an oversized `leaf_size`
   exceeds the dynamic shared-memory limit and fails silently, returning a
   plausible-looking zero histogram. Bound `smem_bytes` and return an FFI error.

Cheap wins inside the same kernel, independent of staging: `pair_statistics`
re-reads `g[b]` from global for every pair (hoist it into registers once per
thread); `pair_norm` + `bin_weight` compute `exp` twice per bin (halve it with a
`n_bins <= 32` register cache); one block per 32×32 leaf pair under-utilises the
SM for small leaves; shared-`double` `atomicAdd` on ~20 addresses is a
serialisation point where a per-warp reduction is usually faster.

## Cluster notes

* Symptom of the current outage: `squeue` shows `R` with growing `TIME`, but
  `gpulog.txt` only ever contains the `== timer` banner — no Python output at
  all, i.e. the stall is before/inside `import jax` (same signature as D003).
  Cancel rather than wait; earlier attempts burned 700 s and a 25 min wall clock
  with zero output.
* Always run on node-local `/tmp` via `scratch/run_gpu.sh`; never time anything
  from `/home` (NFS).
* `scancel -u dforero` clears stragglers; check `squeue` before starting a new
  attempt so a zombie step does not shadow your job on the same node.

## Housekeeping

`gpulog.txt`, `plane_validate.py` and `run_gpu.sh` leaked into the repo root
(untrusted scratch copies). Delete or gitignore them before the next commit.
