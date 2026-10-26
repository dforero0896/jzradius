# Known limitations (PLAN.md §23)

Status: v1. Forward and backward kernels have been executed and validated on a
real GPU (A30); the remaining limitations are the ones below, primarily
performance (item 11) and the eager-only API (item 12).

## 1. Minimum-image regime only (`r_max < L/2`)

`periodic_delta` subtracts a single nearest image per axis:
`d -= L * round(d / L)`. This is exact only when `r_max < L/2`, which is
enforced by `validate_r_max` (`0 < r_max < boxsize/2`) and re-checked in
`RadiusHistogramConfig`. No multiple-image sums are attempted; if a distance
distribution needs contributions from several images, this library cannot
represent it.

## 2. Discrete topology

The neighbour set is a hard cutoff at `r_max`, so the map from positions to the
pair set is piecewise constant. Gradients are those of the *smooth* part
(softmax/Gaussian bin weights) and are correct almost everywhere; crossing
`r_max` or a leaf/work-item boundary changes the set and the gradient jumps.
Soft binning does not remove this discontinuity, and no attempt is made to
detect or report it.

## 3. Atomics and `deterministic=False`

The CUDA forward/backward kernels accumulate with `atomicAdd`. Runs are
usually bitwise identical on the same device, but the hardware provides no
ordering guarantee. `deterministic=True` is rejected with an explicit
`JzRadiusError` in v1 rather than silently ignored. A deterministic path
(sorted accumulation or per-block reduction trees) is future work.

## 4. Softmax tail truncation at `r_max`

Pairs are dropped at `r_max`, so the sum of softmax weights over the
*retained* catalogue is not the full-catalogue sum. For very small `T` the
dropped tail is negligible; for `T` comparable to `r_max` it is not.
Callers who need a normalization that includes the tail must add the analytic
tail themselves — this library reports the truncated sum.

## 5. Small `T` conditioning

`temperature` is floored at `1e-6` (float64) / `1e-4` (float32) by
`validate_temperature`. As `T -> 0` the softmax saturates: weights become 0/1,
`dL/dr` collapses to the nearest bin only, and the VJP becomes poorly
conditioned (`dL/dr = C - A*B` loses precision through cancellation). Expect
gradient noise proportional to `1/T^2`; prefer larger `T` or Gaussian mode for
gradient-based work.

## 6. Coincident particles and `r_safe`

Distances are evaluated as `r = sqrt(r2 + eps)` with `eps = 1e-60` (float64) /
`1e-30` (float32), which keeps `r > 0` for exactly coincident distinct
particles. The backward pass additionally skips pairs with
`r < r_safe = max(1e-12 * L, eps)`, so those pairs contribute to the forward
histogram but receive **zero** gradient. This avoids `1/r` blow-up but means a
zero-separation configuration is not differentiable.

## 7. Static binning parameters / recompilation

`bin_centers`, `temperature`, `r_max`, `boxsize`, `leaf_size`, `mode` and the
pair convention are static arguments of the `custom_vjp` core
(`jax.custom_vjp(nondiff_argnums=...)`). Changing any of them retraces the
forward and recompiles the FFI calls. Changing `n_bins` also changes the output
shape, so a caller looping over bin counts triggers one compilation per count.

## 8. Single device

v1 targets one GPU. There is no sharding, no multi-GPU interaction list, and no
halo exchange. `jztree`'s multi-device tree facility exists upstream but is not
wired into `jzradius`.

## 9. Upstream attribution

`jztree` is consumed as an installed dependency (pinned commit in
`docs/environment_lock.md`); no source is vendored or forked. If code is ever
copied from it, the original copyright and license must be retained and the
copy called out in `docs/discrepancies.md`.

## 10. KNN scaffolding never on the production path

`jztree.knn` was inspected during the M2 upstream audit and is **not** used by
`jzradius`. The production path is Morton-tree adaptation plus dual-tree radius
traversal (`jzradius/traversal.py` for CPU/work-item generation,
`src/cuda/jzradius_cuda.cu` for the device kernels). No approximate-KNN result
is used to claim a radius query.

## 11. Residual host work in the call path

`custom_vjp`'s forward rule builds the jztree hierarchy on the device but still
flattens it and drives the FFI calls from Python. After the M9 work (device-side
dual-tree work queue, D013, and balancing levels over jztree's flat small-N
hierarchies, D014) the wall-clock forward on one A30 (float64, 20 bins,
`r_max=0.05`, `L=1`) is 0.162 s at N=1e4, 0.180 s at N=5e4 and 0.310 s at
N=1e5, with `jax.grad` within 1.08x, i.e. 9.4x faster than v1 at N=1e5. The
host share of the call is 14% at N=1e4 and 45% at N=1e5 (validation,
`jnp.take`, `ffi_call` dispatch, sync) -- that, not the tree build, is what
remains of D012. The PLAN §21 targets (1e6 particles in seconds) are still not
demonstrated: N=1e6 needs the §2.5 scale-tier approval, and removing the Python
overhead needs the jit-able entry point (next item).

## 12. Eager-only public API

Because the forward rule needs concrete host arrays, `jax.jit` around
`radius_softmax_histogram` raises an explicit `JzRadiusError` (D008).
`jax.grad`/`value_and_grad` work eagerly, and the gradient is the analytic
kernel VJP rather than autodiff through a fallback.

## 13. Small-input tree fallback

`jztree`'s leaf-buffer estimate can raise "Tree-Leaf allocation too small" for
small/irregular inputs (e.g. N=17 with `max_leaf_size=32`). In that case
`jzradius` falls back to its own median-split tree, which is exact (same
traversal and kernels) but slower to build. See D011.

## Verification status

- CPU mirror of the shared arithmetic: compiled (`libjzradius_cpu.so`) and
  exercised through the **same typed-FFI/custom_vjp path** as the GPU build;
  237 passed, 3 skipped (skips: the GPU-only jztree adapter test, the
  jztree-dependent FFI test when jztree is not on `PYTHONPATH`, and the optional
  pyfcfc cross-check; all three clear on a GPU node / with pyfcfc).
- CUDA: `nvcc` 12.3 builds `libjzradius_cuda.so` for `sm_80/86/89/90` and
  exports `JzRadiusForward`, `JzRadiusBackward`, `JzRadiusNodeBounds`.
- CUDA **runtime**: exercised on one A30 (driver 610.57.04). The complete suite
  passes on the device -- 239 passed in ~190 s with `JAX_PLATFORMS=cuda,cpu` --
  including forward parity against `reference_histogram` for f32/f64,
  softmax/Gaussian, and position/weight gradients against
  `jax.grad(reference_histogram)` for self and cross correlation.
- External cross-check (independent implementation and adjoint): pyfcfc
  `0.2.0`'s soft histogram and `soft_hist_vjp`. Counts match to `5e-15`
  relative and coordinate gradients to `2e-13` (N = 220, 8 bins, T = 0.01,
  `sigma = T/sqrt(2)`, factor 2 for ordered vs unordered counting). Run with
  `examples/example_soft_hist_jax.py` or `tests/test_external_pyfcfc.py`.
- Timing (not a correctness gate) is in `docs/benchmarks.md`; it reflects
  host-side construction, see item 11.
