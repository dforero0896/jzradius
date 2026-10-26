# Discrepancies Between PLAN.md and Reality

Every entry is also appended to the Deviation Log in `PLAN.md` §25.
Format: plan section, plan claim, observed fact, decision.

## D001 — `jax.lib.xla_extension` removed (plan §4.3)

Plan's `check_env.py` snippet prints
`jax.lib.xla_extension.version()`. In JAX 0.10.2 that attribute is gone
(`hasattr(jax.lib, "xla_extension") == False`; jaxlib 0.10.2).

Decision: `scripts/check_env.py` reports `None` for it and prints
`jax.__version__` / `jaxlib.__version__` instead. No functional dependency on
the removed API anywhere else.

## D002 — `jztree` backend major vs JAX major (plan §4.2)

Plan assumes `pip install "jztree[cuda12|cuda13]"` matched to the stack. The
active environment instead has an **editable** dev build of the pinned clone
(`1.1.1.dev2+g39eea8ec2`) whose `jztree_cuda` comes from the clone's
`build/` directory and reports `CUDA_MAJOR = 12`, while JAX uses the CUDA 13
plugin (`jax-cuda13-plugin 0.10.2`). Import fails with the backend guard.

Decisions:

1. Keep the editable clone (user instruction; allows upstream edits).
2. Import with `JZTREE_SKIP_JAX_CUDA_CHECK=1` for CPU-side API work.
3. `jztree-cu13==1.1.0` installed in `.venv` as fallback, currently shadowed
   by the editable finder.
4. At M5, prove the cu12-backend/cu13-JAX mix with the trivial FFI op. If it
   fails, rebuild the editable extension with CUDA 13 or switch to the
   non-editable `jztree==1.1.0 + jztree-cu13` pair in `.venv`.

## D003 — GPU compute nodes stall on NFS-hosted Python (plan §4.1)

Plan treats GPU nodes as directly usable. On gn02, Python from `/home`
(NFSv3) takes ~18 s to start and `import numpy` exceeds 120 s at ~0 CPU;
login-node execution is normal.

Decision: CPU work proceeds; GPU execution is deferred. Mitigations to try at
M5 (in order): warm-cache rerun, node-local copy of the venv, container
image, or `pip install --target` into node-local `/tmp`. Escalate per §2.8 if
none works. This is a platform issue, not a library defect.

## D004 — `jztree.__version__` is absent (plan §4.3)

Plan's snippet uses `getattr(jztree, "__version__", "unknown")`, which is
correct: the editable build exposes the version only via
`jztree._version.version`. Documented so no code reads a missing attribute.

## D005 — Namespace shadowing of `jztree` by the clone (tooling)

Running Python from the repository root makes `./jztree/` (the clone) a
namespace package and hides the real install; it also affects
`import jztree_cuda`. `scripts/check_env.py` strips the CWD from `sys.path`,
and all tooling should run from `scripts/` or with `cd /tmp` avoided by using
the helper.

## D006 — Unordered self-pair orientation with asymmetric weights (plan §6.5)

For `self_query=True, pair_convention="unique"` the orientation of an
unordered pair depends on the index space. Enumerating `i < j` in the input
space vs a permuted tree-sorted space pairs `wq[i]*ws[j]` the other way round;
with `wq != ws` the two weighted histograms differ (observed ~6% for N=64),
while unweighted pair sets are identical. Decision: `unique` means `i < j` in
input space (CPU/GPU traversal authoritative); document that asymmetric
weights are orientation-dependent for auto-correlation. Classification:
SPEC_CLARIFICATION.

## D007 — `deterministic=True` rejected (plan §7.1, §23.3)

Both kernels accumulate with atomics; no deterministic reduction exists in v1.
`jzradius/histogram.py` raises `JzRadiusError` for `deterministic=True` instead
of silently returning run-to-run-varying sums. Classification: MISSING_FEATURE.

## D008 — Host-side tree build makes the public API eager-only (plan §14.1)

`custom_vjp` makes the *gradient* analytic, but the forward rule needs concrete
host arrays for `jztree`'s tree construction, so `jax.jit` around the public
call cannot trace it. `jax.grad`/`value_and_grad` work eagerly and raise an
actionable `JzRadiusError` under `jit`. Classification: PLATFORM.

## D009 — CUDA compiled but not yet executed (plan §1.5, M5–M8)

`nvcc` 12.3 builds `libjzradius_cuda.so` for sm_80/86/89/90 and exports
`JzRadiusForward`, `JzRadiusBackward`, `JzRadiusNodeBounds`. The numeric path is
validated through the CPU mirror of the same shared header, the same FFI
registration and the same `custom_vjp` rule. GPU execution is blocked by the
cluster stall (D003), so GPU tests skip (never pass) and completion is not
claimed. Classification: PLATFORM.

## D010 — `unique` self pairs are oriented by input index in the kernels (plan §6.5)

v1 enumerated unordered pairs in tree-sorted space, so with asymmetric
`weights_query != weights_source` both the value and the VJPs depended on the
permutation. The kernels now take an `orig` (sorted→original) buffer and
orient each unordered pair by `orig[i] < orig[j]`, swapping the query/source
roles, the weight slots, the accumulation indices and flipping the separation
vector. Evidence: `src/cuda/jzradius_cuda.cu`, `src/cpu/jzradius_cpu.cpp`,
`jzradius/ffi.py`, `jzradius/histogram.py`. Test:
`tests/test_ffi_backward.py::test_self_asymmetric_weights_follow_input_order`.
Classification: CORRECTNESS.

## D011 — Exact in-house tree fallback when jztree under-allocates leaves (plan §9.4)

`jztree`'s leaf buffer estimate raises `Tree-Leaf allocation too small` for
some small inputs (e.g. N=17, `max_leaf_size=32`); `alloc_fac_nodes` does not
help on that path. We retry, then fall back to the median-split tree, which is
exact (same traversal/kernels, only slower to build). Evidence:
`jztree/src/jztree/tree.py:257`, `jzradius/histogram.py::_build_jztree_trees`.
Classification: API_DRIFT.

## D012 — Host-side tree build dominates GPU runtime (plan §21)

Measured on one A30 (float64, 20 bins, r_max=0.05, L=1): fwd 0.76 s (n=1e4),
3.05 s (n=1e5), 18.2 s (n=3e5); `grad` ≈ 1.0x `fwd` because the kernels are
negligible beside the Python/NumPy tree + work-item construction. The >=1e6
scale tier still needs approval; device-side tree construction is the M9 fix.
Classification: PERFORMANCE.

## D013 — Dual-tree work queue runs on the device (plan §11, addendum M9.1)

The host descent in `jzradius/traversal.py` is interpreted Python and was 88%
of the call at N=1e5 (2.59 s of 2.91 s). The descent now runs as one typed-FFI
kernel per BFS level (`JzRadiusWork`/`JzRadiusWorkCpu`), with the prune/expand
decision factored into `jzradius::plan_pair` in the shared arithmetic header so
the CUDA kernel and its CPU mirror cannot drift. The host descent is kept as
the reference implementation and the parity oracle
(`tests/test_device_workitems.py`). Budgets are per-level capacities; an
overflow re-runs that level with 4x buffers, and a level that cannot fit
raises instead of silently dropping interactions. Measured (A30, N=1e5,
leaf_size=32, r_max=0.05): work items 2.59 s -> 9.6 ms, end-to-end forward
2.91 s -> 0.31 s, identical item sets. Classification: PERFORMANCE.

## D014 — Balancing levels on top of a flat jztree hierarchy (plan §9.4, M9.1)

`jztree` derives its tree depth from `TreeConfig.stop_coarsen=1024` and
`coarse_fac=6`, so below ~6e4 particles it returns a *flat* hierarchy: at
N=5e4 the adapted tree had 2168 leaves and 0 internal nodes other than the
virtual root, i.e. the root's children were every leaf. The dual-tree walk then
degenerates to leaf-by-leaf comparison (2 305 879 node-pair visits,
36 node-pair visits per work item). `add_balancing_levels` groups the returned
nodes into 4-ary parents until the top fan-out is <= 4; leaves, particle
ranges and the pair set are unchanged (new boxes are conservative unions), and
`check_tree_invariants` still passes. After: 216 598 visits at N=5e4 (10.6x
fewer), wall forward 1.26 s -> 0.18 s; at N=1e5 0.455 s -> 0.31 s.
Classification: PERFORMANCE.
