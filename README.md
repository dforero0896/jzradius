# jzradius

Differentiable periodic two-point pair counting on GPU via Morton-tree radius
traversal, built on [`jztree`](https://github.com/jstuecker/jztree) and JAX.

`jzradius` computes a soft histogram of pairwise minimum-image distances below
`r_max`, differentiable with respect to particle positions (and optional
weights), using a dual-tree traversal instead of an O(N²) pair list.

## Status

Pre-alpha. The full stack exists: periodic bounds, tree adapter, exact
dual-tree traversal, CUDA forward/backward kernels, typed XLA-FFI host
launchers, build integration and the `custom_vjp` wrapper. The kernels run on
device: the complete suite (`235 passed`) passes on one A30, including forward
parity with the brute-force oracle and position/weight gradients matched
against `jax.grad(reference_histogram)` for self and cross correlation. Known
gaps are performance (the tree build is host-side, D012) and the eager-only
API (`jax.jit` around the public call raises, D008) — see `STATUS.md` and
`docs/limitations.md`. Forward counts and gradients are also cross-checked
against an independent soft-histogram implementation and adjoint (pyfcfc,
`5e-15` / `2e-13` relative), see `docs/examples.md`.

## Install

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e .
# GPU stack must match jztree's backend; see docs/environment_lock.md
```

## Minimal usage

```python
import jax
import jax.numpy as jnp
import jzradius

jax.config.update("jax_enable_x64", True)
pos = jax.random.uniform(jax.random.PRNGKey(0), (256, 3))
centers = jnp.linspace(0.01, 0.2, 20)

def loss(p):
    return jnp.sum(jzradius.radius_softmax_histogram(
        p, bin_centers=centers, temperature=0.05, boxsize=1.0,
        r_max=0.22, self_query=True, pair_convention="unique",
    ))

values, grads = jax.value_and_grad(loss)(pos)
```

The call is eager: build the FFI library once with
`python scripts/build_ext.py` (add `module load cuda/12.3` on the login node for
the CUDA artifact), then import `jzradius`. `jit` around the public function is
not supported in v1 (D008); the gradient itself is the analytic kernel VJP.
`jzradius.reference_histogram` remains available as the O(N²) oracle.

## Tests

```bash
python scripts/build_ext.py            # CPU mirror (+ CUDA when nvcc exists)
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/ -q   # 237 passed, 2 skipped
# on a GPU node (both platforms registered: some tests use jax.io_callback)
JAX_PLATFORMS=cuda,cpu .venv/bin/python -m pytest tests/ -q   # 235 passed
```

Tiered runners (`scripts/run_smoke.py`, `run_unit.py`, `run_integration.py`)
enforce the PLAN §18 caps of 60 s / 300 s / 900 s.

Tolerances, seeds and distributions follow `PLAN.md` §18.

## Examples

`examples/example_soft_hist_jax.py` — adjoint vs finite differences, a
differentiable anisotropic-stretch fit with Adam, a cross-histogram rigid-shift
recovery, and the optional pyfcfc cross-check. Like pyfcfc's examples it writes
2x2 summary figures to `figures/`:

| figure | what it shows |
|---|---|
| `figures/soft_hist_jax.png` | adjoint vs FD scatter, Adam loss, `(a, b)` trajectory, fitted vs target soft histogram |
| `figures/soft_hist_fit.png` | gradient bar chart (FD vs adjoint), loss, `t` trajectory, cross histograms |

Measured (CPU mirror, `boxsize = 100`, `s_max = 40`, 20 bins, `T = 0.5`): FD
check `2.1e-06`; stretch `[1.06, 0.94] → (1.00000, 1.00000)` in 250 steps;
shift `(1.2, -0.8, 0.6) → (1.1999, -0.7999, 0.5996)` in 250 steps. Numbers and
the smoothness discussion: `docs/examples.md`.

Measured A30 timings (float64, r_max = 0.05, 20 bins) are in
`docs/benchmarks.md`: forward 0.76 s at 1e4, 3.05 s at 1e5, 18.2 s at 3e5
particles; the backward pass costs about the same as the forward because the
host-side tree build dominates (D012).

## Layout

```
jzradius/          API, custom_vjp, FFI bindings, traversal, adapter, bounds
src/cuda/          device kernels + shared arithmetic header
src/cpu/           CPU mirror of the same arithmetic (test harness)
scripts/           build_ext, check_env, check_budget, tier runners
docs/              lock, audit, algorithm, ffi notes, testing, limitations
tests/             unit/smoke/integration tiers
```

See `PLAN.md` for the normative specification and `docs/limitations.md` for
known limitations.
