# Examples

Runnable scripts live in `examples/`. They mirror the demonstrations in
pyfcfc's `examples/example_soft_hist_jax.py` and
`examples/example_soft_hist_fit.py`, the closest external counterpart to this
library.  The whole script takes ~3 minutes on the CPU mirror here (two Adam
fits of 250 steps each); on a GPU the same script is much faster because the
per-step cost is dominated by the host-side tree build (D012).

Run anything from the repository root, after building the extension once:

```bash
python scripts/build_ext.py            # CPU mirror; adds CUDA kernels if nvcc exists
python examples/example_soft_hist_jax.py
```

On this cluster the login node needs the `python -S` + explicit `PYTHONPATH`
shim (see `docs/testing.md`, deviation D003); the script itself is plain
Python.

## `example_soft_hist_jax.py`

The library call *is* the differentiable primitive: `radius_softmax_histogram`
is wrapped in `jax.custom_vjp` internally, so `jax.grad`/`jax.value_and_grad`
work on it directly — no user-written custom primitive is required (unlike
pyfcfc, where the example defines one around `py_compute_cf` +
`soft_hist_vjp`).

Four parts:

1. **Adjoint validation.** A random linear functional of the soft histogram,
   differentiated with `jax.grad`, compared against central finite differences
   of the same forward pass (all separations kept below `r_max` so the forward
   is smooth). Measured: `max |grad - FD| / |FD| = 2.1e-06`, i.e. the finite
   difference noise floor with `eps = 1e-6`.

2. **Differentiable model fitting (AP-like test).** 700-particle clustered
   catalogue, anisotropically stretched about the box centre,
   `x = c + (p - c) * (a, a, b)`; `(a, b)` recovered with Adam on
   `0.5 * |hist(theta) - hist(theta_true)|^2`, starting at `[1.06, 0.94]` for a
   true value of `[1, 1]`. Measured on CPU: `a = 1.00000, b = 1.00000` after
   250 Adam steps (102 s), loss `3.7e6 -> 2e-4`.  The fit is exact because the
   forward model is deterministic (same catalogue, same tree), unlike a
   mock-based estimate.

3. **Cross-histogram displacement recovery** (pyfcfc's second example). A
   "randoms" catalogue is rigidly shifted, `x -> x + t`, against a fixed
   "data" catalogue; the 3-vector `t` is recovered from the cross-correlation
   soft histogram by Adam on the same quadratic loss. Gradient check at the
   starting point: `max |jax.grad - FD| / |FD| = 1.6e-07` (bar chart panel).
   Measured on CPU: `t = (1.1999, -0.7999, 0.5996)` vs true
   `(1.2, -0.8, 0.6)` after 250 steps (65 s), loss `1.6e6 -> 0.08`.

4. **External cross-check against pyfcfc** (skipped when not importable). Same
   catalogue, same bins, same kernel width (`sigma = T / sqrt(2)`, since
   pyfcfc's `exp(-(s-c)^2 / (2 sigma^2))` equals this library's
   `exp(-(r-c)^2 / T^2)`), `soft_sigma` mapped accordingly. Conventions differ
   by a factor two — jzradius counts each unordered pair once
   (`pair_convention="unique"`), pyfcfc counts ordered pairs — so the
   comparison is `2 * jzradius == pyfcfc`. Once pyfcfc is installed the example
   prints the residuals. The numbers below are from the dedicated regression
   test `tests/test_external_pyfcfc.py`, which uses a cheaper configuration
   (N = 220 clustered, 8 bins, `r_max` = 0.16, `T` = 0.01) so that it runs in a
   few seconds:

   | quantity | result |
   |---|---|
   | counts vs pyfcfc | `5.1e-15` relative, max abs |
   | counts vs brute-force oracle | `2.5e-12` |
   | coordinate gradients vs `soft_hist_vjp` | `2.3e-13` relative, max abs |

   This is an independent implementation of the same soft-histogram weight and
   its adjoint (pyfcfc: tree-based FCFC + C adjoint; jzradius: Morton-tree
   traversal + CUDA VJP kernel), so it validates both the forward counts and
   the reverse-mode derivative, not just internal self-consistency.

### Why the histograms are smooth (and what it costs)

An earlier version of this example used `boxsize = 1`, `r_max = 0.16`,
`T = 0.01` and tight clusters (spread 0.004). That produced visibly spiky
histograms, but the spikiness was *not* Monte-Carlo noise (the forward pass is
deterministic): with clusters much smaller than the bin width, nearly all
pairwise distances pile up on a handful of discrete inter-cluster separations,
and `T` far below the bin width makes the "soft" kernel effectively a hard
selection, so each bin is a step function. The demo now uses pyfcfc-like
scales — `boxsize = 100`, separations up to `s_max = 40` in 20 bins, `T = 0.5`
(comparable to a bin) and clusters with `spread = 2` — which gives ~3e4 counts
per bin and smooth curves.

Keeping the catalogue inside the central 20% of the box, with `r_max` larger
than every pair separation, is a correctness requirement for the demos: a pair
crossing the hard `r_max` cutoff puts a jump in the loss, which corrupts
finite-difference checks of the gradient (we saw FD and autodiff disagree by a
sign before noticing this).

The price is pair count: the smooth configuration has ~2.4e5 pairs versus a few
thousand in the spiky one, so a forward+backward step costs ~0.3-0.5 s on the
CPU mirror instead of ~0.05 s. The kernels themselves are happy with this (the
CUDA path is used at far larger N in `docs/benchmarks.md`); the per-step cost
here is dominated by the host-side tree build, which is the D012 gap.

### Building pyfcfc for the cross-check

`pyfcfc` is not on PyPI; build the branch that has the adjoint:

```bash
git clone --depth 1 -b pyfcfc-0.2.0 https://github.com/dforero0896/pyfcfc.git .thirdparty/pyfcfc
PYFCFC_NO_LTO=1 pip install --no-build-isolation --no-deps \
    --target .pyfcfc_site .thirdparty/pyfcfc
# then add $PWD/.pyfcfc_site to PYTHONPATH when running the example
```

It needs the system GSL headers (`/usr/include/gsl`, present on this cluster)
and Cython at build time. When `pyfcfc` cannot be imported the example prints a
one-line skip and exits successfully, so the script stays runnable everywhere.

## Not covered here

* GPU timings and scale-tier runs: `docs/benchmarks.md`.
* Test tiers, tolerances and the CPU/GPU recipes: `docs/testing.md`.
