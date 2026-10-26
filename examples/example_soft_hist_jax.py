"""Differentiable pair counting: the jzradius soft histogram inside JAX.

This mirrors the pyfcfc demonstration
(``examples/example_soft_hist_jax.py`` in pyfcfc 0.2.0) on the same footing.

The soft histogram is smooth in the catalogue coordinates and ``jzradius``
ships its reverse-mode derivative as a CUDA/XLA-FFI kernel, wrapped in a
``jax.custom_vjp``.  So no extra primitive is needed here: the library call
*is* the differentiable pair counter.

    hist = jzradius.radius_softmax_histogram(pos, bin_centers=..., ...)
    value, grad = jax.value_and_grad(loss)(pos)

Two demonstrations are run:

1. **Adjoint validation**: the gradient of a random linear functional of the
   pair counts, obtained with ``jax.grad``, against central finite differences
   of the same forward pass.

2. **Differentiable model fitting**: an Alcock-Paczynski-like test.  The
   catalogue is anisotropically stretched, ``x = c + (p - c) * (a, a, b)``, and
   ``(a, b)`` are recovered by Adam on the mismatch between the soft histogram
   of the stretched catalogue and that of the true one.

3. **Cross-histogram displacement recovery** (mirrors pyfcfc's
   ``example_soft_hist_fit.py``): a rigid 3-vector shift of a "randoms"
   catalogue against a fixed "data" catalogue, recovered by Adam from the
   cross-correlation soft histogram, with the three loss gradients checked
   against finite differences and plotted as a bar chart.

4. If ``pyfcfc`` is importable (it is not on PyPI; build it from
   https://github.com/dforero0896/pyfcfc, branch ``pyfcfc-0.2.0``), an external
   cross-check of both the counts and the coordinate gradients is printed.
   The two codes use different conventions, so the comparison is
   ``2 * ours == pyfcfc`` (jzradius counts each unordered pair once, pyfcfc
   counts ordered pairs).

Run from the repository root:

    python examples/example_soft_hist_jax.py

Two 2x2 summary figures are modelled on pyfcfc's
``examples/example_soft_hist_jax.py`` and ``example_soft_hist_fit.py``:
adjoint vs finite differences, the Adam loss curve, the parameter trajectory
and the soft histograms along the fit.  They are saved to
``figures/soft_hist_jax.png`` and ``figures/soft_hist_fit.png``
(matplotlib is optional: without it the script still runs, it just prints the
numbers).
"""

import os
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jzradius import radius_softmax_histogram  # noqa: E402
from jzradius.reference import reference_histogram  # noqa: E402

try:                                     # plotting is optional
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR",
                          tempfile.mkdtemp(prefix="jzradius-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:                        # pragma: no cover
    plt = None

# Anisotropic stretch about a fixed centre, avoiding periodic wrapping so the
# forward pass is smooth in theta (the same trick as the pyfcfc example).
# Lengths are in pyfcfc-like units: a large s_max (~Mpc/h) with many bins, and
# a kernel width comparable to a bin, so the soft histogram is genuinely smooth
# rather than a sum of delta functions.  The catalogue lives in the central
# sub-cube so r_max covers *every* pair: a pair crossing r_max would put a jump
# in the loss and corrupt the finite-difference check.
BOX = 100.0
R_MAX = 40.0                    # < BOX / 2, enforced by the library
TEMPERATURE = 0.5
EDGES = np.linspace(0.0, R_MAX, 21)         # 20 bins, pyfcfc-style sedges
CENTERS = 0.5 * (EDGES[1:] + EDGES[:-1])    # bin centres
SIGMA = TEMPERATURE / np.sqrt(2.0)          # pyfcfc's kernel width
CENTRE = np.array([50.0, 50.0, 50.0])


def soft_hist(x, w=None):
    """Raw soft histogram, differentiable in ``x`` (and ``w`` if given)."""
    return radius_softmax_histogram(
        x,
        bin_centers=CENTERS,
        temperature=TEMPERATURE,
        boxsize=BOX,
        r_max=R_MAX,
        weights_query=w,
        self_query=True,
        pair_convention="unique",
        leaf_size=32,
    )


def make_clustered_box(n, seed=42, n_centers=10, spread=2.0):
    """Mildly clustered catalogue inside the central 20% of the box.

    ``spread`` is comparable to a bin width, so pairwise distances populate the
    bins smoothly instead of piling up on a handful of cluster separations.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.4 * BOX, 0.6 * BOX, (n_centers, 3))
    which = rng.integers(0, n_centers, n)
    pts = centres[which] + rng.normal(scale=spread, size=(n, 3))
    return np.mod(pts, BOX)


# ============================================================================
#  Demo 1: the library's VJP against finite differences
# ============================================================================

def validate_adjoint(rng, n=80):
    pos = BOX * rng.uniform(0.45, 0.55, (n, 3))  # all separations below r_max
    g = jnp.asarray(rng.normal(size=CENTERS.size))

    def loss(x):
        return jnp.dot(g, soft_hist(x))

    gx = np.asarray(jax.grad(loss)(jnp.asarray(pos)))

    fd = np.zeros_like(pos)
    eps = 1e-6
    for i in range(n):
        for c in range(3):
            xp, xm = pos.copy(), pos.copy()
            xp[i, c] += eps
            xm[i, c] -= eps
            fd[i, c] = (float(loss(jnp.asarray(xp))) -
                        float(loss(jnp.asarray(xm)))) / (2.0 * eps)
    return pos, gx, fd


# ============================================================================
#  Demo 2: gradient descent on an anisotropic stretch (AP-like test)
# ============================================================================

def stretch(p, theta):
    """Anisotropic stretch about CENTRE: JAX-differentiable."""
    a, b = theta
    return CENTRE + (p - CENTRE) * jnp.array([a, a, b])


def fit_stretch(p, theta0, theta_true, n_iter=250, lr=1e-2, decay_at=None):
    """Adam on the soft-histogram mismatch; the step is cut 10x late in the run.

    The 10x decay is not cosmetic: each step rebuilds the tree, which changes
    the floating-point summation order and leaves a loss floor of O(10) even at
    the true parameters.  A smaller step once the parameters have converged
    keeps the trajectory from jittering around that floor.
    """
    target = jax.lax.stop_gradient(
        soft_hist(stretch(jnp.asarray(p), jnp.asarray(theta_true))))

    def loss(theta):
        d = soft_hist(stretch(jnp.asarray(p), theta)) - target
        return 0.5 * jnp.dot(d, d)

    value_and_grad = jax.value_and_grad(loss)

    theta = jnp.asarray(theta0)
    m = v = jnp.zeros(2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    if decay_at is None:
        decay_at = int(0.6 * n_iter)
    t0 = time.perf_counter()
    for step in range(1, n_iter + 1):
        step_lr = lr if step <= decay_at else 0.1 * lr
        val, gr = value_and_grad(theta)
        m = b1 * m + (1.0 - b1) * gr
        v = b2 * v + (1.0 - b2) * gr * gr
        mh = m / (1.0 - b1 ** step)
        vh = v / (1.0 - b2 ** step)
        theta = theta - step_lr * mh / (jnp.sqrt(vh) + eps)
        hist.append((float(val), float(theta[0]), float(theta[1])))
        if step % 20 == 0 or step == 1:
            print(f"  step {step:3d}: loss {float(val):12.4f}   "
                  f"a = {float(theta[0]):.5f}  b = {float(theta[1]):.5f}")
    return np.array(hist), np.asarray(theta), time.perf_counter() - t0


# ============================================================================
#  Demo 3: external cross-check against pyfcfc (optional)
# ============================================================================

def cross_check_with_pyfcfc(n=220, seed=7):
    try:
        from pyfcfc.boxes import py_compute_cf, soft_hist_vjp
    except Exception as exc:                        # pragma: no cover
        print(f"  pyfcfc not importable ({exc.__class__.__name__}: {exc}); "
              f"skipping the external cross-check")
        return

    rng = np.random.default_rng(seed)
    pos = make_clustered_box(n, seed=seed, spread=0.006)
    w = np.ones(n)

    res = py_compute_cf([pos], [w], EDGES, None, 1, label=["A"], bin=0,
                        pair=["AA"], soft_hist=True, soft_sigma=SIGMA,
                        box=BOX)
    theirs = res["pairs"]["AA"] * res["normalization"]["AA"]
    ours = 2.0 * np.asarray(soft_hist(jnp.asarray(pos)))

    g = rng.normal(size=CENTERS.size)
    ref = np.asarray(reference_histogram(
        jnp.asarray(pos), jnp.asarray(pos), jnp.asarray(CENTERS),
        temperature=TEMPERATURE, boxsize=BOX, r_max=R_MAX,
        self_query=True, pair_convention="unique"))
    print(f"  counts:  max |2*jzradius - pyfcfc| / max = "
          f"{np.abs(ours - theirs).max() / theirs.max():.3e}")
    print(f"  counts:  max |jzradius - brute force| = "
          f"{np.abs(ours / 2.0 - ref).max():.3e}")

    out = soft_hist_vjp([pos], [w], EDGES, None, 1, pair="AA",
                        cotangent=g, soft_sigma=SIGMA, bin=0, box=BOX,
                        grad_wts=False)
    theirs_g = np.asarray(out["grads"]["A"])
    ours_g = 2.0 * np.asarray(jax.grad(lambda x: jnp.dot(g, soft_hist(x)))(
        jnp.asarray(pos)))
    scale = max(np.abs(theirs_g).max(), np.abs(ours_g).max())
    print(f"  gradients: max |2*d/dx - pyfcfc| / max = "
          f"{np.abs(ours_g - theirs_g).max() / scale:.3e}")


# ============================================================================
#  Figure (pyfcfc-style 2x2 summary)
# ============================================================================

def make_figure(path, gx, fd, rel, hist, theta0, theta_true, theta_fit,
                counts_true, counts_start, counts_fit):
    """Save the 2x2 summary used by pyfcfc's differentiable-histogram demo."""
    if plt is None:
        print("  matplotlib not available; skipping the figure")
        return None

    accent = plt.get_cmap("viridis")(0.3)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

    ax = axes[0, 0]
    lim = max(np.abs(gx).max(), np.abs(fd).max())
    ax.plot([-lim, lim], [-lim, lim], ":", color="0.4", label="y = x")
    ax.scatter(fd.ravel(), gx.ravel(), s=7, alpha=0.6, color=accent)
    ax.set_xlabel("finite-difference gradient")
    ax.set_ylabel("jax.grad through the custom VJP")
    ax.set_title(f"Adjoint validation (max rel err {rel.max():.1e})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.semilogy(hist[:, 0], lw=1.5, color=accent)
    ax.set_xlabel("Adam step")
    ax.set_ylabel(r"loss  $\frac{1}{2}|DD(\theta)-DD(\theta_*)|^2$")
    ax.set_title("Differentiable histogram fitting")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.plot(hist[:, 1], hist[:, 2], "-o", ms=3, lw=1.2, color=accent,
            label="gradient descent")
    ax.plot(theta0[0], theta0[1], "x", ms=10, mew=2.5, color="crimson",
            label="start")
    ax.plot(theta_true[0], theta_true[1], "*", ms=16, color="k", label="truth")
    ax.set_xlabel(r"$a$ ($s_\perp$ stretch)")
    ax.set_ylabel(r"$b$ (line-of-sight stretch)")
    ax.set_title("Parameter recovery trajectory")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.step(np.repeat(EDGES, 2)[1:-1], np.repeat(counts_true, 2), where="mid",
            color="k", lw=1.5, label=r"target  $\theta_*$")
    ax.plot(CENTERS, counts_start, "--", color="crimson", lw=1.3,
            label=f"start ({theta0[0]:.2f}, {theta0[1]:.2f})")
    ax.plot(CENTERS, counts_fit, "-o", ms=3, lw=1.2, color=accent,
            label="fitted")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel("soft DD pair counts")
    ax.set_title("Soft histograms along the fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figure saved to {path}")
    return path


# ============================================================================
#  Demo 3: cross-histogram rigid-shift recovery (pyfcfc's second example)
# ============================================================================

def soft_cross(xq, xs):
    """Cross-correlation soft histogram: all ordered (query, source) pairs."""
    return radius_softmax_histogram(
        xq,
        xs,
        bin_centers=CENTERS,
        temperature=TEMPERATURE,
        boxsize=BOX,
        r_max=R_MAX,
    )


def shifted(x, theta):
    """Rigid translation, JAX-differentiable in ``theta``."""
    return x + jnp.asarray(theta)


def make_cross_catalogues(n=400, seed=11):
    """Two clustered sub-catalogues in the central sub-cube (no wrap issues).

    Kept well inside the box so a shift of a few 1e-2 never wraps, which keeps
    the loss smooth in ``theta`` (the trick used by the pyfcfc example).
    """
    rng = np.random.default_rng(seed)
    data = make_clustered_box(n, seed=seed)
    rand = make_clustered_box(n, seed=seed + 1)
    return data, rand


def fit_shift(data, rand, theta0, theta_true, n_iter=250, lr=3e-2,
              decay_at=None):
    """Adam on ``0.5 |DR(theta) - DR(theta_*)|^2`` for the 3-vector shift."""
    target = jax.lax.stop_gradient(
        soft_cross(jnp.asarray(data), shifted(jnp.asarray(rand), theta_true)))

    def loss(theta):
        d = soft_cross(jnp.asarray(data),
                       shifted(jnp.asarray(rand), theta)) - target
        return 0.5 * jnp.dot(d, d)

    value_and_grad = jax.value_and_grad(loss)

    theta = jnp.asarray(theta0)
    m = v = jnp.zeros(3)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    if decay_at is None:
        decay_at = int(0.6 * n_iter)
    t0 = time.perf_counter()
    for step in range(1, n_iter + 1):
        val, gr = value_and_grad(theta)
        m = b1 * m + (1.0 - b1) * gr
        v = b2 * v + (1.0 - b2) * gr * gr
        mh = m / (1.0 - b1 ** step)
        vh = v / (1.0 - b2 ** step)
        step_lr = lr if step <= decay_at else 0.1 * lr
        theta = theta - step_lr * mh / (jnp.sqrt(vh) + eps)
        hist.append((float(val), *[float(t) for t in theta]))
        if step % 50 == 0 or step == 1:
            print(f"  step {step:3d}: loss {float(val):12.4f}   theta = ("
                  f"{float(theta[0]):+.5f}, {float(theta[1]):+.5f}, "
                  f"{float(theta[2]):+.5f})")
    return np.array(hist), np.asarray(theta), time.perf_counter() - t0


def make_cross_figure(path, fd, grad_chk, rel, hist, theta_true, theta0,
                      counts_true, counts_start, counts_fit):
    """pyfcfc-style 2x2 summary for the cross-histogram displacement fit."""
    if plt is None:
        print("  matplotlib not available; skipping the figure")
        return None

    comps = ["$t_x$", "$t_y$", "$t_z$"]
    cols = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, 2))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))

    ax = axes[0, 0]
    wdt = 0.38
    idx = np.arange(3)
    ax.bar(idx - wdt / 2, fd, wdt, color=cols[0], label="finite differences")
    ax.bar(idx + wdt / 2, grad_chk, wdt, color=cols[1],
           label="jzradius adjoint")
    ax.set_xticks(idx)
    ax.set_xticklabels(comps)
    ax.set_ylabel(r"$dL/d\theta_a$")
    ax.set_title(f"Gradient of the histogram-matching loss "
                 f"(max rel err {rel.max():.1e})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0, 1]
    ax.semilogy(hist[:, 0], lw=1.5, color=cols[0])
    ax.set_xlabel("Adam step")
    ax.set_ylabel(r"loss  $\frac{1}{2}|DR(\theta)-DR(\theta_*)|^2$")
    ax.set_title("Displacement recovery from cross histograms")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1, 0]
    for a in range(3):
        ax.plot(hist[:, 1 + a], lw=1.4,
                color=plt.get_cmap("viridis")(0.2 + 0.6 * a / 2),
                label=comps[a])
        ax.axhline(theta_true[a], ls=":", color="0.4")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("shift parameter")
    ax.set_title("Trajectory of the shift parameters (dotted: truth)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.step(np.repeat(EDGES, 2)[1:-1], np.repeat(counts_true, 2), where="mid",
            color="k", lw=1.5, label=r"target  $\theta_*$")
    ax.plot(CENTERS, counts_start, "--", color="crimson", lw=1.3,
            label="start")
    ax.plot(CENTERS, counts_fit, "-o", ms=3, lw=1.2, color=cols[1],
            label="fitted")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel("soft DR pair counts")
    ax.set_title("Cross histograms along the fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  figure saved to {path}")
    return path


# ============================================================================
#  Main
# ============================================================================

def main():
    rng = np.random.default_rng(7)

    print("1. validating the soft-histogram VJP against finite differences ...")
    pos, gx, fd = validate_adjoint(rng)
    rel = np.abs(gx - fd) / np.maximum(np.abs(fd), 1e-3 * np.abs(fd).max())
    print(f"  max |jax.grad - FD| / |FD| = {rel.max():.3e}")

    print("2. recovering an anisotropic stretch from theta0 = [1.06, 0.94] "
          "(true [1, 1]) ...")
    p = make_clustered_box(700, seed=42)
    theta0 = np.array([1.06, 0.94])
    theta_true = np.array([1.0, 1.0])
    hist, theta_fit, dt = fit_stretch(p, theta0, theta_true)
    print(f"  fit: a = {theta_fit[0]:.5f}, b = {theta_fit[1]:.5f} "
          f"({dt:.1f} s for {len(hist)} Adam steps)")

    print("3. recovering a rigid shift from cross histograms "
          "(true [1.2, -0.8, 0.6]) ...")
    data, rand = make_cross_catalogues()
    t0 = np.array([2.0, 0.0, -1.0])
    t_true = np.array([1.2, -0.8, 0.6])

    # gradient check of the three shift components at the starting point
    target0 = jax.lax.stop_gradient(
        soft_cross(jnp.asarray(data), shifted(jnp.asarray(rand), t_true)))

    def shift_loss(t):
        d = soft_cross(jnp.asarray(data), shifted(jnp.asarray(rand), t)
                       ) - target0
        return 0.5 * jnp.dot(d, d)

    grad_chk = np.asarray(jax.grad(shift_loss)(jnp.asarray(t0)))
    fd_shift = np.zeros(3)
    eps = 1e-6
    for a in range(3):
        tp, tm = t0.copy(), t0.copy()
        tp[a] += eps
        tm[a] -= eps
        fd_shift[a] = (float(shift_loss(jnp.asarray(tp))) -
                       float(shift_loss(jnp.asarray(tm)))) / (2.0 * eps)
    rel_shift = np.abs(grad_chk - fd_shift) / np.maximum(
        np.abs(fd_shift), 1e-8)
    print(f"  max |jax.grad - FD| / |FD| = {rel_shift.max():.3e}")

    hist_s, theta_shift, dt = fit_shift(data, rand, t0, t_true)
    print(f"  fit: theta = ({theta_shift[0]:+.4f}, {theta_shift[1]:+.4f}, "
          f"{theta_shift[2]:+.4f}) vs true ({t_true[0]:+.4f}, "
          f"{t_true[1]:+.4f}, {t_true[2]:+.4f})  "
          f"[{dt:.1f} s, {len(hist_s)} Adam steps]")

    print("4. external cross-check against pyfcfc (if installed) ...")
    cross_check_with_pyfcfc()

    print("5. writing the pyfcfc-style summary figures ...")
    counts_true = np.asarray(soft_hist(stretch(jnp.asarray(p), theta_true)))
    counts_start = np.asarray(soft_hist(stretch(jnp.asarray(p), theta0)))
    counts_fit = np.asarray(soft_hist(stretch(jnp.asarray(p), theta_fit)))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "figures", "soft_hist_jax.png")
    make_figure(os.path.normpath(out), gx, fd, rel, hist, theta0, theta_true,
                theta_fit, counts_true, counts_start, counts_fit)

    dq = jnp.asarray(data)
    counts_true_s = np.asarray(soft_cross(dq, shifted(jnp.asarray(rand), t_true)))
    counts_start_s = np.asarray(soft_cross(dq, shifted(jnp.asarray(rand), t0)))
    counts_fit_s = np.asarray(
        soft_cross(dq, shifted(jnp.asarray(rand), theta_shift)))
    out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "figures", "soft_hist_fit.png")
    make_cross_figure(os.path.normpath(out2), fd_shift, grad_chk, rel_shift,
                      hist_s,
                      t_true, t0, counts_true_s, counts_start_s, counts_fit_s)


if __name__ == "__main__":
    main()
