"""Optional external cross-check against pyfcfc's soft-histogram adjoint.

pyfcfc (https://github.com/dforero0896/pyfcfc, branch ``pyfcfc-0.2.0``) is an
independent implementation of soft pair counting and its reverse-mode
derivative: FCFC's tree-based pair counter plus a C adjoint.  It is not on
PyPI, so this test skips unless it is importable (see ``docs/examples.md`` for
the build recipe, including the ``PYFCFC_NO_LTO=1`` workaround and the
``.pyfcfc_site`` prefix this cluster uses).

Conventions: jzradius counts each unordered pair once
(``pair_convention="unique"``), pyfcfc counts ordered pairs, and pyfcfc's
Gaussian-style soft kernel has ``exp(-(s - c)^2 / (2 sigma^2))`` where jzradius
uses ``exp(-(r - c)^2 / T^2)``; hence ``sigma = T / sqrt(2)`` and the
comparison is ``2 * jzradius == pyfcfc``.  Positions are kept away from the
periodic seams and all separations stay below ``r_max`` so both codes are in
their smooth regime.

Marked ``slow``: it needs the optional dependency and is not part of the
default CPU tier.
"""

import numpy as np
import pytest

pytest.importorskip("pyfcfc", reason="optional external cross-check")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from pyfcfc.boxes import py_compute_cf, soft_hist_vjp  # noqa: E402

from jzradius import radius_softmax_histogram  # noqa: E402

pytestmark = pytest.mark.slow

BOX = 1.0
R_MAX = 0.16
TEMPERATURE = 0.01
EDGES = np.linspace(0.0, R_MAX, 9)
CENTERS = 0.5 * (EDGES[1:] + EDGES[:-1])
SIGMA = TEMPERATURE / np.sqrt(2.0)


def _catalog(n=220, seed=7, spread=0.006):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.3, 0.7, (18, 3))
    which = rng.integers(0, 18, n)
    return np.mod(centres[which] + rng.normal(scale=spread, size=(n, 3)), 1.0)


def _ours(x):
    return radius_softmax_histogram(
        x,
        bin_centers=CENTERS,
        temperature=TEMPERATURE,
        boxsize=BOX,
        r_max=R_MAX,
        self_query=True,
        pair_convention="unique",
    )


def test_counts_match_pyfcfc():
    pos = _catalog()
    weights = np.ones(len(pos))
    res = py_compute_cf([pos], [weights], EDGES, None, 1, label=["A"], bin=0,
                        pair=["AA"], soft_hist=True, soft_sigma=SIGMA,
                        box=BOX)
    theirs = res["pairs"]["AA"] * res["normalization"]["AA"]
    ours = 2.0 * np.asarray(_ours(jnp.asarray(pos)))
    rel = np.abs(ours - theirs).max() / np.abs(theirs).max()
    assert rel < 1e-12, rel


def test_coordinate_gradients_match_pyfcfc():
    pos = _catalog()
    weights = np.ones(len(pos))
    rng = np.random.default_rng(11)
    cotangent = rng.normal(size=CENTERS.size)

    out = soft_hist_vjp([pos], [weights], EDGES, None, 1, pair="AA",
                        cotangent=cotangent, soft_sigma=SIGMA, bin=0,
                        box=BOX, grad_wts=False)
    theirs = np.asarray(out["grads"]["A"])

    def loss(x):
        return jnp.dot(cotangent, _ours(x))

    ours = 2.0 * np.asarray(jax.grad(loss)(jnp.asarray(pos)))
    scale = max(np.abs(theirs).max(), np.abs(ours).max())
    rel = np.abs(ours - theirs).max() / scale
    assert rel < 1e-11, rel
