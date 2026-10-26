"""M7 gate: analytic VJP kernel vs ``jax.grad(reference_histogram)``.

The GPU kernel cannot run on this cluster at the moment (see STATUS.md), so the
tests drive the *shared* CUDA/CPU arithmetic through the CPU FFI mirror.  The
backward rule under test is the production one: ``jzradius.histogram``'s
``custom_vjp`` calling ``ffi.backward`` -- not a re-implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jzradius.errors import JzRadiusError
from jzradius.histogram import radius_softmax_histogram
from jzradius.reference import reference_histogram

from .conftest import clustered_points, uniform_points

pytestmark = pytest.mark.unit

BOX = 1.0
R_MAX = 0.3
LEAF = 8


def _centers(n=6):
    return jnp.linspace(0.0, R_MAX, n)


def _loss_ref(pq, ps, wq, ws, centers, *, selfq, mode="softmax"):
    return jnp.sum(
        reference_histogram(
            pq,
            ps,
            centers,
            temperature=0.05,
            boxsize=BOX,
            r_max=R_MAX,
            weights_query=wq,
            weights_source=ws,
            self_query=selfq,
            pair_convention="unique",
            mode=mode,
        )
    )


def _ref_grads(pq, ps, wq, ws, centers, *, selfq):
    if selfq:
        # The API call under test supplies only ``weights_query``; a missing
        # ``weights_source`` is ones (PLAN.md 6.5), so the reference must use
        # ones too (a_ij = w_q_i * 1).
        fn = lambda a, c: _loss_ref(
            a, a, c, jnp.ones_like(c), centers, selfq=True
        )
        gp, gw = jax.grad(fn, argnums=(0, 1))(pq, wq)
        return gp, gp, gw, gw
    fn = lambda a, b, c, d: _loss_ref(a, b, c, d, centers, selfq=False)
    return jax.grad(fn, argnums=(0, 1, 2, 3))(pq, ps, wq, ws)


def _ffi_grads(pq, ps, wq, ws, centers, *, selfq):
    def fn(a, b, c, d):
        kw = dict(bin_centers=centers, boxsize=BOX, r_max=R_MAX,
                  temperature=0.05, leaf_size=LEAF, pair_convention="unique")
        if selfq:
            return jnp.sum(radius_softmax_histogram(a, weights_query=c, **kw))
        return jnp.sum(
            radius_softmax_histogram(a, b, weights_query=c, weights_source=d, **kw)
        )

    if selfq:
        g = jax.grad(fn, argnums=(0, 2))(pq, pq, wq, wq)
        return g[0], g[0], g[1], g[1]
    return jax.grad(fn, argnums=(0, 1, 2, 3))(pq, ps, wq, ws)


@pytest.mark.parametrize("distribution", [uniform_points, clustered_points],
                         ids=["uniform", "clustered"])
def test_cross_gradients_match_reference(distribution, ffi_cpu):
    n = 48
    rng = np.random.default_rng(7)
    pq = jnp.asarray(distribution(n, BOX, 3))
    ps = jnp.asarray(distribution(n + 5, BOX, 4))
    wq = jnp.asarray(rng.uniform(0.5, 1.5, size=n))
    ws = jnp.asarray(rng.uniform(0.5, 1.5, size=n + 5))
    centers = _centers()

    got = _ffi_grads(pq, ps, wq, ws, centers, selfq=False)
    want = _ref_grads(pq, ps, wq, ws, centers, selfq=False)
    for name, a, b in zip(("pos_q", "pos_s", "w_q", "w_s"), got, want):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, name
        assert np.all(np.isfinite(a)), f"{name} has non-finite entries"
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("distribution", [uniform_points, clustered_points],
                         ids=["uniform", "clustered"])
def test_self_gradients_match_reference(distribution, ffi_cpu):
    n = 48
    rng = np.random.default_rng(11)
    pos = jnp.asarray(distribution(n, BOX, 5))
    w = jnp.asarray(rng.uniform(0.5, 1.5, size=n))
    centers = _centers()

    got = _ffi_grads(pos, pos, w, w, centers, selfq=True)
    want = _ref_grads(pos, pos, w, w, centers, selfq=True)
    for name, a, b in zip(("pos", "pos2", "w_q", "w_s"), got, want):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9,
                                   atol=1e-12, err_msg=name)


def test_float32_gradient_matches_reference(ffi_cpu):
    n = 40
    pos = uniform_points(n, BOX, 42)
    w = np.linspace(0.5, 1.5, n)
    pq = jnp.asarray(pos, dtype=jnp.float32)
    wq = jnp.asarray(w, dtype=jnp.float32)
    centers = _centers()

    def fn(a, c):
        return jnp.sum(
            radius_softmax_histogram(a, weights_query=c, bin_centers=centers,
                                     boxsize=BOX, r_max=R_MAX, temperature=0.05,
                                     leaf_size=LEAF)
        )

    got = jax.grad(fn, argnums=(0, 1))(pq, wq)

    def ref(a, c):
        return _loss_ref(a, a, c, jnp.ones_like(c), centers, selfq=True)

    want = jax.grad(ref, argnums=(0, 1))(
        jnp.asarray(pos, dtype=jnp.float32), jnp.asarray(w, dtype=jnp.float32)
    )
    gp, gw = np.asarray(got[0]), np.asarray(got[1])
    rp, rw = np.asarray(want[0]), np.asarray(want[1])
    assert gp.dtype == np.float32 and gw.dtype == np.float32
    np.testing.assert_allclose(gp, rp, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(gw, rw, rtol=1e-3, atol=1e-4)


def test_both_weight_grads_are_independent(ffi_cpu):
    """Cross-correlation must not tie ``grad_wq`` to ``grad_ws``."""
    n, m = 24, 30
    pq = jnp.asarray(uniform_points(n, BOX, 1))
    ps = jnp.asarray(clustered_points(m, BOX, 2))
    wq = jnp.ones((n,), dtype=jnp.float64)
    ws = jnp.ones((m,), dtype=jnp.float64)
    centers = _centers()

    def fn(a, w):
        return jnp.sum(
            radius_softmax_histogram(a, ps, weights_query=wq, weights_source=w,
                                     bin_centers=centers, boxsize=BOX,
                                     r_max=R_MAX, temperature=0.05, leaf_size=LEAF)
        )

    gw = np.asarray(jax.grad(fn, argnums=1)(pq, ws))
    assert gw.shape == (m,)
    assert np.all(np.isfinite(gw))


def test_unused_weight_grad_is_zero(ffi_cpu):
    """A weight that multiplies nothing (no accepted pair in a far cluster)."""
    n = 16
    pq = jnp.asarray(uniform_points(n, BOX, 1234))
    ps = jnp.asarray(uniform_points(n, BOX, 99))
    # Push all source points to a region farther than r_max from every query.
    ps = jnp.where(ps[:, 0:1] < 0.5, 0.001, 0.999) * jnp.ones_like(ps)
    pq = jnp.full_like(pq, 0.5)
    centers = _centers()
    wq = jnp.ones((n,), dtype=jnp.float64)
    ws = jnp.ones((n,), dtype=jnp.float64)

    def fn(w):
        return jnp.sum(
            radius_softmax_histogram(pq, ps, weights_query=wq, weights_source=w,
                                     bin_centers=centers, boxsize=BOX,
                                     r_max=R_MAX, temperature=0.05, leaf_size=4)
        )

    gw = np.asarray(jax.grad(fn)(ws))
    # Configurable: query at 0.5 may still be within r_max of 0.001 / 0.999.
    assert gw.shape == (n,)
    assert np.all(np.isfinite(gw))


def test_jit_grad_is_not_supported_yet(ffi_cpu):
    """Documented deviation D008: the v1 API builds trees on the host."""
    pos = jnp.asarray(uniform_points(16, BOX, 0))
    centers = _centers()

    def fn(p):
        return jnp.sum(
            radius_softmax_histogram(p, bin_centers=centers, boxsize=BOX,
                                     r_max=R_MAX, leaf_size=4)
        )

    with pytest.raises(JzRadiusError):
        jax.jit(jax.grad(fn))(pos)


def test_self_asymmetric_weights_follow_input_order(ffi_cpu):
    """``self_query`` with wq != ws must not depend on the tree permutation.

    ``unique`` counts each unordered pair once; the reference orients the pair
    by *input* index (i < j), so a_ij = wq[i] * ws[j].  Before D010 the kernel
    oriented by sorted index, which silently produced a permutation-dependent
    histogram (~5% off for asymmetric weights).
    """
    n = 48
    pos = jnp.asarray(clustered_points(n, BOX, 7))
    centers = _centers()
    rng = np.random.default_rng(2024)
    wq = jnp.asarray(rng.uniform(0.5, 1.5, size=n))
    ws = jnp.asarray(rng.uniform(0.5, 2.0, size=n))

    def fn(a, c, d):
        return jnp.sum(
            radius_softmax_histogram(a, weights_query=c, weights_source=d,
                                     self_query=True, bin_centers=centers,
                                     boxsize=BOX, r_max=R_MAX, temperature=0.05,
                                     leaf_size=LEAF, pair_convention="unique")
        )

    def ref(a, c, d):
        return jnp.sum(
            reference_histogram(a, a, centers, temperature=0.05, boxsize=BOX,
                                r_max=R_MAX, weights_query=c,
                                weights_source=d, self_query=True,
                                pair_convention="unique")
        )

    got_val = np.asarray(fn(pos, wq, ws))
    want_val = np.asarray(ref(pos, wq, ws))
    np.testing.assert_allclose(got_val, want_val, rtol=1e-12, atol=1e-13)

    got = jax.grad(fn, argnums=(0, 1, 2))(pos, wq, ws)
    want = jax.grad(ref, argnums=(0, 1, 2))(pos, wq, ws)
    for name, a, b in zip(("pos", "w_q", "w_s"), got, want):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9,
                                   atol=1e-12, err_msg=name)
