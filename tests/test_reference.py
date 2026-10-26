"""Reference histogram tests (PLAN.md 18.2 test_reference)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jzradius.errors import RadiusConfigError
from jzradius.math import soft_weights
from jzradius.reference import (

    reference_histogram,
    reference_pair_set,
    reference_pairs,
)

pytestmark = pytest.mark.unit


def test_single_cross_pair_histogram():
    pos_q = jnp.array([[0.0, 0.0, 0.0]])
    pos_s = jnp.array([[3.0, 0.0, 0.0]])
    centers = jnp.array([1.0, 2.0, 3.0, 4.0])
    temperature = 0.5
    hist = reference_histogram(
        pos_q, pos_s, centers, temperature, boxsize=100.0, r_max=4.0
    )
    expected = soft_weights(jnp.array(3.0), centers, temperature)
    np.testing.assert_allclose(np.asarray(hist), np.asarray(expected), rtol=1e-14)


def test_cross_counts_every_directed_pair_including_coincident():
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    centers = jnp.array([0.5, 1.0])
    hist = reference_histogram(
        pos, pos, centers, 0.5, boxsize=100.0, r_max=2.0
    )
    # Two coincident pairs at r=0 plus (0,1) and (1,0) at r=1.
    assert reference_pairs(pos, pos, 100.0, 2.0)[0].shape == (4,)
    expected = (
        2.0 * soft_weights(jnp.array(0.0), centers, 0.5)
        + 2.0 * soft_weights(jnp.array(1.0), centers, 0.5)
    )
    np.testing.assert_allclose(np.asarray(hist), np.asarray(expected), rtol=1e-14)


def test_auto_unique_counts_unordered_pairs_once():
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    centers = jnp.array([0.5, 1.0, 2.0])
    r_max = 3.0
    i, j, r = reference_pairs(pos, pos, 100.0, r_max, self_query=True)
    pairs = set(zip(np.asarray(i).tolist(), np.asarray(j).tolist()))
    assert pairs == {(0, 1)}  # (1,2) is r=4 > 3, (0,2) is r=5 > 3
    hist = reference_histogram(
        pos, pos, centers, 0.5, boxsize=100.0, r_max=r_max, self_query=True
    )
    expected = soft_weights(jnp.array(1.0), centers, 0.5)
    np.testing.assert_allclose(np.asarray(hist), np.asarray(expected), rtol=1e-14)


def test_directed_half_matches_unique():
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    centers = jnp.array([0.5, 1.0, 1.5])
    kwargs = dict(
        boxsize=100.0, r_max=3.0, self_query=True, mode="softmax"
    )
    unique = reference_histogram(
        pos, pos, centers, 0.5, pair_convention="unique", **kwargs
    )
    directed = reference_histogram(
        pos, pos, centers, 0.5, pair_convention="directed_half", **kwargs
    )
    np.testing.assert_allclose(np.asarray(unique), np.asarray(directed), rtol=1e-14)


def test_weight_combinations():
    pos_q = jnp.array([[0.0, 0.0, 0.0]])
    pos_s = jnp.array([[2.0, 0.0, 0.0]])
    centers = jnp.array([1.0, 2.0, 3.0])
    base = reference_histogram(
        pos_q, pos_s, centers, 0.5, boxsize=100.0, r_max=3.0
    )
    wq = jnp.array([2.0])
    ws = jnp.array([3.0])
    both = reference_histogram(
        pos_q, pos_s, centers, 0.5,
        boxsize=100.0, r_max=3.0,
        weights_query=wq, weights_source=ws,
    )
    only_q = reference_histogram(
        pos_q, pos_s, centers, 0.5,
        boxsize=100.0, r_max=3.0, weights_query=wq,
    )
    only_s = reference_histogram(
        pos_q, pos_s, centers, 0.5,
        boxsize=100.0, r_max=3.0, weights_source=ws,
    )
    np.testing.assert_allclose(np.asarray(both), 6.0 * np.asarray(base), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(only_q), 2.0 * np.asarray(base), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(only_s), 3.0 * np.asarray(base), rtol=1e-14)


def test_peak_bin_is_the_nearest_center():
    pos_q = jnp.array([[0.0, 0.0, 0.0]])
    pos_s = jnp.array([[2.02, 0.0, 0.0]])
    centers = jnp.array([1.0, 2.0, 3.0])
    hist = np.asarray(
        reference_histogram(pos_q, pos_s, centers, 0.2, boxsize=100.0, r_max=4.0)
    )
    assert int(np.argmax(hist)) == 1


def test_periodic_pair_uses_minimum_image():
    boxsize = 10.0
    pos_q = jnp.array([[0.2, 0.0, 0.0]])
    pos_s = jnp.array([[9.4, 0.0, 0.0]])  # MIC distance 0.8
    i, j, r = reference_pairs(pos_q, pos_s, boxsize, 2.0)
    assert r.shape == (1,)
    assert float(r[0]) == pytest.approx(0.8, abs=1e-12)


def test_gaussian_mode_is_unnormalized():
    pos_q = jnp.array([[0.0, 0.0, 0.0]])
    pos_s = jnp.array([[2.0, 0.0, 0.0]])
    centers = jnp.array([1.0, 2.0, 3.0])
    hist = np.asarray(
        reference_histogram(
            pos_q, pos_s, centers, 0.5,
            boxsize=100.0, r_max=4.0, mode="gaussian",
        )
    )
    expected = np.exp(
        -((2.0 - np.asarray(centers)) / 0.5) ** 2
    )
    np.testing.assert_allclose(hist, expected, rtol=1e-14)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    pos_q = jnp.asarray(rng.uniform(0, 10, size=(5, 3)))
    pos_s = jnp.asarray(rng.uniform(0, 10, size=(6, 3)))
    centers = jnp.array([1.0, 2.0, 3.0])
    temperature, boxsize, r_max = 0.5, 10.0, 4.0

    def loss(p):
        return jnp.sum(
            reference_histogram(
                p, pos_s, centers, temperature, boxsize, r_max
            )
        )

    grad = jax.grad(loss)(pos_q)
    h = 1e-6
    for idx in [(0, 0), (2, 1), (4, 2)]:
        plus = pos_q.at[idx].add(h)
        minus = pos_q.at[idx].add(-h)
        numeric = (loss(plus) - loss(minus)) / (2 * h)
        assert float(grad[idx]) == pytest.approx(float(numeric), rel=1e-6, abs=1e-8)


def test_reference_pair_set_is_sorted_and_stable():
    rng = np.random.default_rng(2)
    pos = jnp.asarray(rng.uniform(0, 10, size=(32, 3)))
    a = reference_pair_set(pos, pos, 10.0, 2.5, self_query=True)
    b = reference_pair_set(pos, pos, 10.0, 2.5, self_query=True)
    assert a == b
    assert all((i < j) for i, j, _ in a)
    assert a == sorted(a)


def test_reference_refuses_invalid_shapes():
    with pytest.raises(RadiusConfigError):
        reference_histogram(
            jnp.zeros((3, 2)), jnp.zeros((3, 3)), jnp.array([1.0]), 0.5, 10.0
        )
    with pytest.raises(RadiusConfigError):
        reference_histogram(
            jnp.zeros((3, 3)), jnp.zeros((4, 3)), jnp.array([1.0]), 0.5, 10.0,
            self_query=True,
        )
    with pytest.raises(RadiusConfigError):
        reference_histogram(
            jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.array([1.0]), 0.5, 10.0,
            r_max=5.0,
        )
