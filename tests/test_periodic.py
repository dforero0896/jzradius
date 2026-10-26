"""Periodic minimum-image tests (PLAN.md 18.2 test_periodic)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jzradius.math import periodic_delta, periodic_distance, r_epsilon
from tests.conftest import DISTRIBUTIONS

pytestmark = pytest.mark.unit



def _numpy_mic_delta(delta, boxsize):
    return delta - boxsize * np.round(delta / boxsize)


@pytest.mark.smoke
def test_wraparound_displacements():
    boxsize = 10.0
    cases = {
        9.0: -1.0,
        -9.0: 1.0,
        11.0: 1.0,
        -11.0: -1.0,
        4.0: 4.0,
        -4.0: -4.0,
    }
    for delta, expected in cases.items():
        got = float(periodic_delta(jnp.array(delta), boxsize))
        assert got == pytest.approx(expected, abs=1e-12)


def test_half_box_tie_is_consistent_and_distance_is_half_box():
    boxsize = 10.0
    for delta in (5.0, -5.0):
        d = float(periodic_delta(jnp.array(delta), boxsize))
        assert abs(d) == pytest.approx(5.0)
        r = float(periodic_distance(jnp.array([delta, 0.0, 0.0]), boxsize))
        assert r == pytest.approx(5.0, abs=1e-12)


def test_zero_separation_uses_epsilon():
    r = float(periodic_distance(jnp.zeros(3), 10.0))
    assert r > 0.0
    assert r == pytest.approx(np.sqrt(r_epsilon(jnp.float64)), rel=1e-12)


def test_gradient_through_minimum_image():
    boxsize = 10.0
    delta = jnp.array([1.3, 2.7, -0.4])
    # Stay away from the round() kinks at delta = L/2.
    d_mic = periodic_delta(delta, boxsize)
    r = periodic_distance(delta, boxsize)
    expected = np.asarray(d_mic) / float(r)
    got = np.asarray(jax.grad(lambda x: periodic_distance(x, boxsize))(delta))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("boxsize", [1.0, 10.0, 1234.5])
def test_matches_numpy_mic(boxsize):
    rng = np.random.default_rng(0)
    delta = rng.uniform(-3 * boxsize, 3 * boxsize, size=(500, 3))
    got = np.asarray(periodic_delta(jnp.asarray(delta), boxsize))
    expected = _numpy_mic_delta(delta, boxsize)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", sorted(DISTRIBUTIONS))
def test_pairwise_matrix_matches_scalar_loop(name):
    boxsize = 10.0
    pos = DISTRIBUTIONS[name](17, boxsize, seed=3)
    from jzradius.math import periodic_distance_matrix

    mat = np.asarray(periodic_distance_matrix(pos, pos, boxsize))
    assert mat.shape == (17, 17)
    for i in range(17):
        for j in range(17):
            expected = float(periodic_distance(pos[i] - pos[j], boxsize))
            assert mat[i, j] == pytest.approx(expected, rel=1e-12, abs=1e-12)
