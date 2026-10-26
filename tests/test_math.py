"""Math contract tests (PLAN.md 6.8 / 18.2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jzradius.config import validate_bin_centers, validate_temperature
from jzradius.errors import RadiusConfigError
from jzradius.math import (

    dW_dr,
    gaussian_weights,
    histogram_vjp_r,
    logits,
    logits_derivative,
    softmax_vjp_r,
    softmax_weights,
)

pytestmark = pytest.mark.unit


@pytest.mark.smoke
def test_softmax_normalization_spans_r_and_temperature():
    centers = jnp.linspace(0.0, 5.0, 11)
    r = jnp.linspace(0.0, 6.0, 257)
    for temperature in (0.01, 0.05, 0.5, 2.0):
        w = softmax_weights(r, centers, temperature)
        np.testing.assert_allclose(np.asarray(jnp.sum(w, axis=-1)), 1.0, atol=1e-12)
        assert np.all(np.isfinite(np.asarray(w)))


def test_dW_dr_against_central_difference():
    centers = jnp.array([0.5, 1.5, 2.5, 3.5])
    r0 = 1.234
    temperature = 0.4

    def w_of_r(r):
        return softmax_weights(r, centers, temperature)

    analytic = dW_dr(r0, centers, temperature)
    h = 1e-6
    numeric = (w_of_r(r0 + h) - w_of_r(r0 - h)) / (2.0 * h)
    np.testing.assert_allclose(
        np.asarray(analytic), np.asarray(numeric), rtol=1e-8, atol=1e-10
    )


def test_softmax_vjp_matches_autodiff():
    centers = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    r0 = 2.2
    temperature = 0.3
    g = jnp.array([0.3, -0.7, 1.1, 0.0, 0.5])

    def f(r):
        return jnp.sum(g * softmax_weights(r, centers, temperature))

    expected = jax.grad(f)(r0)
    got = softmax_vjp_r(r0, centers, temperature, g)
    np.testing.assert_allclose(float(expected), float(got), rtol=1e-12, atol=1e-12)


def test_histogram_vjp_scales_linearly_with_pair_weight():
    centers = jnp.array([1.0, 2.0, 3.0])
    r0, temperature = 1.7, 0.2
    g = jnp.array([1.0, 2.0, -0.5])
    base = histogram_vjp_r(r0, centers, temperature, g)
    scaled = histogram_vjp_r(r0, centers, temperature, g, pair_weight=3.0)
    np.testing.assert_allclose(float(scaled), 3.0 * float(base), rtol=1e-12)


def test_gaussian_vjp_matches_autodiff():
    centers = jnp.array([0.0, 2.0])
    r0, temperature = 2.5, 0.5
    g = jnp.array([0.4, -1.3])

    def f(r):
        return jnp.sum(g * gaussian_weights(r, centers, temperature))

    expected = jax.grad(f)(r0)
    got = histogram_vjp_r(r0, centers, temperature, g, mode="gaussian")
    np.testing.assert_allclose(float(expected), float(got), rtol=1e-12, atol=1e-12)


def test_single_bin_softmax_has_zero_gradient_but_gaussian_does_not():
    centers = jnp.array([2.0])
    temperature = 0.5
    g = jnp.array([1.0])

    def f_softmax(r):
        return jnp.sum(g * softmax_weights(r, centers, temperature))

    def f_gaussian(r):
        return jnp.sum(g * gaussian_weights(r, centers, temperature))

    assert abs(float(jax.grad(f_softmax)(2.3))) < 1e-14
    assert abs(float(jax.grad(f_gaussian)(2.3))) > 1e-3


def test_softmax_is_stable_for_tiny_temperature_and_separated_centers():
    centers = jnp.array([0.0, 100.0, 200.0])
    r = jnp.array([0.5, 99.9, 150.0])
    w = softmax_weights(r, centers, temperature=1e-3)
    w = np.asarray(w)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-13)
    # Far-away bins underflow cleanly to exactly zero; r=150 is equidistant
    # from centers 100 and 200, hence the 50/50 split.
    np.testing.assert_allclose(w[0], [1.0, 0.0, 0.0], atol=0.0)
    np.testing.assert_allclose(w[1], [0.0, 1.0, 0.0], atol=0.0)
    np.testing.assert_allclose(w[2], [0.0, 0.5, 0.5], atol=0.0)


def test_logits_and_derivative_formula():
    centers = jnp.array([1.0, 3.0])
    r, temperature = 2.0, 0.5
    alpha = 1.0 / temperature**2
    q = logits(r, centers, temperature)
    qp = logits_derivative(r, centers, temperature)
    np.testing.assert_allclose(np.asarray(q), alpha * -np.array([1.0, 1.0]))
    np.testing.assert_allclose(np.asarray(qp), -2 * alpha * np.array([1.0, -1.0]))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), 1e-9])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(RadiusConfigError):
        validate_temperature(temperature)


def test_bin_centers_validation():
    validate_bin_centers([0.0, 1.0, 2.0])
    for bad in ([], [1.0, 1.0], [2.0, 1.0], [0.0, float("nan")], [[0.0, 1.0]]):
        with pytest.raises(RadiusConfigError):
            validate_bin_centers(bad)
