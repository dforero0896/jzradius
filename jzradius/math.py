"""Differentiable math for periodic radius histograms (PLAN.md section 6).

All functions here are pure JAX and side-effect free.  They operate on the
*scalar* per-pair quantities; pair enumeration lives in :mod:`jzradius.reference`
(CPU oracle) and in the traversal/kernel layers.

Shapes
------
``r``, ``cotangent`` and ``pair_weight`` broadcast against each other.
``centers`` has shape ``(B,)`` and is the last axis of the logits.  A function
that returns bin-resolved values therefore has shape ``broadcast(r.shape) + (B,)``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .errors import RadiusConfigError

_MODES = ("softmax", "gaussian")


def dtype_name(dtype) -> str:
    """Return ``"float32"`` / ``"float64"`` for a JAX/NumPy dtype."""
    name = np.dtype(dtype).name
    if name not in ("float32", "float64"):
        raise TypeError(f"only float32/float64 are supported, got {name!r}")
    return name


def r_epsilon(dtype) -> float:
    """PLAN.md 6.1: ``eps`` added under the square root."""
    return 1e-30 if dtype_name(dtype) == "float32" else 1e-60


def periodic_delta(delta, boxsize):
    """Minimum-image displacement, componentwise (PLAN.md 6.1).

    ``delta - L * round(delta / L)`` with round-to-nearest (ties-to-even).
    The rounding is treated as a constant in reverse mode.
    """
    delta = jnp.asarray(delta)
    boxsize = jnp.asarray(boxsize, dtype=delta.dtype)
    return delta - boxsize * jnp.round(delta / boxsize)


def periodic_distance(delta, boxsize):
    """MIC distance from an already-computed displacement."""
    d = periodic_delta(delta, boxsize)
    return jnp.sqrt(jnp.sum(d * d, axis=-1) + r_epsilon(d.dtype))


def periodic_distance_matrix(pos_i, pos_j, boxsize):
    """Pairwise MIC distances, shape ``pos_i.shape[:-1] + pos_j.shape[:-1]``."""
    pos_i = jnp.asarray(pos_i)
    pos_j = jnp.asarray(pos_j)
    delta = pos_i[..., :, None, :] - pos_j[..., None, :, :]
    return periodic_distance(delta, boxsize)


def logits(r, centers, temperature):
    """``q_b(r) = -(r - c_b)^2 / T^2`` (PLAN.md 6.2)."""
    r = jnp.asarray(r)
    centers = jnp.asarray(centers, dtype=r.dtype)
    alpha = 1.0 / (temperature * temperature)
    diff = r[..., None] - centers
    return -alpha * diff * diff


def logits_derivative(r, centers, temperature):
    """``dq_b/dr = -2 (r - c_b) / T^2`` (PLAN.md 6.2)."""
    r = jnp.asarray(r)
    centers = jnp.asarray(centers, dtype=r.dtype)
    alpha = 1.0 / (temperature * temperature)
    return -2.0 * alpha * (r[..., None] - centers)


def softmax_weights(r, centers, temperature):
    """Stable softmax over bins (PLAN.md 6.3)."""
    q = logits(r, centers, temperature)
    q = q - jnp.max(q, axis=-1, keepdims=True)
    e = jnp.exp(q)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def gaussian_weights(r, centers, temperature):
    """Unnormalized Gaussian bin weights (PLAN.md 6.4)."""
    return jnp.exp(logits(r, centers, temperature))


def soft_weights(r, centers, temperature, mode="softmax"):
    """Dispatch on ``mode``; validates the mode string."""
    if mode == "softmax":
        return softmax_weights(r, centers, temperature)
    if mode == "gaussian":
        return gaussian_weights(r, centers, temperature)
    raise RadiusConfigError(f"mode must be one of {_MODES}, got {mode!r}")


def softmax_logit_mean(weights, logits_deriv):
    """``B = sum_b W_b q'_b``, the softmax score-function term."""
    return jnp.sum(weights * logits_deriv, axis=-1)


def softmax_vjp_r(r, centers, temperature, cotangent):
    """``dL/dr`` per pair for softmax mode, excluding the pair weight ``a_ij``.

    Implements PLAN.md 6.6: ``C - A * B`` with
    ``A = sum g W``, ``B = sum W q'``, ``C = sum g W q'``.
    """
    w = softmax_weights(r, centers, temperature)
    qp = logits_derivative(r, centers, temperature)
    g = jnp.asarray(cotangent)
    a = jnp.sum(g * w, axis=-1)
    b = jnp.sum(w * qp, axis=-1)
    c = jnp.sum(g * w * qp, axis=-1)
    return c - a * b


def gaussian_vjp_r(r, centers, temperature, cotangent):
    """``dL/dr`` per pair for Gaussian mode (PLAN.md 6.6): ``sum g W q'``."""
    w = gaussian_weights(r, centers, temperature)
    qp = logits_derivative(r, centers, temperature)
    return jnp.sum(jnp.asarray(cotangent) * w * qp, axis=-1)


def histogram_vjp_r(
    r, centers, temperature, cotangent, pair_weight=1.0, mode="softmax"
):
    """Full ``dH/dr`` for one pair, including ``a_ij`` (PLAN.md 6.6)."""
    if mode == "softmax":
        base = softmax_vjp_r(r, centers, temperature, cotangent)
    elif mode == "gaussian":
        base = gaussian_vjp_r(r, centers, temperature, cotangent)
    else:
        raise RadiusConfigError(f"mode must be one of {_MODES}, got {mode!r}")
    return pair_weight * base


def softmax_A(r, centers, temperature, cotangent):
    """``A = sum_b g_b W_b``, used for weight cotangents (PLAN.md 6.6)."""
    cotangent = jnp.asarray(cotangent)
    w = softmax_weights(r, centers, temperature)
    return jnp.sum(jnp.asarray(cotangent) * w, axis=-1)


def dW_dr(r, centers, temperature):
    """Analytic ``dW_b/dr = W_b (q'_b - sum_m W_m q'_m)``; for tests."""
    w = softmax_weights(r, centers, temperature)
    qp = logits_derivative(r, centers, temperature)
    return w * (qp - jnp.sum(w * qp, axis=-1, keepdims=True))


def pair_r_safe(boxsize, dtype) -> float:
    """PLAN.md 6.6 guardrail: pairs with ``r < r_safe`` are skipped in reverse mode."""
    return max(1e-12 * float(boxsize), r_epsilon(dtype))
