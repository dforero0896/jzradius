"""Periodic lower-bound tests (PLAN.md 10.5 and 18.2 test_bounds).

This is a mandatory gate: if the bound ever exceeds the true minimum distance,
the traversal may silently miss pairs and no GPU work may proceed.
"""

from __future__ import annotations

import numpy as np
import pytest

from jzradius.bounds import (

    interval_gap_reference,
    lower_bound_boxes,
    lower_bound_spheres,
    periodic_interval_gap,
    prune_tolerance,
)

pytestmark = pytest.mark.unit


def _mic_distance(a, b, boxsize):
    d = np.asarray(a) - np.asarray(b)
    d = d - boxsize * np.round(d / boxsize)
    return float(np.sqrt(np.dot(d, d)))


def _random_boxes(rng, boxsize, n, max_extent):
    centers = rng.uniform(0.0, boxsize, size=(n, 3))
    extents = rng.uniform(0.0, max_extent, size=(n, 3))
    return centers - extents / 2.0, centers + extents / 2.0


@pytest.mark.parametrize("boxsize", [1.0, 10.0, 100.0])
def test_scalar_gap_matches_reference(boxsize):
    rng = np.random.default_rng(0)
    for _ in range(200):
        a0, a1 = np.sort(rng.uniform(-0.2 * boxsize, 1.2 * boxsize, size=2))
        b0, b1 = np.sort(rng.uniform(-0.2 * boxsize, 1.2 * boxsize, size=2))
        got = float(periodic_interval_gap(a0, a1, b0, b1, boxsize))
        expected = interval_gap_reference(a0, a1, b0, b1, boxsize)
        assert got == pytest.approx(expected, abs=1e-12)


def test_seam_spanning_boxes_have_zero_gap():
    boxsize = 10.0
    # A straddles the seam at L; B sits just above 0.
    lb = float(
        lower_bound_boxes(
            np.array([9.0, 0.0, 0.0]),
            np.array([11.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 1.0, 1.0]),
            boxsize,
        )
    )
    assert lb == 0.0


def test_adjacent_boxes_across_the_seam():
    boxsize = 10.0
    # A ends exactly at the seam, B starts exactly at 0: touching in MIC.
    lb = float(
        lower_bound_boxes(
            np.array([9.5, 0.0, 0.0]),
            np.array([10.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 1.0, 1.0]),
            boxsize,
        )
    )
    assert lb == pytest.approx(0.0, abs=1e-12)


@pytest.mark.smoke
def test_identical_and_degenerate_boxes():
    boxsize = 10.0
    point = np.array([3.0, 4.0, 5.0])
    assert float(lower_bound_boxes(point, point, point, point, boxsize)) == 0.0
    assert float(
        lower_bound_boxes(
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
            boxsize,
        )
    ) == 0.0


def test_exactly_half_box_separation():
    boxsize = 10.0
    lb = float(
        lower_bound_boxes(
            np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]),
            np.array([6.0, 1.0, 1.0]), np.array([6.0, 1.0, 1.0]),
            boxsize,
        )
    )
    assert lb == pytest.approx(5.0, abs=1e-12)


def test_property_lower_bound_never_exceeds_true_minimum():
    rng = np.random.default_rng(7)
    boxsize = 10.0
    n_boxes = 400
    n_particles = 6
    lo, hi = _random_boxes(rng, boxsize, 2 * n_boxes, max_extent=0.4 * boxsize)
    lo = lo.reshape(n_boxes, 2, 3)
    hi = hi.reshape(n_boxes, 2, 3)

    for k in range(n_boxes):
        particles = [
            rng.uniform(lo[k, side], hi[k, side], size=(n_particles, 3))
            for side in (0, 1)
        ]
        true_min = min(
            _mic_distance(p, q, boxsize)
            for p in particles[0]
            for q in particles[1]
        )
        lb = float(lower_bound_boxes(lo[k, 0], hi[k, 0], lo[k, 1], hi[k, 1], boxsize))
        assert lb <= true_min + 1e-9, (k, lb, true_min)


def test_bounding_sphere_bound_is_conservative():
    rng = np.random.default_rng(11)
    boxsize = 10.0
    for _ in range(300):
        a = rng.uniform(0, boxsize, size=3)
        b = rng.uniform(0, boxsize, size=3)
        ra, rb = rng.uniform(0.0, 0.2, size=2)

        def sample_in_ball(center, radius, n=5):
            u = rng.normal(size=(n, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            rad = radius * rng.random(n) ** (1.0 / 3.0)
            return center + u * rad[:, None]

        pa = sample_in_ball(a, ra)
        pb = sample_in_ball(b, rb)
        true_min = min(_mic_distance(p, q, boxsize) for p in pa for q in pb)
        lb = float(lower_bound_spheres(a, ra, b, rb, boxsize))
        assert lb <= true_min + 1e-9


def test_prune_tolerance_formula():
    assert prune_tolerance(10.0, np.float64) == pytest.approx(
        8.0 * np.finfo(np.float64).eps * 10.0
    )
    assert prune_tolerance(10.0, np.float32) > prune_tolerance(10.0, np.float64)
