"""Integration tier (PLAN.md §18.1/§18.2): thousands of particles end to end.

These are marked ``integration`` (N <= 20 000, <= 900 s) and ``gpu`` because
the production path uses the jztree hierarchy, whose construction is itself an
FFI kernel.  The oracle is either the brute-force reference (small enough N)
or the exact CPU dual-tree traversal, which enumerates the same pair set.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from jzradius import radius_softmax_histogram
from jzradius.histogram import build_radius_trees
from jzradius.reference import reference_histogram
from jzradius.traversal import traverse_tree_pairs

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

BOX = 1.0
R_MAX = 0.08
TEMPERATURE = 0.05
CENTERS = np.linspace(0.004, 0.076, 12)


def _api(pos):
    return np.asarray(
        radius_softmax_histogram(
            jnp.asarray(pos),
            bin_centers=CENTERS,
            temperature=TEMPERATURE,
            boxsize=BOX,
            r_max=R_MAX,
            self_query=True,
            pair_convention="unique",
        )
    )


@pytest.mark.parametrize("n", [1000, 4000])
def test_matches_exact_traversal(n):
    rng = np.random.default_rng(7)
    pos = rng.uniform(0.0, BOX, (n, 3))
    tree_q, _, _ = build_radius_trees(pos, None, boxsize=BOX, leaf_size=32)
    expected = traverse_tree_pairs(
        tree_q,
        bin_centers=CENTERS,
        temperature=TEMPERATURE,
        boxsize=BOX,
        r_max=R_MAX,
        self_query=True,
        pair_convention="unique",
    )
    got = _api(pos)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected.histogram, rtol=1e-10, atol=1e-12)
    assert expected.stats.n_accepted_pairs > 0


def test_matches_brute_force_reference():
    rng = np.random.default_rng(42)
    n = 1000
    pos = rng.uniform(0.0, BOX, (n, 3))
    expected = np.asarray(
        reference_histogram(
            jnp.asarray(pos),
            jnp.asarray(pos),
            jnp.asarray(CENTERS),
            temperature=TEMPERATURE,
            boxsize=BOX,
            r_max=R_MAX,
            self_query=True,
            pair_convention="unique",
        )
    )
    np.testing.assert_allclose(_api(pos), expected, rtol=1e-10, atol=1e-12)
