"""Exact pair-set equivalence: CPU dual-tree traversal vs brute-force oracle.

Passing this test is a gate for M4 (PLAN.md 11.3, 18.2 pair_set_equivalence):
the traversal must reproduce the *exact set* of accepted ``(i, j)`` pairs, not
merely the same count.
"""

import numpy as np
import pytest

from jzradius import reference_histogram, reference_pair_set
from jzradius.traversal import traverse_tree_pairs
from jzradius.tree_adapter import build_naive_radius_tree

from .conftest import SEEDS, boundary_points, clustered_points, uniform_points

pytestmark = pytest.mark.unit

_DIST = {
    "uniform": uniform_points,
    "clustered": clustered_points,
    "boundary": boundary_points,
}


def _points(dist, n, box, seed):
    return _DIST[dist](n, box, np.random.default_rng(seed))


def _canon(i, j, r):
    return (min(int(i), int(j)), max(int(i), int(j)), float(r))


def _tree_pairs(tree, r_max, box, **kw):
    """Traversal pairs are already in original (input) index space."""
    res = traverse_tree_pairs(tree, r_max=r_max, boxsize=box, **kw)
    got = sorted(_canon(i, j, r) for (i, j), r in zip(res.pairs, res.r))
    return got, res


def _ref_pairs_sorted_space(pos, box, r_max, **kw):
    return reference_pair_set(pos, pos, box, r_max, **kw)


@pytest.mark.parametrize("n", [2, 3, 17, 64, 257])
@pytest.mark.parametrize("dist", ["uniform", "clustered", "boundary"])
@pytest.mark.parametrize("box", [1.0, 100.0])
def test_pair_set_matches_reference_self_query_unique(n, dist, box):
    r_max = 0.15
    pos = _points(dist, n, box, seed=7)
    tree = build_naive_radius_tree(pos, boxsize=box, leaf_size=8)
    got, res = _tree_pairs(tree, r_max, box)

    m = tree.sorted_to_orig
    ref = sorted(
        _canon(m[i], m[j], r) for i, j, r in _ref_pairs_sorted_space(
            tree.sorted_pos, box, r_max, self_query=True, pair_convention="unique"
        )
    )
    assert [p[:2] for p in got] == [p[:2] for p in ref]
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert abs(a[2] - b[2]) <= 1e-12
    assert res.stats.n_accepted_pairs == len(got)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dist", ["uniform", "clustered", "boundary"])
def test_pair_set_matches_reference_across_seeds(dist, seed):
    n, box, r_max = 64, 1.0, 0.2
    pos = _points(dist, n, box, seed)
    tree = build_naive_radius_tree(pos, boxsize=box, leaf_size=8)
    got, res = _tree_pairs(tree, r_max, box)
    m = tree.sorted_to_orig
    ref = sorted(
        _canon(m[i], m[j], r) for i, j, r in reference_pair_set(
            tree.sorted_pos, tree.sorted_pos, box, r_max,
            self_query=True, pair_convention="unique",
        )
    )
    assert [p[:2] for p in got] == [p[:2] for p in ref]
    assert res.stats.n_accepted_pairs == len(got)


@pytest.mark.parametrize("dist", ["uniform", "clustered"])
def test_directed_half_matches_reference(dist):
    n, box, r_max = 64, 1.0, 0.2
    pos = _points(dist, n, box, seed=42)
    tree = build_naive_radius_tree(pos, boxsize=box, leaf_size=8)
    res = traverse_tree_pairs(
        tree, r_max=r_max, boxsize=box, pair_convention="directed_half"
    )
    m = tree.sorted_to_orig
    got = sorted((int(i), int(j)) for i, j in res.pairs)
    ref = sorted(
        (int(m[i]), int(m[j])) for i, j, _ in reference_pair_set(
            tree.sorted_pos, tree.sorted_pos, box, r_max,
            self_query=True, pair_convention="directed_half",
        )
    )
    assert got == ref
    # both directions of every unordered pair are present exactly once
    assert len(got) == len(set(got))
    assert len(got) % 2 == 0


@pytest.mark.parametrize("dist", ["clustered", "boundary"])
def test_cross_correlation_matches_reference(dist):
    box, r_max = 1.0, 0.2
    rng = np.random.default_rng(3)
    pos_q = _DIST[dist](50, box, rng)
    pos_s = _DIST["uniform"](70, box, rng)
    tq = build_naive_radius_tree(pos_q, boxsize=box, leaf_size=8)
    ts = build_naive_radius_tree(pos_s, boxsize=box, leaf_size=8)
    res = traverse_tree_pairs(
        tq, tree_s=ts, self_query=False, r_max=r_max, boxsize=box
    )
    got = sorted(
        (int(i), int(j), float(r)) for (i, j), r in zip(res.pairs, res.r)
    )
    ref = sorted(
        (int(i), int(j), float(r)) for i, j, r in reference_pair_set(
            pos_q, pos_s, box, r_max, self_query=False,
        )
    )
    assert [p[:2] for p in got] == [p[:2] for p in ref]
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert abs(a[2] - b[2]) <= 1e-12
    assert res.stats.n_accepted_pairs == len(got)


@pytest.mark.parametrize("dist", ["uniform", "clustered"])
def test_histogram_matches_reference(dist):
    n, box, r_max = 96, 1.0, 0.3
    centers = np.linspace(0.01, 0.29, 15)
    pos = _points(dist, n, box, seed=0)
    tree = build_naive_radius_tree(pos, boxsize=box, leaf_size=8)
    res = traverse_tree_pairs(
        tree, r_max=r_max, boxsize=box, bin_centers=centers, temperature=0.02,
    )
    ref = np.asarray(
        reference_histogram(
            tree.sorted_pos, tree.sorted_pos, centers, 0.02, box,
            r_max=r_max, self_query=True,
        )
    )
    assert np.all(np.isfinite(res.histogram))
    assert np.allclose(res.histogram, ref, rtol=0, atol=1e-9)


def test_weighted_histogram_matches_reference():
    n, box, r_max = 64, 1.0, 0.3
    centers = np.linspace(0.01, 0.29, 9)
    pos = _points("clustered", n, box, seed=1)
    rng = np.random.default_rng(5)
    wq = rng.uniform(0.5, 2.0, n)
    # For unordered self-pairs the orientation is index-space dependent, so
    # asymmetric query/source weights would change the pair's a_ij under a
    # permutation.  Symmetric weights keep the convention well defined here;
    # asymmetric weights are exercised by the cross-correlation tests.
    ws = wq.copy()
    tree = build_naive_radius_tree(pos, boxsize=box, leaf_size=8)
    res = traverse_tree_pairs(
        tree, r_max=r_max, boxsize=box, bin_centers=centers, temperature=0.05,
        weights_query=wq, weights_source=ws,
    )
    # reference operates in sorted-index space: permute the per-particle weights
    perm = tree.sorted_to_orig
    ref = np.asarray(
        reference_histogram(
            tree.sorted_pos, tree.sorted_pos, centers, 0.05, box,
            r_max=r_max, weights_query=wq[perm], weights_source=ws[perm],
            self_query=True,
        )
    )
    assert np.allclose(res.histogram, ref, rtol=0, atol=1e-9)
