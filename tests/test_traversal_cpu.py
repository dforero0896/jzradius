"""Traversal mechanics on CPU (PLAN.md 11.2, 18.2 traversal_cpu)."""

import numpy as np
import pytest

from jzradius.errors import RadiusConfigError
from jzradius.traversal import TraversalStats, traverse_tree_pairs
from jzradius.tree_adapter import build_naive_radius_tree

from .conftest import clustered_points, uniform_points

pytestmark = pytest.mark.unit


def _tree(n=128, seed=0, dist="clustered", leaf_size=8):
    box = 1.0
    pos = (
        clustered_points(n, box, np.random.default_rng(seed))
        if dist == "clustered"
        else uniform_points(n, box, np.random.default_rng(seed))
    )
    return build_naive_radius_tree(pos, boxsize=box, leaf_size=leaf_size), box


def test_stats_are_populated_and_consistent():
    tree, box = _tree()
    res = traverse_tree_pairs(tree, r_max=0.2, boxsize=box)
    s = res.stats
    assert isinstance(s, TraversalStats)
    assert s.n_node_pairs_visited > 0
    assert s.n_leaf_pairs > 0
    assert s.max_queue_size > 0
    assert s.n_accepted_pairs == len(res.pairs)
    assert s.n_candidate_particle_pairs >= s.n_accepted_pairs
    assert s.n_node_pairs_visited >= s.n_node_pairs_pruned


def test_pruning_is_effective_for_clustered_data():
    tree, box = _tree(n=256, seed=1)
    res = traverse_tree_pairs(tree, r_max=0.05, boxsize=box)
    s = res.stats
    assert s.n_node_pairs_pruned > 0
    # far fewer visits than the O(N^2) leaf-pair enumeration
    assert s.n_node_pairs_visited < tree.n_particles ** 2
    assert s.n_leaf_pairs < tree.n_particles ** 2


def test_tiny_radius_yields_no_pairs():
    tree, box = _tree()
    res = traverse_tree_pairs(tree, r_max=1e-4, boxsize=box)
    assert len(res.pairs) == 0
    assert res.stats.n_accepted_pairs == 0
    assert res.stats.n_node_pairs_pruned > 0


def test_single_particle_has_no_pairs():
    tree = build_naive_radius_tree(np.zeros((1, 3)), boxsize=1.0, leaf_size=8)
    res = traverse_tree_pairs(tree, r_max=0.5, boxsize=1.0)
    assert len(res.pairs) == 0
    assert res.stats.n_candidate_particle_pairs == 0
    assert res.stats.max_queue_size >= 1


def test_traversal_is_deterministic():
    tree, box = _tree()
    a = traverse_tree_pairs(tree, r_max=0.2, boxsize=box)
    b = traverse_tree_pairs(tree, r_max=0.2, boxsize=box)
    assert np.array_equal(a.pairs, b.pairs)
    assert np.array_equal(a.r, b.r)
    assert a.stats == b.stats


def test_pairs_are_sorted_and_within_r_max():
    tree, box = _tree()
    r_max = 0.2
    res = traverse_tree_pairs(tree, r_max=r_max, boxsize=box)
    order = np.lexsort((res.pairs[:, 1], res.pairs[:, 0]))
    assert np.array_equal(order, np.arange(len(res.pairs)))
    assert np.all(res.r <= r_max)
    assert np.all(res.r > 0)


def test_leaf_size_does_not_change_the_pair_set():
    box = 1.0
    pos = uniform_points(200, box, np.random.default_rng(2))
    sets = []
    for leaf_size in (2, 5, 16, 64):
        tree = build_naive_radius_tree(pos, boxsize=box, leaf_size=leaf_size)
        res = traverse_tree_pairs(tree, r_max=0.1, boxsize=box)
        sets.append(set(map(tuple, res.pairs.tolist())))
    assert all(s == sets[0] for s in sets)


def test_rejects_invalid_arguments():
    tree, box = _tree(n=32)
    with pytest.raises(RadiusConfigError):
        traverse_tree_pairs(tree, r_max=0.0, boxsize=box)
    with pytest.raises(RadiusConfigError):
        traverse_tree_pairs(tree, r_max=float("nan"), boxsize=box)
    with pytest.raises(RadiusConfigError):
        traverse_tree_pairs(tree, r_max=0.1, boxsize=box, pair_convention="bogus")
    with pytest.raises(RadiusConfigError):
        traverse_tree_pairs(
            tree, r_max=0.1, boxsize=box,
            weights_query=np.ones(3),          # wrong length
        )
    other = build_naive_radius_tree(
        np.zeros((4, 3)), boxsize=box, leaf_size=2
    )
    with pytest.raises(RadiusConfigError):
        traverse_tree_pairs(
            tree, tree_s=other, self_query=True, r_max=0.1, boxsize=box
        )
