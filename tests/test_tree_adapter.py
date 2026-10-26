"""Tests for the jztree -> RadiusTree adapter (PLAN.md 9.3, 18.2 tree_adapter)."""

import numpy as np
import pytest

from jzradius.errors import TreeAdapterUnavailable
from jzradius.tree_adapter import (
    LevelData,
    RadiusTree,
    build_naive_radius_tree,
    build_radius_tree_from_levels,
    check_tree_invariants,
)

from .conftest import SEEDS, clustered_points, boundary_points, uniform_points

pytestmark = pytest.mark.unit

BOX = 1.0
_DIST = {
    "uniform": uniform_points,
    "clustered": clustered_points,
    "boundary": boundary_points,
}


def _make(dist, seed, n):
    return _DIST[dist](n, BOX, np.random.default_rng(seed))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dist", ["uniform", "clustered", "boundary"])
@pytest.mark.parametrize("n", [2, 3, 17, 64, 257])
def test_naive_tree_satisfies_invariants(dist, seed, n):
    pos = _make(dist, seed, n)
    tree = build_naive_radius_tree(pos, boxsize=BOX, leaf_size=8)
    check_tree_invariants(tree)          # raises on any violation
    assert tree.n_particles == n
    assert tree.root == tree.n_nodes - 1
    assert tree.node_count[tree.root] == n
    assert np.all(tree.node_level[tree.root] > tree.node_level[:-1])


@pytest.mark.parametrize("dist", ["uniform", "clustered"])
def test_leaves_partition_sorted_particles(dist):
    pos = _make(dist, 7, 64)
    tree = build_naive_radius_tree(pos, boxsize=BOX, leaf_size=8)
    leaves = [k for k in range(tree.n_nodes) if tree.is_leaf[k]]
    starts = sorted(int(tree.node_start[k]) for k in leaves)
    assert starts[0] == 0
    covered = 0
    for k in sorted(leaves, key=lambda k: tree.node_start[k]):
        s, c = int(tree.node_start[k]), int(tree.node_count[k])
        assert s == covered
        covered += c
    assert covered == tree.n_particles


@pytest.mark.parametrize("dist", ["clustered", "boundary"])
def test_boxes_are_conservative(dist):
    pos = _make(dist, 1, 128)
    tree = build_naive_radius_tree(pos, boxsize=BOX, leaf_size=8)
    for k in range(tree.n_nodes):
        if tree.is_leaf[k]:
            s = int(tree.node_start[k])
            c = int(tree.node_count[k])
            assert np.all(tree.node_lo[k] <= pos[tree.sorted_to_orig[s:s + c]] + 1e-12)
            assert np.all(tree.node_hi[k] >= pos[tree.sorted_to_orig[s:s + c]] - 1e-12)
        else:
            c0, c1 = int(tree.node_left[k]), int(tree.node_right[k])
            assert np.all(tree.node_lo[k] <= tree.node_lo[c0:c1])
            assert np.all(tree.node_hi[k] >= tree.node_hi[c0:c1])


def test_root_box_is_conservative():
    pos = _make("uniform", 0, 32)
    tree = build_naive_radius_tree(pos, boxsize=BOX, leaf_size=4)
    # the root is tightened to the particle hull (still conservative)
    assert np.allclose(tree.node_lo[tree.root], pos.min(axis=0))
    assert np.allclose(tree.node_hi[tree.root], pos.max(axis=0))


def test_radius_is_enclosing_sphere_of_box():
    tree = build_naive_radius_tree(_make("clustered", 0, 64), boxsize=BOX, leaf_size=4)
    half_diag = 0.5 * np.linalg.norm(tree.node_hi - tree.node_lo, axis=1)
    assert np.allclose(tree.node_radius, half_diag)


def test_build_from_levels_handwritten():
    """Two planes: 4 leaves x 2 particles, one coarsened node, virtual root."""
    rng = np.random.default_rng(0)
    pos = np.asarray(sorted(rng.uniform(0, BOX, (8, 3)).tolist()), dtype=np.float64)
    leaf_splits = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    leaves = LevelData(
        center=pos.reshape(4, 2, 3).mean(axis=1),
        level=np.zeros(4, dtype=np.int64),
        leaf_splits=leaf_splits,
        child_splits=leaf_splits.copy(),
    )
    top = LevelData(
        center=pos.mean(axis=0)[None, :],
        level=np.ones(1, dtype=np.int64),
        leaf_splits=np.array([0, 4], dtype=np.int64),
        child_splits=np.array([0, 4], dtype=np.int64),
    )
    tree = build_radius_tree_from_levels(
        [leaves, top], pos, np.arange(8), boxsize=BOX, leaf_size=2
    )
    assert isinstance(tree, RadiusTree)
    assert tree.n_nodes == 6                     # 4 leaves + node + root
    assert tree.root == 5
    assert tree.n_particles == 8
    assert tree.node_start[5] == 0 and tree.node_count[5] == 8
    assert tree.node_start[4] == 0 and tree.node_count[4] == 8
    assert np.allclose(tree.node_lo[5], pos.min(axis=0))
    assert np.allclose(tree.node_hi[5], pos.max(axis=0))
    # leaves cover particle ranges in order
    for k in range(4):
        assert tree.node_start[k] == 2 * k
        assert tree.node_count[k] == 2
    check_tree_invariants(tree)


def test_from_levels_rejects_incomplete_partition():
    pos = np.zeros((8, 3))
    leaf_splits = np.array([0, 2, 4, 6, 7], dtype=np.int64)   # misses particle 7
    leaves = LevelData(
        center=pos.reshape(4, 2, 3).mean(axis=1),
        level=np.zeros(4, dtype=np.int64),
        leaf_splits=leaf_splits,
        child_splits=leaf_splits.copy(),
    )
    with pytest.raises(TreeAdapterUnavailable):
        build_radius_tree_from_levels([leaves], pos, np.arange(8), boxsize=BOX, leaf_size=2)


def test_from_levels_rejects_bad_shapes():
    pos = np.zeros((8, 3))
    good = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    leaves = LevelData(
        center=pos.reshape(4, 2, 3).mean(axis=1),
        level=np.zeros(4, dtype=np.int64),
        leaf_splits=good,
        child_splits=np.array([0, 4], dtype=np.int64),   # wrong length
    )
    with pytest.raises(TreeAdapterUnavailable):
        build_radius_tree_from_levels([leaves], pos, np.arange(8), boxsize=BOX, leaf_size=2)


def test_invariant_checker_catches_corruption():
    tree = build_naive_radius_tree(_make("uniform", 0, 32), boxsize=BOX, leaf_size=4)
    tree.node_parent[tree.root] = 0
    with pytest.raises(TreeAdapterUnavailable):
        check_tree_invariants(tree)


def test_single_particle_tree():
    tree = build_naive_radius_tree(np.zeros((1, 3)), boxsize=BOX, leaf_size=4)
    assert tree.n_particles == 1
    assert tree.is_leaf[0]
    check_tree_invariants(tree)


@pytest.mark.gpu
def test_adapts_a_real_jztree_hierarchy(monkeypatch):
    """End-to-end adapter check against jztree's own tree planes.

    jztree constructs the Morton tree through an FFI kernel, so this test needs
    a GPU; the synthetic-levels test above covers the adapter on CPU.
    """
    jax = pytest.importorskip("jax")
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("jztree tree construction requires a GPU")
    # D002: the editable clone ships a CUDA 12 backend while JAX uses CUDA 13
    monkeypatch.setenv("JZTREE_SKIP_JAX_CUDA_CHECK", "1")
    jztree = pytest.importorskip("jztree.tree")
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    pos = jnp.asarray(rng.uniform(0, BOX, (300, 3)))
    partz, th = jztree.zsort_and_tree(pos)
    from jzradius.tree_adapter import build_radius_tree_from_jztree

    tree = build_radius_tree_from_jztree(
        th, np.asarray(partz), np.arange(300), boxsize=BOX, leaf_size=32
    )
    check_tree_invariants(tree)
    assert tree.n_nodes > 1
    assert tree.node_count[tree.root] == 300


def test_add_balancing_levels_preserves_leaves_and_boxes():
    """Synthetic coarse planes must not change the leaves or the pair set."""
    from jzradius.tree_adapter import add_balancing_levels

    pos = _make("uniform", 7, 120)
    flat = build_naive_radius_tree(pos, boxsize=BOX, leaf_size=8)
    leaf_ids = np.flatnonzero(flat.is_leaf)  # not contiguous in a pre-order tree
    n_leaf = leaf_ids.size
    order = np.argsort(flat.node_start[leaf_ids])  # leaves in particle order
    counts = flat.node_count[leaf_ids][order]
    leaves = LevelData(
        center=flat.node_center[leaf_ids][order],
        level=np.zeros(n_leaf, dtype=np.int64),
        leaf_splits=np.concatenate([[0], np.cumsum(counts)]),
        child_splits=np.zeros(n_leaf + 1, dtype=np.int64),
    )
    balanced = add_balancing_levels([leaves], group=4)
    assert len(balanced) > 1
    assert int(balanced[-1].center.shape[0]) <= 4
    tree = build_radius_tree_from_levels(
        balanced, flat.sorted_pos, flat.sorted_to_orig, boxsize=BOX, leaf_size=8
    )
    check_tree_invariants(tree)
    np.testing.assert_array_equal(
        tree.node_start[:n_leaf], flat.node_start[leaf_ids][order]
    )
    np.testing.assert_array_equal(tree.node_count[:n_leaf], counts)
    assert tree.node_right[tree.root] - tree.node_left[tree.root] <= 4
