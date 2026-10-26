"""Device-side dual-tree work queue vs the host descent (M9.1, D013).

The public call uses the device queue on GPU and the interpreted host descent
on CPU, so these tests are what keeps the two implementations honest: they
compare the *sets* of leaf-pair rectangles, which is the contract the forward
and backward kernels consume. Both are driven through the same typed-FFI path
(the CPU mirror here, CUDA on a GPU node).
"""

import numpy as np
import pytest

from jzradius.errors import RadiusConfigError, WorkQueueOverflow
from jzradius.ffi import ffi_available
from jzradius.traversal import build_work_items
from jzradius.tree_adapter import build_naive_radius_tree
from jzradius.work_queue import _node_arrays, build_work_items_device

from .conftest import clustered_points, uniform_points

pytestmark = pytest.mark.unit

BOX = 1.0
LEAF = 8


def _trees(n, seed, dist="uniform"):
    gen = uniform_points if dist == "uniform" else clustered_points
    pos = gen(n, BOX, np.random.default_rng(seed))
    tree = build_naive_radius_tree(pos, boxsize=BOX, leaf_size=LEAF)
    return tree, np.asarray(pos)


def _as_set(items):
    return sorted(map(tuple, np.asarray(items, dtype=np.int64).tolist()))


@pytest.mark.parametrize("n", [17, 64, 257, 1000])
@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("r_max", [0.01, 0.05, 0.15])
@pytest.mark.parametrize("unique_pairs", [True, False])
def test_device_queue_matches_host_descent(ffi_cpu, n, seed, r_max, unique_pairs):
    """Same leaf-pair rectangles, bit for bit, for self and ordered modes."""
    tree, _ = _trees(n, seed)
    host = build_work_items(
        tree, None, r_max=r_max, boxsize=BOX, unique_pairs=unique_pairs
    )
    dev = build_work_items_device(
        tree, None, r_max=r_max, boxsize=BOX, unique_pairs=unique_pairs
    )
    assert _as_set(dev) == _as_set(host)


@pytest.mark.parametrize("n", [64, 257])
def test_device_queue_matches_host_descent_cross(ffi_cpu, n):
    """Distinct query/source trees take the non-self descent branch."""
    tree_q, _ = _trees(n, 0)
    tree_s, _ = _trees(n + 13, 1, dist="clustered")
    host = build_work_items(tree_q, tree_s, r_max=0.05, boxsize=BOX)
    dev = build_work_items_device(tree_q, tree_s, r_max=0.05, boxsize=BOX)
    assert _as_set(dev) == _as_set(host)


def test_device_queue_rejects_oversized_queue(ffi_cpu):
    """A queue that cannot fit the caller's budget fails loudly, never truncates."""
    tree, _ = _trees(400, 3)
    with pytest.raises(WorkQueueOverflow):
        build_work_items_device(
            tree, None, r_max=0.4, boxsize=BOX, max_work_items=4
        )


def test_device_queue_empty_when_nothing_is_in_range(ffi_cpu):
    """Disjoint clouds prune at the root, so the queue is empty, not padded."""
    rng = np.random.default_rng(5)
    pos_q = rng.uniform(0.0, 0.2, size=(120, 3))
    pos_s = rng.uniform(0.6, 0.8, size=(120, 3))
    tree_q = build_naive_radius_tree(pos_q, boxsize=BOX, leaf_size=LEAF)
    tree_s = build_naive_radius_tree(pos_s, boxsize=BOX, leaf_size=LEAF)
    dev = build_work_items_device(tree_q, tree_s, r_max=0.01, boxsize=BOX)
    host = build_work_items(tree_q, tree_s, r_max=0.01, boxsize=BOX)
    assert host.shape == (0, 4)
    assert dev.shape == (0, 4)
    assert str(dev.dtype) == "int32"


def test_device_queue_rejects_bad_radius(ffi_cpu):
    tree, _ = _trees(64, 0)
    with pytest.raises(RadiusConfigError):
        build_work_items_device(tree, None, r_max=-1.0, boxsize=BOX)


def test_node_arrays_are_contiguous_and_typed():
    tree, _ = _trees(64, 0)
    arrs = _node_arrays(tree)
    assert arrs["start"].shape == (tree.n_nodes,)
    assert arrs["start"].dtype == np.int32
    assert arrs["lo"].shape == (tree.n_nodes, 3)
    assert arrs["lo"].dtype == np.float64
    # children of internal nodes must stay inside the array
    leaves = arrs["left"] < 0
    assert np.all(arrs["right"][leaves] < 0)
    assert np.all(arrs["left"][~leaves] >= 0)


@pytest.mark.gpu
def test_device_queue_runs_on_cuda():
    """The CUDA translation unit of the same helper, on a real device."""
    import jax

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU device")
    if not ffi_available():
        pytest.skip("CUDA FFI library not loadable")
    tree, _ = _trees(300, 2)
    host = build_work_items(tree, None, r_max=0.05, boxsize=BOX)
    dev = build_work_items_device(tree, None, r_max=0.05, boxsize=BOX)
    assert _as_set(dev) == _as_set(host)
