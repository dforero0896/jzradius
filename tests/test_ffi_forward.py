"""Forward histogram through the typed-FFI kernel (CPU mirror of the CUDA core).

GPU execution is blocked on this cluster, so the same FFI handler is compiled by
g++ and driven through `jax.ffi.ffi_call` on the CPU host.  The CUDA handler is
byte-identical C++ (shared header `src/cuda/jzradius_kernels.cuh`); only the
launch/stream plumbing differs.  Real-GPU smoke tests are marked `gpu` below.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from jzradius import reference_histogram, traverse_tree_pairs
from jzradius import ffi
from jzradius.histogram import build_radius_trees
from jzradius.traversal import build_work_items
from tests.conftest import DISTRIBUTIONS

pytestmark = pytest.mark.unit

BOXES = [1.0, 100.0]


def _ref_hist(pos, centers, *, boxsize, r_max, temperature, self_query, wq, ws, mode):
    return reference_histogram(
        pos,
        pos if self_query else None,
        centers,
        boxsize=boxsize,
        r_max=r_max,
        temperature=temperature,
        self_query=self_query,
        pair_convention="unique",
        weights_query=wq,
        weights_source=ws,
        mode=mode,
    )


@pytest.mark.parametrize("boxsize", BOXES)
@pytest.mark.parametrize("mode", ["softmax", "gaussian"])
@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_public_forward_matches_reference(ffi_cpu, distribution, boxsize, mode, dtype):
    n, leaf, seed = 64, 8, 7
    pos = distribution(n, boxsize, seed, dtype=dtype)
    centers = np.linspace(0.05 * boxsize, 0.3 * boxsize, 6)
    r_max, temperature = 0.3 * boxsize, 0.05 * boxsize
    wq = jnp.asarray(np.random.default_rng(3).uniform(0.5, 1.5, n), dtype=dtype)
    from jzradius.histogram import radius_softmax_histogram

    got = radius_softmax_histogram(
        pos,
        bin_centers=centers,
        boxsize=boxsize,
        r_max=r_max,
        temperature=temperature,
        weights_query=wq,
        weights_source=wq,
        self_query=True,
        leaf_size=leaf,
        mode=mode,
    )
    ref = _ref_hist(
        pos, centers, boxsize=boxsize, r_max=r_max, temperature=temperature,
        self_query=True, wq=wq, ws=wq, mode=mode,
    )
    tol = 1e-5 if dtype == jnp.float32 else 1e-9
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=tol, atol=tol)


@pytest.mark.parametrize("boxsize", BOXES)
def test_cross_forward_matches_reference(ffi_cpu, distribution, boxsize):
    nq, ns, leaf, seed = 64, 48, 8, 11
    pq = distribution(nq, boxsize, 3)
    ps = distribution(ns, boxsize, seed)
    centers = np.linspace(0.02 * boxsize, 0.25 * boxsize, 5)
    r_max, temperature = 0.25 * boxsize, 0.04 * boxsize
    wq = jnp.asarray(np.random.default_rng(3).uniform(0.5, 1.5, nq))
    ws = jnp.asarray(np.random.default_rng(4).uniform(0.5, 1.5, ns))
    from jzradius.histogram import radius_softmax_histogram

    got = radius_softmax_histogram(
        pq, ps, centers, boxsize=boxsize, r_max=r_max, temperature=temperature,
        weights_query=wq, weights_source=ws, leaf_size=leaf,
    )
    ref = reference_histogram(
        pq, ps, centers, boxsize=boxsize, r_max=r_max, temperature=temperature,
        weights_query=wq, weights_source=ws,
    )
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-9, atol=1e-9)


def test_directed_half_equals_unique(ffi_cpu):
    from jzradius.histogram import radius_softmax_histogram

    pos = DISTRIBUTIONS["clustered"](32, 1.0, 5)
    centers = np.linspace(0.0, 0.3, 5)
    kw = dict(boxsize=1.0, r_max=0.3, temperature=0.05, leaf_size=8)
    unique = radius_softmax_histogram(pos, bin_centers=centers, pair_convention="unique", **kw)
    half = radius_softmax_histogram(pos, bin_centers=centers, pair_convention="directed_half", **kw)
    np.testing.assert_allclose(np.asarray(half), np.asarray(unique), rtol=1e-9, atol=1e-9)


def test_ffi_kernel_agrees_with_cpu_traversal(ffi_cpu):
    """Drive the kernel directly with the same work items as the CPU traversal."""
    pos = DISTRIBUTIONS["uniform"](48, 1.0, 42)
    centers = np.linspace(0.0, 0.3, 4)
    tree_q, tree_s, self_query = build_radius_trees(
        np.asarray(pos, dtype=np.float64), None, boxsize=1.0, leaf_size=8
    )
    work = build_work_items(tree_q, tree_s, r_max=0.3, boxsize=1.0)
    params = ffi.build_params(
        n_q=48, n_s=48, n_bins=centers.size, n_work=work.shape[0],
        self_query=True, triangular=True, mode="softmax", r_max=0.3,
        boxsize=1.0, temperature=0.05, eps=1e-60, r_safe=1e-12, pair_scale=1.0,
    )
    perm = jnp.asarray(tree_q.sorted_to_orig, dtype=jnp.int32)
    sorted_pos = jnp.take(pos, perm, axis=0)
    ones = jnp.ones((48,), dtype=jnp.float64)
    got = ffi.forward(
        sorted_pos, sorted_pos, jnp.asarray(centers), jnp.asarray(work, dtype=jnp.int32),
        ones, ones, perm, params,
    )
    got = np.asarray(got)
    res = traverse_tree_pairs(
        tree_q, tree_s=tree_s, r_max=0.3, boxsize=1.0, bin_centers=centers,
        temperature=0.05, self_query=True,
    )
    np.testing.assert_allclose(got, np.asarray(res.histogram), rtol=1e-12, atol=1e-12)


def test_empty_interaction_is_all_zeros(ffi_cpu):
    from jzradius.histogram import radius_softmax_histogram

    pos = DISTRIBUTIONS["uniform"](16, 1.0, 1)
    got = radius_softmax_histogram(
        pos, bin_centers=np.linspace(0.0, 1e-4, 3), boxsize=1.0, r_max=1e-4,
        temperature=0.05, leaf_size=4,
    )
    np.testing.assert_array_equal(np.asarray(got), 0.0)


@pytest.mark.gpu
def test_forward_on_cuda_backend():
    pytest.importorskip("jztree")
    from jzradius.histogram import radius_softmax_histogram

    pos = DISTRIBUTIONS["uniform"](32, 1.0, 2)
    got = radius_softmax_histogram(
        pos, bin_centers=np.linspace(0.0, 0.3, 4), boxsize=1.0, r_max=0.3,
        leaf_size=8,
    )
    assert np.all(np.isfinite(np.asarray(got)))
