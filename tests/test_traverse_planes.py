"""Plane-native traversal op (`ffi.traverse`) on the CPU mirror.

Production runs the walk inside one device op driven by C++ over jztree's
packed planes (PLAN_ADDENDUM Rev 3).  jztree's hierarchy can only be built on a
GPU, so here we build a *synthetic* levelised hierarchy straight from particle
order: plane 0 chunks the particles into leaves, and each higher plane groups
``fanout`` consecutive nodes of the plane below it.  The op must not care where
the planes came from -- it only needs a conservative hierarchy -- so this
exercises the real code path (node materialisation, bottom-up boxes,
refine/scan/scatter, sentinel padding, capacity-sized launches) on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jzradius import ffi
from jzradius.errors import JzRadiusError
from jzradius.reference import reference_histogram

pytestmark = pytest.mark.unit

BOX = 1.0
LEAF = 8
FANOUT = 4


def grouped_planes(n_particles: int, *, leaf_size: int = LEAF, fanout: int = FANOUT):
    """Levelised packed planes: leaves chunk particles, parents group children."""
    split0 = np.arange(0, n_particles + 1, leaf_size, dtype=np.int32)
    if split0[-1] != n_particles:
        split0 = np.concatenate([split0, np.array([n_particles], np.int32)])
    planes = [split0]  # plane 0: leaf -> particle splits
    while len(planes[-1]) - 1 > 1:
        child = planes[-1]
        n_child = child.size - 1
        starts = list(range(0, n_child, fanout))
        splits = np.array(starts + [n_child], dtype=np.int32)
        planes.append(splits)
    # Packed n2n: plane p>0 holds child *indices* into plane p-1.
    n2n_chunks = [planes[0]]
    for split in planes[1:]:
        n2n_chunks.append(split.astype(np.int32))
    n2n = np.concatenate(n2n_chunks)
    n2n_ispl = np.concatenate(
        [[0], np.cumsum([c.size for c in n2n_chunks])]
    ).astype(np.int32)

    # n2l: node -> leaf range.  Leaves are their own range.
    n_lv = len(planes)
    n_leaves = planes[0].size - 1
    n2l_chunks = []
    leaf_lo = np.arange(n_leaves, dtype=np.int32)
    n2l_chunks.append(np.concatenate([leaf_lo, [n_leaves]]).astype(np.int32))
    for p in range(1, n_lv):
        child_ispl = planes[p - 1] - 1  # unused; child leaf ranges are contiguous
        prev = n2l_chunks[-1]
        split = planes[p]
        out = np.empty(split.size, dtype=np.int32)
        for i in range(split.size - 1):
            out[i] = prev[int(split[i])]
        out[-1] = n_leaves
        n2l_chunks.append(out)
    n2l = np.concatenate(n2l_chunks)
    n2l_ispl = n2n_ispl.copy()
    return {
        "n2n": jnp.asarray(n2n, dtype=jnp.int32),
        "n2n_ispl": jnp.asarray(n2n_ispl, dtype=jnp.int32),
        "n2l": jnp.asarray(n2l, dtype=jnp.int32),
        "n2l_ispl": jnp.asarray(n2l_ispl, dtype=jnp.int32),
        "n_nodes": int(n2n.size - n_lv),
        "nlevels": n_lv,
    }


def _traverse(pos, planes, *, r_max=0.15, boxsize=BOX, unique_pairs=True, cap=None):
    cap = cap or min(max(1024, 16 * planes["n_nodes"]), 1 << 22)
    prune_tol = 8.0 * np.finfo(np.float64).eps * boxsize
    work, counts = ffi.traverse(
        planes["n2n"],
        planes["n2n_ispl"],
        planes["n2l"],
        planes["n2l_ispl"],
        pos,
        work_cap=cap,
        self_query=True,
        unique_pairs=unique_pairs,
        r_max=r_max,
        prune_tol=prune_tol,
        boxsize=boxsize,
    )
    return work, counts, cap


def _forward(pos, work, cap, *, centers, r_max, temperature, mode="softmax",
             pair_scale=1.0, unique_pairs=True):
    n = pos.shape[0]
    perm = jnp.arange(n, dtype=jnp.int32)
    params = ffi.build_params(
        n_q=n,
        n_s=n,
        n_bins=int(centers.size),
        n_work=cap,
        self_query=True,
        triangular=unique_pairs,
        mode=mode,
        r_max=r_max,
        boxsize=BOX,
        temperature=temperature,
        eps=1e-60,
        r_safe=1e-12 * BOX,
        pair_scale=pair_scale,
    )
    w = jnp.ones(n, dtype=jnp.float64)
    return ffi.forward(
        pos, pos, jnp.asarray(centers), work, w, w, perm, params
    )


def _reference(pos_np, centers, r_max, temperature, *, mode="softmax",
               pair_convention="unique"):
    n = pos_np.shape[0]
    return reference_histogram(
        pos_np,
        pos_np,
        jnp.asarray(centers),
        temperature,
        BOX,
        r_max=r_max,
        weights_query=jnp.ones(n),
        weights_source=jnp.ones(n),
        self_query=True,
        pair_convention=pair_convention,
        mode=mode,
    )


@pytest.mark.parametrize("n", [17, 64, 257])
@pytest.mark.parametrize("r_max", [0.08, 0.2])
def test_traverse_forward_matches_reference(n, r_max):
    rng = np.random.default_rng(3 * n + int(r_max * 1000))
    pos_np = rng.uniform(0.0, BOX, size=(n, 3))
    pos = jnp.asarray(pos_np)
    centers = np.linspace(0.005, r_max * 0.95, 6)
    planes = grouped_planes(n)

    work, counts, cap = _traverse(pos, planes, r_max=r_max)
    counts_np = np.asarray(counts)
    assert counts_np[1] == 0
    assert counts_np[2] >= 1  # at least one level of C++-driven plane walk

    got = _forward(pos, work, cap, centers=centers, r_max=r_max,
                   temperature=0.05)
    want = _reference(pos_np, centers, r_max, 0.05)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=1e-10, atol=1e-12)


def test_traverse_gaussian_matches_reference():
    n, r_max = 120, 0.2
    rng = np.random.default_rng(11)
    pos_np = rng.uniform(0.0, BOX, size=(n, 3))
    pos = jnp.asarray(pos_np)
    centers = np.linspace(0.01, r_max * 0.9, 5)
    planes = grouped_planes(n)
    work, counts, cap = _traverse(pos, planes, r_max=r_max)
    assert int(np.asarray(counts)[1]) == 0
    got = _forward(pos, work, cap, centers=centers, r_max=r_max,
                   temperature=0.05, mode="gaussian")
    want = _reference(pos_np, centers, r_max, 0.05, mode="gaussian")
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=1e-10, atol=1e-12)


def test_traverse_directed_half_equals_unique():
    n, r_max = 96, 0.18
    rng = np.random.default_rng(5)
    pos = jnp.asarray(rng.uniform(0.0, BOX, size=(n, 3)))
    centers = np.linspace(0.005, r_max * 0.9, 4)
    planes = grouped_planes(n)

    w_unique, c_u, cap_u = _traverse(pos, planes, r_max=r_max, unique_pairs=True)
    w_dihalf, c_d, cap_d = _traverse(pos, planes, r_max=r_max, unique_pairs=False)
    # An ordered walk mirrors every non-diagonal leaf pair, so it needs more
    # interaction rectangles than the unordered walk (diagonal rectangles are
    # shared and enumerate both orientations inside the kernel).
    assert int(np.asarray(c_d)[0]) > int(np.asarray(c_u)[0])

    got_u = _forward(pos, w_unique, cap_u, centers=centers, r_max=r_max,
                     temperature=0.05)
    got_d = _forward(pos, w_dihalf, cap_d, centers=centers, r_max=r_max,
                     temperature=0.05, pair_scale=0.5, unique_pairs=False)
    np.testing.assert_allclose(np.asarray(got_d), np.asarray(got_u),
                               rtol=1e-12, atol=0)


def test_traverse_pads_tail_with_sentinels():
    n, r_max = 40, 0.15
    rng = np.random.default_rng(2)
    pos = jnp.asarray(rng.uniform(0.0, BOX, size=(n, 3)))
    planes = grouped_planes(n)
    work, counts, cap = _traverse(pos, planes, r_max=r_max)
    n_work = int(np.asarray(counts)[0])
    assert 0 < n_work < cap
    tail = np.asarray(work)[n_work:]
    assert tail.size > 0
    assert (tail == -1).all()


def test_traverse_is_deterministic():
    n, r_max = 150, 0.12
    rng = np.random.default_rng(9)
    pos = jnp.asarray(rng.uniform(0.0, BOX, size=(n, 3)))
    planes = grouped_planes(n)
    a, _, _ = _traverse(pos, planes, r_max=r_max)
    b, _, _ = _traverse(pos, planes, r_max=r_max)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_traverse_flags_overflow_instead_of_truncating():
    n, r_max = 120, 0.25  # large radius: many interactions
    rng = np.random.default_rng(13)
    pos = jnp.asarray(rng.uniform(0.0, BOX, size=(n, 3)))
    planes = grouped_planes(n)
    work, counts, _ = _traverse(pos, planes, r_max=r_max, cap=4)
    assert int(np.asarray(counts)[1]) == 1
    assert int(np.asarray(counts)[0]) == 0


def test_traverse_tiny_radius_yields_empty_histogram():
    """Pruning is conservative, so rectangles may survive a tiny r_max; what
    must stay exact is the histogram they produce (identically zero here)."""
    n = 60
    rng = np.random.default_rng(17)
    pos_np = rng.uniform(0.0, BOX, size=(n, 3))
    pos = jnp.asarray(pos_np)
    centers = np.linspace(0.0001, 1e-6, 3)
    planes = grouped_planes(n)
    work, counts, cap = _traverse(pos, planes, r_max=1e-6)
    got = _forward(pos, work, cap, centers=centers, r_max=1e-6, temperature=1e-3)
    np.testing.assert_array_equal(np.asarray(got), np.zeros(centers.size))


def test_traverse_rejects_inconsistent_planes():
    n = 33
    rng = np.random.default_rng(19)
    pos = jnp.asarray(rng.uniform(0.0, BOX, size=(n, 3)))
    planes = grouped_planes(n)
    bad = planes["n2n_ispl"][:-1]
    with pytest.raises(Exception):
        ffi.traverse(
            planes["n2n"],
            bad,
            planes["n2l"],
            planes["n2l_ispl"],
            pos,
            work_cap=1024,
            self_query=True,
            unique_pairs=True,
            r_max=0.15,
            prune_tol=0.0,
            boxsize=BOX,
        )
