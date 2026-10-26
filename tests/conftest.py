"""Shared pytest fixtures and JAX configuration."""

from __future__ import annotations

import jax

# PLAN.md requires float64 by default; JAX disables x64 unless asked.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

SEEDS = [0, 1, 7, 42, 1234]


def uniform_points(n, boxsize, seed, dtype=jnp.float64):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.0, boxsize, size=(n, 3)), dtype=dtype)


def clustered_points(n, boxsize, seed, dtype=jnp.float64):
    """Cosmology-like clusters: a few lognormal blobs."""
    rng = np.random.default_rng(seed)
    n_clusters = max(1, n // 64)
    centers = rng.uniform(0.0, boxsize, size=(n_clusters, 3))
    labels = rng.integers(0, n_clusters, size=n)
    offsets = rng.normal(scale=0.01 * boxsize, size=(n, 3))
    pts = centers[labels] + offsets
    return jnp.asarray(np.mod(pts, boxsize), dtype=dtype)


def boundary_points(n, boxsize, seed, dtype=jnp.float64):
    """Concentrated near the 0/L faces to exercise the seam."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 0.02 * boxsize, size=(n, 3))
    flip = rng.random(size=(n, 3)) < 0.5
    pts = np.where(flip, pts, boxsize - pts)
    return jnp.asarray(np.mod(pts, boxsize), dtype=dtype)


DISTRIBUTIONS = {
    "uniform": uniform_points,
    "clustered": clustered_points,
    "boundary": boundary_points,
}


@pytest.fixture(params=SEEDS)
def seed(request):
    return request.param


@pytest.fixture(params=sorted(DISTRIBUTIONS))
def distribution(request):
    return DISTRIBUTIONS[request.param]


@pytest.fixture(scope="session")
def ffi_cpu():
    """Build (once) and load the CPU mirror of the CUDA FFI kernels."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    lib = root / "jzradius" / "_libs" / "libjzradius_cpu.so"
    if not lib.exists():
        proc = subprocess.run(
            [sys.executable, str(root / "scripts" / "build_ext.py"), "--no-cuda"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not lib.exists():
            pytest.skip(
                "CPU FFI library unavailable: "
                + (proc.stderr.strip().splitlines() or ["no g++"])[-1]
            )
    from jzradius import ffi

    if not ffi.ffi_available("cpu"):
        ffi._load()
    if not ffi.ffi_available("cpu"):
        pytest.skip("CPU FFI library could not be loaded")
    return lib
