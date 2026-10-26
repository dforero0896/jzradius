"""Typed XLA FFI bindings for the jzradius kernels (PLAN.md 1.5, M5).

The compiled artifacts live in ``jzradius/_libs``:

* ``libjzradius_cpu.so``  (CPU mirror, used when no GPU is present),
* ``libjzradius_cuda.so`` (CUDA kernels, registered for platform ``"CUDA"``).

Both export the same three typed-FFI entry points:

``jzradius_forward``      histogram of accepted pairs
``jzradius_backward``     VJP wrt positions and pair weights
``jzradius_node_bounds``  periodic node-node distance lower bounds

The Python wrappers call ``jax.ffi.ffi_call`` directly.  They are *not*
differentiable by themselves; :mod:`jzradius.histogram` wraps them in
``jax.custom_vjp`` and supplies the residual arrays.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .errors import JzRadiusError

__all__ = [
    "active_platform",
    "build_params",
    "ffi_available",
    "forward",
    "backward",
    "node_bounds",
]

_LIB_DIR = Path(__file__).resolve().parent / "_libs"
_LIB_NAMES = {"cpu": "libjzradius_cpu.so", "CUDA": "libjzradius_cuda.so"}
_TARGETS = {
    "forward": "jzradius_forward",
    "backward": "jzradius_backward",
    "node_bounds": "jzradius_node_bounds",
}
_SYMBOLS = {
    "cpu": {
        "forward": "JzRadiusForwardCpu",
        "backward": "JzRadiusBackwardCpu",
        "node_bounds": "JzRadiusNodeBoundsCpu",
    },
    "CUDA": {
        "forward": "JzRadiusForward",
        "backward": "JzRadiusBackward",
        "node_bounds": "JzRadiusNodeBounds",
    },
}

_REGISTERED: Dict[str, ctypes.CDLL] = {}


class FfiUnavailable(JzRadiusError):
    """Raised when the compiled FFI library for the active platform is missing."""


def active_platform() -> str:
    """Return the FFI platform name matching the active JAX backend."""
    return "CUDA" if jax.default_backend() == "gpu" else "cpu"


def _load(platform: str) -> ctypes.CDLL:
    if platform in _REGISTERED:
        return _REGISTERED[platform]
    path = _LIB_DIR / _LIB_NAMES[platform]
    if not path.exists():
        hint = (
            "run `python scripts/build_ext.py`"
            if platform == "cpu"
            else "run `python scripts/build_ext.py` on a machine with nvcc"
        )
        raise FfiUnavailable(
            f"jzradius FFI library for platform {platform!r} not found at {path}; "
            f"{hint}"
        )
    lib = ctypes.CDLL(os.fspath(path))
    for kind, symbol in _SYMBOLS[platform].items():
        func = getattr(lib, symbol)
        func.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(
            _TARGETS[kind], jax.ffi.pycapsule(func), platform=platform
        )
    _REGISTERED[platform] = lib
    return lib


def ffi_available(platform: str | None = None) -> bool:
    """Return True when the FFI artifact for ``platform`` can be loaded."""
    platform = platform or active_platform()
    try:
        _load(platform)
        return True
    except (FfiUnavailable, OSError):
        return False


def build_params(
    *,
    n_q: int,
    n_s: int,
    n_bins: int,
    n_work: int,
    self_query: bool,
    triangular: bool,
    mode: str,
    r_max: float,
    boxsize: float,
    temperature: float,
    eps: float,
    r_safe: float,
    pair_scale: float,
) -> Dict[str, object]:
    """Build the attribute dictionary for the typed FFI calls."""
    if mode not in ("softmax", "gaussian"):
        raise JzRadiusError(f"unknown weight mode {mode!r}")
    if self_query and n_q != n_s:
        raise JzRadiusError("self_query requires equally sized query/source arrays")
    return {
        "self_query": 1 if self_query else 0,
        "unique_pairs": 1 if (self_query and triangular) else 0,
        "mode": 0 if mode == "softmax" else 1,
        "r_max": float(r_max),
        "boxsize": float(boxsize),
        "temperature": float(temperature),
        "eps": float(eps),
        "r_safe": float(r_safe),
        "pair_scale": float(pair_scale),
    }


def _attrs(params: Dict[str, object]) -> Dict[str, object]:
    return {
        "self_query": np.int32(params["self_query"]),
        "unique_pairs": np.int32(params["unique_pairs"]),
        "mode": np.int32(params["mode"]),
        "r_max": np.float64(params["r_max"]),
        "boxsize": np.float64(params["boxsize"]),
        "temperature": np.float64(params["temperature"]),
        "eps": np.float64(params["eps"]),
        "r_safe": np.float64(params["r_safe"]),
        "pair_scale": np.float64(params["pair_scale"]),
    }


def _check_inputs(qpos, spos, centers, work, wq, ws, orig=None) -> None:
    if qpos.dtype != spos.dtype or qpos.dtype != wq.dtype or qpos.dtype != ws.dtype:
        raise JzRadiusError("positions and weights must share one dtype")
    if qpos.dtype not in (jnp.float32, jnp.float64):
        raise JzRadiusError("only float32/float64 are supported")
    if qpos.ndim != 2 or qpos.shape[1] != 3:
        raise JzRadiusError(f"qpos must have shape (N, 3), got {qpos.shape}")
    if spos.ndim != 2 or spos.shape[1] != 3:
        raise JzRadiusError(f"spos must have shape (N, 3), got {spos.shape}")
    if work.ndim != 2 or work.shape[1] != 4 or work.dtype != jnp.int32:
        raise JzRadiusError(
            f"work must be int32 with shape (W, 4), got {work.shape}/{work.dtype}"
        )
    if centers.ndim != 1 or centers.dtype != jnp.float64:
        raise JzRadiusError("bin centers must be a float64 vector")
    if orig is not None:
        if orig.ndim != 1 or orig.dtype != jnp.int32:
            raise JzRadiusError("orig must be an int32 vector")
        if orig.shape[0] != qpos.shape[0]:
            raise JzRadiusError(
                f"orig must have one entry per query particle, got {orig.shape[0]}"
            )


def forward(
    qpos,
    spos,
    centers,
    work,
    wq,
    ws,
    orig,
    params: Dict[str, object],
    *,
    platform: str | None = None,
):
    """Histogram of accepted pairs via the typed FFI forward kernel.

    ``orig`` maps sorted positions back to caller indices; the kernel uses it
    only to orient unordered self-pairs by input index (D010).
    """
    platform = platform or active_platform()
    _load(platform)
    _check_inputs(qpos, spos, centers, work, wq, ws, orig)
    result = jax.ffi.ffi_call(
        _TARGETS["forward"],
        jax.ShapeDtypeStruct((centers.shape[0],), qpos.dtype),
        vmap_method="sequential",
    )(qpos, spos, centers, work, wq, ws, orig, **_attrs(params))
    return result


def backward(
    qpos,
    spos,
    centers,
    work,
    wq,
    ws,
    orig,
    cotangent,
    params: Dict[str, object],
    *,
    platform: str | None = None,
) -> Tuple[object, object, object, object]:
    """VJP wrt positions and pair weights (device-side analytic backward)."""
    platform = platform or active_platform()
    _load(platform)
    _check_inputs(qpos, spos, centers, work, wq, ws, orig)
    n_q, n_s = qpos.shape[0], spos.shape[0]
    result = jax.ffi.ffi_call(
        _TARGETS["backward"],
        (
            jax.ShapeDtypeStruct(qpos.shape, qpos.dtype),
            jax.ShapeDtypeStruct(spos.shape, spos.dtype),
            jax.ShapeDtypeStruct((n_q,), qpos.dtype),
            jax.ShapeDtypeStruct((n_s,), qpos.dtype),
        ),
        vmap_method="sequential",
    )(qpos, spos, centers, work, wq, ws, orig, cotangent, **_attrs(params))
    return tuple(result)


def node_bounds(
    qlo, qhi, slo, shi, boxsize: float, *, platform: str | None = None
):
    """Periodic node-node lower bounds (PLAN.md 10.3) evaluated on device."""
    platform = platform or active_platform()
    _load(platform)
    result = jax.ffi.ffi_call(
        _TARGETS["node_bounds"],
        jax.ShapeDtypeStruct((qlo.shape[0],), jnp.float64),
        vmap_method="sequential",
    )(qlo, qhi, slo, shi, boxsize=np.float64(boxsize))
    return result
