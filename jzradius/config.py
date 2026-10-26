"""Configuration objects and validation for :mod:`jzradius`."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import RadiusConfigError

_MODES = ("softmax", "gaussian")
_PAIR_CONVENTIONS = ("unique", "directed_half")
_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class RadiusHistogramConfig:
    """Static configuration for a radius histogram call.

    Fields match PLAN.md section 7.1.  ``temperature``, ``boxsize`` and
    ``r_max`` are kept here for convenience, but the public function also
    accepts them as dynamic arguments.
    """

    boxsize: float
    r_max: float | None = None
    temperature: float = 0.05
    mode: str = "softmax"
    self_query: bool = False
    pair_convention: str = "unique"
    leaf_size: int = 32
    dtype: str = "float64"
    deterministic: bool = False
    block_size: int = 256
    max_work_items: int = 1 << 22
    max_queue_growth_factor: int = 4

    def __post_init__(self) -> None:
        validate_config(self)


def validate_config(cfg: RadiusHistogramConfig) -> None:
    """Raise :class:`RadiusConfigError` on an invalid configuration."""
    _require_positive_finite("boxsize", cfg.boxsize)
    if cfg.mode not in _MODES:
        raise RadiusConfigError(f"mode must be one of {_MODES}, got {cfg.mode!r}")
    if cfg.pair_convention not in _PAIR_CONVENTIONS:
        raise RadiusConfigError(
            f"pair_convention must be one of {_PAIR_CONVENTIONS}, "
            f"got {cfg.pair_convention!r}"
        )
    if cfg.dtype not in _DTYPES:
        raise RadiusConfigError(f"dtype must be one of {_DTYPES}, got {cfg.dtype!r}")
    if cfg.leaf_size < 1:
        raise RadiusConfigError(f"leaf_size must be >= 1, got {cfg.leaf_size}")
    if cfg.block_size < 1:
        raise RadiusConfigError(f"block_size must be >= 1, got {cfg.block_size}")
    if cfg.max_work_items < 1:
        raise RadiusConfigError("max_work_items must be >= 1")
    if cfg.max_queue_growth_factor < 1:
        raise RadiusConfigError("max_queue_growth_factor must be >= 1")
    validate_temperature(cfg.temperature)
    if cfg.r_max is not None:
        validate_r_max(cfg.r_max, cfg.boxsize)


def validate_temperature(temperature: float) -> None:
    _require_positive_finite("temperature", temperature)
    # PLAN.md 6.2: reject temperatures that make alpha = 1/T^2 overflow.
    if temperature < 1e-6:
        raise RadiusConfigError(
            f"temperature must be >= 1e-6 for numerical safety, got {temperature}"
        )


def validate_r_max(r_max: float, boxsize: float) -> None:
    _require_positive_finite("r_max", r_max)
    if r_max >= boxsize / 2:
        raise RadiusConfigError(
            f"r_max={r_max} must be < boxsize/2={boxsize / 2}; "
            "v1 assumes the single-image regime"
        )


def validate_bin_centers(bin_centers) -> np.ndarray:
    """Validate and return bin centers as a 1-D float64 numpy array."""
    centers = np.asarray(bin_centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 1:
        raise RadiusConfigError(
            f"bin_centers must be a non-empty 1-D array, got shape {centers.shape}"
        )
    if not np.all(np.isfinite(centers)):
        raise RadiusConfigError("bin_centers must be finite")
    if centers.size > 1 and not np.all(np.diff(centers) > 0):
        raise RadiusConfigError("bin_centers must be strictly increasing")
    if np.any(centers < 0):
        raise RadiusConfigError("bin_centers must be non-negative")
    return centers


def default_r_max(
    bin_centers, temperature: float, median_spacing: float | None = None
) -> float:
    """PLAN.md 7.2: ``max(c) + max(6*T, 0.1 * median_spacing)``.

    ``median_spacing`` is the median inter-particle spacing; when omitted (for
    backward compatibility) the median bin spacing is used instead.
    """
    centers = np.asarray(bin_centers, dtype=np.float64)
    if median_spacing is None:
        spacing = np.median(np.diff(centers)) if centers.size > 1 else 0.0
    else:
        spacing = float(median_spacing)
    return float(centers.max() + max(6.0 * temperature, 0.1 * spacing))


def r_safe(boxsize: float, dtype: str = "float64") -> float:
    """PLAN.md 6.6: minimum separation below which the pair is skipped."""
    if dtype == "float32":
        return max(1e-12 * boxsize, 1e-30)
    return max(1e-12 * boxsize, 1e-60)


def _require_positive_finite(name: str, value: float) -> None:
    try:
        ok = math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise RadiusConfigError(f"{name} must be a positive finite number, got {value!r}")
