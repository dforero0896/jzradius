"""jzradius: differentiable periodic radius histograms on GPU.

The public entry point is :func:`radius_softmax_histogram`: a pair histogram of
periodically wrapped separations with soft (softmax or Gaussian) bin weights,
plus its analytic VJP, backed by typed XLA-FFI kernels.

Also exported are the pure-JAX building blocks (:mod:`jzradius.math`,
:mod:`jzradius.reference`, :mod:`jzradius.bounds`), the tree adapter and the CPU
reference traversal, which are what the tests check the kernels against.
"""

from __future__ import annotations

from . import bounds, math, reference
from .config import (
    RadiusHistogramConfig,
    default_r_max,
    r_safe,
    validate_bin_centers,
    validate_config,
    validate_r_max,
    validate_temperature,
)
from .histogram import radius_softmax_histogram
from .errors import (
    ApproximateModeWarning,
    JzRadiusError,
    RadiusConfigError,
    ScaleTierNotApproved,
    TreeAdapterUnavailable,
    WorkQueueOverflow,
)
from .math import (
    dW_dr,
    gaussian_weights,
    histogram_vjp_r,
    logits,
    logits_derivative,
    periodic_delta,
    periodic_distance,
    periodic_distance_matrix,
    soft_weights,
    softmax_vjp_r,
    softmax_weights,
)
from .reference import reference_histogram, reference_pair_set, reference_pairs
from .traversal import TraversalResult, TraversalStats, traverse_tree_pairs
from .tree_adapter import (
    RadiusTree,
    build_naive_radius_tree,
    build_radius_tree_from_jztree,
    build_radius_tree_from_levels,
    check_tree_invariants,
)

__version__ = "1.1.0.dev0"

__all__ = [
    "ApproximateModeWarning",
    "JzRadiusError",
    "RadiusConfigError",
    "RadiusHistogramConfig",
    "RadiusTree",
    "ScaleTierNotApproved",
    "TraversalResult",
    "TraversalStats",
    "TreeAdapterUnavailable",
    "WorkQueueOverflow",
    "__version__",
    "bounds",
    "build_naive_radius_tree",
    "build_radius_tree_from_jztree",
    "build_radius_tree_from_levels",
    "check_tree_invariants",
    "dW_dr",
    "default_r_max",
    "gaussian_weights",
    "histogram_vjp_r",
    "logits",
    "logits_derivative",
    "math",
    "periodic_delta",
    "periodic_distance",
    "periodic_distance_matrix",
    "radius_softmax_histogram",
    "r_safe",
    "reference",
    "reference_histogram",
    "reference_pair_set",
    "reference_pairs",
    "soft_weights",
    "softmax_vjp_r",
    "softmax_weights",
    "traverse_tree_pairs",
    "validate_bin_centers",
    "validate_config",
    "validate_r_max",
    "validate_temperature",
]
