"""Exceptions and warnings for :mod:`jzradius`."""

from __future__ import annotations


class JzRadiusError(RuntimeError):
    """Base class for jzradius runtime failures."""


class WorkQueueOverflow(JzRadiusError):
    """The interaction/work queue exceeded its hard capacity.

    This must never be swallowed: dropping interactions silently would turn an
    exact radius count into a truncated one.
    """


class TreeAdapterUnavailable(JzRadiusError):
    """Upstream ``jztree`` could not be used to build a :class:`RadiusTree`."""


class ScaleTierNotApproved(JzRadiusError):
    """A scale tier above ``integration`` was requested without approval."""


class ApproximateModeWarning(UserWarning):
    """A result relies on an approximation that is not exact radius counting."""


class RadiusConfigError(ValueError):
    """Invalid user configuration (box size, temperature, bin centers, ...)."""
