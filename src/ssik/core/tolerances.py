"""Numeric tolerances for kinematic-structure predicates.

Real-world URDFs produce axes and origins that are *nearly* but not exactly
aligned with canonical kinematic structure -- axes at ``(0, 0, 0.99999998)``
from Xacro arithmetic rounding, rpy matrices whose orthogonality check differs
from identity by 1e-12 after accumulation. The predicates that decide "are
these three axes intersecting?" or "parallel?" need explicit tolerances so
that behaviour is predictable and so downstream error messages can say "this
chain is 3e-6 away from three-consecutive-parallel; did you mean to classify
it that way?"

Every public predicate consumes a :class:`TolerancePolicy`. The default is
calibrated for metric-scale 6-DOF chains (link lengths < 2m); override for
millimeter-scale micro-arms or kilometer-scale astronomy mounts.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_TOLERANCE_POLICY", "TolerancePolicy"]


@dataclass(frozen=True)
class TolerancePolicy:
    """Tolerances for kinematic-structure predicates.

    Attributes:
        axis_parallel: cross-product magnitude below which two unit-vector
            axes are considered parallel (or anti-parallel). For unit
            vectors ``||a x b|| = sin(theta)`` so the default ``1e-8``
            accepts axes differing by up to ~1 microradian.
        axis_intersect: perpendicular distance below which two lines in
            3D are considered to intersect. Default matches the
            ``axis_parallel`` tolerance in spirit -- metric-scale chains.
    """

    axis_parallel: float = 1e-8
    axis_intersect: float = 1e-8


DEFAULT_TOLERANCE_POLICY = TolerancePolicy()
