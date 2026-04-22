"""Structural description of a kinematic chain.

:class:`TopologyReport` is the output of :func:`describe_topology`; it captures
which canonical kinematic-family conditions hold on a POE-normalized chain.
The dispatcher (Phase C) will consume this to pick a closed-form solver;
in this phase it's also usable as a standalone "what kind of arm is this?"
diagnostic.

The report is deliberately minimal right now: boolean/location facts about
the two Pieper-class conditions the dispatcher cares about first (three
consecutive intersecting axes, three consecutive parallel axes). Near-miss
diagnostics (distance-to-predicate for each triple, for user-visible
"your chain is 3e-6 away from three-parallel" error messages) will land
alongside the dispatcher in Phase C where they have a concrete consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.predicates import (
    three_consecutive_intersecting,
    three_consecutive_parallel,
)

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from ssik._kinbody import KinBody

__all__ = ["TopologyReport", "describe_topology"]


@dataclass(frozen=True, kw_only=True)
class TopologyReport:
    """Structural description of a POE-normalized kinematic chain.

    Attributes:
        dof: number of active joints in the chain.
        three_consecutive_intersecting: if any triple of consecutive joint
            axes share a common point, the ``(i, i+1, i+2)`` indices of
            the first such triple; otherwise ``None``. This is the
            Pieper spherical-wrist / spherical-shoulder condition.
        three_consecutive_parallel: if any triple of consecutive joint axes
            are all pairwise parallel, the ``(i, i+1, i+2)`` indices of the
            first such triple; otherwise ``None``. This is the three-parallel
            condition (UR-class arms).
    """

    dof: int
    three_consecutive_intersecting: tuple[int, int, int] | None
    three_consecutive_parallel: tuple[int, int, int] | None


def describe_topology(
    kb: KinBody,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> TopologyReport:
    """Classify the kinematic structure of a POE-normalized chain.

    :param kb: a :class:`KinBody` returned by
        :func:`ssik._urdf.load_urdf_kinbody_normalized`. Passing a
        non-normalized KinBody yields undefined results.
    :param policy: tolerances for the axis-parallel / axis-intersect
        predicates. Defaults to :data:`ssik.core.tolerances.DEFAULT_TOLERANCE_POLICY`.
    """
    return TopologyReport(
        dof=kb.GetDOF(),
        three_consecutive_intersecting=three_consecutive_intersecting(kb.joints, policy),
        three_consecutive_parallel=three_consecutive_parallel(kb.joints, policy),
    )
