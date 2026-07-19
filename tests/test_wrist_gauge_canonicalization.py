"""Gauge-invariant spherical-wrist handling (#377).

A spherical-wrist 6R whose last joint's URDF frame sits a fixed distance *along
its own rotation axis* from the wrist-center intersection (the flange offset --
ABB IRB 6700 and other standard industrial arms) was mis-classified: the strict
``three_consecutive_intersecting`` predicate requires the wrist origins to
coincide with the intersection, so dispatch fell back to the ~100x-slower
``ikgeo.general_6r``.

The fix is gauge-invariant and scoped to the spherical-solve path -- no global
representation change:

* Dispatch routes on :func:`ssik.kinematics.predicates.axes_meet_at_common_point`
  -- the true Pieper condition (wrist axes concurrent), independent of origin
  placement.
* :func:`ssik._kinbody.canonicalize_spherical_wrist` slides the along-axis offset
  onto the intersection (into the tool transform, an exact gauge move) at solver
  entry, on a copy.

This pins: the fix routes + solves at machine precision; canonicalization is
FK-identical and does not mutate its input; construction is untouched (so 7R
SRS/jointlock arms and baked artifacts are unaffected -- the iiwa #155
rejection still holds); and an *off-axis* offset (a genuinely non-spherical
wrist) is left alone.
"""

from __future__ import annotations

import numpy as np

from ssik._kinbody import JointSpec, build_kinbody, canonicalize_spherical_wrist
from ssik.core.dispatcher import dispatch
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import three_consecutive_intersecting
from ssik.solvers.ikgeo import spherical_two_parallel

_Z = np.array([0.0, 0.0, 1.0])
_Y = np.array([0.0, 1.0, 0.0])
_X = np.array([1.0, 0.0, 0.0])


def _trans(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _spherical_wrist_6r(flange: np.ndarray) -> list[JointSpec]:
    """Anthropomorphic spherical-wrist 6R: shoulder/elbow axes parallel (the
    ``spherical_two_parallel`` class) and wrist axes (3, 4, 5) concurrent at one
    point. ``flange`` is the last joint's frame offset from that point -- along
    +x it is an along-axis gauge offset; along +z it is off-axis.
    """
    return [
        JointSpec(parent_link_T=_trans(0, 0, 0), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0, 0.5), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0.5, 0, 0), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0.5, 0, 0), axis=_X, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0, 0), axis=_Y, joint_type="revolute"),
        JointSpec(
            parent_link_T=_trans(float(flange[0]), float(flange[1]), float(flange[2])),
            axis=_X,
            joint_type="revolute",
        ),
    ]


def test_along_axis_flange_routes_to_spherical_and_solves() -> None:
    """The along-axis flange offset routes to the Tier-0 spherical solver and
    solves at machine precision (#377)."""
    kb = build_kinbody(_spherical_wrist_6r(np.array([0.2, 0.0, 0.0])))
    plan = dispatch(kb)
    assert plan.solver_name == "ikgeo.spherical_two_parallel", plan.solver_name
    assert plan.tier == 0

    rng = np.random.default_rng(0)
    worst = 0.0
    solved = 0
    for _ in range(200):
        q = rng.uniform(-2.0, 2.0, size=6)
        t = poe_forward_kinematics(kb, q)
        sols, _ = spherical_two_parallel.solve(kb, t)
        if sols:
            solved += 1
            worst = max(
                worst, min(float(np.max(np.abs(poe_forward_kinematics(kb, s.q) - t))) for s in sols)
            )
    assert solved == 200, f"only solved {solved}/200"
    assert worst < 1e-9, f"worst FK closure {worst:.2e}"


def test_off_axis_flange_stays_general_6r() -> None:
    """An *off-axis* last-joint offset is a genuinely non-spherical wrist (the
    axis misses the intersection): it is NOT routed to the spherical solver."""
    kb = build_kinbody(_spherical_wrist_6r(np.array([0.0, 0.0, 0.2])))
    assert dispatch(kb).solver_name == "ikgeo.general_6r"


def test_canonicalize_is_fk_identical_and_non_mutating() -> None:
    """``canonicalize_spherical_wrist`` returns an FK-identical copy and leaves
    the input untouched -- gauge-invariance without side effects."""
    kb = build_kinbody(_spherical_wrist_6r(np.array([0.2, 0.0, 0.0])))
    before_left = kb.joints[-1].T_left[:3, 3].copy()
    before_right = kb.joints[-1].T_right[:3, 3].copy()

    kb_canon = canonicalize_spherical_wrist(kb)

    # Input unchanged.
    assert np.array_equal(kb.joints[-1].T_left[:3, 3], before_left)
    assert np.array_equal(kb.joints[-1].T_right[:3, 3], before_right)
    # Copy is re-gauged: the 0.2 moved from the joint frame into the tool.
    assert not np.allclose(kb_canon.joints[-1].T_left[:3, 3], before_left)
    # FK identical at random configs.
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.uniform(-3.0, 3.0, size=6)
        assert (
            np.max(np.abs(poe_forward_kinematics(kb, q) - poe_forward_kinematics(kb_canon, q)))
            < 1e-12
        )


def test_construction_does_not_gauge_the_wrist() -> None:
    """Building the KinBody must NOT canonicalize -- gauge-invariance lives in the
    solver path only, so nothing outside a spherical solve sees a changed
    representation (this is what keeps 7R artifacts + the #155 contract intact)."""
    flange = np.array([0.2, 0.0, 0.0])
    kb = build_kinbody(_spherical_wrist_6r(flange))
    # The flange still sits in the last joint's T_left (as authored), not folded
    # into the tool transform.
    assert np.allclose(kb.joints[-1].T_left[:3, 3], flange)
    assert np.allclose(kb.joints[-1].T_right[:3, 3], np.zeros(3))
    # And the strict predicate still rejects the un-gauged wrist.
    assert three_consecutive_intersecting(kb.joints) is None


def test_canonicalize_noop_on_canonical_and_nonspherical() -> None:
    """No-op (returns the same object) when the wrist is already canonical or
    the chain has no spherical wrist -- safe to call unconditionally."""
    canonical = build_kinbody(_spherical_wrist_6r(np.array([0.0, 0.0, 0.0])))
    assert canonicalize_spherical_wrist(canonical) is canonical

    off_axis = build_kinbody(_spherical_wrist_6r(np.array([0.0, 0.0, 0.2])))
    assert canonicalize_spherical_wrist(off_axis) is off_axis


def test_seven_r_untouched_by_construction() -> None:
    """A 7R chain with an along-axis wrist offset is left exactly as built and is
    not routed to a 6R spherical solver -- the SRS/jointlock path owns it."""
    specs = [
        JointSpec(parent_link_T=_trans(0, 0, 0.2 * i), axis=_Z, joint_type="revolute")
        for i in range(7)
    ]
    kb = build_kinbody(specs)
    assert np.allclose(kb.joints[-1].T_right[:3, 3], np.zeros(3))
