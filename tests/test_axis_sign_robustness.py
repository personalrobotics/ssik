"""Axis-sign (gauge) robustness of the analytic solvers.

A revolute joint's axis direction is a gauge freedom: flipping ``a -> -a`` and
negating the joint angle ``q -> -q`` leaves FK unchanged, because
``R(-a, q) == R(a, -q)``. A URDF author is free to point any joint axis either
way, so an authored ``-a`` must solve exactly as well as ``+a``.

Every solver must therefore be *sign-robust*: for any single joint whose axis is
negated (a distinct-but-valid arm), ``solve(fk(q))`` must still round-trip to
machine precision. A parallel/intersecting predicate treats ``±a`` as parallel/
concurrent, so a negated axis re-dispatches to the *same* solver -- the solver
cannot lean on an axis-sign it never enforced.

``ikgeo.three_parallel`` violated this: it collapses the parallel trio (joints
1, 2, 3) onto ``axes[1]`` and uses the signed total ``theta14 = q1+q2+q3+q4``, so
a trio joint authored anti-parallel to ``axes[1]`` produced candidates in the
wrong sign convention and FK-failed (Standard Bots core/spark, 0 solutions). The
fix normalizes each anti-parallel trio joint back to its physical convention.
The wider audit (SRS, spherical_two_parallel/intersecting, general_6r,
spherical_shoulder 7R) found every other family already robust; these tests pin
the property so it cannot silently regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik._kinbody import JointSpec, build_kinbody
from ssik.core.dispatcher import dispatch
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.manipulator import Manipulator

_Z = np.array([0.0, 0.0, 1.0])
_Y = np.array([0.0, 1.0, 0.0])
_X = np.array([1.0, 0.0, 0.0])


def _trans(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


# A UR-style 6R: base ``z``, then the parallel shoulder/elbow/wrist-1 trio
# (joints 1, 2, 3 about ``y``), then wrist-2 ``z`` and wrist-3 ``y`` --
# the ``ikgeo.three_parallel`` class.
_THREE_PARALLEL = [
    JointSpec(parent_link_T=_trans(0, 0, 0.1), axis=_Z, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0, 0.1, 0.1), axis=_Y, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0.4, 0, 0), axis=_Y, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0.4, 0, 0), axis=_Y, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0, 0.1, 0), axis=_Z, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0, 0, 0.1), axis=_Y, joint_type="revolute"),
]

# An anthropomorphic arm with a parallel shoulder/elbow pair and a spherical
# wrist -- the ``ikgeo.spherical_two_parallel`` class (mirrors the #377 test).
_TWO_PARALLEL = [
    JointSpec(parent_link_T=_trans(0, 0, 0), axis=_Z, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0, 0, 0.5), axis=_Y, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0.5, 0, 0), axis=_Y, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0.5, 0, 0), axis=_X, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0, 0, 0), axis=_Y, joint_type="revolute"),
    JointSpec(parent_link_T=_trans(0, 0, 0), axis=_X, joint_type="revolute"),
]


def _flip_axis(specs: list[JointSpec], i: int) -> list[JointSpec]:
    """Return a copy of ``specs`` with joint ``i``'s axis negated."""
    return [
        JointSpec(
            parent_link_T=s.parent_link_T,
            axis=(-s.axis if j == i else s.axis),
            joint_type=s.joint_type,
        )
        for j, s in enumerate(specs)
    ]


def _worst_roundtrip(kb, seed: int, n: int = 100) -> tuple[int, float]:
    """``solve(fk(q))`` over ``n`` random poses; return (solved, worst FK error)."""
    m = Manipulator(kb)
    rng = np.random.default_rng(seed)
    solved = 0
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(-2.0, 2.0, size=len(kb.joints))
        t = poe_forward_kinematics(kb, q)
        sols = m.solve(t, respect_limits=False)
        if sols:
            solved += 1
            worst = max(
                worst,
                min(float(np.max(np.abs(poe_forward_kinematics(kb, s.q) - t))) for s in sols),
            )
    return solved, worst


@pytest.mark.parametrize("flip", range(6))
def test_three_parallel_is_sign_robust(flip: int) -> None:
    """Negating *any* joint axis (notably a parallel-trio joint 1/2/3, the bug
    case) keeps the arm on ``three_parallel`` and solving at machine precision."""
    kb = build_kinbody(_flip_axis(_THREE_PARALLEL, flip))
    assert dispatch(kb).solver_name == "ikgeo.three_parallel"
    solved, worst = _worst_roundtrip(kb, seed=100 + flip)
    assert solved == 100, f"flip j{flip}: only solved {solved}/100"
    assert worst < 1e-9, f"flip j{flip}: worst FK closure {worst:.2e}"


def test_three_parallel_anti_parallel_trio_regression() -> None:
    """The exact failure mode: joints 2 and 3 authored anti-parallel to the trio
    reference axis (Standard Bots core/spark). Pre-fix this returned 0 solutions;
    it must now round-trip at machine precision."""
    specs = _flip_axis(_flip_axis(_THREE_PARALLEL, 2), 3)
    kb = build_kinbody(specs)
    assert dispatch(kb).solver_name == "ikgeo.three_parallel"
    solved, worst = _worst_roundtrip(kb, seed=7)
    assert solved == 100, f"anti-parallel trio: only solved {solved}/100"
    assert worst < 1e-9, f"anti-parallel trio: worst FK closure {worst:.2e}"


@pytest.mark.parametrize("flip", range(6))
def test_spherical_two_parallel_is_sign_robust(flip: int) -> None:
    """A representative already-robust family stays robust under any axis flip --
    guards against a future same-sign assumption creeping into the SP-per-joint
    solvers (the audit found these use actual per-joint axes, not a signed sum)."""
    kb = build_kinbody(_flip_axis(_TWO_PARALLEL, flip))
    assert dispatch(kb).solver_name == "ikgeo.spherical_two_parallel"
    solved, worst = _worst_roundtrip(kb, seed=200 + flip)
    assert solved == 100, f"flip j{flip}: only solved {solved}/100"
    assert worst < 1e-9, f"flip j{flip}: worst FK closure {worst:.2e}"
