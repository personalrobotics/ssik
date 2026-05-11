"""Unit tests for :mod:`ssik.postprocess`.

Operates on synthetic ``Solution`` lists -- no IK round-trip needed.
Each function is tested for its core contract plus edge cases:

* ``respect_limits``: drops out-of-range, keeps in-range, ignores
  ``limits=None`` joints, preserves input order.
* ``wrap_to_limits``: wraps ``q ± 2*pi`` when that lands in range,
  leaves alone if neither wrap fits, no-op for prismatic and
  ``limits=None`` joints, preserves order, preserves other ``Solution``
  fields.
* ``nearest_to_seed``: sorts by wrap-to-pi distance under ``wrap_l2``
  / ``wrap_linf``; stable on ties.
* ``max_solutions``: truncates correctly including edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik._kinbody import JointSpec, KinBody, build_kinbody
from ssik.core.solution import Solution
from ssik.postprocess import (
    max_solutions,
    nearest_to_seed,
    respect_limits,
    wrap_to_limits,
)


def _kb_with_limits(limits: list[tuple[float, float] | None]) -> KinBody:
    """Build a small KinBody with one joint per entry in ``limits``."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    specs = [
        JointSpec(
            parent_link_T=np.eye(4),
            axis=z_axis,
            joint_type="revolute",
            limits=lim,
        )
        for lim in limits
    ]
    return build_kinbody(specs)


def _sol(q: list[float]) -> Solution:
    return Solution(
        q=np.asarray(q, dtype=np.float64),
        fk_residual=0.0,
    )


# ---------------------------------------------------------------------------
# respect_limits
# ---------------------------------------------------------------------------


def test_respect_limits_keeps_in_range() -> None:
    kb = _kb_with_limits([(-1.0, 1.0), (-2.0, 2.0)])
    sols = [_sol([0.5, 1.5]), _sol([-0.9, -1.9])]
    out = respect_limits(sols, kb)
    assert len(out) == 2


def test_respect_limits_drops_out_of_range() -> None:
    kb = _kb_with_limits([(-1.0, 1.0), (-2.0, 2.0)])
    sols = [_sol([0.5, 1.5]), _sol([1.5, 0.0])]  # second violates joint 0
    out = respect_limits(sols, kb)
    assert len(out) == 1
    assert out[0].q[0] == 0.5


def test_respect_limits_ignores_none_joints() -> None:
    """``limits=None`` (continuous) joints don't constrain anything."""
    kb = _kb_with_limits([None, (-1.0, 1.0)])
    sols = [_sol([100.0, 0.5]), _sol([-100.0, 1.5])]
    out = respect_limits(sols, kb)
    # First survives (100.0 unconstrained, 0.5 in range);
    # second drops (1.5 out of range on joint 1).
    assert len(out) == 1
    assert out[0].q[0] == 100.0


def test_respect_limits_boundary_inclusive() -> None:
    kb = _kb_with_limits([(-1.0, 1.0)])
    sols = [_sol([-1.0]), _sol([1.0])]
    out = respect_limits(sols, kb)
    assert len(out) == 2  # both boundaries accepted


def test_respect_limits_preserves_order() -> None:
    kb = _kb_with_limits([(-1.0, 1.0)])
    sols = [_sol([0.1]), _sol([0.2]), _sol([0.3])]
    out = respect_limits(sols, kb)
    assert [float(s.q[0]) for s in out] == pytest.approx([0.1, 0.2, 0.3])


def test_respect_limits_rejects_wrong_q_length() -> None:
    kb = _kb_with_limits([(-1.0, 1.0)])
    bad = _sol([0.0, 0.0])  # 2-vector for 1-DOF kb
    with pytest.raises(ValueError, match="q-length"):
        respect_limits([bad], kb)


# ---------------------------------------------------------------------------
# wrap_to_limits
# ---------------------------------------------------------------------------


def test_wrap_to_limits_brings_q_into_range() -> None:
    """q=4.0 with limits [-pi, pi] wraps to ~ -2.283 which IS in range."""
    kb = _kb_with_limits([(-np.pi, np.pi)])
    sols = [_sol([4.0])]
    out = wrap_to_limits(sols, kb)
    assert len(out) == 1
    assert -np.pi <= out[0].q[0] <= np.pi
    # 4.0 - 2*pi ≈ -2.2832
    assert abs(out[0].q[0] - (4.0 - 2 * np.pi)) < 1e-12


def test_wrap_to_limits_leaves_in_range_alone() -> None:
    kb = _kb_with_limits([(-np.pi, np.pi)])
    sols = [_sol([1.5])]
    out = wrap_to_limits(sols, kb)
    assert out[0].q[0] == 1.5  # unchanged


def test_wrap_to_limits_no_wrap_fits() -> None:
    """q=3.0 with limits [0.5, 1.0] -- no integer 2*pi wrap lands in
    that narrow range, so q stays at 3.0 (later filtered by respect_limits)."""
    kb = _kb_with_limits([(0.5, 1.0)])
    sols = [_sol([3.0])]
    out = wrap_to_limits(sols, kb)
    assert out[0].q[0] == 3.0  # unchanged; respect_limits would drop it


def test_wrap_to_limits_skips_none_joints() -> None:
    """Continuous joints (limits=None) are never wrapped."""
    kb = _kb_with_limits([None])
    sols = [_sol([10.0])]
    out = wrap_to_limits(sols, kb)
    assert out[0].q[0] == 10.0


def test_wrap_to_limits_skips_prismatic() -> None:
    """Prismatic joints have no rotational periodicity; never wrap."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    specs = [
        JointSpec(
            parent_link_T=np.eye(4),
            axis=z_axis,
            joint_type="prismatic",
            limits=(-1.0, 1.0),
        )
    ]
    kb = build_kinbody(specs)
    sols = [_sol([5.0])]  # out of range, but never wrapped
    out = wrap_to_limits(sols, kb)
    assert out[0].q[0] == 5.0


def test_wrap_to_limits_preserves_other_fields() -> None:
    kb = _kb_with_limits([(-np.pi, np.pi)])
    original = Solution(
        q=np.array([4.0]),
        fk_residual=1.5e-9,
        refinement_used="lm",
    )
    out = wrap_to_limits([original], kb)
    wrapped = out[0]
    assert wrapped.fk_residual == 1.5e-9
    assert wrapped.refinement_used == "lm"


def test_wrap_to_limits_prefers_smallest_k() -> None:
    """When multiple wraps land in range, pick smallest |k| (closest to original)."""
    # If limits are very wide, both q+2*pi and q-2*pi could fit. The smallest-k
    # wrap is q itself (k=0), so an in-range value should never get wrapped.
    kb = _kb_with_limits([(-3 * np.pi, 3 * np.pi)])
    sols = [_sol([0.5])]
    out = wrap_to_limits(sols, kb)
    assert out[0].q[0] == 0.5


# ---------------------------------------------------------------------------
# nearest_to_seed
# ---------------------------------------------------------------------------


def test_nearest_to_seed_sorts_l2() -> None:
    sols = [_sol([0.0, 0.0]), _sol([0.5, 0.5]), _sol([0.1, 0.1])]
    seed = np.array([0.0, 0.0])
    out = nearest_to_seed(sols, seed)
    # Order: [0,0] (dist 0), [0.1,0.1] (~0.14), [0.5,0.5] (~0.71)
    assert [float(s.q[0]) for s in out] == pytest.approx([0.0, 0.1, 0.5])


def test_nearest_to_seed_uses_wrap_metric() -> None:
    """q=3.0 and seed=-3.0 are wrap-distance ~0.28, NOT 6.0."""
    sols = [_sol([3.0]), _sol([0.5])]
    seed = np.array([-3.0])
    out = nearest_to_seed(sols, seed)
    # Wrap dist for 3.0 vs -3.0: wrap(6.0) = 6.0 - 2*pi ≈ -0.28, |...| = 0.28
    # Wrap dist for 0.5 vs -3.0: 3.5 (no wrap needed)
    # So 3.0 wins despite "looking" farther on raw subtraction.
    assert out[0].q[0] == 3.0
    assert out[1].q[0] == 0.5


def test_nearest_to_seed_linf_metric() -> None:
    sols = [_sol([0.5, 0.0]), _sol([0.3, 0.3])]
    seed = np.array([0.0, 0.0])
    out = nearest_to_seed(sols, seed, metric="wrap_linf")
    # linf: [0.5, 0] -> max(0.5, 0) = 0.5; [0.3, 0.3] -> 0.3
    # So [0.3, 0.3] should come first.
    assert out[0].q[0] == 0.3
    assert out[1].q[0] == 0.5


def test_nearest_to_seed_stable_sort() -> None:
    # Two solutions at same distance -- stable sort preserves input order.
    sols = [_sol([1.0, 0.0]), _sol([-1.0, 0.0])]
    seed = np.array([0.0, 0.0])
    out = nearest_to_seed(sols, seed)
    # Both at distance 1.0; first one stays first.
    assert out[0].q[0] == 1.0
    assert out[1].q[0] == -1.0


def test_nearest_to_seed_unknown_metric_raises() -> None:
    sols = [_sol([0.0])]
    seed = np.array([0.0])
    with pytest.raises(ValueError, match="unknown metric"):
        nearest_to_seed(sols, seed, metric="manhattan")


# ---------------------------------------------------------------------------
# max_solutions
# ---------------------------------------------------------------------------


def test_max_solutions_truncates() -> None:
    sols = [_sol([float(i)]) for i in range(5)]
    out = max_solutions(sols, k=3)
    assert len(out) == 3
    assert [float(s.q[0]) for s in out] == [0.0, 1.0, 2.0]


def test_max_solutions_k_larger_than_input() -> None:
    sols = [_sol([0.0]), _sol([1.0])]
    out = max_solutions(sols, k=10)
    assert len(out) == 2


def test_max_solutions_k_zero() -> None:
    sols = [_sol([0.0]), _sol([1.0])]
    out = max_solutions(sols, k=0)
    assert out == []


def test_max_solutions_k_negative() -> None:
    sols = [_sol([0.0]), _sol([1.0])]
    out = max_solutions(sols, k=-3)
    assert out == []


# ---------------------------------------------------------------------------
# Pipeline composition (end-to-end recipe)
# ---------------------------------------------------------------------------


def test_pipeline_compose_franka_recipe() -> None:
    """The canonical pipeline: wrap into range -> drop violations ->
    sort by distance -> top-k. Verifies the composition produces a
    sensible output without any individual step swallowing the others."""
    kb = _kb_with_limits([(-np.pi, np.pi), (-1.0, 1.0)])
    sols = [
        _sol([4.0, 0.0]),  # joint 0 wraps to ~-2.28 which is in range; joint 1 ok
        _sol([0.5, 0.5]),  # both in range
        _sol([0.0, 1.5]),  # joint 1 out of range -> drops
        _sol([0.1, 0.1]),  # both in range
        _sol([10.0, 0.0]),  # joint 0 wraps to ~3.72 (out of [-pi, pi]); drops
    ]
    seed = np.array([0.0, 0.0])

    sols = wrap_to_limits(sols, kb)
    sols = respect_limits(sols, kb)
    sols = nearest_to_seed(sols, seed)
    sols = max_solutions(sols, k=2)

    assert len(sols) == 2
    # Closest to seed should come first.
    assert sols[0].q[0] == pytest.approx(0.1)
    assert sols[0].q[1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# End-to-end: real IK output -> postprocess pipeline
# ---------------------------------------------------------------------------


def test_franka_pipeline_real_ik_output() -> None:
    """Run real Franka 7R IK, then put the output through the canonical
    pipeline. Verifies the postprocess helpers compose with actual solver
    output (not just synthetic Solutions).

    Acceptance: after `wrap_to_limits + respect_limits`, every surviving
    solution lies inside every joint's range. After `nearest_to_seed`, the
    first one is the closest to ``q_current``. After `max_solutions(k=1)`,
    we have exactly one.
    """
    import sys
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    sys.path.insert(0, str(fixtures))
    from franka_panda import franka_panda_specs

    from ssik.kinematics._scalar3 import _mat4_mat4
    from ssik.solvers.jointlock.seven_r import solve as seven_r_solve
    from ssik.subproblems._rotation import rotation_matrix

    kb = build_kinbody(franka_panda_specs())

    # FK on a known q gives a reachable target.
    q_true = np.array([0.1, 0.2, -0.1, -1.5, 0.0, 1.7, -0.5], dtype=np.float64)
    T = np.eye(4)
    for j, qi in zip(kb.joints, q_true, strict=True):
        R = np.eye(4)
        R[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = _mat4_mat4(_mat4_mat4(T, _mat4_mat4(j.T_left, R)), j.T_right)

    sols, is_ls = seven_r_solve(kb, T)
    assert not is_ls
    assert len(sols) > 0

    # Run the canonical pipeline.
    sols = wrap_to_limits(sols, kb)
    sols = respect_limits(sols, kb)
    assert len(sols) > 0, "all Franka solutions filtered out by limits"

    # Every survivor strictly within every joint's limits.
    for sol in sols:
        for i, joint in enumerate(kb.joints):
            if joint.limits is None:
                continue
            lo, hi = joint.limits
            assert lo <= sol.q[i] <= hi, (
                f"joint {i} q={sol.q[i]} outside [{lo}, {hi}] -- respect_limits failed"
            )

    # Sort by distance to seed; q_true is the seed so it should rank first.
    sols = nearest_to_seed(sols, q_true)
    # The closest should be very close (sub-radian on every joint, l2 < 1.0).
    closest_diffs = [
        abs(((sols[0].q[i] - q_true[i]) + np.pi) % (2 * np.pi) - np.pi) for i in range(7)
    ]
    assert max(closest_diffs) < 0.5, (
        f"closest IK solution to seed has unexpectedly large diffs: {closest_diffs}"
    )

    # Truncate to top-1.
    sols = max_solutions(sols, k=1)
    assert len(sols) == 1
