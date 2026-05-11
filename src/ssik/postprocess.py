"""Composable post-processing filters for IK solutions.

ssik's analytical IK kernel returns the full geometric solution set --
every branch the math admits, regardless of joint limits, distance to a
preferred configuration, or any other application-level concern. This
module provides the building blocks the application layer composes on top.

Convention follows IKFast (OpenRAVE wrapper handles limits + collision +
nearest-to-seed in a separate Python layer above the generated kernel) and
EAIK (``mj_manipulator/franka.py`` wraps EAIK's pure IK with limit /
seed / collision logic on the Python side).

Each function takes ``list[Solution]`` (and any extra args) and returns
``list[Solution]`` -- pure transforms, easy to test, easy to compose.
The four shipping with v0.1 cover ~95% of real-world post-processing:

  * :func:`respect_limits` -- drop solutions outside any joint's range
  * :func:`wrap_to_limits` -- try ``q ± 2*pi`` per joint to bring solutions in
  * :func:`nearest_to_seed` -- sort by wrap-to-pi distance to a reference q
  * :func:`take_first` -- truncate to the first ``k``

Production pipeline pattern::

    from franka_panda_ik import _KB, solve
    from ssik.postprocess import (
        respect_limits, wrap_to_limits, nearest_to_seed, take_first,
    )

    sols = solve(T_target, respect_limits=False)
    sols = wrap_to_limits(sols, _KB)
    sols = respect_limits(sols, _KB)
    sols = nearest_to_seed(sols, q_current)
    sols = take_first(sols, k=4)

Out of scope for v0.1 (separate issues / future work):

  * Collision filtering (needs a collision backend such as FCL).
  * Trajectory-context filters (continuous q-trajectory with smoothness).
  * Reachability / dexterity scoring.

These can be follow-ups; the four above cover the common case and form a
self-contained module that compiles cleanly to a single shared ``.so``
in Phase 4 (no per-arm specialisation, no symbolic precompute).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution

__all__ = [
    "nearest_to_seed",
    "respect_limits",
    "take_first",
    "wrap_to_limits",
]


def respect_limits(sols: list[Solution], kb: KinBody) -> list[Solution]:
    """Drop solutions where any joint's q value is outside its reachable range.

    Joints with ``limits=None`` are unconstrained (continuous joints, or
    fixtures that don't supply limits) and never reject a solution. Joints
    with ``limits=(lo, hi)`` reject any solution where ``q[i] < lo`` or
    ``q[i] > hi`` strictly; values exactly on the boundary are accepted.

    :param sols: candidate solutions (e.g. output of an ssik solver's
        ``solve()``).
    :param kb: the same :class:`KinBody` used for the IK call. Joint limits
        come from ``kb.joints[i].limits``.
    :returns: filtered solutions; preserves input order.
    """
    n_joints = len(kb.joints)
    kept: list[Solution] = []
    for sol in sols:
        if len(sol.q) != n_joints:
            raise ValueError(f"solution q-length {len(sol.q)} doesn't match kb DOF {n_joints}")
        within = True
        for i, joint in enumerate(kb.joints):
            if joint.limits is None:
                continue
            lo, hi = joint.limits
            if sol.q[i] < lo or sol.q[i] > hi:
                within = False
                break
        if within:
            kept.append(sol)
    return kept


def wrap_to_limits(sols: list[Solution], kb: KinBody) -> list[Solution]:
    """Try wrapping each joint's q value by ``±2*pi`` integer multiples to
    bring it into the joint's reachable range.

    A revolute joint at ``q = 3.0`` with limits ``[-pi, pi]`` is FK-equivalent
    to ``q - 2*pi ≈ -3.28``, which is *also* outside the range, so neither
    wrap fits and the solution stays at ``q = 3.0`` (and would be dropped by
    :func:`respect_limits` if called next). A joint at ``q = 4.0`` with
    limits ``[-pi, pi]`` wraps to ``q - 2*pi ≈ -2.28`` which is in range:
    we keep the wrapped value.

    Joints with ``limits=None`` are left unchanged (no constraint to wrap
    into). Prismatic joints are left unchanged (no rotational periodicity).

    Search is over ``k ∈ {-2, -1, 0, +1, +2}`` integer multiples of ``2*pi``;
    that covers any joint whose limits span up to ±5*pi (more than enough for
    any commercial arm). The smallest-|k| wrap that lands in range wins,
    biasing toward the original value.

    :param sols: candidate solutions.
    :param kb: the same :class:`KinBody` used for the IK call.
    :returns: solutions with each q-vector adjusted joint-wise; preserves
        input order; returns ``Solution`` instances with the wrapped q
        and other fields unchanged.
    """
    n_joints = len(kb.joints)
    out: list[Solution] = []
    for sol in sols:
        if len(sol.q) != n_joints:
            raise ValueError(f"solution q-length {len(sol.q)} doesn't match kb DOF {n_joints}")
        q_new = np.asarray(sol.q, dtype=np.float64).copy()
        for i, joint in enumerate(kb.joints):
            if joint.limits is None or joint.joint_type != "revolute":
                continue
            lo, hi = joint.limits
            q_i = float(q_new[i])
            if lo <= q_i <= hi:
                continue
            # Try wraps with smallest |k| first.
            best = q_i
            best_in_range = False
            for k in (1, -1, 2, -2):
                candidate = q_i + 2.0 * np.pi * k
                if lo <= candidate <= hi:
                    best = candidate
                    best_in_range = True
                    break
            if best_in_range:
                q_new[i] = best
        out.append(replace(sol, q=q_new))
    return out


def _wrap_to_pi(angle: float) -> float:
    """Wrap a single angle to the canonical ``[-pi, pi]`` representative."""
    return float(((angle + np.pi) % (2.0 * np.pi)) - np.pi)


def nearest_to_seed(
    sols: list[Solution],
    q_seed: NDArray[np.float64],
    *,
    metric: str = "wrap_l2",
) -> list[Solution]:
    """Sort solutions by joint-space distance to a reference configuration.

    The "wrap-to-pi" distance treats angle differences modulo ``2*pi``, so
    e.g. ``q=3.0`` and ``q_seed=-3.0`` are at distance
    ``|wrap(3.0 - (-3.0))| = |wrap(6.0)| = |6.0 - 2*pi| ≈ 0.28``, not 6.0.
    This is the right metric for revolute-joint similarity.

    :param sols: candidate solutions.
    :param q_seed: reference joint configuration (length matches the chain's
        DOF).
    :param metric: one of ``"wrap_l2"`` (default; sum-of-squares of
        wrap-to-pi differences) or ``"wrap_linf"`` (max wrap-to-pi
        difference). ``wrap_l2`` is smooth and prefers configurations that
        are uniformly close; ``wrap_linf`` is hard-cap and prefers
        configurations whose worst-joint deviation is small.
    :returns: solutions sorted by ascending distance to ``q_seed``. Stable
        sort: ties preserve input order.
    """
    if metric not in ("wrap_l2", "wrap_linf"):
        raise ValueError(f"unknown metric {metric!r}; expected 'wrap_l2' or 'wrap_linf'")
    seed = np.asarray(q_seed, dtype=np.float64)

    def distance(sol: Solution) -> float:
        diffs = [_wrap_to_pi(float(sol.q[i] - seed[i])) for i in range(len(seed))]
        if metric == "wrap_l2":
            return float(np.sqrt(sum(d * d for d in diffs)))
        # wrap_linf
        return float(max(abs(d) for d in diffs))

    # Python's sort is stable, so ties preserve input order.
    return sorted(sols, key=distance)


def take_first(sols: list[Solution], k: int) -> list[Solution]:
    """Truncate to the first ``k`` solutions.

    Use after :func:`nearest_to_seed` (or any other ranking) to keep only
    the top-``k`` matches. ``k <= 0`` returns an empty list.

    Renamed from ``max_solutions`` in v1.0 to avoid name collision with
    the ``max_solutions`` kwarg on ``Manipulator.solve`` / artifact
    ``solve()`` -- they have different shapes (kwarg is an int passed in;
    this function takes ``(sols, k)``).

    :param sols: candidate solutions, typically already sorted.
    :param k: maximum number of solutions to keep.
    :returns: ``sols[:max(k, 0)]``.
    """
    return list(sols[: max(k, 0)])
