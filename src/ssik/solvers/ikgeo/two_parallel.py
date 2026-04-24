"""Tier-1 univariate-search 6R solver: two parallel shoulder-elbow axes.

Handles any 6R kinematic chain where joints ``(1, 2)`` are parallel
(``axes[1] || axes[2]``) and no stronger wrist specialization matches.
Weaker than ``three_parallel`` (which requires three consecutive
parallel axes).

Algorithm: port of the BSD-3 [ik-geo Rust reference][ikgeo]'s
``two_parallel`` (Elias & Wen, arXiv:2211.05737). 1D search over
``theta_0`` with an inner SP6 call per sample.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/mod.rs
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.predicates import axis_parallel
from ssik.solvers.ikgeo._univariate import search_1d_matched
from ssik.subproblems import sp1, sp6

__all__ = ["solve"]

_SEARCH_SAMPLES = 200


def _rot_mat(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """3x3 rotation matrix around ``axis`` by ``angle`` (Rodrigues)."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )


def _wrap_to_pi(angle: float) -> float:
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[NDArray[np.float64]], bool]:
    if len(kb.joints) != 6:
        raise ValueError(f"two_parallel requires a 6-DOF chain; got {len(kb.joints)} joints")
    if not axis_parallel(kb.joints[1].axis, kb.joints[2].axis, policy):
        raise ValueError(
            "two_parallel requires joints (1, 2) to be parallel; "
            "they are not. Check the chain's topology."
        )

    axes = [j.axis for j in kb.joints]
    p = [kb.joints[i].T_left[:3, 3].copy() for i in range(6)]
    p.append(kb.joints[-1].T_right[:3, 3].copy())

    r_home = kb.joints[-1].T_right[:3, :3]
    t_target = np.asarray(T_target, dtype=np.float64)
    r_06 = t_target[:3, :3] @ r_home.T
    p_0t = t_target[:3, 3]

    p_16 = p_0t - p[0] - r_06 @ p[6]

    p2_norm = float(np.linalg.norm(p[2]))

    def _branches_at(q1: float) -> list[tuple[tuple[float, float], float]]:
        """At a given q1, return [((q6, q4), alignment_error), ...] for every
        SP6 branch. Used by `search_1d_matched` to track geometric branches
        across adjacent q1 samples by (q6, q4)-proximity rather than index.
        """
        r_01 = _rot_mat(axes[0], q1)
        h1 = r_06.T @ r_01 @ axes[1]
        sp_h = [h1, axes[1], h1, axes[1]]
        sp_k = [-axes[5], axes[3], -axes[5], axes[3]]
        sp_p = [p[5], p[4], axes[4], -axes[4]]
        d1 = float(axes[1] @ (r_01.T @ p_16 - p[1] - p[2] - p[3]))
        d2 = 0.0
        sols, _ = sp6.solve(sp_h, sp_k, sp_p, d1, d2, policy)
        branches: list[tuple[tuple[float, float], float]] = []
        for q6, q4 in sols:
            r_34 = _rot_mat(axes[3], q4)
            r_56 = _rot_mat(axes[5], q6)
            t23, _ = sp1.solve(
                axes[1],
                r_34 @ axes[4],
                r_01.T @ r_06 @ r_56.T @ axes[4],
                policy,
            )
            r_13 = _rot_mat(axes[1], t23)
            delta = (
                r_01.T @ p_16
                - p[1]
                - r_13 @ p[3]
                - r_13 @ r_34 @ p[4]
                - r_01.T @ r_06 @ r_56.T @ p[5]
            )
            err = float(np.linalg.norm(delta)) - p2_norm
            branches.append(((q6, q4), err))
        return branches

    q1_and_branch = search_1d_matched(_branches_at, -np.pi, np.pi, _SEARCH_SAMPLES)

    candidates: list[NDArray[np.float64]] = []
    for q1, (q6, q4) in q1_and_branch:
        r_01 = _rot_mat(axes[0], q1)
        r_34 = _rot_mat(axes[3], q4)
        r_56 = _rot_mat(axes[5], q6)

        t23, _ = sp1.solve(
            axes[1],
            r_34 @ axes[4],
            r_01.T @ r_06 @ r_56.T @ axes[4],
            policy,
        )
        r_13 = _rot_mat(axes[1], t23)

        delta = (
            r_01.T @ p_16 - p[1] - r_13 @ p[3] - r_13 @ r_34 @ p[4] - r_01.T @ r_06 @ r_56.T @ p[5]
        )
        q2, _ = sp1.solve(axes[1], p[2], delta, policy)
        q5, _ = sp1.solve(
            -axes[4],
            r_34.T @ axes[1],
            r_56 @ r_06.T @ r_01 @ axes[1],
            policy,
        )
        q3 = _wrap_to_pi(t23 - q2)
        candidates.append(np.array([q1, q2, q3, q4, q5, q6]))

    num_tol = policy.subproblem_numerical
    dedup_tol = policy.subproblem_dedup
    verified: list[NDArray[np.float64]] = []
    for q in candidates:
        if float(np.linalg.norm(_forward_kinematics(kb, q) - t_target)) < num_tol:
            verified.append(q)

    solutions: list[NDArray[np.float64]] = []
    for q in verified:
        if not any(_q_close(q, existing, dedup_tol) for existing in solutions):
            solutions.append(q)

    return solutions, len(solutions) == 0


def _q_close(a: NDArray[np.float64], b: NDArray[np.float64], tol: float) -> bool:
    for ai, bi in zip(a, b, strict=True):
        diff = float(((float(ai) - float(bi) + np.pi) % (2 * np.pi)) - np.pi)
        if abs(diff) > tol:
            return False
    return True


def _forward_kinematics(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = _rot_mat(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T
