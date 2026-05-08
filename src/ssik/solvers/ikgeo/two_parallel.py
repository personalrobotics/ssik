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

import logging

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import axis_parallel
from ssik.refinement import kinbody_jacobian, verify_candidates
from ssik.solvers.ikgeo._univariate import search_1d_matched
from ssik.subproblems import sp1, sp6
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["solve"]

_SEARCH_SAMPLES = 200
_SOLVER_NAME = "ikgeo.two_parallel"
_LOG = logging.getLogger(__name__)


def _wrap_to_pi(angle: float) -> float:
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    max_solutions: int | None = None,
) -> tuple[list[Solution], bool]:
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
        r_01 = rotation_matrix(axes[0], q1)
        h1 = r_06.T @ r_01 @ axes[1]
        sp_h = [h1, axes[1], h1, axes[1]]
        sp_k = [-axes[5], axes[3], -axes[5], axes[3]]
        sp_p = [p[5], p[4], axes[4], -axes[4]]
        d1 = float(axes[1] @ (r_01.T @ p_16 - p[1] - p[2] - p[3]))
        d2 = 0.0
        sols, _ = sp6.solve(sp_h, sp_k, sp_p, d1, d2, policy)
        branches: list[tuple[tuple[float, float], float]] = []
        for q6, q4 in sols:
            r_34 = rotation_matrix(axes[3], q4)
            r_56 = rotation_matrix(axes[5], q6)
            t23, _ = sp1.solve(
                axes[1],
                r_34 @ axes[4],
                r_01.T @ r_06 @ r_56.T @ axes[4],
                policy,
            )
            r_13 = rotation_matrix(axes[1], t23)
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
        r_01 = rotation_matrix(axes[0], q1)
        r_34 = rotation_matrix(axes[3], q4)
        r_56 = rotation_matrix(axes[5], q6)

        t23, _ = sp1.solve(
            axes[1],
            r_34 @ axes[4],
            r_01.T @ r_06 @ r_56.T @ axes[4],
            policy,
        )
        r_13 = rotation_matrix(axes[1], t23)

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

    solutions = verify_candidates(
        candidates,
        fk_fn=lambda q: poe_forward_kinematics(kb, q),
        jacobian_fn=lambda q: kinbody_jacobian(kb, q),
        t_target=t_target,
        fk_atol=policy.subproblem_numerical,
        dedup_atol=policy.subproblem_dedup,
        solver_name=_SOLVER_NAME,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
        max_solutions=max_solutions,
    )
    _LOG.info(
        "%s: %d candidates from %d q1-search branches -> %d solutions (is_ls=%s)",
        _SOLVER_NAME,
        len(candidates),
        len(q1_and_branch),
        len(solutions),
        len(solutions) == 0,
    )
    return solutions, len(solutions) == 0
