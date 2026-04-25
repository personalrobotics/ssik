"""Tier-2 general 6R IK solver: fully-general bivariate search.

Handles ANY 6R kinematic chain without relying on parallel / intersecting /
wrist specializations. This is the most general (and computationally
heaviest) of the IK-Geo family. Broken into a 2D grid search over the first
two joints with an inner SP5 solve + SP1 closure, then Nelder-Mead
refinement.

Algorithm: port of the BSD-3 [ik-geo Rust reference][ikgeo]'s
``gen_six_dof``. At each ``(theta_0, theta_1)`` sample:

1. Compute ``p_63 = Rot(-axes[1], q1) @ (Rot(-axes[0], q0) @ p_16 - p[1])
   - p[2]``, the wrist-center position in frame-2 coordinates.
2. SP5 solves for ``(theta_2, theta_3, theta_4)`` as the remaining
   shoulder-to-wrist chain. Up to 4 triples per ``(q0, q1)``.
3. For each triple, compute ``R_05`` (cumulative rotation through
   joints 0-4), then the alignment error is
   ``|R_05 @ axes[5] - R_06 @ axes[5]|`` -- zero when the full 6D pose
   closes.
4. :func:`ssik.solvers.ikgeo._bivariate.search_2d` samples on a 100x100
   grid, picks local minima, and refines each via Nelder-Mead.
5. At each refined minimum, SP1 recovers ``theta_5``.

Output: up to ~8 IK solutions (determined by how many distinct minima
``search_2d`` finds). Precision is limited by Nelder-Mead convergence
tolerance (1e-6 sd) plus SP5's polynomial machinery.

**Cost note**: ~10000 SP5 calls per IK solve (the 100x100 grid). Each
IK takes seconds, not milliseconds. Use the specialised tier-0 / tier-1
solvers when they match. Dispatcher uses this only as the fallback for
arms that match no specialisation.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/mod.rs
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.solvers.ikgeo._bivariate import search_2d
from ssik.subproblems import sp1, sp5

__all__ = ["solve"]

_GRID_N = 100


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


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[NDArray[np.float64]], bool]:
    """Analytic + bivariate-search IK for any 6R chain.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerances (forwarded to SP5 and SP1).
    :returns: ``(solutions, is_ls)``. Up to 8 length-6 joint vectors,
        each reproducing ``T_target`` under FK to within the
        subproblem-residual tolerance.
    """
    if len(kb.joints) != 6:
        raise ValueError(f"gen_six_dof requires a 6-DOF chain; got {len(kb.joints)} joints")

    axes = [j.axis for j in kb.joints]
    p = [kb.joints[i].T_left[:3, 3].copy() for i in range(6)]
    p.append(kb.joints[-1].T_right[:3, 3].copy())

    r_home = kb.joints[-1].T_right[:3, :3]
    t_target = np.asarray(T_target, dtype=np.float64)
    r_06 = t_target[:3, :3] @ r_home.T
    p_0t = t_target[:3, 3]

    p_16 = p_0t - p[0] - r_06 @ p[6]

    target_axis5 = r_06 @ axes[5]

    def _alignment_error(q1: float, q2: float) -> NDArray[np.float64]:
        """4-vector of ``|R_05 @ axes[5] - R_06 @ axes[5]|`` per SP5 branch."""
        errors = np.full(4, np.inf, dtype=np.float64)
        p_63 = _rot_mat(-axes[1], q2) @ (_rot_mat(-axes[0], q1) @ p_16 - p[1]) - p[2]
        triples, _ = sp5.solve(-p[3], p_63, p[4], p[5], -axes[2], axes[3], axes[4], policy)
        for i, (q3, q4, q5) in enumerate(triples):
            r_05 = (
                _rot_mat(axes[0], q1)
                @ _rot_mat(axes[1], q2)
                @ _rot_mat(axes[2], q3)
                @ _rot_mat(axes[3], q4)
                @ _rot_mat(axes[4], q5)
            )
            errors[i] = float(np.linalg.norm(r_05 @ axes[5] - target_axis5))
        return errors

    minima = search_2d(_alignment_error, (-np.pi, -np.pi), (np.pi, np.pi), _GRID_N)

    candidates: list[NDArray[np.float64]] = []
    for q1, q2, branch_idx in minima:
        p_63 = _rot_mat(-axes[1], q2) @ (_rot_mat(-axes[0], q1) @ p_16 - p[1]) - p[2]
        triples, _ = sp5.solve(-p[3], p_63, p[4], p[5], -axes[2], axes[3], axes[4], policy)
        if branch_idx >= len(triples):
            continue
        q3, q4, q5 = triples[branch_idx]

        r_05 = (
            _rot_mat(axes[0], q1)
            @ _rot_mat(axes[1], q2)
            @ _rot_mat(axes[2], q3)
            @ _rot_mat(axes[3], q4)
            @ _rot_mat(axes[4], q5)
        )
        z_axis = np.array([0.0, 0.0, 1.0])
        q6, _ = sp1.solve(axes[5], z_axis, r_05.T @ r_06 @ z_axis, policy)

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
    """POE forward kinematics for the composed IK post-verification."""
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = _rot_mat(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T
