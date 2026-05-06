"""Native 7R analytical IK for SRS-class arms (Singh-Kreutz 1989).

SRS = Spherical-Roll-Spherical: shoulder axes (joints 0, 1, 2) meet at
one point ``S``, joint 3 is the elbow, wrist axes (joints 4, 5, 6) meet
at one point ``W``. The redundancy is parameterised by the elbow swivel
angle ``θ`` (the elbow's position on a circle perpendicular to the
shoulder-wrist axis).

Targets (predicate-driven; auto-classified by
:func:`ssik.kinematics.predicates.is_srs_7r`):

- KUKA iiwa LBR 14 / 7 / R820 / R14
- Flexiv Rizon 4 / 10 (when fixture lands; #80)
- Kinova Gen3 7-DOF (when fixture lands)
- Sawyer / Baxter (Rethink)
- Kassow KR810 / KR1410

Algorithm (verified end-to-end on iiwa14 — see
``docs/specs/srs_7r_singh_kreutz.md``):

1. Compute target wrist pivot ``W_t = T.p - T.R @ ee_offset_local``.
2. Cosine rule on shoulder-to-wrist distance gives elbow joint ``q_3``
   (2 branches: elbow up / elbow down).
3. For each swivel sample ``θ``, place elbow ``E_t`` on the circle
   perpendicular to ``SW`` at the geometrically-determined radius.
4. Recover ``(q_0, q_1)`` from the elbow direction (2 branches via the
   sign of ``q_1``).
5. Recover ``q_2`` from the wrist pivot constraint via a single
   ``atan2`` (signed angle from the q_2=0 wrist pivot to ``W_t`` around
   the upper arm axis).
6. Recover the wrist triple ``(q_4, q_5, q_6)`` from the residual
   rotation via ZYZ Euler decomposition (2 branches via the sign of
   ``q_5``).

Total: 2 (q_3) × 2 (q_1) × 1 (q_2) × 2 (q_5) = 8 candidates per swivel.
FK closure filters spurious; cluster-merge (wrap-to-π) deduplicates
across swivel samples.

Per-arm cold-cache cost: ~0 (no symbolic precompute -- the algorithm is
purely numeric). Hot-path: dominated by branch enumeration ×
swivel-sweep × per-branch FK closure check. Pure-Python; sub-millisecond
target reachable with #186 Cython compile if the bench gate misses.

References:

- Singh-Kreutz 1989: original closed-form 7R-SRS derivation.
- EAIK (Ostermeier 2024, arXiv:2409.14815): production C++ implementation.
- IK-Geo (Elias-Wen 2022, arXiv:2211.05737): subproblem decomposition
  family that this solver shares (SP1 atan2 for ``q_2``, ZYZ Euler for
  the wrist triple).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import (
    SrsClassification,
    is_srs_7r,
    joint_origins,
)
from ssik.refinement import dedup_by_wrap_close
from ssik.subproblems._rotation import rotation_matrix

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from ssik._kinbody import KinBody

__all__ = ["solve"]

_SOLVER_NAME = "seven_r.srs"
_DEFAULT_SWIVEL_SAMPLES = 16
_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-arm geometric setup (cheap; runs once per (kb, T_target) call).
# ---------------------------------------------------------------------------


def _arm_constants(kb: KinBody, cls: SrsClassification) -> tuple[float, float, NDArray[np.float64]]:
    """Compute (L_se, L_ew, ee_offset_local) from the chain at q=0."""
    origins = joint_origins(kb.joints)
    L_se = float(np.linalg.norm(origins[cls.elbow_index] - cls.shoulder_pivot))
    L_ew = float(np.linalg.norm(origins[cls.elbow_index] - cls.wrist_pivot))
    ee_home = poe_forward_kinematics(kb, np.zeros(len(kb.joints)))[:3, 3]
    ee_offset_local = ee_home - cls.wrist_pivot
    return L_se, L_ew, ee_offset_local


def _swivel_basis(
    u_sw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two orthonormal vectors spanning the plane perpendicular to ``u_sw``."""
    # Pick a reference vector that's not too aligned with u_sw.
    ref = np.array([0.0, 0.0, 1.0]) if abs(u_sw[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    u_perp1 = ref - np.dot(ref, u_sw) * u_sw
    u_perp1 /= np.linalg.norm(u_perp1)
    u_perp2 = np.cross(u_sw, u_perp1)
    return u_perp1, u_perp2


# ---------------------------------------------------------------------------
# FK helpers (branch verification + per-branch consistency).
# ---------------------------------------------------------------------------


def _joint_origin_at_q(kb: KinBody, q: NDArray[np.float64], joint_idx: int) -> NDArray[np.float64]:
    """Return joint ``joint_idx``'s origin in the world frame at config ``q``."""
    T = np.eye(4)
    for i, j in enumerate(kb.joints):
        T = T @ j.T_left
        if i == joint_idx:
            return T[:3, 3].copy()
        R = rotation_matrix(j.axis, q[i])
        T_full = np.eye(4)
        T_full[:3, :3] = R
        T = T @ T_full @ j.T_right
    return T[:3, 3].copy()


def _orientation_up_to_joint(
    kb: KinBody, q: NDArray[np.float64], joint_idx: int
) -> NDArray[np.float64]:
    """Return the 3x3 world rotation of joint ``joint_idx``'s frame BEFORE its
    own rotation (i.e., after applying joints ``0..joint_idx-1`` and joint
    ``joint_idx``'s ``T_left``)."""
    T = np.eye(4)
    for i, j in enumerate(kb.joints):
        T = T @ j.T_left
        if i == joint_idx:
            return T[:3, :3].copy()
        R = rotation_matrix(j.axis, q[i])
        T_full = np.eye(4)
        T_full[:3, :3] = R
        T = T @ T_full @ j.T_right
    return T[:3, :3].copy()


# ---------------------------------------------------------------------------
# Per-branch solver (one swivel angle, one elbow branch, one shoulder branch,
# one wrist branch).
# ---------------------------------------------------------------------------


def _solve_one_branch(
    kb: KinBody,
    cls: SrsClassification,
    L_se: float,
    L_ew: float,
    R_target: NDArray[np.float64],
    W_t: NDArray[np.float64],
    swivel_data: tuple[NDArray[np.float64], float, float, NDArray[np.float64], NDArray[np.float64]],
    swivel: float,
    q_3_signed: float,
    q_1_sign: int,
    q_5_sign: int,
    consistency_tol: float = 1e-7,
) -> NDArray[np.float64] | None:
    """Solve one of the 8 branches at the given swivel angle.

    Returns a 7-vector ``q`` if the branch is consistent (FK-traceable), or
    ``None`` if the branch is degenerate (gimbal lock, branch inconsistent
    with the chain's home orientation, etc.).
    """
    u_sw, x_c, r_circle, u_perp1, u_perp2 = swivel_data
    S = cls.shoulder_pivot

    # Step 4a: elbow position on the swivel circle.
    E_t = S + x_c * u_sw + r_circle * (np.cos(swivel) * u_perp1 + np.sin(swivel) * u_perp2)

    # Step 4b: recover (q_0, q_1) from elbow direction.
    # For ZY shoulder: d = R_z(q_0) R_y(q_1) (0, 0, 1) =
    #   (sin(q_1) cos(q_0), sin(q_1) sin(q_0), cos(q_1)).
    d = (E_t - S) / L_se
    cos_q1 = float(np.clip(d[2], -1.0, 1.0))
    q_1 = q_1_sign * np.arccos(cos_q1)
    sin_q1 = np.sin(q_1)
    if abs(sin_q1) > 1e-9:
        q_0 = np.arctan2(d[1] / sin_q1, d[0] / sin_q1)
    else:
        # Gimbal lock: elbow on the shoulder z-axis. q_0 is free; pick 0.
        q_0 = 0.0

    # Verify (q_0, q_1) actually places the elbow at E_t.
    E_check = S + L_se * np.array(
        [
            np.sin(q_1) * np.cos(q_0),
            np.sin(q_1) * np.sin(q_0),
            np.cos(q_1),
        ]
    )
    if np.linalg.norm(E_check - E_t) > consistency_tol:
        return None

    # Step 5: recover q_2 from wrist pivot constraint.
    # Build q_partial = (q_0, q_1, 0, q_3, 0, 0, 0); the wrist pivot at
    # this q is some specific point. Rotating q_2 about the upper-arm axis
    # rotates this wrist-pivot offset around the upper arm. q_2 is the
    # signed angle from the q_2=0 offset to W_t in the plane perpendicular
    # to the upper arm.
    q_partial = np.array([q_0, q_1, 0.0, q_3_signed, 0.0, 0.0, 0.0], dtype=np.float64)
    W_at_q2_zero = _joint_origin_at_q(kb, q_partial, 5)

    u_upper = (E_t - S) / L_se  # upper arm direction
    v0 = W_at_q2_zero - E_t  # from elbow to q_2=0 wrist pivot (in world)
    v_t = W_t - E_t  # from elbow to target wrist pivot

    # Project both onto the plane perpendicular to u_upper.
    v0_perp = v0 - np.dot(v0, u_upper) * u_upper
    vt_perp = v_t - np.dot(v_t, u_upper) * u_upper
    if np.linalg.norm(v0_perp) < 1e-9 or np.linalg.norm(vt_perp) < 1e-9:
        # Singular: lower arm aligned with upper arm (can happen at
        # full extension, q_3 near 0 / 2π). q_2 is free in this branch.
        return None

    v0_perp_n = v0_perp / np.linalg.norm(v0_perp)
    vt_perp_n = vt_perp / np.linalg.norm(vt_perp)
    cos_q2 = float(np.dot(v0_perp_n, vt_perp_n))
    sin_q2 = float(np.dot(np.cross(v0_perp_n, vt_perp_n), u_upper))
    q_2 = np.arctan2(sin_q2, cos_q2)

    # Verify q_2 places W at W_t.
    q_check = np.array([q_0, q_1, q_2, q_3_signed, 0.0, 0.0, 0.0], dtype=np.float64)
    W_check = _joint_origin_at_q(kb, q_check, 5)
    if np.linalg.norm(W_check - W_t) > consistency_tol:
        return None

    # Step 6: wrist triple from residual rotation.
    # Compute the world rotation up to and including joint 3, then the world
    # rotation of joint 4's PRE-rotation frame (joint 4's T_left applied).
    R_pre_wrist = _orientation_up_to_joint(kb, q_check, 4)
    # The remaining rotation that joints 4-6 must produce, in the body frame
    # at joint 4 (pre-rotation). We also account for the last joint's T_right
    # (any tool rotation).
    R_post_wrist = kb.joints[6].T_right[:3, :3]
    R_res = R_pre_wrist.T @ R_target @ R_post_wrist.T

    # ZYZ Euler decomposition for axes (joint 4 = z, joint 5 = y, joint 6 = z
    # in iiwa14's body frame at the wrist). Generalizes by reading the
    # actual body-frame axes from kb.joints[4..6].axis -- but for SRS arms
    # these are typically the canonical (z, y, z) at q=0.
    # R_z(q_4) R_y(q_5) R_z(q_6) = R_res.
    # q_5 = ±acos(R_res[2,2]); q_4, q_6 from off-diagonals.
    cos_q5 = float(np.clip(R_res[2, 2], -1.0, 1.0))
    q_5 = q_5_sign * np.arccos(cos_q5)
    sin_q5 = np.sin(q_5)
    if abs(sin_q5) > 1e-9:
        q_4 = np.arctan2(q_5_sign * R_res[1, 2], q_5_sign * R_res[0, 2])
        q_6 = np.arctan2(q_5_sign * R_res[2, 1], q_5_sign * -R_res[2, 0])
    else:
        # Gimbal lock at q_5 = 0 or π: q_4 + q_6 (or q_4 - q_6) is determined
        # but split is arbitrary; pick q_4 = 0, q_6 absorbs the rest.
        q_4 = 0.0
        if cos_q5 > 0:
            q_6 = np.arctan2(-R_res[0, 1], R_res[0, 0])
        else:
            q_6 = np.arctan2(R_res[0, 1], -R_res[0, 0])

    return np.array([q_0, q_1, q_2, q_3_signed, q_4, q_5, q_6], dtype=np.float64)


# ---------------------------------------------------------------------------
# Public solve.
# ---------------------------------------------------------------------------


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    swivel_samples: int | NDArray[np.float64] = _DEFAULT_SWIVEL_SAMPLES,
    allow_refinement: bool = False,  # noqa: ARG001 -- accepted for solver-API parity; SRS reaches machine precision algebraically
    refinement_max_iters: int = 15,  # noqa: ARG001 -- ditto
    max_solutions: int | None = None,
    fk_atol: float | None = None,
) -> tuple[list[Solution], bool]:
    """Native SRS-class 7R analytical IK via Singh-Kreutz parameterization.

    :param kb: POE-normalized 7R :class:`KinBody`. Topology must be SRS-class
        (verified via :func:`ssik.kinematics.predicates.is_srs_7r`).
    :param T_target: 4x4 target pose in the base frame.
    :param policy: tolerance policy.
    :param swivel_samples: int N for uniform sweep over [-π, π], or an
        explicit array of swivel angles.
    :param allow_refinement: accepted for parity with other solvers; the SRS
        algorithm is closed-form and reaches machine precision algebraically,
        so refinement is unnecessary on non-singular inputs.
    :param max_solutions: optional cap. When set, stops sampling once that
        many deduplicated solutions have been found.
    :param fk_atol: FK closure tolerance for accepting a candidate. Default
        ``policy.subproblem_numerical``.

    :returns: ``(solutions, is_ls)``. ``is_ls=True`` iff zero candidates
        passed FK closure.

    :raises ValueError: if ``kb`` is not 7-DOF or not SRS-class.
    """
    if len(kb.joints) != 7:
        raise ValueError(f"seven_r.srs requires a 7-DOF chain; got {len(kb.joints)}")
    cls = is_srs_7r(kb, policy)
    if cls is None:
        raise ValueError(
            "seven_r.srs requires SRS-class topology (shoulder axes 0,1,2 "
            "concurrent + wrist axes 4,5,6 concurrent). Use "
            "ssik.kinematics.predicates.is_srs_7r to check."
        )

    L_se, L_ew, ee_offset_local = _arm_constants(kb, cls)

    t_target = np.asarray(T_target, dtype=np.float64)
    R_target = t_target[:3, :3]
    p_target = t_target[:3, 3]
    W_t = p_target - R_target @ ee_offset_local

    # Step 2: shoulder-to-wrist
    SW = W_t - cls.shoulder_pivot
    d_sw = float(np.linalg.norm(SW))
    if d_sw > L_se + L_ew or d_sw < abs(L_se - L_ew):
        # Target wrist out of reach.
        return [], True
    u_sw = SW / d_sw

    # Step 3: q_3 candidates from cosine rule.
    cos_int = float(np.clip((L_se**2 + L_ew**2 - d_sw**2) / (2.0 * L_se * L_ew), -1.0, 1.0))
    base_q3 = np.pi - np.arccos(cos_int)
    q_3_branches = (base_q3, -base_q3)

    # Step 4 setup: swivel circle.
    x_c = (L_se**2 - L_ew**2 + d_sw**2) / (2.0 * d_sw)
    r_circle = float(np.sqrt(max(L_se**2 - x_c**2, 0.0)))
    u_perp1, u_perp2 = _swivel_basis(u_sw)
    swivel_data = (u_sw, x_c, r_circle, u_perp1, u_perp2)

    # Swivel grid.
    if isinstance(swivel_samples, int):
        swivels = np.linspace(-np.pi, np.pi, swivel_samples, endpoint=False)
    else:
        swivels = np.asarray(swivel_samples, dtype=np.float64).ravel()

    fk_threshold = fk_atol if fk_atol is not None else policy.subproblem_numerical

    candidates: list[Solution] = []
    branch_id = 0
    for swivel in swivels:
        for q_3_signed in q_3_branches:
            for q_1_sign in (+1, -1):
                for q_5_sign in (+1, -1):
                    q = _solve_one_branch(
                        kb,
                        cls,
                        L_se,
                        L_ew,
                        R_target,
                        W_t,
                        swivel_data,
                        float(swivel),
                        q_3_signed,
                        q_1_sign,
                        q_5_sign,
                    )
                    if q is None:
                        continue
                    # FK closure check
                    T_fk = poe_forward_kinematics(kb, q)
                    fk_residual = float(np.linalg.norm(T_fk - t_target))
                    if fk_residual > fk_threshold:
                        continue
                    candidates.append(
                        Solution(
                            q=q,
                            fk_residual=fk_residual,
                            refinement_used="none",
                            refinement_iters=0,
                            branch_id=branch_id,
                            solver_name=_SOLVER_NAME,
                        )
                    )
                    branch_id += 1
                    if max_solutions is not None and len(candidates) >= max_solutions:
                        # Early-exit AFTER dedup so the user gets up to
                        # max_solutions UNIQUE IKs.
                        deduped = dedup_by_wrap_close(candidates, policy.subproblem_dedup)
                        if len(deduped) >= max_solutions:
                            return deduped[:max_solutions], False

    if not candidates:
        return [], True

    deduped = dedup_by_wrap_close(candidates, policy.subproblem_dedup)
    if max_solutions is not None:
        deduped = deduped[:max_solutions]
    return deduped, False
