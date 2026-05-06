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

Total: 2 (q_3) x 2 (q_1) x 1 (q_2) x 2 (q_5) = 8 candidates per swivel.
FK closure filters spurious; cluster-merge (wrap-to-π) deduplicates
across swivel samples.

Per-arm cold-cache cost: ~0 (no symbolic precompute -- the algorithm is
purely numeric). Hot-path: dominated by branch enumeration x
swivel-sweep x per-branch FK closure check. Pure-Python; sub-millisecond
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
from ssik.subproblems import sp1
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


def _frame_at_joint(
    kb: KinBody, q: NDArray[np.float64], joint_idx: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(R, p)`` -- the world frame at joint ``joint_idx`` BEFORE its
    own rotation (i.e., after joints ``0..joint_idx-1`` and joint
    ``joint_idx``'s ``T_left``).

    Tracks rotation and position separately to avoid 4x4 matmul overhead.
    POE-normalized chains have pure-translation ``T_left`` and (for non-final
    joints) pure-translation ``T_right``, so position propagation reduces to
    ``p_new = p + R @ t`` and rotation propagation to one matmul per joint.

    Replaces the older ``_joint_origin_at_q`` + ``_orientation_up_to_joint``
    helpers with a single 2-3x faster version.
    """
    R = np.eye(3)
    p = np.zeros(3)
    for i, j in enumerate(kb.joints):
        # Apply T_left's translation (rotation block of T_left is identity
        # in POE-normalized chains).
        p = p + R @ j.T_left[:3, 3]
        if i == joint_idx:
            return R, p
        # Apply joint i's rotation.
        R = R @ rotation_matrix(j.axis, float(q[i]))
        # Apply T_right (translation always; rotation only matters on the
        # last joint but applying R @ I on the others is cheap and avoids
        # an expensive `array_equal` test in the hot loop).
        T_right = j.T_right
        R = R @ T_right[:3, :3]
        p = p + R @ T_right[:3, 3]
    return R, p


# ---------------------------------------------------------------------------
# Per-branch solver (one swivel angle, one elbow branch, one shoulder branch,
# one wrist branch).
# ---------------------------------------------------------------------------


def _solve_pre_wrist(
    kb: KinBody,
    cls: SrsClassification,
    L_se: float,
    R_target: NDArray[np.float64],
    W_t: NDArray[np.float64],
    swivel_data: tuple[NDArray[np.float64], float, float, NDArray[np.float64], NDArray[np.float64]],
    swivel: float,
    q_3_signed: float,
    q_1_sign: int,
    policy: TolerancePolicy,
) -> tuple[float, float, float, NDArray[np.float64]] | None:
    """Solve the SHOULDER + ELBOW triple (q_0, q_1, q_2, q_3) at given swivel
    + branch signs. Returns (q_0, q_1, q_2, R_pre_wrist) where R_pre_wrist
    is the world orientation of joint 4's pre-rotation frame (joints 0..3
    applied + joint 4's ``T_left``). Wrist branches share these.

    This is the part of the algorithm that doesn't depend on the wrist
    branch sign (q_5 sign), so we factor it out to compute it once per
    swivel x q_3 x q_1 combination instead of once per full branch.

    Reuses subproblem-composition style from
    :mod:`ssik.solvers.ikgeo.spherical`: the shoulder + elbow analytical
    closed-form, then a single ``sp1`` for q_2 from the wrist-pivot
    constraint.
    """
    u_sw, x_c, r_circle, u_perp1, u_perp2 = swivel_data
    S = cls.shoulder_pivot

    # Step 4a: elbow position on the swivel circle.
    E_t = S + x_c * u_sw + r_circle * (np.cos(swivel) * u_perp1 + np.sin(swivel) * u_perp2)

    # Step 4b: recover (q_0, q_1) from elbow direction analytically.
    # For ZY shoulder: d = R_z(q_0) R_y(q_1) (0, 0, 1) =
    #   (sin(q_1) cos(q_0), sin(q_1) sin(q_0), cos(q_1)).
    d = (E_t - S) / L_se
    cos_q1 = float(np.clip(d[2], -1.0, 1.0))
    q_1 = q_1_sign * np.arccos(cos_q1)
    sin_q1 = np.sin(q_1)
    if abs(sin_q1) > 1e-9:  # noqa: SIM108
        q_0 = np.arctan2(d[1] / sin_q1, d[0] / sin_q1)
    else:
        # Gimbal lock: elbow on the shoulder z-axis. q_0 is free; pick 0.
        q_0 = 0.0

    # Step 5: recover q_2 from wrist pivot constraint via a single SP1.
    # The "q_2 = 0 wrist pivot" is computed by partial FK once at q_2 = 0;
    # the SP1 then rotates that pivot offset (in plane perp to upper arm)
    # to align with the target wrist pivot. q_2 is unique up to a 2π
    # ambiguity that SP1 resolves.
    q_partial = np.array([q_0, q_1, 0.0, q_3_signed, 0.0, 0.0, 0.0], dtype=np.float64)
    _, W_at_q2_zero = _frame_at_joint(kb, q_partial, 5)

    u_upper = (E_t - S) / L_se  # upper arm direction
    p_from = W_at_q2_zero - E_t  # from elbow to q_2=0 wrist pivot
    p_to = W_t - E_t  # from elbow to target wrist pivot
    if np.linalg.norm(p_from) < 1e-9 or np.linalg.norm(p_to) < 1e-9:
        return None
    q_2, _ = sp1.solve(u_upper, p_from, p_to, policy)

    # Compute the world orientation of joint 4's pre-rotation frame (used
    # by the wrist triple; doesn't depend on q_5 sign so we cache it).
    q_post = np.array([q_0, q_1, q_2, q_3_signed, 0.0, 0.0, 0.0], dtype=np.float64)
    R_pre_wrist, _ = _frame_at_joint(kb, q_post, 4)

    return q_0, q_1, q_2, R_pre_wrist


def _solve_wrist_triple(
    kb: KinBody,
    R_target: NDArray[np.float64],
    R_pre_wrist: NDArray[np.float64],
    q_5_sign: int,
) -> tuple[float, float, float] | None:
    """Solve (q_4, q_5, q_6) from the residual rotation via ZYZ Euler.

    For SRS arms with canonical wrist axes (z, y, z) in the body frame at
    joint 4 (post joint-4-T_left), the residual rotation
    ``R_res = R_pre_wrist^T @ R_target @ R_post_wrist^T`` factorises as
    ``R_z(q_4) R_y(q_5) R_z(q_6) = R_res``. The two branches via the sign
    of ``q_5`` are enumerated by the caller.
    """
    R_post_wrist = kb.joints[6].T_right[:3, :3]
    R_res = R_pre_wrist.T @ R_target @ R_post_wrist.T

    cos_q5 = float(np.clip(R_res[2, 2], -1.0, 1.0))
    q_5 = q_5_sign * np.arccos(cos_q5)
    sin_q5 = np.sin(q_5)
    if abs(sin_q5) > 1e-9:
        q_4 = np.arctan2(q_5_sign * R_res[1, 2], q_5_sign * R_res[0, 2])
        q_6 = np.arctan2(q_5_sign * R_res[2, 1], q_5_sign * -R_res[2, 0])
    else:
        # Gimbal lock at q_5 = 0 or π: q_4 + q_6 (or q_4 - q_6) is
        # determined but the split is arbitrary; pick q_4 = 0.
        q_4 = 0.0
        if cos_q5 > 0:
            q_6 = float(np.arctan2(-R_res[0, 1], R_res[0, 0]))
        else:
            q_6 = float(np.arctan2(R_res[0, 1], -R_res[0, 0]))
    return q_4, q_5, q_6


# ---------------------------------------------------------------------------
# Public solve.
# ---------------------------------------------------------------------------


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    swivel_samples: int | NDArray[np.float64] = _DEFAULT_SWIVEL_SAMPLES,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
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
                # Solve the shoulder + elbow + q_2 once per (swivel, q_3, q_1)
                # since the wrist branch sign doesn't affect them.
                pre_wrist = _solve_pre_wrist(
                    kb,
                    cls,
                    L_se,
                    R_target,
                    W_t,
                    swivel_data,
                    float(swivel),
                    q_3_signed,
                    q_1_sign,
                    policy,
                )
                if pre_wrist is None:
                    continue
                q_0, q_1, q_2, R_pre_wrist = pre_wrist
                for q_5_sign in (+1, -1):
                    wrist = _solve_wrist_triple(kb, R_target, R_pre_wrist, q_5_sign)
                    if wrist is None:
                        continue
                    q_4, q_5, q_6 = wrist
                    q = np.array([q_0, q_1, q_2, q_3_signed, q_4, q_5, q_6], dtype=np.float64)
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
