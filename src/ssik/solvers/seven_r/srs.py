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
# Vectorised helpers (batch over swivel axis).
# ---------------------------------------------------------------------------


def _rodrigues_batch(axis: NDArray[np.float64], angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Batched Rodrigues: ``(N,)`` angles around fixed ``axis`` -> ``(N, 3, 3)``
    rotation matrices. Used by :func:`_frame_at_joint_batch` to apply each
    joint's rotation to the whole swivel batch in a single broadcast.
    """
    c = np.cos(angles)
    s = np.sin(angles)
    one_minus_c = 1.0 - c
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=np.float64)
    K2 = K @ K
    eye = np.eye(3, dtype=np.float64)
    return (
        eye[None, :, :]
        + s[:, None, None] * K[None, :, :]
        + one_minus_c[:, None, None] * K2[None, :, :]
    )


def _frame_at_joint_batch(
    kb: KinBody,
    q_batch: NDArray[np.float64],
    joint_idx: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Batched frame-at-joint walk. ``q_batch`` is ``(N, 7)``; returns
    ``(R: (N, 3, 3), p: (N, 3))`` -- the world frame at ``joint_idx`` BEFORE
    its own rotation, evaluated for each row of ``q_batch``.

    Walks the chain joint-by-joint, applying each joint's rotation and
    translation as a batched ``(N, 3, 3)`` / ``(N, 3)`` op. Eliminates the
    ~6 ms Python interpretation overhead that the per-call scalar variant
    incurred on the 16-swivel sweep (~1.9x speedup on iiwa14 strict).
    """
    n = q_batch.shape[0]
    R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    p = np.zeros((n, 3), dtype=np.float64)
    for i, j in enumerate(kb.joints):
        # T_left translation (rotation block is I in POE-normalised chains).
        p = p + R @ j.T_left[:3, 3]
        if i == joint_idx:
            return R, p
        # Joint rotation: batched Rodrigues over the (N,) angle slice.
        Ri = _rodrigues_batch(j.axis, q_batch[:, i])
        R = R @ Ri
        # T_right -- rotation block only matters on the last joint, but the
        # frame-at-joint contract returns BEFORE the joint at ``joint_idx``,
        # so ``joint_idx <= 5`` callers never reach the final joint's
        # ``T_right`` rotation block. Skip the identity-multiplication
        # branch test here for speed.
        T_right = j.T_right
        Rt_block = T_right[:3, :3]
        if not np.array_equal(Rt_block, np.eye(3)):
            R = R @ Rt_block
        p = p + R @ T_right[:3, 3]
    return R, p


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
    reach_slack: float = 0.0,
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
    :param reach_slack: slacken the cosine-rule reach check by this many
        meters in both directions (#200). Default ``0.0`` preserves strict-SRS
        behaviour. Approximate-SRS callers (:mod:`ssik.solvers.seven_r.srs_polished`)
        pass ``2 * max_drift_m`` so the offset between approximated and true
        shoulder/wrist pivots doesn't push borderline-reachable poses past
        ``L_se + L_ew``. Spurious candidates from slackening fail FK closure
        downstream; the cost is a few extra LM-polish iterations on those
        seeds, not incorrect IKs.

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
    # Reach check with optional slack (#200): approximate-SRS callers slacken
    # by ``2 * max_drift_m`` so offsets between approximated and true pivots
    # don't reject borderline-reachable poses (notably elbow-singular q_3 ≈ 0).
    if d_sw > L_se + L_ew + reach_slack or d_sw < max(0.0, abs(L_se - L_ew) - reach_slack):
        # Target wrist out of reach.
        return [], True
    if d_sw < 1e-12:
        # Shoulder coincides with target wrist -- pathological configuration
        # the algorithm can't parameterise (no SW direction). Empty solution
        # set; LM-from-seed in srs_polished can recover.
        return [], True
    u_sw = SW / d_sw

    # Layer 2 of #223: clamp d_sw strictly inside the cosine-rule envelope
    # ONLY for approximate-SRS callers (reach_slack > 0). Without clamp,
    # d_sw exactly at the boundary gives r_circle = 0 (swivel circle
    # collapses) and a degenerate q_2 atan2 that LM polish struggles to
    # recover from. Clamping keeps r_circle > 0 so the standard 16-swivel
    # sweep produces 16 nearby candidates instead of 16 collapsed copies.
    #
    # Strict-SRS callers (reach_slack=0; iiwa14 etc.) skip the clamp:
    # introducing a 1e-6 m offset in d_sw translates to ~3e-7 FK residual
    # which exceeds their 1e-10 strict-SRS contract. The original boundary
    # handling via ``np.clip`` on ``cos_int`` and ``max(..., 0)`` on r_circle
    # squared keeps strict-SRS behaviour byte-stable.
    if reach_slack > 0.0:
        _SINGULAR_EPS = 1e-6
        d_sw_eff = float(
            np.clip(d_sw, abs(L_se - L_ew) + _SINGULAR_EPS, L_se + L_ew - _SINGULAR_EPS)
        )
    else:
        d_sw_eff = d_sw

    # Step 3: q_3 candidates from cosine rule.
    cos_int = float(np.clip((L_se**2 + L_ew**2 - d_sw_eff**2) / (2.0 * L_se * L_ew), -1.0, 1.0))
    base_q3 = np.pi - np.arccos(cos_int)
    q_3_branches = (base_q3, -base_q3)

    # Step 4 setup: swivel circle (uses the clamped d_sw_eff so r_circle > 0).
    x_c = (L_se**2 - L_ew**2 + d_sw_eff**2) / (2.0 * d_sw_eff)
    r_circle = float(np.sqrt(max(L_se**2 - x_c**2, 0.0)))
    u_perp1, u_perp2 = _swivel_basis(u_sw)
    swivel_data = (u_sw, x_c, r_circle, u_perp1, u_perp2)

    # Swivel grid.
    if isinstance(swivel_samples, int):
        swivels = np.linspace(-np.pi, np.pi, swivel_samples, endpoint=False)
    else:
        swivels = np.asarray(swivel_samples, dtype=np.float64).ravel()

    fk_threshold = fk_atol if fk_atol is not None else policy.subproblem_numerical

    # Vectorised inner loop: batch all swivel-derived geometry into (N,)-shaped
    # arrays so the q_0/q_1/q_2 + R_pre_wrist + wrist-triple computation runs
    # under one numpy broadcast per (q_3_signed, q_1_sign, q_5_sign) outer
    # iteration -- 8 outer iterations vs the old 16 * 2 * 2 * 2 = 128. The
    # frame_at_joint walk happens in batched form (_frame_at_joint_batch). The
    # per-candidate FK closure check is still per-row because Cython's
    # ``poe_forward_kinematics`` is already optimised at ~12 us per call.
    u_sw, x_c, r_circle, u_perp1, u_perp2 = swivel_data
    N = swivels.shape[0]
    cs = np.cos(swivels)
    sn = np.sin(swivels)
    S = cls.shoulder_pivot
    E_t = (
        S
        + x_c * u_sw
        + r_circle * (cs[:, None] * u_perp1[None, :] + sn[:, None] * u_perp2[None, :])
    )  # (N, 3)
    R_post_wrist = kb.joints[6].T_right[:3, :3]

    # Layer 3 of #223: detect near-kinematic-singularity (r_circle small)
    # ONLY for approximate-SRS callers (those passing reach_slack > 0). At
    # the singularity the swivel circle collapses to a near-single point and
    # the wrist-pivot SP1 for q_2 becomes numerically unstable on arms whose
    # shoulder pivot is approximate (Gen3: 12 mm offset). Re-parameterising
    # to sweep q_2 directly produces a 1-parameter family of seeds that
    # LM polish (in srs_polished) can close to machine precision.
    #
    # Strict-SRS callers (reach_slack=0; iiwa14, etc.) keep the SP1 atan2
    # path. The atan2 may be numerically loose at the exact singularity,
    # but on a strict-SRS arm the algebraic candidates close to machine
    # precision regardless -- there's no offset to compound the error.
    # The strict-SRS test suite asserts FK closure at 1e-13, which the
    # SP1 path meets and the q_2-sweep would not.
    #
    # Threshold 1e-2 m (1 cm): empirically separates the SP1-stable regime
    # from the q_2-redundancy regime on Gen3-class arms.
    _SINGULAR_R_CIRCLE = 1e-2
    _is_singular: bool = reach_slack > 0.0 and r_circle < _SINGULAR_R_CIRCLE

    # Elbow direction d (N, 3) -- same for both q_1 signs because cos(q_1)
    # depends only on d[:, 2].
    d = (E_t - S) / L_se
    cos_q1_raw = np.clip(d[:, 2], -1.0, 1.0)

    candidates: list[Solution] = []
    branch_id = 0

    for q_3_signed in q_3_branches:
        for q_1_sign in (+1, -1):
            q_1 = q_1_sign * np.arccos(cos_q1_raw)
            sin_q1 = np.sin(q_1)
            q_0 = np.zeros(N, dtype=np.float64)
            non_gimbal_q1 = np.abs(sin_q1) > 1e-9
            if non_gimbal_q1.any():
                q_0[non_gimbal_q1] = np.arctan2(
                    d[non_gimbal_q1, 1] / sin_q1[non_gimbal_q1],
                    d[non_gimbal_q1, 0] / sin_q1[non_gimbal_q1],
                )

            # Frame at joint 5 with q_partial = (q_0, q_1, 0, q_3, 0, 0, 0).
            q_partial = np.zeros((N, 7), dtype=np.float64)
            q_partial[:, 0] = q_0
            q_partial[:, 1] = q_1
            q_partial[:, 3] = q_3_signed
            _, W_at_q2_zero = _frame_at_joint_batch(kb, q_partial, 5)

            if _is_singular:
                # At the kinematic singularity, the wrist-pivot SP1 is
                # degenerate -- replace q_2's atan2 derivation with a
                # direct sweep of the swivel-grid values, treating q_2 as
                # the redundancy parameter (#223 layer 3). The 16 swivel
                # samples become 16 q_2 samples; the wrist triple computes
                # correctly for each. LM polish closes the small offset
                # induced by clamping d_sw to d_sw_eff.
                q_2 = swivels
            else:
                # SP1 vectorised: q_2 around upper-arm axis maps q_2=0 wrist
                # pivot into target W_t. Closed-form:
                # q = atan2(u . (p1 x p2), p1.p2 - (u.p1)(u.p2)).
                u_upper = d  # (N, 3)
                p_from = W_at_q2_zero - E_t
                p_to = W_t - E_t  # (3,) -> broadcasts to (N, 3) below
                up_dot_pf = (u_upper * p_from).sum(axis=1)
                up_dot_pt = (u_upper * p_to).sum(axis=1)
                cross_pf_pt = np.cross(p_from, p_to)
                num = (u_upper * cross_pf_pt).sum(axis=1)
                den = (p_from * p_to).sum(axis=1) - up_dot_pf * up_dot_pt
                q_2 = np.arctan2(num, den)

            # Frame at joint 4 with q_full_pre = (q_0, q_1, q_2, q_3, 0, 0, 0).
            q_post = q_partial.copy()
            q_post[:, 2] = q_2
            R_pre_wrist, _ = _frame_at_joint_batch(kb, q_post, 4)

            # Wrist triple via ZYZ Euler -- vectorised. R_res = R_pre^T R_target R_post^T.
            R_res = R_pre_wrist.transpose(0, 2, 1) @ R_target @ R_post_wrist.T
            cos_q5_raw = np.clip(R_res[:, 2, 2], -1.0, 1.0)

            for q_5_sign in (+1, -1):
                q_5 = q_5_sign * np.arccos(cos_q5_raw)
                sin_q5 = np.sin(q_5)
                q_4 = np.zeros(N, dtype=np.float64)
                q_6 = np.zeros(N, dtype=np.float64)
                ng = np.abs(sin_q5) > 1e-9
                if ng.any():
                    q_4[ng] = np.arctan2(q_5_sign * R_res[ng, 1, 2], q_5_sign * R_res[ng, 0, 2])
                    q_6[ng] = np.arctan2(q_5_sign * R_res[ng, 2, 1], q_5_sign * -R_res[ng, 2, 0])
                gimbal = ~ng
                if gimbal.any():
                    # Gimbal lock at q_5 = 0 or π: q_4 + q_6 (or q_4 - q_6) is
                    # determined; pick q_4 = 0 and recover q_6 from R_res[:2, :2].
                    cos_q5_pos = cos_q5_raw[gimbal] > 0
                    R_gimbal = R_res[gimbal]
                    q_6_g = np.where(
                        cos_q5_pos,
                        np.arctan2(-R_gimbal[:, 0, 1], R_gimbal[:, 0, 0]),
                        np.arctan2(R_gimbal[:, 0, 1], -R_gimbal[:, 0, 0]),
                    )
                    q_6[gimbal] = q_6_g

                # Build q-array (N, 7) and FK-verify each row.
                q_full = np.column_stack([q_0, q_1, q_2, np.full(N, q_3_signed), q_4, q_5, q_6])
                for i in range(N):
                    T_fk = poe_forward_kinematics(kb, q_full[i])
                    fk_residual = float(np.linalg.norm(T_fk - t_target))
                    if fk_residual > fk_threshold:
                        continue
                    candidates.append(
                        Solution(
                            q=q_full[i],
                            fk_residual=fk_residual,
                            refinement_used="none",
                            refinement_iters=0,
                            branch_id=branch_id,
                            solver_name=_SOLVER_NAME,
                        )
                    )
                    branch_id += 1
                    if max_solutions is not None and len(candidates) >= max_solutions:
                        deduped = dedup_by_wrap_close(candidates, policy.subproblem_dedup)
                        if len(deduped) >= max_solutions:
                            return deduped[:max_solutions], False

    if not candidates:
        return [], True

    deduped = dedup_by_wrap_close(candidates, policy.subproblem_dedup)
    if max_solutions is not None:
        deduped = deduped[:max_solutions]
    return deduped, False
