"""Native 7R analytical IK for SRS-class arms (Singh-Kreutz 1989).

SRS = Spherical-Roll-Spherical: shoulder axes (joints 0, 1, 2) meet at
one point ``S``, joint 3 is the elbow, wrist axes (joints 4, 5, 6) meet
at one point ``W``. The redundancy is parameterised by the elbow swivel
angle ``θ`` (the elbow's position on a circle perpendicular to the
shoulder-wrist axis).

Targets (predicate-driven; auto-classified by
:func:`ssik.kinematics.predicates.is_srs_7r` -- *any* concurrent-axis
shoulder + wrist, not just canonical Z*Z):

- KUKA iiwa LBR 14 / 7 / R820 / R14 (canonical z-y-z)
- Flexiv Rizon 4 / 10 (when fixture lands; #80)
- Kinova Gen3 7-DOF (approximate; via ``srs_polished``)
- Sawyer / Baxter (Rethink)
- Kassow KR810 / KR1410
- Enactic OpenArm v2.0, Galaxea R1 Pro -- non-Z*Z (#354)

Two extraction paths share the swivel-circle geometry:

**Canonical (z-y-z shoulder + wrist, ``u_home`` along +z; iiwa-class).**
Fully vectorised over the swivel sweep, byte-identical to the original
solver:

1. Wrist pivot ``W_t = T.p - T.R @ ee_offset_local``.
2. Cosine rule on ``|S - W_t|`` gives elbow ``q_3`` (2 branches).
3. Per swivel ``θ``, place elbow ``E_t`` on the reach circle.
4. ``(q_0, q_1)`` from the elbow direction; ``q_2`` from the wrist-pivot
   SP1; ``(q_4, q_5, q_6)`` from ZYZ-Euler of the residual.

**General (any concurrent-axis SRS; #354).** Pure rotation algebra per
swivel, valid for a tilted/offset elbow (axis not perpendicular to the
upper arm, ``S-E-W`` not straight at ``q_3 = 0``) and non-Z*Z triples:

1-3. Same wrist pivot + swivel-circle elbow placement.
4. Build the shoulder rotation ``R_sh`` from a frame alignment placing
   ``E_t`` plus a roll about the upper arm: ``q_3`` (elbow) from
   Subproblem 4 on the wrist-pivot latitude ``d . (W_t - E_t)`` (invariant
   under the roll), then the roll ``φ`` from an SP1.
5. Recover ``(q_0, q_1, q_2)`` from ``decompose_3axis(R_sh, n0, n1, n2)``
   and ``(q_4, q_5, q_6)`` from ``decompose_3axis`` of the residual --
   the generalized Davenport (arbitrary three-axis) decomposition.

Both paths emit up to 8 candidates per swivel; FK closure filters
spurious, cluster-merge (wrap-to-π) deduplicates across swivels.

Per-arm cold-cache cost: ~0 (no symbolic precompute). Pure-Python;
canonical path is sub-millisecond, the general path ~4-5 ms on the full
16-swivel sweep (vectorized over the swivel batch, #368; sub-millisecond
for ``max_solutions=1``).

References:

- Singh-Kreutz 1989: original closed-form 7R-SRS derivation.
- Shuster & Markley 2003: generalized (arbitrary-axis) Euler angles --
  the basis for the general shoulder/wrist decomposition.
- EAIK (Ostermeier 2024, arXiv:2409.14815): production C++ implementation.
- IK-Geo (Elias-Wen 2022, arXiv:2211.05737): subproblem family (SP1, SP4).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics._generalized_euler import (
    _axis_angle_matrix,
    _cross3,
    _norm3,
    decompose_3axis_batch,
)
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import (
    SrsClassification,
    _classify_srs_7r_geometric,
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


def _arm_constants(
    kb: KinBody, cls: SrsClassification
) -> tuple[float, float, NDArray[np.float64], list[NDArray[np.float64]]]:
    """Compute (L_se, L_ew, ee_offset_local, joint_origins) from the chain at q=0."""
    origins = joint_origins(kb.joints)
    L_se = float(np.linalg.norm(origins[cls.elbow_index] - cls.shoulder_pivot))
    L_ew = float(np.linalg.norm(origins[cls.elbow_index] - cls.wrist_pivot))
    ee_home = poe_forward_kinematics(kb, np.zeros(len(kb.joints)))[:3, 3]
    ee_offset_local = ee_home - cls.wrist_pivot
    return L_se, L_ew, ee_offset_local, origins


def _swivel_basis(
    u_sw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two orthonormal vectors spanning the plane perpendicular to ``u_sw``."""
    # Pick a reference vector that's not too aligned with u_sw.
    ref = np.array([0.0, 0.0, 1.0]) if abs(u_sw[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    u_perp1 = ref - np.dot(ref, u_sw) * u_sw
    u_perp1 /= _norm3(u_perp1)
    u_perp2 = _cross3(u_sw, u_perp1)
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


def _rodrigues_axes_batch(
    axes: NDArray[np.float64], angles: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rodrigues matrices for *per-row* ``axes`` ``(N, 3)`` and ``angles``
    ``(N,)`` -> ``(N, 3, 3)``. Unlike :func:`_rodrigues_batch` (fixed axis),
    the rotation axis varies per row -- needed for the shoulder-roll about the
    per-swivel elbow direction ``d_hat``."""
    c = np.cos(angles)
    s = np.sin(angles)
    omc = 1.0 - c
    ax, ay, az = axes[:, 0], axes[:, 1], axes[:, 2]
    n = axes.shape[0]
    K = np.zeros((n, 3, 3), dtype=np.float64)
    K[:, 0, 1] = -az
    K[:, 0, 2] = ay
    K[:, 1, 0] = az
    K[:, 1, 2] = -ax
    K[:, 2, 0] = -ay
    K[:, 2, 1] = ax
    K2 = K @ K
    return (
        np.eye(3)[None, :, :]
        + s[:, None, None] * K
        + omc[:, None, None] * K2
    )


def _min_rotation_batch(
    u_home: NDArray[np.float64], v_batch: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Vectorised :func:`_min_rotation`: minimal rotations carrying ``u_home``
    onto each row of ``v_batch`` ``(N, 3)`` -> ``(N, 3, 3)``. The rare
    (anti)parallel rows (cross product degenerate) fall back to the scalar
    path to preserve its exact perpendicular-axis choice."""
    u = u_home / _norm3(u_home)
    vn = v_batch / np.linalg.norm(v_batch, axis=1, keepdims=True)
    c = vn @ u  # (N,)
    axis = np.cross(np.broadcast_to(u, vn.shape), vn)  # (N, 3)
    an = np.linalg.norm(axis, axis=1)  # (N,)
    ok = an > 1e-9
    axu = np.zeros_like(axis)
    axu[ok] = axis[ok] / an[ok, None]
    out = _rodrigues_axes_batch(axu, np.arccos(np.clip(c, -1.0, 1.0)))
    for idx in np.nonzero(~ok)[0]:
        out[idx] = _min_rotation(u_home, v_batch[idx])
    return out


def _sp4_branches_batch(
    h: NDArray[np.float64],
    k: NDArray[np.float64],
    p: NDArray[np.float64],
    delta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Vectorised :func:`_sp4_branches` over ``(N, 3)`` vector stacks.

    :returns: ``(q_a, q_b, valid)`` -- the two elbow branches ``(N,)`` and a
        ``(N,)`` mask that is ``False`` where the projection is unreachable
        (``|C| > |A,B|``) or the amplitude degenerates. Both branches coincide
        where ``off -> 0``; the duplicate is collapsed downstream by dedup."""
    h_dot_k = np.einsum("ni,ni->n", h, k)
    k_dot_p = np.einsum("ni,ni->n", k, p)
    a_coef = np.einsum("ni,ni->n", h, p) - h_dot_k * k_dot_p
    b_coef = np.einsum("ni,ni->n", h, np.cross(k, p))
    c_const = delta - h_dot_k * k_dot_p
    amplitude = np.hypot(a_coef, b_coef)
    valid = amplitude >= 1e-12
    ratio = np.where(valid, c_const / np.where(valid, amplitude, 1.0), 2.0)
    valid &= np.abs(ratio) <= 1.0 + 1e-9
    base = np.arctan2(b_coef, a_coef)
    off = np.arccos(np.clip(ratio, -1.0, 1.0))
    return base + off, base - off, valid


def _shoulder_angles_zyz(
    d: NDArray[np.float64], q_1_sign: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Canonical-ZYZ shoulder: recover ``(q_0, q_1)`` (each ``(N,)``) aiming
    the home upper arm (along ``+z``) onto each elbow direction ``d``
    (``(N, 3)``) for ``z``-``y`` shoulder axes. ``cos q_1 == d_z``; ``q_0``
    is the azimuth. The two ``q_1_sign`` branches are the elbow-direction
    preimages. Vectorised + byte-identical to the pre-#354 solver. The
    general (non-ZYZ) path recovers the shoulder via ``decompose_3axis``
    instead.
    """
    cos_q1 = np.clip(d[:, 2], -1.0, 1.0)
    q_1 = q_1_sign * np.arccos(cos_q1)
    sin_q1 = np.sin(q_1)
    q_0 = np.zeros(d.shape[0], dtype=np.float64)
    ng = np.abs(sin_q1) > 1e-9
    if ng.any():
        q_0[ng] = np.arctan2(d[ng, 1] / sin_q1[ng], d[ng, 0] / sin_q1[ng])
    return q_0, q_1


def _min_rotation(u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimal rotation matrix carrying the direction of ``u`` onto ``v``."""
    u = u / _norm3(u)
    v = v / _norm3(v)
    c = float(u @ v)
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-12:
        # Antiparallel: rotate pi about any axis perpendicular to u.
        perp = _cross3(u, np.array([1.0, 0.0, 0.0]))
        if _norm3(perp) < 1e-6:
            perp = _cross3(u, np.array([0.0, 1.0, 0.0]))
        return _axis_angle_matrix(perp / _norm3(perp), np.pi)
    axis = _cross3(u, v)
    return _axis_angle_matrix(axis / _norm3(axis), float(np.arccos(c)))


def _sp4_branches(
    h: NDArray[np.float64], k: NDArray[np.float64], p: NDArray[np.float64], delta: float
) -> tuple[float, ...]:
    """Subproblem 4: solve ``h . (Rot(k, q) p) == delta`` for ``q``.

    ``A cos q + B sin q == C`` with ``A = h.p - (h.k)(k.p)``,
    ``B = h.(k x p)``, ``C = delta - (h.k)(k.p)`` -- up to two roots
    ``atan2(B, A) +/- arccos(C / |A,B|)`` (the elbow-up / elbow-down pair).
    Empty when the projection is unreachable (``|C| > |A,B|``).

    Scalar per-swivel form retained alongside :func:`_sp4_branches_batch`
    for the exact in-limits swivel resolver (``_swivel_limits``, #369),
    which walks candidate swivels one at a time rather than as a batch.
    """
    h_dot_k = float(h @ k)
    k_dot_p = float(k @ p)
    a_coef = float(h @ p) - h_dot_k * k_dot_p
    b_coef = float(h @ _cross3(k, p))
    c_const = delta - h_dot_k * k_dot_p
    amplitude = float(np.hypot(a_coef, b_coef))
    if amplitude < 1e-12:
        return ()
    ratio = c_const / amplitude
    if abs(ratio) > 1.0 + 1e-9:
        return ()
    base = float(np.arctan2(b_coef, a_coef))
    off = float(np.arccos(np.clip(ratio, -1.0, 1.0)))
    return (base + off,) if off < 1e-12 else (base + off, base - off)


def _verify_fk(
    sols: list[Solution],
    kb: KinBody,
    t_target: NDArray[np.float64],
    fk_threshold: float,
) -> list[Solution]:
    """Compute the FK residual for each candidate and drop anything past
    ``fk_threshold``. Returns the surviving solutions with their
    ``fk_residual`` field filled in.

    Run after dedup so we evaluate FK on the surviving ~50 unique IKs
    rather than the raw 128 candidates (#246). For strict-SRS arms the
    algebra is exact, so the threshold check is defensive belt-and-braces;
    for approximate-SRS callers (``reach_slack > 0``) it's load-bearing.
    """
    out: list[Solution] = []
    for s in sols:
        T_fk = poe_forward_kinematics(kb, s.q)
        _d = (T_fk - t_target).ravel()
        fk_residual = math.sqrt(float(_d @ _d))
        if fk_residual <= fk_threshold:
            # Direct construction beats dataclasses.replace (~10 us per
            # call) on the warm-path post-dedup loop.
            out.append(
                Solution(
                    q=s.q,
                    fk_residual=fk_residual,
                    refinement_used=s.refinement_used,
                )
            )
    return out


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
    # Use the geometric-only helper here, not the strict ``is_srs_7r``:
    # ``srs_polished`` (the approximate-SRS path used by Gen3 / OpenArm /
    # other non-canonical-Euler arms) calls ``srs.solve`` with reach-slack
    # to get warm-start candidates that its LM polish then rescues. Those
    # arms fail the Z*Z check in the strict predicate but are still
    # axis-concurrent SRS. The dispatcher's tier-0 gate is the strict
    # ``is_srs_7r``; the public ``srs.solve`` entrypoint only refuses
    # geometrically non-SRS chains.
    cls = _classify_srs_7r_geometric(kb, policy)
    if cls is None:
        raise ValueError(
            "seven_r.srs requires SRS-class topology (shoulder axes 0,1,2 "
            "concurrent + wrist axes 4,5,6 concurrent). Use "
            "ssik.kinematics.predicates.is_srs_7r to check."
        )

    L_se, L_ew, ee_offset_local, origins = _arm_constants(kb, cls)

    # Home upper-arm vector (S -> elbow at q=0). Cheap; needed by both paths.
    upper_home = origins[cls.elbow_index] - cls.shoulder_pivot
    u_home = upper_home / float(np.linalg.norm(upper_home))
    # Canonical-ZYZ fast path (iiwa-class): shoulder z-y with the home upper
    # arm along +z, wrist z-y-z. The original vectorised extraction hardcodes
    # these literals; keep it byte-identical + sub-millisecond there and fall
    # to the general Davenport extraction for any other concurrent-axis SRS
    # arm (#354 -- e.g. Galaxea R1 Pro's y-x-z shoulder / z-y-x wrist). The
    # axes are already unit in a POE-normalized chain, so check them directly
    # (no list-comp normalize -- this runs on the iiwa hot path every call).
    _EZ = np.array([0.0, 0.0, 1.0])
    _EY = np.array([0.0, 1.0, 0.0])
    j = kb.joints
    canonical_zyz = bool(
        np.allclose(j[0].axis, _EZ)
        and np.allclose(j[1].axis, _EY)
        and np.allclose(u_home, _EZ)
        and np.allclose(j[4].axis, _EZ)
        and np.allclose(j[5].axis, _EY)
        and np.allclose(j[6].axis, _EZ)
    )
    # Approximate-SRS callers (``reach_slack > 0``; only ``srs_polished``,
    # which is Z*Z-gated so the arm is canonical z-y-z up to its small pivot
    # drift) keep the canonical path: it is the ZYZ warm-start factory their
    # LM polish + near-singular q_2-sweep (#223) were tuned against. The
    # general path targets exact concurrent-axis arms (reach_slack == 0) and
    # carries no reach-slack / elbow-singular handling.
    use_canonical = canonical_zyz or reach_slack > 0.0

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

    # Elbow direction d (N, 3).
    d = (E_t - S) / L_se

    candidates: list[Solution] = []

    def _append(q_vec: NDArray[np.float64]) -> list[Solution] | None:
        """Append a finite candidate; return the capped verified set if
        ``max_solutions`` is already satisfied (short-circuit), else None.

        Singh-Kreutz on strict-SRS is exact closed-form -- intermediate
        clamps + atan2 introduce ≤ 1e-13 drift, well below ``fk_threshold``.
        FK verify is deferred to one post-dedup pass so we skip it on the
        candidates dedup will eliminate (~10% on iiwa14; #246). Only non-finite
        q-vectors are dropped here, as a safety net for the edge cases above.
        """
        if not np.all(np.isfinite(q_vec)):
            return None
        candidates.append(Solution(q=q_vec, fk_residual=0.0, refinement_used="none"))
        if max_solutions is not None and len(candidates) >= max_solutions:
            deduped_ = dedup_by_wrap_close(candidates, policy.subproblem_dedup)
            verified_ = _verify_fk(deduped_, kb, t_target, fk_threshold)
            if len(verified_) >= max_solutions:
                return verified_[:max_solutions]
        return None

    if use_canonical:
        # --- Canonical ZYZ sweep (iiwa-class): cosine-rule elbow + vectorised
        # ZYZ-Euler wrist. Byte-identical to the pre-#354 solver. ---
        for q_3_signed in q_3_branches:
            for q_1_sign in (+1, -1):
                q_0, q_1 = _shoulder_angles_zyz(d, q_1_sign)

                # Frame at joint 5 with q_partial = (q_0, q_1, 0, q_3, 0, 0, 0).
                q_partial = np.zeros((N, 7), dtype=np.float64)
                q_partial[:, 0] = q_0
                q_partial[:, 1] = q_1
                q_partial[:, 3] = q_3_signed
                _, W_at_q2_zero = _frame_at_joint_batch(kb, q_partial, 5)

                if _is_singular:
                    # At the kinematic singularity the wrist-pivot SP1 is
                    # degenerate -- sweep the swivel grid as q_2 directly
                    # (#223 layer 3); LM polish closes the clamp offset.
                    q_2 = swivels
                else:
                    # SP1 vectorised: q_2 around the upper-arm axis maps the
                    # q_2=0 wrist pivot onto W_t.
                    u_upper = d  # (N, 3)
                    p_from = W_at_q2_zero - E_t
                    p_to = W_t - E_t  # (3,) -> broadcasts to (N, 3)
                    up_dot_pf = (u_upper * p_from).sum(axis=1)
                    up_dot_pt = (u_upper * p_to).sum(axis=1)
                    cross_pf_pt = np.cross(p_from, p_to)
                    num = (u_upper * cross_pf_pt).sum(axis=1)
                    den = (p_from * p_to).sum(axis=1) - up_dot_pf * up_dot_pt
                    q_2 = np.arctan2(num, den)

                q_post = q_partial.copy()
                q_post[:, 2] = q_2
                R_pre_wrist, _ = _frame_at_joint_batch(kb, q_post, 4)
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
                        q_6[ng] = np.arctan2(
                            q_5_sign * R_res[ng, 2, 1], q_5_sign * -R_res[ng, 2, 0]
                        )
                    gimbal = ~ng
                    if gimbal.any():
                        # Gimbal lock at q_5 = 0 or π: q_4 + q_6 (or q_4 - q_6)
                        # is determined; pick q_4 = 0, recover q_6 from R_res.
                        cos_q5_pos = cos_q5_raw[gimbal] > 0
                        R_gimbal = R_res[gimbal]
                        q_6_g = np.where(
                            cos_q5_pos,
                            np.arctan2(-R_gimbal[:, 0, 1], R_gimbal[:, 0, 0]),
                            np.arctan2(R_gimbal[:, 0, 1], -R_gimbal[:, 0, 0]),
                        )
                        q_6[gimbal] = q_6_g

                    q_full = np.column_stack([q_0, q_1, q_2, np.full(N, q_3_signed), q_4, q_5, q_6])
                    finite = np.all(np.isfinite(q_full), axis=1)
                    for i in np.where(finite)[0]:
                        capped = _append(q_full[i])
                        if capped is not None:
                            return capped, False
    else:
        # --- General SRS sweep (#354): full Davenport shoulder + Davenport
        # wrist for any concurrent-axis SRS arm. Handles a tilted/offset elbow
        # (R1 Pro's elbow axis is not perpendicular to the upper arm and S-E-W
        # is not straight at q_3=0) and a non-ZYZ shoulder/wrist, where the
        # canonical path's cosine rule, q_2-as-roll SP1, and ZYZ extraction are
        # all invalid. Pure rotation algebra (no FK frame walks); each swivel
        # yields up to 2 (q_3) x 2 (shoulder) x 2 (wrist) = 8 candidates. ---
        # Unit axes + home forearm (built lazily: the canonical hot path above
        # never needs them, and the per-call cost matters at max_solutions=1).
        n_axes = [
            np.asarray(jt.axis, dtype=np.float64) / _norm3(np.asarray(jt.axis, dtype=np.float64))
            for jt in kb.joints
        ]
        n0_axis, n1_axis, n2_axis = n_axes[0], n_axes[1], n_axes[2]
        n3_axis = n_axes[cls.elbow_index]
        n4_axis, n5_axis, n6_axis = n_axes[4], n_axes[5], n_axes[6]
        forearm_home = cls.wrist_pivot - origins[cls.elbow_index]  # elbow -> wrist at q = 0

        # Batched over the N swivels (mirrors the canonical path's structure:
        # discrete branch enumeration outside, one broadcast per branch). The
        # elbow direction ``d`` and the target wrist offset vary per swivel; the
        # joint axes are constant, so the Davenport coefficients reduce to
        # scalars inside ``decompose_3axis_batch``.
        uppers = E_t - S  # (N, 3): S -> elbow (target)
        wrist_vecs = W_t - E_t  # (N, 3): elbow -> wrist pivot (target)
        # R0: reference shoulder rotations placing the elbow (upper_home -> upper).
        r0 = _min_rotation_batch(upper_home, uppers)  # (N, 3, 3)
        k_elbow = np.einsum("nij,j->ni", r0, n3_axis)  # (N, 3)
        v_forearm0 = np.einsum("nij,j->ni", r0, forearm_home)  # (N, 3)
        # q_3 (elbow): wrist-pivot latitude along the upper arm fixes it; SP4 on
        # the forearm rotated by R0. Two elbow branches, both computed.
        delta_sp4 = np.einsum("ni,ni->n", wrist_vecs, d)  # wrist_vec . d_hat
        q3a, q3b, sp4_valid = _sp4_branches_batch(d, k_elbow, v_forearm0, delta_sp4)

        # Wrist post-rotation folded once (r_res = r_pre_elbow^T @ R_target @ R_post^T).
        M_wrist = R_target @ R_post_wrist.T  # (3, 3)
        dw = np.einsum("ni,ni->n", d, wrist_vecs)  # d_hat . wrist_vec (roll-invariant)

        for q3 in (q3a, q3b):  # each (N,)
            rot_n3_q3 = _rodrigues_batch(n3_axis, q3)  # (N, 3, 3)
            g = np.einsum("nij,j->ni", rot_n3_q3, forearm_home)  # axis_angle(n3,q3) @ forearm
            g = np.einsum("nij,nj->ni", r0, g)  # r0 @ g (forearm dir before roll)
            # phi (shoulder roll about d_hat) mapping g onto wrist_vec: SP1.
            g_perp = g - d * np.einsum("ni,ni->n", d, g)[:, None]
            w_perp = wrist_vecs - d * dw[:, None]
            num = np.einsum("ni,ni->n", d, np.cross(g_perp, w_perp))
            den = np.einsum("ni,ni->n", g_perp, w_perp)
            phi = np.arctan2(num, den)
            phi[np.linalg.norm(g_perp, axis=1) < 1e-9] = 0.0

            r_sh = _rodrigues_axes_batch(d, phi) @ r0  # (N, 3, 3)
            r_pre_elbow = r_sh @ rot_n3_q3  # (N, 3, 3)
            r_res = np.einsum("nij,jk->nik", r_pre_elbow.transpose(0, 2, 1), M_wrist)

            sh_branches, sh_ok = decompose_3axis_batch(r_sh, n0_axis, n1_axis, n2_axis)
            wr_branches, wr_ok = decompose_3axis_batch(r_res, n4_axis, n5_axis, n6_axis)
            if not (sh_ok and wr_ok):
                continue
            for s0, s1, s2 in sh_branches:  # up to 2 shoulder branches
                for w4, w5, w6 in wr_branches:  # up to 2 wrist branches
                    q_full = np.stack([s0, s1, s2, q3, w4, w5, w6], axis=1)  # (N, 7)
                    keep = sp4_valid & np.all(np.isfinite(q_full), axis=1)
                    for row in np.nonzero(keep)[0]:
                        candidates.append(
                            Solution(q=q_full[row], fk_residual=0.0, refinement_used="none")
                        )

    if not candidates:
        return [], True

    deduped = dedup_by_wrap_close(candidates, policy.subproblem_dedup)
    # Final FK pass on dedup'd survivors: fills in fk_residual with the
    # actual measured closure and drops any candidate that's drifted past
    # ``fk_threshold``. This is the FK guarantee the public API promises.
    verified = _verify_fk(deduped, kb, t_target, fk_threshold)
    if not verified:
        return [], True
    if max_solutions is not None:
        verified = verified[:max_solutions]
    return verified, False
