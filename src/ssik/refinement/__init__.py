"""Universal numerical-refinement layer.

Newton-on-spatial-Jacobian polish over the SE(3) log residual. Callable by
*any* solver that produces an algebraic candidate close to a true IK
solution: tier-1 univariate-search near a sample-grid miss, tier-2
bivariate grid-search final convergence, tier-2 Raghavan-Roth ill-
conditioned-pencil polish, and joint-locked 7R inner solver.

Design constraints (see GitHub #74):

- **Opt-in by default off.** Solvers' public ``solve()`` accepts
  ``allow_refinement: bool = False``. Default behaviour is pure algebraic
  -- candidates that don't already meet ``fk_atol`` get dropped, not
  polished. When ``True``, each near-miss gets one
  :func:`lm_refine` pass; the resulting :class:`~ssik.core.solution.Solution`
  reports ``refinement_used="lm"`` and ``refinement_iters``.
- **FK-tolerance-driven termination.** Iterate until ``||r|| < fk_atol``
  or ``max_iters`` hit. No divergence-abort heuristic: Newton trajectories
  can be non-monotonic near saddles / under step-clipping, and aggressive
  early termination misses real recoveries.
- **Analytical Jacobian preferred.** Callers pass ``jacobian_fn`` when
  they have a closed-form spatial Jacobian (DH chain, POE chain). When
  ``None``, central-differences fallback. Analytical is ~50x faster.
- **No scipy.** Hand-rolled Newton is one LAPACK ``solve`` per iter.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import cython
import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution

# 2*pi as a typed module-level constant -- referenced inside the dedup
# hot loop. Cython types this as a C ``double``; pure Python sees it as
# a regular ``float``.
_TWO_PI: float = 2.0 * math.pi

# Cached read-only 4x4 identity reused inside ``kinbody_jacobian``: each
# call avoids ``len(joints)+1`` per-iteration ``np.eye(4)`` allocations.
# Same pattern as the artifact orchestrator's ``_FK_EYE4`` (#146).
_FK_EYE4 = np.eye(4, dtype=np.float64)
_FK_EYE4.flags.writeable = False

__all__ = [
    "kinbody_jacobian",
    "lm_refine",
    "lm_refine_batch",
    "numerical_jacobian",
    "se3_log_residual",
    "verify_candidates",
]


def se3_log_residual(t_err: NDArray[np.float64]) -> NDArray[np.float64]:
    """6-vector residual ``(translation, rotation_axis-angle)`` for SE(3) error.

    For ``t_err = T_target @ FK(q)^{-1}``, returns the local twist
    coordinate that drives ``q`` toward the target. Translation is read
    directly; rotation uses the antisymmetric ``vee((R - R.T) / 2) =
    sin(angle) * axis`` formulation (precise at all scales, including
    near identity) combined with ``atan2(sin_angle, cos_angle)`` for
    the magnitude (well-conditioned at every angle).

    Three regimes:

    1. Generic (``sin_angle > 1e-9``): ``rot_err = (angle / sin_angle)
       * skew_vec``. The scale factor is well-defined and converges to
       1 as ``angle -> 0``.
    2. Near identity (``sin_angle <= 1e-9`` and ``cos_angle > 0``): the
       skew vector is itself the angle-scaled axis to first order
       (``skew_vec ≈ angle * axis``); return as-is.
    3. Near π (``sin_angle <= 1e-9`` and ``cos_angle < 0``): the skew
       vector vanishes; recover the rotation axis from the dominant
       eigenvector of ``R + I`` (which has rank 1 with eigenvalue 2
       along the axis at exactly π).

    This formulation fixes the precision-loss bug in the prior
    ``arccos(0.5 * (trace - 1))`` implementation, which silently zeroed
    the rotation residual when the rotation error was below ~3e-8 rad.
    See #199 for the bug + fix discussion. Newton trajectories that
    used to stop at ~1e-8 Frobenius FK now converge to machine
    precision.
    """
    trans_err = t_err[:3, 3]
    r_err = t_err[:3, :3]

    # vee((R - R.T) / 2) = sin(angle) * axis -- exact at all scales,
    # since float64 captures off-diagonal differences faithfully near
    # identity (the lossy step in the old formulation was the trace).
    skew_vec = np.array(
        [
            0.5 * (r_err[2, 1] - r_err[1, 2]),
            0.5 * (r_err[0, 2] - r_err[2, 0]),
            0.5 * (r_err[1, 0] - r_err[0, 1]),
        ]
    )
    sin_angle = float(np.linalg.norm(skew_vec))
    cos_angle = max(-1.0, min(1.0, 0.5 * (float(np.trace(r_err)) - 1.0)))

    # atan2 is well-conditioned at every angle (unlike arccos near 0 or π).
    angle = math.atan2(sin_angle, cos_angle)

    if sin_angle > 1e-9:
        rot_err = (angle / sin_angle) * skew_vec
    elif cos_angle > 0:
        # Near identity: skew_vec ≈ angle * axis to leading order.
        rot_err = skew_vec
    else:
        # Near π: skew_vec vanished; recover axis from R + I.
        r_plus_i = r_err + np.eye(3)
        norms = np.linalg.norm(r_plus_i, axis=0)
        idx = int(np.argmax(norms))
        rot_err = (angle / norms[idx]) * r_plus_i[:, idx]
    return np.concatenate([trans_err, rot_err])


def numerical_jacobian(
    q: NDArray[np.float64],
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    """Central-difference 6xN spatial Jacobian as a fallback when no
    analytical Jacobian is provided.

    Costs 2N FK evaluations per call; analytical Jacobians are typically
    ~50x faster. Used when a solver doesn't yet have an analytical
    Jacobian for its kinematic representation.
    """
    n = q.shape[0]
    j = np.zeros((6, n), dtype=np.float64)
    t_base = fk_fn(q)
    for i in range(n):
        q_p = q.copy()
        q_p[i] += eps
        q_m = q.copy()
        q_m[i] -= eps
        # Central-difference the SE(3) log residual contribution per axis.
        r_p = se3_log_residual(fk_fn(q_p) @ np.linalg.inv(t_base))
        r_m = se3_log_residual(fk_fn(q_m) @ np.linalg.inv(t_base))
        j[:, i] = (r_p - r_m) / (2.0 * eps)
    return j


@cython.ccall
@cython.locals(
    n=cython.int,
    i=cython.int,
    qi=cython.double,
    ax=cython.double,
    ay=cython.double,
    az=cython.double,
    norm=cython.double,
    inv_norm=cython.double,
    c=cython.double,
    s=cython.double,
    oc=cython.double,
    r00=cython.double,
    r01=cython.double,
    r02=cython.double,
    r10=cython.double,
    r11=cython.double,
    r12=cython.double,
    r20=cython.double,
    r21=cython.double,
    r22=cython.double,
    a00=cython.double,
    a01=cython.double,
    a02=cython.double,
    a03=cython.double,
    a10=cython.double,
    a11=cython.double,
    a12=cython.double,
    a13=cython.double,
    a20=cython.double,
    a21=cython.double,
    a22=cython.double,
    a23=cython.double,
    p00=cython.double,
    p01=cython.double,
    p02=cython.double,
    p03=cython.double,
    p10=cython.double,
    p11=cython.double,
    p12=cython.double,
    p13=cython.double,
    p20=cython.double,
    p21=cython.double,
    p22=cython.double,
    p23=cython.double,
    pm00=cython.double,
    pm01=cython.double,
    pm02=cython.double,
    pm03=cython.double,
    pm10=cython.double,
    pm11=cython.double,
    pm12=cython.double,
    pm13=cython.double,
    pm20=cython.double,
    pm21=cython.double,
    pm22=cython.double,
    pm23=cython.double,
    l00=cython.double,
    l01=cython.double,
    l02=cython.double,
    l03=cython.double,
    l10=cython.double,
    l11=cython.double,
    l12=cython.double,
    l13=cython.double,
    l20=cython.double,
    l21=cython.double,
    l22=cython.double,
    l23=cython.double,
    rt00=cython.double,
    rt01=cython.double,
    rt02=cython.double,
    rt03=cython.double,
    rt10=cython.double,
    rt11=cython.double,
    rt12=cython.double,
    rt13=cython.double,
    rt20=cython.double,
    rt21=cython.double,
    rt22=cython.double,
    rt23=cython.double,
    zx=cython.double,
    zy=cython.double,
    zz=cython.double,
    px=cython.double,
    py=cython.double,
    pz=cython.double,
)
def kinbody_jacobian(
    kb: object,  # ssik._kinbody.KinBody (avoid import cycle)
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Closed-form 6xN spatial Jacobian for a POE-form ``KinBody``.

    The spatial Jacobian relates joint rates to the spatial twist of
    the end-effector pose, ``T(q+dq) @ T(q)^{-1} ~ exp([J @ dq])``,
    matching the convention used by :func:`se3_log_residual` (which
    reads off the spatial twist of ``T_target @ T_q^{-1}``). For
    revolute joint i with axis ``z_i`` (world frame at ``q``) and
    origin ``p_i`` (world frame at ``q``)::

        J[:3, i] = -z_i x p_i = p_i x z_i    (linear, spatial)
        J[3:, i] =  z_i                      (angular, spatial)

    Note: this is *not* the so-called "geometric" or "hybrid" Jacobian
    ``z_i x (p_e - p_i)`` that maps to end-effector POINT velocity in
    world coordinates -- ``lm_refine`` consumes a spatial twist, so the
    spatial form is what's needed for Newton-on-SE(3)-log convergence.
    The two differ by ``z_i x p_e`` per column.

    Hand-rolled scalar-inline body (mirrors
    :func:`ssik.kinematics.poe_fk.poe_forward_kinematics` -- both walk
    the same chain). The accumulator is carried as 12 doubles (the
    bottom row ``[0, 0, 0, 1]`` is implicit); per-joint cost is two
    inlined 4x4 matmuls plus a 3-vec rotation, with no per-iteration
    numpy allocations or dispatch. On Franka 7R the inner ``lm_refine``
    iter drops from 144 us (numpy-heavy body) to ~25 us; verify_candidates
    drops from 21 ms to ~6 ms for the locked-Franka HP path.
    """
    joints = kb.joints  # type: ignore[attr-defined]
    n = len(joints)
    j_arr = np.zeros((6, n), dtype=np.float64)
    # Accumulator T_acc: top 3 rows of a 4x4 (bottom row is implicit
    # [0, 0, 0, 1]). Initialised to identity.
    a00 = 1.0
    a01 = 0.0
    a02 = 0.0
    a03 = 0.0
    a10 = 0.0
    a11 = 1.0
    a12 = 0.0
    a13 = 0.0
    a20 = 0.0
    a21 = 0.0
    a22 = 1.0
    a23 = 0.0
    for i in range(n):
        joint = joints[i]
        axis = joint.axis
        ax = float(axis[0])
        ay = float(axis[1])
        az = float(axis[2])
        norm = (ax * ax + ay * ay + az * az) ** 0.5
        inv_norm = 1.0 / norm
        ax = ax * inv_norm
        ay = ay * inv_norm
        az = az * inv_norm
        # T_left scalars.
        Tl = joint.T_left
        l00 = float(Tl[0, 0])
        l01 = float(Tl[0, 1])
        l02 = float(Tl[0, 2])
        l03 = float(Tl[0, 3])
        l10 = float(Tl[1, 0])
        l11 = float(Tl[1, 1])
        l12 = float(Tl[1, 2])
        l13 = float(Tl[1, 3])
        l20 = float(Tl[2, 0])
        l21 = float(Tl[2, 1])
        l22 = float(Tl[2, 2])
        l23 = float(Tl[2, 3])
        # P = T_acc @ T_left  (frame just BEFORE joint i acts in world).
        # Top 3 rows of homogeneous matrix product; bottom row is implicit.
        p00 = a00 * l00 + a01 * l10 + a02 * l20
        p01 = a00 * l01 + a01 * l11 + a02 * l21
        p02 = a00 * l02 + a01 * l12 + a02 * l22
        p03 = a00 * l03 + a01 * l13 + a02 * l23 + a03
        p10 = a10 * l00 + a11 * l10 + a12 * l20
        p11 = a10 * l01 + a11 * l11 + a12 * l21
        p12 = a10 * l02 + a11 * l12 + a12 * l22
        p13 = a10 * l03 + a11 * l13 + a12 * l23 + a13
        p20 = a20 * l00 + a21 * l10 + a22 * l20
        p21 = a20 * l01 + a21 * l11 + a22 * l21
        p22 = a20 * l02 + a21 * l12 + a22 * l22
        p23 = a20 * l03 + a21 * l13 + a22 * l23 + a23
        # z_i = R_pre @ axis_unit (joint axis in world frame at q).
        zx = p00 * ax + p01 * ay + p02 * az
        zy = p10 * ax + p11 * ay + p12 * az
        zz = p20 * ax + p21 * ay + p22 * az
        # p_i = translation of P (joint origin in world frame at q).
        px = p03
        py = p13
        pz = p23
        # Spatial Jacobian column: linear = p_i x z_i, angular = z_i.
        j_arr[0, i] = py * zz - pz * zy
        j_arr[1, i] = pz * zx - px * zz
        j_arr[2, i] = px * zy - py * zx
        j_arr[3, i] = zx
        j_arr[4, i] = zy
        j_arr[5, i] = zz
        # Now advance T_acc. For revolute: T_acc_new = P @ R(axis, q[i])
        # @ T_right. Build Rodrigues 3x3 and inline the two matmuls.
        if joint.joint_type == "prismatic":
            # Joint contribution: translation by q[i] along axis. Equivalent
            # to T_acc_new[:3,:3] = P[:3,:3], translation += P[:3,:3] @ (q*axis).
            qi = float(q[i])
            pm00 = p00
            pm01 = p01
            pm02 = p02
            pm03 = p03 + qi * (p00 * ax + p01 * ay + p02 * az)
            pm10 = p10
            pm11 = p11
            pm12 = p12
            pm13 = p13 + qi * (p10 * ax + p11 * ay + p12 * az)
            pm20 = p20
            pm21 = p21
            pm22 = p22
            pm23 = p23 + qi * (p20 * ax + p21 * ay + p22 * az)
        else:
            qi = float(q[i])
            c = math.cos(qi)
            s = math.sin(qi)
            oc = 1.0 - c
            r00 = c + ax * ax * oc
            r01 = ax * ay * oc - az * s
            r02 = ax * az * oc + ay * s
            r10 = ay * ax * oc + az * s
            r11 = c + ay * ay * oc
            r12 = ay * az * oc - ax * s
            r20 = az * ax * oc - ay * s
            r21 = az * ay * oc + ax * s
            r22 = c + az * az * oc
            # PM = P @ R (only the 3x3 rotation block; translation
            # column unchanged because R has trivial translation).
            pm00 = p00 * r00 + p01 * r10 + p02 * r20
            pm01 = p00 * r01 + p01 * r11 + p02 * r21
            pm02 = p00 * r02 + p01 * r12 + p02 * r22
            pm03 = p03
            pm10 = p10 * r00 + p11 * r10 + p12 * r20
            pm11 = p10 * r01 + p11 * r11 + p12 * r21
            pm12 = p10 * r02 + p11 * r12 + p12 * r22
            pm13 = p13
            pm20 = p20 * r00 + p21 * r10 + p22 * r20
            pm21 = p20 * r01 + p21 * r11 + p22 * r21
            pm22 = p20 * r02 + p21 * r12 + p22 * r22
            pm23 = p23
        # T_right scalars.
        Tr = joint.T_right
        rt00 = float(Tr[0, 0])
        rt01 = float(Tr[0, 1])
        rt02 = float(Tr[0, 2])
        rt03 = float(Tr[0, 3])
        rt10 = float(Tr[1, 0])
        rt11 = float(Tr[1, 1])
        rt12 = float(Tr[1, 2])
        rt13 = float(Tr[1, 3])
        rt20 = float(Tr[2, 0])
        rt21 = float(Tr[2, 1])
        rt22 = float(Tr[2, 2])
        rt23 = float(Tr[2, 3])
        # T_acc_new = PM @ T_right.
        a00 = pm00 * rt00 + pm01 * rt10 + pm02 * rt20
        a01 = pm00 * rt01 + pm01 * rt11 + pm02 * rt21
        a02 = pm00 * rt02 + pm01 * rt12 + pm02 * rt22
        a03 = pm00 * rt03 + pm01 * rt13 + pm02 * rt23 + pm03
        a10 = pm10 * rt00 + pm11 * rt10 + pm12 * rt20
        a11 = pm10 * rt01 + pm11 * rt11 + pm12 * rt21
        a12 = pm10 * rt02 + pm11 * rt12 + pm12 * rt22
        a13 = pm10 * rt03 + pm11 * rt13 + pm12 * rt23 + pm13
        a20 = pm20 * rt00 + pm21 * rt10 + pm22 * rt20
        a21 = pm20 * rt01 + pm21 * rt11 + pm22 * rt21
        a22 = pm20 * rt02 + pm21 * rt12 + pm22 * rt22
        a23 = pm20 * rt03 + pm21 * rt13 + pm22 * rt23 + pm23
    return j_arr


def lm_refine(
    q_seed: NDArray[np.float64],
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t_target: NDArray[np.float64],
    *,
    fk_atol: float = 1e-9,
    max_iters: int = 15,
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    step_clip: float = 0.5,
    divergence_factor: float = 5.0,
    divergence_min_iters: int = 4,
) -> tuple[NDArray[np.float64], float, int] | None:
    """Newton-Raphson polish on the SE(3) log residual.

    For seeds within ~30 deg of a true solution, converges to machine
    precision in 1-5 iterations.

    :param q_seed: starting joint vector. Length matches ``fk_fn`` input.
    :param fk_fn: ``q -> 4x4`` forward kinematics callable.
    :param t_target: 4x4 target end-effector pose.
    :param fk_atol: convergence threshold on ``||se3_log_residual||``.
    :param max_iters: cap. If hit without convergence, returns ``None``.
    :param jacobian_fn: optional ``q -> 6xN`` spatial Jacobian callable.
        When ``None``, central-difference numeric Jacobian is used (slow).
    :param step_clip: per-component absolute clip on the Newton step;
        keeps trajectories from overshooting in the linearisation regime.
    :param divergence_factor: early-abort multiplier applied to the
        best-so-far residual. After ``divergence_min_iters`` iterations,
        if ``||r|| > divergence_factor * r_best`` the trajectory is
        clearly outside the basin of attraction and Newton is aborted
        with a ``None`` return. Set to ``inf`` to disable. Default
        ``5.0`` -- empirically convergent trajectories from algebraic
        seeds stay within ~1.5x of best-so-far on locked-Franka
        multiplicity-4 clusters; divergers exceed 5x within 5 iters.
    :param divergence_min_iters: number of iterations before the
        divergence check arms (default 4). Allows the first uphill
        step a converger sometimes takes (~30% bump on a hard seed)
        without triggering abort.
    :returns: ``(q_refined, fk_residual, iters)`` on convergence,
        ``None`` if ``max_iters`` was reached without ``||r|| < fk_atol``
        or the divergence guard fired.

    On divergent seeds (algebraic spurious roots, multi-root cluster
    members from a different IK branch, etc.) the early-abort saves
    ~10 wasted Newton iterations per seed -- on locked-Franka HP that
    is the difference between 39 ms and 13 ms in ``verify_candidates``.
    """
    q = q_seed.astype(np.float64).copy()
    r_best: float = float("inf")
    for it in range(max_iters):
        t_q = fk_fn(q)
        t_diff = t_target @ np.linalg.inv(t_q)
        r = se3_log_residual(t_diff)
        norm = float(np.linalg.norm(r))
        if norm < fk_atol:
            return (q, norm, it)
        if norm < r_best:
            r_best = norm
        elif it >= divergence_min_iters and norm > divergence_factor * r_best:
            # Trajectory is clearly outside the basin of attraction.
            # See divergence_factor docstring for the rationale.
            return None
        j_s = jacobian_fn(q) if jacobian_fn is not None else numerical_jacobian(q, fk_fn)
        try:
            dq = np.linalg.solve(j_s, r)
        except np.linalg.LinAlgError:
            # Singular Jacobian (kinematic singularity) -> Tikhonov-damped LSQ.
            damping = max(1e-9, 1e-6 * norm)
            n = j_s.shape[1]
            jtj = j_s.T @ j_s + damping * np.eye(n)
            dq = np.linalg.solve(jtj, j_s.T @ r)
        dq = np.clip(dq, -step_clip, step_clip)
        q = q + dq
    # Final convergence check after max_iters.
    t_check = fk_fn(q)
    final_r = float(np.linalg.norm(se3_log_residual(t_target @ np.linalg.inv(t_check))))
    if final_r > fk_atol:
        return None
    return (q, final_r, max_iters)


def _se3_log_residual_batch(t_err: NDArray[np.float64]) -> NDArray[np.float64]:
    """Batched 6-vector SE(3) log residual for a stack of error matrices.

    :param t_err: shape ``(N, 4, 4)``.
    :returns: shape ``(N, 6)`` where columns 0-2 are translation and 3-5
        rotation axis-angle, computed via the same antisymmetric-vee +
        atan2 formulation as :func:`se3_log_residual` but vectorised.

    This is the batched twin of :func:`se3_log_residual` (post-#199 fix
    that replaced the lossy trace-arccos formulation). Used by
    :func:`lm_refine_batch` to compute residuals for N candidates at
    once.
    """
    n = t_err.shape[0]
    trans_err = t_err[:, :3, 3]  # (N, 3)
    r_err = t_err[:, :3, :3]  # (N, 3, 3)

    skew = 0.5 * np.stack(
        [
            r_err[:, 2, 1] - r_err[:, 1, 2],
            r_err[:, 0, 2] - r_err[:, 2, 0],
            r_err[:, 1, 0] - r_err[:, 0, 1],
        ],
        axis=-1,
    )  # (N, 3)
    sin_angle = np.linalg.norm(skew, axis=1)  # (N,)
    cos_angle = np.clip(0.5 * (np.trace(r_err, axis1=1, axis2=2) - 1.0), -1.0, 1.0)
    angle = np.arctan2(sin_angle, cos_angle)  # (N,)

    rot_err = np.empty((n, 3), dtype=np.float64)
    # Generic regime: sin_angle > 1e-9
    generic_mask = sin_angle > 1e-9
    if generic_mask.any():
        scale = (angle[generic_mask] / sin_angle[generic_mask])[:, None]
        rot_err[generic_mask] = scale * skew[generic_mask]
    # Near identity (sin_angle <= 1e-9, cos_angle > 0)
    near_id_mask = (~generic_mask) & (cos_angle > 0)
    if near_id_mask.any():
        rot_err[near_id_mask] = skew[near_id_mask]
    # Near pi (sin_angle <= 1e-9, cos_angle < 0)
    near_pi_mask = (~generic_mask) & (cos_angle <= 0)
    if near_pi_mask.any():
        # Per-element fallback (rare; near-pi candidates are spurious in
        # most polish trajectories).
        for idx in np.where(near_pi_mask)[0]:
            r_plus_i = r_err[idx] + np.eye(3)
            norms = np.linalg.norm(r_plus_i, axis=0)
            i = int(np.argmax(norms))
            rot_err[idx] = (angle[idx] / norms[i]) * r_plus_i[:, i]

    return np.concatenate([trans_err, rot_err], axis=1)


def lm_refine_batch(
    q_seeds: NDArray[np.float64],
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t_target: NDArray[np.float64],
    *,
    fk_atol: float = 1e-9,
    max_iters: int = 30,
    step_clip: float = 0.5,
    divergence_factor: float = 2.0,
    divergence_min_iters: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Batched Newton polish for ``N`` seeds against a single target T.

    Synchronises the iter loop across all candidates so per-iteration
    Python dispatch overhead in ``np.linalg.{solve, inv, norm}`` amortises
    over ``N``. ``fk_fn`` and ``jacobian_fn`` are still called per-
    candidate per-iter (no batched FK at the kinematics level), but the
    linear-algebra portion runs as one batched operation per iteration.

    Empirically ~30-50% faster than the sequential :func:`lm_refine`
    loop in :mod:`ssik.solvers.seven_r.srs_polished` on Gen3 (75-128
    candidates per IK call).

    :param q_seeds: ``(N, dof)`` array of initial joint vectors.
    :param fk_fn: ``q -> 4x4`` forward kinematics callable, scalar input.
    :param jacobian_fn: ``q -> 6xN_dof`` spatial Jacobian, scalar input.
    :param t_target: ``(4, 4)`` target end-effector pose.
    :param fk_atol: Frobenius residual threshold for convergence (the
        contract metric the rest of ssik uses for ``Solution.fk_residual``).
    :param max_iters: per-candidate iteration cap.
    :param step_clip: per-component absolute clip on the Newton step.
    :param divergence_factor: early-abort multiplier; same semantics as
        :func:`lm_refine`. Default 2.0 matches the seven_r.srs_polished
        tuning from #203.
    :param divergence_min_iters: armed-after iter count.

    :returns: ``(q_polished, fk_residuals, iters_used)`` where
        ``fk_residuals[i] < fk_atol`` indicates candidate ``i`` converged;
        otherwise the candidate is unconverged or diverged. Callers gate
        on the residual.
    """
    q_seeds = np.asarray(q_seeds, dtype=np.float64)
    if q_seeds.ndim != 2:
        raise ValueError(f"q_seeds must be (N, dof); got shape {q_seeds.shape}")
    n, dof = q_seeds.shape
    q = q_seeds.copy()
    active = np.ones(n, dtype=bool)
    r_best = np.full(n, np.inf)
    fk_residuals = np.full(n, np.inf)
    iters_used = np.zeros(n, dtype=np.intp)

    # Pre-allocate stacked workspace buffers.
    fk_arr = np.empty((n, 4, 4), dtype=np.float64)
    jac_arr = np.empty((n, 6, dof), dtype=np.float64)
    eye_dof = np.eye(dof, dtype=np.float64)

    for it in range(max_iters):
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            break

        # Sequential FK + Jacobian per active candidate. Batching this
        # would need batched poe_forward_kinematics + kinbody_jacobian
        # (Cython rewrite). The Python dispatch overhead per call is
        # ~3-5 us; total per pose-call is ~150 ms wall, of which most
        # is actual Cython compute. Batched linalg below saves the
        # other ~150 ms of np.linalg dispatch.
        for idx in active_idx:
            fk_arr[idx] = fk_fn(q[idx])
            jac_arr[idx] = jacobian_fn(q[idx])

        # Batched Frobenius residual (the contract metric).
        diff = fk_arr[active_idx] - t_target
        frob = np.linalg.norm(diff.reshape(active_idx.size, -1), axis=1)

        # Per-candidate convergence + divergence check.
        for k, idx in enumerate(active_idx):
            r = float(frob[k])
            if r < fk_atol:
                fk_residuals[idx] = r
                iters_used[idx] = it
                active[idx] = False
                continue
            if r < r_best[idx]:
                r_best[idx] = r
            elif it >= divergence_min_iters and r > divergence_factor * r_best[idx]:
                # Diverged: stop tracking this candidate.
                fk_residuals[idx] = np.inf
                iters_used[idx] = it
                active[idx] = False

        # Recompute the still-active subset for this iter's Newton step.
        active_now = active[active_idx]
        step_idx = active_idx[active_now]
        if step_idx.size == 0:
            continue

        fk_step = fk_arr[step_idx]
        jac_step = jac_arr[step_idx]

        # Batched twist computation: T_err = t_target @ inv(fk).
        # np.linalg.inv broadcasts over the leading dim.
        t_err = t_target @ np.linalg.inv(fk_step)
        twist = _se3_log_residual_batch(t_err)  # (N_step, 6)

        # Batched Newton step via normal equations:
        #   dq = (J^T @ J + lambda * I)^{-1} @ J^T @ r
        # Using normal equations (instead of lstsq) lets us batch via
        # np.linalg.solve, which broadcasts across the leading batch dim.
        jtj = np.einsum("nji,njk->nik", jac_step, jac_step)  # (N_step, dof, dof)
        # Tikhonov damping: small fixed value handles near-singular Jacobians
        # without per-iter conditioning probes.
        jtj_damped = jtj + 1e-9 * eye_dof
        jtr = np.einsum("nji,nj->ni", jac_step, twist)  # (N_step, dof)
        # numpy.linalg.solve with batched LHS (N, dof, dof) and RHS as
        # (N, dof, 1) returns (N, dof, 1); squeeze back to (N, dof).
        dq = np.linalg.solve(jtj_damped, jtr[..., None])[..., 0]
        dq = np.clip(dq, -step_clip, step_clip)

        q[step_idx] += dq

    # Final pass for candidates that exhausted max_iters: record their
    # current Frobenius residual (may be > fk_atol; caller decides).
    for idx in np.where(active)[0]:
        T_check = fk_fn(q[idx])
        fk_residuals[idx] = float(np.linalg.norm(T_check - t_target))
        iters_used[idx] = max_iters
        active[idx] = False

    return q, fk_residuals, iters_used


@cython.ccall
@cython.locals(
    i=cython.int,
    n=cython.int,
    diff=cython.double,
    ai=cython.double,
    bi=cython.double,
)
def _q_close(a: NDArray[np.float64], b: NDArray[np.float64], tol: float) -> bool:
    """Element-wise wrap-to-pi closeness for joint-angle vectors.

    Early-exits on the first per-element mismatch -- when ``a`` and ``b``
    are clearly not the same solution (mod 2pi) the typical case bails
    out within 1-2 elements, much cheaper than a full broadcasted compare.
    Cython compiles this to a typed scalar loop with no Python-object
    boxing in the inner arithmetic.
    """
    n = len(a)
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        diff = ((ai - bi + math.pi) % _TWO_PI) - math.pi
        if abs(diff) > tol:
            return False
    return True


def _dedup_scalar(candidates: list[Solution], tol: float) -> list[Solution]:
    """Pairwise scan with early-exit -- fast path under Cython compile.

    Inner per-pair check via :func:`_q_close` short-circuits on first
    mismatched joint. Cython compiles the scalar arithmetic to a typed
    C loop; the inner cost is near-memcpy when most pairs disagree
    quickly.
    """
    deduped: list[Solution] = []
    for cand in candidates:
        match_idx = -1
        for j, existing in enumerate(deduped):
            if _q_close(cand.q, existing.q, tol):
                match_idx = j
                break
        if match_idx == -1:
            deduped.append(cand)
        elif cand.fk_residual < deduped[match_idx].fk_residual:
            deduped[match_idx] = cand
    return deduped


def _dedup_numpy(candidates: list[Solution], tol: float) -> list[Solution]:
    """Numpy-broadcast all-pairs compare -- fast path under pure Python.

    For untyped Python, batching the wrap-to-pi compare through one
    broadcasted ``mod 2pi`` over an ``(M, N_dof)`` array beats the
    per-element scalar loop because the per-numpy-call overhead amortises
    over all M existing candidates. Pure-Python interpretation of the
    scalar loop has too much per-iteration overhead to compete.
    """
    deduped: list[Solution] = []
    n_dof = candidates[0].q.shape[0]
    arr = np.empty((0, n_dof), dtype=np.float64)
    for cand in candidates:
        if arr.shape[0] > 0:
            diffs = (cand.q - arr + np.pi) % (2 * np.pi) - np.pi
            matches = np.all(np.abs(diffs) < tol, axis=1)
            idxs = np.where(matches)[0]
            if len(idxs) > 0:
                j = int(idxs[0])
                if cand.fk_residual < deduped[j].fk_residual:
                    deduped[j] = cand
                    arr[j] = cand.q
                continue
        deduped.append(cand)
        arr = np.vstack([arr, cand.q[np.newaxis, :]])
    return deduped


def dedup_by_wrap_close(candidates: list[Solution], tol: float) -> list[Solution]:
    """Dedup :class:`Solution` candidates by wrap-to-pi joint-angle closeness.

    For each cluster of solutions whose joint vectors agree (mod 2pi) within
    ``tol`` per joint, keeps the candidate with the lowest ``fk_residual``.
    Streaming order matches the first-match-wins semantics so output
    ordering is stable.

    Two implementations under one entry point. Cython compiles
    ``cython.compiled`` to ``True`` and dead-strips the other branch:

    - Compiled: :func:`_dedup_scalar`. Per-pair early-exit at C scalar
      speed; ~25% faster than the numpy-broadcast variant on realistic
      Franka 7R workloads (200 cands -> 64 unique: 1.25 ms vs 1.62 ms).
    - Pure Python: :func:`_dedup_numpy`. Numpy-broadcast all-pairs
      compare; faster than the scalar loop in interpreted form because
      the per-numpy-call overhead amortises over M existing candidates.

    Without this dispatch, picking either implementation regresses one
    of the two install paths (wheel users vs source-install users).
    """
    if not candidates:
        return []
    if cython.compiled:
        return _dedup_scalar(candidates, tol)
    return _dedup_numpy(candidates, tol)


def verify_candidates(
    candidates: Iterable[NDArray[np.float64]],
    *,
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t_target: NDArray[np.float64],
    fk_atol: float,
    solver_name: str = "",  # accepted for back-compat; v1.0 drops it from Solution
    dedup_atol: float | None = None,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    max_solutions: int | None = None,
) -> list[Solution]:
    """FK-verify, optionally polish, and dedup IK candidates.

    Common back-half of every ssik solver:

    1. Iterate ``candidates``. For each ``q``, compute ``fk_residual =
       ||fk_fn(q) - t_target||_F``.
    2. ``fk_residual <= fk_atol``: wrap in :class:`Solution` with
       ``refinement_used="none"``.
    3. ``fk_residual > fk_atol`` AND ``allow_refinement=True``: run one
       :func:`lm_refine` pass; on success wrap with ``refinement_used="lm"``.
       On failure or when ``allow_refinement=False``: drop the candidate.
    4. If ``dedup_atol`` is given, collapse solutions whose joint vectors
       agree (mod 2pi) within ``dedup_atol`` per joint -- keep the one
       with lower ``fk_residual``.

    The pattern is identical across every solver in
    :mod:`ssik.solvers.ikgeo` and :mod:`ssik.solvers.jointlock`; centralising
    it here keeps one source of truth for the algebraic-first / opt-in-
    refinement / dedup-by-residual contract that GitHub #74 specifies.

    :param candidates: raw joint vectors produced by the solver's algebraic
        machinery (typically the SP1-SP6 back-substitution branches).
    :param fk_fn: ``q -> 4x4`` forward kinematics (POE chain or DH chain).
    :param t_target: 4x4 target pose in the FK frame.
    :param fk_atol: closure threshold in Frobenius norm.
    :param solver_name: tag stored on each returned :class:`Solution`.
    :param dedup_atol: per-joint wrap-to-pi tolerance for collapsing
        equivalent solutions. ``None`` skips deduplication (useful when the
        solver has its own dedup invariant).
    :param allow_refinement: opt into Newton polish for near-misses.
    :param refinement_max_iters: cap on Newton iterations per candidate.
    :param jacobian_fn: optional analytical Jacobian for ``lm_refine``.
    :param max_solutions: optional early-exit cap (#198). When set, stop
        iterating ``candidates`` once the deduped count reaches the cap.
        Default ``None`` preserves full enumeration. The check runs after
        every appended candidate; the post-dedup gate guarantees the cap
        is met by *unique* solutions.
    """
    if max_solutions is not None and max_solutions < 1:
        raise ValueError(f"max_solutions must be >= 1 or None; got {max_solutions}")
    verified: list[Solution] = []
    for q in candidates:
        q = np.asarray(q, dtype=np.float64)
        fk_resid = float(np.linalg.norm(fk_fn(q) - t_target))
        appended = False
        if fk_resid <= fk_atol:
            verified.append(
                Solution(
                    q=q,
                    fk_residual=fk_resid,
                    refinement_used="none",
                )
            )
            appended = True
        elif allow_refinement:
            refined = lm_refine(
                q,
                fk_fn,
                t_target,
                fk_atol=fk_atol,
                max_iters=refinement_max_iters,
                jacobian_fn=jacobian_fn,
            )
            if refined is not None:
                q_ref, resid, _iters = refined
                verified.append(
                    Solution(
                        q=q_ref,
                        fk_residual=resid,
                        refinement_used="lm",
                    )
                )
                appended = True

        # Early-exit gate (#198): once we have the requested number of
        # *unique* solutions, stop iterating remaining candidates. The
        # post-dedup check is mandatory because algebraic enumeration
        # often emits duplicate q-vectors across branches.
        if appended and max_solutions is not None and len(verified) >= max_solutions:
            if dedup_atol is None:
                return verified[:max_solutions]
            deduped = dedup_by_wrap_close(verified, dedup_atol)
            if len(deduped) >= max_solutions:
                return deduped[:max_solutions]

    if dedup_atol is None:
        if max_solutions is not None:
            return verified[:max_solutions]
        return verified
    deduped_final = dedup_by_wrap_close(verified, dedup_atol)
    if max_solutions is not None:
        return deduped_final[:max_solutions]
    return deduped_final
