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

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution

__all__ = [
    "kinbody_jacobian",
    "lm_refine",
    "numerical_jacobian",
    "se3_log_residual",
    "verify_candidates",
]


def se3_log_residual(t_err: NDArray[np.float64]) -> NDArray[np.float64]:
    """6-vector residual ``(translation, rotation_axis-angle)`` for SE(3) error.

    For ``t_err = T_target @ FK(q)^{-1}``, returns the local twist
    coordinate that drives ``q`` toward the target. Translation is read
    directly; rotation is via Rodrigues' formula ``log(R) -> axis*angle``
    with cosine clamping for numerical safety near identity.
    """
    trans_err = t_err[:3, 3]
    r_err = t_err[:3, :3]
    cos_a = max(-1.0, min(1.0, 0.5 * (np.trace(r_err) - 1.0)))
    angle = float(np.arccos(cos_a))
    if angle < 1e-9:
        rot_err = np.zeros(3)
    else:
        s = 1.0 / (2.0 * np.sin(angle))
        rot_err = np.array(
            [
                s * (r_err[2, 1] - r_err[1, 2]) * angle,
                s * (r_err[0, 2] - r_err[2, 0]) * angle,
                s * (r_err[1, 0] - r_err[0, 1]) * angle,
            ]
        )
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


def kinbody_jacobian(
    kb: object,  # ssik._kinbody.KinBody (avoid import cycle)
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Closed-form 6xN spatial Jacobian for a POE-form ``KinBody``.

    For each joint i, the column is the i-th screw axis in the world
    frame at config ``q``::

        J[:3, i] = z_i x (p_e - p_i)    (linear-velocity component)
        J[3:, i] = z_i                  (angular-velocity component)

    where ``z_i`` is the joint-i axis in the world frame at ``q``, and
    ``p_i`` / ``p_e`` are the joint-i origin and end-effector position
    respectively.
    """
    joints = kb.joints  # type: ignore[attr-defined]
    n = len(joints)
    cum: list[NDArray[np.float64]] = [np.eye(4, dtype=np.float64)]
    for joint, qi in zip(joints, q, strict=True):
        rot = np.eye(4, dtype=np.float64)
        c = float(np.cos(float(qi)))
        s = float(np.sin(float(qi)))
        ax = joint.axis / np.linalg.norm(joint.axis)
        x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
        oc = 1.0 - c
        rot[:3, :3] = np.array(
            [
                [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
                [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
                [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
            ]
        )
        cum.append(cum[-1] @ joint.T_left @ rot @ joint.T_right)
    p_e = cum[-1][:3, 3]
    j = np.zeros((6, n), dtype=np.float64)
    for i, joint in enumerate(joints):
        # Frame just *before* joint i acts: cum[i] @ T_left[i].
        t_pre = cum[i] @ joint.T_left
        z_i = t_pre[:3, :3] @ (joint.axis / np.linalg.norm(joint.axis))
        p_i = t_pre[:3, 3]
        j[:3, i] = np.cross(z_i, p_e - p_i)
        j[3:, i] = z_i
    return j


def lm_refine(
    q_seed: NDArray[np.float64],
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t_target: NDArray[np.float64],
    *,
    fk_atol: float = 1e-9,
    max_iters: int = 15,
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    step_clip: float = 0.5,
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
    :returns: ``(q_refined, fk_residual, iters)`` on convergence,
        ``None`` if ``max_iters`` was reached without ``||r|| < fk_atol``.

    No divergence-abort: Newton can be non-monotonic near a saddle or
    under step-clipping; aggressive early termination misses recoveries.
    Trust ``max_iters`` + final residual check instead.
    """
    q = q_seed.astype(np.float64).copy()
    for it in range(max_iters):
        t_q = fk_fn(q)
        t_diff = t_target @ np.linalg.inv(t_q)
        r = se3_log_residual(t_diff)
        norm = float(np.linalg.norm(r))
        if norm < fk_atol:
            return (q, norm, it)
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


def _q_close(a: NDArray[np.float64], b: NDArray[np.float64], tol: float) -> bool:
    """Element-wise wrap-to-pi closeness for joint-angle vectors."""
    for ai, bi in zip(a, b, strict=True):
        diff = float(((float(ai) - float(bi) + np.pi) % (2 * np.pi)) - np.pi)
        if abs(diff) > tol:
            return False
    return True


def verify_candidates(
    candidates: Iterable[NDArray[np.float64]],
    *,
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t_target: NDArray[np.float64],
    fk_atol: float,
    solver_name: str,
    dedup_atol: float | None = None,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
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
    """
    verified: list[Solution] = []
    for branch_idx, q in enumerate(candidates):
        q = np.asarray(q, dtype=np.float64)
        fk_resid = float(np.linalg.norm(fk_fn(q) - t_target))
        if fk_resid <= fk_atol:
            verified.append(
                Solution(
                    q=q,
                    fk_residual=fk_resid,
                    refinement_used="none",
                    refinement_iters=0,
                    branch_id=branch_idx,
                    solver_name=solver_name,
                )
            )
            continue
        if not allow_refinement:
            continue
        refined = lm_refine(
            q,
            fk_fn,
            t_target,
            fk_atol=fk_atol,
            max_iters=refinement_max_iters,
            jacobian_fn=jacobian_fn,
        )
        if refined is None:
            continue
        q_ref, resid, iters = refined
        verified.append(
            Solution(
                q=q_ref,
                fk_residual=resid,
                refinement_used="lm",
                refinement_iters=iters,
                branch_id=branch_idx,
                solver_name=solver_name,
            )
        )

    if dedup_atol is None:
        return verified
    deduped: list[Solution] = []
    for cand in verified:
        dup_idx: int | None = None
        for j, existing in enumerate(deduped):
            if _q_close(cand.q, existing.q, dedup_atol):
                dup_idx = j
                break
        if dup_idx is None:
            deduped.append(cand)
        elif cand.fk_residual < deduped[dup_idx].fk_residual:
            deduped[dup_idx] = cand
    return deduped
