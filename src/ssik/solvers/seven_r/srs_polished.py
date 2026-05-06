"""Approximate-SRS 7R analytical IK + LM polish.

Wraps :mod:`ssik.solvers.seven_r.srs` (Singh-Kreutz, strict-SRS) for
arms whose URDF axes don't *exactly* meet at common shoulder/wrist
points but whose drift is small enough to fit inside Newton's basin
of attraction (~3-5 cm task space empirically, gated by
``max_drift_m``).

Algorithm:

1. Run :func:`ssik.kinematics.predicates.is_approximately_srs_7r` to
   accept the chain (refusing any arm whose drift exceeds the basin).
2. Pass the relaxed-policy SRS classification to
   :func:`ssik.solvers.seven_r.srs.solve` with a permissive ``fk_atol``
   so it returns all algebraic candidates -- the candidates' FK
   residuals are ~``max_drift_m`` because the solver assumes axes
   meet exactly.
3. LM-polish each candidate against the **original** (non-snapped)
   URDF FK. Newton converges in 4-15 iterations from any seed inside
   the basin; divergent seeds are dropped.
4. Cluster-merge to drop duplicate IKs that polished into the same
   solution (different SRS branches may collapse under perturbation).

Targets:

- **Kinova Gen3 7-DOF**: 12 mm shoulder + 0.4 mm wrist drift -- 16-30x
  faster than ``jointlock + HP`` at machine-precision FK closure.
- Future arms with similar drift profiles (auto via predicate; no
  per-arm hardcoding).

Refused (drift exceeds gate; falls back to ``jointlock + HP``):

- Flexiv Rizon 4 / 10 (151 mm wrist drift).
- Kassow KR810 (111 mm wrist drift).
- Anthropomorphic 7R like Franka Panda (wrist axes don't meet at
  one point in any home configuration).

References:

- Original Singh-Kreutz solver: :mod:`ssik.solvers.seven_r.srs`.
- Refinement layer: :mod:`ssik.refinement` (LM polish, kinbody
  Jacobian).
- HP locked-7R perturbation pattern (precedent for "approximate
  algebra + LM polish"): see :mod:`ssik.solvers.husty_pfurner`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import is_approximately_srs_7r
from ssik.refinement import dedup_by_wrap_close, kinbody_jacobian
from ssik.solvers.seven_r import srs

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from ssik._kinbody import KinBody

__all__ = ["solve"]

_SOLVER_NAME = "seven_r.srs_polished"
_LOG = logging.getLogger(__name__)
_DEFAULT_MAX_DRIFT_M = 0.04


def _polish_to_frobenius(
    q_seed: NDArray[np.float64],
    kb: KinBody,
    T_target: NDArray[np.float64],
    *,
    fk_atol: float,
    max_iters: int,
    step_clip: float = 0.5,
    divergence_factor: float = 5.0,
) -> tuple[NDArray[np.float64], float, int] | None:
    """Newton polish on the spatial twist of ``T_target @ T_q^{-1}``,
    convergent on the **Frobenius** ``||FK(q) - T_target||_F`` norm.

    This is a workaround for the near-identity precision loss in
    :func:`ssik.refinement.se3_log_residual` (the ``cos_a = (trace-1)/2``
    formulation rounds to ``1.0`` in float64 when the rotation error is
    below ~3e-8, silently zeroing the rotation part of the residual).
    See #199.

    The fix: compute the rotation part of the twist via the
    skew-symmetric vee of ``R_err - R_err.T``, which preserves
    precision down to machine epsilon. Translation is read directly
    from ``T_err``. The convergence gate is the actual Frobenius FK
    residual, so candidates can't slip through with hidden rotation
    error.
    """
    q = q_seed.astype(np.float64).copy()
    r_best = float("inf")
    for it in range(max_iters):
        T_q = poe_forward_kinematics(kb, q)
        # Frobenius gate -- the contract callers expect.
        frob = float(np.linalg.norm(T_q - T_target))
        if frob < fk_atol:
            return q, frob, it
        if frob < r_best:
            r_best = frob
        elif it >= 4 and frob > divergence_factor * r_best:
            return None
        # Robust spatial twist: read translation directly, rotation via
        # skew-vee of (R_err - R_err.T) / 2 (works at machine precision
        # near identity).
        T_err = T_target @ np.linalg.inv(T_q)
        r_err = T_err[:3, :3]
        omega = 0.5 * np.array(
            [
                r_err[2, 1] - r_err[1, 2],
                r_err[0, 2] - r_err[2, 0],
                r_err[1, 0] - r_err[0, 1],
            ]
        )
        twist = np.concatenate([T_err[:3, 3], omega])
        J = kinbody_jacobian(kb, q)
        try:
            dq = np.linalg.solve(J, twist)
        except np.linalg.LinAlgError:
            damping = max(1e-9, 1e-6 * frob)
            n = J.shape[1]
            dq = np.linalg.solve(J.T @ J + damping * np.eye(n), J.T @ twist)
        dq = np.clip(dq, -step_clip, step_clip)
        q = q + dq
    # Final convergence check.
    T_q = poe_forward_kinematics(kb, q)
    frob = float(np.linalg.norm(T_q - T_target))
    if frob < fk_atol:
        return q, frob, max_iters
    return None


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    max_drift_m: float = _DEFAULT_MAX_DRIFT_M,
    swivel_samples: int | NDArray[np.float64] = 16,
    polish_max_iters: int = 30,
    polish_fk_atol: float = 1e-12,
    max_solutions: int | None = None,
) -> tuple[list[Solution], bool]:
    """Approximate-SRS 7R IK with LM polish.

    :param kb: POE-normalized 7R :class:`KinBody`.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerance policy. ``policy.subproblem_dedup`` controls
        the cluster-merge gate after polish.
    :param max_drift_m: refusal gate; arms whose shoulder/wrist drift
        exceeds this value (default 4 cm) raise :class:`ValueError`.
        Tuned to keep the snap-and-polish trajectory inside Newton's
        basin (~3-5 cm task space empirically).
    :param swivel_samples: int N for uniform sweep over [-π, π], or an
        explicit array of swivel angles. Forwarded to the inner SRS
        solver.
    :param polish_max_iters: per-candidate Newton iteration cap.
    :param polish_fk_atol: target FK closure for accepting a polished
        candidate. Default 1e-12 reaches the bulletproof fixture
        contract (FK ≤ 1e-10).
    :param max_solutions: optional cap on the number of polished
        IKs returned.

    :returns: ``(solutions, is_ls)``. ``is_ls=True`` iff zero
        candidates polished to within ``polish_fk_atol``.

    :raises ValueError: if ``kb`` is not 7-DOF or its drift exceeds
        ``max_drift_m`` (i.e. arm is not approximately SRS).
    """
    if len(kb.joints) != 7:
        raise ValueError(f"seven_r.srs_polished requires a 7-DOF chain; got {len(kb.joints)}")

    cls = is_approximately_srs_7r(kb, max_drift_m=max_drift_m, policy=policy)
    if cls is None:
        raise ValueError(
            f"seven_r.srs_polished requires approximately-SRS topology with "
            f"max axis drift <= {max_drift_m} m. Use "
            f"ssik.kinematics.predicates.is_approximately_srs_7r to check."
        )

    # Step 1: build a relaxed policy that lets the inner SRS solver
    # accept the approximate pivots.
    relaxed_policy = replace(policy, axis_intersect=max(max_drift_m, policy.axis_intersect))

    # Step 2: run inner SRS with permissive FK tolerance to capture all
    # algebraic candidates. They will have FK residual ~max_drift_m
    # because the solver assumes axes meet exactly; LM polish corrects.
    raw, _is_ls = srs.solve(
        kb,
        T_target,
        policy=relaxed_policy,
        swivel_samples=swivel_samples,
        fk_atol=10.0,
        # Don't cap raw candidates here -- some won't polish, and
        # we want to maximise survivors.
        max_solutions=None,
    )

    if not raw:
        # SRS produced nothing even at huge fk_atol; truly unreachable.
        return [], True

    # Step 3: LM polish each candidate against the original URDF FK,
    # using a Frobenius-FK convergence gate to avoid the near-identity
    # precision loss in se3_log_residual (#199).
    polished: list[Solution] = []
    for c in raw:
        result = _polish_to_frobenius(
            c.q,
            kb,
            T_target,
            fk_atol=polish_fk_atol,
            max_iters=polish_max_iters,
        )
        if result is None:
            continue
        q_polished, fk_frob, iters = result
        polished.append(
            replace(
                c,
                q=q_polished,
                fk_residual=fk_frob,
                refinement_used="lm",
                refinement_iters=iters,
                solver_name=_SOLVER_NAME,
            )
        )

    if not polished:
        return [], True

    # Step 4: cluster-merge. Different SRS branches may polish into
    # the same true IK; dedup keeps one representative per cluster.
    deduped = dedup_by_wrap_close(polished, policy.subproblem_dedup)

    if max_solutions is not None and len(deduped) > max_solutions:
        deduped = deduped[:max_solutions]

    _LOG.info(
        "seven_r.srs_polished: drift_shoulder=%.4f m  drift_wrist=%.4f m  "
        "raw=%d  polished=%d  deduped=%d",
        cls.shoulder_drift_m,
        cls.wrist_drift_m,
        len(raw),
        len(polished),
        len(deduped),
    )

    return deduped, False
