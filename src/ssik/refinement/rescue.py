"""T-perturbation rescue for measure-zero RR rank-deficiency ridges (#319).

Five of the eight outstanding coverage gaps share one root cause: at
specific q-space ridges (e.g. CRX-10iA/L's q3 ~ -pi/2 + roll-axis
triple), the Raghavan-Roth pencil ``m_quad`` is structurally
rank-deficient. The algebraic solver still extracts roots but they
fail FK closure; the genuine analytical solutions exist arbitrarily
close in q-space but are not algebraically reachable from the direct
RR path at the exact ridge point.

This module provides a small, opt-in rescue layer: perturb the
target pose by a small SE(3) increment, re-solve at the perturbed
pose (which sits off-ridge in the well-conditioned regime), then
Newton-refine each candidate back to the original ``T_target`` via
``lm_refine``. Empirically recovers 4-17 unique sols on the
falsifying examples of #298 (CRX), #304 (Rizon 4), and #280 (Kassow)
with FK closure at the 1e-10 to 1e-12 range.

Design intent:

- **Opt-in.** Callers explicitly invoke ``rescue_via_T_perturbation``
  when the direct ``solve()`` returns an empty list. The default
  solver path stays purely analytical -- ssik's first-class promise
  is "analytical IK", not "numerical IK". The rescue is a deliberate
  user-controlled fallback for the measure-zero ridge cases.
- **Deterministic.** Uses a fixed RNG seed by default so test
  fingerprints stay stable across runs.
- **Cheap when fired.** ~8 perturbations x normal solve cost; well
  under 1 ms additional latency on tier-0 / SRS arms, ~50-100 ms on
  HP-class jointlock 7R arms.
- **FK-closure gated.** Only returns candidates that LM-refine back
  to the original ``T_target`` within ``fk_atol``. No tolerance
  loosening, no papering over.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.refinement import lm_refine


def rescue_via_T_perturbation(
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    solve_fn: Callable[..., list[Solution]],
    T_target: NDArray[np.float64],
    *,
    n_perturbations: int = 8,
    perturbation_scale_m: float = 5e-3,
    perturbation_scale_rad: float = 5e-3,
    fk_atol: float = 1e-8,
    refinement_max_iters: int = 20,
    dedup_atol: float = 1e-4,
    seed: int = 20260608,
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> list[Solution]:
    """Recover IK solutions at q-space ridges via T-perturbation + LM polish.

    Perturbs ``T_target`` by random SE(3) increments, re-solves at each
    perturbed pose (which sits off the rank-deficient ridge), then
    Newton-refines each candidate back to the original ``T_target``.
    Returns the unique FK-closing solutions.

    Intended call site::

        sols = module.solve(T_target)
        if not sols:
            sols = rescue_via_T_perturbation(
                module.fk, module.solve, T_target,
            )

    :param fk_fn: per-arm forward kinematics callable (typically
        ``<arm>_ik.fk``).
    :param solve_fn: per-arm IK solve callable (typically
        ``<arm>_ik.solve``). Called with ``T_pert`` and
        ``respect_limits=False`` to maximize the candidate set.
    :param T_target: 4x4 SE(3) target pose. The pose the rescued
        solutions must close at.
    :param n_perturbations: how many random T-perturbations to try.
        Default 8 -- empirically ~50-100% per-trial success rate on
        Group A reproducers, so 8 trials hits >99% combined recovery.
    :param perturbation_scale_m: translation magnitude (each axis,
        each trial). Default 5 mm.
    :param perturbation_scale_rad: rotation magnitude (each axis,
        each trial). Default 5 mrad.
    :param fk_atol: SE(3) log-residual threshold for accepted
        solutions (consumed by :func:`lm_refine` internally). Default
        1e-8 -- empirically achieved by all rescued candidates on the
        Group A reproducers, and ~2-3 orders of magnitude below
        typical robot repeatability, so rescue solutions are
        operationally indistinguishable from analytical ones.
    :param refinement_max_iters: cap on Newton iterations per
        candidate. Default 20 -- empirically converges in 3-8 iters
        on Group A reproducers.
    :param dedup_atol: wrap-to-pi joint-angle tolerance for
        collapsing equivalent solutions. Default 1e-4 rad.
    :param seed: RNG seed for the perturbation directions.
        Deterministic by default so test fingerprints are stable.
    :param jacobian_fn: optional analytical spatial Jacobian for the
        LM-refine step. When ``None``, ``lm_refine`` falls back to
        central-difference Jacobian (~50x slower).

    :returns: list of :class:`Solution` whose FK closes at the
        original ``T_target`` within ``fk_atol``. Each carries
        ``refinement_used="lm"`` to flag that the rescue path fired.
        Empty list iff none of the ``n_perturbations`` trials
        produced a solution that refined back to ``T_target``.
    """
    rng = np.random.default_rng(seed)
    refined: list[Solution] = []
    refined_qs: list[NDArray[np.float64]] = []

    for _ in range(n_perturbations):
        dx = rng.standard_normal(3) * perturbation_scale_m
        # Small-angle so3 perturbation: vector w in R^3 with magnitude
        # |w| ~ perturbation_scale_rad becomes a rotation of angle |w|
        # about axis w / |w|. Using Rodrigues' formula on the small
        # vector keeps the perturbation well-conditioned at small
        # scales without the cos/sin breakdown of an explicit
        # axis-angle build.
        w = rng.standard_normal(3) * perturbation_scale_rad
        angle = float(np.linalg.norm(w))
        if angle > 0:
            axis = w / angle
            K = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            R_delta = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        else:
            R_delta = np.eye(3)
        dT = np.eye(4)
        dT[:3, :3] = R_delta
        dT[:3, 3] = dx
        T_pert = T_target @ dT

        try:
            pert_sols = solve_fn(T_pert, respect_limits=False)
        except TypeError:
            # Per-arm solve functions that don't accept respect_limits
            # (extremely rare in shipping artifacts but worth handling).
            pert_sols = solve_fn(T_pert)

        for sol in pert_sols:
            q_seed = np.asarray(sol.q, dtype=np.float64)
            result = lm_refine(
                q_seed,
                fk_fn,
                T_target,
                fk_atol=fk_atol,
                max_iters=refinement_max_iters,
                jacobian_fn=jacobian_fn,
            )
            if result is None:
                continue
            q_ref, fk_resid, _iters = result
            if fk_resid > fk_atol:
                continue

            # Wrap-to-pi dedup against accepted solutions so far.
            is_dup = False
            for q_existing in refined_qs:
                diff = (q_ref - q_existing + np.pi) % (2.0 * np.pi) - np.pi
                if float(np.linalg.norm(diff)) < dedup_atol:
                    is_dup = True
                    break
            if is_dup:
                continue

            refined.append(
                Solution(
                    q=q_ref,
                    fk_residual=float(fk_resid),
                    refinement_used="lm",
                )
            )
            refined_qs.append(q_ref)

    return refined
