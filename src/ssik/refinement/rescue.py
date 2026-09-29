"""T-perturbation rescue for measure-zero RR rank-deficiency ridges (#319).

Five of the eight outstanding coverage gaps share one root cause: at
specific q-space ridges (e.g. CRX-10iA/L's q3 ~ -pi/2 + roll-axis
triple), the Raghavan-Roth pencil ``m_quad`` is structurally
rank-deficient. The algebraic solver still extracts roots but they
fail FK closure; the genuine analytical solutions exist arbitrarily
close in q-space but are not algebraically reachable from the direct
RR path at the exact ridge point.

This module provides a small rescue layer: perturb the target pose by a
small SE(3) increment, re-solve at the perturbed pose (which sits off-ridge
in the well-conditioned regime), then Newton-polish each candidate back to
the original ``T_target``. Empirically recovers 4-17 unique sols on the
falsifying examples of #298 (CRX), #304 (Rizon 4), and #280 (Kassow) with
FK closure at the 1e-10 to 1e-12 range.

Design intent:

- **Empty-gated.** ``solve()`` calls it only when the analytical path
  returns no (in-limits) solution; ``allow_rescue=False`` is the
  guaranteed-analytical escape.
- **One definition, two backends (#622).** The native backend
  (``cpp/include/ssik_cpp/rescue.hpp``) implements this exact algorithm:
  the same perturbation sequence and the same polish, so both backends
  recover the same set (up to float round-off in the polish) and rescue
  output is reproducible on every platform. See "The definition" below;
  a change here must be mirrored there.
- **Cheap when fired.** 16 perturbations x normal solve cost; well
  under 1 ms additional latency on tier-0 / SRS arms, ~50-100 ms on
  HP-class jointlock 7R arms.
- **FK-closure gated.** Only returns candidates that polish back to the
  original ``T_target`` within ``fk_atol``. No tolerance loosening, no
  papering over.

The definition
--------------

Perturbation ``i`` (``i = 0 .. n_perturbations - 1``) uses the direction
``u_i`` in R^6, the ``(i + 1)``-th point of the R_6 Kronecker (generalised
golden ratio) low-discrepancy sequence mapped to a zero-mean box of unit
per-axis variance::

    u_i[k] = sqrt(3) * (2 * frac((i + 1) * ALPHA[k]) - 1),  frac(x) = x - floor(x)

with ``ALPHA[k] = frac(phi^-(k + 1))`` for the real root ``phi`` of
``x^7 = x + 1`` (``_KRONECKER_ALPHA``, tabulated as double literals). Every
step is a correctly rounded IEEE operation, so the directions are
bit-identical in both languages and on every platform; no pseudo-random
generator is involved (#622: the two backends used PCG64 and mt19937_64,
and libstdc++ / libc++ even disagree on ``std::normal_distribution``). With
``m = scale_multipliers[i % len]``, ``dx = u_i[:3] * perturbation_scale_m * m``
and ``w = u_i[3:] * perturbation_scale_rad * m`` give
``T_pert = T_target @ [Rot(w / |w|, |w|), dx]``.

Each perturbed pose is re-solved with ``respect_limits=False``, and every
candidate is polished back to ``T_target`` by the fixed-damping batch Newton
of :func:`ssik.refinement.lm_refine_batch`: at most ``refinement_max_iters``
steps ``dq = clip((J^T J + 1e-9 I)^-1 J^T log(T_target FK(q)^-1), +-0.5)``,
stopping as soon as the Frobenius residual drops below 1e-12, aborting a
trajectory whose residual exceeds 5x its best after 4 iterations, and
accepting the end point when its residual is at most ``fk_atol``. A candidate
that polish does not accept is polished again from its perturbed solution, the
same way except with the damping scaled by the residual,
``1e-9 * clip(||log(T_target FK(q)^-1)||_2, 1e-5, 1)``
(``residual_scaled_damping``), and is accepted by the same rule. (#622 chose
this polish over the adaptive single-candidate ``lm_refine`` on recovery: the
adaptive one's stall guard and undamped 6R step drop candidates at singular
poses, e.g. ur16e returned [] there. #646 added the second polish: the fixed
damping shortens the step along a Jacobian direction with ``sigma < ~3e-5``,
so near a singular solution it stalls above ``fk_atol``; a piper pose at a
wrist singularity, sigma_min 3e-6, came back [] although its own in-limits
configuration was an exact solution. The scaled damping alone instead loses
exactly singular 7R solutions that the fixed damping reaches, so the second
polish only adds to the first.) Accepted
solutions are deduplicated in perturbation order, then candidate order, by
the L2 wrap-to-pi distance ``dedup_atol``; the first one seen is kept.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.refinement import lm_refine_batch, numerical_jacobian
from ssik.subproblems._rotation import rotation_matrix

# Tight polish target tried before the loose acceptability gate. A rescue whose
# perturbation landed off-ridge (the common case) is well-conditioned and
# converges to machine precision in a step or two; aiming for it there means we
# return the *exact* solution rather than one that merely cleared ``fk_atol``
# (the #384 gen3 case: the polish stopped at ~6e-9 the instant it dipped under
# the 1e-8 gate, leaving the machine precision it would reach one iteration
# later unused). Genuinely rank-deficient ridge poses (#319) stall above this
# and are accepted at ``fk_atol`` instead, so ridge recovery is unchanged.
_TIGHT_POLISH_FK_ATOL = 1e-12

# The polish's divergence guard (see the module docstring). Looser than
# lm_refine_batch's default 2.0 / 2 (tuned for srs_polished, #203): that
# tighter guard aborts rescuable perturbed candidates whose trajectory dips
# before converging (piper lost 3 of 6 on a ridge pose, #405).
_DIVERGENCE_FACTOR = 5.0
_DIVERGENCE_MIN_ITERS = 4

# frac(phi^-(k+1)), k = 0..5, for the real root phi of x^7 = x + 1: the R_6
# Kronecker sequence's increments. Mirrored in cpp/include/ssik_cpp/rescue.hpp.
_KRONECKER_ALPHA = (
    0.8986537126286993,
    0.8075784952213448,
    0.7257334129697598,
    0.6521830259439717,
    0.5860866975779695,
    0.5266889867007359,
)
_SQRT3 = 1.7320508075688772


def perturbation_direction(i: int) -> tuple[float, ...]:
    """``u_i`` of the module docstring: the rescue's ``i``-th unscaled SE(3)
    perturbation, translation components first. Deterministic and
    platform-independent (no RNG)."""
    out = []
    for a in _KRONECKER_ALPHA:
        x = (i + 1) * a
        out.append(_SQRT3 * (2.0 * (x - math.floor(x)) - 1.0))
    return tuple(out)


def _perturbation(i: int, scale_m: float, scale_rad: float, mult: float) -> NDArray[np.float64]:
    """The 4x4 increment ``dT`` of perturbation ``i``; ``T_pert = T_target @ dT``.
    Operation order matches the native port so the two agree to the last bit
    (up to libm's sin/cos)."""
    u = perturbation_direction(i)
    dT = np.eye(4)
    w0, w1, w2 = (u[3] * scale_rad * mult, u[4] * scale_rad * mult, u[5] * scale_rad * mult)
    angle = math.sqrt(w0 * w0 + w1 * w1 + w2 * w2)
    if angle > 0.0:
        dT[:3, :3] = rotation_matrix(np.array([w0 / angle, w1 / angle, w2 / angle]), angle)
    dT[0, 3] = u[0] * scale_m * mult
    dT[1, 3] = u[1] * scale_m * mult
    dT[2, 3] = u[2] * scale_m * mult
    return dT


def rescue_via_T_perturbation(
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    solve_fn: Callable[..., list[Solution]],
    T_target: NDArray[np.float64],
    *,
    n_perturbations: int = 16,
    perturbation_scale_m: float = 5e-3,
    perturbation_scale_rad: float = 5e-3,
    scale_multipliers: tuple[float, ...] = (1.0, 2.0, 4.0, 10.0),
    fk_atol: float = 1e-8,
    refinement_max_iters: int = 20,
    dedup_atol: float = 1e-3,
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> list[Solution]:
    """Recover IK solutions at q-space ridges via T-perturbation + Newton polish.

    Perturbs ``T_target`` by the deterministic SE(3) increments of the module
    docstring, re-solves at each perturbed pose (which sits off the
    rank-deficient ridge), then Newton-polishes each candidate back to the
    original ``T_target``. Returns the unique FK-closing solutions.

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
    :param n_perturbations: how many T-perturbations to try. Default 16 --
        combined with the escalating ``scale_multipliers`` this gives robust
        cross-platform margin.
    :param perturbation_scale_m: base translation magnitude (per axis, unit
        variance over the sequence), scaled by ``scale_multipliers``.
        Default 5 mm.
    :param perturbation_scale_rad: base rotation magnitude (per axis, as
        above), scaled by ``scale_multipliers``. Default 5 mrad.
    :param scale_multipliers: cycled per-perturbation multipliers applied
        to the base scales. Default ``(1, 2, 4, 10)`` -> 5/10/20/50 mm +
        mrad. The large multiples land firmly off the rank-deficient
        ridge, so recovery does not depend on BLAS-backend-sensitive
        near-ridge numerics.
    :param fk_atol: Frobenius FK residual an accepted solution must meet.
        Default 1e-8 -- ~2-3 orders of magnitude below typical robot
        repeatability, so rescue solutions are operationally
        indistinguishable from analytical ones.
    :param refinement_max_iters: cap on Newton iterations per
        candidate. Default 20 -- empirically converges in 3-8 iters
        on Group A reproducers.
    :param dedup_atol: wrap-to-pi joint-angle tolerance for
        collapsing equivalent solutions. Default 1e-3 rad, matching the
        solvers' ``subproblem_dedup`` (#534): the shared finalize does not
        re-dedup, so a tighter rescue dedup leaked near-duplicate pairs in
        ``(1e-4, 1e-3)`` -- pairs the solver already treats as one -- into the
        returned set. Deduping at the solver's own tolerance cannot drop a
        genuinely-distinct solution.
    :param jacobian_fn: optional analytical spatial Jacobian for the
        polish. When ``None``, a central-difference Jacobian is used
        (~50x slower, same algorithm).

    :returns: list of :class:`Solution` whose FK closes at the
        original ``T_target`` within ``fk_atol``. Each carries
        ``refinement_used="lm"`` to flag that the rescue path fired.
        Empty list iff none of the ``n_perturbations`` trials
        produced a solution that refined back to ``T_target``.
    """
    jac: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    if jacobian_fn is not None:
        jac = jacobian_fn
    else:

        def jac(q: NDArray[np.float64]) -> NDArray[np.float64]:
            return numerical_jacobian(q, fk_fn)

    refined: list[Solution] = []
    refined_qs: list[NDArray[np.float64]] = []

    for i in range(n_perturbations):
        # Escalating scale schedule: cycle through ``scale_multipliers`` so
        # successive perturbations span a range of magnitudes (default
        # 5/10/20/50 mm + mrad). Small near-ridge perturbations are
        # numerically marginal -- whether they clear the rank-deficient
        # ridge is BLAS-backend-sensitive (the #319 CI flake: Accelerate
        # recovered, OpenBLAS did not). The large multiples land firmly
        # off-ridge in the well-conditioned regime and recover reliably on
        # any backend, so the schedule gives robust cross-platform margin.
        mult = scale_multipliers[i % len(scale_multipliers)] if scale_multipliers else 1.0
        dT = _perturbation(i, perturbation_scale_m, perturbation_scale_rad, mult)
        T_pert = T_target @ dT

        try:
            pert_sols = solve_fn(T_pert, respect_limits=False)
        except TypeError:
            # Per-arm solve functions that don't accept respect_limits
            # (extremely rare in shipping artifacts but worth handling).
            pert_sols = solve_fn(T_pert)

        if not pert_sols:
            continue

        # Polish every perturbed candidate back to the original T_target (the
        # module docstring's polish). Most perturbed candidates on a redundant
        # arm are swivel solutions to T_pert that do NOT converge back to a
        # ridge T_target; the divergence guard aborts those in a few iterations.
        q_seeds = np.array([sol.q for sol in pert_sols], dtype=np.float64)
        q_polished, fk_resids, _iters = lm_refine_batch(
            q_seeds,
            fk_fn,
            jac,
            T_target,
            fk_atol=min(_TIGHT_POLISH_FK_ATOL, fk_atol),
            max_iters=refinement_max_iters,
            divergence_factor=_DIVERGENCE_FACTOR,
            divergence_min_iters=_DIVERGENCE_MIN_ITERS,
        )
        # Second polish, from the seed, for the candidates the first did not
        # accept (the module docstring; #646).
        retry = ~(fk_resids <= fk_atol)
        if retry.any():
            q_polished[retry], fk_resids[retry], _ = lm_refine_batch(
                q_seeds[retry],
                fk_fn,
                jac,
                T_target,
                fk_atol=min(_TIGHT_POLISH_FK_ATOL, fk_atol),
                max_iters=refinement_max_iters,
                divergence_factor=_DIVERGENCE_FACTOR,
                divergence_min_iters=_DIVERGENCE_MIN_ITERS,
                residual_scaled_damping=True,
            )

        for q_ref, fk_resid in zip(q_polished, fk_resids, strict=True):
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
