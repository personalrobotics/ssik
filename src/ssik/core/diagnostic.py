"""Diagnostic record for ``solve(T, explain=True)`` (#265).

Today's failure mode for ``solve(T)`` is opaque: callers get an empty
``list[Solution]`` and have to guess whether the pose was unreachable,
near-singular, filtered by joint limits, or hit a solver bug. The
``Diagnostic`` returned alongside the solutions on explain mode lets the
caller attribute failures concretely.

Counts are aggregate-only by default (cheap); per-candidate dispositions
are out of scope for the v1.1.0 introduction. Per-IK ``Solution.fk_residual``
already lets a caller inspect individual survivors.

Usage::

    arm = ssik.prebuilt.iiwa14_ik
    sols, diag = arm.solve(T_target, explain=True)
    if not sols:
        print(diag.summary())
        # solver: seven_r.srs (tier 0)
        # 0 raw candidates -> 0 returned (pose appears unreachable
        # within tolerance policy.subproblem_numerical = 1e-05).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Diagnostic:
    """Per-call diagnostic record from ``solve(T, explain=True)``.

    All counts are post-filter aggregates -- the solver's own
    enumeration cost (number of raw algebraic candidates before any
    filtering) is in ``raw_candidates``; each subsequent filter is a
    delta. ``final_count == len(returned_solutions)``.

    The fields are deliberately data-only (no methods that re-run the
    solve, no live references to the kinbody). A diagnostic is a
    snapshot, safe to log / serialise.
    """

    solver_name: str
    """Dispatched solver, e.g. ``"ikgeo.three_parallel"``."""

    solver_tier: int
    """0 = closed-form, 1 = univariate search, 2 = bivariate search."""

    dispatch_reason: str
    """Human-readable explanation of why the dispatcher picked this solver.
    Often the most useful field for 'why did ssik refuse my arm?' triage."""

    raw_candidates: int
    """Raw analytical candidate count returned by the inner solver, before
    any post-processing. ``0`` means the solver itself returned nothing --
    pose is outside the analytical reach or the solver's preconditions
    aren't met. ``>0`` with ``final_count == 0`` means every candidate was
    filtered (limits, etc.)."""

    dropped_by_limits: int
    """Candidates that fell outside URDF joint limits (after the ``q ± 2π``
    rescue pass). ``0`` when ``respect_limits=False`` was passed."""

    dropped_by_max_solutions: int
    """Surviving candidates truncated by the ``max_solutions`` cap."""

    final_count: int
    """Number of solutions in the returned list."""

    max_fk_residual: float
    """Worst ``fk_residual`` among returned solutions. ``nan`` when
    ``final_count == 0``."""

    refinement_engaged: int
    """How many candidates ran through Levenberg-Marquardt polish (when
    ``allow_refinement=True`` was passed). ``0`` when refinement was off."""

    fk_atol: float
    """The FK-closure threshold the solver used. Useful when the user
    customised the tolerance policy and wants the live value back."""

    warnings: tuple[str, ...] = field(default_factory=tuple)
    """Optional conditioning / robustness flags raised during the solve.
    Empty tuple in the common-path. Today: forward-compatible reservation;
    populated by the HP / RR solvers once they wire up conditioning checks
    (#178)."""

    def summary(self) -> str:
        """One-paragraph human-readable summary for logging / triage.

        Distinguishes the four common cases:
        - reachable (returned >=1 IK)
        - unreachable (0 raw candidates)
        - all-filtered (raw candidates > 0, final 0 -- joint limits etc.)
        - capped (max_solutions truncation)
        """
        lines = [
            f"solver: {self.solver_name} (tier {self.solver_tier})",
            f"dispatch: {self.dispatch_reason}",
        ]
        if self.final_count > 0:
            lines.append(
                f"  -> {self.raw_candidates} candidates "
                f"-> {self.final_count} returned (max FK {self.max_fk_residual:.1e}, "
                f"threshold {self.fk_atol:.0e})"
            )
            if self.dropped_by_limits:
                lines.append(f"  filtered by joint limits: {self.dropped_by_limits}")
            if self.dropped_by_max_solutions:
                lines.append(f"  capped by max_solutions: {self.dropped_by_max_solutions}")
        elif self.raw_candidates == 0:
            lines.append(
                "  -> 0 raw candidates: pose appears unreachable "
                "(or outside this solver's analytical envelope)"
            )
        else:
            lines.append(f"  -> {self.raw_candidates} raw candidates, all filtered:")
            if self.dropped_by_limits:
                lines.append(
                    f"     dropped by joint limits: {self.dropped_by_limits} "
                    f"(pass respect_limits=False for the raw geometric set)"
                )
        if self.refinement_engaged:
            lines.append(f"  LM polish engaged on {self.refinement_engaged} candidates")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


__all__ = ["Diagnostic"]
