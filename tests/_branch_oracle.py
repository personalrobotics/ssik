"""A branch oracle that shares no algebra with the solvers (#573).

M7 needs to answer "did the solver return *every* branch?", and the honest
difficulty is finding an answer to check against. Most of our existing gates
compare one ssik path to another: native against Python, one solver against a
second on a shared fixture. Those catch implementation drift, but a *shared*
omission passes every one of them, and #571 is exactly that. The general-6R
Raghavan-Roth path drops generalized eigenvalues at infinity, so a configuration
whose linearity joint sits at ``pi`` is unrepresentable in the tan-half-angle
coordinate. Re-running the same algebra in higher precision would miss it
identically: the omission is in the representation, not the arithmetic.

So the oracle here works in joint space and touches no chart at all. It runs
damped Levenberg-Marquardt on ``FK(q) = T`` from many random starts and keeps
what converges. ``q = pi`` is unremarkable to it. On #571's fixture it recovers
eight branches, including the one the solver drops, at machine precision.

The cost of independence is that this is a *statistical* oracle: nothing
guarantees a given number of restarts lands in every basin. It is trustworthy
only with a stopping rule, so :func:`enumerate_branches` escalates its restart
budget until the branch count stops growing, and reports the budget it needed.
On the same fixture 400 restarts found seven branches (a different seven than
the solver's) and 2000 found all eight, which is the entire reason the rule
exists rather than a fixed count.

Use it for curated fixtures and regression cases, not in a hot loop: a
stabilized run is thousands of LM solves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.numerical import lm_multi_restart

__all__ = ["OracleResult", "enumerate_branches", "wrapped_linf"]

_TWO_PI = 2.0 * np.pi

# Escalating restart budgets. Each round re-runs from scratch with a larger
# count and unions the result, so the branch set only grows; the run stops once
# it has stopped growing (see `stable_rounds`).
_BUDGETS = (500, 1000, 2000, 4000, 8000)

# LM must converge much tighter than the solvers' own fk_atol: the oracle is the
# reference, so its residuals should be at machine precision rather than at the
# acceptance threshold.
_FK_ATOL = 1e-12
_MAX_ITERS = 200

# Two configurations count as the same branch within this wrapped L-infinity
# distance. Far below the separation between genuine branches (#571's nearest
# pair is 0.59 rad) and far above LM's converged spread within one basin.
_DEDUP_TOL = 1e-4


def wrapped_linf(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Max per-joint difference, each wrapped to ``[-pi, pi]``."""
    d = (np.asarray(a) - np.asarray(b) + np.pi) % _TWO_PI - np.pi
    return float(np.max(np.abs(d)))


@dataclass(frozen=True)
class OracleResult:
    """Branches found, and how hard the oracle had to work to be sure."""

    branches: tuple[NDArray[np.float64], ...]
    worst_fk: float
    budget_used: int
    stabilized: bool
    """False when the count was still growing at the largest budget, meaning the
    set is a lower bound rather than an answer. Assertions must not treat an
    unstabilized result as completeness evidence."""

    def __len__(self) -> int:
        return len(self.branches)

    def contains(self, q: NDArray[np.float64], tol: float = 1e-6) -> bool:
        return any(wrapped_linf(q, b) <= tol for b in self.branches)

    def nearest(self, q: NDArray[np.float64]) -> float:
        if not self.branches:
            return float("inf")
        return min(wrapped_linf(q, b) for b in self.branches)


def _merge(found: list[NDArray[np.float64]], q: NDArray[np.float64], tol: float) -> bool:
    """Add ``q`` unless an equivalent configuration is already present."""
    for b in found:
        if wrapped_linf(q, b) <= tol:
            return False
    found.append(q)
    return True


def enumerate_branches(
    kb: KinBody,
    t_target: NDArray[np.float64],
    *,
    dedup_tol: float = _DEDUP_TOL,
    fk_atol: float = _FK_ATOL,
    budgets: tuple[int, ...] = _BUDGETS,
    stable_rounds: int = 2,
) -> OracleResult:
    """Every branch of ``FK(q) = t_target`` this can find, without using a chart.

    Escalates through ``budgets`` until the branch count repeats for
    ``stable_rounds`` consecutive rounds, then reports that set. A result whose
    count was still growing at the last budget is returned with
    ``stabilized=False``.

    :param dedup_tol: wrapped L-infinity distance below which two configurations
        are the same branch.
    :param fk_atol: LM convergence threshold. Tight on purpose: this is the
        reference the solvers are measured against.
    """
    policy = replace(DEFAULT_TOLERANCE_POLICY, subproblem_numerical=fk_atol)
    found: list[NDArray[np.float64]] = []
    counts: list[int] = []
    budget_used = 0

    for budget in budgets:
        budget_used = budget
        sols, _ = lm_multi_restart.solve(
            kb, t_target, policy=policy, n_restarts=budget, refinement_max_iters=_MAX_ITERS
        )
        for s in sols:
            q = np.asarray(s.q, dtype=np.float64)
            if float(np.linalg.norm(poe_forward_kinematics(kb, q) - t_target)) <= fk_atol:
                _merge(found, q, dedup_tol)
        counts.append(len(found))
        if len(counts) > stable_rounds and len(set(counts[-(stable_rounds + 1) :])) == 1:
            break

    stabilized = len(counts) > stable_rounds and len(set(counts[-(stable_rounds + 1) :])) == 1
    worst = max(
        (float(np.linalg.norm(poe_forward_kinematics(kb, q) - t_target)) for q in found),
        default=0.0,
    )
    # Deterministic order so failure messages are reproducible.
    found.sort(key=lambda q: [round(float(v), 9) for v in q])
    return OracleResult(tuple(found), worst, budget_used, stabilized)
