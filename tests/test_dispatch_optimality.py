"""Dispatch coverage-optimality gate: when an arm matches more than one
tier-0 7R solver predicate, the dispatcher must pick the one with the best
coverage (#442).

Several 7R solvers are selected by predicates that can match the SAME arm --
notably the two *approximate* predicates (``is_approximately_srs_7r`` and
``is_approximately_spherical_shoulder_7r``), and the strict SRS predicate is a
subset of the approximate one. Which solver wins is decided purely by dispatch
order, and nothing else validates that the chosen solver actually maximizes
coverage.

j2s7s300 (#440) exposed this: it matches BOTH approximate predicates. The
dispatcher checked approximate-spherical-shoulder first and routed it to
``spherical_shoulder_polished`` at ~38% coverage, when ``srs_polished`` covers
100%. The arm dispatched "successfully" to a solver that silently dropped 62%
of reachable poses.

This gate, for every shipped 7R arm, enumerates the tier-0 solvers whose
predicate matches, measures each one's coverage, and asserts the *dispatched*
solver is within tolerance of the best. A dispatch-order regression that
downgrades an arm's pick fails here even though the arm still "dispatches".

See #442.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from ssik.core.dispatcher import dispatch
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY as POL
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import is_approximately_srs_7r, is_srs_7r
from ssik.prebuilt._manifest import load_manifest
from ssik.solvers.seven_r import (
    spherical_shoulder,
    spherical_shoulder_polished,
    srs,
    srs_polished,
)
from ssik.solvers.seven_r.spherical_shoulder import is_spherical_shoulder_7r
from ssik.solvers.seven_r.spherical_shoulder_polished import (
    is_approximately_spherical_shoulder_7r,
)

if TYPE_CHECKING:
    from ssik._kinbody import KinBody

_MANIFEST = load_manifest()
_ARMS_7R = [a for a in _MANIFEST.values() if a.dof == 7]

_N_POSES = 40
# The dispatched solver's coverage must be within this of the best matching
# candidate. Generous: the point is to catch a systemic mis-route (j2s7's
# 100% vs 38%), not to police last-percent differences between two equally-good
# solvers on borderline poses.
_COVERAGE_SLACK = 0.15
_FK_TOL = 1e-6

# Every tier-0 7R solver, guarded by the predicate the dispatcher uses to pick
# it. Each entry: (solver_name, predicate(kb) -> bool, solve(kb, T, policy)).
# The predicate guard is what lets us call each solver only on arms it accepts
# (the solvers raise on a non-matching topology).
_Candidate = tuple[str, Callable[["KinBody"], bool], Callable[..., Any]]
_CANDIDATES: list[_Candidate] = [
    ("seven_r.srs", lambda kb: is_srs_7r(kb, POL) is not None, srs.solve),
    (
        "seven_r.srs_polished",
        lambda kb: is_approximately_srs_7r(kb, max_drift_m=0.04, policy=POL) is not None,
        srs_polished.solve,
    ),
    (
        "seven_r.spherical_shoulder",
        lambda kb: is_spherical_shoulder_7r(kb, POL),
        spherical_shoulder.solve,
    ),
    (
        "seven_r.spherical_shoulder_polished",
        lambda kb: is_approximately_spherical_shoulder_7r(kb, policy=POL),
        spherical_shoulder_polished.solve,
    ),
]


def _coverage(solve_fn: Callable[..., Any], kb: KinBody) -> float:
    """Fraction of ``_N_POSES`` in-limits poses the raw solver FK-closes."""
    rng = np.random.default_rng(20260728)
    limits = [(j.limits if j.limits is not None else (-np.pi, np.pi)) for j in kb.joints]
    covered = 0
    for _ in range(_N_POSES):
        q = np.array([rng.uniform(lo, hi) for lo, hi in limits])
        t_target = poe_forward_kinematics(kb, q)
        result = solve_fn(kb, t_target, POL)
        sols = result[0] if isinstance(result, tuple) else result
        if sols and any(
            float(np.max(np.abs(poe_forward_kinematics(kb, np.asarray(s.q)) - t_target))) < _FK_TOL
            for s in sols
        ):
            covered += 1
    return covered / _N_POSES


@pytest.mark.parametrize("arm", _ARMS_7R, ids=[a.name for a in _ARMS_7R])
def test_dispatch_picks_best_coverage_solver(arm: object) -> None:
    """For every 7R arm that matches >1 tier-0 solver predicate, the dispatched
    solver must cover within ``_COVERAGE_SLACK`` of the best matching candidate.

    Guards dispatch precedence between overlapping predicates: a reordering that
    downgrades an arm's pick (j2s7 -> spherical_shoulder_polished at 38% instead
    of srs_polished at 100%) fails here.
    """
    name = arm.name  # type: ignore[attr-defined]
    mod = importlib.import_module(arm.hier_module or f"ssik.prebuilt.{name}")  # type: ignore[attr-defined]
    kb = mod._KB

    matching = [(sname, solve) for (sname, pred, solve) in _CANDIDATES if pred(kb)]
    if len(matching) < 2:
        pytest.skip(f"{name}: only one tier-0 solver matches; no dispatch ambiguity")

    covs = {sname: _coverage(solve, kb) for sname, solve in matching}
    dispatched = dispatch(kb).solver_name
    assert dispatched in covs, (
        f"{name}: dispatched to {dispatched!r}, not among the matching tier-0 "
        f"candidates {sorted(covs)}"
    )
    best_name = max(covs, key=lambda k: covs[k])
    assert covs[dispatched] >= covs[best_name] - _COVERAGE_SLACK, (
        f"{name}: dispatched to {dispatched!r} (coverage {covs[dispatched]:.0%}) but "
        f"{best_name!r} covers {covs[best_name]:.0%} -- dispatch is not picking the "
        f"max-coverage solver among matching candidates {covs}. Fix the dispatch "
        f"precedence, don't widen this gate."
    )


def test_j2s7_dispatch_gap_is_real() -> None:
    """Teeth: j2s7s300 matches both approximate predicates, the spherical-
    shoulder solver genuinely under-covers it, and dispatch correctly picks
    srs_polished. This is the concrete regression the gate guards (#440/#442)."""
    mod = importlib.import_module("ssik.prebuilt.kinova.j2s7s300_ik")
    kb = mod._KB
    assert is_approximately_srs_7r(kb, max_drift_m=0.04, policy=POL) is not None
    assert is_approximately_spherical_shoulder_7r(kb, policy=POL)
    cov_srs = _coverage(srs_polished.solve, kb)
    cov_ss = _coverage(spherical_shoulder_polished.solve, kb)
    assert cov_srs > cov_ss + _COVERAGE_SLACK, (
        f"expected a real coverage gap: srs_polished={cov_srs:.0%} vs "
        f"spherical_shoulder_polished={cov_ss:.0%}"
    )
    assert dispatch(kb).solver_name == "seven_r.srs_polished"
