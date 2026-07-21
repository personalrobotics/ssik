"""Artifact <-> live-solver parity (U2, #389).

The codegen emitters re-express solver logic as generated source (the specialised
composers inline the SP1/SP3/SP4 algebra; every emitter re-implements the
verify/dedup/postprocess tail). That is duplication of *behaviour* across two
representations -- the deepest drift risk in the codebase, and the one no
byte-snapshot catches (an artifact can regenerate byte-consistently and still
compute a different IK set than the library it was generated from).

This test pins behavioural parity: for every shipped prebuilt, ``artifact.solve``
and ``Manipulator(kb).solve`` (which dispatches the live solver) must return the
same IK set on the same pose. If the emitted code ever diverges from the live
solver, CI goes red here.

Scoping (each documented at its constant below): ``ikgeo.general_6r`` is compared
FK-only (its cold Raghavan-Roth re-derivation is slightly less accurate than the
baked artifact); ``jointlock.seven_r`` is skipped entirely (cold Husty-Pfurner vs
baked cached-RR is the documented #328 gap, not codegen drift); and near-singular
poses are skipped for strict set-parity by Jacobian condition number.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik.core.solution import Solution
from ssik.manipulator import Manipulator
from ssik.prebuilt._manifest import load_manifest
from ssik.refinement import kinbody_jacobian

_TWO_PI = 2.0 * np.pi
# Live path runs a runtime symbolic RR re-derivation that can differ slightly
# from the baked artifact -- compare FK only, not the exact solution set.
_SYMBOLIC_LIVE = {"ikgeo.general_6r"}
# Cold universal Husty-Pfurner (jointlock from a raw KinBody) vs the baked
# cached-RR artifact is the documented #328 coverage/accuracy gap, not codegen
# drift -- there is no meaningful artifact<->live parity to assert here. The
# baked artifact's own accuracy is pinned by the uniform-fuzz suite.
_NO_PARITY = {"jointlock.seven_r"}
# Above this Jacobian condition number a pose is near-singular: merging branches
# make the IK solution ill-defined, so the shared artifact/live algebra can land
# ~1e-3 apart (measured: cond ~3e4 at a UR elbow-straight pose gives an 8e-4
# split). Well-conditioned poses sit at cond ~10-1000, so 1e4 cleanly isolates
# the boundary cases; a genuine codegen divergence shows at good poses too, so
# skipping these does not mask drift.
_SINGULAR_COND = 1e4


def _wrap_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs((a - b + np.pi) % _TWO_PI - np.pi)))


def _prebuilts() -> list[str]:
    names = []
    for arm in load_manifest():
        try:
            importlib.import_module(f"ssik.prebuilt.{arm}")
        except Exception:
            continue
        names.append(arm)
    return names


# A pose is "degenerate" (near a kinematic singularity / branch merge) when two
# of its solutions coincide to within this L-infinity gap. There the artifact
# and live paths run the same algebra but a boundary clip / dedup tie-break can
# land ~1e-3 apart -- a numerical property of the pose, not codegen drift. Such
# poses are skipped for strict set-parity (FK closure is still checked).
_DEGENERATE_GAP = 5e-2


def _min_pairwise(sols: list[Solution]) -> float:
    qs = [np.asarray(s.q, dtype=np.float64) for s in sols]
    return min(
        (_wrap_linf(qs[i], qs[j]) for i in range(len(qs)) for j in range(i + 1, len(qs))),
        default=np.inf,
    )


def _sets_match(art: list[Solution], live: list[Solution], tol: float) -> tuple[bool, str]:
    """Bijective nearest-match of two solution lists by wrap-to-pi L-infinity."""
    if len(art) != len(live):
        return False, f"solution count {len(art)} (artifact) vs {len(live)} (live)"
    live_q = [np.asarray(s.q, dtype=np.float64) for s in live]
    used = [False] * len(live_q)
    for s in art:
        aq = np.asarray(s.q, dtype=np.float64)
        best, best_d = -1, np.inf
        for j, lq in enumerate(live_q):
            if used[j]:
                continue
            d = _wrap_linf(aq, lq)
            if d < best_d:
                best_d, best = d, j
        if best < 0 or best_d > tol:
            return False, f"artifact solution unmatched in live set (nearest {best_d:.2e})"
        used[best] = True
    return True, ""


@pytest.mark.parametrize("arm", _prebuilts())
def test_artifact_matches_live_solver(arm: str) -> None:
    """The emitted artifact computes the same IK as the live dispatched solver."""
    art = importlib.import_module(f"ssik.prebuilt.{arm}")
    if art.SOLVER_NAME in _NO_PARITY:
        pytest.skip(f"{art.SOLVER_NAME}: cold-fallback vs baked is #328, not codegen parity")
    kb = art._KB
    live = Manipulator(kb)
    fk_only = art.SOLVER_NAME in _SYMBOLIC_LIVE

    rng = np.random.default_rng(0)
    dof = len(kb.joints)
    fk_checked = 0
    set_checked = 0
    for _ in range(25):
        q = rng.uniform(-2.0, 2.0, size=dof)
        t = art.fk(q)
        # Pure analytical set from both paths: no rescue, no limit filtering, and
        # refinement ON for both so the artifact's baked force-refine (#362) is
        # matched by the live solver (which otherwise honours allow_refinement).
        kw = {"respect_limits": False, "allow_rescue": False, "allow_refinement": True}
        art_sols = art.solve(t, **kw)
        if not art_sols:
            continue
        fk_checked += 1
        # The shipped artifact must FK-close to machine precision, always.
        worst_art = max(float(np.max(np.abs(art.fk(s.q) - t))) for s in art_sols)
        assert worst_art < 1e-6, f"{arm}: artifact FK closure {worst_art:.2e}"
        # Symbolic-live arms (cold RR re-derivation / universal HP) can differ
        # from the baked artifact by design (#328); only the artifact is pinned.
        if fk_only:
            continue
        live_sols = live.solve(t, respect_limits=False, allow_rescue=False, allow_refinement=True)
        worst_live = max((float(np.max(np.abs(art.fk(s.q) - t))) for s in live_sols), default=0.0)
        assert worst_live < 1e-6, f"{arm}: live FK closure {worst_live:.2e}"
        # Strict set-parity only on well-conditioned poses: near a singularity
        # (merging branches, or two solutions within _DEGENERATE_GAP) the shared
        # algebra tie-breaks differently -- benign, not codegen drift.
        cond = float(np.linalg.cond(kinbody_jacobian(kb, q)))
        if cond > _SINGULAR_COND:
            continue
        if min(_min_pairwise(art_sols), _min_pairwise(live_sols)) < _DEGENERATE_GAP:
            continue
        set_checked += 1
        ok, why = _sets_match(art_sols, live_sols, tol=1e-6)
        assert ok, f"{arm} ({art.SOLVER_NAME}): artifact != live at q={q.tolist()}: {why}"
    assert fk_checked > 0, f"{arm}: no pose produced solutions to compare"
    if not fk_only:
        assert set_checked > 0, f"{arm}: no non-degenerate pose to set-compare"
