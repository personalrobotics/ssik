"""T-perturbation rescue (#319) — coverage on Group A reproducers.

Five of the eight outstanding coverage gaps share the same root cause:
``m_quad`` is structurally rank-deficient on a measure-zero ridge in
q-space, the analytical RR path returns 0 sols at the exact ridge, but
genuine analytical solutions exist arbitrarily close. The T-perturbation
rescue (see :mod:`ssik.refinement.rescue`) recovers them by perturbing
the target, re-solving, and LM-refining each candidate back to the
original ``T_target``.

These tests pin coverage on the falsifying examples that landed #298,
#304, and #280 — at three levels: the analytical path still returns 0
there (baseline), the standalone rescue recovers them, and bulletproof
``solve()`` (allow_rescue=True, the default) recovers them with no
explicit opt-in. If any layer regresses, this file fires.

#309 OpenArm right is exercised via the uniform-fuzz sweep rather than
here; #286 PiPER's stored q* no longer reproduces (the sweep moved on).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik.refinement.rescue import rescue_via_T_perturbation

# Group A reproducers from the issue tracker. Each entry: (arm_module,
# q*, n_expected_min). n_expected_min is a conservative lower bound: the
# escalating-scale schedule gives large cross-platform margin (measured
# default-seed recovery: CRX 4 (structural), Kassow 24, Rizon 4 26), so
# >=4 holds even under BLAS-backend numeric drift. A lower bound (not an
# exact count) keeps the test stable across platforms and future numerics.
GROUP_A = [
    pytest.param(
        "fanuc_crx10ial_ik",
        np.array([0.0, -0.21484375, -1.57884726, 0.25, -0.21484375, 0.0]),
        4,
        id="298_crx",
    ),
    pytest.param(
        "rizon4_ik",
        np.array([0.0, 1.0, -2.0, 0.375, -2.0, 1.0, 0.0]),
        4,
        id="304_rizon4",
    ),
    pytest.param(
        "kassow_kr810_ik",
        np.array([0.0, 1.0, 2.5, 0.5, 1.0, 1.0, 0.0]),
        4,
        id="280_kassow",
    ),
    # #299 gen3 exercises the *thin-wrapper* rescue path (seven_r.srs_polished),
    # which #358 wired to match the orchestrator arms above. Raw srs_polished
    # returns 0 sols here; the rescue recovers dozens (86 on macOS).
    pytest.param(
        "gen3_ik",
        np.array([0.51171875, 0.51171875, 1.01953125, 0.625, -0.6484375, -2.75, 0.875]),
        4,
        id="299_gen3",
    ),
]


@pytest.mark.parametrize(("arm_name", "q_star", "n_expected_min"), GROUP_A)
def test_direct_solve_at_ridge_returns_zero(
    arm_name: str, q_star: np.ndarray, n_expected_min: int
) -> None:
    """Establishes baseline: the direct analytical path returns 0 sols
    at the ridge point. If this ever starts returning non-zero,
    something fixed the ridge upstream and the rescue test below would
    need to be updated to a still-failing q*."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    T = mod.fk(q_star)
    # allow_rescue=False isolates the purely-analytical path: bulletproof
    # solve() now auto-recovers these ridges via the rescue (#319), so the
    # "direct returns 0" baseline must opt out of that fallback.
    sols = mod.solve(T, respect_limits=False, allow_rescue=False)
    assert len(sols) == 0, (
        f"{arm_name}: direct solve at the stored ridge q* now returns "
        f"{len(sols)} sols; the ridge is healed and this reproducer is "
        "stale. Update the test parametrise table or close the upstream issue."
    )


@pytest.mark.parametrize(("arm_name", "q_star", "n_expected_min"), GROUP_A)
def test_rescue_recovers_solutions_at_ridge(
    arm_name: str, q_star: np.ndarray, n_expected_min: int
) -> None:
    """The T-perturbation rescue should recover ≥ ``n_expected_min``
    unique analytical solutions at the ridge q*, each with FK closure
    well below 1e-6."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    T = mod.fk(q_star)

    rescued = rescue_via_T_perturbation(mod.fk, mod.solve, T)

    assert len(rescued) >= n_expected_min, (
        f"{arm_name}: T-rescue recovered {len(rescued)} sols at q*; "
        f"expected at least {n_expected_min}"
    )

    # Each rescued solution must close the original T at the documented
    # fk_atol; reverify independently here (defence against any
    # rescue-internal bug that might emit wrongly-tagged sols).
    for sol in rescued:
        T_check = mod.fk(sol.q)
        fk_err = float(np.linalg.norm(T_check - T))
        assert fk_err < 1e-6, (
            f"{arm_name}: rescued sol q={sol.q.tolist()} has independent "
            f"FK error {fk_err:.2e}, above 1e-6 ceiling"
        )
        assert sol.refinement_used == "lm", (
            f"{arm_name}: rescued sol should be tagged refinement_used='lm', "
            f"got {sol.refinement_used!r}"
        )


@pytest.mark.parametrize(("arm_name", "q_star", "n_expected_min"), GROUP_A)
def test_bulletproof_solve_auto_recovers_ridge(
    arm_name: str, q_star: np.ndarray, n_expected_min: int
) -> None:
    """Default ``solve()`` (allow_rescue=True) must itself recover the ridge.

    This is the #319 bulletproof contract: callers don't invoke the rescue
    explicitly -- ``solve()`` returns the IK at a reachable ridge directly,
    tagged ``refinement_used="lm"`` and FK-closed to machine precision."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    T = mod.fk(q_star)

    sols = mod.solve(T, respect_limits=False)

    assert len(sols) >= n_expected_min, (
        f"{arm_name}: bulletproof solve() recovered {len(sols)} sols at the "
        f"ridge q*; expected at least {n_expected_min}"
    )
    for sol in sols:
        assert sol.refinement_used == "lm", (
            f"{arm_name}: ridge sol should be tagged 'lm', got {sol.refinement_used!r}"
        )
        fk_err = float(np.linalg.norm(mod.fk(sol.q) - T))
        assert fk_err < 1e-6, f"{arm_name}: ridge sol FK error {fk_err:.2e} above 1e-6 ceiling"


def test_bulletproof_solve_skips_rescue_when_unreachable() -> None:
    """A pose just beyond the workspace must return ``[]`` without the rescue
    firing -- the reachability verdict / reach-sphere gate it out so an
    unreachable target stays cheap and honest (no fabricated near-misses)."""
    from ssik.prebuilt import fanuc_crx10ial_ik as crx

    # A genuine arm pose pushed radially out past the CRX's ~1.675 m reach.
    T = crx.fk(np.array([0.3, 0.5, 0.2, 0.4, 0.6, -0.3]))
    T[:3, 3] *= 1.0 + 0.5 / float(np.linalg.norm(T[:3, 3]))

    assert crx.solve(T, respect_limits=False) == []
    assert crx.solve(T, respect_limits=False, allow_rescue=False) == []


def test_rescue_returns_empty_when_T_is_truly_unreachable() -> None:
    """When ``T_target`` is genuinely unreachable (well outside the arm's
    workspace), the rescue should return an empty list rather than
    fabricating bogus near-misses. We use a target 10 m in front of the
    CRX base, far beyond its ~1.5 m reach."""
    from ssik.prebuilt import fanuc_crx10ial_ik

    T_unreachable = np.eye(4)
    T_unreachable[:3, 3] = [10.0, 0.0, 1.0]
    rescued = rescue_via_T_perturbation(
        fanuc_crx10ial_ik.fk,
        fanuc_crx10ial_ik.solve,
        T_unreachable,
    )
    assert rescued == [], (
        f"rescue returned {len(rescued)} sols for an obviously-unreachable "
        "target 10m from the CRX base; expected empty list"
    )


def test_rescue_idempotent_when_direct_solve_succeeds() -> None:
    """When the direct solve already has solutions (no ridge), the
    rescue should still work and not corrupt them — though typical use
    only invokes rescue on empty direct-solve results."""
    from ssik.prebuilt import fanuc_crx10ial_ik

    q_healthy = np.array([0.3, 0.5, 0.2, 0.4, 0.6, -0.3])
    T = fanuc_crx10ial_ik.fk(q_healthy)
    direct = fanuc_crx10ial_ik.solve(T, respect_limits=False)
    assert direct, "expected direct solve to succeed on the healthy seed"

    rescued = rescue_via_T_perturbation(
        fanuc_crx10ial_ik.fk,
        fanuc_crx10ial_ik.solve,
        T,
    )
    # The rescue may find some / all / more of the direct solutions
    # depending on which perturbation regions happen to be reachable.
    # The contract is just that whatever it returns is FK-correct.
    for sol in rescued:
        T_check = fanuc_crx10ial_ik.fk(sol.q)
        fk_err = float(np.linalg.norm(T_check - T))
        assert fk_err < 1e-6, (
            f"rescue produced FK-incorrect sol on healthy pose: "
            f"q={sol.q.tolist()}, fk_err={fk_err:.2e}"
        )


def test_rescue_polishes_well_conditioned_to_machine_precision_384() -> None:
    """#384 regression: a rescue whose perturbation lands off-ridge is
    well-conditioned and must be polished to machine precision, not merely to
    the loose 1e-8 acceptability gate.

    The gen3 pose below routes to ``srs_polished``, which returns [] (its
    approximate pivots don't clear the strict polish filter here), so the
    thin-wrapper rescue fires. Its solutions FK-close to <=~6e-9 under the old
    early-stopping polish -- above gen3's 1e-9 fuzz ceiling -- even though one
    more Newton step reaches ~5e-13. The tight-then-loose polish now delivers
    that: every returned branch closes well under the ceiling.
    """
    from ssik.prebuilt import gen3_ik

    q_star = np.array([0.125, 0.46875, 2.125, 0.4765625, 1.5, 1.0, 0.5])
    T = gen3_ik.fk(q_star)
    sols = gen3_ik.solve(T, respect_limits=False)
    assert sols, "expected the rescue to recover solutions at this gen3 pose"
    worst = max(s.fk_residual for s in sols)
    assert worst < 1e-9, (
        f"rescue left a branch under-polished at {worst:.2e} (> gen3 1e-9 ceiling); "
        f"the tight polish should reach machine precision on this well-conditioned pose"
    )
