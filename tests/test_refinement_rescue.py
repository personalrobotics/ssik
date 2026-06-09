"""T-perturbation rescue (#319) — coverage on Group A reproducers.

Five of the eight outstanding coverage gaps share the same root cause:
``m_quad`` is structurally rank-deficient on a measure-zero ridge in
q-space, the analytical RR path returns 0 sols at the exact ridge, but
genuine analytical solutions exist arbitrarily close. The T-perturbation
rescue (see :mod:`ssik.refinement.rescue`) recovers them by perturbing
the target, re-solving, and LM-refining each candidate back to the
original ``T_target``.

These tests pin the rescue's coverage on the falsifying examples that
landed each of #298, #304, and #280 in the issue tracker. If the rescue
ever stops recovering them, this file fires.

#286 PiPER and #309 OpenArm right are *not* tested here yet — #286's
stored q* no longer reproduces (the Hypothesis sweep moved on to a
different example), and #309's reproducer needs an independent
verification pass.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik.refinement.rescue import rescue_via_T_perturbation

# Group A reproducers from the issue tracker. Each entry: (arm_module,
# q*, n_expected_min) — n_expected_min is the empirically observed
# minimum recovered sols. The rescue should recover at least that many
# on every run (the RNG is fixed-seeded in the rescue, so the count is
# deterministic; using a lower-bound gate so future RNG / numerics
# improvements that find more sols don't break the test).
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
    sols = mod.solve(T, respect_limits=False)
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
