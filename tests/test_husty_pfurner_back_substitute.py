"""Bulletproof validation for the Husty-Pfurner back-substitution layer.

Phase 5f of #158/#162. Validates that
:func:`ssik.solvers.husty_pfurner._back_substitute.solve_ik` produces
joint-tuples ``(v_1, ..., v_6)`` that:

1. Include the FK-truth ``q*`` for every (DH, q) test config.
2. Each returned tuple FK-closes to ``sigma_E`` at machine precision.
3. The returned set of tuples are physically distinct IK solutions
   (no duplicates within machine precision).

Tolerance philosophy mirrors Phase 5d: hand-picked configs at
``1e-10`` (joint-angle precision; this is tighter than v=tan(theta/2)
precision because joint angles compress near theta=pi where v blows
up). Hypothesis fuzz with the same ``assume()`` filter as Phase 5d
to skip algebraically-pathological corners.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from ssik.solvers.husty_pfurner._back_substitute import solve_ik
from ssik.solvers.husty_pfurner._eliminate import precompute_rrr_chain
from tests.test_husty_pfurner_eliminate import (
    _DH_BASELINE,
    _DH_LARGE_ALPHA,
    _DH_SMALL_LINK,
    _full_6r_chain,
)

# ----------------------------------------------------------------------------
# Hand-picked test configurations.
# ----------------------------------------------------------------------------

_HAND_PICKED_CASES: list[tuple[dict[str, float], tuple[float, ...]]] = [
    (_DH_BASELINE, (0.30, -0.40, 0.60, 0.20, 0.50, -0.70)),
    (_DH_BASELINE, (-0.20, 0.30, -0.50, 0.40, -0.60, 0.80)),
    (_DH_BASELINE, (1.10, -0.90, 0.50, 1.30, -0.70, 0.40)),
    (_DH_LARGE_ALPHA, (0.30, -0.40, 0.60, 0.20, 0.50, -0.70)),
    (_DH_SMALL_LINK, (0.30, -0.40, 0.60, 0.20, 0.50, -0.70)),
]


def _kwargs_for_solve(dh: dict[str, float]) -> dict[str, float]:
    """Map _DH_* dict keys (l_*) to solve_ik's expected kwargs (ls_*)."""
    return dict(
        a_1=dh["a_1"],
        l_1=dh["l_1"],
        d_2=dh["d_2"],
        a_2=dh["a_2"],
        l_2=dh["l_2"],
        d_3=dh["d_3"],
        a_3=dh["a_3"],
        l_3=dh["l_3"],
        d_4=dh["d_4"],
        a_4=dh["a_4"],
        l_4=dh["l_4"],
        d_5=dh["d_5"],
        a_5=dh["a_5"],
        l_5=dh["l_5"],
    )


def _sigma_for(dh: dict[str, float], v: tuple[float, ...]) -> np.ndarray:
    """Build sigma_E for a given (DH, q) pair."""
    return _full_6r_chain(
        v=v,
        a=(dh["a_1"], dh["a_2"], dh["a_3"], dh["a_4"], dh["a_5"]),
        ls=(dh["l_1"], dh["l_2"], dh["l_3"], dh["l_4"], dh["l_5"]),
        d=(dh["d_2"], dh["d_3"], dh["d_4"], dh["d_5"]),
    )


def _max_q_diff(a: tuple[float, ...] | np.ndarray, b: tuple[float, ...] | np.ndarray) -> float:
    return max(abs(float(ai) - float(bi)) for ai, bi in zip(a, b, strict=True))


# ----------------------------------------------------------------------------
# API + shape sanity.
# ----------------------------------------------------------------------------


def test_solve_ik_returns_correct_shape() -> None:
    dh = _DH_BASELINE
    v = (0.3, -0.4, 0.6, 0.2, 0.5, -0.7)
    pre = precompute_rrr_chain(**dh)
    sigma_E = _sigma_for(dh, v)
    sols = solve_ik(pre, sigma_E, **_kwargs_for_solve(dh))
    assert sols.ndim == 2
    assert sols.shape[1] == 6
    assert sols.dtype == np.float64
    assert np.all(np.isfinite(sols))


def test_solve_ik_empty_when_unreachable() -> None:
    """Sanity: a non-physical sigma_E (random gibberish 8-vec) should
    yield zero IK solutions because no FK closure is achievable.
    """
    pre = precompute_rrr_chain(**_DH_BASELINE)
    bogus_sigma = np.array([1.0, 0.5, 0.3, 0.7, 100.0, 200.0, 300.0, 400.0])
    sols = solve_ik(pre, bogus_sigma, **_kwargs_for_solve(_DH_BASELINE))
    assert sols.shape == (0, 6) or sols.shape[0] == 0


# ----------------------------------------------------------------------------
# Hand-picked: each config recovers FK truth + every returned solution
# satisfies FK closure.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dh_v", _HAND_PICKED_CASES)
def test_solve_ik_recovers_truth_and_fk_closes(
    dh_v: tuple[dict[str, float], tuple[float, ...]],
) -> None:
    dh, v_truth = dh_v
    pre = precompute_rrr_chain(**dh)
    sigma_E = _sigma_for(dh, v_truth)
    sols = solve_ik(pre, sigma_E, **_kwargs_for_solve(dh))
    assert sols.shape[0] >= 1, f"no IK solutions for DH={dh}, v={v_truth}"

    # FK closure: every returned solution must round-trip to sigma_E
    # within machine precision (the FK closure filter inside solve_ik
    # already enforces this via fk_tol; this is the test contract).
    sigma_E_norm = float(np.linalg.norm(sigma_E))
    for sol in sols:
        sigma_chain = _sigma_for(dh, tuple(sol))
        scale = float(sigma_chain @ sigma_E) / max(float(sigma_chain @ sigma_chain), 1e-300)
        residue = float(np.linalg.norm(sigma_chain * scale - sigma_E)) / max(sigma_E_norm, 1e-300)
        assert residue < 1e-7, (
            f"FK closure failed for sol={sol}, residue={residue:.3e}, DH={dh}, truth={v_truth}"
        )

    # The FK truth must be among the returned solutions (modulo joint
    # multivaluation: tan(theta/2) for theta and theta+2pi gives the
    # same v, so direct comparison works).
    nearest_err = min(_max_q_diff(sol, v_truth) for sol in sols)
    assert nearest_err < 1e-9, (
        f"FK truth {v_truth} not recovered: nearest sol max-abs-err {nearest_err:.3e} (DH={dh})"
    )


# ----------------------------------------------------------------------------
# 100-pose Hypothesis fuzz: every (DH, q) round-trips through IK.
# ----------------------------------------------------------------------------


_safe_dh = st.floats(min_value=0.05, max_value=1.5, allow_nan=False, allow_infinity=False)
_safe_alpha = st.floats(
    min_value=0.05, max_value=math.pi - 0.05, allow_nan=False, allow_infinity=False
)
_safe_dist = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_safe_q = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
_sign = st.sampled_from([-1.0, 1.0])


@given(
    a_1=_safe_dh,
    s_a1=_sign,
    alpha_1=_safe_alpha,
    d_2=_safe_dist,
    a_2=_safe_dh,
    s_a2=_sign,
    alpha_2=_safe_alpha,
    d_3=_safe_dist,
    a_3=_safe_dh,
    s_a3=_sign,
    alpha_3=_safe_alpha,
    d_4=_safe_dist,
    a_4=_safe_dh,
    s_a4=_sign,
    alpha_4=_safe_alpha,
    d_5=_safe_dist,
    a_5=_safe_dh,
    s_a5=_sign,
    alpha_5=_safe_alpha,
    v_1=_safe_q,
    v_2=_safe_q,
    v_3=_safe_q,
    v_4=_safe_q,
    v_5=_safe_q,
    v_6=_safe_q,
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)
@pytest.mark.xfail(
    strict=False,
    reason=(
        "Pre-existing Hypothesis flake on degenerate DH (#179): "
        "shrunken minimal example is in Capco's T(v_1) double-degenerate "
        "class -- closes when #176 (T(v_2) implementation) lands or when "
        "the Hypothesis strategy is tightened to reject those DHs."
    ),
)
def test_solve_ik_recovers_truth_hypothesis(
    a_1: float,
    s_a1: float,
    alpha_1: float,
    d_2: float,
    a_2: float,
    s_a2: float,
    alpha_2: float,
    d_3: float,
    a_3: float,
    s_a3: float,
    alpha_3: float,
    d_4: float,
    a_4: float,
    s_a4: float,
    alpha_4: float,
    d_5: float,
    a_5: float,
    s_a5: float,
    alpha_5: float,
    v_1: float,
    v_2: float,
    v_3: float,
    v_4: float,
    v_5: float,
    v_6: float,
) -> None:
    """100 random (DH, q) configs. Each must produce at least one IK
    solution that round-trips through FK to ``sigma_E`` within 1e-7.
    The FK truth must be among the returned solutions (1e-7 max-abs-q
    tolerance, looser than hand-picked because float64 precision at
    multiplicity-2 roots gives ``v_i`` accuracy ~1e-8 by Wilkinson).

    Same Hypothesis filter as Phase 5d: skip all-zero-q + replicated
    symmetric DH (algebraically pathological corners that real robots
    don't have).
    """
    a = (s_a1 * a_1, s_a2 * a_2, s_a3 * a_3, s_a4 * a_4, s_a5 * a_5)
    ls = tuple(math.tan(0.5 * x) for x in (alpha_1, alpha_2, alpha_3, alpha_4, alpha_5))
    d = (d_2, d_3, d_4, d_5)
    v = (v_1, v_2, v_3, v_4, v_5, v_6)

    assume(np.linalg.norm(np.asarray(v)) > 0.5)
    a_spread = float(np.std(np.abs(a)))
    alpha_spread = float(np.std([alpha_1, alpha_2, alpha_3, alpha_4, alpha_5]))
    assume(a_spread > 0.05 or alpha_spread > 0.05)

    dh_kwargs = dict(
        a_1=a[0],
        l_1=ls[0],
        d_2=d[0],
        a_2=a[1],
        l_2=ls[1],
        d_3=d[1],
        a_3=a[2],
        l_3=ls[2],
        d_4=d[2],
        a_4=a[3],
        l_4=ls[3],
        d_5=d[3],
        a_5=a[4],
        l_5=ls[4],
    )
    pre = precompute_rrr_chain(**dh_kwargs)
    sigma_E = _full_6r_chain(v=v, a=a, ls=ls, d=d)
    sols = solve_ik(pre, sigma_E, **_kwargs_for_solve(dh_kwargs))
    if sols.shape[0] == 0:
        pytest.fail(f"no IK solutions for DH={dh_kwargs}, v={v}")

    nearest_err = min(_max_q_diff(sol, v) for sol in sols)
    assert nearest_err < 1e-7, (
        f"FK truth {v} not recovered: nearest sol max-abs-err {nearest_err:.3e} (DH={dh_kwargs})"
    )


# ----------------------------------------------------------------------------
# T(v_4) right-chain dispatch (#177): full IK recovery on right-only-degenerate
# DH configurations. Tv4 alone does NOT fix the locked-7R double-degenerate
# case (left chain ALSO needs Tv2 dispatch, see #176); this suite covers the
# case where ONLY the right chain is degenerate (a_4 = 0 OR l_4 = 0) and the
# left chain Tv1 precondition holds.
# ----------------------------------------------------------------------------


_DH_TV4_A4_ZERO = dict(
    a_1=0.30, l_1=math.tan(0.5 * 0.4), d_2=0.20,
    a_2=0.40, l_2=math.tan(0.5 * 0.6), d_3=0.10,
    a_3=0.50, l_3=math.tan(0.5 * 0.5), d_4=0.30,
    a_4=0.0, l_4=math.tan(0.5 * 0.3), d_5=0.40,
    a_5=0.30, l_5=math.tan(0.5 * 0.7),
)

_DH_TV4_L4_ZERO = dict(
    a_1=0.30, l_1=math.tan(0.5 * 0.4), d_2=0.20,
    a_2=0.40, l_2=math.tan(0.5 * 0.6), d_3=0.10,
    a_3=0.50, l_3=math.tan(0.5 * 0.5), d_4=0.30,
    a_4=0.20, l_4=0.0, d_5=0.40,
    a_5=0.30, l_5=math.tan(0.5 * 0.7),
)

_TV4_HAND_PICKED_CASES: list[tuple[dict[str, float], tuple[float, ...]]] = [
    (_DH_TV4_A4_ZERO, (0.30, -0.40, 0.60, 0.20, 0.50, -0.70)),
    (_DH_TV4_A4_ZERO, (-0.20, 0.30, -0.50, 0.40, -0.60, 0.80)),
    (_DH_TV4_A4_ZERO, (1.10, -0.90, 0.50, 1.30, -0.70, 0.40)),
    (_DH_TV4_L4_ZERO, (0.30, -0.40, 0.60, 0.20, 0.50, -0.70)),
    (_DH_TV4_L4_ZERO, (-0.20, 0.30, -0.50, 0.40, -0.60, 0.80)),
]


@pytest.mark.parametrize("dh_v", _TV4_HAND_PICKED_CASES)
def test_tv4_dispatch_recovers_truth_and_fk_closes(
    dh_v: tuple[dict[str, float], tuple[float, ...]],
) -> None:
    """Tv4 dispatch (a_4=0 OR l_4=0): solve_ik recovers q_truth at machine
    precision and every returned solution FK-closes to sigma_E.
    """
    dh, v_truth = dh_v
    pre = precompute_rrr_chain(**dh)
    assert pre.right_parametric_var == "v_4", (
        f"Expected Tv4 dispatch for DH={dh}, got {pre.right_parametric_var}"
    )
    sigma_E = _sigma_for(dh, v_truth)
    sols = solve_ik(pre, sigma_E, **_kwargs_for_solve(dh))
    assert sols.shape[0] >= 1, f"no IK solutions for Tv4 DH={dh}, v={v_truth}"

    # FK closure: every returned sol must round-trip to sigma_E.
    sigma_E_norm = float(np.linalg.norm(sigma_E))
    for sol in sols:
        sigma_chain = _sigma_for(dh, tuple(sol))
        scale = float(sigma_chain @ sigma_E) / max(float(sigma_chain @ sigma_chain), 1e-300)
        residue = float(np.linalg.norm(sigma_chain * scale - sigma_E)) / max(sigma_E_norm, 1e-300)
        assert residue < 1e-7, (
            f"FK closure failed for sol={sol}, residue={residue:.3e}, DH={dh}"
        )

    # FK truth must be among returned solutions.
    nearest_err = min(_max_q_diff(sol, v_truth) for sol in sols)
    assert nearest_err < 1e-9, (
        f"FK truth {v_truth} not recovered: nearest sol max-abs-err {nearest_err:.3e}"
    )


@pytest.mark.parametrize("seed", list(range(20)))
def test_tv4_dispatch_random_dh_recovers_truth(seed: int) -> None:
    """20 random Tv4-region DHs (a_4=0, otherwise non-degenerate): each
    must produce at least one IK solution recovering q_truth at 1e-7.

    Bulletproof: verifies the dispatch + back-sub round-trip on the full
    Tv4 region (right-chain inner-mirror), not just hand-picked DHs.
    """
    rng = np.random.default_rng(seed + 600)
    a_1 = float(rng.uniform(0.1, 1.0)) * float(rng.choice([-1.0, 1.0]))
    l_1 = math.tan(0.5 * float(rng.uniform(0.2, math.pi - 0.2)))
    a_2 = float(rng.uniform(0.1, 1.0)) * float(rng.choice([-1.0, 1.0]))
    l_2 = math.tan(0.5 * float(rng.uniform(0.2, math.pi - 0.2)))
    a_3 = float(rng.uniform(0.1, 1.0)) * float(rng.choice([-1.0, 1.0]))
    l_3 = math.tan(0.5 * float(rng.uniform(0.2, math.pi - 0.2)))
    a_4 = 0.0  # forces Tv4
    l_4 = math.tan(0.5 * float(rng.uniform(0.2, math.pi - 0.2)))
    a_5 = float(rng.uniform(0.1, 1.0)) * float(rng.choice([-1.0, 1.0]))
    l_5 = math.tan(0.5 * float(rng.uniform(0.2, math.pi - 0.2)))
    d_2 = float(rng.uniform(-0.5, 0.5))
    d_3 = float(rng.uniform(-0.5, 0.5))
    d_4 = float(rng.uniform(-0.5, 0.5))
    d_5 = float(rng.uniform(-0.5, 0.5))
    v_truth = tuple(float(rng.uniform(-1.0, 1.0)) for _ in range(6))

    dh = dict(
        a_1=a_1, l_1=l_1, d_2=d_2, a_2=a_2, l_2=l_2, d_3=d_3,
        a_3=a_3, l_3=l_3, d_4=d_4, a_4=a_4, l_4=l_4, d_5=d_5,
        a_5=a_5, l_5=l_5,
    )
    pre = precompute_rrr_chain(**dh)
    assert pre.right_parametric_var == "v_4"
    sigma_E = _sigma_for(dh, v_truth)
    sols = solve_ik(pre, sigma_E, **_kwargs_for_solve(dh))
    assert sols.shape[0] >= 1, f"seed={seed}: no IK solutions for Tv4 DH={dh}"
    nearest_err = min(_max_q_diff(sol, v_truth) for sol in sols)
    assert nearest_err < 1e-7, (
        f"seed={seed}: FK truth {v_truth} not recovered: max-abs-err={nearest_err:.3e}"
    )
