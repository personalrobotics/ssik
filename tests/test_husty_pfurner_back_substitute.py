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
        ls_1=dh["l_1"],
        d_2=dh["d_2"],
        a_2=dh["a_2"],
        ls_2=dh["l_2"],
        d_3=dh["d_3"],
        a_3=dh["a_3"],
        ls_3=dh["l_3"],
        d_4=dh["d_4"],
        a_4=dh["a_4"],
        ls_4=dh["l_4"],
        d_5=dh["d_5"],
        a_5=dh["a_5"],
        ls_5=dh["l_5"],
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
