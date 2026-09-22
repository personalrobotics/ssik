"""Seeded tracking must measure the joint motion it actually commands (#562 step 2).

``rewrap_to_seed`` (#562 step 1) fixed the *representative*: a seeded solve now
returns the in-limit winding nearest the seed. It did not fix the *metric*.
``nearest_to_seed`` ranks, and ``within_seed_tolerance`` filters, by the
wrap-to-pi distance ``|wrap(q_i - seed_i)|`` -- the right notion for a joint whose
range is at most a turn, and the wrong one for a joint whose range spans more.

On a wide-limit joint (UR family, ``[-2*pi, 2*pi]``) the rewrapped representative
can sit more than pi from the seed, because the nearer winding is out of limits.
The true motion is then ``|q_i - seed_i|``, which the wrap metric under-reports by
a full turn. Two documented guarantees break:

* ``seed_tolerance`` is described as "the hard guarantee for trajectory tracking
  (no joint jumps more than ``tolerance``)" -- but it admits joints that jump far
  further.
* ``max_solutions=1`` with a seed is documented as the trajectory-tracking idiom,
  returning the nearest IK -- but it can return a branch that moves a joint a full
  revolution more than an available alternative.

The fix ranks and filters on ``|q_i - seed_i|`` for a joint with finite limits,
falling back to the wrap distance only for a continuous joint, where every winding
is reachable and the mod-2*pi distance *is* the true one. These tests were written
against the broken behaviour (as strict xfails) and now pin the fixed one, in both
the Python pipeline and its C++ mirror in ``cpp/include/ssik_cpp/finalize.hpp``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik.prebuilt.universal_robots import ur5e_ik as m

_TWO_PI = 2.0 * np.pi


def _wrap_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(((a - b + np.pi) % _TWO_PI) - np.pi)))


def _true_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


# A pose whose seeded solution set separates the two metrics. The seed sits on
# joint 1's lower limit (-2*pi, in limits), where one branch's only in-limit
# winding is a full turn away: it reads as 0.208 rad under wrap-to-pi and is
# 6.273 rad of real motion, while another branch is 0.242 rad away by both.
_SEED = np.array(
    [2.784346619651, -_TWO_PI, 0.201158842855, -5.115638900669, -0.501080827821, -4.405911714569]
)
_Q_TRUE = np.array([2.800491, -6.253941, -0.040885, -5.122635, -0.674865, -4.614309])


def _seed_case() -> tuple[np.ndarray, np.ndarray]:
    """``(T, seed)`` for the separating pose, with the seed verified in limits."""
    lims = np.array([j.limits for j in m._KB.joints])
    assert np.all((lims[:, 0] <= _SEED) & (lims[:, 1] >= _SEED)), "seed must be in limits"
    return m.fk(_Q_TRUE), _SEED


# ---------------------------------------------------------------------------
# What is already right (#562 step 1) -- guard against regression by the fix.
# ---------------------------------------------------------------------------


def test_seeded_solve_recovers_an_exact_seed() -> None:
    """Seeding with a configuration that is itself a solution returns that
    configuration, not its principal-value representative."""
    q = np.array([6.0, -0.6, 0.9, 5.9, -0.5, 0.2])
    sols = m.solve(m.fk(q), q_seed=q, max_solutions=1)
    assert sols
    assert _true_linf(sols[0].q, q) < 1e-9


def test_every_returned_solution_is_in_limits() -> None:
    """Rewrapping to the seed must not push a joint out of its range."""
    T, seed = _seed_case()
    lims = np.array([j.limits for j in m._KB.joints])
    for s in m.solve(T, q_seed=seed):
        assert np.all(s.q >= lims[:, 0] - 1e-12)
        assert np.all(s.q <= lims[:, 1] + 1e-12)


def test_the_two_metrics_actually_disagree_here() -> None:
    """The premise of the tests below: at this pose the wrap-nearest solution and
    the truly nearest solution are different branches. If this ever stops holding,
    the fixture has drifted and those tests are no longer testing anything."""
    T, seed = _seed_case()
    sols = m.solve(T, q_seed=seed)
    assert len(sols) >= 2
    wrap = [_wrap_linf(s.q, seed) for s in sols]
    true = [_true_linf(s.q, seed) for s in sols]
    assert int(np.argmin(wrap)) != int(np.argmin(true))
    assert max(true) - min(true) > np.pi  # the gap is a real turn, not rounding


# ---------------------------------------------------------------------------
# The fix (#562 step 2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("native", [True, False])
def test_top1_is_the_truly_nearest_solution(native: bool) -> None:
    """``max_solutions=1`` with a seed is the trajectory-tracking idiom: it must
    return the solution that moves the joints least, measured as they move.

    Before the fix the returned branch moved a joint 6.273 rad where another
    solution at the same pose moved it 0.242 rad -- a gratuitous full revolution,
    on both backends."""
    T, seed = _seed_case()
    all_sols = m.solve(T, q_seed=seed, native=native)
    best = min(_true_linf(s.q, seed) for s in all_sols)
    top = m.solve(T, q_seed=seed, max_solutions=1, native=native)
    assert top
    assert _true_linf(top[0].q, seed) == pytest.approx(best, abs=1e-9)


def test_ranking_is_monotone_in_true_joint_motion() -> None:
    """``nearest_to_seed`` orders by distance to the seed; on a wide-limit chain
    that ordering must be by the motion actually commanded."""
    T, seed = _seed_case()
    d = [_true_linf(s.q, seed) for s in m.solve(T, q_seed=seed)]
    assert d == sorted(d)


@pytest.mark.parametrize("tolerance", [0.5, 1.9])
def test_seed_tolerance_bounds_the_true_joint_motion(tolerance: float) -> None:
    """``seed_tolerance`` is documented as a hard per-joint bound ("no joint jumps
    more than ``tolerance``"). Every surviving solution must satisfy it in real
    joint motion; an empty result is the honest answer when none does."""
    T, seed = _seed_case()
    for s in m.solve(T, q_seed=seed, seed_tolerance=tolerance):
        assert _true_linf(s.q, seed) <= tolerance + 1e-9


def test_unwinding_at_the_limit_is_not_silently_within_tolerance() -> None:
    """Following a path that drives a joint past +2*pi forces an unwind of nearly a
    full turn -- physically unavoidable, since no other in-limit winding exists.
    A caller who asked for a 0.2 rad bound must not be handed it unannounced."""
    q = np.array([6.20, -0.6, 0.9, 0.0, -0.5, 0.2])
    seed = m.solve(m.fk(q), q_seed=q, max_solutions=1)[0].q
    q_next = q.copy()
    q_next[0] = 6.36  # past +2*pi = 6.2832, so joint 0 must unwind
    kept = m.solve(m.fk(q_next), q_seed=seed, seed_tolerance=0.2, max_solutions=1)
    assert all(_true_linf(s.q, seed) <= 0.2 + 1e-9 for s in kept)
