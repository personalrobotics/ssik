"""``Chart.restrict``: narrowing a branch's domain by an arbitrary constraint.

A chart is the geometric object -- ``FK^-1(T)`` and nothing else. Anything that is
not a property of the kinematics (self-collision, a task-space obstacle, a keep-out
region) is a restriction the caller composes on top, which is what ``restrict``
is. It returns a :class:`Chart`, so the narrowing survives into ``q``, ``contains``,
``curve`` and ``in_limits``, and restrictions chain.

Two forms, deliberately different in strength:

* a **margin** (float, ``>= 0`` feasible) has its zero bisected, so boundaries are
  exact to ``tol`` however coarse the sampling grid is;
* a **boolean** is a step, so its boundary is only as good as the grid, and each
  kept interval is shrunk by one step unless the caller opts out.

The tests below pin that distinction, since it is the whole reason both forms
exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik.chart import charts
from ssik.prebuilt.franka import panda_ik as pk

_Q = np.array([0.3, -0.4, 0.2, -1.8, 0.1, 1.6, 0.5])


@pytest.fixture(scope="module")
def family():
    return charts(pk._KB, pk.fk(_Q), solver_name=pk.SOLVER_NAME)


@pytest.fixture(scope="module")
def chart(family):
    return family.charts[0]


def _cut(chart) -> float:
    """A threshold on joint 3 that genuinely splits at least one domain interval.

    The midpoint of the chart's *global* range is not enough: joint 3 can occupy
    two separated bands on two separated intervals, and a cut in the gap selects
    whole intervals without ever crossing the constraint. Bisecting the single
    interval with the widest span guarantees a real crossing.
    """
    best, span = None, -1.0
    for lo, hi in chart.domain:
        v = chart.q(np.linspace(lo, hi, 64))[:, 3]
        v = v[np.isfinite(v)]
        if v.size and float(v.max() - v.min()) > span:
            best, span = float(0.5 * (v.max() + v.min())), float(v.max() - v.min())
    assert best is not None, "fixture has no interval to split"
    assert span > 1e-3, "fixture has no interval to split"
    return best


def test_margin_boundaries_are_bisected_not_sampled(chart) -> None:
    """A margin's zero is found exactly, so a boundary introduced by the
    restriction sits on the constraint -- not on the nearest grid point."""
    cut = _cut(chart)
    r = chart.restrict(lambda qs: cut - qs[:, 3], samples=21)  # deliberately coarse
    assert r.domain
    parent_edges = {round(v, 9) for lo, hi in chart.domain for v in (lo, hi)}
    introduced = [t for lo, hi in r.domain for t in (lo, hi) if round(t, 9) not in parent_edges]
    assert introduced, "the restriction must introduce at least one new boundary"
    for t in introduced:
        assert chart.q(t)[3] == pytest.approx(cut, abs=1e-8)


def test_restriction_actually_holds_along_the_curve(chart) -> None:
    """``curve`` on a restricted chart may not sample a forbidden configuration."""
    cut = _cut(chart)
    r = chart.restrict(lambda qs: cut - qs[:, 3])
    sampled = np.concatenate([qs[:, 3] for _, qs in r.curve(400)])
    assert sampled.max() <= cut + 1e-9


def test_boolean_is_conservative_by_default(chart) -> None:
    """A step predicate cannot resolve better than its grid, so the default keeps
    strictly less than the margin form -- and opting out recovers it."""
    cut = _cut(chart)
    margin = chart.restrict(lambda qs: cut - qs[:, 3])
    shrunk = chart.restrict(lambda qs: qs[:, 3] < cut)
    raw = chart.restrict(lambda qs: qs[:, 3] < cut, conservative=False)
    assert shrunk.domain, "the shrink must not annihilate resolvable intervals"
    for lo, hi in shrunk.domain:
        assert any(lo >= a - 1e-9 and hi <= b + 1e-9 for a, b in margin.domain)
    assert len(raw.domain) == len(margin.domain)


def test_clearance_shifts_the_boundary(chart) -> None:
    """``clearance`` is an offset on the margin, so it keeps strictly less."""
    cut = _cut(chart)
    tight = chart.restrict(lambda qs: cut - qs[:, 3])
    banded = chart.restrict(lambda qs: cut - qs[:, 3], clearance=0.02)
    assert sum(b - a for a, b in banded.domain) < sum(b - a for a, b in tight.domain)


def test_restrictions_compose_and_preserve_branch_identity(chart) -> None:
    """Chaining intersects, and the label is the branch's identity across poses --
    restricting must not invent a new branch."""
    cut = _cut(chart)
    first = chart.restrict(lambda qs: cut - qs[:, 3])
    both = first.restrict(lambda qs: qs[:, 5] - 0.5)
    assert both.label == chart.label
    assert sum(b - a for a, b in both.domain) <= sum(b - a for a, b in first.domain) + 1e-9
    for lo, hi in both.domain:
        assert any(lo >= a - 1e-9 and hi <= b + 1e-9 for a, b in first.domain)


def test_contains_and_in_limits_respect_the_restriction(chart) -> None:
    cut = _cut(chart)
    r = chart.restrict(lambda qs: cut - qs[:, 3])
    forbidden = [t for lo, hi in chart.domain for t in (0.5 * (lo + hi),) if chart.q(t)[3] > cut]
    for t in forbidden:
        assert chart.contains(t)
        assert not r.contains(t)
    for lo, hi in r.in_limits():
        assert any(lo >= a - 1e-9 and hi <= b + 1e-9 for a, b in r.domain)


def test_an_impossible_constraint_empties_the_chart(chart) -> None:
    """Empty is a real answer: this branch is entirely forbidden."""
    assert chart.restrict(lambda qs: np.full(len(qs), -1.0)).domain == ()


def test_a_vacuous_constraint_returns_the_same_domain(chart) -> None:
    r = chart.restrict(lambda qs: np.full(len(qs), 1.0))
    assert len(r.domain) == len(chart.domain)
    for (a, b), (c, d) in zip(r.domain, chart.domain, strict=True):
        assert a == pytest.approx(c, abs=1e-9)
        assert b == pytest.approx(d, abs=1e-9)


def test_family_restrict_drops_emptied_charts(family) -> None:
    kept = family.restrict(lambda qs: np.full(len(qs), 1.0))
    assert len(kept) == len(family)
    gone = family.restrict(lambda qs: np.full(len(qs), -1.0))
    assert len(gone) == 0


def test_predicate_never_sees_off_branch_nan(chart) -> None:
    """``q(t)`` is NaN outside the branch; those samples are infeasible by
    construction and must not reach the caller's predicate."""
    seen: list[np.ndarray] = []

    def fn(qs):
        seen.append(qs.copy())
        return np.full(len(qs), 1.0)

    chart.restrict(fn)
    assert seen
    assert all(np.all(np.isfinite(block)) for block in seen)


def test_restricted_family_keeps_its_pose_context(family) -> None:
    """A restricted family narrows the charts and nothing else. The pose-level
    methods (``gap``, ``escape``, ``drift_to_merge``) need the family's ``kb``,
    ``T`` and ``solver_name``; re-constructing without them leaves a family that
    looks fine until one of those is called."""
    restricted = family.restrict(lambda qs: np.full(len(qs), 1.0))
    assert len(restricted) == len(family)
    plain = family.gap(family.charts[0], family.charts[1])
    narrowed = restricted.gap(restricted.charts[0], restricted.charts[1])
    for a, b in zip(plain, narrowed, strict=True):
        assert np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    restricted.escape(restricted.charts[0], restricted.charts[1])  # must not raise
