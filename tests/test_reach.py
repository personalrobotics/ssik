"""``ssik.reach.reach_along`` (request A5): every branch's reachable stretches
along a one-parameter family of poses, their ends placed as critical points,
against a per-pose oracle -- the branch's margin scanned densely and its sign
changes bracketed to the same tolerance -- on the Panda and the iiwa14, along a
line and along a screw; and the ends it reports are where the margin changes sign."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq  # type: ignore[import-untyped]

import ssik
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.chart import charts
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.reach import ABSENT, Label, Reach, reach_along

FIXTURES = Path(__file__).parent / "fixtures"
_ARMS = {
    "franka_panda": ("panda_link0", "panda_link8"),
    "kuka_iiwa14": ("base", "iiwa_link_ee_kuka"),
}
LINE = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # slide along the tool axis
SCREW = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 2.0])  # advance 0.5 m per rad spun about it
MATCH = 5e-8  # the oracle's own precision on a bisected fold


def _kb(name: str):
    base, ee = _ARMS[name]
    return load_urdf_kinbody_normalized(FIXTURES / f"{name}.urdf", base, ee)


def _pose(kb, seed: int):
    rng = np.random.default_rng(seed)
    q = np.array([rng.uniform(lo, hi) for lo, hi in (j.limits for j in kb.joints)])
    return poe_forward_kinematics(kb, q)


def oracle(reach: Reach, scan: int = 161) -> dict[Label, list[tuple[float, float]]]:
    """Per-pose: every branch's margin at ``scan`` stops, each sign change bracketed
    by Brent's method (a branch's appearance is a step, which it bisects)."""
    lo, hi = reach.s_range
    stops = np.linspace(lo, hi, scan)
    labels = {label for s in stops for label in reach._family.margins(float(s))}
    out: dict[Label, list[tuple[float, float]]] = {}
    for label in sorted(labels):
        m = np.array([reach._family.margins(float(s)).get(label, (ABSENT, 0.0))[0] for s in stops])
        f = lambda s, lab=label: reach.margin(lab, float(s))  # noqa: E731
        runs: list[tuple[float, float]] = []
        start = None
        for i, inside in enumerate(m >= 0):
            if inside and start is None:
                start = stops[0] if i == 0 else brentq(f, stops[i - 1], stops[i], xtol=1e-9)
            if start is not None and (not inside or i == len(stops) - 1):
                end = stops[-1] if inside else brentq(f, stops[i - 1], stops[i], xtol=1e-9)
                runs.append((float(start), float(end)))
                start = None
        if runs:
            out[label] = runs
    return out


@pytest.mark.parametrize("arm", sorted(_ARMS))
@pytest.mark.parametrize("twist", ["line", "screw"])
def test_reach_matches_the_per_pose_oracle(arm: str, twist: str) -> None:
    kb = _kb(arm)
    xi = LINE if twist == "line" else SCREW
    for seed in (0, 1):
        reach = reach_along(kb, _pose(kb, seed), xi, (-0.4, 0.4))
        expected = oracle(reach)
        assert sorted(reach.intervals) == sorted(expected), (arm, twist, seed)
        for label, intervals in expected.items():
            ours = reach.intervals[label]
            assert len(ours) == len(intervals), (arm, twist, seed, label, ours, intervals)
            for (a, b), (c, d) in zip(ours, intervals, strict=True):
                where = (arm, twist, seed, label, (a, b), (c, d))
                assert abs(a - c) < MATCH, where
                assert abs(b - d) < MATCH, where


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_every_end_is_a_sign_change_of_its_branch(arm: str) -> None:
    kb = _kb(arm)
    reach = reach_along(kb, _pose(kb, 2), LINE, (-0.4, 0.4))
    lo, hi = reach.s_range
    for end in reach.ends:
        inside = next(
            1.0 if abs(end.s - a) < 1e-12 else -1.0
            for a, b in reach.intervals[end.label]
            if abs(end.s - a) < 1e-12 or abs(end.s - b) < 1e-12
        )
        if end.kind == "workspace":
            assert not reach._family.exists(end.s - inside * 1e-6)
            assert reach.margin(end.label, end.s + inside * 1e-6) >= 0
            continue
        assert reach.margin(end.label, end.s + inside * 2e-8) >= -1e-9, end
        assert reach.margin(end.label, end.s - inside * 2e-8) < 1e-9, end
    assert all(lo <= e.s <= hi for e in reach.ends)


def test_manipulator_reach_along_and_margin() -> None:
    arm = ssik.Manipulator.from_prebuilt("panda")
    q0 = np.array([0.3, -0.4, 0.2, -1.8, 0.1, 1.6, 0.5])
    reach = arm.reach_along(arm.fk(q0), LINE, (-0.2, 0.2))
    assert reach.workspace
    assert reach.intervals
    label = reach.labels[0]
    a, b = reach.intervals[label][0]
    mid = 0.5 * (a + b)
    assert reach.margin(label, mid) > 0
    assert np.allclose(reach.pose(0.0), arm.fk(q0))
    family = charts(arm.kinbody, reach.pose(mid), solver_name=arm.solver_name)
    assert any(tuple(c.label) == label and c.margin()[0] > 0 for c in family)
