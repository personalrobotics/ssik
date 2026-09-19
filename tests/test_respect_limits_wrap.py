"""``respect_limits="wrap"`` (request A4): the full geometric set, each joint
wrapped into its range where a ``+- 2*pi`` representative exists, nothing
dropped -- the raw ``False`` set is unwrapped, so a limit-margin score on it
misreads in-range branches as violations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ssik
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.postprocess import wrap_to_limits

FIXTURES = Path(__file__).parent / "fixtures"


def test_wrap_mode_wraps_without_dropping() -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / "kuka_iiwa14.urdf", "base", "iiwa_link_ee_kuka")
    arm = ssik.Manipulator(kb)
    limits = [j.limits for j in kb.joints]
    assert all(lim is not None for lim in limits)
    lo = np.array([lim[0] for lim in limits if lim is not None])
    hi = np.array([lim[1] for lim in limits if lim is not None])
    rng = np.random.default_rng(3)
    n_raw_out = n_wrap_out = 0
    for _ in range(5):
        q = np.array([rng.uniform(a, b) for a, b in zip(lo, hi, strict=True)])
        T = arm.fk(q)
        raw = arm.solve(T, respect_limits=False)
        wrapped = arm.solve(T, respect_limits="wrap")
        assert len(wrapped) == len(raw)  # nothing dropped
        for a, b in zip(wrapped, wrap_to_limits(raw, kb), strict=True):
            assert np.allclose(a.q, b.q)
        n_raw_out += sum(int(np.any(s.q < lo) or np.any(s.q > hi)) for s in raw)
        n_wrap_out += sum(int(np.any(s.q < lo) or np.any(s.q > hi)) for s in wrapped)
    assert n_wrap_out <= n_raw_out
    sols, diag = arm.solve(T, respect_limits="wrap", explain=True)
    assert diag.dropped_by_limits == 0
    assert len(sols) == len(raw)
