"""``respect_limits`` monotonicity invariant across every shipped prebuilt.

The default ``solve()`` path filters the raw geometric solution set down to
in-limit branches (with a ``q +/- 2pi`` rescue). So the unfiltered set is a
*superset in pose coverage*: if ``solve(T, respect_limits=True)`` finds an IK
for a pose, ``solve(T, respect_limits=False)`` must too. Anything else means
the raw path is dropping solutions the limited path keeps -- a solver bug.

This guards against the iiwa7-class regression (deferred in #424): iiwa7
returns 30 solutions with limits but 0 without, which would make the
``respect_limits=False`` uniform-fuzz gate fail a genuinely-working arm.
Every arm shipped today satisfies the invariant; this test keeps it that way.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik.prebuilt._manifest import load_manifest

_MANIFEST = load_manifest()
_ARM_NAMES = sorted(_MANIFEST)


@pytest.mark.parametrize("arm_name", _ARM_NAMES)
def test_respect_limits_is_superset(arm_name: str) -> None:
    """For poses the default (limited) path solves, the raw path must solve
    too. Sampled in a moderate reachable range shared by all arms."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    rng = np.random.default_rng(0)
    for _ in range(16):
        # A conservative range well inside every shipped arm's joint limits,
        # so FK(q) is genuinely reachable and both paths should solve it.
        q = rng.uniform(-1.2, 1.2, size=mod.DOF)
        t_target = mod.fk(q)
        if not mod.solve(t_target):
            continue  # default path did not cover this pose; nothing to assert
        assert mod.solve(t_target, respect_limits=False), (
            f"{arm_name}: solve(respect_limits=False) returned no IK for a pose "
            f"the default path solved (q={q.tolist()}). The raw geometric set "
            f"must be a superset of the limit-filtered set."
        )
