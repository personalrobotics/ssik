"""HP must never raise on any pose -- it returns ``([], is_ls=True)`` for
poses it cannot solve, never an unhandled exception.

Regression for the singular-matrix crash: on general (non-locked) 6R
geometries such as the FANUC CRX-10iA/L, HP's interpolation-based
elimination hits a singular Cramer/pencil matrix on a large fraction of
poses. Before the fix, ``eliminate_uw_pairs`` re-raised that
``LinAlgError`` (and a second unguarded ``np.linalg.solve`` in the
back-substitution path could raise too), so ``solve()`` crashed instead
of degrading to "no solution here". ~59/100 random FANUC poses crashed.

HP is never *dispatched* on general 6R (Raghavan-Roth is the tier-2
solver for those; HP only runs on locked-7R sub-chains). But a caller
that invokes HP directly on such an arm must get the empty-result
contract, not an exception.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.husty_pfurner.general_6r import solve as hp_solve

REPO = Path(__file__).resolve().parent.parent
FANUC_URDF = REPO / "tests" / "fixtures" / "fanuc_crx10ial.urdf"

# A pose that reproduced the singular-matrix crash before the fix
# (every drop index singular in ``eliminate_uw_pairs``).
_CRASHING_Q = np.array(
    [
        2.245637299378165,
        -2.930568260297326,
        1.442967726722393,
        -2.0379158390962826,
        2.281920468786123,
        0.2605085298868298,
    ]
)


@pytest.fixture(scope="module")
def fanuc_kb():
    return load_urdf_kinbody_normalized(FANUC_URDF, "base_link", "flange")


def test_hp_pinned_singular_pose_does_not_crash(fanuc_kb):
    """The exact pose that used to raise ``LinAlgError`` now returns
    cleanly, and an empty result carries ``is_ls=True``."""
    T = poe_forward_kinematics(fanuc_kb, _CRASHING_Q)
    sols, is_ls = hp_solve(fanuc_kb, T, allow_refinement=True)
    # Whatever HP returns, it must not raise. If it found nothing, the
    # least-squares flag must say so (the documented empty contract).
    if not sols:
        assert is_ls, "HP returned no solutions but is_ls=False"
    # Any returned solution must be a real FK-closing IK, not a spurious
    # degenerate root that slipped through.
    for s in sols:
        assert s.fk_residual < 1e-6, f"spurious HP solution: fk={s.fk_residual:.3e}"


def test_hp_never_raises_on_fanuc_fuzz(fanuc_kb):
    """No random FANUC pose escapes an unhandled ``LinAlgError`` from HP.
    Before the fix, ~59/100 crashed."""
    rng = np.random.default_rng(0)
    crashes = 0
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, size=6)
        T = poe_forward_kinematics(fanuc_kb, q)
        try:
            sols, is_ls = hp_solve(fanuc_kb, T, allow_refinement=True)
        except np.linalg.LinAlgError:
            crashes += 1
            continue
        if not sols:
            assert is_ls, "HP returned no solutions but is_ls=False"
    assert crashes == 0, f"HP raised LinAlgError on {crashes}/100 FANUC poses"
