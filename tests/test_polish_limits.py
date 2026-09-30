"""The general_6r polish never takes an accepted candidate across a joint limit (#644).

The polish of accepted candidates (#636) runs before the limit filter. If it
moved a candidate that has a value (or a +-2 pi winding) within the 1e-9 limit
band to just outside the band, the filter would drop a configuration the
solver had accepted. End to end this happens only at exact-limit ties, where
which side of the band the polished root lands on is round-off, so no shipped
pose pins it reproducibly across platforms. These cases build the failure
deterministically instead: a regular root ``q*`` placed 5e-9 rad past a limit
of the FANUC CRX-20iA/L (joint 3, limits +-3 pi / 2, a span over 2 pi), and an
accepted candidate ``q0`` on that limit whose polish converges to ``q*``.

Oracle: joint-limit arithmetic, independent of the polish. Every value
``q0_i + 2 pi k`` inside ``[lo_i - 1e-9, hi_i + 1e-9]`` must also be inside
for the returned configuration, and a rejected polish returns ``q0`` exactly
(the accept-or-keep rule). The control case shows the rule does not block an
ordinary polish that stays inside the band. The native mirror of this test is
``cpp/tests/test_polish_limits.cpp``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.postprocess import _LIMIT_BAND
from ssik.prebuilt.fanuc import crx20ial_ik
from ssik.refinement.polish import POLISH_TARGET, Chain, polish_accepted

_KB = crx20ial_ik._KB
_J = 2  # joint 3: limits +-3 pi / 2
# Every joint of the arm is limited.
_LIMITS = [lim for lim in (j.limits for j in _KB.joints) if lim is not None]
assert len(_LIMITS) == len(_KB.joints)
_LO = _LIMITS[_J][0]
# A regular configuration (sigma_min(J) = 0.034) with joint 3 on its lower limit.
_Q_ON_LIMIT = np.array(
    [
        1.5310050577906944,
        2.169188913994496,
        _LO,
        -1.363346478405587,
        2.7887203364781827,
        3.6332152303383065,
    ]
)


def _in_band_windings(x: float, lo: float, hi: float) -> set[int]:
    ks = range(math.floor((lo - x) / (2 * math.pi)) - 1, math.ceil((hi - x) / (2 * math.pi)) + 2)
    return {k for k in ks if lo - _LIMIT_BAND <= x + 2 * math.pi * k <= hi + _LIMIT_BAND}


def _case(root_offset: float, winding: int) -> tuple[np.ndarray, np.ndarray]:
    """``(q0, T)``: the root has joint 3 at ``lo + root_offset`` (+ 2 pi winding);
    the candidate has it on the limit, the rest of the root unchanged."""
    q_star = _Q_ON_LIMIT.copy()
    q_star[_J] = _LO + root_offset + 2 * math.pi * winding
    q0 = q_star.copy()
    q0[_J] = _LO + 2 * math.pi * winding
    return q0, poe_forward_kinematics(_KB, q_star)


@pytest.mark.parametrize(
    ("root_offset", "winding", "polished"),
    [
        (-5e-9, 0, False),  # the polish would take the value 4e-9 past the band
        (-5e-9, 1, False),  # ... or keep the value (near +pi/2) but lose its -3 pi/2 winding
        (+5e-9, 0, True),  # control: the root is inside, so the polish is kept
    ],
    ids=["past-limit", "lost-winding", "inside-control"],
)
def test_polish_keeps_every_in_limit_winding(
    root_offset: float, winding: int, polished: bool
) -> None:
    q0, t_target = _case(root_offset, winding)
    residual0 = float(np.linalg.norm(poe_forward_kinematics(_KB, q0) - t_target))
    assert 1e-12 < residual0 <= 1e-5  # an accepted candidate that polish would improve

    q, r, ok = polish_accepted(q0[None], t_target, Chain.from_kinbody(_KB))

    for i, (lo, hi) in enumerate(_LIMITS):
        assert _in_band_windings(q0[i], lo, hi) <= _in_band_windings(q[0, i], lo, hi), i
    assert bool(ok[0]) is polished
    if polished:
        assert r[0] <= POLISH_TARGET
    else:
        assert np.array_equal(q[0], q0)
