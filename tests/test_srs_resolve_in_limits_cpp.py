"""C++ SRS swivel-limits resolver vs the Python oracle (#515).

``resolve_in_limits`` (exact path) is the rare-case bulletproofing the SRS
prebuilt keeps around the blind swivel sweep: when the sweep samples no in-limits
candidate for a reachable in-limits pose, this recovers the exact solution by
enumerating the <=8 IK branches and taking each branch's feasible-swivel arc
centre. The C++ port (``ssik::srs_swivel::resolve_in_limits``) must reproduce the
Python result set exactly, on both the canonical-ZYZ arms (iiwa14) and the
general concurrent-axis arms (r1pro, openarm) -- ``_Branch.q`` is the general
Davenport solve, so the same port covers all of them.

Skips when the test-only extension isn't built (the cpp CI job builds it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY as _POL
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.seven_r._swivel_limits import _joint_limits, resolve_in_limits
from ssik.solvers.seven_r.srs import (  # type: ignore[attr-defined]
    _arm_constants,
    _classify_srs_7r_geometric,
)
from tests._cpp_backend import _load_ext, cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")

FIXTURES = Path(__file__).parent / "fixtures"

# (name, base, ee, is_canonical) -- a canonical-ZYZ arm + general concurrent-axis
# arms, so the port is exercised on both the offset-free and the general path.
# iiwa14 uses iiwa_link_ee: the ee link carries a rotated home flange
# (R_home != I), so this also exercises the #517 ee-frame gauge-normalization in
# the baked SrsConsts (before that fix the resolver returned nothing here).
_SRS_ARMS = [
    ("kuka_iiwa14", "iiwa_link_0", "iiwa_link_ee", True),
    ("r1pro_left", "left_arm_base_link", "left_arm_link7", False),
    ("openarm_left", "openarm_left_base_link", "openarm_left_ee_base_link", False),
]


def _cpp_args(kb: Any, cls: Any) -> dict[str, Any]:
    """Bake the SrsConsts (base + branch-enumeration extras) for the C++ call."""
    l_se, l_ew, ee_offset, origins = _arm_constants(kb, cls)
    j = kb.joints
    return {
        "axes": np.array([jt.axis for jt in j], dtype=np.float64),
        "t_left": np.array([jt.T_left for jt in j], dtype=np.float64),
        "t_right": np.array([jt.T_right for jt in j], dtype=np.float64),
        "types": np.array([0 if jt.joint_type == "revolute" else 1 for jt in j], dtype=np.int32),
        "l_se": float(l_se),
        "l_ew": float(l_ew),
        "ee_offset": np.asarray(ee_offset, dtype=np.float64),
        "shoulder_pivot": np.asarray(cls.shoulder_pivot, dtype=np.float64),
        "r_post_wrist": np.asarray(j[6].T_right[:3, :3], dtype=np.float64),
        "elbow_index": int(cls.elbow_index),
        "upper_home": np.asarray(origins[cls.elbow_index] - cls.shoulder_pivot, dtype=np.float64),
        "forearm_home": np.asarray(cls.wrist_pivot - origins[cls.elbow_index], dtype=np.float64),
    }


def _sorted_qs(sols: list[Any]) -> list[np.ndarray]:
    """Solutions -> q vectors, ordered canonically for set comparison."""
    qs = [np.asarray(s.q if hasattr(s, "q") else s, dtype=np.float64) for s in sols]
    return sorted(qs, key=lambda q: tuple(np.round(q, 6)))


def _match(py: list[np.ndarray], cpp: list[np.ndarray], tol: float = 1e-9) -> bool:
    if len(py) != len(cpp):
        return False
    return all(np.allclose(p, c, atol=tol) for p, c in zip(py, cpp, strict=True))


@pytest.mark.parametrize(
    ("name", "base", "ee", "canonical"),
    _SRS_ARMS,
    ids=[a[0] for a in _SRS_ARMS],
)
def test_cpp_resolve_matches_python(name: str, base: str, ee: str, canonical: bool) -> None:
    ext = _load_ext()
    kb = load_urdf_kinbody_normalized(FIXTURES / f"{name}.urdf", base, ee)
    cls = _classify_srs_7r_geometric(kb, _POL)
    assert cls is not None, f"{name}: not classified SRS-class"
    args = _cpp_args(kb, cls)
    lims = _joint_limits(kb)
    lo = np.array([a for a, _ in lims], dtype=np.float64)
    hi = np.array([b for _, b in lims], dtype=np.float64)
    fk_atol = float(_POL.subproblem_numerical)

    rng = np.random.default_rng(0)
    mismatch = nonempty = 0
    for _ in range(200):
        q = np.array([rng.uniform(a, b) for a, b in lims])
        T = poe_forward_kinematics(kb, q)

        py = _sorted_qs(resolve_in_limits(kb, T))
        cpp = _sorted_qs(
            ext.srs_resolve_in_limits(
                args["axes"],
                args["t_left"],
                args["t_right"],
                args["types"],
                args["l_se"],
                args["l_ew"],
                args["ee_offset"],
                args["shoulder_pivot"],
                args["r_post_wrist"],
                args["elbow_index"],
                args["upper_home"],
                args["forearm_home"],
                lo,
                hi,
                np.asarray(T, dtype=np.float64),
                fk_atol,
            )
        )
        if not _match(py, cpp):
            mismatch += 1
        nonempty += len(py) > 0

    assert mismatch == 0, f"{name}: {mismatch}/200 resolve_in_limits mismatches vs Python"
    # Sanity: the fuzz actually drove solutions through the resolver.
    assert nonempty > 150, f"{name}: only {nonempty}/200 poses produced a solution"
