"""Property-based fuzzing of the POE URDF normalizer.

The normalizer in [src/ssik/_urdf.py](../src/ssik/_urdf.py) is load-bearing for
every downstream solver in the rebuild (see #37). If it silently mis-rotates
an axis or drops a fixed-joint offset, every subproblem solver built on top
inherits the bug and misdiagnoses as a dispatcher/subproblem error rather than
a loader error. The hand-written tests in
[test_urdf_normalize.py](test_urdf_normalize.py) cover a handful of fixtures;
this file fuzzes a much larger input distribution:

- Chain length 1 to 5 active joints, with up to 3 fixed joints interspersed.
- Each origin has a random ``xyz`` in [-2, 2]³ and a random ``rpy`` in [-π, π]³.
- Each active joint picks revolute or prismatic, and a random (normalized) axis.

The properties asserted on every generated URDF:

1. **FK equivalence** — raw and normalized loaders produce identical FK at
   random ``q``.
2. **POE invariants** — on the normalized chain, ``T_left`` is pure translation
   for every joint, ``T_right`` is identity for joints ``0..n-2``, and every
   axis is unit-length (within numerical tolerance).
3. **Home pose** — at ``q = 0``, the normalized chain's FK matches the raw
   chain's FK (which is a sanity check that also catches any bug in ``T_right``
   on the last joint).

Separate regression cases cover real-world URDF rough edges we've seen in the
wild: axes not-quite-unit-length from Xacro rounding, near-orthogonal rpy,
fixed-joints with rotation in the middle of a chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from ssik._urdf import load_urdf_kinbody, load_urdf_kinbody_normalized

# --------------------------------------------------------------------------- #
# FK helpers (revolute + prismatic aware)
# --------------------------------------------------------------------------- #


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation as a 4x4."""
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return np.eye(4)
    k = axis / norm
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return T


def _translate_along(axis: np.ndarray, distance: float) -> np.ndarray:
    """Pure translation by ``distance * axis`` as a 4x4."""
    T = np.eye(4)
    T[:3, 3] = axis * distance
    return T


def _fk(kb: Any, q: np.ndarray) -> np.ndarray:
    """Compose the chain ``T_left @ motion(axis, qᵢ) @ T_right`` per joint.

    Handles both revolute (axis-angle rotation) and prismatic (translation
    along axis) joints.
    """
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        if j.joint_type == "revolute":
            motion = _axis_angle(j.axis, qi)
        else:  # prismatic
            motion = _translate_along(j.axis, qi)
        T = T @ j.T_left @ motion @ j.T_right
    return T


# --------------------------------------------------------------------------- #
# URDF emission from a Python chain spec
# --------------------------------------------------------------------------- #


def _urdf_text(chain: list[dict[str, Any]]) -> str:
    n = len(chain)
    link_names = ["base_link", *[f"link_{i}" for i in range(1, n)], "ee_link"]
    lines = ['<?xml version="1.0"?>', '<robot name="fuzz">']
    for name in link_names:
        lines.append(f'  <link name="{name}"/>')
    for i, joint in enumerate(chain):
        xyz = " ".join(f"{v!r}" for v in joint["xyz"])
        rpy = " ".join(f"{v!r}" for v in joint["rpy"])
        lines.append(f'  <joint name="j{i}" type="{joint["type"]}">')
        lines.append(f'    <parent link="{link_names[i]}"/>')
        lines.append(f'    <child link="{link_names[i + 1]}"/>')
        lines.append(f'    <origin xyz="{xyz}" rpy="{rpy}"/>')
        if joint["axis"] is not None:
            axis = " ".join(f"{v!r}" for v in joint["axis"])
            lines.append(f'    <axis xyz="{axis}"/>')
        if joint["type"] in ("revolute", "prismatic"):
            lines.append('    <limit effort="1" velocity="1" lower="-3.14" upper="3.14"/>')
        lines.append("  </joint>")
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Hypothesis strategies
# --------------------------------------------------------------------------- #

_finite = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False, width=64)
_angle = st.floats(
    min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False, width=64
)


@st.composite
def _unit_axis(draw: st.DrawFn) -> tuple[float, float, float]:
    """Sample a direction in ℝ³ and normalize. Rejects near-zero vectors."""
    v = np.array([draw(_finite), draw(_finite), draw(_finite)])
    norm = float(np.linalg.norm(v))
    assume(norm > 1e-3)  # reject degenerate (would fail FK tolerance checks)
    v = v / norm
    return (float(v[0]), float(v[1]), float(v[2]))


@st.composite
def _joint(draw: st.DrawFn, joint_type: str) -> dict[str, Any]:
    return {
        "type": joint_type,
        "xyz": (draw(_finite), draw(_finite), draw(_finite)),
        "rpy": (draw(_angle), draw(_angle), draw(_angle)),
        "axis": draw(_unit_axis()) if joint_type in ("revolute", "prismatic") else None,
    }


@st.composite
def _chain(draw: st.DrawFn) -> list[dict[str, Any]]:
    """A URDF chain of 1 to 5 active joints, up to 3 fixed joints interleaved.

    Active joints are mostly revolute (more common in 6R arms) but sometimes
    prismatic. Fixed joints can appear anywhere — first, between actives, or
    trailing the last active.
    """
    n_active = draw(st.integers(min_value=1, max_value=5))
    active_types = [
        draw(st.sampled_from(["revolute", "revolute", "revolute", "prismatic"]))
        for _ in range(n_active)
    ]
    active_joints = [draw(_joint(t)) for t in active_types]

    n_fixed = draw(st.integers(min_value=0, max_value=3))
    fixed_joints = [draw(_joint("fixed")) for _ in range(n_fixed)]

    # Shuffle via index slots: pick n_fixed insertion points (with repetition)
    # into the active sequence (length n_active + 1 boundary positions).
    chain: list[dict[str, Any]] = list(active_joints)
    for _ in range(n_fixed):
        pos = draw(st.integers(min_value=0, max_value=len(chain)))
        chain.insert(pos, fixed_joints.pop())
    return chain


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #

_SLOW = [HealthCheck.function_scoped_fixture, HealthCheck.too_slow]


@given(_chain())
@settings(max_examples=150, deadline=None, suppress_health_check=_SLOW)
def test_raw_and_normalized_fk_agree_at_random_q(
    tmp_path_factory: pytest.TempPathFactory, chain: list[dict[str, Any]]
) -> None:
    tmp_path = tmp_path_factory.mktemp("fuzz_fk")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(_urdf_text(chain))

    kb_raw = load_urdf_kinbody(urdf, "base_link", "ee_link")
    kb_norm = load_urdf_kinbody_normalized(urdf, "base_link", "ee_link")

    assert len(kb_raw.joints) == len(kb_norm.joints)
    n = len(kb_raw.joints)
    assume(n > 0)  # strategy guarantees ≥1 active, belt-and-suspenders

    # Use a deterministic seed so example shrinking keeps reproducing the same q.
    rng = np.random.default_rng(len(chain))
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, n)
        T_raw = _fk(kb_raw, q)
        T_norm = _fk(kb_norm, q)
        assert np.allclose(T_raw, T_norm, atol=1e-8), (
            f"FK divergence at q={q.tolist()}:\n"
            f"  max |Δ| = {np.max(np.abs(T_raw - T_norm))}\n"
            f"  chain    = {chain}"
        )


@given(_chain())
@settings(max_examples=150, deadline=None, suppress_health_check=_SLOW)
def test_normalized_t_left_is_pure_translation(
    tmp_path_factory: pytest.TempPathFactory, chain: list[dict[str, Any]]
) -> None:
    tmp_path = tmp_path_factory.mktemp("fuzz_tleft")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(_urdf_text(chain))

    kb = load_urdf_kinbody_normalized(urdf, "base_link", "ee_link")
    for j in kb.joints:
        assert np.allclose(j.T_left[:3, :3], np.eye(3), atol=1e-10), (
            f"joint {j.name}: T_left has rotation, POE invariant broken"
        )
        assert j.T_left[3, 3] == 1.0
        assert np.all(j.T_left[3, :3] == 0.0)


@given(_chain())
@settings(max_examples=150, deadline=None, suppress_health_check=_SLOW)
def test_normalized_t_right_identity_except_last(
    tmp_path_factory: pytest.TempPathFactory, chain: list[dict[str, Any]]
) -> None:
    tmp_path = tmp_path_factory.mktemp("fuzz_tright")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(_urdf_text(chain))

    kb = load_urdf_kinbody_normalized(urdf, "base_link", "ee_link")
    for j in kb.joints[:-1]:
        assert np.allclose(j.T_right, np.eye(4), atol=1e-10), (
            f"joint {j.name}: T_right != I in a non-final slot, POE invariant broken"
        )


@given(_chain())
@settings(max_examples=150, deadline=None, suppress_health_check=_SLOW)
def test_normalized_axes_are_unit_length(
    tmp_path_factory: pytest.TempPathFactory, chain: list[dict[str, Any]]
) -> None:
    tmp_path = tmp_path_factory.mktemp("fuzz_axes")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(_urdf_text(chain))

    kb = load_urdf_kinbody_normalized(urdf, "base_link", "ee_link")
    for j in kb.joints:
        norm = float(np.linalg.norm(j.axis))
        assert abs(norm - 1.0) < 1e-9, (
            f"joint {j.name}: ||axis|| = {norm} (expected 1.0) — "
            "POE normalization should preserve axis unit length since the "
            "input is already unit and r_cum is orthonormal."
        )


@given(_chain())
@settings(max_examples=150, deadline=None, suppress_health_check=_SLOW)
def test_fk_at_q_zero_matches_raw_loader(
    tmp_path_factory: pytest.TempPathFactory, chain: list[dict[str, Any]]
) -> None:
    """At ``q = 0`` every joint's motion is identity, so FK is determined
    entirely by the cumulative ``T_left @ T_right`` chain. This is a stronger
    check than the generic FK-agreement test because it isolates the home pose
    — any bug in the last joint's ``T_right`` (which absorbs cumulative
    rotation + trailing fixed-joint offset) shows up here immediately.
    """
    tmp_path = tmp_path_factory.mktemp("fuzz_home")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(_urdf_text(chain))

    kb_raw = load_urdf_kinbody(urdf, "base_link", "ee_link")
    kb_norm = load_urdf_kinbody_normalized(urdf, "base_link", "ee_link")

    q_zero = np.zeros(len(kb_raw.joints))
    assert np.allclose(_fk(kb_raw, q_zero), _fk(kb_norm, q_zero), atol=1e-10)


# --------------------------------------------------------------------------- #
# Regression cases — real-world rough edges we care about
# --------------------------------------------------------------------------- #


def _write_urdf_text(path: Path, body: str) -> Path:
    path.write_text(f'<?xml version="1.0"?><robot name="t">{body}</robot>')
    return path


def test_axis_near_unit_length_accepted(tmp_path: Path) -> None:
    """Xacro evaluated expressions often produce axes like (0, 0, 0.99999998)
    — not exactly unit but well within tolerance. The normalizer must still
    produce a unit-length axis in the output.
    """
    urdf = _write_urdf_text(
        tmp_path / "robot.urdf",
        """
        <link name="base"/><link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/><child link="tip"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <axis xyz="0 0 0.99999998"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody_normalized(urdf, "base", "tip")
    assert abs(float(np.linalg.norm(kb.joints[0].axis)) - 1.0) < 1e-9


def test_fixed_joint_rotation_fuses_into_subsequent_axis(tmp_path: Path) -> None:
    """Explicit regression: a Rz(π/2) fixed joint in the middle should rotate
    the next active joint's axis, not inject a rotation into its T_left.
    """
    urdf = _write_urdf_text(
        tmp_path / "robot.urdf",
        """
        <link name="base"/>
        <link name="mid"/>
        <link name="post"/>
        <link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/><child link="mid"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <axis xyz="1 0 0"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        <joint name="rot_fixed" type="fixed">
          <parent link="mid"/><child link="post"/>
          <origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/>
        </joint>
        <joint name="j1" type="revolute">
          <parent link="post"/><child link="tip"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <axis xyz="1 0 0"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody_normalized(urdf, "base", "tip")
    assert len(kb.joints) == 2
    # Joint 0's axis is unchanged (+x).
    assert np.allclose(kb.joints[0].axis, [1, 0, 0], atol=1e-10)
    # Joint 1's axis: original +x rotated by Rz(π/2) becomes +y.
    assert np.allclose(kb.joints[1].axis, [0, 1, 0], atol=1e-10)
    # T_left for both joints is still pure translation.
    for j in kb.joints:
        assert np.allclose(j.T_left[:3, :3], np.eye(3), atol=1e-10)


def test_prismatic_with_rotated_parent_origin(tmp_path: Path) -> None:
    """Prismatic joint whose parent origin has a non-identity rotation —
    the prismatic axis should be rotated into the base frame in the
    normalized output.
    """
    urdf = _write_urdf_text(
        tmp_path / "robot.urdf",
        """
        <link name="base"/><link name="tip"/>
        <joint name="j0" type="prismatic">
          <parent link="base"/><child link="tip"/>
          <origin xyz="0 0 0" rpy="0 1.5707963267948966 0"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="0" upper="0.5"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody_normalized(urdf, "base", "tip")
    # Axis is (0, 0, 1) in joint frame; after Ry(π/2) it becomes +x in base.
    assert np.allclose(kb.joints[0].axis, [1, 0, 0], atol=1e-10)
    # T_left is pure translation.
    assert np.allclose(kb.joints[0].T_left[:3, :3], np.eye(3), atol=1e-10)
