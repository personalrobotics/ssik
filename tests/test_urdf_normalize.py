"""Tests for the POE-normalized URDF adapter.

The normalization rewrites a URDF-loaded chain so every joint's ``T_left``
is a pure translation and every joint's ``axis`` is in the base frame at
q=0. The final joint's ``T_right`` absorbs the cumulative home-pose
rotation so FK at q=0 matches the URDF exactly.

What we verify here:

1. **FK equivalence at many q** — the normalized and native-adapter chains
   are kinematically identical. This is the central POE claim.
2. **Structure is encoded correctly** — ``T_left`` is pure translation for
   every joint; joint axes are in base frame (match what ``urchin.link_fk``
   would compute for each joint's axis direction at q=0).
3. **Trailing fixed joints and intermediate fixed joints** — same
   FK as the native adapter for chains that include fixed segments.
4. **Mimic / planar / floating** rejections propagate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody, load_urdf_kinbody_normalized

FIXTURES = Path(__file__).parent / "fixtures"


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation as 4x4, matching the per-joint POE FK convention
    for a revolute joint."""
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return np.eye(4)
    k = axis / norm
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return T


def _fk(kinbody: Any, q: np.ndarray) -> np.ndarray:
    """Compose the chain ``T_left @ R(axis, qᵢ) @ T_right`` per joint."""
    T = np.eye(4)
    for j, qi in zip(kinbody.joints, q, strict=True):
        T = T @ j.T_left @ _axis_angle(j.axis, qi) @ j.T_right
    return T


def _write_urdf(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "robot.urdf"
    path.write_text(f'<?xml version="1.0"?><robot name="t">{body}</robot>')
    return path


# ---------------------------------------------------------------------------
# Core FK-equivalence on the UR5 fixture (our DH-style URDF)
# ---------------------------------------------------------------------------


def test_fk_equivalence_on_ur5() -> None:
    """Normalized and native chains give identical FK at 20 random q."""
    kb_orig = load_urdf_kinbody(FIXTURES / "ur5.urdf", "base_link", "wrist_3_link")
    kb_norm = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "wrist_3_link")

    assert len(kb_orig.joints) == len(kb_norm.joints) == 6

    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, 6)
        assert np.allclose(_fk(kb_orig, q), _fk(kb_norm, q), atol=1e-9), (
            f"FK divergence at q={q.tolist()}"
        )


def test_ur5_normalized_exposes_parallel_axes() -> None:
    """UR5's three inner parallel axes (joints 1, 2, 3 and also joint 5) should
    all be co-aligned along the base-frame y-axis after normalization. That's
    the structure :func:`ssik.kinematics.predicates.three_consecutive_parallel`
    needs to see for a Pieper-class solve.

    Sign convention follows the manufacturer URDF (per #311 the fixture is now
    ``robot_descriptions / ur5_description``, not the v1.2 textbook-DH model
    that had axes flipped to ``(0, -1, 0)``).
    """
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "wrist_3_link")

    # Joints 1, 2, 3, 5 should have axis ≈ (0, +1, 0).
    for idx in (1, 2, 3, 5):
        ax = kb.joints[idx].axis
        assert np.allclose(ax, [0.0, 1.0, 0.0], atol=1e-10), (
            f"joint {idx} axis in base frame = {ax.tolist()}, expected ≈ (0, 1, 0)"
        )
    # Joint 0 should be pure +z, joint 4 should be pure -z.
    assert np.allclose(kb.joints[0].axis, [0.0, 0.0, 1.0], atol=1e-10)
    assert np.allclose(kb.joints[4].axis, [0.0, 0.0, -1.0], atol=1e-10)


def test_ur5_normalized_t_left_is_pure_translation() -> None:
    """Every ``T_left`` must be a pure translation in POE normalization."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "wrist_3_link")
    for j in kb.joints:
        assert np.allclose(j.T_left[:3, :3], np.eye(3), atol=1e-10), (
            f"joint {j.name}: T_left has rotation, POE invariant broken"
        )


def test_ur5_normalized_t_right_identity_except_last() -> None:
    """Only the last joint's ``T_right`` carries the home-pose rotation."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "wrist_3_link")
    for j in kb.joints[:-1]:
        assert np.allclose(j.T_right, np.eye(4), atol=1e-10), (
            f"joint {j.name}: T_right != I, POE invariant broken"
        )


# ---------------------------------------------------------------------------
# Synthesized URDFs — fixed-joint handling, type mapping, rejections
# ---------------------------------------------------------------------------


def test_fixed_joint_between_actives(tmp_path: Path) -> None:
    """Fixed joint with rotation in its origin, between two active joints —
    the rotation contributes to cumulative ``r_cum`` and shows up in the
    second active joint's base-frame axis, not as a rotation in T_left."""
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="mid"/>
        <link name="pre_tip"/>
        <link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/>
          <child link="mid"/>
          <origin xyz="0 0 0.1" rpy="0 0 0"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        <joint name="rot_fixed" type="fixed">
          <parent link="mid"/>
          <child link="pre_tip"/>
          <origin xyz="0 0 0" rpy="1.5707963267948966 0 0"/>
        </joint>
        <joint name="j1" type="revolute">
          <parent link="pre_tip"/>
          <child link="tip"/>
          <origin xyz="0 0 0.2" rpy="0 0 0"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    kb_orig = load_urdf_kinbody(urdf, "base", "tip")
    kb_norm = load_urdf_kinbody_normalized(urdf, "base", "tip")

    # Both should have 2 active joints.
    assert len(kb_orig.joints) == len(kb_norm.joints) == 2

    # FK agreement at random q.
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.uniform(-np.pi, np.pi, 2)
        assert np.allclose(_fk(kb_orig, q), _fk(kb_norm, q), atol=1e-9)

    # Normalization: j1's axis in base frame should reflect the intermediate
    # Rx(π/2) rotation — original (0,0,1) becomes (0,-1,~0).
    assert np.allclose(kb_norm.joints[0].axis, [0, 0, 1], atol=1e-10)
    assert np.allclose(kb_norm.joints[1].axis, [0, -1, 0], atol=1e-9)


def test_trailing_fixed_rolled_into_t_right(tmp_path: Path) -> None:
    """A trailing fixed joint after the last active joint — its offset and
    any rotation must end up in the last joint's ``T_right``."""
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="hinge"/>
        <link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/>
          <child link="hinge"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        <joint name="tail" type="fixed">
          <parent link="hinge"/>
          <child link="tip"/>
          <origin xyz="0 0 0.5" rpy="0 1.5707963267948966 0"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody_normalized(urdf, "base", "tip")
    assert len(kb.joints) == 1
    # T_right should encode (0, 0, 0.5) translation and Ry(π/2) rotation.
    T_right = kb.joints[0].T_right
    assert np.allclose(T_right[:3, 3], [0, 0, 0.5], atol=1e-10)
    expected_r = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)  # Ry(π/2)
    assert np.allclose(T_right[:3, :3], expected_r, atol=1e-10)


def test_continuous_maps_to_revolute(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/><link name="tip"/>
        <joint name="j0" type="continuous">
          <parent link="base"/><child link="tip"/>
          <axis xyz="0 0 1"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody_normalized(urdf, "base", "tip")
    assert kb.joints[0].joint_type == "revolute"


def test_prismatic(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/><link name="tip"/>
        <joint name="j0" type="prismatic">
          <parent link="base"/><child link="tip"/>
          <origin xyz="0 0 0.1" rpy="1.5707963267948966 0 0"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="0" upper="0.5"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody_normalized(urdf, "base", "tip")
    assert kb.joints[0].joint_type == "prismatic"
    # Axis should be rotated by Rx(π/2): (0,0,1) → (0,-1,0).
    assert np.allclose(kb.joints[0].axis, [0, -1, 0], atol=1e-10)


def test_mimic_rejected(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/><link name="mid"/><link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/><child link="mid"/><axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        <joint name="j1" type="revolute">
          <parent link="mid"/><child link="tip"/><axis xyz="0 0 1"/>
          <mimic joint="j0" multiplier="0.5"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    with pytest.raises(NotImplementedError, match="mimic"):
        load_urdf_kinbody_normalized(urdf, "base", "tip")


def test_planar_rejected(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/><link name="tip"/>
        <joint name="j0" type="planar">
          <parent link="base"/><child link="tip"/><axis xyz="0 0 1"/>
        </joint>
        """,
    )
    with pytest.raises(NotImplementedError, match="planar"):
        load_urdf_kinbody_normalized(urdf, "base", "tip")


def test_all_fixed_rejected(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/><link name="tip"/>
        <joint name="j0" type="fixed">
          <parent link="base"/><child link="tip"/>
        </joint>
        """,
    )
    with pytest.raises(ValueError, match="no active joints"):
        load_urdf_kinbody_normalized(urdf, "base", "tip")


def test_missing_link_rejected(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/><link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/><child link="tip"/><axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    with pytest.raises(ValueError, match="no link named"):
        load_urdf_kinbody_normalized(urdf, "base", "nope")
