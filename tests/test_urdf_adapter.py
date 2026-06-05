"""Tests for the URDF → KinBody adapter.

Unit tests use synthesized URDF strings to probe the adapter's handling of
fixed-joint fusion, mimic rejection, and joint-type mapping. One integration
test loads the real ``ur5.urdf`` fixture and cross-verifies FK against the
hand-built DH fixture in ``tests/fixtures/ur5.py``.

The URDF-adapter tests are fast — they do not invoke ``generateIkSolver``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody

FIXTURES = Path(__file__).parent / "fixtures"


def _write_urdf(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "robot.urdf"
    path.write_text(f'<?xml version="1.0"?><robot name="t">{body}</robot>')
    return path


def test_minimal_single_joint(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/>
          <child link="tip"/>
          <origin xyz="0 0 0.1" rpy="0 0 0"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody(urdf, "base", "tip")
    assert [link.name for link in kb.links] == ["base", "tip"]
    assert len(kb.joints) == 1
    j = kb.joints[0]
    assert j.joint_type == "revolute"
    assert j.dof_index == 0
    assert np.allclose(j.T_left[:3, 3], [0, 0, 0.1])
    assert np.allclose(j.axis, [0, 0, 1])


def test_continuous_maps_to_revolute(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="tip"/>
        <joint name="j0" type="continuous">
          <parent link="base"/>
          <child link="tip"/>
          <axis xyz="1 0 0"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody(urdf, "base", "tip")
    assert kb.joints[0].joint_type == "revolute"


def test_prismatic(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="tip"/>
        <joint name="j0" type="prismatic">
          <parent link="base"/>
          <child link="tip"/>
          <axis xyz="0 1 0"/>
          <limit effort="1" velocity="1" lower="0" upper="0.5"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody(urdf, "base", "tip")
    assert kb.joints[0].joint_type == "prismatic"


def test_fixed_joint_fuses_into_next_active(tmp_path: Path) -> None:
    """Fixed joint between base and active joint — its origin should be
    baked into the active joint's T_left."""
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="mount"/>
        <link name="tip"/>
        <joint name="mount_fixed" type="fixed">
          <parent link="base"/>
          <child link="mount"/>
          <origin xyz="0.5 0 0" rpy="0 0 0"/>
        </joint>
        <joint name="j0" type="revolute">
          <parent link="mount"/>
          <child link="tip"/>
          <origin xyz="0 0 0.1" rpy="0 0 0"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody(urdf, "base", "tip")
    # Chain is [base → tip] active-wise — the "mount" link is fused.
    assert [link.name for link in kb.links] == ["base", "tip"]
    j = kb.joints[0]
    # T_left = mount_fixed.origin @ j0.origin = Trans(0.5,0,0) @ Trans(0,0,0.1)
    assert np.allclose(j.T_left[:3, 3], [0.5, 0, 0.1])


def test_trailing_fixed_fuses_into_previous_T_right(tmp_path: Path) -> None:
    """Fixed joint after the last active joint should bake into that
    joint's T_right."""
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="pre_ee"/>
        <link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/>
          <child link="pre_ee"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        <joint name="tail_fixed" type="fixed">
          <parent link="pre_ee"/>
          <child link="tip"/>
          <origin xyz="0 0 0.25" rpy="0 0 0"/>
        </joint>
        """,
    )
    kb = load_urdf_kinbody(urdf, "base", "tip")
    assert len(kb.joints) == 1
    j = kb.joints[0]
    assert np.allclose(j.T_right[:3, 3], [0, 0, 0.25])


def test_mimic_joint_rejected(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="mid"/>
        <link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/>
          <child link="mid"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        <joint name="j1" type="revolute">
          <parent link="mid"/>
          <child link="tip"/>
          <axis xyz="0 0 1"/>
          <mimic joint="j0" multiplier="0.5"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    with pytest.raises(NotImplementedError, match="mimic"):
        load_urdf_kinbody(urdf, "base", "tip")


def test_planar_joint_rejected(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="tip"/>
        <joint name="j0" type="planar">
          <parent link="base"/>
          <child link="tip"/>
          <axis xyz="0 0 1"/>
        </joint>
        """,
    )
    with pytest.raises(NotImplementedError, match="planar"):
        load_urdf_kinbody(urdf, "base", "tip")


def test_missing_link_raises(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="tip"/>
        <joint name="j0" type="revolute">
          <parent link="base"/>
          <child link="tip"/>
          <axis xyz="0 0 1"/>
          <limit effort="1" velocity="1" lower="-1" upper="1"/>
        </joint>
        """,
    )
    with pytest.raises(ValueError, match="no link named"):
        load_urdf_kinbody(urdf, "base", "nope")


def test_all_fixed_chain_raises(tmp_path: Path) -> None:
    urdf = _write_urdf(
        tmp_path,
        """
        <link name="base"/>
        <link name="tip"/>
        <joint name="j0" type="fixed">
          <parent link="base"/>
          <child link="tip"/>
        </joint>
        """,
    )
    with pytest.raises(ValueError, match="no active joints"):
        load_urdf_kinbody(urdf, "base", "tip")


def test_ur5_urdf_loads_as_six_joint_chain() -> None:
    """Smoke check that the UR5 URDF adapter produces a 6-joint POE chain.

    The original incarnation of this test compared adapter output against
    a hand-built classical-DH UR5 in ``tests/fixtures/ur5.py``. Per #311,
    ``tests/fixtures/ur5.urdf`` was replaced with the manufacturer URDF
    (with the real ~135 mm physical shoulder offset), so a comparison
    against textbook DH no longer makes sense -- the two describe
    different physical chains. The end-to-end kinematic-correctness
    claim is now locked in by
    :mod:`tests.test_prebuilt_fixture_parity`, which asserts
    ``module.fk(q) == upstream.fk(q)`` at machine precision against
    ``robot_descriptions / ur5_description``.
    """
    kb = load_urdf_kinbody(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    assert len(kb.joints) == 6
    assert kb.GetDOF() == 6
