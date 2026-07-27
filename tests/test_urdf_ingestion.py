"""URDF ingestion robustness: mesh-free loading of vendor URDFs that declare
``xmlns:xacro`` but reference meshes by ROS ``package://`` (ABB YuMi, FANUC).

Regression for the Wave 1 onboarding friction (#425): such files went through
the xacro processor (because ``_is_xacro`` matched the namespace) and failed
with ``PackageNotFoundError`` before any mesh-stripping. The fix strips meshes
as plain XML first when no real xacro expansion is needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ssik._urdf import (
    load_urdf_kinbody_normalized,
    needs_xacro_expansion,
    strip_urdf_to_fixture,
)

# A minimal 2-revolute chain that declares the xacro namespace and references
# meshes by an unresolvable ``package://`` -- exactly the YuMi/FANUC shape.
_PACKAGE_MESH_URDF = """<?xml version="1.0"?>
<robot name="probe" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base">
    <visual><geometry><mesh filename="package://probe_desc/meshes/base.stl"/></geometry></visual>
    <collision><geometry><mesh filename="package://probe_desc/meshes/base.stl"/></geometry></collision>
  </link>
  <link name="link1">
    <visual><geometry><mesh filename="package://probe_desc/meshes/l1.stl"/></geometry></visual>
  </link>
  <link name="link2">
    <visual><geometry><mesh filename="package://probe_desc/meshes/l2.stl"/></geometry></visual>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def test_namespace_only_urdf_does_not_need_expansion() -> None:
    """A URDF that only declares ``xmlns:xacro`` (no macros / substitutions)
    must NOT be routed through the xacro processor."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.urdf"
        src.write_text(_PACKAGE_MESH_URDF)
        assert needs_xacro_expansion(src) is False


def test_macro_urdf_needs_expansion() -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.urdf"
        src.write_text('<robot name="p"><xacro:property name="a" value="${1+1}"/></robot>')
        assert needs_xacro_expansion(src) is True


def test_package_mesh_urdf_strips_and_loads() -> None:
    """The end-to-end ingestion path: strip meshes as plain XML (dropping the
    unresolvable ``package://`` refs), then load the mesh-free chain -- no ROS
    workspace, no ``PackageNotFoundError``."""
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.urdf"
        src.write_text(_PACKAGE_MESH_URDF)
        dest = Path(d) / "probe_fixture.urdf"
        strip_urdf_to_fixture(src, dest)
        assert "package://" not in dest.read_text()
        kb = load_urdf_kinbody_normalized(dest, "base", "link2")
        assert len(kb.joints) == 2


# ---------------------------------------------------------------------------
# base/ee auto-detection (suggest_base_ee)
# ---------------------------------------------------------------------------


def _rev(name: str, parent: str, child: str) -> str:
    return (
        f'<joint name="{name}" type="revolute">'
        f'<parent link="{parent}"/><child link="{child}"/>'
        '<origin xyz="0.2 0 0"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
    )


# world -> base_link (fixed) -> l1 -> l2 -> wrist (3R), then a gripper with two
# revolute fingers branching off ``gripper_base`` past the wrist.
_ARM_WITH_GRIPPER = f"""<?xml version="1.0"?>
<robot name="arm">
  <link name="world"/><link name="base_link"/>
  <link name="l1"/><link name="l2"/><link name="wrist"/>
  <link name="gripper_base"/><link name="finger_left"/><link name="finger_right"/>
  <joint name="fixed_world" type="fixed">
    <parent link="world"/><child link="base_link"/><origin xyz="0 0 0.05"/></joint>
  {_rev("j1", "base_link", "l1")}
  {_rev("j2", "l1", "l2")}
  {_rev("j3", "l2", "wrist")}
  <joint name="wrist_to_gripper" type="fixed">
    <parent link="wrist"/><child link="gripper_base"/><origin xyz="0.05 0 0"/></joint>
  {_rev("fl", "gripper_base", "finger_left")}
  {_rev("fr", "gripper_base", "finger_right")}
</robot>
"""


def test_suggest_base_ee_skips_leading_fixed_and_gripper() -> None:
    """base folds past ``world->base_link``; ee backs off the gripper fingers
    to the wrist flange."""
    from ssik._urdf import suggest_base_ee

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "arm.urdf"
        src.write_text(_ARM_WITH_GRIPPER)
        base, ee, _ = suggest_base_ee(src)
        assert base == "base_link"
        assert ee == "wrist"
