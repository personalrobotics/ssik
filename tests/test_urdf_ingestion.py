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
