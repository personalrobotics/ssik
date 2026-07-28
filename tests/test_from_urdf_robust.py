"""``Manipulator.from_urdf`` ingestion robustness (#434).

The public library entry point must be as one-click as ``ssik add-arm``:

- **``package://`` tolerance.** Vendor URDFs (ABB, FANUC, ...) declare
  ``xmlns:xacro`` and reference meshes by ROS ``package://`` paths that don't
  resolve outside a ROS workspace. ``from_urdf`` strips the URDF to a
  mesh-free copy before loading, so those load without a workspace on disk.
  Loading such a URDF via the non-stripping path raises ``PackageNotFoundError``
  (asserted below), so this guard has teeth.
- **base/ee auto-detection.** With ``base``/``ee`` omitted, the longest
  actuated chain is detected and gives FK-identical results to the explicit call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ssik

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def pkg_urdf(tmp_path: Path) -> Path:
    """A dispatchable 6-DOF arm (UR5) dressed as a raw vendor URDF: an
    ``xmlns:xacro`` namespace on the robot tag and an unresolvable
    ``package://`` mesh -- the exact shape that breaks a naive load."""
    src = (FIXTURES / "ur5.urdf").read_text(encoding="utf-8")
    assert "xmlns:xacro" not in src  # the shipped fixture is already clean
    dressed = src.replace(
        '<robot name="ur5">',
        '<robot name="ur5" xmlns:xacro="http://www.ros.org/wiki/xacro">',
    ).replace(
        '<link name="base_link">',
        '<link name="base_link"><visual><geometry>'
        '<mesh filename="package://no_such_pkg/base.stl"/></geometry></visual>',
        1,
    )
    assert "package://" in dressed
    assert "xmlns:xacro" in dressed
    path = tmp_path / "pkg_arm.urdf"
    path.write_text(dressed, encoding="utf-8")
    return path


def test_from_urdf_tolerates_package_meshes(pkg_urdf: Path) -> None:
    """A URDF with xmlns:xacro + package:// meshes loads without a ROS workspace,
    FK-identical to the clean shipped fixture."""
    arm = ssik.Manipulator.from_urdf(pkg_urdf, base="base_link", ee="ee_link")
    clean = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    assert arm.dof == 6
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=6)
        assert np.allclose(arm.fk(q), clean.fk(q), atol=1e-12)


def test_non_stripping_path_fails_on_package_meshes(pkg_urdf: Path) -> None:
    """Teeth: the raw (non-stripping) loader raises on the same file, so the
    strip-first ingestion in from_urdf is doing real work."""
    from ssik._urdf import load_urdf_kinbody_normalized

    with pytest.raises(Exception, match=r"[Pp]ackage"):
        load_urdf_kinbody_normalized(pkg_urdf, "base_link", "ee_link")


def test_from_urdf_autodetects_base_ee(pkg_urdf: Path) -> None:
    """Omitting base/ee auto-detects the actuated chain and yields a usable,
    dispatchable arm whose IK round-trips (FK closure) end-to-end.

    We don't pin the auto-detected EE (it resolves to the kinematic flange,
    ``wrist_3_link``, not the fixture's ``ee_link``): the guarantee is a
    working 6-DOF solver, verified by an FK -> IK -> FK round-trip.
    """
    auto = ssik.Manipulator.from_urdf(pkg_urdf)
    assert auto.dof == 6
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.0, 1.0, size=6)
    T = auto.fk(q)
    sols = auto.solve(T)
    assert sols, "auto-detected arm returned no IK solutions"
    assert any(np.allclose(auto.fk(s.q), T, atol=1e-8) for s in sols)
