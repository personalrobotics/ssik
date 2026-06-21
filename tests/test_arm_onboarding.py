"""Arm-onboarding tooling (#341): URDF stripping + one-click bench regen.

Guards the friction-reduction helpers: ``ssik._urdf.strip_urdf_to_fixture``
(shared by ``ssik add-arm`` and ``scripts/strip_urdf_fixture.py``) and the
surgical ``[bench]`` rewrite in ``scripts/regen_bench.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody_normalized, strip_urdf_to_fixture
from ssik.kinematics.poe_fk import poe_forward_kinematics

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"

_URDF_WITH_GEOMETRY = """<?xml version="1.0"?>
<robot name="toy">
  <material name="red"><color rgba="1 0 0 1"/></material>
  <link name="base">
    <visual><geometry><mesh filename="package://toy/base.stl"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
    <inertial><mass value="1"/></inertial>
  </link>
  <link name="l1"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <gazebo reference="base"/>
</robot>
"""


def test_strip_drops_nonkinematic_keeps_joints(tmp_path: Path) -> None:
    src = tmp_path / "toy.urdf"
    src.write_text(_URDF_WITH_GEOMETRY)
    dest = tmp_path / "toy_stripped.urdf"
    n_links, n_joints = strip_urdf_to_fixture(src, dest)
    assert (n_links, n_joints) == (2, 1)
    text = dest.read_text()
    for dropped in ("<visual", "<collision", "<inertial", "<material", "<gazebo", "package://"):
        assert dropped not in text, f"{dropped} survived strip"
    # Joint kinematics are kept.
    assert "<joint" in text
    assert 'xyz="0 0 0.3"' in text
    assert "<axis" in text


def test_strip_preserves_fk(tmp_path: Path) -> None:
    """Stripping a real fixture changes no kinematics -- FK is identical."""
    src = FIXTURES / "rizon4.urdf"
    dest = tmp_path / "rizon4_stripped.urdf"
    strip_urdf_to_fixture(src, dest)
    kb_src = load_urdf_kinbody_normalized(src, "base_link", "flange")
    kb_dst = load_urdf_kinbody_normalized(dest, "base_link", "flange")
    rng = np.random.default_rng(0)
    for _ in range(8):
        q = rng.uniform(-1.5, 1.5, size=len(kb_src.joints))
        drift = np.abs(poe_forward_kinematics(kb_dst, q) - poe_forward_kinematics(kb_src, q)).max()
        assert drift < 1e-12, f"FK drift {drift:.2e}"


def test_strip_raises_on_broken_joint(tmp_path: Path) -> None:
    bad = tmp_path / "bad.urdf"
    bad.write_text('<robot name="x"><joint name="j" type="revolute"/></robot>')
    with pytest.raises(ValueError, match="parent/child"):
        strip_urdf_to_fixture(bad, tmp_path / "out.urdf")


def _load_regen_bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "regen_bench", REPO / "scripts" / "regen_bench.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_regen_bench_update_is_surgical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``update_manifest_bench`` rewrites only the five bench values, preserving
    every comment and hand-curated field."""
    rb = _load_regen_bench()
    manifest = tmp_path / "M.toml"
    manifest.write_text(
        "# top comment\n"
        "[arms.foo_ik]\n"
        'display_name = "Foo"  # curated\n'
        "dof = 6\n"
        "\n"
        "[arms.foo_ik.bench]\n"
        "ms_mean = 1.0\n"
        "ms_ci95 = 0.1\n"
        "max_fk = 1.0e-9\n"
        "sols_min = 4\n"
        "sols_max = 8\n"
    )
    monkeypatch.setattr(rb, "MANIFEST", manifest)
    rb.update_manifest_bench(
        "foo_ik",
        {"ms_mean": 2.5, "ms_ci95": 0.2, "max_fk": 3e-6, "sols_min": 2, "sols_max": 10},
    )
    out = manifest.read_text()
    # Curated content untouched.
    assert "# top comment" in out
    assert 'display_name = "Foo"  # curated' in out
    assert "dof = 6" in out
    # Bench values rewritten.
    assert "ms_mean = 2.50" in out
    assert "max_fk = 3.0e-06" in out
    assert "sols_max = 10" in out
    assert "ms_mean = 1.0\n" not in out
