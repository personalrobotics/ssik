"""Format-adapter registry (#470): detection + the two built-in adapters.

The registry (:mod:`ssik._formats`) is what makes ingestion pluggable -- a new
source format (URDF, MJCF, future USD/SDF) is one ``register(FormatAdapter(...))``
call, and every format-agnostic consumer (``ssik add-arm``, ``regen_artifacts``,
the manifest validator) resolves through it. These tests pin the contract those
consumers rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssik import _formats as formats


def test_builtin_kinds_registered() -> None:
    assert set(formats.kinds()) >= {"urdf", "mjcf"}


def test_get_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown format kind"):
        formats.get("usd")


def test_detect_by_extension() -> None:
    assert formats.detect("arm.urdf").kind == "urdf"
    assert formats.detect("arm.xacro").kind == "urdf"
    assert formats.detect("arm.mjcf").kind == "mjcf"


def test_detect_unknown_extension_raises() -> None:
    with pytest.raises(ValueError, match="no registered format"):
        formats.detect("arm.step")


def test_detect_xml_disambiguated_by_root_tag(tmp_path: Path) -> None:
    """``.xml`` is claimed by both URDF and MJCF; the XML root element decides."""
    robot = tmp_path / "r.xml"
    robot.write_text('<robot name="x"><link name="a"/></robot>')
    mujoco = tmp_path / "m.xml"
    mujoco.write_text('<mujoco model="x"><worldbody/></mujoco>')
    assert formats.detect(robot).kind == "urdf"
    assert formats.detect(mujoco).kind == "mjcf"


def test_adapter_fields_present() -> None:
    for kind in ("urdf", "mjcf"):
        a = formats.get(kind)
        assert a.kind == kind
        assert a.fixture_suffix.startswith(".")
        assert a.scaffold_loader_module
        assert a.scaffold_loader_func
        assert callable(a.load)
        assert callable(a.vendor)
        assert callable(a.suggest_base_ee)
