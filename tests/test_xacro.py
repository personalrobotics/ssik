"""Xacro description support (#327): detection, expansion, substitution args.

Gated on the optional ``xacrodoc`` dependency (the ``xacro`` extra; also in the
dev group so CI runs these).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xacrodoc")
pytest.importorskip("urchin")

from ssik._urdf import (
    _is_xacro,
    load_urdf_kinbody_normalized,
    process_xacro,
)
from ssik.kinematics.poe_fk import poe_forward_kinematics

FIXTURES = Path(__file__).parent / "fixtures"
XACRO = FIXTURES / "toy2r.urdf.xacro"


def test_is_xacro_by_extension_and_content(tmp_path: Path) -> None:
    assert _is_xacro(XACRO)
    plain = tmp_path / "plain.urdf"
    plain.write_text("<robot name='r'><link name='base_link'/></robot>")
    assert not _is_xacro(plain)
    # a .urdf that is actually xacro (namespace) is detected by content
    sneaky = tmp_path / "sneaky.urdf"
    sneaky.write_text("<robot xmlns:xacro='http://www.ros.org/wiki/xacro' name='r'/>")
    assert _is_xacro(sneaky)


def test_process_xacro_expands_default_and_subargs() -> None:
    default = process_xacro(XACRO)
    assert "<xacro:" not in default  # macros expanded
    assert "xmlns:xacro" not in default  # namespace gone
    assert 'xyz="0.3 0 0"' in default  # default len2
    overridden = process_xacro(XACRO, {"len2": "0.55"})
    assert 'xyz="0.55 0 0"' in overridden


def test_load_xacro_builds_kinbody() -> None:
    kb = load_urdf_kinbody_normalized(XACRO, "base_link", "link2")
    assert len(kb.joints) == 2
    # link2 sits L2=0.3 along x and 0.2 up at q=0.
    T = poe_forward_kinematics(kb, np.zeros(2))
    assert T[:3, 3] == pytest.approx([0.3, 0.0, 0.2])


def test_load_xacro_honors_subargs() -> None:
    kb = load_urdf_kinbody_normalized(XACRO, "base_link", "link2", xacro_args={"len2": "0.55"})
    T = poe_forward_kinematics(kb, np.zeros(2))
    assert T[:3, 3] == pytest.approx([0.55, 0.0, 0.2])


def test_vendor_xacro_to_plain_urdf(tmp_path: Path) -> None:
    """The ``add-arm`` path -- expand xacro then strip -- yields a plain,
    FK-equivalent fixture (no xacro tags, no mesh/visual)."""
    from ssik._urdf import _as_plain_urdf, strip_urdf_to_fixture

    dest = tmp_path / "vendored.urdf"
    with _as_plain_urdf(XACRO) as plain:
        strip_urdf_to_fixture(plain, dest)
    text = dest.read_text()
    assert "<xacro:" not in text
    assert "xmlns:xacro" not in text
    assert "<visual" not in text
    assert "package://" not in text

    kb_vendored = load_urdf_kinbody_normalized(dest, "base_link", "link2")
    kb_src = load_urdf_kinbody_normalized(XACRO, "base_link", "link2")
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=len(kb_src.joints))
        drift = np.abs(poe_forward_kinematics(kb_vendored, q) - poe_forward_kinematics(kb_src, q))
        assert drift.max() < 1e-12, f"vendored xacro FK drift {drift.max():.2e}"
