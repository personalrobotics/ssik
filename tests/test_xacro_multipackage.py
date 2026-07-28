"""Multi-package ROS xacro resolution in the ingestion pipeline.

Industrial vendor descriptions (ABB, KUKA, FANUC via ros-industrial) split a
robot across sibling ROS packages: ``irb120_3_58.xacro`` in ``abb_irb120_support``
does ``<xacro:include filename="$(find abb_resources)/..."/>`` for shared
materials. xacrodoc can't resolve ``$(find abb_resources)`` unless told where the
sibling packages live, so ``process_xacro`` (and thus ``ssik add-arm`` /
``from_urdf``) failed on every multi-package vendor description with
``PackageNotFoundError``.

``process_xacro`` now auto-discovers the source's workspace root (the parent of
its own ROS package) and registers it with xacrodoc, so ``$(find <sibling>)``
resolves with no manual ``ROS_PACKAGE_PATH`` or ``--package-path``. This test
uses a self-contained two-package workspace (``robot_support`` includes
``$(find shared_resources)/common.xacro``) so it needs no external checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_WS = Path(__file__).parent / "fixtures" / "multipkg_ws"
_ARM_XACRO = _WS / "robot_support" / "urdf" / "arm.urdf.xacro"


def _xacrodoc_available() -> bool:
    try:
        import xacrodoc  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _xacrodoc_available(), reason="xacrodoc not installed")
def test_process_xacro_resolves_sibling_package() -> None:
    """``process_xacro`` auto-resolves ``$(find shared_resources)`` (a sibling
    package) without any manual package-path configuration. The arm's joint
    origin uses ``${link_len}`` defined in the sibling's common.xacro, so a
    failed include would either raise or leave the property undefined."""
    from ssik._urdf import process_xacro

    urdf = process_xacro(_ARM_XACRO)
    assert "<robot" in urdf
    # 0.5 is the link_len property from the sibling package's common.xacro,
    # substituted into the joint origin -> proves the include resolved.
    assert "0.5" in urdf


@pytest.mark.skipif(not _xacrodoc_available(), reason="xacrodoc not installed")
def test_discover_package_roots_finds_workspace() -> None:
    """The workspace root (parent of the source's own ROS package) is discovered
    so xacrodoc's ``look_in`` finds every sibling package under it."""
    from ssik._urdf import _discover_package_roots

    roots = _discover_package_roots(_ARM_XACRO)
    assert _WS.resolve() in [r.resolve() for r in roots]


@pytest.mark.skipif(not _xacrodoc_available(), reason="xacrodoc not installed")
def test_multipackage_xacro_loads_as_kinbody() -> None:
    """End-to-end: a multi-package xacro flows through the ingestion path
    (expand -> strip -> POE-normalize) to a usable KinBody."""
    import tempfile

    from ssik._urdf import _as_plain_urdf, load_urdf_kinbody_normalized, strip_urdf_to_fixture

    with _as_plain_urdf(_ARM_XACRO) as plain:
        stripped = Path(tempfile.mktemp(suffix=".urdf"))
        strip_urdf_to_fixture(plain, stripped)
    kb = load_urdf_kinbody_normalized(stripped, "base_link", "tool")
    assert len(kb.joints) == 1
    # link_len=0.5 from the sibling package landed in the joint offset.
    assert abs(float(kb.joints[0].T_left[2, 3]) - 0.5) < 1e-12
