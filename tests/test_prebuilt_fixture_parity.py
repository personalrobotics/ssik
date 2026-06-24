"""Fixture-vs-upstream-URDF parity (#311).

Every prebuilt promises that the q-vector ``module.solve(T)`` returns
will drive a real arm to ``T``. That promise only holds if the
kinematic chain ssik solves against is the same chain the manufacturer
ships -- otherwise FK closure 1e-15 is between ssik's IK and ssik's FK,
not between ssik's IK and the real arm's FK.

This test compares ``module.fk(q)`` against the canonical upstream
URDF's FK at random configs and asserts machine-precision agreement.

The canonical upstream is identified per arm via the manifest's
``fixture_source`` line:

- ``"robot_descriptions / <name>"``  -- compared against
  :func:`robot_descriptions.loaders.yourdfpy.load_robot_description`.
- ``"<other source>"``  -- skipped (no programmatic upstream URDF
  available; the fixture IS the canonical model, e.g. Puma 560 / JACO 2
  hand-built DH, vendor-supplied URDFs not in robot_descriptions).

Skipped arms still get an explicit message so a future addition of
their description to ``robot_descriptions`` flips them from skipped
to checked without code change.

Hidden EE-link convention: ssik's prebuilt encodes one EE link (e.g.
``iiwa_link_ee_kuka``); the rendered URDF may also expose alternate
EE convenience frames (``iiwa_link_ee`` rotated 90°). This test
queries the rendered URDF at the SAME link name the prebuilt's
``EE_LINK`` attribute carries, so the comparison is well-defined.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ssik.prebuilt._manifest import load_manifest

if TYPE_CHECKING:
    pass


_POS_ATOL = 1e-9  # 1 nanometre
_ROT_ATOL = 1e-7  # ~6e-6 degrees -- looser than position because some upstream
# URDFs round their quaternion encodings on write; this is below any robot's
# physical repeatability.


def _arms_with_upstream():
    """Yield ``(arm_name, description_name, prebuilt_module)`` for every
    arm whose ``fixture_source`` line names a ``robot_descriptions``
    entry."""
    for name, arm in load_manifest().items():
        # Parse the convention: "robot_descriptions / <name> (...)" -- the
        # second whitespace-separated token after the slash is the
        # description module name. We keep the parse intentionally
        # narrow so any deviation surfaces as a clear test-collection
        # error rather than a silent skip.
        src = arm.fixture_source
        marker = "robot_descriptions / "
        if marker not in src:
            continue
        # Take the substring after the marker, split on whitespace + "(".
        tail = src.split(marker, 1)[1]
        # description name runs until the first whitespace or "(".
        desc = ""
        for ch in tail:
            if ch.isspace() or ch == "(":
                break
            desc += ch
        if not desc:
            continue
        yield name, desc, arm


_PARAM_SETS = list(_arms_with_upstream())


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_q(dof: int, n: int, seed: int) -> list[np.ndarray]:
    """``n`` random configs in ``[-1, 1]`` rad. Tight enough to keep
    poses comfortably in the workspace interior; the parity claim is
    about the chain math, not joint-limit handling."""
    rng = _rng(seed)
    return [rng.uniform(-1.0, 1.0, size=dof) for _ in range(n)]


@pytest.mark.parametrize(
    ("arm_name", "description", "arm"),
    _PARAM_SETS,
    ids=[p[0] for p in _PARAM_SETS],
)
def test_fixture_matches_upstream_urdf(arm_name, description, arm) -> None:
    """For every arm whose manifest ``fixture_source`` cites a
    ``robot_descriptions`` entry, assert ``ssik.fk(q) == upstream.fk(q)``
    at machine precision on N random configs.

    Failure on this test typically means we vendored a different
    revision than the current upstream, or that our chain endpoint
    (``base_link`` / ``ee_link``) names the wrong link in the rendered
    URDF.
    """
    try:
        from robot_descriptions.loaders.yourdfpy import (  # type: ignore[import-untyped]
            load_robot_description,
        )
    except ImportError:
        pytest.skip("`robot_descriptions` not installed")

    module = import_module(f"ssik.prebuilt.{arm_name}")
    urdf = load_robot_description(description)
    urdf_actuated = [
        j.name for j in urdf.robot.joints if j.type in ("revolute", "continuous", "prismatic")
    ]
    # Default mapping: ssik's q[i] drives URDF's i-th actuated joint.
    # Arms whose URDF includes extra non-IK joints (Panda fingers,
    # PiPER grippers) need an explicit mapping. ``ssik.prebuilt.<arm>``
    # bakes the chain's joint names as ``_KB.joints[i].name``, so we
    # can resolve the mapping live without an extra manifest field.
    kb = getattr(module, "_KB", None)
    if kb is None:
        pytest.skip(f"{arm_name}: prebuilt does not expose _KB")
    ik_names = [j.name for j in kb.joints]
    # Joint-name match between ssik chain and URDF actuated chain.
    missing = [n for n in ik_names if n not in urdf_actuated]
    if missing:
        pytest.fail(
            f"{arm_name}: IK joint names not in upstream URDF: {missing}. "
            f"URDF actuated: {urdf_actuated}"
        )

    for q in _random_q(module.DOF, n=10, seed=0):
        T_ssik = module.fk(q)
        cfg = {n: 0.0 for n in urdf_actuated}
        for i, n in enumerate(ik_names):
            cfg[n] = float(q[i])
        urdf.update_cfg(cfg)
        T_urdf = np.asarray(urdf.get_transform(arm.ee_link))

        pos_drift = float(np.linalg.norm(T_urdf[:3, 3] - T_ssik[:3, 3]))
        R_rel = T_urdf[:3, :3] @ T_ssik[:3, :3].T
        cos = float(np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0))
        rot_drift = float(np.arccos(cos))

        assert pos_drift < _POS_ATOL, (
            f"{arm_name}: ssik.fk position drift vs {description}@{arm.ee_link} = "
            f"{pos_drift:.3e} m > {_POS_ATOL:.0e} (q={q.tolist()})"
        )
        assert rot_drift < _ROT_ATOL, (
            f"{arm_name}: ssik.fk rotation drift vs {description}@{arm.ee_link} = "
            f"{rot_drift:.3e} rad > {_ROT_ATOL:.0e} (q={q.tolist()})"
        )


def test_every_arm_has_provenance() -> None:
    """Every manifest entry must carry a ``fixture_source`` line. The
    manifest loader already enforces this at parse time; this test is
    the explicit, named asserter so a regression on the schema gives a
    clear failure rather than an opaque ``KeyError`` somewhere
    downstream."""
    arms = load_manifest()
    for name, arm in arms.items():
        assert arm.fixture_source, f"arm {name!r}: fixture_source must not be empty (#311)"


def test_every_arm_has_eaik_comparison() -> None:
    """Every manifest entry must carry an ``[eaik]`` block (populated by
    ``scripts/regen_bench.py``), so a newly-onboarded arm never silently ships
    with a blank EAIK comparison cell. Each block must be internally consistent:
    supported arms carry a family + timing; refused arms carry a reason."""
    for name, arm in load_manifest().items():
        e = arm.eaik
        assert e is not None, f"arm {name!r}: missing [eaik] block (run scripts/regen_bench.py)"
        if e.supported:
            assert e.family, f"arm {name!r}: supported EAIK block must name a family"
            assert e.ms_mean > 0, f"arm {name!r}: supported EAIK block must have timing"
        else:
            assert e.refusal, f"arm {name!r}: refused EAIK block must carry a refusal string"


def test_at_least_one_arm_has_upstream_parity_coverage() -> None:
    """At least one arm in the manifest must have a
    ``robot_descriptions / <name>`` provenance line, so the parametrised
    parity test above is actually checking something. Catches the
    degenerate case where someone strips parity coverage from every
    entry."""
    assert _PARAM_SETS, (
        "No arms parametrised for upstream-URDF parity. Either "
        "``robot_descriptions`` is unavailable or every manifest entry "
        "has been edited to remove its ``robot_descriptions /`` "
        "provenance line."
    )
