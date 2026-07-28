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
    """Yield ``(arm_name, description_name, arm)`` for every arm that declares
    an ``upstream_description`` (its ``robot_descriptions`` module name).

    Reads the dedicated ``upstream_description`` field, NOT the human
    ``fixture_source`` prose (#444): a typo or annotation in the prose can no
    longer silently drop an arm from parity or crash the loader. ``None`` means
    "no programmatic URDF upstream" -- a deliberate, reviewed choice that this
    test skips visibly."""
    for name, arm in load_manifest().items():
        if arm.upstream_description is not None:
            yield name, arm.upstream_description, arm


_PARAM_SETS = list(_arms_with_upstream())

# Floor on parity coverage: if a manifest edit silently drops arms from parity
# (a broken migration, a bulk field removal), the count regresses below this and
# CI fails. Bump when the roster grows. As of the #444 migration: 26 arms carry
# an upstream_description.
_MIN_UPSTREAM_ARMS = 24


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
    # ``upstream_description`` is authoritative (not parsed from prose), so a
    # load failure here means the field names a wrong/absent robot_descriptions
    # module -- fail loudly with that diagnosis rather than a raw traceback (#444).
    try:
        urdf = load_robot_description(description)
    except Exception as err:  # surface any loader failure clearly
        pytest.fail(
            f"{arm_name}: upstream_description={description!r} did not load a URDF "
            f"({type(err).__name__}: {err}). Fix the manifest's upstream_description "
            f"(it must name a robot_descriptions module that exposes a URDF), or set "
            f"it to nothing if the arm has no programmatic URDF upstream."
        )
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
        # Compare in the arm's BASE frame, not the URDF root/world frame:
        # ssik.fk is base_link -> ee_link, so the upstream pose must be
        # taken relative to base_link too. For arms whose base_link is not
        # the URDF root (e.g. YuMi's ``yumi_body``, offset from ``world``),
        # ``get_transform(ee)`` alone is world->ee and drifts by the
        # world->base offset -- a false failure.
        T_urdf = np.asarray(urdf.get_transform(arm.ee_link, arm.base_link))

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


def test_upstream_parity_coverage_above_floor() -> None:
    """The number of arms with upstream-URDF parity coverage must not silently
    regress below ``_MIN_UPSTREAM_ARMS``. A per-arm count floor (not just
    "at least one") catches a broken migration or a bulk removal of
    ``upstream_description`` fields that would quietly gut parity coverage while
    leaving one arm to keep the suite green (#444)."""
    n = len(_PARAM_SETS)
    assert n >= _MIN_UPSTREAM_ARMS, (
        f"only {n} arms carry an upstream_description (floor {_MIN_UPSTREAM_ARMS}); "
        f"parity coverage regressed. Check for a broken manifest migration or "
        f"removed upstream_description fields."
    )
