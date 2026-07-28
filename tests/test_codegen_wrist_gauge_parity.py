"""Codegen<->live parity under the wrist FLANGE-OFFSET gauge (never-again guard).

Companion to :mod:`test_codegen_gauge_parity` (which guards the axis-sign gauge).
This closes the dimension that let the composer canonicalization bug ship: the
live ``ikgeo`` spherical-wrist solvers call ``canonicalize_spherical_wrist``
(re-gauging a URDF flange offset onto the wrist intersection), but their codegen
composers did **not** -- so for any arm whose last wrist joint is placed along
its own axis (FANUC M-710iC, Kinova j2s6s300) the emitted artifact baked the
wrong shoulder-to-wrist consolidation and returned **zero** solutions, while the
live solver returned the correct set. Puma 560 (canonical wrist) masked it; every
Wave 2 industrial Pieper arm would have shipped broken (#424).

The fix (#433) added ``canonicalize_spherical_wrist`` to **all three**
spherical-wrist composers -- ``spherical_two_parallel``,
``spherical_two_intersecting``, and generic ``spherical``. This test guards all
three systematically (#435): for each, it synthesizes an arm with and without a
wrist flange offset, emits the artifact from codegen in-process, and asserts full
coverage at machine-precision FK closure. No shipped fixture required: if any
spherical-wrist composer ever drops the canonicalization, CI goes red here
regardless of roster. Verified to fail (0 coverage / math-domain error on the
offset case) if a composer's ``canonicalize_spherical_wrist`` call is removed.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from ssik._kinbody import JointSpec, KinBody, build_kinbody
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import dispatch
from ssik.kinematics.poe_fk import poe_forward_kinematics

_Z = np.array([0.0, 0.0, 1.0])
_Y = np.array([0.0, 1.0, 0.0])
_X = np.array([1.0, 0.0, 0.0])


def _trans(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _wrist(flange: float) -> list[JointSpec]:
    """Shared spherical wrist (axes 3, 4, 5 intersect). ``flange`` offsets the
    last wrist joint along its own axis -- a still-spherical but non-canonical
    wrist (the FANUC/j2s6 shape) that every composer must re-gauge. Concurrency
    of the wrist axes is a local property, preserved under any shoulder."""
    return [
        JointSpec(parent_link_T=_trans(0.4, 0, 0.1), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0, 0), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0, flange), axis=_Z, joint_type="revolute"),
    ]


def _spherical_two_parallel_specs(flange: float) -> list[JointSpec]:
    """Puma-class: base z, parallel shoulder/elbow about y, spherical wrist."""
    return [
        JointSpec(parent_link_T=_trans(0, 0, 0.3), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0.1, 0.2), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0.4, 0, 0), axis=_Y, joint_type="revolute"),
        *_wrist(flange),
    ]


def _spherical_two_intersecting_specs(flange: float) -> list[JointSpec]:
    """Intersecting-shoulder: ||p[1]|| = 0 (joints 0, 1 share an origin) and
    axes[1] not parallel to axes[2], spherical wrist. (IRB120 / xArm6 class.)"""
    return [
        JointSpec(parent_link_T=_trans(0, 0, 0.3), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0, 0), axis=_Y, joint_type="revolute"),  # p[1]=0
        JointSpec(parent_link_T=_trans(0.4, 0, 0.2), axis=_X, joint_type="revolute"),  # not || y
        *_wrist(flange),
    ]


def _spherical_specs(flange: float) -> list[JointSpec]:
    """Generic spherical-wrist: no shoulder specialisation (||p[1]|| != 0 and
    axes[1] not parallel to axes[2]), spherical wrist. Routes to ``ikgeo.spherical``
    (SP5 shoulder)."""
    return [
        JointSpec(parent_link_T=_trans(0, 0, 0.3), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0.1, 0.2), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0.4, 0, 0.2), axis=_X, joint_type="revolute"),
        *_wrist(flange),
    ]


# Each entry: (solver_name, specs factory). All three spherical-wrist composers
# call canonicalize_spherical_wrist; all three are guarded here.
_TOPOLOGIES: list[tuple[str, Callable[[float], list[JointSpec]]]] = [
    ("ikgeo.spherical_two_parallel", _spherical_two_parallel_specs),
    ("ikgeo.spherical_two_intersecting", _spherical_two_intersecting_specs),
    ("ikgeo.spherical", _spherical_specs),
]


def _emit_and_import(kb: KinBody, expected_solver: str, tag: str) -> object:
    plan = dispatch(kb)
    assert plan.solver_name == expected_solver, f"{tag}: {plan.solver_name} != {expected_solver}"
    name = f"_wrist_gauge_{tag}_{uuid.uuid4().hex[:8]}"
    src = emit_artifact(kb=kb, plan=plan, module_name=name, output_path=None).source
    path = Path(tempfile.gettempdir()) / f"{name}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        path.unlink(missing_ok=True)
    return mod


@pytest.mark.parametrize(("solver", "specs_fn"), _TOPOLOGIES, ids=[t[0] for t in _TOPOLOGIES])
@pytest.mark.parametrize(("tag", "flange"), [("canonical", 0.0), ("flange_offset", 0.15)])
def test_emitted_artifact_matches_live_under_flange_gauge(
    tag: str, flange: float, solver: str, specs_fn: Callable[[float], list[JointSpec]]
) -> None:
    """Every spherical-wrist composer emits an artifact that solves every
    reachable pose (coverage) at machine-precision FK closure, with or without a
    wrist flange offset. The offset case is broken (0 coverage / math-domain
    error) if the composer skips ``canonicalize_spherical_wrist`` (#424/#433)."""
    kb = build_kinbody(specs_fn(flange))
    art = _emit_and_import(kb, solver, f"{solver.split('.')[-1]}_{tag}")
    rng = np.random.default_rng(0)
    covered = 0
    worst_fk = 0.0
    n = 40
    for _ in range(n):
        q = rng.uniform(-1.2, 1.2, size=6)
        t_target = poe_forward_kinematics(kb, q)
        sols = art.solve(t_target, respect_limits=False)  # type: ignore[attr-defined]
        if sols:
            covered += 1
            worst_fk = max(worst_fk, max(float(s.fk_residual) for s in sols))
    assert covered == n, (
        f"{solver} {tag}: emitted artifact covered {covered}/{n} reachable poses. "
        f"A composer that skips canonicalize_spherical_wrist breaks the "
        f"flange-offset case."
    )
    assert worst_fk < 1e-8, f"{solver} {tag}: worst FK {worst_fk:.2e} on emitted artifact"
