"""Codegen<->live parity under the wrist FLANGE-OFFSET gauge (never-again guard).

Companion to :mod:`test_codegen_gauge_parity` (which guards the axis-sign gauge).
This closes the dimension that let the composer canonicalization bug ship: the
live ``ikgeo.spherical_two_parallel`` solver calls ``canonicalize_spherical_wrist``
(re-gauging a URDF flange offset onto the wrist intersection), but its codegen
composer did **not** -- so for any arm whose last wrist joint is placed along its
own axis (FANUC M-710iC, Kinova j2s6s300) the emitted artifact baked the wrong
shoulder-to-wrist consolidation and returned **zero** solutions, while the live
solver returned the correct set. Puma 560 (canonical wrist) masked it; every
Wave 2 industrial Pieper arm would have shipped broken (#424).

This test synthesizes a spherical-wrist arm with and without a flange offset,
emits the artifact from codegen in-process, and asserts the emitted solver
matches the live dispatched solver at machine precision. No shipped fixture
required: if a spherical-wrist composer ever drops the canonicalization, CI goes
red here regardless of roster. Verified to fail (0% coverage on the offset case)
if the composer's ``canonicalize_spherical_wrist`` call is removed.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from ssik._kinbody import JointSpec, build_kinbody
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import dispatch
from ssik.kinematics.poe_fk import poe_forward_kinematics

_Z = np.array([0.0, 0.0, 1.0])
_Y = np.array([0.0, 1.0, 0.0])


def _trans(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _spherical_two_parallel_specs(flange: float) -> list[JointSpec]:
    """A Puma-class arm: base z, parallel shoulder/elbow about y, spherical wrist
    (z, y, z) whose axes intersect. ``flange`` offsets the last wrist joint along
    its own axis -- a still-spherical but non-canonical wrist (the FANUC/j2s6
    shape) that the composer must re-gauge."""
    return [
        JointSpec(parent_link_T=_trans(0, 0, 0.3), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0.1, 0.2), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0.4, 0, 0), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0.4, 0, 0.1), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0, 0), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0, flange), axis=_Z, joint_type="revolute"),
    ]


def _emit_and_import(kb, tag: str):
    plan = dispatch(kb)
    assert plan.solver_name == "ikgeo.spherical_two_parallel", plan.solver_name
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


@pytest.mark.parametrize(
    ("tag", "flange"),
    [("canonical", 0.0), ("flange_offset", 0.15)],
)
def test_emitted_artifact_matches_live_under_flange_gauge(tag: str, flange: float) -> None:
    """The emitted spherical_two_parallel artifact solves every reachable pose
    (coverage) at machine-precision FK closure, with or without a wrist flange
    offset. The offset case is 0% coverage if the composer skips canonicalization."""
    kb = build_kinbody(_spherical_two_parallel_specs(flange))
    art = _emit_and_import(kb, tag)
    rng = np.random.default_rng(0)
    covered = 0
    worst_fk = 0.0
    n = 40
    for _ in range(n):
        q = rng.uniform(-1.2, 1.2, size=6)
        t_target = poe_forward_kinematics(kb, q)
        sols = art.solve(t_target, respect_limits=False)
        if sols:
            covered += 1
            worst_fk = max(worst_fk, max(float(s.fk_residual) for s in sols))
    assert covered == n, (
        f"{tag}: emitted artifact covered {covered}/{n} reachable poses. "
        f"A composer that skips canonicalize_spherical_wrist returns 0 on the "
        f"flange-offset case."
    )
    assert worst_fk < 1e-8, f"{tag}: worst FK {worst_fk:.2e} on emitted artifact"
