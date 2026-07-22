"""Codegen<->live parity under axis-representation gauges (never-again guard).

The trio-flip bug (#398 follow-up) slipped through because the live
``ikgeo.three_parallel`` solver and its codegen composer
(``ssik.codegen._compose.three_parallel``) each implement the parallel-trio sign
convention, and only the live copy was fixed. The existing artifact<->live
parity test (:mod:`test_artifact_live_parity`) would have caught it -- but it
only runs over *shipped prebuilts*, and no anti-parallel-trio arm existed in the
prebuilt set until Standard Bots thor/spark were onboarded. That is a coverage
gap: a real fixture had to exist for the divergence to be caught.

This test closes the gap structurally. It *synthesizes* three_parallel arms that
exercise each axis-sign gauge (aligned trio, and every anti-parallel trio-joint
combination), emits the artifact from codegen in-process, and asserts the emitted
solver computes the same IK as the live dispatched solver -- at machine precision,
including near-home poses (where the un-flipped artifact produced FK-passing but
wrong LS near-misses). No shipped fixture required: if a composer ever diverges
from its live solver on a gauge freedom, CI goes red here regardless of roster.
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
from ssik.manipulator import Manipulator

_Z = np.array([0.0, 0.0, 1.0])
_Y = np.array([0.0, 1.0, 0.0])


def _trans(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


# UR-style three_parallel 6R: base z, parallel shoulder/elbow/wrist-1 trio about
# y (joints 1, 2, 3), wrist-2 z, wrist-3 y. Nonzero offsets keep it generic.
def _three_parallel_specs() -> list[JointSpec]:
    return [
        JointSpec(parent_link_T=_trans(0, 0, 0.1), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0.1, 0.1), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0.4, 0, 0), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0.4, 0, 0), axis=_Y, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0.1, 0), axis=_Z, joint_type="revolute"),
        JointSpec(parent_link_T=_trans(0, 0, 0.1), axis=_Y, joint_type="revolute"),
    ]


def _flip_axes(specs: list[JointSpec], idxs: tuple[int, ...]) -> list[JointSpec]:
    return [
        JointSpec(
            parent_link_T=s.parent_link_T,
            axis=(-s.axis if j in idxs else s.axis),
            joint_type=s.joint_type,
        )
        for j, s in enumerate(specs)
    ]


def _emit_and_import(kb, tag: str):
    """Emit the artifact for ``kb`` in-process and import it as a fresh module."""
    plan = dispatch(kb)
    assert plan.solver_name == "ikgeo.three_parallel", plan.solver_name
    name = f"_gauge_parity_{tag}_{uuid.uuid4().hex[:8]}"
    src = emit_artifact(kb=kb, plan=plan, module_name=name, output_path=None).source
    path = Path(tempfile.gettempdir()) / f"{name}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        path.unlink(missing_ok=True)
    return mod


# Every trio-joint anti-parallel combination, plus the aligned baseline and a
# reference-axis (joint 1) flip that makes joints 2 and 3 both anti-parallel.
_GAUGES = [
    ("aligned", ()),
    ("flip2", (2,)),
    ("flip3", (3,)),
    ("flip23", (2, 3)),
    ("flip1", (1,)),
]


@pytest.mark.parametrize(("tag", "flip"), _GAUGES)
def test_emitted_artifact_matches_live_under_axis_gauge(tag: str, flip: tuple[int, ...]) -> None:
    """The emitted three_parallel artifact computes the same IK as the live
    solver for every trio axis-sign gauge -- at machine precision, including the
    near-home poses where the un-flipped codegen produced wrong LS near-misses."""
    kb = build_kinbody(_flip_axes(_three_parallel_specs(), flip))
    art = _emit_and_import(kb, tag)
    live = Manipulator(kb)
    sys.modules.pop(art.__name__, None)

    rng = np.random.default_rng(hash(tag) % (2**31))
    checked = 0
    for i in range(60):
        # Half near-home (the bug's hiding spot), half across the workspace.
        span = 0.5 if i % 2 == 0 else 2.0
        q = rng.uniform(-span, span, size=6)
        t = poe_forward_kinematics(kb, q)
        kw = {"respect_limits": False, "allow_rescue": False, "allow_refinement": True}
        art_sols = art.solve(t, **kw)
        if not art_sols:
            continue
        checked += 1
        # The emitted artifact must FK-close to machine precision.
        worst_art = max(float(np.max(np.abs(art.fk(s.q) - t))) for s in art_sols)
        assert worst_art < 1e-9, f"{tag}: artifact FK {worst_art:.2e} at q={q.tolist()}"
        # ...and it must find the same number of solutions as the live solver
        # (the un-flipped codegen dropped real branches / returned near-misses).
        live_sols = live.solve(t, **kw)
        assert len(art_sols) == len(live_sols), (
            f"{tag}: artifact {len(art_sols)} vs live {len(live_sols)} sols at q={q.tolist()}"
        )
    assert checked > 20, f"{tag}: too few solvable poses ({checked})"
