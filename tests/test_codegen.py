"""End-to-end correctness for :mod:`ssik.core.codegen`.

Builds the artifact for each shipped fixture, imports it as a Python
module, and verifies that ``solve(T)`` produces the same solutions as
calling the underlying ssik solver directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ssik._kinbody import KinBody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import dispatch
from ssik.core.tolerances import TolerancePolicy
from ssik.solvers.ikgeo import spherical_two_parallel, three_parallel
from ssik.subproblems._rotation import rotation_matrix

FIXTURES = Path(__file__).parent / "fixtures"


def _fk_poe(kb: KinBody, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def _import_artifact(source: str, module_name: str, tmp_path: Path) -> object:
    """Write ``source`` to ``tmp_path/<module_name>.py`` and import it."""
    artifact_path = tmp_path / f"{module_name}.py"
    artifact_path.write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, artifact_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_emit_ur5_artifact_in_memory(tmp_path: Path) -> None:
    """Render the artifact for UR5 and check the rendered source has the
    public surface we promised."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    plan = dispatch(kb)
    result = emit_artifact(
        kb=kb,
        plan=plan,
        module_name="ur5_ik_test",
        output_path=None,
        arm_label="UR5",
    )
    # Public-API surface check. UR5 (three_parallel) now uses the
    # specialised emitter, so the artifact's body is inlined math + a
    # runtime SP6 import (not a thin wrapper around the full solver).
    for needle in (
        "def solve(",
        'SOLVER_NAME = "ikgeo.three_parallel"',
        "SOLVER_TIER = 0",
        "EXPECTED_MS_MEDIAN =",
        "FLOP_BUDGET =",
        "DISPATCH_REASON =",
        "_KB = _build_kb()",
        "_sp6_runtime",  # specialised three_parallel imports SP6 runtime
        "_solve_algebraic",  # specialised emitter shape
    ):
        assert needle in result.source, f"missing {needle!r}"
    assert result.module_name == "ur5_ik_test"
    assert result.output_path is None


@pytest.mark.parametrize(
    ("urdf", "base", "ee", "module_name", "label", "direct_solver"),
    [
        ("ur5.urdf", "base_link", "ee_link", "ur5_ik_emit", "UR5", three_parallel),
        (
            "puma560.urdf",
            "base_link",
            "wrist_3_link",
            "puma560_ik_emit",
            "Puma 560",
            spherical_two_parallel,
        ),
    ],
)
def test_emit_then_import_then_roundtrip(
    tmp_path: Path,
    urdf: str,
    base: str,
    ee: str,
    module_name: str,
    label: str,
    direct_solver: object,
) -> None:
    """Full pipeline: dispatch -> emit -> import -> solve -> compare with the
    direct solver call. The artifact must produce the same solutions as the
    direct path on identical inputs.
    """
    kb = load_urdf_kinbody_normalized(FIXTURES / urdf, base, ee)
    plan = dispatch(kb)
    artifact_path = tmp_path / f"{module_name}.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name=module_name,
        output_path=str(artifact_path),
        arm_label=label,
    )
    assert artifact_path.exists()

    spec = importlib.util.spec_from_file_location(module_name, artifact_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    # Sanity: artifact exports the documented public constants.
    assert plan.solver_name == mod.SOLVER_NAME
    assert plan.tier == mod.SOLVER_TIER
    assert isinstance(mod.DISPATCH_REASON, str)

    # Roundtrip: pick a random seeded q, FK it, solve, verify all returned
    # solutions FK-close on the target. Then check the artifact's output
    # matches the direct solver's output (same #solutions; q-vectors agree
    # set-wise within machine precision).
    rng = np.random.default_rng(seed=42)
    for trial in range(5):
        q_star = rng.uniform(-1.0, 1.0, size=6)
        T_star = _fk_poe(kb, q_star)
        sols_artifact = mod.solve(T_star)
        sols_direct, _is_ls_direct = direct_solver.solve(kb, T_star)  # type: ignore[attr-defined]
        assert len(sols_artifact) == len(sols_direct), f"trial {trial}: solution count disagrees"
        for sol in sols_artifact:
            T_check = _fk_poe(kb, sol.q)
            assert np.allclose(T_check, T_star, atol=1e-9), (
                f"trial {trial}: artifact q={sol.q.tolist()} fails FK"
            )


def test_artifact_solve_accepts_policy_and_refinement_kwargs(tmp_path: Path) -> None:
    """The emitted ``solve()`` exposes the underlying solver's kwargs:
    ``policy``, ``allow_refinement``, ``refinement_max_iters``. This is
    the rich-API contract the artifact must preserve.
    """
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    plan = dispatch(kb)
    artifact_path = tmp_path / "ur5_ik_kwargs.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name="ur5_ik_kwargs",
        output_path=str(artifact_path),
        arm_label="UR5",
    )
    spec = importlib.util.spec_from_file_location("ur5_ik_kwargs", artifact_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ur5_ik_kwargs"] = mod
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(seed=11)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    T_star = _fk_poe(kb, q_star)

    # Default call: defaults match the underlying solver's defaults.
    sols_default = mod.solve(T_star)
    assert sols_default

    # Custom policy: tighter FK-closure threshold; result still closes.
    strict = TolerancePolicy(subproblem_numerical=1e-9)
    sols_strict = mod.solve(T_star, policy=strict)
    assert sols_strict
    for sol in sols_strict:
        T_check = _fk_poe(kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-8), "strict-policy result fails FK"

    # Newton refinement: opt-in path with cap on iterations.
    sols_refined = mod.solve(T_star, allow_refinement=True, refinement_max_iters=8)
    assert sols_refined
    for sol in sols_refined:
        T_check = _fk_poe(kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-9)
