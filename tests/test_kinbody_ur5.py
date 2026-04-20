"""Integration smoke: hand-built UR5 chain + vendored IKFastSolver.

Marked ``slow`` — symbolic IK for a 6R arm runs on the order of minutes and
is not appropriate for the default CI pass. Opt in with ``pytest -m slow``.

Coverage:
- ``test_forward_kinematics_chain_runs`` — exercises every shim method the
  solver calls (via ``forwardKinematicsChain``). No full IK generation.
- ``test_generate_ik_solver_produces_output`` — the literal #5 criterion:
  ``generateIkSolver`` runs to completion and returns a non-None chaintree.
- ``test_ur5_pfk_matches_shim_fk`` — the solver's internal symbolic FK
  (``chaintree.Pfk``) agrees numerically with our independent shim FK.
  Catches convention bugs (``T_left``/``T_right``, axis orientation, DH
  decomposition) before we trust any IK output.
- ``test_ur5_fk_ik_roundtrip`` — full correctness gate: pick q*, compute
  target position P* via our FK, feed (P*, free joints) through the
  generated IK chaintree, verify at least one candidate solution
  re-produces P* within tolerance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def ur5_chaintree() -> Any:
    """Build the UR5 Translation3D chaintree once per module (~2 min)."""
    from fixtures.ur5 import ur5_specs
    from ikfastpy._kinbody import build_kinbody
    from ikfastpy._vendor.ikfast import IKFastSolver

    kb = build_kinbody(ur5_specs())
    solver = IKFastSolver(kinbody=kb)
    return solver.generateIkSolver(
        baselink="base_link",
        eelink="ee_link",
        freeindices=[3, 4, 5],
        solvefn=IKFastSolver.solveFullIK_Translation3D,
    )


def test_forward_kinematics_chain_runs() -> None:
    """Cheaper than a full IK solve: exercises *every* shim method the solver
    invokes on the chain (context manager, GetDOF, GetJointFromDOFIndex,
    GetChain, GetName, IsStatic, GetHierarchy*, Is{Revolute,Prismatic,Mimic},
    GetDOFIndex) without the combinatorial cost of ``generateIkSolver``.
    """
    from fixtures.ur5 import ur5_specs
    from ikfastpy._kinbody import build_kinbody
    from ikfastpy._vendor.ikfast import IKFastSolver

    kb = build_kinbody(ur5_specs())
    solver = IKFastSolver(kinbody=kb)
    chainlinks = kb.GetChain("base_link", "ee_link", returnjoints=False)
    chainjoints = kb.GetChain("base_link", "ee_link", returnjoints=True)
    links_raw, jointvars = solver.forwardKinematicsChain(chainlinks, chainjoints)

    assert len(jointvars) == 6
    assert len(links_raw) >= 1


def test_generate_ik_solver_produces_output(ur5_chaintree: Any) -> None:
    """Full issue #5 criterion: ``generateIkSolver`` returns sympy output."""
    assert ur5_chaintree is not None


def test_ur5_pfk_matches_shim_fk(ur5_chaintree: Any) -> None:
    """Stage A of correctness validation: ikfast's internal symbolic FK
    agrees with our independent FK at a random joint configuration.

    If this fails, the shim's joint-transform convention is wrong and no
    subsequent IK result can be trusted.
    """
    import sympy

    from fixtures.ur5 import ur5_fk

    q_star = [0.3, -0.7, 0.9, 1.1, -0.5, 0.2]

    T_shim = ur5_fk(q_star)
    p_shim = T_shim[:3, 3]

    subs = {sympy.Symbol(f"j{i}"): q_star[i] for i in range(6)}
    p_ikfast = np.array(
        [float(ur5_chaintree.Pfk[i].subs(subs).evalf()) for i in range(3)],
        dtype=np.float64,
    )

    assert np.allclose(p_shim, p_ikfast, atol=1e-9), (
        f"shim FK {p_shim} vs ikfast Pfk {p_ikfast} diverge; diff={p_shim - p_ikfast}"
    )


def test_ur5_fk_ik_roundtrip(ur5_chaintree: Any) -> None:
    """Stage B of correctness validation: pick a random joint config q*,
    compute the target EE position via our FK, feed (P*, free joints) into
    the generated IK chaintree, and verify at least one candidate solution
    reproduces P* within tolerance.

    This is the real correctness gate. Passing means the sympy-1.14 compat
    fixes in PR #30 (``SimplifyAtan2`` oscillation guard + ``isValidSolution``
    TypeError widening) didn't silently drop or corrupt solutions on the
    path to a real Translation3D IK.
    """
    from fixtures.ur5 import ur5_fk
    from fk_ik_eval import eval_chaintree

    q_star = [0.3, -0.7, 0.9, 1.1, -0.5, 0.2]
    p_star = ur5_fk(q_star)[:3, 3]

    q_free = {"j3": q_star[3], "j4": q_star[4], "j5": q_star[5]}
    candidates = eval_chaintree(
        ur5_chaintree, q_free=q_free, target_pos=(p_star[0], p_star[1], p_star[2])
    )

    assert len(candidates) > 0, "chaintree walker produced no candidate solutions"

    matches: list[dict[str, float]] = []
    for cand in candidates:
        full_q = [cand[f"j{i}"] for i in range(6)]
        p_cand = ur5_fk(full_q)[:3, 3]
        if np.allclose(p_cand, p_star, atol=1e-6):
            matches.append(cand)

    assert len(matches) > 0, (
        f"no candidate reproduces target within 1e-6. "
        f"q*={q_star}, P*={p_star.tolist()}, "
        f"candidates={candidates}"
    )
