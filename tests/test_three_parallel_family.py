"""Cross-backend FK-roundtrip sweep across the whole three-parallel 6R family.

Every arm that dispatches to ``ikgeo.three_parallel`` (17 at time of writing:
the UR sizes, CR5, Nova5, Z1, and the anti-parallel-trio Standard Bots
core/spark/thor) is swept through the *same* correctness assertions against
both backends (#499). Manifest-driven, so a new three-parallel arm is covered
automatically.

This complements two existing suites without duplicating them:

- ``test_three_parallel`` runs the deep 500-pose hypothesis fuzz + singular +
  #56 coverage, but only on UR5. Here we go wide (all 17 arms) at moderate
  depth -- catching *structural* per-arm bugs the single-arm depth can't, e.g.
  the ``trio_flip`` sign path that only the anti-parallel Standard Bots arms
  exercise.
- ``test_prebuilt_uniform_fuzz`` fuzzes the *artifacts* (``<arm>.solve`` with
  limits + refinement); here we validate the core solver (Python live + the
  native C++ backend) directly.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.prebuilt._manifest import load_manifest

_THREE_PARALLEL_ARMS = [
    arm.name for arm in load_manifest().values() if arm.solver == "ikgeo.three_parallel"
]

_N_POSES = 40
_SEED_RECOVERY_TOL = 5e-4  # SP6 near-double-root q-space envelope (see test_three_parallel)


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _q_matches(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return all(abs(_wrap(float(ai - bi))) < tol for ai, bi in zip(a, b, strict=True))


@pytest.mark.parametrize("arm_name", _THREE_PARALLEL_ARMS)
def test_family_fk_roundtrip(arm_name: str, three_parallel_backend: Any) -> None:
    """Each family arm: random reachable poses solve, every returned IK
    FK-closes within the arm's ceiling, and the seeded q* is recovered.

    Run against both the Python reference and the native C++ backend, so the
    C++ solver is held to the same bar across every family geometry -- with no
    assertion logic written in C++.
    """
    manifest = load_manifest()
    arm = manifest[arm_name]
    kb = importlib.import_module(f"ssik.prebuilt.{arm_name}")._KB
    ceiling = float(arm.fk_ceiling_fuzz)

    ranges = [
        (float(j.limits[0]), float(j.limits[1])) if j.limits else (-np.pi, np.pi) for j in kb.joints
    ]
    rng = np.random.default_rng(0)

    for _ in range(_N_POSES):
        q_star = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t_star = np.asarray(poe_forward_kinematics(kb, q_star), dtype=np.float64)

        solutions, is_ls = three_parallel_backend(kb, t_star)
        assert not is_ls, f"{arm_name}: unexpected is_ls at reachable pose"
        assert 1 <= len(solutions) <= 8, f"{arm_name}: got {len(solutions)} solutions"

        for sol in solutions:
            worst = float(np.abs(poe_forward_kinematics(kb, sol.q) - t_star).max())
            assert worst < ceiling, (
                f"{arm_name}: returned IK FK-closes to {worst:.2e} (ceiling {ceiling:.0e})"
            )

        assert any(_q_matches(s.q, q_star, _SEED_RECOVERY_TOL) for s in solutions), (
            f"{arm_name}: seeded q*={q_star.tolist()} not recovered"
        )
