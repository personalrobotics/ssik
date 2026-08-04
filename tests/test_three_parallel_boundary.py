"""Boundary-behaviour parity for three_parallel across both backends (#499).

Two things the reachable-pose family sweep (``test_three_parallel_family``)
doesn't touch, both run against ``{python, cpp}``:

1. **In-solve Newton rescue.** Random reachable poses close analytically, so
   ``allow_refinement=True`` is a no-op there. At a genuinely near-singular pose
   the closed form emits a candidate that FK-misses just past the 1e-7 gate; the
   forced-refinement path (what the shipped artifact uses) Newton-polishes it and
   keeps it if it now closes. We pin such a pose and assert the *robust*
   invariants of that wired path:

   - **Soundness**: every refined solution FK-closes within the arm's ceiling.
   - **Monotonic + additive**: refinement never drops a solution the analytical
     pass already found; it only adds rescued ones (norefine set is a subset of
     the refine set).

   We deliberately do NOT assert q-space parity of the *rescued* solution across
   backends: at a near-singular pose that solution is genuinely non-unique (it
   FK-closes at ~8e-8 with a q that differs by BLAS backend -- the same
   representative ambiguity documented in ``test_three_parallel`` #56). Nor do we
   assert the rescue strictly fires on every platform (whether the candidate
   lands just inside or just outside the 1e-7 gate is backend-dependent). The
   pinned pose fires on the reference; the assertions hold on any platform.

2. **Unreachable targets.** A pose outside the workspace must return the empty
   set with ``is_ls=True`` -- identically on both backends.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from ssik._native import _NATIVE_SOLVERS
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.prebuilt._manifest import load_manifest


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _q_close(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return all(abs(_wrap(float(ai - bi))) < tol for ai, bi in zip(a, b, strict=True))


# Near-singular poses (wrist-pitch ~ +/-pi/2) where the closed form emits a
# candidate just past the 1e-7 gate that Newton rescues -- so
# allow_refinement=True yields more solutions than allow_refinement=False.
# Found by biased search (see git history); one pinned pose is enough to
# exercise the wired rescue path, as one is enough for the #56 regression.
_RESCUE_POSES: list[tuple[str, list[float]]] = [
    (
        "z1_ik",
        [
            1.5828215897533102,
            0.9832636327213744,
            -0.5320754096685629,
            1.5535114891012707,
            -1.4919333724514705,
            1.5121741405632485,
        ],
    ),
]


@pytest.mark.parametrize(("arm_name", "q_star"), _RESCUE_POSES, ids=[a for a, _ in _RESCUE_POSES])
def test_in_solve_refinement_is_sound_and_additive(
    arm_name: str, q_star: list[float], three_parallel_backend: Any
) -> None:
    """At a near-singular pose, the forced-refinement path is sound and additive.

    Runs on both backends: every refined solution FK-closes within ceiling, and
    refinement only adds solutions (never drops the analytical ones).
    """
    arm = load_manifest()[arm_name]
    kb = importlib.import_module(f"ssik.prebuilt.{arm_name}")._KB
    ceiling = float(arm.fk_ceiling_fuzz)
    t_star = np.asarray(poe_forward_kinematics(kb, np.array(q_star)), dtype=np.float64)

    no_refine, _ = three_parallel_backend(kb, t_star, allow_refinement=False)
    refined, is_ls = three_parallel_backend(kb, t_star, allow_refinement=True)

    assert not is_ls, f"{arm_name}: unexpected is_ls at reachable near-singular pose"

    # Soundness: every refined solution reproduces the target under FK.
    for sol in refined:
        worst = float(np.abs(poe_forward_kinematics(kb, sol.q) - t_star).max())
        assert worst < ceiling, (
            f"{arm_name}: refined IK FK-closes to {worst:.2e} (ceiling {ceiling:.0e})"
        )

    # Monotonic + additive: refinement never loses an analytical solution.
    assert len(refined) >= len(no_refine), f"{arm_name}: refinement dropped solutions"
    for sol in no_refine:
        assert any(_q_close(sol.q, r.q, 1e-6) for r in refined), (
            f"{arm_name}: an analytical solution vanished under refinement"
        )


_UNREACHABLE_TARGETS = {
    "far_translation": np.array(
        [[1.0, 0, 0, 10.0], [0, 1.0, 0, 10.0], [0, 0, 1.0, 10.0], [0, 0, 0, 1.0]]
    ),
    "far_below": np.array([[1.0, 0, 0, 0.0], [0, 1.0, 0, 0.0], [0, 0, 1.0, -5.0], [0, 0, 0, 1.0]]),
}


@pytest.mark.parametrize("target_name", list(_UNREACHABLE_TARGETS))
def test_unreachable_returns_empty_is_ls(target_name: str, three_parallel_backend: Any) -> None:
    """An unreachable target returns the empty set with is_ls=True on both backends."""
    kb = importlib.import_module("ssik.prebuilt.ur5_ik")._KB
    t = _UNREACHABLE_TARGETS[target_name]
    solutions, is_ls = three_parallel_backend(kb, t)
    assert solutions == [] or len(solutions) == 0, f"{target_name}: expected no solutions"
    assert is_ls, f"{target_name}: expected is_ls=True for unreachable target"


_FAMILY_ARMS = [a.name for a in load_manifest().values() if a.solver in _NATIVE_SOLVERS]


@pytest.mark.parametrize("arm_name", _FAMILY_ARMS)
def test_rescue_is_dormant_across_family(arm_name: str) -> None:
    """The T-perturbation rescue never fires for the native geometric families.

    This is the assumption that lets the native artifact layer (#503/#510) omit
    the full ``rescue_via_T_perturbation`` port: the analytical path is complete
    for these geometric families (three_parallel, spherical_two_parallel), so
    ``allow_rescue`` changes nothing on reachable poses. If a future geometry or
    tolerance change makes rescue start firing here, this goes red -- the signal
    that the C++ artifact layer now needs the rescue port (deferred to the RR/HP
    families where rescue is load-bearing).
    """
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    kb = mod._KB
    ranges = [
        (float(j.limits[0]), float(j.limits[1])) if j.limits else (-np.pi, np.pi) for j in kb.joints
    ]
    rng = np.random.default_rng(0)
    for _ in range(40):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t = np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)
        with_rescue = mod.solve(t, allow_rescue=True, respect_limits=False)
        without_rescue = mod.solve(t, allow_rescue=False, respect_limits=False)
        assert len(with_rescue) == len(without_rescue), (
            f"{arm_name}: rescue fired at a reachable pose -- the C++ artifact "
            f"layer's rescue omission (#503) is no longer valid"
        )
