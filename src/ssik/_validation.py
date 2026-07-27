"""Coverage + FK-closure validation for a prebuilt artifact.

The onboarding gate: sample poses across an arm's real joint limits, FK + solve
with the **shipped** ``module.solve``, and assert that a minimum fraction return
at least one IK (coverage) and every returned IK FK-closes within a ceiling.

A broken or degenerate arm (0 solutions, locked ``[0, 0]`` joint limits, a wrong
end-effector gauge) fails the *coverage* assertion. "FK <= tol on every retained
IK" is vacuously true at zero coverage, so coverage is asserted separately -- the
lesson from the Wave 1 onboarding (#423), where a 0-solution arm passed the whole
gate.

One tested implementation, shared by the ``ssik add-arm`` test scaffold (and any
future ``add-arm`` build-time validation), so the gate can't silently rot in
generated code.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class _Solution(Protocol):
    fk_residual: float


class _SolvableArm(Protocol):
    """The surface a prebuilt artifact exposes that this validator needs."""

    DOF: int
    _KB: object  # exposes ``.joints`` with per-joint ``.limits``

    def fk(self, q: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def solve(self, t_target: NDArray[np.float64]) -> list[_Solution]: ...


def joint_bounds(module: _SolvableArm) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(lo, hi)`` per-joint sampling bounds from the artifact's baked KinBody.
    Continuous / limitless joints sample ``[-pi, pi]``."""
    lo: list[float] = []
    hi: list[float] = []
    for j in module._KB.joints:  # type: ignore[attr-defined]
        lim = getattr(j, "limits", None)
        if lim is None:
            lo.append(-np.pi)
            hi.append(np.pi)
        else:
            lo.append(float(lim[0]))
            hi.append(float(lim[1]))
    return np.array(lo), np.array(hi)


def check_solve_coverage(
    module: _SolvableArm, *, n_poses: int = 64, seed: int = 0
) -> tuple[float, float]:
    """Sample ``n_poses`` within the arm's joint limits, ``solve(fk(q))`` via the
    shipped path, and return ``(coverage_fraction, worst_fk)``.

    ``coverage_fraction`` is the fraction of poses that returned >= 1 IK;
    ``worst_fk`` is the largest FK residual across all returned solutions (0.0 if
    none were returned).
    """
    lo, hi = joint_bounds(module)
    rng = np.random.default_rng(seed)
    covered = 0
    worst_fk = 0.0
    for _ in range(n_poses):
        q = rng.uniform(lo, hi)
        sols = module.solve(module.fk(q))
        if sols:
            covered += 1
            worst_fk = max(worst_fk, max(float(s.fk_residual) for s in sols))
    return covered / n_poses, worst_fk


def assert_solve_coverage(
    module: _SolvableArm,
    *,
    min_coverage: float = 0.95,
    fk_ceiling: float = 1e-6,
    n_poses: int = 64,
    seed: int = 0,
) -> tuple[float, float]:
    """Raise :class:`AssertionError` if coverage ``< min_coverage`` or the worst
    FK residual is ``>= fk_ceiling``. Returns ``(coverage, worst_fk)`` on success.

    This is the bulletproof gate: a broken arm (few/no solutions) trips the
    coverage assertion instead of passing vacuously.
    """
    name = getattr(module, "__name__", "arm")
    coverage, worst_fk = check_solve_coverage(module, n_poses=n_poses, seed=seed)
    if coverage < min_coverage:
        raise AssertionError(
            f"{name}: coverage {coverage:.0%} < {min_coverage:.0%} "
            f"({round(coverage * n_poses)}/{n_poses} reachable poses returned an IK). "
            f"A broken or degenerate arm returns few/no solutions."
        )
    if worst_fk >= fk_ceiling:
        raise AssertionError(f"{name}: worst FK {worst_fk:.2e} >= {fk_ceiling:.0e} ceiling")
    return coverage, worst_fk
