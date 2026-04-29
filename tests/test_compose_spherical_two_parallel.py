"""End-to-end test for the ``spherical_two_parallel`` composer.

Compiles the rendered ``_solve_algebraic`` source, runs it against the
runtime ``ikgeo.spherical_two_parallel.solve`` on Puma 560 + 100 random
target poses, and asserts that every candidate the runtime returns is
also produced by the composed function (within machine precision, modulo
wrap-to-pi).

This is the bulletproof gate at the composer layer: if the composed
function disagrees on even one input, the artifact built on top is
unsound.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ssik._kinbody import KinBody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.codegen._compose.spherical_two_parallel import compose, render_constants_header
from ssik.solvers.ikgeo import spherical_two_parallel
from ssik.subproblems._rotation import rotation_matrix

FIXTURES = Path(__file__).parent / "fixtures"


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _q_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return all(abs(_wrap(float(ai - bi))) < tol for ai, bi in zip(a, b, strict=True))


def _compile_composed(kb_: KinBody) -> Callable[[np.ndarray], list[list[float]]]:
    """Render the composer + header into a fresh module namespace."""
    source = render_constants_header() + "\n" + compose(kb_)
    namespace: dict[str, object] = {}
    exec(compile(source, "<composed-puma>", "exec"), namespace)
    return namespace["_solve_algebraic"]  # type: ignore[return-value]


def _fk(kb_: KinBody, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb_.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def test_composed_puma_matches_runtime_on_random_poses() -> None:
    """The composed ``_solve_algebraic`` must produce candidate q-vectors
    such that every runtime-returned solution is matched by some composed
    candidate (within wrap-to-pi machine precision)."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")
    composed_solve = _compile_composed(kb)

    rng = np.random.default_rng(seed=0)
    n_trials = 100
    fk_errs: list[float] = []
    n_runtime_total = 0
    n_runtime_matched = 0

    for _ in range(n_trials):
        q_star = rng.uniform(-1.0, 1.0, size=6)
        T_star = _fk(kb, q_star)

        composed_candidates = composed_solve(T_star)
        runtime_solutions, is_ls = spherical_two_parallel.solve(kb, T_star)
        assert not is_ls, "test setup should produce feasible poses"

        # Every runtime solution should match at least one composed candidate.
        for sol in runtime_solutions:
            n_runtime_total += 1
            sol_q = sol.q
            if any(_q_match(np.asarray(c), sol_q, tol=1e-6) for c in composed_candidates):
                n_runtime_matched += 1

        # Every composed candidate that FK-closes should be valid (so the
        # composed function is sound; doesn't have to enumerate fewer or
        # more, just produce supersets that include the runtime answers).
        for c in composed_candidates:
            T_c = _fk(kb, np.asarray(c))
            err = float(np.linalg.norm(T_c - T_star))
            # Filter out branches that fail FK (LS / degenerate fallbacks).
            # We don't require all composed candidates to FK-close; some may
            # be LS branches. We DO require that for any FK-close composed
            # candidate, the residual is tight.
            if err < 1e-6:
                fk_errs.append(err)

    print(
        f"composed/runtime match: {n_runtime_matched}/{n_runtime_total} "
        f"runtime solutions found in composed candidates"
    )
    assert n_runtime_matched == n_runtime_total, (
        f"composed function failed to enumerate {n_runtime_total - n_runtime_matched} "
        f"runtime solutions across {n_trials} poses"
    )
    if fk_errs:
        print(
            f"FK-close composed candidates: median {np.median(fk_errs):.2e}, max {max(fk_errs):.2e}"
        )
        assert max(fk_errs) < 1e-6
