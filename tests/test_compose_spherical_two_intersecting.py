"""End-to-end test for the ``spherical_two_intersecting`` composer.

Compiles the rendered ``_solve_algebraic`` source on the synthetic
two-intersecting-shoulder fixture from test_spherical_two_intersecting.py,
runs it against the runtime ``ikgeo.spherical_two_intersecting.solve`` on
50 random target poses, and asserts every runtime solution is matched by a
composed candidate (within wrap-to-pi machine precision).

This is the bulletproof gate at the composer layer for the SP3+SP2+SP4+SP1
chain: if disagreement on any input, the artifact is unsound.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.codegen._compose.spherical_two_intersecting import (
    compose,
    render_constants_header,
)
from ssik.solvers.ikgeo import spherical_two_intersecting
from ssik.subproblems._rotation import rotation_matrix


def _build_synthetic_kb() -> KinBody:
    """Synthetic 6R arm: spherical wrist at (3, 4, 5), p[1] = 0
    (joints 0, 1 share an origin). Mirrors the
    `synthetic_spherical_two_intersecting_kb` fixture."""
    a2, a3, d4 = 0.45, 0.06, 0.38
    tilt_theta = np.deg2rad(15.0)
    axes = [
        np.array([0.0, 0.0, 1.0]),
        np.array([np.sin(tilt_theta), -np.cos(tilt_theta), 0.0]),
        np.array([np.sin(tilt_theta), -np.cos(tilt_theta), 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    t_lefts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([a2, 0.0, 0.0]),
        np.array([a3, 0.0, 0.0]),
        np.array([0.0, 0.0, d4]),
        np.array([0.0, 0.0, 0.0]),
    ]
    link_names = ["base_link", *(f"link_{i}" for i in range(1, 6)), "ee_link"]
    links = [Link(name=n) for n in link_names]
    joints: list[Joint] = []
    for i in range(6):
        t_left_i = np.eye(4)
        t_left_i[:3, 3] = t_lefts[i]
        t_right_i = np.eye(4)
        joints.append(
            Joint(
                name=f"joint_{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=t_left_i,
                T_right=t_right_i,
                axis=axes[i],
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _q_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> bool:
    return all(abs(_wrap(float(ai - bi))) < tol for ai, bi in zip(a, b, strict=True))


def _compile_composed(kb: KinBody) -> Callable[[np.ndarray], list[list[float]]]:
    source = render_constants_header() + "\n" + compose(kb)
    namespace: dict[str, object] = {}
    exec(compile(source, "<composed-sti>", "exec"), namespace)
    return namespace["_solve_algebraic"]  # type: ignore[return-value]


def _fk(kb: KinBody, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def test_composed_synthetic_matches_runtime() -> None:
    """The composed _solve_algebraic must produce candidates that include
    every runtime solution (modulo wrap-to-pi)."""
    kb = _build_synthetic_kb()
    composed_solve = _compile_composed(kb)

    rng = np.random.default_rng(seed=0)
    n_trials = 50
    n_runtime_total = 0
    n_runtime_matched = 0

    for _ in range(n_trials):
        q_star = rng.uniform(-1.0, 1.0, size=6)
        T_star = _fk(kb, q_star)

        composed_candidates = composed_solve(T_star)
        runtime_solutions, is_ls = spherical_two_intersecting.solve(kb, T_star)
        if is_ls or not runtime_solutions:
            continue

        for sol in runtime_solutions:
            n_runtime_total += 1
            sol_q = sol.q
            if any(_q_match(np.asarray(c), sol_q, tol=1e-6) for c in composed_candidates):
                n_runtime_matched += 1

    assert n_runtime_matched == n_runtime_total, (
        f"composed function failed to enumerate {n_runtime_total - n_runtime_matched} "
        f"of {n_runtime_total} runtime solutions across {n_trials} poses"
    )
    assert n_runtime_total > 50, (
        f"too few runtime solutions ({n_runtime_total}) for a meaningful test; "
        f"check fixture and trial count"
    )
