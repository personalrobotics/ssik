"""End-to-end test for the ``spherical`` composer (SP5 runtime + post-chain inlined)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.codegen._compose.spherical import compose, render_constants_header
from ssik.solvers.ikgeo import spherical
from ssik.subproblems._rotation import rotation_matrix


def _build_synthetic_kb() -> KinBody:
    """Synthetic spherical-wrist arm matching test_ikgeo_spherical's
    `_build_generic_spherical_arm`."""
    tilt = np.deg2rad(25.0)
    axes = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([np.sin(tilt), -np.cos(tilt), 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    d1, a1, a2, a3, d3, d4 = 0.2, 0.04, 0.5, 0.08, -0.12, 0.4
    t_lefts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([a1, 0.0, d1]),
        np.array([a2, 0.0, 0.0]),
        np.array([a3, d3, 0.0]),
        np.array([0.0, 0.0, d4]),
        np.array([0.0, 0.0, 0.0]),
    ]
    link_names = ["base_link", *(f"link_{i}" for i in range(1, 6)), "ee_link"]
    links = [Link(name=n) for n in link_names]
    joints: list[Joint] = []
    for i in range(6):
        t_left_i = np.eye(4)
        t_left_i[:3, 3] = t_lefts[i]
        joints.append(
            Joint(
                name=f"joint_{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=t_left_i,
                T_right=np.eye(4),
                axis=axes[i],
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _q_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    return all(abs(_wrap(float(ai - bi))) < tol for ai, bi in zip(a, b, strict=True))


def _compile_composed(kb: KinBody) -> Callable[[np.ndarray], list[list[float]]]:
    source = render_constants_header() + "\nimport numpy as np\n" + compose(kb)
    namespace: dict[str, object] = {"np": np}
    exec(compile(source, "<composed-sph>", "exec"), namespace)
    return namespace["_solve_algebraic"]  # type: ignore[return-value]


def _fk(kb: KinBody, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def test_composed_synthetic_matches_runtime() -> None:
    kb = _build_synthetic_kb()
    composed_solve = _compile_composed(kb)

    rng = np.random.default_rng(seed=2)
    n_trials = 30
    n_runtime_total = 0
    n_runtime_matched = 0

    for _ in range(n_trials):
        q_star = rng.uniform(-1.0, 1.0, size=6)
        T_star = _fk(kb, q_star)

        composed_candidates = composed_solve(T_star)
        runtime_solutions, is_ls = spherical.solve(kb, T_star)
        if is_ls or not runtime_solutions:
            continue

        for sol in runtime_solutions:
            n_runtime_total += 1
            if any(_q_match(np.asarray(c), sol.q, tol=1e-4) for c in composed_candidates):
                n_runtime_matched += 1

    assert n_runtime_matched == n_runtime_total, (
        f"composed function failed to enumerate {n_runtime_total - n_runtime_matched} "
        f"of {n_runtime_total} runtime solutions across {n_trials} poses"
    )
    assert n_runtime_total > 0
