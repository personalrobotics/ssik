"""Per-IK timing for ssik.solvers.ikgeo.two_parallel on an IK-Geo-style
random 6R arm with axes[2] = axes[1] (the two-parallel constraint).

Tier-1 univariate-search solver -- expect higher per-IK time than tier-0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _bench_lib import run_bench

from ssik._kinbody import Joint, KinBody, Link
from ssik.solvers.ikgeo import two_parallel


def _build_random_two_parallel(seed: int = 7) -> KinBody:
    rng = np.random.default_rng(seed)

    def _rnorm() -> np.ndarray:
        v = rng.standard_normal(3)
        return v / float(np.linalg.norm(v))

    axes = [_rnorm() for _ in range(6)]
    axes[2] = axes[1].copy()
    t_lefts = [rng.standard_normal(3) for _ in range(6)]
    tool_p = rng.standard_normal(3)

    link_names = ["base_link", *(f"link_{i}" for i in range(1, 6)), "ee_link"]
    links = [Link(name=n) for n in link_names]
    joints: list[Joint] = []
    for i in range(6):
        t_left_i = np.eye(4)
        t_left_i[:3, 3] = t_lefts[i]
        t_right_i = np.eye(4)
        if i == 5:
            t_right_i[:3, 3] = tool_p
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


run_bench(
    solver_label="ikgeo.two_parallel (random IK-Geo TwoParallelSetup)",
    solver_call=lambda kb, T: two_parallel.solve(kb, T),
    kb=_build_random_two_parallel(),
    n_dof=6,
    n_poses=50,
)
