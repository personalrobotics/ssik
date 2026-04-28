"""Per-IK timing for ssik.solvers.ikgeo.spherical on a synthetic
spherical-wrist 6R arm (no parallel/intersecting shoulder specialization).

Same arm as test_ikgeo_spherical.py's _build_generic_spherical_arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _bench_lib import run_bench

from ssik._kinbody import Joint, KinBody, Link
from ssik.solvers.ikgeo import spherical


def _build_generic_spherical_arm(tilt_deg: float = 25.0) -> KinBody:
    tilt = np.deg2rad(tilt_deg)
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


run_bench(
    solver_label="ikgeo.spherical (synthetic, generic spherical wrist)",
    solver_call=lambda kb, T: spherical.solve(kb, T),
    kb=_build_generic_spherical_arm(),
    n_dof=6,
)
