"""Profile dispatch + IK for the four candidate non-SRS 7R fixtures.

For each of Rizon 4 / Gen3 / xArm7 / Franka (control): build a fixture,
run a 16-sample lock-sweep at the auto-selected lock joint, and report
which inner-solver each sample dispatches to. Then time a default
``solve(T)`` plus ``max_solutions=1``.

Honest classification:

  - **Pieper wedge**: most samples dispatch to a fast tier-0/1 solver
    (spherical_two_parallel / three_parallel / spherical /
    spherical_two_intersecting). IK ~ms-class.

  - **Tier-2 trap**: most samples dispatch to gen_six_dof. IK ~tens of
    seconds.

iiwa is NOT in this script -- it's already known to be a tier-2 trap
until #143 lands. This script answers "do the *other* three 7R fixtures
share iiwa's fate, or do they actually IK fast like Franka?"
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests" / "fixtures"))

from ssik._kinbody import JointSpec, build_kinbody  # noqa: E402
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY  # noqa: E402
from ssik.kinematics.poe_fk import poe_forward_kinematics  # noqa: E402
from ssik.solvers.jointlock import seven_r  # noqa: E402
from ssik.solvers.jointlock.seven_r import (  # noqa: E402
    _DEFAULT_SAMPLES,
    _lock_joint,
    _topology_rank,
    choose_lock_joint,
)


def _quat_wxyz_to_rot(q: tuple[float, float, float, float]) -> NDArray[np.float64]:
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _xform(pos, quat) -> NDArray[np.float64]:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rot(quat)
    T[:3, 3] = pos
    return T


# Inline chain transcriptions from mujoco_menagerie. Each entry:
# (pos, quat_wxyz, axis_xyz, range_lo, range_hi). Quat (1,0,0,0) is identity.

# flexiv_rizon4/flexiv_rizon4.xml
RIZON4 = [
    ((0.0, 0.0, 0.155), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0), -2.8798, 2.8798),
    ((0.0, 0.03, 0.21), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -2.3562, 2.3562),
    ((0.0, 0.035, 0.205), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -3.0543, 3.0543),
    ((-0.02, -0.03, 0.19), (0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0), -1.9548, 2.7751),
    ((-0.02, 0.025, 0.195), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0), -3.0543, 3.0543),
    ((0.0, 0.03, 0.19), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1.4835, 4.6251),
    ((-0.015, 0.073, 0.11), (0.707107, 0.0, -0.707107, 0.0), (0.0, 0.0, 1.0), -3.0543, 3.0543),
]

# kinova_gen3/gen3.xml
GEN3 = [
    ((0.0, 0.0, 0.15643), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -3.14159, 3.14159),
    ((0.0, 0.005375, -0.12838), (1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.24, 2.24),
    ((0.0, -0.21038, -0.006375), (1.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -3.14159, 3.14159),
    ((0.0, 0.006375, -0.21038), (1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.57, 2.57),
    ((0.0, -0.20843, -0.006375), (1.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -3.14159, 3.14159),
    ((0.0, 0.00017505, -0.10593), (1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.09, 2.09),
    ((0.0, -0.10593, -0.00017505), (1.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -3.14159, 3.14159),
]

# ufactory_xarm7/xarm7.xml
XARM7 = [
    ((0.0, 0.0, 0.267), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -6.2832, 6.2832),
    ((0.0, 0.0, 0.0), (1.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.059, 2.0944),
    ((0.0, -0.293, 0.0), (1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -6.2832, 6.2832),
    ((0.0525, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -0.19198, 3.927),
    ((0.0775, -0.3425, 0.0), (1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -6.2832, 6.2832),
    ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -1.69297, 3.14159),
    ((0.076, 0.097, 0.0), (1.0, -1.0, 0.0, 0.0), (0.0, 0.0, 1.0), -6.2832, 6.2832),
]


def _make_specs(chain: list[tuple], name: str) -> list[JointSpec]:
    specs: list[JointSpec] = []
    for i, (pos, quat, axis, lo, hi) in enumerate(chain):
        specs.append(
            JointSpec(
                parent_link_T=_xform(pos, quat),
                axis=np.array(axis, dtype=np.float64),
                joint_type="revolute",
                child_link_T=np.eye(4, dtype=np.float64),
                name=f"{name}_joint{i + 1}",
                limits=(lo, hi),
            )
        )
    return specs


def profile_arm(name: str, chain: list[tuple]) -> None:
    print(f"\n=== {name} ===")
    kb = build_kinbody(_make_specs(chain, name.lower()))
    print(f"  fixture: {len(kb.joints)} joints")

    # Topology dispatch profile across the auto-selected lock joint.
    lock_idx = choose_lock_joint(kb, DEFAULT_TOLERANCE_POLICY)
    print(f"  chosen lock_idx: {lock_idx}")

    joint_lim = kb.joints[lock_idx].limits
    if joint_lim is None:
        lo, hi = -np.pi, np.pi
    else:
        lo, hi = joint_lim
    samples = np.linspace(lo, hi, _DEFAULT_SAMPLES, endpoint=False)
    dispatch: Counter[str] = Counter()
    for q_lock in samples:
        sub = _lock_joint(kb, lock_idx, float(q_lock))
        _, solver_name = _topology_rank(sub, DEFAULT_TOLERANCE_POLICY)
        dispatch[solver_name] += 1
    print(f"  dispatch over {_DEFAULT_SAMPLES} samples:")
    for solver_name, count in sorted(dispatch.items(), key=lambda x: -x[1]):
        print(f"    {count:>3}x  {solver_name}")

    # Classify wedge vs trap.
    fast_count = sum(n for s, n in dispatch.items() if "gen_six_dof" not in s)
    trap_count = dispatch.get("gen_six_dof", 0) + dispatch.get("reversed:gen_six_dof", 0)
    if fast_count >= trap_count:
        print(
            f"  classification: PIEPER WEDGE "
            f"({fast_count}/{_DEFAULT_SAMPLES} samples land in fast solvers)"
        )
    else:
        print(
            f"  classification: TIER-2 TRAP "
            f"({trap_count}/{_DEFAULT_SAMPLES} samples fall through to gen_six_dof)"
        )

    # Quick IK timing -- only run if classification says wedge, otherwise skip
    # to save time (tier-2 takes ~30s/IK).
    if fast_count >= trap_count:
        rng = np.random.default_rng(42)
        q_star = rng.uniform(-0.8, 0.8, size=7)
        T = poe_forward_kinematics(kb, q_star)

        # Default solve
        try:
            t0 = time.perf_counter()
            sols, is_ls = seven_r.solve(kb, T)
            t_default = (time.perf_counter() - t0) * 1000
            print(f"  default solve: {len(sols)} sols, is_ls={is_ls}, {t_default:.1f} ms")
            if sols:
                err = float(np.max(np.abs(poe_forward_kinematics(kb, sols[0].q) - T)))
                print(f"    FK closure on first sol: {err:.2e}")
        except Exception as e:
            print(f"  default solve raised: {e!r}")

        # max_solutions=1
        try:
            t0 = time.perf_counter()
            sols1, is_ls1 = seven_r.solve(kb, T, max_solutions=1)
            t_max1 = (time.perf_counter() - t0) * 1000
            print(f"  max_solutions=1: {len(sols1)} sols, is_ls={is_ls1}, {t_max1:.1f} ms")
        except Exception as e:
            print(f"  max_solutions=1 raised: {e!r}")
    else:
        print("  skipping IK timing (tier-2 trap, would take >30 s per IK)")


def main() -> None:
    print("Profiling non-SRS 7R candidates: which arms IK fast vs fall through to tier-2?\n")
    print("Per #80: classify each as Pieper wedge (fast inner dispatches) or tier-2 trap (")
    print("most samples land in gen_six_dof, IK ~tens of seconds per call).")

    # Franka 7R as the control -- known fast wedge.
    sys.path.insert(0, str(REPO / "tests" / "fixtures"))
    from franka_panda import franka_panda_specs  # type: ignore[import-not-found]

    print("\n=== Franka Panda (control, known wedge) ===")
    fkb = build_kinbody(franka_panda_specs())
    flock = choose_lock_joint(fkb, DEFAULT_TOLERANCE_POLICY)
    fjoint_lim = fkb.joints[flock].limits
    flo, fhi = fjoint_lim if fjoint_lim is not None else (-np.pi, np.pi)
    fsamples = np.linspace(flo, fhi, _DEFAULT_SAMPLES, endpoint=False)
    fdispatch: Counter[str] = Counter()
    for q_lock in fsamples:
        sub = _lock_joint(fkb, flock, float(q_lock))
        _, solver_name = _topology_rank(sub, DEFAULT_TOLERANCE_POLICY)
        fdispatch[solver_name] += 1
    print(f"  chosen lock_idx: {flock}")
    print(f"  dispatch over {_DEFAULT_SAMPLES} samples:")
    for solver_name, count in sorted(fdispatch.items(), key=lambda x: -x[1]):
        print(f"    {count:>3}x  {solver_name}")

    profile_arm("Rizon 4", RIZON4)
    profile_arm("Gen3", GEN3)
    profile_arm("xArm7", XARM7)


if __name__ == "__main__":
    main()
