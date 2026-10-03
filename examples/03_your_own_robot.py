"""Example 03: your own robot -- from a URDF to a deployable IK module.

Any serial 6R or 7R arm can be loaded from its URDF. This example uses a URDF
written below, a 6R arm with the Kinova JACO 2's geometry (Kinova's published
DH table for the 60-degree wrist). That wrist makes it **non-Pieper**: no three
consecutive joint axes meet at a point or are parallel, so no closed-form
subproblem decomposition exists and EAIK-style solvers refuse it. ssik solves
it with a Raghavan-Roth elimination (``ikgeo.general_6r``) and returns every
branch at machine-precision FK closure.

That elimination is derived symbolically for each arm, and the derivation
takes from seconds to about a minute. ``Manipulator.from_urdf(...).solve()`` pays
it on its first call in every new process. ``ssik build`` pays it once and
bakes the result into a Python module you import. This example:

1. Loads the URDF with ``Manipulator.from_urdf`` and shows the dispatch.
2. Runs ``ssik build`` on it (the one-time cost, measured) and imports the
   emitted ``my_arm_ik`` module.
3. Solves with it: a pose ``q*`` inside the limits is recovered, every branch
   closes FK, and the branches outside the joint limits are filtered, with the
   reason for each.

Pieper-class arms (UR, Puma, most industrial 6R, SRS 7R) need no derivation:
``from_urdf`` solves them in closed form from the first call.

Needs ``pip install 'ssik[urdf]'`` (included in ``ssik[demo]``)::

    python examples/03_your_own_robot.py           # about 10 s, most of it ssik build
    python examples/03_your_own_robot.py --live    # also time the live first solve

The script checks its own claims at the end and exits non-zero if one fails.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import ssik

# The JACO 2's DH geometry (Kinova, 6-DOF curved wrist), written as URDF joint
# origins: each joint's origin is the previous link's DH transform
# Trans_z(d) Trans_x(a) Rot_x(alpha); tool0 sits on the last joint's axis, at
# the last link's offset d. Joints 2 and 3 carry illustrative limits (the
# ranges the JACO 2 gives its shoulder and elbow); the rest are continuous.
MY_ARM_URDF = """\
<robot name="my_arm">
  <link name="base_link"/>
  <link name="link_1"/> <link name="link_2"/> <link name="link_3"/>
  <link name="link_4"/> <link name="link_5"/> <link name="link_6"/>
  <link name="tool0"/>
  <joint name="joint_1" type="continuous">
    <parent link="base_link"/> <child link="link_1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/> <axis xyz="0 0 1"/>
  </joint>
  <joint name="joint_2" type="revolute">
    <parent link="link_1"/> <child link="link_2"/>
    <origin xyz="0 0 0.2755" rpy="1.5707963267948966 0 0"/> <axis xyz="0 0 1"/>
    <limit lower="0.820305" upper="5.46288" effort="40" velocity="0.6"/>
  </joint>
  <joint name="joint_3" type="revolute">
    <parent link="link_2"/> <child link="link_3"/>
    <origin xyz="0.41 0 0" rpy="3.141592653589793 0 0"/> <axis xyz="0 0 1"/>
    <limit lower="0.331613" upper="5.95157" effort="40" velocity="0.6"/>
  </joint>
  <joint name="joint_4" type="continuous">
    <parent link="link_3"/> <child link="link_4"/>
    <origin xyz="0 0 -0.0098" rpy="1.5707963267948966 0 0"/> <axis xyz="0 0 1"/>
  </joint>
  <joint name="joint_5" type="continuous">
    <parent link="link_4"/> <child link="link_5"/>
    <origin xyz="0 0 -0.25008" rpy="1.0471975511965976 0 0"/> <axis xyz="0 0 1"/>
  </joint>
  <joint name="joint_6" type="continuous">
    <parent link="link_5"/> <child link="link_6"/>
    <origin xyz="0 0 -0.08556" rpy="1.0471975511965976 0 0"/> <axis xyz="0 0 1"/>
  </joint>
  <joint name="flange" type="fixed">
    <parent link="link_6"/> <child link="tool0"/>
    <origin xyz="0 0 -0.20278" rpy="0 0 0"/>
  </joint>
</robot>
"""

Q_STAR = np.array([0.4, 2.2, 2.0, 1.0, -0.8, 0.3])  # inside the limits above


def wrapped(a: np.ndarray, b: np.ndarray) -> float:
    """Largest per-joint difference between two configurations, on the circle."""
    d = np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi
    return float(np.max(np.abs(d)))


def outside_limits(q: np.ndarray, limits: list[tuple[float, float] | None]) -> list[str]:
    """The joints of ``q`` with no in-limit representative ``q_i + 2*pi*k``."""
    bad = []
    for i, (qi, lim) in enumerate(zip(q, limits, strict=True)):
        if lim is None:
            continue
        lo, hi = lim
        k = np.ceil((lo - qi) / (2 * np.pi))  # the first turn at or above lo
        if qi + 2 * np.pi * k > hi:
            bad.append(f"joint {i + 1} = {qi:.3f} (limits [{lo:.3f}, {hi:.3f}])")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live",
        action="store_true",
        help="also time Manipulator.from_urdf's first solve (derives the elimination again)",
    )
    args = parser.parse_args()
    checks: list[tuple[str, bool]] = []

    with tempfile.TemporaryDirectory() as tmp:
        urdf = Path(tmp) / "my_arm.urdf"
        urdf.write_text(MY_ARM_URDF)

        # -------------------------------------------------------------------
        # 1. Load the chain from base_link to tool0. Without base= and ee=,
        #    from_urdf picks the longest actuated chain and stops at its last
        #    joint (link_6 here), skipping trailing fixed frames such as tool0.
        # -------------------------------------------------------------------
        arm = ssik.Manipulator.from_urdf(urdf, base="base_link", ee="tool0")
        print(arm)
        print(f"  solver {arm.solver_name}: no closed form for this arm, so the first")
        print("  solve derives a Raghavan-Roth elimination for its geometry.\n")
        checks.append(("the arm dispatches to general_6r", arm.solver_name == "ikgeo.general_6r"))

        # -------------------------------------------------------------------
        # 2. ssik build: the one-time derivation, baked into a module.
        # -------------------------------------------------------------------
        out = Path(tmp) / "my_arm_ik.py"
        flags = ["--base", "base_link", "--ee", "tool0"]
        print(f"$ ssik build my_arm.urdf {' '.join(flags)} --out my_arm_ik.py")
        print("  (this derives the elimination for this arm: expect up to a minute, once)")
        t0 = time.perf_counter()
        # ``python -m ssik.cli`` is the ``ssik`` console script, run with this
        # interpreter so the example does not depend on PATH.
        proc = subprocess.run(
            [sys.executable, "-m", "ssik.cli", "build", str(urdf), *flags, "--out", str(out)],
            capture_output=True,
            text=True,
        )
        build_s = time.perf_counter() - t0
        for line in proc.stdout.splitlines():
            if any(key in line for key in ("Best solver", "Wrote", "poses solved", "FAILED")):
                print("  " + line.strip())
        print(f"  ssik build took {build_s:.1f} s\n")
        if proc.returncode != 0:
            print(proc.stdout + proc.stderr)
        checks.append(("ssik build succeeds", proc.returncode == 0 and out.is_file()))
        if proc.returncode != 0:
            return _report(checks)

        sys.path.insert(0, tmp)
        my_arm_ik = importlib.import_module("my_arm_ik")

        # -------------------------------------------------------------------
        # 3. Solve with the built module.
        # -------------------------------------------------------------------
        T = my_arm_ik.fk(Q_STAR)
        t0 = time.perf_counter()
        sols = my_arm_ik.solve(T)
        first_ms = (time.perf_counter() - t0) * 1e3
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            my_arm_ik.solve(T)
            times.append(time.perf_counter() - t0)
        print("import my_arm_ik; my_arm_ik.solve(T):")
        warm_ms = np.median(times) * 1e3
        print(f"  first call in this process {first_ms:.1f} ms; then median {warm_ms:.2f} ms")
        print(f"  q*                         {Q_STAR.tolist()}")
        for i, s in enumerate(sols):
            print(f"  [{i}] q = {np.round(s.q, 3).tolist()}  fk_residual {s.fk_residual:.1e}")
        worst_fk = max(s.fk_residual for s in sols)
        recovered = min(wrapped(s.q, Q_STAR) for s in sols)
        checks.append(("every returned branch closes FK below 1e-9", worst_fk < 1e-9))
        checks.append(("q* is recovered", recovered < 1e-9))

        # Joint limits: the solver's geometric branches, then the limit pass.
        raw = my_arm_ik.solve(T, respect_limits=False)
        dropped = [r for r in raw if outside_limits(r.q, arm.joint_limits)]
        print(f"\n  respect_limits=False: {len(raw)} geometric branches; {len(sols)} are in limits")
        for r in dropped:
            why = "; ".join(outside_limits(r.q, arm.joint_limits))
            print(f"    dropped {np.round(r.q, 3).tolist()}: {why}")
        print("  A finite joint is moved into its range by whole turns where one fits:")
        for s in sols:
            (r,) = [r for r in raw if wrapped(r.q, s.q) < 1e-9]
            for i in np.flatnonzero(np.abs(r.q - s.q) > 1.0):
                print(
                    f"    joint {i + 1} = {r.q[i]:.3f} from the solver is reported as {s.q[i]:.3f}"
                )
        checks.append(
            (
                "the limit pass drops exactly the out-of-limit branches",
                len(sols) == len(raw) - len(dropped),
            )
        )
        checks.append(("some branch is out of limits here", bool(dropped)))

        # -------------------------------------------------------------------
        # The live path, on request: the same answers, after its own derivation.
        # -------------------------------------------------------------------
        if args.live:
            t0 = time.perf_counter()
            live = arm.solve(T)
            live_s = time.perf_counter() - t0
            print(f"\nManipulator.from_urdf(...).solve(T), first call: {live_s:.1f} s")
            gap = max(min(wrapped(a.q, b.q) for b in sols) for a in live)
            print(f"  {len(live)} solutions, each within {gap:.1e} rad of the built module's")
            same = len(live) == len(sols) and gap < 1e-6
            checks.append(("the live solve returns the built module's set", same))
        sys.path.remove(tmp)

    print()
    return _report(checks)


def _report(checks: list[tuple[str, bool]]) -> int:
    for name, ok in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
