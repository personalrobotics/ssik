"""Strip a URDF down to a kinematics-only test fixture.

ssik's test fixtures under ``tests/fixtures/`` are kinematics-only URDFs: links
keep only their names, joints keep origin / axis / parent / child / limit, and
everything irrelevant to IK (``<visual>``, ``<collision>``, ``<inertial>``,
``<material>``, ``<gazebo>``, ``<transmission>``) is removed. That keeps
fixtures small, free of unresolved ``package://`` mesh paths, and focused on the
geometry the solvers use.

This is a thin CLI over :func:`ssik._urdf.strip_urdf_to_fixture` (shared with
``ssik add-arm``). Forward kinematics of the stripped fixture are identical to
the source -- only non-kinematic elements are dropped.

Usage::

    uv run python scripts/strip_urdf_fixture.py <source.urdf> tests/fixtures/<arm>.urdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from ssik._urdf import strip_urdf_to_fixture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source URDF to strip")
    parser.add_argument("dest", type=Path, help="output kinematics-only URDF")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source URDF not found: {args.source}")

    n_links, n_joints = strip_urdf_to_fixture(args.source, args.dest)
    mesh_refs = sum(
        1 for line in args.dest.read_text().splitlines() if "package://" in line or "<mesh" in line
    )
    print(f"wrote {args.dest}: {n_links} links, {n_joints} joints, {mesh_refs} mesh refs")
    if mesh_refs:
        print("WARNING: mesh references remain -- inspect the source for non-standard structure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
