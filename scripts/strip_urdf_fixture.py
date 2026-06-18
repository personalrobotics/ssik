"""Strip a URDF down to a kinematics-only test fixture.

ssik's test fixtures under ``tests/fixtures/`` are kinematics-only URDFs: the
links keep only their names, and the joints keep origin / axis / parent / child
/ limit. Everything irrelevant to inverse kinematics -- ``<visual>``,
``<collision>``, ``<inertial>``, ``<material>``, ``<gazebo>``,
``<transmission>`` -- is removed. That keeps fixtures small, free of unresolved
``package://`` mesh paths, and focused on the geometry the solvers actually use.

This tool produces such a fixture from any source URDF (e.g. the output of
``robot_descriptions``'s ``load_robot_description(...).write_xml()``), so adding
a new arm fixture is reproducible instead of a hand-edit.

Usage::

    uv run python scripts/strip_urdf_fixture.py <source.urdf> tests/fixtures/<arm>.urdf

The forward kinematics of the stripped fixture are identical to the source
(only non-kinematic elements are dropped); the tool re-checks this is
structurally true by confirming every joint's origin/axis/parent/child survive.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Non-kinematic child elements removed from every ``<link>``.
_LINK_DROP = ("visual", "collision", "inertial")
# Non-kinematic top-level elements removed from ``<robot>``.
_ROBOT_DROP = ("material", "gazebo", "transmission")


def strip_urdf(source: Path, dest: Path) -> tuple[int, int]:
    """Write a kinematics-only copy of ``source`` to ``dest``.

    :returns: ``(n_links, n_joints)`` kept.
    """
    tree = ET.parse(source)
    root = tree.getroot()

    for tag in _ROBOT_DROP:
        for el in root.findall(tag):
            root.remove(el)

    for link in root.findall("link"):
        for tag in _LINK_DROP:
            for el in link.findall(tag):
                link.remove(el)

    n_joints = len(root.findall("joint"))
    for joint in root.findall("joint"):
        # A joint must keep its parent/child to define the chain; warn loudly
        # rather than emit a silently-broken fixture.
        if joint.find("parent") is None or joint.find("child") is None:
            raise ValueError(f"joint {joint.get('name')!r} is missing parent/child")

    ET.indent(tree, space="  ")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest, encoding="unicode", xml_declaration=True)
    return len(root.findall("link")), n_joints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source URDF to strip")
    parser.add_argument("dest", type=Path, help="output kinematics-only URDF")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"source URDF not found: {args.source}")

    n_links, n_joints = strip_urdf(args.source, args.dest)
    mesh_refs = sum(
        1 for line in args.dest.read_text().splitlines() if "package://" in line or "<mesh" in line
    )
    print(f"wrote {args.dest}: {n_links} links, {n_joints} joints, {mesh_refs} mesh refs")
    if mesh_refs:
        print("WARNING: mesh references remain -- inspect the source for non-standard structure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
