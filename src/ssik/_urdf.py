"""URDF → KinBody adapter.

Loads a URDF via `urchin` and builds a :class:`ssik._kinbody.KinBody` for
the chain from ``base_link`` to ``ee_link``. Fixed joints in the chain are
fused into the adjacent active joint's ``T_left`` (or the previous active
joint's ``T_right`` if they trail the last active joint).

``urchin`` is imported lazily so the rest of the package stays installable
without it; loading a URDF without the ``urdf`` extra raises a clear message.

Private for now; the public :class:`Manipulator.from_urdf` entry point (#12)
will wrap this.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import Joint, JointType, KinBody, Link, build_poe_kinbody

if TYPE_CHECKING:  # pragma: no cover — typing only
    from urchin import Joint as UrchinJoint

__all__ = ["load_urdf_kinbody", "load_urdf_kinbody_normalized", "strip_urdf_to_fixture"]

# Non-kinematic elements stripped when vendoring a URDF as a test fixture:
# everything irrelevant to inverse kinematics. Keeps fixtures small and free of
# unresolved ``package://`` mesh paths.
_FIXTURE_LINK_DROP = ("visual", "collision", "inertial")
_FIXTURE_ROBOT_DROP = ("material", "gazebo", "transmission")


def strip_urdf_to_fixture(source: Path, dest: Path) -> tuple[int, int]:
    """Write a kinematics-only copy of ``source`` to ``dest``.

    Drops ``<visual>`` / ``<collision>`` / ``<inertial>`` from every link and
    top-level ``<material>`` / ``<gazebo>`` / ``<transmission>``, keeping link
    names and the full joint definitions (origin / axis / parent / child /
    limit) -- so forward kinematics are identical, but the file is small and has
    no unresolved mesh paths. Used to vendor ``tests/fixtures/*.urdf``.

    :returns: ``(n_links, n_joints)`` kept.
    :raises ValueError: if any joint is missing its ``parent``/``child`` (which
        would yield a silently-broken chain).
    """
    tree = ET.parse(source)
    root = tree.getroot()
    for tag in _FIXTURE_ROBOT_DROP:
        for el in root.findall(tag):
            root.remove(el)
    for link in root.findall("link"):
        for tag in _FIXTURE_LINK_DROP:
            for el in link.findall(tag):
                link.remove(el)
    for joint in root.findall("joint"):
        if joint.find("parent") is None or joint.find("child") is None:
            raise ValueError(f"joint {joint.get('name')!r} is missing parent/child")
    ET.indent(tree, space="  ")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest, encoding="unicode", xml_declaration=True)
    return len(root.findall("link")), len(root.findall("joint"))


_IDENTITY: NDArray[np.float64] = np.eye(4, dtype=np.float64)


def _import_urchin() -> object:
    try:
        import urchin
    except ImportError as err:
        raise ImportError(
            "URDF loading requires the optional 'urdf' extra: "
            "`pip install ssik[urdf]` (or `uv add urchin`)."
        ) from err
    return urchin


def _walk_chain(urdf: object, base_link: str, ee_link: str) -> list[UrchinJoint]:
    """Return the ordered list of URDF joints from ``base_link`` to ``ee_link``,
    walking the public ``joints`` list (no private-graph internals).
    """
    joints = urdf.joints  # type: ignore[attr-defined]
    link_map = urdf.link_map  # type: ignore[attr-defined]
    if base_link not in link_map:
        raise ValueError(f"URDF has no link named {base_link!r}")
    if ee_link not in link_map:
        raise ValueError(f"URDF has no link named {ee_link!r}")

    child_to_joint = {j.child: j for j in joints}
    path: list[UrchinJoint] = []
    cur = ee_link
    visited: set[str] = set()
    while cur != base_link:
        if cur in visited:
            raise ValueError(f"URDF contains a cycle reaching link {cur!r}")
        visited.add(cur)
        joint = child_to_joint.get(cur)
        if joint is None:
            raise ValueError(
                f"no joint leads to link {cur!r}; chain from {ee_link!r} to {base_link!r} is broken"
            )
        path.append(joint)
        cur = joint.parent
    path.reverse()  # base → EE order
    return path


def _map_joint_type(urdf_type: str, name: str) -> JointType:
    if urdf_type in ("revolute", "continuous"):
        return "revolute"
    if urdf_type == "prismatic":
        return "prismatic"
    if urdf_type in ("planar", "floating"):
        raise NotImplementedError(
            f"joint {name!r}: URDF type {urdf_type!r} is not supported by the "
            "kinbody shim (only revolute, continuous, prismatic, fixed are allowed)."
        )
    raise ValueError(f"joint {name!r}: unknown URDF joint type {urdf_type!r}")


def load_urdf_kinbody(
    urdf_path: str | Path,
    base_link: str,
    ee_link: str,
    *,
    lazy_load_meshes: bool = True,
) -> KinBody:
    """Load a URDF file and build a :class:`KinBody` for the chain between
    ``base_link`` and ``ee_link``.

    Active joints (revolute / continuous / prismatic) become :class:`Joint`s in
    the kinbody. Fixed joints are fused into the adjacent active joint's
    ``T_left``, or into the previous active joint's ``T_right`` if they trail
    the last active joint. Mimic joints and planar/floating joints are not
    supported and raise.

    :param lazy_load_meshes: forwarded to ``urchin.URDF.load``; default ``True``
        since we don't need mesh geometry for kinematics.
    """
    urchin = _import_urchin()
    urdf = urchin.URDF.load(str(urdf_path), lazy_load_meshes=lazy_load_meshes)  # type: ignore[attr-defined]

    chain = _walk_chain(urdf, base_link, ee_link)

    links: list[Link] = [Link(name=base_link)]
    joints: list[Joint] = []
    pending_fixed: NDArray[np.float64] = _IDENTITY.copy()
    dof_index = 0

    for uj in chain:
        if uj.mimic is not None:
            raise NotImplementedError(
                f"joint {uj.name!r} mimics joint {uj.mimic.joint!r}; "
                "mimic joints are not supported by the kinbody shim."
            )
        if uj.joint_type == "fixed":
            pending_fixed = pending_fixed @ np.asarray(uj.origin, dtype=np.float64)
            continue

        joint_type = _map_joint_type(uj.joint_type, uj.name)
        T_left = pending_fixed @ np.asarray(uj.origin, dtype=np.float64)
        pending_fixed = _IDENTITY.copy()
        if uj.joint_type == "continuous" or uj.limit is None:
            limits: tuple[float, float] | None = None
        else:
            limits = (float(uj.limit.lower), float(uj.limit.upper))

        child_link = Link(name=uj.child)
        joints.append(
            Joint(
                name=uj.name,
                dof_index=dof_index,
                parent_link=links[-1],
                T_left=T_left,
                T_right=_IDENTITY.copy(),
                axis=np.asarray(uj.axis, dtype=np.float64),
                joint_type=joint_type,
                limits=limits,
            )
        )
        links.append(child_link)
        dof_index += 1

    if not joints:
        raise ValueError(
            f"chain from {base_link!r} to {ee_link!r} contains no active joints (all fixed)"
        )

    # Trailing fixed joints after the last active joint: bake into T_right.
    if not np.allclose(pending_fixed, _IDENTITY):
        last = joints[-1]
        joints[-1] = Joint(
            name=last.name,
            dof_index=last.dof_index,
            parent_link=last.parent_link,
            T_left=last.T_left,
            T_right=last.T_right @ pending_fixed,
            axis=last.axis,
            joint_type=last.joint_type,
            limits=last.limits,
        )

    # The last link we appended carries the active-chain's child name; rename
    # it to the user-supplied ee_link to preserve URDF semantics.
    links[-1] = Link(name=ee_link)
    # Rewire the last joint's reference since we just replaced the Link object.
    # (Joint.parent_link points at the *previous* link in the chain — the
    # replacement is the final link, not a parent — so no rewire needed.)

    return KinBody(links=links, joints=joints)


def load_urdf_kinbody_normalized(
    urdf_path: str | Path,
    base_link: str,
    ee_link: str,
    *,
    lazy_load_meshes: bool = True,
) -> KinBody:
    """Load a URDF and build a **POE-normalized** :class:`KinBody` for the
    chain between ``base_link`` and ``ee_link``.

    The resulting chain is kinematically identical to :func:`load_urdf_kinbody`
    (same FK at every ``q``) but uses a different internal encoding that
    exposes the robot's kinematic structure to the topology dispatcher:

    - Each active joint's ``T_left`` is a **pure translation** (no rotation).
    - Each active joint's ``axis`` is expressed in the **base frame at q=0**.
    - ``T_right`` is identity for all but the last active joint; the last
      one absorbs any trailing fixed-joint offset **and** the cumulative
      home-pose orientation so that FK(q=0) exactly matches the URDF.

    This is the :ref:`POE (product-of-exponentials) encoding
    <https://en.wikipedia.org/wiki/Product_of_exponentials_formula>`_
    flattened into our ``T_left @ R(axis, q) @ T_right`` chain format.

    **Why this matters.** Topology predicates like
    :func:`ssik.kinematics.predicates.three_consecutive_parallel` and
    :func:`ssik.kinematics.predicates.three_consecutive_intersecting`
    inspect joint axes *in each joint's local frame*. With URDF's native
    encoding, joint axes live in frames that accumulate arbitrary ``rpy``
    rotations from upstream joint origins, so structural patterns like
    "three parallel axes" (UR5) are hidden behind
    symbolic rotation products that sympy can't simplify into recognizable
    form. POE-normalizing the chain puts axes directly in the base frame at
    q=0, making the structure visible. See #33 for the full analysis.

    :param lazy_load_meshes: forwarded to ``urchin.URDF.load``.
    """
    urchin = _import_urchin()
    urdf = urchin.URDF.load(str(urdf_path), lazy_load_meshes=lazy_load_meshes)  # type: ignore[attr-defined]

    chain = _walk_chain(urdf, base_link, ee_link)

    # Accumulate base-frame rotation and position as we walk through joint
    # origins. At q=0 every joint's rotation is identity, so only the origin
    # transforms contribute.
    r_cum = np.eye(3, dtype=np.float64)
    pos_cum = np.zeros(3, dtype=np.float64)
    prev_active_pos = np.zeros(3, dtype=np.float64)

    # Per-active-joint records: (name, joint_type, axis_base, P_offset, limits).
    # ``limits`` is ``None`` for URDF ``continuous`` joints (free rotation, no
    # range) and for any joint without a ``<limit>`` element. Otherwise it's
    # ``(lower, upper)`` from urchin.
    records: list[
        tuple[
            str,
            JointType,
            NDArray[np.float64],
            NDArray[np.float64],
            tuple[float, float] | None,
        ]
    ] = []

    for uj in chain:
        if uj.mimic is not None:
            raise NotImplementedError(
                f"joint {uj.name!r} mimics joint {uj.mimic.joint!r}; "
                "mimic joints are not supported by the kinbody shim."
            )

        origin = np.asarray(uj.origin, dtype=np.float64)
        r_origin = origin[:3, :3]
        t_origin = origin[:3, 3]

        # Advance base-frame cumulative state through this joint's origin.
        pos_cum = pos_cum + r_cum @ t_origin
        r_cum = r_cum @ r_origin

        if uj.joint_type == "fixed":
            # Fixed joints don't emit records; their origin already folded in.
            continue

        joint_type = _map_joint_type(uj.joint_type, uj.name)
        axis_base = r_cum @ np.asarray(uj.axis, dtype=np.float64)
        p_offset = pos_cum - prev_active_pos
        # Continuous URDF joints (free rotation) have no <limit>; urchin
        # returns ``j.limit = None``. Same handling for any joint with no
        # explicit limit element. Limited joints get ``(lower, upper)``.
        if uj.joint_type == "continuous" or uj.limit is None:
            limits = None
        else:
            limits = (float(uj.limit.lower), float(uj.limit.upper))
        records.append((uj.name, joint_type, axis_base, p_offset, limits))
        prev_active_pos = pos_cum.copy()

    if not records:
        raise ValueError(
            f"chain from {base_link!r} to {ee_link!r} contains no active joints (all fixed)"
        )

    # Any trailing fixed-joint offset + the home orientation live in the
    # final active joint's T_right. The "final offset" is the translation
    # from the last active joint to the EE; the home rotation is r_cum.
    p_final = pos_cum - prev_active_pos
    r_ee_home = r_cum.copy()
    final_t_right = np.eye(4, dtype=np.float64)
    final_t_right[:3, 3] = p_final
    final_rotation = np.eye(4, dtype=np.float64)
    final_rotation[:3, :3] = r_ee_home
    final_t_right = final_t_right @ final_rotation

    return build_poe_kinbody(records, final_t_right, base_link, ee_link)
