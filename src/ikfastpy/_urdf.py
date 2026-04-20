"""URDF → KinBody adapter.

Loads a URDF via `urchin` and builds a :class:`ikfastpy._kinbody.KinBody` for
the chain from ``base_link`` to ``ee_link``. Fixed joints in the chain are
fused into the adjacent active joint's ``T_left`` (or the previous active
joint's ``T_right`` if they trail the last active joint).

``urchin`` is imported lazily so the rest of the package stays installable
without it; loading a URDF without the ``urdf`` extra raises a clear message.

Private for now; the public :class:`Manipulator.from_urdf` entry point (#12)
will wrap this.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ikfastpy._kinbody import Joint, JointType, KinBody, Link

if TYPE_CHECKING:  # pragma: no cover — typing only
    from urchin import Joint as UrchinJoint

__all__ = ["load_urdf_kinbody"]

_IDENTITY: NDArray[np.float64] = np.eye(4, dtype=np.float64)


def _import_urchin() -> object:
    try:
        import urchin
    except ImportError as err:
        raise ImportError(
            "URDF loading requires the optional 'urdf' extra: "
            "`pip install ikfastpy[urdf]` (or `uv add urchin`)."
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
        )

    # The last link we appended carries the active-chain's child name; rename
    # it to the user-supplied ee_link to preserve URDF semantics.
    links[-1] = Link(name=ee_link)
    # Rewire the last joint's reference since we just replaced the Link object.
    # (Joint.parent_link points at the *previous* link in the chain — the
    # replacement is the final link, not a parent — so no rewire needed.)

    return KinBody(links=links, joints=joints)
