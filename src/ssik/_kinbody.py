"""Linear kinematic chain (POE-form) consumed by every ssik solver.

A :class:`KinBody` is a flat list of :class:`Joint` objects bracketed by
:class:`Link` objects. Each joint carries the per-joint POE factorisation:

* ``T_left`` — 4x4 rigid transform from the parent link's frame to the
  joint frame.
* ``axis`` — unit 3-vector in the joint frame (post-``T_left``).
* ``T_right`` — 4x4 rigid transform from the joint frame to the child
  link's frame (typically identity except on the last joint, where it
  carries any tool offset).
* ``joint_type`` — ``"revolute"`` or ``"prismatic"``.

The forward kinematics for joint ``i`` is
``T_left @ R_axis(joint.axis, q) @ T_right`` for revolute, with the
analogous translation for prismatic.

Input is a flat list of :class:`JointSpec` (one per joint) passed to
:func:`build_kinbody`, which assembles ``N + 1`` links bracketing the
``N`` joints and returns a :class:`KinBody`. URDF and MJCF loaders
build the spec list; users can also build it by hand for prototype
geometries.

This module is private. The public solver entry points
(``ikgeo.general_6r.solve``, etc.) accept :class:`KinBody` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

__all__ = ["Joint", "JointSpec", "KinBody", "Link", "build_kinbody"]

JointType = Literal["revolute", "prismatic"]


@dataclass(frozen=True)
class JointSpec:
    """User-facing description of a single joint in a linear chain.

    The joint's total transform (at joint value ``q``) is
    ``parent_link_T @ R(q) @ child_link_T``, where ``R(q)`` is a rotation about
    ``axis`` for revolute joints, or a translation along ``axis`` for prismatic.
    ``child_link_T`` defaults to identity, which is enough for chains expressed
    purely in terms of "where the next joint sits" (e.g., simple axis lists).
    It is optional but necessary to express classical DH (where the ``d``/``a``
    translations happen *after* the joint rotation).

    ``limits`` is an optional ``(lo, hi)`` pair giving the joint's reachable
    range (radians for revolute, metres for prismatic). ``None`` means
    unconstrained / unspecified -- ssik solvers don't filter by limits
    inside the kernel (that lives in :mod:`ssik.postprocess`); the limits
    are kinematic data downstream code can consume. The single in-kernel
    consumer is :func:`ssik.solvers.jointlock.seven_r.solve`, which
    clamps the default ``lock_samples`` sweep to ``[lo, hi]`` when
    available so we don't waste samples outside the joint's range.
    """

    parent_link_T: NDArray[np.float64]
    axis: NDArray[np.float64]
    joint_type: JointType
    child_link_T: NDArray[np.float64] | None = None
    name: str | None = None
    limits: tuple[float, float] | None = None


@dataclass(eq=False)
class Link:
    """Rigid body in the kinematic chain.

    Equality is by name so that ``joint.GetHierarchyParentLink() == chainlinks[i]``
    works across object identities. (Solvers compare links by name when
    reasoning about which joint connects which links in a topology.)
    """

    name: str

    def GetName(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Link):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(eq=False)
class Joint:
    """Single-DOF revolute or prismatic joint.

    Multi-axis joints (spherical, universal) are not supported — ``iaxis`` must
    be ``0``. Mimic/passive joints are not supported either; the relevant stubs
    raise if the solver tries to treat this joint as mimic.

    ``limits`` mirrors :attr:`JointSpec.limits` -- an ``(lo, hi)`` pair or
    ``None`` for unconstrained / unspecified. Solvers don't filter by limits
    inside the kernel; downstream code in :mod:`ssik.postprocess` does.
    """

    name: str
    dof_index: int
    parent_link: Link
    T_left: NDArray[np.float64]
    T_right: NDArray[np.float64]
    axis: NDArray[np.float64]
    joint_type: JointType
    limits: tuple[float, float] | None = None

    def _check_iaxis(self, iaxis: int) -> None:
        if iaxis != 0:
            raise ValueError(f"joint {self.name!r} is single-DOF; iaxis must be 0, got {iaxis}")

    def GetName(self) -> str:
        return self.name

    def GetDOF(self) -> int:
        return 1

    def GetDOFIndex(self) -> int:
        return self.dof_index

    def IsStatic(self) -> bool:
        return False

    def IsRevolute(self, iaxis: int) -> bool:
        self._check_iaxis(iaxis)
        return self.joint_type == "revolute"

    def IsPrismatic(self, iaxis: int) -> bool:
        self._check_iaxis(iaxis)
        return self.joint_type == "prismatic"

    def IsMimic(self, iaxis: int) -> bool:
        self._check_iaxis(iaxis)
        return False

    def GetMimicEquation(self, iaxis: int) -> object:
        raise NotImplementedError("mimic joints are not supported by this shim")

    def GetHierarchyParentLink(self) -> Link:
        return self.parent_link

    def GetInternalHierarchyLeftTransform(self) -> NDArray[np.float64]:
        return self.T_left.copy()

    def GetInternalHierarchyRightTransform(self) -> NDArray[np.float64]:
        return self.T_right.copy()

    def GetInternalHierarchyAxis(self, iaxis: int) -> NDArray[np.float64]:
        self._check_iaxis(iaxis)
        return self.axis.copy()


@dataclass
class KinBody:
    """Linear kinematic chain: ``N+1`` links bracketing ``N`` joints."""

    links: list[Link]
    joints: list[Joint]
    _dof_to_joint: list[Joint] = field(init=False)
    _links_by_name: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.links) != len(self.joints) + 1:
            raise ValueError(
                f"links ({len(self.links)}) must be exactly one more than "
                f"joints ({len(self.joints)}) for a linear chain"
            )
        self._dof_to_joint = [
            j for j in self.joints for _ in range(j.GetDOF()) if j.GetDOFIndex() >= 0
        ]
        self._links_by_name = {link.name: i for i, link in enumerate(self.links)}
        if len(self._links_by_name) != len(self.links):
            raise ValueError("link names must be unique")

    def __enter__(self) -> KinBody:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def GetDOF(self) -> int:
        return len(self._dof_to_joint)

    def GetJointFromDOFIndex(self, idof: int) -> Joint:
        return self._dof_to_joint[idof]

    @overload
    def GetChain(self, baselink: str, eelink: str, returnjoints: Literal[True]) -> list[Joint]: ...
    @overload
    def GetChain(self, baselink: str, eelink: str, returnjoints: Literal[False]) -> list[Link]: ...
    @overload
    def GetChain(
        self, baselink: str, eelink: str, returnjoints: bool
    ) -> list[Link] | list[Joint]: ...
    def GetChain(
        self,
        baselink: str,
        eelink: str,
        returnjoints: bool = True,
    ) -> list[Link] | list[Joint]:
        try:
            i0 = self._links_by_name[baselink]
            i1 = self._links_by_name[eelink]
        except KeyError as err:
            raise ValueError(f"unknown link name: {err.args[0]!r}") from err
        if i0 > i1:
            raise ValueError(f"baselink {baselink!r} must precede eelink {eelink!r} in the chain")
        if returnjoints:
            return list(self.joints[i0:i1])
        return list(self.links[i0 : i1 + 1])


def build_kinbody(
    specs: list[JointSpec],
    *,
    base_link_name: str = "base_link",
    ee_link_name: str = "ee_link",
) -> KinBody:
    """Assemble a POE-normalised :class:`KinBody` from a list of :class:`JointSpec`.

    Each joint's ``axis`` in the input ``JointSpec`` is interpreted as the
    rotation axis in the **post-T_left frame** (the local convention).
    The output :class:`KinBody` is POE-normalised: ``axis`` is expressed
    in the **base frame at q=0**, ``T_left`` carries only translation,
    and the cumulative home rotation lives in the last joint's
    ``T_right``. This matches the convention produced by
    :func:`ssik._urdf.load_urdf_kinbody_normalized`, so downstream
    predicates and solvers can compare ``joint.axis`` directly without
    having to walk the chain to recover world-frame axes.

    The forward kinematics is preserved bit-exactly: the normalisation
    is a representation change, not a numerical transform on the chain.

    :param specs: list of :class:`JointSpec`, one per joint.
    :param base_link_name: name for the chain's base link.
    :param ee_link_name: name for the end-effector link.
    """
    if not specs:
        raise ValueError("at least one JointSpec is required")

    n = len(specs)
    link_names = [base_link_name, *[f"link_{i}" for i in range(1, n)], ee_link_name]
    if len(set(link_names)) != len(link_names):
        raise ValueError(
            f"base_link_name ({base_link_name!r}) collides with ee_link_name "
            f"({ee_link_name!r}) or an auto-generated intermediate link name"
        )
    links = [Link(name=name) for name in link_names]

    # First pass: validate inputs and walk the chain at q=0 accumulating
    # cumulative rotation R_cum and position pos_cum. At each active joint,
    # record (axis_world, joint_origin) -- the rotation axis in the base
    # frame and the world-frame position of the rotation axis.
    R_cum = np.eye(3, dtype=np.float64)
    pos_cum = np.zeros(3, dtype=np.float64)
    records: list[
        tuple[
            str,
            JointType,
            NDArray[np.float64],
            NDArray[np.float64],
            tuple[float, float] | None,
        ]
    ] = []
    for i, spec in enumerate(specs):
        T_left = np.ascontiguousarray(spec.parent_link_T, dtype=np.float64)
        if T_left.shape != (4, 4):
            raise ValueError(f"spec[{i}].parent_link_T must be 4x4, got {T_left.shape}")
        axis_local = np.ascontiguousarray(spec.axis, dtype=np.float64)
        if axis_local.shape != (3,):
            raise ValueError(f"spec[{i}].axis must be shape (3,), got {axis_local.shape}")

        # Advance through T_left's rotation + translation.
        pos_cum = pos_cum + R_cum @ T_left[:3, 3]
        R_cum = R_cum @ T_left[:3, :3]

        # Joint axis in world frame, joint position in world (R(axis, 0) is
        # identity so neither R_cum nor pos_cum changes from the joint
        # rotation itself).
        axis_world = R_cum @ axis_local
        joint_origin = pos_cum.copy()

        # Advance through T_right's rotation + translation.
        if spec.child_link_T is None:
            T_right = np.eye(4, dtype=np.float64)
        else:
            T_right = np.ascontiguousarray(spec.child_link_T, dtype=np.float64)
            if T_right.shape != (4, 4):
                raise ValueError(f"spec[{i}].child_link_T must be 4x4, got {T_right.shape}")
        pos_cum = pos_cum + R_cum @ T_right[:3, 3]
        R_cum = R_cum @ T_right[:3, :3]

        joint_name = spec.name if spec.name is not None else f"j{i}"
        records.append((joint_name, spec.joint_type, axis_world, joint_origin, spec.limits))

    # Second pass: emit POE-normalised joints. T_left is pure translation
    # (offset between consecutive joint origins in world); T_right is
    # identity except on the last joint, which carries (final_offset,
    # home_rotation). pos_cum and R_cum at this point are the FK at q=0.
    joints: list[Joint] = []
    prev_origin = np.zeros(3, dtype=np.float64)
    for i, (joint_name, joint_type, axis_world, joint_origin, limits) in enumerate(records):
        new_T_left = np.eye(4, dtype=np.float64)
        new_T_left[:3, 3] = joint_origin - prev_origin

        if i == n - 1:
            # Last joint absorbs the final offset (joint_n -> EE) plus the
            # cumulative home rotation.
            new_T_right = np.eye(4, dtype=np.float64)
            new_T_right[:3, 3] = pos_cum - joint_origin
            R_part = np.eye(4, dtype=np.float64)
            R_part[:3, :3] = R_cum
            new_T_right = new_T_right @ R_part
        else:
            new_T_right = np.eye(4, dtype=np.float64)

        joints.append(
            Joint(
                name=joint_name,
                dof_index=i,
                parent_link=links[i],
                T_left=new_T_left,
                T_right=new_T_right,
                axis=axis_world,
                joint_type=joint_type,
                limits=limits,
            )
        )
        prev_origin = joint_origin

    return KinBody(links=links, joints=joints)
