"""Minimal duck-typed kinbody for the vendored IKFast solver.

The upstream `ikfast.py` solver was written against OpenRAVE's `KinBody` API.
This module provides *exactly* the subset of that API that the solver actually
calls, nothing more. The surface was established by auditing every call site
in `ssik/_vendor/ikfast.py`; see the issue #5 for the catalog.

Input format is a flat list of :class:`JointSpec` (one per joint). For each
joint the spec gives:

* ``parent_link_T`` — 4x4 rigid transform from the parent link's frame to the
  joint's frame.
* ``axis`` — unit 3-vector in the joint frame (post-``parent_link_T``).
* ``joint_type`` — ``"revolute"`` or ``"prismatic"``.

The factory :func:`build_kinbody` assembles a linear chain of ``N + 1`` links
bracketing the ``N`` joints and returns a :class:`KinBody` ready to hand to
:class:`ssik._vendor.ikfast.IKFastSolver`.

This module is private. The public API (``Manipulator``) will wrap it; see #12.
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
    """

    parent_link_T: NDArray[np.float64]
    axis: NDArray[np.float64]
    joint_type: JointType
    child_link_T: NDArray[np.float64] | None = None
    name: str | None = None


@dataclass(eq=False)
class Link:
    """Rigid body in the kinematic chain.

    Equality is by name so that ``joint.GetHierarchyParentLink() == chainlinks[i]``
    (see ikfast.py:1650) works across object identities.
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
    """

    name: str
    dof_index: int
    parent_link: Link
    T_left: NDArray[np.float64]
    T_right: NDArray[np.float64]
    axis: NDArray[np.float64]
    joint_type: JointType

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
    """Assemble a :class:`KinBody` from a linear list of :class:`JointSpec`.

    Each joint gets ``T_left = spec.parent_link_T`` and ``T_right = identity``,
    with ``dof_index = i`` (contiguous, one DOF per joint).
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

    joints: list[Joint] = []
    for i, spec in enumerate(specs):
        T_left = np.ascontiguousarray(spec.parent_link_T, dtype=np.float64)
        if T_left.shape != (4, 4):
            raise ValueError(f"spec[{i}].parent_link_T must be 4x4, got {T_left.shape}")
        axis = np.ascontiguousarray(spec.axis, dtype=np.float64)
        if axis.shape != (3,):
            raise ValueError(f"spec[{i}].axis must be shape (3,), got {axis.shape}")
        if spec.child_link_T is None:
            T_right = np.eye(4, dtype=np.float64)
        else:
            T_right = np.ascontiguousarray(spec.child_link_T, dtype=np.float64)
            if T_right.shape != (4, 4):
                raise ValueError(f"spec[{i}].child_link_T must be 4x4, got {T_right.shape}")
        joints.append(
            Joint(
                name=spec.name if spec.name is not None else f"j{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_left,
                T_right=T_right,
                axis=axis,
                joint_type=spec.joint_type,
            )
        )
    return KinBody(links=links, joints=joints)
