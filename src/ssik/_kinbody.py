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

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ssik.core.tolerances import TolerancePolicy

__all__ = [
    "Joint",
    "JointSpec",
    "KinBody",
    "Link",
    "build_kinbody",
    "build_poe_kinbody",
    "canonicalize_spherical_wrist",
]

# A revolute joint's frame origin may sit anywhere along its own axis without
# changing kinematics; below this the residual off-axis component of a wrist
# offset is treated as zero (a genuine spherical wrist meets to ~1e-15).
_WRIST_PERP_ATOL = 1e-9

# An axis vector shorter than this carries no usable direction (degenerate).
_AXIS_MIN_NORM = 1e-9
# An axis already this close to unit length is passed through unchanged, so a
# clean (already-normalized) chain -- e.g. every URDF axis, which urchin unitizes
# on load -- keeps its exact bytes and baked artifacts don't drift. Only a
# meaningfully non-unit axis (a hand-built JointSpec / from_axes row) is scaled.
_AXIS_UNIT_TOL = 1e-9

JointType = Literal["revolute", "prismatic"]


def _unit_axis(axis: NDArray[np.float64], i: int) -> NDArray[np.float64]:
    """Return ``axis`` scaled to unit length.

    A joint axis denotes only a *direction*: its magnitude is physically
    meaningless, but the Rodrigues kernel (``rotate`` / ``rotation_matrix``) and
    every predicate assume ``|axis| == 1`` (``predicates`` documents that a
    non-unit axis silently misbehaves rather than raising). The URDF loader gets
    unit axes for free from ``urchin``, but the direct constructors
    (``from_axes`` / ``from_dh`` / ``from_transforms`` / raw ``JointSpec``) trust
    the caller, so we normalize here -- the one construction chokepoint -- to
    make all paths robust to the axis-length gauge. A near-zero axis has no
    direction and is a construction error.
    """
    n = float(np.linalg.norm(axis))
    if n < _AXIS_MIN_NORM:
        raise ValueError(f"joint {i} axis is degenerate (norm {n:.2e}); it has no direction")
    if abs(n - 1.0) <= _AXIS_UNIT_TOL:
        # Already unit -- return the exact input (no float division), so a
        # normalized chain stays bit-identical and baked artifacts don't shift.
        return axis
    return axis / n


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
        # Normalize to unit -- the Rodrigues kernel + predicates assume |axis|=1.
        # R_cum below is a rotation (norm-preserving), so axis_world stays unit.
        axis_local = _unit_axis(axis_local, i)

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


# POE record: (joint name, joint type, axis in base frame, translation offset
# from the previous joint's frame, optional (lo, hi) limits).
PoeRecord = tuple[
    str, "JointType", NDArray[np.float64], NDArray[np.float64], "tuple[float, float] | None"
]


def build_poe_kinbody(
    records: list[PoeRecord],
    final_t_right: NDArray[np.float64],
    base_link: str,
    ee_link: str,
) -> KinBody:
    """Construct a POE-normalized :class:`KinBody` from per-joint records.

    Each record's ``T_left`` is the pure translation ``p_offset`` and its axis
    is already in the base frame at ``q=0`` (the POE encoding). ``final_t_right``
    absorbs the trailing offset + home orientation on the last joint so
    ``FK(q=0)`` matches the source. Shared by the URDF and MJCF loaders so the
    construction (and synthesized intermediate link names) lives in one place.
    """
    n = len(records)
    if n == 0:
        raise ValueError(f"chain from {base_link!r} to {ee_link!r} has no active joints")
    link_names = [base_link, *[f"_poe_link_{i}" for i in range(1, n)], ee_link]
    links = [Link(name=name) for name in link_names]
    joints: list[Joint] = []
    for i, (name, joint_type, axis_base, p_offset, limits) in enumerate(records):
        # Normalize to unit (urchin already does for URDF; belt-and-suspenders
        # for any other record source). Rodrigues + predicates assume |axis|=1.
        axis_base = _unit_axis(np.ascontiguousarray(axis_base, dtype=np.float64), i)
        t_left = np.eye(4, dtype=np.float64)
        t_left[:3, 3] = p_offset
        t_right = final_t_right if i == n - 1 else np.eye(4, dtype=np.float64)
        joints.append(
            Joint(
                name=name,
                dof_index=i,
                parent_link=links[i],
                T_left=t_left,
                T_right=t_right,
                axis=axis_base,
                joint_type=joint_type,
                limits=limits,
            )
        )
    return KinBody(links=links, joints=joints)


def canonicalize_spherical_wrist(kb: KinBody, policy: TolerancePolicy | None = None) -> KinBody:
    """Return ``kb`` with its spherical wrist expressed in the canonical gauge.

    The ik-geo ``spherical`` family consolidates the wrist as
    ``p[3] = T_left[3] + T_left[4] + T_left[5]`` (a telescoping sum that reduces
    to the *last* wrist joint's origin) and reads the tool-flange offset from
    ``T_right[5]``. A URDF that places the last wrist joint's frame a fixed
    distance *along its own rotation axis* from the wrist-center intersection --
    the flange offset, standard on industrial arms like the ABB IRB 6700 (#377)
    -- breaks that: the offset lands in the consolidated ``p[3]`` instead of the
    tool term, so the wrist center is computed wrong and the solver returns
    nothing.

    A revolute joint's origin may slide freely along its own axis without
    changing kinematics (a translation along the axis commutes with the joint
    rotation: ``R(axis, q)^-1 @ Trans(d) @ R(axis, q) == Trans(d)`` for
    ``d || axis``). This returns a copy with the last wrist joint slid onto the
    axis intersection and the along-axis offset moved into ``T_right`` -- exactly
    FK-identical, and canonical for the solver's consolidation. It is a no-op
    (returns ``kb`` unchanged) when the last three axes are not concurrent or the
    wrist is already canonical, so it is safe to call unconditionally at solver
    entry. Only the *along-axis* component is a gauge freedom; an off-axis origin
    is a genuinely non-spherical wrist and is left untouched.

    Gauge-invariance lives here, in the solver path, rather than in construction:
    the un-gauged representation is what other solvers (SRS / jointlock / HP) and
    the baked artifacts depend on, so nothing outside the spherical solve is
    perturbed (#377).
    """
    # Local imports keep the module's import surface flat and cycle-free
    # (``predicates`` imports this module only under TYPE_CHECKING).
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.kinematics.predicates import axes_meet_at_common_point

    if policy is None:
        policy = DEFAULT_TOLERANCE_POLICY

    joints = kb.joints
    n = len(joints)
    if n < 3:
        return kb
    last = joints[-1]
    if not last.IsRevolute(0):
        return kb

    p = axes_meet_at_common_point(joints, (n - 3, n - 2, n - 1), policy)
    if p is None:
        return kb

    # Last wrist joint origin in the base frame (T_left is a pure translation
    # post-normalization, so origins accumulate along the chain).
    origin = np.zeros(3, dtype=np.float64)
    for j in joints:
        origin = origin + j.T_left[:3, 3]

    axis = last.axis
    delta = origin - p
    along = float(np.dot(delta, axis)) * axis
    if float(np.linalg.norm(delta - along)) > _WRIST_PERP_ATOL:
        return kb  # off-axis: genuinely non-spherical, not a gauge freedom
    if float(np.linalg.norm(along)) < _WRIST_PERP_ATOL:
        return kb  # already canonical

    new_left = last.T_left.copy()
    new_left[:3, 3] = new_left[:3, 3] - along
    new_right = last.T_right.copy()
    new_right[:3, 3] = new_right[:3, 3] + along
    new_joints = [*joints[:-1], replace(last, T_left=new_left, T_right=new_right)]
    return KinBody(links=list(kb.links), joints=new_joints)
