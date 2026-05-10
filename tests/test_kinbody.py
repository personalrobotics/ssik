"""Unit tests for :class:`ssik._kinbody.KinBody`.

These exercise the structural invariants of the kinbody chain: link/joint
counts, name uniqueness, dataclass behaviour. Full integration tests
through the public solvers live in ``tests/test_ikgeo_*.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik._kinbody import JointSpec, KinBody, Link, build_kinbody


def _spec(
    joint_type: str = "revolute",
    limits: tuple[float, float] | None = None,
) -> JointSpec:
    return JointSpec(
        parent_link_T=np.eye(4),
        axis=np.array([0.0, 0.0, 1.0]),
        joint_type=joint_type,  # type: ignore[arg-type]
        limits=limits,
    )


def test_link_equality_is_by_name() -> None:
    a = Link(name="l0")
    b = Link(name="l0")
    c = Link(name="l1")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_link_equality_rejects_non_link() -> None:
    assert (Link(name="l0") == "l0") is False


def test_build_kinbody_requires_specs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_kinbody([])


def test_build_kinbody_rejects_bad_transform_shape() -> None:
    bad = JointSpec(
        parent_link_T=np.eye(3),
        axis=np.array([0.0, 0.0, 1.0]),
        joint_type="revolute",
    )
    with pytest.raises(ValueError, match="must be 4x4"):
        build_kinbody([bad])


def test_build_kinbody_rejects_bad_axis_shape() -> None:
    bad = JointSpec(
        parent_link_T=np.eye(4),
        axis=np.array([0.0, 0.0, 1.0, 0.0]),
        joint_type="revolute",
    )
    with pytest.raises(ValueError, match="shape \\(3,\\)"):
        build_kinbody([bad])


def test_build_kinbody_link_count_and_names() -> None:
    kb = build_kinbody([_spec(), _spec(), _spec()])
    names = [link.name for link in kb.links]
    assert names == ["base_link", "link_1", "link_2", "ee_link"]
    assert len(kb.joints) == 3


def test_build_kinbody_rejects_name_collision() -> None:
    with pytest.raises(ValueError, match="collides"):
        build_kinbody([_spec(), _spec()], base_link_name="link_1", ee_link_name="ee_link")


def test_kinbody_dof_and_joint_lookup() -> None:
    kb = build_kinbody([_spec(), _spec("prismatic")])
    assert kb.GetDOF() == 2
    assert kb.GetJointFromDOFIndex(0) is kb.joints[0]
    assert kb.GetJointFromDOFIndex(1) is kb.joints[1]


def test_kinbody_context_manager_is_noop() -> None:
    kb = build_kinbody([_spec()])
    with kb as inner:
        assert inner is kb


def test_get_chain_full() -> None:
    kb = build_kinbody([_spec(), _spec(), _spec()])
    chainlinks = kb.GetChain("base_link", "ee_link", returnjoints=False)
    chainjoints = kb.GetChain("base_link", "ee_link", returnjoints=True)
    assert [link.name for link in chainlinks] == ["base_link", "link_1", "link_2", "ee_link"]
    assert len(chainjoints) == 3
    assert len(chainlinks) == len(chainjoints) + 1


def test_get_chain_partial() -> None:
    kb = build_kinbody([_spec(), _spec(), _spec()])
    chainlinks = kb.GetChain("link_1", "link_2", returnjoints=False)
    chainjoints = kb.GetChain("link_1", "link_2", returnjoints=True)
    assert [link.name for link in chainlinks] == ["link_1", "link_2"]
    assert len(chainjoints) == 1


def test_get_chain_unknown_link_raises() -> None:
    kb = build_kinbody([_spec()])
    with pytest.raises(ValueError, match="unknown link"):
        kb.GetChain("base_link", "nope", returnjoints=True)


def test_get_chain_reversed_raises() -> None:
    kb = build_kinbody([_spec(), _spec()])
    with pytest.raises(ValueError, match="must precede"):
        kb.GetChain("ee_link", "base_link", returnjoints=True)


def test_joint_parent_link_equality_drives_chain_orientation() -> None:
    """Each joint's parent link must compare equal (by name) to the
    corresponding link in :meth:`KinBody.GetChain`. Solvers rely on this
    equality to align joint orientation with traversal direction when
    reasoning about which joint connects which two links.
    """
    kb = build_kinbody([_spec(), _spec(), _spec()])
    chainlinks = kb.GetChain("base_link", "ee_link", returnjoints=False)
    chainjoints = kb.GetChain("base_link", "ee_link", returnjoints=True)
    for i, joint in enumerate(chainjoints):
        assert joint.GetHierarchyParentLink() == chainlinks[i]


def test_joint_transform_flat_is_row_major_16() -> None:
    """A 4x4 transform's ``T.flat`` iterator yields 16 row-major scalars,
    the layout solvers expect when serialising / extracting matrix entries.
    """
    kb = build_kinbody([_spec()])
    j = kb.joints[0]
    T = j.GetInternalHierarchyLeftTransform()
    flat = list(T.flat)
    assert len(flat) == 16
    assert T.shape == (4, 4)


def test_joint_axis_has_len_and_indexing() -> None:
    kb = build_kinbody([_spec()])
    j = kb.joints[0]
    axis = j.GetInternalHierarchyAxis(0)
    assert len(axis) == 3
    assert axis[2] == pytest.approx(1.0)


def test_joint_type_predicates() -> None:
    kb = build_kinbody([_spec("revolute"), _spec("prismatic")])
    assert kb.joints[0].IsRevolute(0) is True
    assert kb.joints[0].IsPrismatic(0) is False
    assert kb.joints[1].IsRevolute(0) is False
    assert kb.joints[1].IsPrismatic(0) is True
    assert kb.joints[0].IsStatic() is False
    assert kb.joints[0].IsMimic(0) is False


def test_joint_iaxis_out_of_range_raises() -> None:
    kb = build_kinbody([_spec()])
    with pytest.raises(ValueError, match="single-DOF"):
        kb.joints[0].IsRevolute(1)


def test_mimic_equation_raises() -> None:
    kb = build_kinbody([_spec()])
    with pytest.raises(NotImplementedError):
        kb.joints[0].GetMimicEquation(0)


def test_joint_transforms_returned_are_copies() -> None:
    """Defensive: callers may mutate returned transforms (and sympy
    serialisation in the build pipeline assumes immutability). Make sure
    KinBody returns fresh arrays.
    """
    kb = build_kinbody([_spec()])
    j = kb.joints[0]
    T1 = j.GetInternalHierarchyLeftTransform()
    T1[0, 0] = 99.0
    T2 = j.GetInternalHierarchyLeftTransform()
    assert T2[0, 0] == pytest.approx(1.0)


def test_kinbody_rejects_mismatched_link_joint_counts() -> None:
    links = [Link(name="a"), Link(name="b")]
    with pytest.raises(ValueError, match="one more than"):
        KinBody(links=links, joints=[])


def test_custom_joint_names_preserved() -> None:
    spec = JointSpec(
        parent_link_T=np.eye(4),
        axis=np.array([0.0, 0.0, 1.0]),
        joint_type="revolute",
        name="shoulder_pan",
    )
    kb = build_kinbody([spec])
    assert kb.joints[0].GetName() == "shoulder_pan"


def test_joint_limits_default_none() -> None:
    """Joints without explicit limits default to ``limits=None`` (continuous /
    unspecified). This is the silent backwards-compat path for fixtures that
    don't supply limits."""
    kb = build_kinbody([_spec(), _spec()])
    for j in kb.joints:
        assert j.limits is None


def test_joint_limits_propagate_through_build_kinbody() -> None:
    """``JointSpec.limits`` propagate to ``Joint.limits`` after POE
    normalisation. Limits are kinematic data; the normalisation is a frame
    change, not a re-parametrisation, so limits stay attached to the same
    physical joint."""
    specs = [
        _spec(limits=(-1.0, 1.0)),
        _spec(limits=None),  # continuous middle joint
        _spec(limits=(-2.5, 0.5)),
    ]
    kb = build_kinbody(specs)
    assert kb.joints[0].limits == (-1.0, 1.0)
    assert kb.joints[1].limits is None
    assert kb.joints[2].limits == (-2.5, 0.5)
