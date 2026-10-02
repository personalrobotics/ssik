"""Tests for :meth:`ssik.Manipulator.body_twists` and :meth:`ssik.Manipulator.free_tail`.

A target that fixes less than a pose (a tool axis on a line, a tool plane on a plane, a
tool point on a point) is a coset ``T0 exp(span(G))`` of body-frame twists ``G``. The last
``k`` joints drop out of such a target exactly when their body twists lie in ``span(G)``:
the tail of the product of exponentials is then an element of the same group. These tests
pin the twists against FK, the membership count against the geometric reading (axis on the
line / normal to the plane / through the point), and the claim itself: freed joints set to
any values keep the primitive where it is.

Derivation and the checks against ssik's charts: self-motion-charts
``derivations/coset_targets.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik.chart import se3_exp

FIXTURES = Path(__file__).parent / "fixtures"

ARMS = {
    "franka_panda": ("franka_panda.urdf", "panda_link0", "panda_link8"),
    "kuka_iiwa14": ("kuka_iiwa14.urdf", "base", "iiwa_link_ee_kuka"),
    "ur5e": ("ur5e.urdf", "world", "tool0"),
}


def _arm(name: str) -> ssik.Manipulator:
    urdf, base, ee = ARMS[name]
    return ssik.Manipulator.from_urdf(FIXTURES / urdf, base=base, ee=ee)


def _random_q(arm: ssik.Manipulator, rng: np.random.Generator) -> np.ndarray:
    lims = [lim if lim is not None else (-np.pi, np.pi) for lim in arm.joint_limits]
    return np.array([rng.uniform(lo, hi) for lo, hi in lims])


# Body-frame generators of each primitive's stabilizer, in ssik's (v, w) order, built from
# plain geometry: a rotation about the line through p along a is (p x a, a).


def _line(p: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.array([np.r_[a, 0, 0, 0], np.r_[np.cross(p, a), a]])


def _plane(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return np.array([np.r_[u, 0, 0, 0], np.r_[np.cross(n, u), 0, 0, 0], np.r_[np.cross(p, n), n]])


def _point(p: np.ndarray) -> np.ndarray:
    return np.array([np.r_[np.cross(p, e), e] for e in np.eye(3)])


def _axis(arm: ssik.Manipulator, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Joint ``j``'s axis in the flange frame at q = 0, as (point on it, unit direction),
    read off FK alone -- independent of ``body_twists``: ``fk(0)^-1 fk(phi e_j)`` is a
    rotation about that axis."""
    phi = 0.7
    q = np.zeros(arm.dof)
    q[j] = phi
    D = np.linalg.inv(arm.fk(np.zeros(arm.dof))) @ arm.fk(q)
    R, t = D[:3, :3], D[:3, 3]
    a = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(phi))
    p, *_ = np.linalg.lstsq(np.eye(3) - R, t, rcond=None)  # the fixed line; lstsq picks p _|_ a
    return p, a / np.linalg.norm(a)


def _last_axis(arm: ssik.Manipulator) -> tuple[np.ndarray, np.ndarray]:
    return _axis(arm, arm.dof - 1)


def _perpendicular(a: np.ndarray) -> np.ndarray:
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    return u / np.linalg.norm(u)


@pytest.mark.parametrize("name", list(ARMS))
def test_body_twists_rebuild_fk(name: str) -> None:
    """FK(q) = FK(0) exp(B_1 q_1) ... exp(B_n q_n): the twists are the chain's own."""
    arm = _arm(name)
    B = arm.body_twists()
    assert B.shape == (arm.dof, 6)
    M = arm.fk(np.zeros(arm.dof))
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = _random_q(arm, rng)
        T = M.copy()
        for b, x in zip(B, q, strict=True):
            T = T @ se3_exp(b * x)
        assert np.abs(T - arm.fk(q)).max() < 1e-12


@pytest.mark.parametrize("name", list(ARMS))
def test_last_joint_turns_the_flange_about_its_own_twist(name: str) -> None:
    """FK(q) exp(phi B_n) = FK(q + phi e_n), the identity a free spin rests on."""
    arm = _arm(name)
    b = arm.body_twists()[-1]
    rng = np.random.default_rng(1)
    for _ in range(10):
        q, phi = _random_q(arm, rng), rng.uniform(-np.pi, np.pi)
        q2 = q.copy()
        q2[-1] += phi
        assert np.abs(arm.fk(q) @ se3_exp(b * phi) - arm.fk(q2)).max() < 1e-12


@pytest.mark.parametrize("name", list(ARMS))
def test_tool_axis_line_frees_the_last_joint(name: str) -> None:
    arm = _arm(name)
    p, a = _last_axis(arm)
    assert arm.free_tail(_line(p, a)) == 1
    assert arm.free_tail(_line(p, -a)) == 1  # the line, not its orientation


@pytest.mark.parametrize("name", list(ARMS))
def test_line_off_the_axis_frees_nothing(name: str) -> None:
    arm = _arm(name)
    p, a = _last_axis(arm)
    assert arm.free_tail(_line(p + 0.05 * _perpendicular(a), a)) == 0
    assert arm.free_tail(_line(p, _perpendicular(a))) == 0


@pytest.mark.parametrize("name", list(ARMS))
def test_plane_normal_to_the_axis_frees_the_last_joint_wherever_it_is(name: str) -> None:
    arm = _arm(name)
    p, a = _last_axis(arm)
    offset = np.array([0.3, -0.2, 0.1])
    assert arm.free_tail(_plane(p, a)) == 1
    assert arm.free_tail(_plane(p + offset, a)) == 1


@pytest.mark.parametrize("name", list(ARMS))
def test_plane_holding_the_axis_frees_nothing(name: str) -> None:
    arm = _arm(name)
    p, a = _last_axis(arm)
    assert arm.free_tail(_plane(p, _perpendicular(a))) == 0


def test_spherical_wrist_centre_frees_all_three_wrist_joints() -> None:
    """iiwa14: joints 5-7 meet at the wrist centre, so a point target there frees them."""
    arm = _arm("kuka_iiwa14")
    # the centre: the point on joint 7's axis closest to joint 6's
    (p6, a6), (p7, a7) = _axis(arm, 5), _axis(arm, 6)
    n = np.cross(a7, a6)
    s = np.dot(np.cross(p6 - p7, a6), n) / (n @ n)
    centre = p7 + s * a7
    assert arm.free_tail(_point(centre)) == 3
    assert arm.free_tail(_point(centre + 0.05 * a7)) == 1  # further along joint 7 only


def test_freed_joints_keep_the_primitive_in_place() -> None:
    """The claim itself: any values of the freed joints leave the target satisfied."""
    arm = _arm("kuka_iiwa14")
    on_axis, _ = _last_axis(arm)  # a point on joint 7's axis: frees at least the last joint
    k = arm.free_tail(_point(on_axis))
    assert k >= 1
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = _random_q(arm, rng)
        q2 = q.copy()
        q2[-k:] = _random_q(arm, rng)[-k:]
        x1 = arm.fk(q) @ np.r_[on_axis, 1.0]
        x2 = arm.fk(q2) @ np.r_[on_axis, 1.0]
        assert np.abs(x1 - x2).max() < 1e-12
        q3 = q.copy()
        q3[-k - 1] += 0.3  # the first joint outside the tail does move it
        assert np.abs(arm.fk(q3) @ np.r_[on_axis, 1.0] - x1).max() > 1e-3


def test_free_tail_takes_one_twist_and_an_empty_set() -> None:
    arm = _arm("franka_panda")
    b = arm.body_twists()[-1]
    assert arm.free_tail(b) == 1
    assert arm.free_tail(np.zeros((0, 6))) == 0


def test_free_tail_rejects_a_bad_shape() -> None:
    arm = _arm("franka_panda")
    with pytest.raises(ValueError, match=r"\(d, 6\)"):
        arm.free_tail(np.zeros((2, 4)))
