"""Chain reversal for POE-normalised kinematic chains.

For a 6R chain with forward kinematics

    FK(q) = T_pre @ T_left[0] @ R(a[0], q[0]) @ T_left[1] @ R(a[1], q[1])
            @ ... @ T_left[n-1] @ R(a[n-1], q[n-1]) @ T_right[n-1]

(POE-normalised: each ``T_left[i]`` is a pure translation, ``T_right[i]``
is identity for ``i < n-1``, and the last ``T_right`` carries the home
rotation), the **reversed chain** has

    FK_rev(q') = (FK(reverse(q')))^{-1}

so solving IK on the reversed chain with target ``T_target^{-1}`` and then
flipping the result is equivalent to solving the original IK problem.
This module provides :func:`reverse_kinematic_chain` and a
:func:`map_reversed_q` helper.

Why this matters: ssik's tier-0 IK solvers expect specific kinematic
patterns at canonical positions (parallel triple at sub-chain ``(1,2,3)``,
spherical wrist at ``(3,4,5)``). Some arms — Franka being the canonical
example — have those same patterns but at the *base* of the sub-chain
after locking the redundant joint. Reversing the chain lands the pattern
at the canonical position so the existing solvers work unchanged.

EAIK calls this the ``REVERSED`` kinematic class. See #121 Level 1.

References:
- IKS C++ source: ``EAIK/src/CPP/src/IK/6R_IK.cpp`` (``reverse_kinematic_chain``)
- Discussion: GitHub issue #121
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import Joint, KinBody, Link

__all__ = ["map_reversed_q", "reverse_kinematic_chain"]


def reverse_kinematic_chain(kb: KinBody) -> KinBody:
    """Return the reversed POE-normalised chain.

    Math: with the original chain expressed as

        FK(q) = trans(p_0) @ R(a_0, q_0) @ trans(p_1) @ R(a_1, q_1)
                @ ... @ trans(p_{n-1}) @ R(a_{n-1}, q_{n-1})
                @ trans(p_n) @ R_home

    inverting and propagating ``R_home^T`` through the chain via the
    conjugation identity ``R @ R(a, q) = R(R @ a, q) @ R`` yields

        FK(q)^{-1} = trans(-R_home^T p_n)
                     @ R(-R_home^T a_{n-1}, q_{n-1}) @ trans(-R_home^T p_{n-1})
                     @ R(-R_home^T a_{n-2}, q_{n-2}) @ trans(-R_home^T p_{n-2})
                     @ ...
                     @ R(-R_home^T a_0, q_0) @ trans(-R_home^T p_0)
                     @ R_home^T

    which is POE-form for the reversed chain with sign-flipped axes (so
    ``q_rev[i] = q[n-1-i]`` directly without sign flips). Reversed-chain
    parameters:

      - ``axis_rev[i] = -R_home^T @ axis[n-1-i]`` for ``i in 0..n-1``
      - ``T_left_rev[i].translation = -R_home^T @ p[n-i]`` for ``i in 0..n-1``
      - ``T_right_rev[n-1] = trans(-R_home^T @ p_0) @ rotate(R_home^T)``
      - ``T_right_rev[i] = identity`` for ``i in 0..n-2``

    where ``p_0`` is the base→joint-0 translation, ``p_i`` for
    ``i in 1..n-1`` is the joint-(i-1)→joint-i translation, and ``p_n``
    is the joint-(n-1)→EE translation.

    :param kb: a POE-normalised :class:`KinBody`.
    :returns: a new :class:`KinBody` representing the reversed chain.
    """
    joints = kb.joints
    n = len(joints)
    if n == 0:
        raise ValueError("cannot reverse an empty chain")

    # Extract POE-form translations p_0..p_n.
    # p_i for i=0..n-1 is the T_left translation of joint i.
    # p_n is the T_right translation of the last joint (joint-to-EE).
    translations: list[NDArray[np.float64]] = [joints[i].T_left[:3, 3].copy() for i in range(n)]
    last_t_right = joints[-1].T_right
    translations.append(last_t_right[:3, 3].copy())

    # Home rotation = rotation block of last joint's T_right.
    R_home = last_t_right[:3, :3].copy()
    R_home_T = R_home.T

    # Reversed joints. Sign-flipped axis convention: q_rev[i] = q[n-1-i]
    # (no per-joint sign flip needed when we negate the axis).
    new_links = [Link(name=f"_rev_link_{i}") for i in range(n + 1)]
    new_joints: list[Joint] = []
    for i in range(n):
        # Reversed joint i corresponds to original joint (n - 1 - i).
        orig_j = joints[n - 1 - i]

        # Axis: -R_home^T @ a_{n-1-i}
        axis_rev = -R_home_T @ orig_j.axis

        # T_left translation: -R_home^T @ p_{n-i}
        T_left_rev = np.eye(4, dtype=np.float64)
        T_left_rev[:3, 3] = -R_home_T @ translations[n - i]

        if i < n - 1:
            T_right_rev = np.eye(4, dtype=np.float64)
        else:
            # Last reversed joint absorbs:
            # trans(-R_home^T @ p_0) @ rotate(R_home^T)
            T_right_rev = np.eye(4, dtype=np.float64)
            T_right_rev[:3, :3] = R_home_T
            T_right_rev[:3, 3] = -R_home_T @ translations[0]

        new_joints.append(
            Joint(
                name=f"{orig_j.name}__rev",
                dof_index=i,
                parent_link=new_links[i],
                T_left=T_left_rev,
                T_right=T_right_rev,
                axis=axis_rev,
                joint_type=orig_j.joint_type,
            )
        )

    return KinBody(links=new_links, joints=new_joints)


def map_reversed_q(q_rev: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map a reversed-chain joint-angle vector back to original-chain ordering.

    With the sign-flipped-axis convention used by :func:`reverse_kinematic_chain`,
    the mapping is just a reverse-order permutation: ``q[i] = q_rev[n - 1 - i]``.
    """
    return np.flip(np.asarray(q_rev, dtype=np.float64)).copy()
