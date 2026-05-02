"""Forward kinematics for POE-normalized kinematic chains.

The shared implementation used by every analytical solver's post-verify
step. Before #137 Slice 4 step 1b this lived as seven byte-identical
copies in ``ssik.solvers.ikgeo.*``; consolidating to a single module
means we optimise it exactly once for every inner-solver hot path.

#137 Slice 4 step 3b (mirroring step 3 in the artifact orchestrator
template, #152): the body now uses hand-rolled scalar 4x4 matmul +
inline Rodrigues rather than ``rotation_matrix`` + numpy ``@``. Each
numpy ``@`` on a 4x4 has ~3 us of dispatch overhead that compounds
over the per-IK inner-solver verify path: Franka 7R calls this ~94
times per default IK via the lock sweep, so removing 3 numpy ``@``s
per call * 94 calls = ~850 us per IK saved.
"""

from __future__ import annotations

import math

import cython
import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody

__all__ = ["poe_forward_kinematics"]


@cython.ccall
@cython.locals(
    n=cython.int,
    i=cython.int,
    ax=cython.double,
    ay=cython.double,
    az=cython.double,
    qi=cython.double,
    c=cython.double,
    s=cython.double,
    oc=cython.double,
    r00=cython.double,
    r01=cython.double,
    r02=cython.double,
    r10=cython.double,
    r11=cython.double,
    r12=cython.double,
    r20=cython.double,
    r21=cython.double,
    r22=cython.double,
    l00=cython.double,
    l01=cython.double,
    l02=cython.double,
    l03=cython.double,
    l10=cython.double,
    l11=cython.double,
    l12=cython.double,
    l13=cython.double,
    l20=cython.double,
    l21=cython.double,
    l22=cython.double,
    l23=cython.double,
    m00=cython.double,
    m01=cython.double,
    m02=cython.double,
    m03=cython.double,
    m10=cython.double,
    m11=cython.double,
    m12=cython.double,
    m13=cython.double,
    m20=cython.double,
    m21=cython.double,
    m22=cython.double,
    m23=cython.double,
    t00=cython.double,
    t01=cython.double,
    t02=cython.double,
    t03=cython.double,
    t10=cython.double,
    t11=cython.double,
    t12=cython.double,
    t13=cython.double,
    t20=cython.double,
    t21=cython.double,
    t22=cython.double,
    t23=cython.double,
    n00=cython.double,
    n01=cython.double,
    n02=cython.double,
    n03=cython.double,
    n10=cython.double,
    n11=cython.double,
    n12=cython.double,
    n13=cython.double,
    n20=cython.double,
    n21=cython.double,
    n22=cython.double,
    n23=cython.double,
    a00=cython.double,
    a01=cython.double,
    a02=cython.double,
    a03=cython.double,
    a10=cython.double,
    a11=cython.double,
    a12=cython.double,
    a13=cython.double,
    a20=cython.double,
    a21=cython.double,
    a22=cython.double,
    a23=cython.double,
    b00=cython.double,
    b01=cython.double,
    b02=cython.double,
    b03=cython.double,
    b10=cython.double,
    b11=cython.double,
    b12=cython.double,
    b13=cython.double,
    b20=cython.double,
    b21=cython.double,
    b22=cython.double,
    b23=cython.double,
)
def poe_forward_kinematics(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """POE forward kinematics for a normalized :class:`KinBody` at config ``q``.

    Walks the chain joint-by-joint, applying ``T_left @ Joint(axis, q) @ T_right``
    in order, where ``Joint`` is a rotation for revolute joints and a translation
    along ``axis`` for prismatic joints. Returns the 4x4 base-to-end pose.

    Hand-rolled scalar 4x4 matmul + inline Rodrigues. The accumulator is
    carried as 12 scalars (the bottom row ``[0, 0, 0, 1]`` is implicit);
    no per-call ``np.eye(4)`` allocations and no per-joint numpy ``@``
    dispatch.
    """
    joints = kb.joints
    n = len(joints)
    a00 = 1.0
    a01 = 0.0
    a02 = 0.0
    a03 = 0.0
    a10 = 0.0
    a11 = 1.0
    a12 = 0.0
    a13 = 0.0
    a20 = 0.0
    a21 = 0.0
    a22 = 1.0
    a23 = 0.0
    for i in range(n):
        joint = joints[i]
        axis = joint.axis
        ax = float(axis[0])
        ay = float(axis[1])
        az = float(axis[2])
        qi = float(q[i])
        # T_left entries.
        Tl = joint.T_left
        l00 = float(Tl[0, 0])
        l01 = float(Tl[0, 1])
        l02 = float(Tl[0, 2])
        l03 = float(Tl[0, 3])
        l10 = float(Tl[1, 0])
        l11 = float(Tl[1, 1])
        l12 = float(Tl[1, 2])
        l13 = float(Tl[1, 3])
        l20 = float(Tl[2, 0])
        l21 = float(Tl[2, 1])
        l22 = float(Tl[2, 2])
        l23 = float(Tl[2, 3])
        if joint.joint_type == "prismatic":
            # Joint transform is translation by qi along axis. Composed with
            # T_left: rotation block unchanged from L33, translation column
            # gets ``L33 @ (qi * axis)`` added on top of Lt.
            m00 = l00
            m01 = l01
            m02 = l02
            m03 = l03 + qi * (l00 * ax + l01 * ay + l02 * az)
            m10 = l10
            m11 = l11
            m12 = l12
            m13 = l13 + qi * (l10 * ax + l11 * ay + l12 * az)
            m20 = l20
            m21 = l21
            m22 = l22
            m23 = l23 + qi * (l20 * ax + l21 * ay + l22 * az)
        else:
            # Revolute path: Rodrigues 3x3 rotation, then ``M = T_left @ R``.
            c = math.cos(qi)
            s = math.sin(qi)
            oc = 1.0 - c
            r00 = c + ax * ax * oc
            r01 = ax * ay * oc - az * s
            r02 = ax * az * oc + ay * s
            r10 = ay * ax * oc + az * s
            r11 = c + ay * ay * oc
            r12 = ay * az * oc - ax * s
            r20 = az * ax * oc - ay * s
            r21 = az * ay * oc + ax * s
            r22 = c + az * az * oc
            m00 = l00 * r00 + l01 * r10 + l02 * r20
            m01 = l00 * r01 + l01 * r11 + l02 * r21
            m02 = l00 * r02 + l01 * r12 + l02 * r22
            m03 = l03
            m10 = l10 * r00 + l11 * r10 + l12 * r20
            m11 = l10 * r01 + l11 * r11 + l12 * r21
            m12 = l10 * r02 + l11 * r12 + l12 * r22
            m13 = l13
            m20 = l20 * r00 + l21 * r10 + l22 * r20
            m21 = l20 * r01 + l21 * r11 + l22 * r21
            m22 = l20 * r02 + l21 * r12 + l22 * r22
            m23 = l23
        # T_right entries.
        Tr = joint.T_right
        t00 = float(Tr[0, 0])
        t01 = float(Tr[0, 1])
        t02 = float(Tr[0, 2])
        t03 = float(Tr[0, 3])
        t10 = float(Tr[1, 0])
        t11 = float(Tr[1, 1])
        t12 = float(Tr[1, 2])
        t13 = float(Tr[1, 3])
        t20 = float(Tr[2, 0])
        t21 = float(Tr[2, 1])
        t22 = float(Tr[2, 2])
        t23 = float(Tr[2, 3])
        # N = M @ T_right
        n00 = m00 * t00 + m01 * t10 + m02 * t20
        n01 = m00 * t01 + m01 * t11 + m02 * t21
        n02 = m00 * t02 + m01 * t12 + m02 * t22
        n03 = m00 * t03 + m01 * t13 + m02 * t23 + m03
        n10 = m10 * t00 + m11 * t10 + m12 * t20
        n11 = m10 * t01 + m11 * t11 + m12 * t21
        n12 = m10 * t02 + m11 * t12 + m12 * t22
        n13 = m10 * t03 + m11 * t13 + m12 * t23 + m13
        n20 = m20 * t00 + m21 * t10 + m22 * t20
        n21 = m20 * t01 + m21 * t11 + m22 * t21
        n22 = m20 * t02 + m21 * t12 + m22 * t22
        n23 = m20 * t03 + m21 * t13 + m22 * t23 + m23
        # T_acc = T_acc @ N
        b00 = a00 * n00 + a01 * n10 + a02 * n20
        b01 = a00 * n01 + a01 * n11 + a02 * n21
        b02 = a00 * n02 + a01 * n12 + a02 * n22
        b03 = a00 * n03 + a01 * n13 + a02 * n23 + a03
        b10 = a10 * n00 + a11 * n10 + a12 * n20
        b11 = a10 * n01 + a11 * n11 + a12 * n21
        b12 = a10 * n02 + a11 * n12 + a12 * n22
        b13 = a10 * n03 + a11 * n13 + a12 * n23 + a13
        b20 = a20 * n00 + a21 * n10 + a22 * n20
        b21 = a20 * n01 + a21 * n11 + a22 * n21
        b22 = a20 * n02 + a21 * n12 + a22 * n22
        b23 = a20 * n03 + a21 * n13 + a22 * n23 + a23
        a00, a01, a02, a03 = b00, b01, b02, b03
        a10, a11, a12, a13 = b10, b11, b12, b13
        a20, a21, a22, a23 = b20, b21, b22, b23
    return np.array(
        [[a00, a01, a02, a03], [a10, a11, a12, a13], [a20, a21, a22, a23], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
