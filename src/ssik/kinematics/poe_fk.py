"""Forward kinematics for POE-normalized kinematic chains.

The shared implementation used by every analytical solver's post-verify
step. Before #137 Slice 4 step 1b this lived as seven byte-identical
copies in ``ssik.solvers.ikgeo.*``; consolidating to a single module
means the next-stage Cython source rewrite (#147) optimises it exactly
once, and the same per-call ``np.eye(4)`` allocation fix applies to
every solver simultaneously.

The cached ``_FK_EYE4`` constant + reused ``rot`` buffer match the
artifact orchestrator pattern from #146; the runtime hot path through
the inner 6R solvers gets the same default-call speedup.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["poe_forward_kinematics"]


# Cached read-only 4x4 identity. Hot-path callers do ``_FK_EYE4.copy()``
# instead of ``np.eye(4)`` to avoid the per-call diagonal-init cost --
# this matters under a 7R lock-sweep that calls back into the inner 6R
# solver's verify pass 24x per IK.
_FK_EYE4 = np.eye(4, dtype=np.float64)
_FK_EYE4.flags.writeable = False


def poe_forward_kinematics(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """POE forward kinematics for a normalized :class:`KinBody` at config ``q``.

    Walks the chain joint-by-joint, applying ``T_left @ Rot(axis, q) @ T_right``
    in order. Returns the 4x4 base-to-end pose.

    Performance: a single ``rot`` buffer is reused across joints; only its
    3x3 rotation block is overwritten per iteration (the bottom row stays
    ``[0, 0, 0, 1]``). One numpy 4x4 alloc total instead of one per joint.
    """
    T = _FK_EYE4.copy()
    rot = _FK_EYE4.copy()
    for j, qi in zip(kb.joints, q, strict=True):
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T
