"""Body-frame joint twists, and which trailing joints a partial target leaves free.

Every chain is a product of exponentials in body form,

    FK(q) = FK(0) exp(B_1 q_1) ... exp(B_n q_n),

with ``B_i`` joint ``i``'s unit twist at ``q = 0`` seen from the end-effector frame, in
ssik's ``(v, w)`` order (linear, angular -- the order of :func:`ssik.chart.se3_exp` and
:func:`ssik.refinement.kinbody_jacobian`). A revolute joint about the line through ``p``
along ``a`` has ``B = (p x a, a)``; a prismatic joint along ``a`` has ``B = (a, 0)``.

A target that fixes less than a pose -- a tool axis on a line, a tool plane on a plane, a
tool point on a point -- is a coset ``T0 exp(span(G))``: ``G`` spans the primitive's
stabilizer, body-frame twists composed on the right. If the last ``k`` twists
``B_{n-k+1}, ..., B_n`` lie in ``span(G)``, the tail ``exp(B_{n-k+1} q_{n-k+1}) ...
exp(B_n q_n)`` is an element of that group, so

    FK(q) in T0 exp(span(G))   <=>   FK(0) exp(B_1 q_1) ... exp(B_{n-k} q_{n-k}) in T0 exp(span(G)),

and those ``k`` joints drop out of the target: any values inside their limits keep it
satisfied. Read geometrically, a revolute joint whose axis is the line ``a`` is freed by

    a line L    iff  a == L                  (the tool axis on the target line)
    a plane P   iff  a is normal to P        (at any position: every normal spin is in G)
    a point p   iff  p lies on a             (a spherical wrist centred on p frees all three)

The test is linear, :func:`free_tail`, so it needs no model of the primitive: only its
twists. Derivation and its checks against the chart API: self-motion-charts
``derivations/coset_targets.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik._kinbody import KinBody

__all__ = ["body_twists", "free_tail"]

# Membership residual, relative to |B_i|. The twists are exact geometry, so a real member
# lands at rounding (1e-15) and a real miss far above; 1e-9 sits between the two.
_FREE_TOL = 1e-9


def _adjoint(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Adjoint of ``T`` acting on ``(v, w)`` twists."""
    R, p = T[:3, :3], T[:3, 3]
    P = np.array([[0.0, -p[2], p[1]], [p[2], 0.0, -p[0]], [-p[1], p[0], 0.0]])
    out = np.zeros((6, 6))
    out[:3, :3] = out[3:, 3:] = R
    out[:3, 3:] = P @ R
    return out


def body_twists(kb: KinBody) -> NDArray[np.float64]:
    """``(n, 6)`` joint twists at ``q = 0`` in the end-effector frame, ``(v, w)`` order.

    Walks the chain at ``q = 0`` (``T_left @ T_right`` per joint), so it does not rely on
    the chain being POE-normalised.
    """
    T = np.eye(4)
    space = []
    for joint in kb.joints:
        T = T @ joint.T_left
        a = T[:3, :3] @ np.asarray(joint.axis, dtype=np.float64)
        a = a / np.linalg.norm(a)
        if joint.joint_type == "prismatic":
            space.append(np.r_[a, 0.0, 0.0, 0.0])
        else:
            space.append(np.r_[np.cross(T[:3, 3], a), a])
        T = T @ joint.T_right
    Ad = _adjoint(np.asarray(np.linalg.inv(T), dtype=np.float64))
    return np.asarray(np.asarray(space, dtype=np.float64) @ Ad.T, dtype=np.float64)


def free_tail(twists: ArrayLike, generators: ArrayLike, *, tol: float = _FREE_TOL) -> int:
    """How many of the last joints a target leaves free: the longest suffix of ``twists``
    inside ``span(generators)``.

    :param twists: ``(n, 6)`` body twists, as :func:`body_twists` returns them.
    :param generators: ``(d, 6)`` body-frame twists spanning the target's stabilizer, in
        the end-effector frame, ``(v, w)`` order; a single ``(6,)`` twist is one generator.
        Their scale and basis do not matter, only their span.
    :param tol: largest residual of a twist against the span, relative to its norm, that
        still counts as inside.
    :returns: ``k``: joints ``n-k+1..n`` drop out of the target. A suffix, so a free joint
        with a non-free joint after it does not count.
    :raises ValueError: when ``generators`` is not ``(d, 6)``.
    """
    B = np.asarray(twists, dtype=np.float64)
    G = np.asarray(generators, dtype=np.float64)
    if G.ndim == 1 and G.shape == (6,):
        G = G[None, :]
    if G.ndim != 2 or G.shape[1] != 6:
        raise ValueError(f"generators must be (d, 6) twists; got shape {G.shape}")
    if not len(G):
        return 0
    k = 0
    for b in B[::-1]:
        c, *_ = np.linalg.lstsq(G.T, b, rcond=None)
        if np.linalg.norm(G.T @ c - b) > tol * np.linalg.norm(b):
            break
        k += 1
    return k
