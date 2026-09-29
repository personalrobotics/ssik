"""Machine-precision polish of accepted general-6R candidates.

The Raghavan-Roth (``ikgeo.general_6r``) path accepts an algebraic candidate
when its FK residual is at most the acceptance gate (``subproblem_numerical``,
1e-5). Its eigen-solve is ill-conditioned, so an accepted angle is only as
accurate as ``residual / sigma_min(J)``: near a joint limit it can be off by
~1e-5 rad, and which side of the limit it lands on then follows the machine's
LAPACK kernels. A few Newton steps from the accepted candidate reach round-off.

This is distinct from ``allow_refinement`` (#74), which tries to rescue a
candidate that *fails* the gate and stays opt-in. Polish touches only
candidates that already passed, runs by default, and either returns the same
branch at machine precision or leaves the candidate exactly as it was.

One definition, two backends
----------------------------

The native backend (``cpp/include/ssik_cpp/polish.hpp``) implements exactly
this algorithm; a change here must be mirrored there.

Given an accepted candidate ``q0``, the target ``T`` and the FK round-off
floor ``f = same_root_floor(T)``::

    r0 = ||FK(q0) - T||_F;  if r0 <= f: return q0 unchanged
    q, best = q0, (q0, r0)
    for k in 1 .. POLISH_MAX_ITERS:
        dq = clip((J^T J + 1e-9 I)^-1 J^T log(T FK(q)^-1), +-0.5)   # J = J(q)
        if k == 1: eta = ||dq||_2
        q = q + dq;  r = ||FK(q) - T||_F
        if r >= best.r: break          # stopped improving
        best = (q, r)
        if r <= f: break               # at round-off
    accept best iff best.r <= POLISH_TARGET and ||best.q - q0||_2 <= 2 eta
    otherwise return (q0, r0) unchanged

The step is :func:`ssik.refinement.lm_refine_batch`'s (and the rescue's): the
fixed-damping normal equations solved by partial-pivot LU.

Why the acceptance rule cannot swap branches. ``eta`` is the first Newton
step, the linearised distance from ``q0`` to its own root (at most
``r0 / sigma_min(J)``). By the Newton-Kantorovich theorem, when the iteration
from ``q0`` converges quadratically (``h <= 1/2``) it converges to the unique
root in the ball of radius ``2 eta`` about ``q0``, and never leaves that ball.
A polished point farther than ``2 eta`` from ``q0`` is therefore a trajectory
that left the candidate's own basin -- possibly toward a twin branch -- and is
rejected. A candidate that does not reach ``POLISH_TARGET`` (a singular pose,
where the damping shortens the steps, or a root that is not simple) is also
left exactly as the solver produced it. So polish never moves a candidate by
more than twice its own error estimate, never makes its residual worse, and
runs before same-root deduplication, which then sees each branch at
machine precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssik.refinement import _se3_log_residual_batch, same_root_floor

__all__ = ["POLISH_MAX_ITERS", "POLISH_TARGET", "Chain", "polish_accepted"]

# Residual a polished candidate must reach to replace the original: seven
# orders of magnitude below the 1e-5 acceptance gate and well above FK
# round-off (~1e-15 for a metre-scale arm), so a regular root always reaches
# it in one or two steps.
POLISH_TARGET = 1e-12

# Newton steps. Convergence is quadratic from an accepted candidate: 1e-5 ->
# ~1e-10 -> round-off, so two steps suffice at a regular pose; four leave room
# for moderately conditioned ones.
POLISH_MAX_ITERS = 4

# lm_refine_batch's (and the rescue polish's) step parameters.
_DAMPING = 1e-9
_STEP_CLIP = 0.5

_EYE4 = np.eye(4, dtype=np.float64)


class Chain:
    """A revolute chain ``FK(q) = prod_i left_i Rot(axis_i, q_i) right_i``.

    Evaluates FK and the spatial Jacobian of a batch of configurations in one
    vectorised pass. Jacobian column ``i`` is ``(p_i x z_i ; z_i)``, joint
    ``i``'s screw in the world frame: the convention of
    :func:`ssik.refinement.kinbody_jacobian`, and the one the twist of
    :func:`ssik.refinement.se3_log_residual` is measured in, so the polish's
    Newton step converges quadratically.
    """

    def __init__(
        self, left: NDArray[np.float64], axes: NDArray[np.float64], right: NDArray[np.float64]
    ) -> None:
        self.left = np.asarray(left, dtype=np.float64)
        axes = np.asarray(axes, dtype=np.float64)
        self.axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        self.right = np.asarray(right, dtype=np.float64)
        self.dof = self.axes.shape[0]
        self._left_is_eye = bool(np.all(self.left == _EYE4))

    @classmethod
    def from_kinbody(cls, kb: Any) -> Chain:
        """The POE chain of a revolute :class:`~ssik._kinbody.KinBody`."""
        joints = kb.joints
        if any(j.joint_type != "revolute" for j in joints):
            raise ValueError("Chain is revolute-only")
        return cls(
            np.array([j.T_left for j in joints]),
            np.array([j.axis for j in joints]),
            np.array([j.T_right for j in joints]),
        )

    @classmethod
    def from_dh(cls, dh: tuple[Any, Any, Any]) -> Chain:
        """A standard distal DH chain ``(alpha, a, d)``: ``Rz(q) Tz(d) Tx(a) Rx(alpha)``."""
        alpha, a, d = (np.asarray(x, dtype=np.float64) for x in dh)
        n = alpha.shape[0]
        right = np.broadcast_to(_EYE4, (n, 4, 4)).copy()
        ca, sa = np.cos(alpha), np.sin(alpha)
        right[:, 0, 3] = a
        right[:, 1, 1], right[:, 1, 2] = ca, -sa
        right[:, 2, 1], right[:, 2, 2], right[:, 2, 3] = sa, ca, d
        axes = np.zeros((n, 3))
        axes[:, 2] = 1.0
        return cls(np.broadcast_to(_EYE4, (n, 4, 4)).copy(), axes, right)

    def fk(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """FK ``(N, 4, 4)`` for ``q`` of shape ``(N, dof)``."""
        links = self._links(q)
        acc = links[:, 0]
        for i in range(1, self.dof):
            acc = acc @ links[:, i]
        return acc

    def _links(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-joint link transforms ``left_i Rot(axis_i, q_i) right_i``, ``(N, dof, 4, 4)``."""
        q = np.asarray(q, dtype=np.float64)
        n = q.shape[0]
        c, s = np.cos(q), np.sin(q)
        oc = 1.0 - c
        ax, ay, az = self.axes[:, 0], self.axes[:, 1], self.axes[:, 2]
        rot = np.zeros((n, self.dof, 4, 4))
        rot[..., 0, 0] = c + ax * ax * oc
        rot[..., 0, 1] = ax * ay * oc - az * s
        rot[..., 0, 2] = ax * az * oc + ay * s
        rot[..., 1, 0] = ay * ax * oc + az * s
        rot[..., 1, 1] = c + ay * ay * oc
        rot[..., 1, 2] = ay * az * oc - ax * s
        rot[..., 2, 0] = az * ax * oc - ay * s
        rot[..., 2, 1] = az * ay * oc + ax * s
        rot[..., 2, 2] = c + az * az * oc
        rot[..., 3, 3] = 1.0
        links = rot @ self.right if self._left_is_eye else self.left @ rot @ self.right
        return links

    def fk_jacobian(
        self, q: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(FK (N, 4, 4), spatial Jacobian (N, 6, dof))`` for ``q`` of shape ``(N, dof)``."""
        n = q.shape[0]
        links = self._links(q)
        jac = np.empty((n, 6, self.dof))
        acc = np.broadcast_to(_EYE4, (n, 4, 4))
        for i in range(self.dof):
            p = acc if self._left_is_eye else acc @ self.left[i]
            z = p[:, :3, :3] @ self.axes[i]
            o = p[:, :3, 3]
            jac[:, 0, i] = o[:, 1] * z[:, 2] - o[:, 2] * z[:, 1]
            jac[:, 1, i] = o[:, 2] * z[:, 0] - o[:, 0] * z[:, 2]
            jac[:, 2, i] = o[:, 0] * z[:, 1] - o[:, 1] * z[:, 0]
            jac[:, 3:, i] = z
            acc = acc @ links[:, i]
        return np.array(acc), jac


def _inv_rigid(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of a stack of rigid transforms ``(N, 4, 4)``."""
    rt = np.swapaxes(t[:, :3, :3], 1, 2)
    out = np.zeros_like(t)
    out[:, :3, :3] = rt
    out[:, :3, 3] = -(rt @ t[:, :3, 3, None])[..., 0]
    out[:, 3, 3] = 1.0
    return out


def polish_accepted(
    q0: NDArray[np.float64],
    t_target: NDArray[np.float64],
    chain: Chain,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Polish accepted candidates to machine precision (module docstring).

    :param q0: ``(N, dof)`` candidates that already passed the acceptance gate.
    :param t_target: ``(4, 4)`` target pose, in the frame of ``chain``.
    :param chain: the chain whose FK the candidates close.
    :returns: ``(q, residual, polished)``, shapes ``(N, dof)``, ``(N,)`` and
        ``(N,)``. Where ``polished`` is true the row is the polished candidate
        and its residual; elsewhere the acceptance rule rejected the polish and
        the row is ``q0``'s, unchanged (its residual as evaluated here; callers
        that already measured it keep their own value).
    """
    q0 = np.asarray(q0, dtype=np.float64)
    n, dof = q0.shape
    t_target = np.asarray(t_target, dtype=np.float64)
    if n == 0:
        return q0.copy(), np.empty(0, dtype=np.float64), np.zeros(0, dtype=bool)
    floor = same_root_floor(t_target)

    fk, jac = chain.fk_jacobian(q0)
    r0 = np.linalg.norm((fk - t_target).reshape(n, -1), axis=1)
    q = q0.copy()
    best_q = q0.copy()
    best_r = r0.copy()
    eta = np.zeros(n, dtype=np.float64)
    active = r0 > floor
    eye = np.eye(dof, dtype=np.float64)

    for k in range(POLISH_MAX_ITERS):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        j = jac[idx]
        jt = np.swapaxes(j, 1, 2)
        twist = _se3_log_residual_batch(t_target @ _inv_rigid(fk[idx]))
        dq = np.linalg.solve(jt @ j + _DAMPING * eye, jt @ twist[..., None])[..., 0]
        dq = np.clip(dq, -_STEP_CLIP, _STEP_CLIP)
        if k == 0:
            eta[idx] = np.linalg.norm(dq, axis=1)
        q[idx] += dq
        fk[idx] = chain.fk(q[idx])
        r = np.linalg.norm((fk[idx] - t_target).reshape(idx.size, -1), axis=1)
        improved = r < best_r[idx]
        better = idx[improved]
        best_q[better] = q[better]
        best_r[better] = r[improved]
        active[idx] = improved & (r > floor)
        # The Jacobian is needed only where another step follows.
        more = np.flatnonzero(active)
        if more.size and k + 1 < POLISH_MAX_ITERS:
            _, jac[more] = chain.fk_jacobian(q[more])

    ok = (best_r <= POLISH_TARGET) & (best_r < r0)
    ok &= np.linalg.norm(best_q - q0, axis=1) <= 2.0 * eta
    return np.where(ok[:, None], best_q, q0), np.where(ok, best_r, r0), ok
