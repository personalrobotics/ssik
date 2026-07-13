"""Exact feasible-swivel (joint-limit-aware) resolution for SRS-class 7R (#359).

The elbow swivel ``psi`` is the 1-DOF redundancy. For a fixed IK branch every
joint is a closed-form function ``q_i(psi)`` because the shoulder rotation is

    R_sh(psi) = Rot(u_sw, psi) @ R_sh(0)

(rotating the elbow about the shoulder-wrist axis *is* the swivel). So the set
of ``psi`` with all joints inside their limits is computed *exactly* -- no
sampling -- and we return in-limits solutions directly. The uniform swivel
sweep in :mod:`ssik.solvers.seven_r.srs` samples ``psi`` blindly and can miss a
narrow in-limits arc (a reachable in-limits pose then returns ``[]``); this
closes that gap.

Method (Shimizu, Kim, Kakuya & Freeman, IEEE T-RO 24(5), 2008), generalised
here to arbitrary concurrent axes -- the same generalisation that #356 applied
to the base SRS solver -- so the closed forms hold for non-Z*Z shoulders/wrists:

1. Enumerate the <=8 discrete branches (2 elbow x 2 shoulder-sign x 2 wrist-sign)
   natively at ``psi = 0`` from the general-path geometry (``R_sh(0)``, ``q_3``).
2. Per branch, intersect the 6 non-elbow joints' feasible-``psi`` arcs -- the
   parameter-agnostic feasible-interval core in :mod:`._feasible_param` (shared
   with the non-SRS locked-joint redundancy of #148); union over branches.
3. Sample representative ``psi`` (arc centres = max limit margin) and emit the
   wrapped-to-limits joint vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics._generalized_euler import _axis_angle_matrix as _rot
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import (
    SrsClassification,
    _classify_srs_7r_geometric,
    is_approximately_srs_7r,
)
from ssik.refinement import dedup_by_wrap_close, kinbody_jacobian, lm_refine_batch
from ssik.solvers.seven_r._feasible_param import (
    PARAM_GRID,
    feasible_arcs,
    to_limits,
)
from ssik.solvers.seven_r.srs import (
    _arm_constants,
    _min_rotation,
    _rodrigues_batch,
    _sp4_branches,
    _swivel_basis,
)

if TYPE_CHECKING:  # pragma: no cover
    from ssik._kinbody import KinBody

_EPS = 1e-9


def _signed_angle(k: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Angle about unit ``k`` carrying ``a``'s perp component onto ``b``'s."""
    ap = a - k * (k @ a)
    bp = b - k * (k @ b)
    return float(np.arctan2(k @ np.cross(ap, bp), ap @ bp))


def _signed_angle_batch(
    k: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Vectorised :func:`_signed_angle` over ``a, b`` of shape ``(N, 3)``."""
    ap = a - k[None, :] * (a @ k)[:, None]
    bp = b - k[None, :] * (b @ k)[:, None]
    out: NDArray[np.float64] = np.arctan2(np.cross(ap, bp) @ k, (ap * bp).sum(axis=1))
    return out


def _triple_phase(
    m0: NDArray[np.float64], m1: NDArray[np.float64], m2: NDArray[np.float64]
) -> tuple[float, float, float]:
    """``rho, delta, gamma`` with ``m0 . Rot(m1,q) m2 == rho cos(q - delta) + gamma``."""
    a = float((m0 @ m2) - (m0 @ m1) * (m1 @ m2))
    b = float(m0 @ np.cross(m1, m2))
    return float(np.hypot(a, b)), float(np.arctan2(b, a)), float((m0 @ m1) * (m1 @ m2))


class _Branch:
    """One discrete IK branch: fixed elbow ``(R_sh0, q3)`` + shoulder/wrist sign.

    ``q(psi)`` is the closed-form, continuous joint vector along the swivel.
    """

    __slots__ = (
        "R_post",
        "R_sh0",
        "R_t",
        "_ps",
        "_pw",
        "n",
        "q3",
        "s_sgn",
        "u_sw",
        "w_sgn",
    )

    def __init__(
        self,
        n: list[NDArray[np.float64]],
        u_sw: NDArray[np.float64],
        R_sh0: NDArray[np.float64],
        q3: float,
        R_t: NDArray[np.float64],
        R_post: NDArray[np.float64],
        s_sgn: int,
        w_sgn: int,
    ) -> None:
        self.n = n
        self.u_sw = u_sw
        self.R_sh0 = R_sh0
        self.q3 = q3
        self.R_t = R_t
        self.R_post = R_post
        self.s_sgn = s_sgn
        self.w_sgn = w_sgn
        self._ps = _triple_phase(n[0], n[1], n[2])
        self._pw = _triple_phase(n[4], n[5], n[6])

    def q(self, psi: float) -> NDArray[np.float64]:
        n = self.n
        R_sh = _rot(self.u_sw, psi) @ self.R_sh0
        rho, dlt, gam = self._ps
        q1 = dlt + self.s_sgn * float(
            np.arccos(np.clip((n[0] @ R_sh @ n[2] - gam) / rho, -1.0, 1.0))
        )
        q0 = _signed_angle(n[0], _rot(n[1], q1) @ n[2], R_sh @ n[2])
        q2 = -_signed_angle(n[2], _rot(n[1], -q1) @ n[0], R_sh.T @ n[0])
        R_res = (R_sh @ _rot(n[3], self.q3)).T @ self.R_t @ self.R_post.T
        rho, dlt, gam = self._pw
        q5 = dlt + self.w_sgn * float(
            np.arccos(np.clip((n[4] @ R_res @ n[6] - gam) / rho, -1.0, 1.0))
        )
        q4 = _signed_angle(n[4], _rot(n[5], q5) @ n[6], R_res @ n[6])
        q6 = -_signed_angle(n[6], _rot(n[5], -q5) @ n[4], R_res.T @ n[4])
        return np.array([q0, q1, q2, self.q3, q4, q5, q6], dtype=np.float64)

    def q_grid(self, psis: NDArray[np.float64]) -> NDArray[np.float64]:
        """Vectorised :meth:`q` over a ``(N,)`` swivel grid -> ``(N, 7)``."""
        n = self.n
        rsh = _rodrigues_batch(self.u_sw, psis) @ self.R_sh0  # (N,3,3)
        rho, dlt, gam = self._ps
        q1 = dlt + self.s_sgn * np.arccos(
            np.clip((np.einsum("i,nij,j->n", n[0], rsh, n[2]) - gam) / rho, -1.0, 1.0)
        )
        rsh_n2 = rsh @ n[2]
        q0 = _signed_angle_batch(n[0], _rodrigues_batch(n[1], q1) @ n[2], rsh_n2)
        q2 = -_signed_angle_batch(
            n[2], _rodrigues_batch(n[1], -q1) @ n[0], rsh.transpose(0, 2, 1) @ n[0]
        )
        m = rsh @ _rot(n[3], self.q3)  # (N,3,3)
        rres = m.transpose(0, 2, 1) @ (self.R_t @ self.R_post.T)  # (N,3,3)
        rho, dlt, gam = self._pw
        q5 = dlt + self.w_sgn * np.arccos(
            np.clip((np.einsum("i,nij,j->n", n[4], rres, n[6]) - gam) / rho, -1.0, 1.0)
        )
        q4 = _signed_angle_batch(n[4], _rodrigues_batch(n[5], q5) @ n[6], rres @ n[6])
        q6 = -_signed_angle_batch(
            n[6], _rodrigues_batch(n[5], -q5) @ n[4], rres.transpose(0, 2, 1) @ n[4]
        )
        return np.stack([q0, q1, q2, np.full_like(q0, self.q3), q4, q5, q6], axis=1)


def _branch_arcs(branch: _Branch, limits: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # The elbow q3 is fixed along the swivel: pre-check it, then sweep the rest.
    if not (limits[3][0] <= branch.q3 <= limits[3][1]):
        return []
    q_grid = branch.q_grid(PARAM_GRID)  # (N, 7) -- one batched eval per branch
    return feasible_arcs(branch.q, q_grid, (0, 1, 2, 4, 5, 6), limits, PARAM_GRID)


def _enumerate_branches(
    kb: KinBody, cls: SrsClassification, T: NDArray[np.float64]
) -> list[_Branch]:
    """All <=8 branches, with each ``R_sh(0)`` built natively at ``psi=0``."""
    n = [np.asarray(j.axis, dtype=np.float64) / float(np.linalg.norm(j.axis)) for j in kb.joints]
    L_se, L_ew, ee, origins = _arm_constants(kb, cls)
    S = cls.shoulder_pivot
    R_t = T[:3, :3]
    W_t = T[:3, 3] - R_t @ ee
    R_post = kb.joints[6].T_right[:3, :3]
    upper_home = origins[cls.elbow_index] - S
    forearm_home = cls.wrist_pivot - origins[cls.elbow_index]
    n3 = n[cls.elbow_index]

    d_sw = float(np.linalg.norm(W_t - S))
    if d_sw < _EPS or not (abs(L_se - L_ew) < d_sw < L_se + L_ew):
        return []
    u_sw = (W_t - S) / d_sw
    x_c = (L_se**2 - L_ew**2 + d_sw**2) / (2.0 * d_sw)
    r_circle = float(np.sqrt(max(L_se**2 - x_c**2, 0.0)))
    u_p1, _u_p2 = _swivel_basis(u_sw)

    # psi = 0 geometry (cos 0 = 1, sin 0 = 0 -> elbow on u_p1)
    elbow0 = S + x_c * u_sw + r_circle * u_p1
    upper0 = elbow0 - S
    d0 = upper0 / L_se
    r0 = _min_rotation(upper_home, upper0)
    wrist_vec0 = W_t - elbow0

    branches: list[_Branch] = []
    for q3 in _sp4_branches(d0, r0 @ n3, r0 @ forearm_home, float(wrist_vec0 @ d0)):
        g = r0 @ (_rot(n3, q3) @ forearm_home)
        g_perp = g - d0 * float(d0 @ g)
        w_perp = wrist_vec0 - d0 * float(d0 @ wrist_vec0)
        if float(np.linalg.norm(g_perp)) < _EPS:
            phi = 0.0
        else:
            phi = float(np.arctan2(float(d0 @ np.cross(g_perp, w_perp)), float(g_perp @ w_perp)))
        R_sh0 = _rot(d0, phi) @ r0
        for s_sgn in (+1, -1):
            for w_sgn in (+1, -1):
                branches.append(_Branch(n, u_sw, R_sh0, q3, R_t, R_post, s_sgn, w_sgn))
    return branches


def feasible_in_limits_solutions(
    kb: KinBody,
    cls: SrsClassification,
    T: NDArray[np.float64],
    limits: list[tuple[float, float]],
    *,
    max_solutions: int | None = None,
    samples_per_arc: int = 1,
) -> list[NDArray[np.float64]]:
    """In-limits IK solutions via feasible-swivel resolution.

    Returns wrapped-to-limits joint vectors. ``samples_per_arc=1`` (the exact-SRS
    default) takes each arc's centre (maximum joint-limit margin); >1 spreads
    that many interior samples per arc -- used by the approximate-SRS path, where
    each seed is LM-polished and only some stay in-limits after the polish shift.
    Empty iff no in-limits solution exists (target unreachable-in-limits).
    """
    T = np.asarray(T, dtype=np.float64)
    out: list[NDArray[np.float64]] = []
    for branch in _enumerate_branches(kb, cls, T):
        for a, b in _branch_arcs(branch, limits):
            psis = (
                [0.5 * (a + b)]
                if samples_per_arc == 1
                else np.linspace(a, b, samples_per_arc + 2)[1:-1].tolist()
            )
            for psi in psis:
                q = branch.q(float(psi))
                out.append(
                    np.array([to_limits(float(q[i]), limits[i][0], limits[i][1]) for i in range(7)])
                )
                if max_solutions is not None and len(out) >= max_solutions:
                    return out
    return out


def _joint_limits(kb: KinBody) -> list[tuple[float, float]]:
    lims: list[tuple[float, float]] = []
    for j in kb.joints:
        lo_hi = j.limits
        if lo_hi is None or lo_hi[0] is None or lo_hi[1] is None:
            lims.append((-np.pi, np.pi))
        else:
            lims.append((float(lo_hi[0]), float(lo_hi[1])))
    return lims


_APPROX_MAX_DRIFT_M = 0.04  # matches seven_r.srs_polished's default gate
_APPROX_SAMPLES_PER_ARC = 5  # LM shift can push the arc centre out of limits
_APPROX_FK_ATOL = 1e-8


def _in_limits(q: NDArray[np.float64], limits: list[tuple[float, float]]) -> bool:
    return all(lo - 1e-9 <= q[i] <= hi + 1e-9 for i, (lo, hi) in enumerate(limits))


def resolve_in_limits(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    max_solutions: int | None = None,
) -> list[Solution]:
    """Joint-limit-aware IK for SRS-class 7R via feasible-swivel resolution.

    Intended as the ``respect_limits`` fallback for the SRS-family prebuilt
    ``solve()``: when the blind swivel sweep samples no in-limits candidate, the
    feasible-swivel arcs recover the in-limits solution(s) (#359). Returns
    FK-verified, in-limits :class:`Solution` objects; empty when the chain is not
    SRS-class or no in-limits solution exists.

    Exactly-concurrent SRS chains are solved in closed form (machine precision).
    Approximately-SRS chains (e.g. Kinova Gen3) run the resolver on the best-fit
    pivots to seed candidates, then LM-polish each to machine-precision FK against
    the true URDF, keeping those still in-limits (#370).
    """
    if len(kb.joints) != 7:
        return []
    T = np.asarray(T_target, dtype=np.float64)
    limits = _joint_limits(kb)

    cls = _classify_srs_7r_geometric(kb, policy)
    if cls is not None:
        fk_atol = policy.subproblem_numerical
        exact: list[Solution] = []
        for q in feasible_in_limits_solutions(kb, cls, T, limits, max_solutions=max_solutions):
            residual = float(np.linalg.norm(poe_forward_kinematics(kb, q) - T))
            if residual <= fk_atol:
                exact.append(Solution(q=q, fk_residual=residual, refinement_used="none"))
        return exact

    # Approximately-SRS (#370): best-fit resolver seeds + LM polish.
    approx = is_approximately_srs_7r(kb, max_drift_m=_APPROX_MAX_DRIFT_M, policy=policy)
    if approx is None:
        return []
    seeds = feasible_in_limits_solutions(
        kb, approx.base, T, limits, samples_per_arc=_APPROX_SAMPLES_PER_ARC
    )
    if not seeds:
        return []

    def _fk(q: NDArray[np.float64]) -> NDArray[np.float64]:
        t: NDArray[np.float64] = poe_forward_kinematics(kb, q)
        return t

    def _jac(q: NDArray[np.float64]) -> NDArray[np.float64]:
        j: NDArray[np.float64] = kinbody_jacobian(kb, q)
        return j

    q_polished, residuals, _iters = lm_refine_batch(np.asarray(seeds), _fk, _jac, T)
    polished: list[Solution] = [
        Solution(q=q_polished[i], fk_residual=float(residuals[i]), refinement_used="lm")
        for i in range(q_polished.shape[0])
        if residuals[i] <= _APPROX_FK_ATOL and _in_limits(q_polished[i], limits)
    ]
    out = dedup_by_wrap_close(polished, policy.subproblem_dedup)
    return out[:max_solutions] if max_solutions is not None else out
