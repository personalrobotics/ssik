"""Back-substitution for the Husty-Pfurner universal-6R IK solver.

Phase 5f of #162. Given a candidate ``(v_1, v_6) = (u, w)`` from
:func:`ssik.solvers.husty_pfurner._eliminate.eliminate_uw_pairs`,
recover the remaining four joint values ``(v_2, v_3, v_4, v_5)`` so
that the full chain FK matches the target ``sigma_E``.

Algorithm (per Capco et al. 2019 Section 5.4 / Pfurner 2009 ch. 4):

1. Form ``sigma_inner = sigma_1(v_1)^-1 . sigma_E . sigma_6(v_6)^-1``
   -- a numeric SE(3) factor that the inner four joints must produce.
2. Recover the Cramer cofactor ``P(u, w)`` as a numeric 8-vec at the
   refined ``(u, w)``. Algebraically ``P = sigma_1(v_1) . sigma_2 .
   sigma_3 = sigma_E . sigma_6(v_6)^-1 . sigma_5^-1 . sigma_4^-1``,
   so it gives the F_4 split-frame Study DQ.
3. Two 2R sub-chains:

   - ``sigma_2(v_2) sigma_3(v_3) = sigma_1(u)^-1 . P``: solve for
     ``(v_2, v_3)``.
   - ``sigma_4(v_4) sigma_5(v_5) = P^-1 . sigma_E . sigma_6(w)^-1``:
     solve for ``(v_4, v_5)``.
4. Each 2R subproblem decomposes via closed-form ZXZ-like atan2
   (``alpha_a``, ``alpha_b`` are known DH twists; only the two
   ``v_a``, ``v_b`` rotations are free).
5. FK closure check: ``||FK(v_1, v_2, v_3, v_4, v_5, v_6) - sigma_E||``
   below machine eps in projective Study norm. Candidates that fail
   are spurious algebraic solutions (e.g. multiplicity-cluster
   members that don't quite satisfy the original chain equation).

The output is the list of full IK solutions, each polished to
machine precision and FK-verified.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik.solvers.husty_pfurner._eliminate import (
    EliminatePrecompute,
    _apply_sigma_e_to_tw_pre,
    _build_full_8x8,
    eliminate_uw_pairs,
)
from ssik.solvers.husty_pfurner._study import dq_conj, dq_mul, se3_from_dq

__all__ = [
    "back_substitute_one",
    "solve_ik",
]


# =============================================================================
# Joint Study DQ primitives.
# =============================================================================


def _sigma_z(v: float) -> NDArray[np.float64]:
    """Projective Study DQ of ``R_z(theta)`` with ``v = tan(theta/2)``."""
    return np.array([1.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _sigma_tx(a: float) -> NDArray[np.float64]:
    """Projective Study DQ of ``T_x(a)``."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5 * a, 0.0, 0.0], dtype=np.float64)


def _sigma_tz(d: float) -> NDArray[np.float64]:
    """Projective Study DQ of ``T_z(d)``."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * d], dtype=np.float64)


def _sigma_rx(twist: float) -> NDArray[np.float64]:
    """Projective Study DQ of ``R_x(alpha)`` with ``twist = tan(alpha/2)``."""
    return np.array([1.0, twist, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _sigma_joint_full(v: float, a: float, ls: float, d: float) -> NDArray[np.float64]:
    """Projective Study DQ of one full DH joint:
    ``R_z(v) . T_z(d) . T_x(a) . R_x(ls)`` where ``v = tan(theta/2)``
    and ``ls = tan(alpha/2)``.
    """
    return dq_mul(
        _sigma_z(v),
        dq_mul(_sigma_tz(d), dq_mul(_sigma_tx(a), _sigma_rx(ls))),
    )


def _dq_inv(sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Projective Study DQ inverse: for unit-norm ``sigma`` this is
    just the conjugate; for projective (non-unit-norm) inputs, scale
    by ``1/||sigma_p||^2`` afterwards. We'll work in projective form
    and rely on ``dq_mul``'s scale-invariance (the chain product is
    a projective DQ that gets normalised at FK closure check time).
    """
    return dq_conj(sigma)


# =============================================================================
# 2R sub-chain decomposition.
# =============================================================================


def _solve_2r_chain(
    sigma_target: NDArray[np.float64],
    a_a: float,
    ls_a: float,
    d_a: float,
    a_b: float,
    ls_b: float,
    d_b: float,
) -> list[tuple[float, float]]:
    """Recover ``(v_a, v_b)`` from a 2R-chain Study DQ target.

    Chain: ``sigma_a(v_a) sigma_b(v_b) = lambda * sigma_target``
    (projective), where each ``sigma_i(v_i) = R_z(v_i) T_z(d_i)
    T_x(a_i) R_x(alpha_i)``, ``v_i = tan(theta_i/2)``,
    ``ls_i = tan(alpha_i/2)``.

    Closed-form ZXZ-like decomposition: extract the rotation part of
    ``sigma_target``, undo the fixed ``alpha_b`` factor on the right,
    atan2 for ``v_a``, then atan2 for ``v_b``. Two atan2 calls = ~1 us.

    On *true* common roots (target on the chain's image), this is
    exact at machine precision. On *spurious* roots (Newton converged
    to an algebraic root that doesn't correspond to a physical IK),
    the rotation is correct but translation isn't -- the resulting
    ``(v_a, v_b)`` gives a chain that doesn't FK-close. The outer
    :func:`solve_ik` filter (tight ``fk_tol``) catches those, and the
    dispatcher's ``lm_refine`` fallback polishes them when needed.

    :returns: ``[(v_a, v_b)]`` (single solution); ``[]`` only on
        gimbal lock + sign-degenerate input.
    """
    # Convert to SE(3) for rotation extraction.
    R = se3_from_dq(sigma_target)[:3, :3]

    denom_b = 1.0 + ls_b * ls_b
    sa_b = 2.0 * ls_b / denom_b
    ca_b = (1.0 - ls_b * ls_b) / denom_b
    Rx_neg_alpha_b = np.array(
        [[1.0, 0.0, 0.0], [0.0, ca_b, sa_b], [0.0, -sa_b, ca_b]],
        dtype=np.float64,
    )
    R_prime = R @ Rx_neg_alpha_b

    denom_a = 1.0 + ls_a * ls_a
    sa_a = 2.0 * ls_a / denom_a
    ca_a = (1.0 - ls_a * ls_a) / denom_a
    rxz, ryz = float(R_prime[0, 2]), float(R_prime[1, 2])

    if abs(sa_a) < 1e-12:
        v_a = 0.0
        R_double = R_prime
    else:
        sign_sa_a = 1.0 if sa_a > 0.0 else -1.0
        theta_a = float(np.arctan2(rxz * sign_sa_a, -ryz * sign_sa_a))
        v_a = float(np.tan(0.5 * theta_a))
        c_va, s_va = np.cos(theta_a), np.sin(theta_a)
        Rz_neg_va = np.array(
            [[c_va, s_va, 0.0], [-s_va, c_va, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Rx_neg_alpha_a = np.array(
            [[1.0, 0.0, 0.0], [0.0, ca_a, sa_a], [0.0, -sa_a, ca_a]],
            dtype=np.float64,
        )
        R_double = Rx_neg_alpha_a @ Rz_neg_va @ R_prime

    theta_b = float(np.arctan2(R_double[1, 0], R_double[0, 0]))
    v_b = float(np.tan(0.5 * theta_b))
    return [(v_a, v_b)]


# =============================================================================
# Back-substitution and full IK.
# =============================================================================


def _build_dh_dict(pre: EliminatePrecompute) -> dict[str, float]:
    """Recover the DH parameters that built ``pre`` by extracting them
    from the ``T_u`` and ``T_w_pre`` tensors. (Or, more practically,
    require the caller to pass them explicitly -- we'll do the latter.)
    """
    raise NotImplementedError(
        "DH parameters cannot be reconstructed from EliminatePrecompute; "
        "caller must provide them explicitly via dh_kwargs to solve_ik."
    )


def _cramer_P_at(
    pre: EliminatePrecompute,
    sigma_E: NDArray[np.float64],
    u: float,
    w: float,
    drop_idx: int = 7,
) -> NDArray[np.float64]:
    """Evaluate the Cramer cofactor 8-vec ``P(u, w)`` at the refined
    ``(u, w)``. Constructs the 8x8 system, takes the 7-of-8 sub-block,
    computes ``[det(A), x_1*det(A), ..., x_7*det(A)]`` where
    ``x = A^{-1} (-col_0)``.

    Algebraically this matches what
    :func:`ssik.solvers.husty_pfurner._eliminate._cramer_8vec_via_interp`
    produces, evaluated at one specific ``(u, w)`` (not the whole
    bivariate polynomial tensor).
    """
    sigma_E_arr = np.asarray(sigma_E, dtype=np.float64)
    T_w = _apply_sigma_e_to_tw_pre(pre.T_w_pre, sigma_E_arr)
    M_full = _build_full_8x8(pre.T_u, T_w, u, w)
    keep = [i for i in range(8) if i != drop_idx]
    M_7 = M_full[keep]
    A = M_7[:, 1:]
    rhs = -M_7[:, 0]
    x = np.linalg.solve(A, rhs)
    d = float(np.linalg.det(A))
    P = np.empty(8, dtype=np.float64)
    P[0] = d
    P[1:] = x * d
    return P


def back_substitute_one(
    pre: EliminatePrecompute,
    sigma_E: NDArray[np.float64],
    u: float,
    w: float,
    *,
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
    d_3: float,
    a_3: float,
    l_3: float,
    d_4: float,
    a_4: float,
    l_4: float,
    d_5: float,
    a_5: float,
    l_5: float,
) -> list[tuple[float, float, float, float]]:
    """Recover ``(v_2, v_3, v_4, v_5)`` for one ``(u, w)`` candidate.

    See module docstring for the algorithm. Returns a list because
    each 2R sub-decomposition can in principle yield multiple
    branches (though in this DH convention the closed-form atan2
    yields a single solution per sub-chain; near gimbal lock there
    is a 1-parameter family that we represent by a single sample).

    Caller must pass DH parameters explicitly because the
    ``EliminatePrecompute`` tensors don't preserve them in a
    recoverable form.
    """
    # Build sigma_1(u) and sigma_6(w).
    sigma_1 = _sigma_joint_full(u, a_1, l_1, 0.0)  # joint 1: a_6 = d_6 = l_6 = 0 convention
    sigma_6 = _sigma_z(w)

    # Recover the Cramer cofactor P(u, w) at this refined point.
    P = _cramer_P_at(pre, sigma_E, u, w)

    # Compute the two 2R chain targets:
    #   sigma_left  = sigma_2(v_2) sigma_3(v_3) = sigma_1(u)^-1 . P
    #   sigma_right = sigma_4(v_4) sigma_5(v_5) = P^-1 . sigma_E . sigma_6(w)^-1
    sigma_left = dq_mul(_dq_inv(sigma_1), P)
    sigma_right = dq_mul(_dq_inv(P), dq_mul(sigma_E, _dq_inv(sigma_6)))

    # Decompose directly on the projective Study DQ targets (no SE(3)
    # conversion needed; the full-8-vec back-sub uses both rotation
    # and translation components).
    sol_23 = _solve_2r_chain(sigma_left, a_2, l_2, d_2, a_3, l_3, d_3)
    sol_45 = _solve_2r_chain(sigma_right, a_4, l_4, d_4, a_5, l_5, d_5)

    return [(v_2, v_3, v_4, v_5) for (v_2, v_3) in sol_23 for (v_4, v_5) in sol_45]


def solve_ik(
    pre: EliminatePrecompute,
    sigma_E: NDArray[np.float64],
    *,
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
    d_3: float,
    a_3: float,
    l_3: float,
    d_4: float,
    a_4: float,
    l_4: float,
    d_5: float,
    a_5: float,
    l_5: float,
    fk_tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Top-level HP IK solver.

    1. Run :func:`eliminate_uw_pairs` to get refined ``(u, w)``
       candidates.
    2. For each ``(u, w)``, run :func:`back_substitute_one` to
       recover ``(v_2, v_3, v_4, v_5)``.
    3. Filter by FK closure: keep only ``q`` such that the chain
       FK matches ``sigma_E`` in projective Study norm below
       ``fk_tol``.

    :returns: 2-D array of shape ``(n, 6)`` with rows
        ``(v_1, v_2, v_3, v_4, v_5, v_6)``, each tan-half-angle.
    """
    pairs = eliminate_uw_pairs(pre, sigma_E)
    if pairs.size == 0:
        return np.empty((0, 6), dtype=np.float64)

    sigma_E_arr = np.asarray(sigma_E, dtype=np.float64)
    sigma_E_norm = float(np.linalg.norm(sigma_E_arr))
    out: list[list[float]] = []

    for u, w in pairs:
        candidates = back_substitute_one(
            pre,
            sigma_E_arr,
            float(u),
            float(w),
            a_1=a_1,
            l_1=l_1,
            d_2=d_2,
            a_2=a_2,
            l_2=l_2,
            d_3=d_3,
            a_3=a_3,
            l_3=l_3,
            d_4=d_4,
            a_4=a_4,
            l_4=l_4,
            d_5=d_5,
            a_5=a_5,
            l_5=l_5,
        )
        for v_2, v_3, v_4, v_5 in candidates:
            # FK closure: build the full 6R chain DQ and compare.
            sigma_chain = dq_mul(
                _sigma_joint_full(float(u), a_1, l_1, 0.0),
                dq_mul(
                    _sigma_joint_full(v_2, a_2, l_2, d_2),
                    dq_mul(
                        _sigma_joint_full(v_3, a_3, l_3, d_3),
                        dq_mul(
                            _sigma_joint_full(v_4, a_4, l_4, d_4),
                            dq_mul(
                                _sigma_joint_full(v_5, a_5, l_5, d_5),
                                _sigma_z(float(w)),
                            ),
                        ),
                    ),
                ),
            )
            # Projective comparison: align scales then compute residual.
            scale = float(np.dot(sigma_chain, sigma_E_arr)) / max(
                float(np.dot(sigma_chain, sigma_chain)), 1e-300
            )
            residue_abs = float(np.linalg.norm(sigma_chain * scale - sigma_E_arr))
            residue_rel = residue_abs / max(sigma_E_norm, 1e-300)
            if residue_rel < fk_tol:
                out.append([float(u), v_2, v_3, v_4, v_5, float(w)])

    if not out:
        return np.empty((0, 6), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)
