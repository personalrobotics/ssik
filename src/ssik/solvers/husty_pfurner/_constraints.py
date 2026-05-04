"""Constraint quadrics for the Husty-Pfurner algorithm (Phase 5c).

Implements the parametrised hyperplane equations from Capco et al. 2019
Section 3 / equation (5). The HP algorithm splits the 6R chain at the
middle frame F_4 and represents each side's workspace as a parametrised
3-space in P^7 (4 hyperplane equations in the Study coordinates
``(x_0, x_1, x_2, x_3, y_0, y_1, y_2, y_3)``).

**Phase 5c step 1 scope** (this file): the SIMPLIFIED form -- four
hyperplanes for the inner-2-chain workspace ``V_1 = {R_z(v_2) T_x(a_2)
R_x(l_2) R_z(v_3) | v_2, v_3 in R}`` of the RRR case. Equation (5) of
the paper, with the ``a_1, l_1 -> a_2, l_2`` substitution noted in
Capco's RRR paragraph after eq. (5).

**Subsequent phases** (5c.2 / 5c.3 / ...) will:

- Apply Capco eq. (4) change of variables to inject the parametrising
  joint ``v_1`` and produce ``T(v_1)`` -- the actual 3-space the
  HP elimination consumes.
- Add the analogous ``T(v_3)`` parametrisation (Capco's V_3 case in
  the RRR setup) and ``T(v_2)`` (the harder fallback used when the
  primary parametrisations land on the Study quadric).
- Add the 6R/P variants RRP, RPR, RPP, PRR, PPR (Capco's per-pattern
  case files).

**Coordinate convention**: Capco's algorithm parametrises rotations
via tan-half-angle: ``v = tan(theta/2)``, ``l = tan(alpha/2)``. The
Study coordinates are then projective polynomials in those parameters,
which makes the hyperplane equations linear in ``(x_0, ..., y_3)``
with coefficients polynomial in DH parameters.

Algorithmic reference: Capco, Loquias, Manongsong, Nemenzo (2019),
'Inverse Kinematics of Some General 6R/P Manipulators',
arXiv 1906.07813.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "hyperplane_residuals",
    "tv1_hyperplanes_rrr",
    "tv1_symbolic_in_v1",
    "tv3_hyperplanes_rrr",
    "tv3_symbolic_in_v3",
    "tv6_hyperplanes_rrr",
    "tv6_symbolic_in_v6",
    "v1_hyperplanes_rrr",
]


# Canonical sympy symbols used by all symbolic helpers in this module.
# Cached once at import time to avoid repeated allocation; tests and
# downstream code (``_eliminate.py``) reuse these for substitution.
_V1_SYM = sp.symbols("v_1", real=True)
_V3_SYM = sp.symbols("v_3", real=True)
_V6_SYM = sp.symbols("v_6", real=True)


def v1_hyperplanes_rrr(a_2: float, l_2: float) -> NDArray[np.float64]:
    """4x8 coefficient matrix of the ``V_1`` hyperplanes for the RRR case.

    Per Capco et al. 2019 equation (5), with the ``a_1 -> a_2`` and
    ``l_1 -> l_2`` substitution from the RRR adaptation in the paragraph
    after eq. (5)::

        a_2 l_2  x_0                 + 2     y_0                 = 0
                -a_2 x_1                     + 2 l_2 y_1         = 0
                       -a_2 x_2                     + 2 l_2 y_2  = 0
                              a_2 l_2 x_3                  + 2 y_3 = 0

    Each row of the returned matrix gives the coefficients of
    ``(x_0, x_1, x_2, x_3, y_0, y_1, y_2, y_3)`` in one hyperplane.

    These four hyperplanes vanish identically on the projective Study
    DQ of ``R_z(v_2) T_x(a_2) R_x(l_2) R_z(v_3)`` for any ``v_2, v_3 in R``
    (when ``v = tan(theta/2)`` and ``l = tan(alpha/2)``).

    **Precondition**: ``a_2 != 0`` and ``l_2 != 0`` (alpha_2 not equal
    to ``0`` or ``pi``, modulo the half-angle convention). The
    ``a_1 = l_1 = 0`` degenerate case (Capco eq. before (5)) yields a
    different linear 3-space; future steps will dispatch the degenerate
    branch via a separate function.
    """
    al = a_2 * l_2
    two_l = 2.0 * l_2
    return np.array(
        [
            [al, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, -a_2, 0.0, 0.0, 0.0, two_l, 0.0, 0.0],
            [0.0, 0.0, -a_2, 0.0, 0.0, 0.0, two_l, 0.0],
            [0.0, 0.0, 0.0, al, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )


def hyperplane_residuals(
    coeffs: NDArray[np.float64], sigma: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate the K hyperplane equations at a Study 8-vec ``sigma``.

    Returns a ``(K,)`` array of linear-form values ``coeffs @ sigma``.
    For ``sigma`` taken from the parametrised workspace these
    constraints describe (e.g. ``v1_hyperplanes_rrr`` evaluated on a
    DQ from ``R_z(v_2) T_x(a_2) R_x(l_2) R_z(v_3)``), every entry is
    zero up to floating-point noise.
    """
    return coeffs @ sigma


# ---------------------------------------------------------------------------
# Phase 5c step 2: T(v_1) -- V_1 hyperplanes after change of variables (Capco
# eq. 4) to inject the parametrising joint v_1.
#
# Strategy: symbolic preprocessing via sympy, then ``sp.lambdify`` to a
# numpy-vectorised callable. The lambdify is one-time at module-import; the
# runtime ``tv1_hyperplanes_rrr`` is pure numpy substitution.
# ---------------------------------------------------------------------------


def _quat_mul_sym(p: sp.Matrix, q: sp.Matrix) -> sp.Matrix:
    """Hamilton product of two sympy 4-vec quaternions."""
    return sp.Matrix(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
            p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
        ]
    )


def _dq_mul_sym(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    """Symbolic dual-quaternion product of two 8-vec sympy matrices."""
    pa = sp.Matrix([a[0], a[1], a[2], a[3]])
    qa = sp.Matrix([a[4], a[5], a[6], a[7]])
    pb = sp.Matrix([b[0], b[1], b[2], b[3]])
    qb = sp.Matrix([b[4], b[5], b[6], b[7]])
    p = _quat_mul_sym(pa, pb)
    q = _quat_mul_sym(pa, qb) + _quat_mul_sym(qa, pb)
    return sp.Matrix([p[0], p[1], p[2], p[3], q[0], q[1], q[2], q[3]])


def _dq_conj_sym(s: sp.Matrix) -> sp.Matrix:
    """Symbolic dual-quaternion conjugate (negate imaginary parts of both halves)."""
    return sp.Matrix([s[0], -s[1], -s[2], -s[3], s[4], -s[5], -s[6], -s[7]])


@lru_cache(maxsize=1)
def _build_tv1_rrr_symbolic() -> tuple[sp.Matrix, tuple[sp.Symbol, ...]]:
    """Build the symbolic 4x8 ``T(v_1)`` coefficient matrix and the ordered
    tuple of free symbols ``(v_1, a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3)``.

    Symbolic pipeline:

    1. Build ``LEFT = R_z(v_1) T_x(a_1) R_x(l_1) T_z(d_2)`` and
       ``RIGHT = T_z(d_3) T_x(a_3) R_x(l_3)`` as projective sympy 8-vecs.
    2. Compute ``tau = LEFT^* * sigma * RIGHT^*`` symbolically; this maps a
       point ``sigma`` in ``V_L`` to a (scalar-multiple of a) point in
       ``V_1`` per Capco eq. (4).
    3. Apply the four ``V_1`` hyperplanes from eq. (5) to ``tau``.
    4. Extract the coefficient of each ``sigma`` component to get the
       ``T(v_1)`` 4x8 coefficient matrix.

    Cached so all callers (lambdified runtime + symbolic-substitution helpers
    used by elimination) share one symbolic preprocessing pass.
    """
    v_1 = _V1_SYM
    a_1, l_1, d_2 = sp.symbols("a_1 l_1 d_2", real=True)
    a_2, l_2 = sp.symbols("a_2 l_2", real=True)
    d_3, a_3, l_3 = sp.symbols("d_3 a_3 l_3", real=True)
    x_syms = sp.symbols("x_0 x_1 x_2 x_3 y_0 y_1 y_2 y_3", real=True)
    sigma_sym = sp.Matrix(x_syms)

    half = sp.Rational(1, 2)
    one = sp.Integer(1)
    zero = sp.Integer(0)

    def rz(v: sp.Symbol) -> sp.Matrix:
        return sp.Matrix([one, zero, zero, v, zero, zero, zero, zero])

    def tx(a: sp.Symbol) -> sp.Matrix:
        return sp.Matrix([one, zero, zero, zero, zero, half * a, zero, zero])

    def tz(d: sp.Symbol) -> sp.Matrix:
        return sp.Matrix([one, zero, zero, zero, zero, zero, zero, half * d])

    def rx(t: sp.Symbol) -> sp.Matrix:
        return sp.Matrix([one, t, zero, zero, zero, zero, zero, zero])

    left = _dq_mul_sym(rz(v_1), _dq_mul_sym(tx(a_1), _dq_mul_sym(rx(l_1), tz(d_2))))
    right = _dq_mul_sym(tz(d_3), _dq_mul_sym(tx(a_3), rx(l_3)))

    left_conj = _dq_conj_sym(left)
    right_conj = _dq_conj_sym(right)

    tau = _dq_mul_sym(left_conj, _dq_mul_sym(sigma_sym, right_conj))

    v1_coeffs = sp.Matrix(
        [
            [a_2 * l_2, zero, zero, zero, sp.Integer(2), zero, zero, zero],
            [zero, -a_2, zero, zero, zero, sp.Integer(2) * l_2, zero, zero],
            [zero, zero, -a_2, zero, zero, zero, sp.Integer(2) * l_2, zero],
            [zero, zero, zero, a_2 * l_2, zero, zero, zero, sp.Integer(2)],
        ]
    )

    hyperplanes_vl = v1_coeffs * tau
    c_new_sym = sp.zeros(4, 8)
    for i in range(4):
        expr_i = sp.expand(hyperplanes_vl[i])
        for j in range(8):
            c_new_sym[i, j] = expr_i.coeff(x_syms[j])

    return c_new_sym, (v_1, a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3)


@lru_cache(maxsize=1)
def _build_tv1_rrr_lambdified() -> Callable[..., NDArray[np.float64]]:
    """Lambdified runtime version of ``T(v_1)``: numpy callable taking
    ``(v_1, a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3)`` and returning a 4x8
    numpy array.
    """
    c_new_sym, args = _build_tv1_rrr_symbolic()
    return sp.lambdify(args, c_new_sym, modules="numpy")  # type: ignore[no-any-return]


def tv1_symbolic_in_v1(
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
    d_3: float,
    a_3: float,
    l_3: float,
) -> sp.Matrix:
    """Return the 4x8 ``T(v_1)`` coefficient matrix as a sympy ``Matrix``
    with ``v_1`` symbolic and DH parameters substituted numerically.

    Used by the elimination pipeline (Phase 5d) to build the linear system
    over ``C(v_1, v_6)``. ``ssik.solvers.husty_pfurner._constraints._V1_SYM``
    is the sympy symbol used for ``v_1`` (cached at module import).
    """
    c_sym, (_v_1, p_a1, p_l1, p_d2, p_a2, p_l2, p_d3, p_a3, p_l3) = _build_tv1_rrr_symbolic()
    subs = {
        p_a1: sp.Float(a_1),
        p_l1: sp.Float(l_1),
        p_d2: sp.Float(d_2),
        p_a2: sp.Float(a_2),
        p_l2: sp.Float(l_2),
        p_d3: sp.Float(d_3),
        p_a3: sp.Float(a_3),
        p_l3: sp.Float(l_3),
    }
    return c_sym.subs(subs)


def tv1_hyperplanes_rrr(
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
    d_3: float,
    a_3: float,
    l_3: float,
    v_1: float,
) -> NDArray[np.float64]:
    """4x8 coefficient matrix of ``T(v_1)`` for the RRR case at parameter ``v_1``.

    Applies Capco eq. (4) change of variables to the V_1 hyperplanes
    (see :func:`v1_hyperplanes_rrr`) so the resulting 4 hyperplanes
    contain the full left-chain workspace ``V_L = {sigma_1(v_1)
    sigma_2(v_2) sigma_3(v_3) | v_2, v_3 in R}`` with ``v_1`` injected
    as a numeric parameter.

    DH parameter convention (Capco): rotations parametrised by tan-half-angle,
    so ``v_1 = tan(theta_1 / 2)`` and ``l_1 = tan(alpha_1 / 2)``. Distances
    ``a_i`` and offsets ``d_i`` are linear DH parameters.

    The argument order matches the natural DH walk: parameters of joints
    1, 2, 3 in sequence, ``v_1`` last (it's the parametrising free variable).

    For any ``v_1, v_2, v_3 in R``, the 4 hyperplanes returned here vanish
    on the projective Study DQ of the full RRR chain
    ``sigma_1(v_1) sigma_2(v_2) sigma_3(v_3)`` within floating-point noise.
    """
    fn = _build_tv1_rrr_lambdified()
    result = np.asarray(
        fn(v_1, a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3),
        dtype=np.float64,
    )
    if result.shape != (4, 8):
        raise RuntimeError(  # pragma: no cover
            f"tv1_hyperplanes_rrr produced shape {result.shape}, expected (4, 8)"
        )
    return result


# ---------------------------------------------------------------------------
# Phase 5c step 3: T(v_3) -- alternative parametrisation by the third joint.
#
# T(v_1) requires ``a_2 != 0 ∧ l_2 != 0`` (V_1's eq. (5) precondition). Real
# arms with ``l_2 = ±1`` (alpha_2 = ±90deg, the most common DH twist on
# industrial 6R like UR5, Puma) do NOT satisfy this. ``T(v_3)`` is the
# alternative: parametrise V_L by ``v_3`` instead of ``v_1``, using V_3's
# hyperplanes (which depend on ``a_1, l_1``).
#
# Structure mirrors T(v_1) but with the change of variables on the RIGHT
# only:
#
#   V_L = V_3 · POST(v_3)
#       = {R_z(v_1) T_x(a_1) R_x(l_1) R_z(v_2)} ·
#         T_z(d_2) T_x(a_2) R_x(l_2) R_z(v_3) T_z(d_3) T_x(a_3) R_x(l_3)
#
# So ``tau = sigma · POST^*`` and the V_3 hyperplanes (eq. 5 with ``a_1, l_1``
# directly, no index swap) vanish on tau.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_tv3_rrr_symbolic() -> tuple[sp.Matrix, tuple[sp.Symbol, ...]]:
    """Symbolic 4x8 ``T(v_3)`` Matrix and ordered ``(v_3, a_1, l_1, d_2,
    a_2, l_2, d_3, a_3, l_3)`` symbol tuple.

    Differs from ``T(v_1)`` in two ways: V_3 hyperplanes use ``(a_1, l_1)``
    directly (Capco eq. 5, no index swap), and the change of variables
    is one-sided on the right (``tau = sigma · POST^*``).
    """
    v_3 = _V3_SYM
    a_1, l_1 = sp.symbols("a_1 l_1", real=True)
    d_2, a_2, l_2 = sp.symbols("d_2 a_2 l_2", real=True)
    d_3, a_3, l_3 = sp.symbols("d_3 a_3 l_3", real=True)
    x_syms = sp.symbols("x_0 x_1 x_2 x_3 y_0 y_1 y_2 y_3", real=True)
    sigma_sym = sp.Matrix(x_syms)

    half = sp.Rational(1, 2)
    one = sp.Integer(1)
    zero = sp.Integer(0)

    def rz(v: sp.Symbol) -> sp.Matrix:
        return sp.Matrix([one, zero, zero, v, zero, zero, zero, zero])

    def tx(a: sp.Symbol) -> sp.Matrix:
        return sp.Matrix([one, zero, zero, zero, zero, half * a, zero, zero])

    def tz(d: sp.Symbol) -> sp.Matrix:
        return sp.Matrix([one, zero, zero, zero, zero, zero, zero, half * d])

    def rx(t: sp.Symbol) -> sp.Matrix:
        return sp.Matrix([one, t, zero, zero, zero, zero, zero, zero])

    post = _dq_mul_sym(
        tz(d_2),
        _dq_mul_sym(
            tx(a_2),
            _dq_mul_sym(
                rx(l_2),
                _dq_mul_sym(rz(v_3), _dq_mul_sym(tz(d_3), _dq_mul_sym(tx(a_3), rx(l_3)))),
            ),
        ),
    )
    post_conj = _dq_conj_sym(post)
    tau = _dq_mul_sym(sigma_sym, post_conj)

    v3_coeffs = sp.Matrix(
        [
            [a_1 * l_1, zero, zero, zero, sp.Integer(2), zero, zero, zero],
            [zero, -a_1, zero, zero, zero, sp.Integer(2) * l_1, zero, zero],
            [zero, zero, -a_1, zero, zero, zero, sp.Integer(2) * l_1, zero],
            [zero, zero, zero, a_1 * l_1, zero, zero, zero, sp.Integer(2)],
        ]
    )

    hyperplanes_vl = v3_coeffs * tau
    c_new_sym = sp.zeros(4, 8)
    for i in range(4):
        expr_i = sp.expand(hyperplanes_vl[i])
        for j in range(8):
            c_new_sym[i, j] = expr_i.coeff(x_syms[j])

    return c_new_sym, (v_3, a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3)


@lru_cache(maxsize=1)
def _build_tv3_rrr_lambdified() -> Callable[..., NDArray[np.float64]]:
    """Lambdified runtime function for ``T(v_3)`` in RRR. Numpy callable
    taking ``(v_3, a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3)``.
    """
    c_new_sym, args = _build_tv3_rrr_symbolic()
    return sp.lambdify(args, c_new_sym, modules="numpy")  # type: ignore[no-any-return]


def tv3_symbolic_in_v3(
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
    d_3: float,
    a_3: float,
    l_3: float,
) -> sp.Matrix:
    """Return the 4x8 ``T(v_3)`` coefficient matrix as a sympy ``Matrix``
    with ``v_3`` symbolic and DH parameters substituted numerically.

    Parallel to :func:`tv1_symbolic_in_v1` for the alternative left-chain
    parametrisation. The free symbol is
    ``ssik.solvers.husty_pfurner._constraints._V3_SYM``.
    """
    c_sym, (_v_3, p_a1, p_l1, p_d2, p_a2, p_l2, p_d3, p_a3, p_l3) = _build_tv3_rrr_symbolic()
    subs = {
        p_a1: sp.Float(a_1),
        p_l1: sp.Float(l_1),
        p_d2: sp.Float(d_2),
        p_a2: sp.Float(a_2),
        p_l2: sp.Float(l_2),
        p_d3: sp.Float(d_3),
        p_a3: sp.Float(a_3),
        p_l3: sp.Float(l_3),
    }
    return c_sym.subs(subs)


def tv3_hyperplanes_rrr(
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
    d_3: float,
    a_3: float,
    l_3: float,
    v_3: float,
) -> NDArray[np.float64]:
    """4x8 coefficient matrix of ``T(v_3)`` for the RRR case at parameter ``v_3``.

    The alternative parametrisation of V_L when ``T(v_1)`` lies in the
    Study quadric (which happens iff ``l_2 = ±1`` in Capco's convention
    -- alpha_2 = ±pi/2, very common on industrial arms like UR5 / Puma).

    DH parameter convention (Capco): rotations parametrised by
    tan-half-angle, so ``v_3 = tan(theta_3 / 2)`` and ``l_i = tan(alpha_i
    / 2)``. Distances ``a_i`` and offsets ``d_i`` are linear DH parameters.

    Argument order matches T(v_1) (DH params for joints 1-3 then the
    parametrising joint ``v_3`` last).

    For any ``v_1, v_2, v_3 in R``, the 4 hyperplanes vanish on the
    projective Study DQ of the full RRR chain
    ``sigma_1(v_1) sigma_2(v_2) sigma_3(v_3)`` within floating-point noise.

    Precondition: ``a_1 != 0 ∧ l_1 != 0`` (V_3's eq. (5) precondition).
    For arms with both ``l_1 = 0`` and ``l_2 = ±1``, neither T(v_1) nor
    T(v_3) applies; the T(v_2) fallback (next phase) is required.
    """
    fn = _build_tv3_rrr_lambdified()
    result = np.asarray(
        fn(v_3, a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3),
        dtype=np.float64,
    )
    if result.shape != (4, 8):
        raise RuntimeError(  # pragma: no cover
            f"tv3_hyperplanes_rrr produced shape {result.shape}, expected (4, 8)"
        )
    return result


# ---------------------------------------------------------------------------
# Phase 5d step 1: T(v_6) -- right-chain hyperplanes parametrised by joint 6.
#
# Per Capco eq. (6), the right chain
#     V_R = {sigma_E sigma_6^{-1}(v_6) sigma_5^{-1}(v_5) sigma_4^{-1}(v_4)}
# can be obtained from the LEFT-chain T(v_1) construction by:
#
#   1. Parameter substitutions (sign flips + index re-mapping):
#        v_1 -> -v_6, a_1 -> -a_5, l_1 -> -l_5, d_2 -> -d_5
#        v_2 -> -v_5, a_2 -> -a_4, l_2 -> -l_4
#        v_3 -> -v_4, a_3 -> 0,   l_3 -> 0,    d_3 -> -d_4
#   2. A final change of variables ``sigma -> sigma_E^* . sigma`` (left
#      multiplication by the conjugate of the target end-effector pose).
#
# Convention: a_6 = d_6 = l_6 = 0 -- the EE site / last-link offset must be
# absorbed into ``sigma_E`` by the caller (left-multiplying the end-pose
# by the joint-6 EE offset DQ before passing it here).
# ---------------------------------------------------------------------------


def _quat_left_mult_matrix(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 4x4 matrix ``L(p)`` such that ``L(p) @ q == p * q`` for
    any quaternion ``q``.

    Hamilton product written as a linear map in ``q``.
    """
    p0, p1, p2, p3 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    return np.array(
        [
            [p0, -p1, -p2, -p3],
            [p1, p0, -p3, p2],
            [p2, p3, p0, -p1],
            [p3, -p2, p1, p0],
        ],
        dtype=np.float64,
    )


def _dq_left_mult_matrix(eta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 8x8 matrix ``M_eta`` such that ``M_eta @ sigma == eta * sigma``
    for any 8-vec dual quaternion ``sigma`` (using ``ssik.solvers.husty_pfurner._study.dq_mul``).

    Block structure: top-left and bottom-right are ``L(eta_p)`` (the rotation
    part of ``eta`` acts on both halves of ``sigma``); bottom-left is
    ``L(eta_q)`` (the translation part of ``eta`` mixes ``sigma_p`` into
    the result's translation half); top-right is zero (``sigma_q`` does
    not contribute to the rotation half of the product).
    """
    eta_p = eta[:4]
    eta_q = eta[4:]
    Lp = _quat_left_mult_matrix(eta_p)
    Lq = _quat_left_mult_matrix(eta_q)
    M = np.zeros((8, 8), dtype=np.float64)
    M[:4, :4] = Lp
    M[4:, :4] = Lq
    M[4:, 4:] = Lp
    return M


def tv6_hyperplanes_rrr(
    a_4: float,
    l_4: float,
    d_4: float,
    a_5: float,
    l_5: float,
    d_5: float,
    sigma_E: NDArray[np.float64],
    v_6: float,
) -> NDArray[np.float64]:
    """4x8 coefficient matrix of ``T(v_6)`` for the right chain in 6R/RRR.

    Reuses the lambdified ``T(v_1)`` machinery via Capco eq. (6) parameter
    substitutions, then applies the ``sigma_E^*``-left-multiplication
    change of variables.

    :param a_4, l_4, d_4: DH parameters for joint 4 (``l_4 = tan(alpha_4/2)``).
    :param a_5, l_5, d_5: DH parameters for joint 5.
    :param sigma_E: 8-vec Study DQ of the target end-effector pose. Caller
        must absorb the joint-6 EE offset into ``sigma_E`` (since this
        function assumes ``a_6 = d_6 = l_6 = 0`` per Capco's convention).
    :param v_6: tan-half-angle of joint-6 rotation.

    Precondition: ``a_5 != 0 ∧ l_5 != 0`` (mirrors V_1's eq. 5
    precondition, transposed for the right chain).

    Returns a 4x8 array; each row is the coefficients of ``(x_0, ..., y_3)``
    in one of four hyperplanes that vanish on
    ``V_R = sigma_E . sigma_6^{-1} . sigma_5^{-1} . sigma_4^{-1}``.
    """
    coeffs_pre_sigma_e = tv1_hyperplanes_rrr(
        a_1=-a_5,
        l_1=-l_5,
        d_2=-d_5,
        a_2=-a_4,
        l_2=-l_4,
        d_3=-d_4,
        a_3=0.0,
        l_3=0.0,
        v_1=-v_6,
    )
    sigma_e_arr = np.asarray(sigma_E, dtype=np.float64)
    if sigma_e_arr.shape != (8,):
        raise ValueError(f"sigma_E must be 8-vec, got shape {sigma_e_arr.shape}")
    sigma_e_conj = np.array(
        [
            sigma_e_arr[0],
            -sigma_e_arr[1],
            -sigma_e_arr[2],
            -sigma_e_arr[3],
            sigma_e_arr[4],
            -sigma_e_arr[5],
            -sigma_e_arr[6],
            -sigma_e_arr[7],
        ],
        dtype=np.float64,
    )
    M_e = _dq_left_mult_matrix(sigma_e_conj)
    return coeffs_pre_sigma_e @ M_e


def tv6_symbolic_in_v6(
    a_4: float,
    l_4: float,
    d_4: float,
    a_5: float,
    l_5: float,
    d_5: float,
    sigma_E: NDArray[np.float64],
) -> sp.Matrix:
    """Return the 4x8 ``T(v_6)`` coefficient matrix as a sympy ``Matrix``
    with ``v_6`` symbolic and DH + ``sigma_E`` substituted numerically.

    Mirrors :func:`tv6_hyperplanes_rrr` but keeps ``v_6`` symbolic for use
    in the elimination pipeline. The free symbol is
    ``ssik.solvers.husty_pfurner._constraints._V6_SYM``.

    Implementation: build ``T(v_1)`` symbolic with the Capco eq. (6)
    parameter substitutions (``v_1 -> -v_6``, ``a_1 -> -a_5``, etc.) so
    the result is a sympy matrix in ``v_6``. Then apply the
    ``sigma_E^*`` left-multiplication numerically (since ``sigma_E`` is
    given as a fully-numeric 8-vec).
    """
    sigma_e_arr = np.asarray(sigma_E, dtype=np.float64)
    if sigma_e_arr.shape != (8,):
        raise ValueError(f"sigma_E must be 8-vec, got shape {sigma_e_arr.shape}")
    # Step 1-3: substituted T(v_1) symbolic. We DO NOT call
    # tv1_symbolic_in_v1 directly because that returns a Matrix in v_1 --
    # we want it in v_6. Substitute v_1 -> -v_6 after.
    tv1_sub = tv1_symbolic_in_v1(
        a_1=-a_5,
        l_1=-l_5,
        d_2=-d_5,
        a_2=-a_4,
        l_2=-l_4,
        d_3=-d_4,
        a_3=0.0,
        l_3=0.0,
    )
    # Substitute v_1 -> -v_6.
    coeffs_pre_sigma_e = tv1_sub.subs(_V1_SYM, -_V6_SYM)

    # Step 4: numerical sigma_E^* left-mult.
    sigma_e_conj = np.array(
        [
            sigma_e_arr[0],
            -sigma_e_arr[1],
            -sigma_e_arr[2],
            -sigma_e_arr[3],
            sigma_e_arr[4],
            -sigma_e_arr[5],
            -sigma_e_arr[6],
            -sigma_e_arr[7],
        ],
        dtype=np.float64,
    )
    M_e = _dq_left_mult_matrix(sigma_e_conj)
    M_e_sym = sp.Matrix(M_e.tolist())
    return coeffs_pre_sigma_e * M_e_sym
