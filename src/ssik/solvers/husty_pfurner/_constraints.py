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
    "TV2_RRR_CASE_KEYS",
    "hyperplane_residuals",
    "tv1_hyperplanes_rrr",
    "tv1_symbolic_in_v1",
    "tv2_hyperplanes_rrr",
    "tv2_rrr_case_for",
    "tv2_symbolic_in_v2",
    "tv3_hyperplanes_rrr",
    "tv3_symbolic_in_v3",
    "tv4_hyperplanes_rrr",
    "tv4_symbolic_in_v4",
    "tv6_hyperplanes_rrr",
    "tv6_symbolic_in_v6",
    "v1_hyperplanes_rrr",
]


# Canonical sympy symbols used by all symbolic helpers in this module.
# Cached once at import time to avoid repeated allocation; tests and
# downstream code (``_eliminate.py``) reuse these for substitution.
_V1_SYM = sp.symbols("v_1", real=True)
_V2_SYM = sp.symbols("v_2", real=True)
_V3_SYM = sp.symbols("v_3", real=True)
_V4_SYM = sp.symbols("v_4", real=True)
_V6_SYM = sp.symbols("v_6", real=True)


# Capco's RRR T(v_2) sub-case keys. Each one identifies which DH
# parameter pair is zero in the kinematic chain that triggers the
# double-degenerate branch (``T(v_1)`` AND ``T(v_3)`` both lie in the
# Study quadric). See `which_case.py:get_Tvd2_key1` and
# `rrr.py:Tv2_cases` in Capco's Zenodo 3157441 reference code.
TV2_RRR_CASE_KEYS: tuple[str, ...] = (
    "[a_1=0,a_2=0]",
    "[a_1=0,l_2=0]",
    "[l_1=0,a_2=0]",
    "[l_1=0,l_2=0]",
)


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


def _quat_right_mult_matrix(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 4x4 matrix ``R(p)`` such that ``R(p) @ q == q * p`` for
    any quaternion ``q``.

    Right-mult is non-commutative with left-mult; the column ordering
    differs from :func:`_quat_left_mult_matrix`.
    """
    p0, p1, p2, p3 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    return np.array(
        [
            [p0, -p1, -p2, -p3],
            [p1, p0, p3, -p2],
            [p2, -p3, p0, p1],
            [p3, p2, -p1, p0],
        ],
        dtype=np.float64,
    )


def _dq_right_mult_matrix(eta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 8x8 matrix ``M_eta`` such that ``M_eta @ sigma == sigma * eta``
    for any 8-vec dual quaternion ``sigma`` (using ``ssik.solvers.husty_pfurner._study.dq_mul``).

    Block structure mirrors :func:`_dq_left_mult_matrix` but with
    right-mult quaternion matrices: top-left and bottom-right are
    ``R(eta_p)``; bottom-left is ``R(eta_q)``.
    """
    eta_p = eta[:4]
    eta_q = eta[4:]
    Rp = _quat_right_mult_matrix(eta_p)
    Rq = _quat_right_mult_matrix(eta_q)
    M = np.zeros((8, 8), dtype=np.float64)
    M[:4, :4] = Rp
    M[4:, :4] = Rq
    M[4:, 4:] = Rp
    return M


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


# ---------------------------------------------------------------------------
# Phase 5c.4b / GitHub #177: T(v_4) -- right-chain mirror of T(v_3).
#
# Used when T(v_6) is structurally degenerate (typically because joint 4's
# DH (a_4, l_4) lands on a Capco-Tv1 nullspace, e.g., locked-Franka after
# joint-4 lock with a_4 = 0). Built by parameter-mirroring T(v_3): right
# chain joints (4, 5, 6) map to left-analog joints (3, 2, 1) with sign
# flips, and the parametric symbol substitutes v_3 -> -v_4.
#
# Precondition: a_5 != 0 AND l_5 != 0 (Tv3's (a_1, l_1) precondition under
# the mirror substitution a_1 -> -a_5, l_1 -> -l_5). Same precondition as
# T(v_6); the difference is structural -- T(v_4) survives some DH
# pathologies that null T(v_6) coefficients.
# ---------------------------------------------------------------------------


def tv4_hyperplanes_rrr(
    a_4: float,
    l_4: float,
    d_4: float,
    a_5: float,
    l_5: float,
    d_5: float,
    sigma_E: NDArray[np.float64],
    v_4: float,
) -> NDArray[np.float64]:
    """4x8 coefficient matrix of ``T(v_4)`` for the right chain in 6R/RRR.

    Mirror of :func:`tv6_hyperplanes_rrr` but built from ``T(v_3)`` rather
    than ``T(v_1)``. Parametrises the innermost joint of the right chain
    (joint 4) instead of the outermost (joint 6). Used when ``T(v_6)``
    coefficients structurally vanish for the given DH (e.g. locked-Franka
    with ``a_4 = 0``).

    :param a_4, l_4, d_4: DH parameters for joint 4 (``l_4 = tan(alpha_4/2)``).
    :param a_5, l_5, d_5: DH parameters for joint 5.
    :param sigma_E: 8-vec Study DQ of the target end-effector pose. Caller
        must absorb the joint-6 EE offset into ``sigma_E`` (since this
        function assumes ``a_6 = d_6 = l_6 = 0`` per Capco's convention).
    :param v_4: tan-half-angle of joint-4 rotation.

    Precondition: ``a_5 != 0 ∧ l_5 != 0`` (Tv3's (a_1, l_1) precondition
    under the right-chain mirror substitution).

    Returns a 4x8 array; each row is the coefficients of ``(x_0, ..., y_3)``
    in one of four hyperplanes that vanish on
    ``V_R = sigma_E . sigma_6^{-1} . sigma_5^{-1} . sigma_4^{-1}``.
    """
    coeffs_pre_sigma_e = tv3_hyperplanes_rrr(
        a_1=-a_5,
        l_1=-l_5,
        d_2=-d_5,
        a_2=-a_4,
        l_2=-l_4,
        d_3=-d_4,
        a_3=0.0,
        l_3=0.0,
        v_3=-v_4,
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


def tv4_symbolic_in_v4(
    a_4: float,
    l_4: float,
    d_4: float,
    a_5: float,
    l_5: float,
    d_5: float,
    sigma_E: NDArray[np.float64],
) -> sp.Matrix:
    """Return the 4x8 ``T(v_4)`` coefficient matrix as a sympy ``Matrix``
    with ``v_4`` symbolic and DH + ``sigma_E`` substituted numerically.

    Mirrors :func:`tv4_hyperplanes_rrr` but keeps ``v_4`` symbolic for use
    in the elimination pipeline. The free symbol is
    ``ssik.solvers.husty_pfurner._constraints._V4_SYM``.

    Implementation: build ``T(v_3)`` symbolic with the right-chain mirror
    substitutions (``a_1 -> -a_5``, ``l_1 -> -l_5``, ...) so the result is
    a sympy matrix in ``v_3``. Then substitute ``v_3 -> -v_4``. Then apply
    the ``sigma_E^*`` left-multiplication numerically.
    """
    sigma_e_arr = np.asarray(sigma_E, dtype=np.float64)
    if sigma_e_arr.shape != (8,):
        raise ValueError(f"sigma_E must be 8-vec, got shape {sigma_e_arr.shape}")
    tv3_sub = tv3_symbolic_in_v3(
        a_1=-a_5,
        l_1=-l_5,
        d_2=-d_5,
        a_2=-a_4,
        l_2=-l_4,
        d_3=-d_4,
        a_3=0.0,
        l_3=0.0,
    )
    coeffs_pre_sigma_e = tv3_sub.subs(_V3_SYM, -_V4_SYM)

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


# ---------------------------------------------------------------------------
# Phase 5c.4 / GitHub #176: T(v_2) -- the double-degenerate parametrization.
#
# Triggered for RRR when BOTH T(v_1) and T(v_3) lie in the Study quadric:
#
#     (a_2 = 0 OR l_2 = 0)   <-- T(v_1) eq. 5 simplified-form precondition fail
#     AND
#     (a_1 = 0 OR l_1 = 0)   <-- T(v_3) precondition fail (mirror of above)
#
# In this case neither of the closed-form 4x8 hyperplane matrices applies.
# Capco's documented fix (paper Section 5.4 step 2; reference giac code
# ``rrr.py:Tv2_cases`` in Zenodo 3157441) is a 12x16 kernel construction:
#
#  1. Build the symbolic Study DQ chain
#       t = sigma_1(v_1, d_1=0, a_1, l_1)
#         . sigma_2(v_2, d_2, a_2, l_2)
#         . R_z(v_3)            [joint-3 DH absorbed in Tv2_full]
#  2. Form the bilinear expression
#       eqn = sum_i  t[i] * (cf[i] + v_2 * df[i])
#     where (cf, df) are 16 unknown scalars (cf[0..7] for the constant
#     part, df[0..7] for the v_2-linear part).
#  3. Extract coefficients of v_1^v1i * v_2^v2i * v_3^v3i for
#     (v1i, v2i, v3i) in {0,1} x {0,1,2} x {0,1}: 12 row vectors of
#     length 16. Stack as a 12x16 matrix M.
#  4. Substitute the DH-degeneracy condition (e.g. a_1=0 AND a_2=0) into
#     M. The kernel of the resulting matrix has dimension 4 (one basis
#     vector per V_L hyperplane).
#  5. Each kernel basis vector v_k of length 16 gives a hyperplane
#       H_k(xy, v_2) = (v_k[0:8] + v_2 * v_k[8:16]) . xy
#     linear in v_2 and in the Study coords xy.
#
# This yields 4 hyperplanes whose coefficients are linear in v_2, the
# parametrising joint, exactly mirroring the structure ``T(v_1)`` /
# ``T(v_3)`` produces. Downstream elimination (``_eliminate.py``) then
# treats v_2 as the parametric variable u in the Sylvester pencil.
#
# The derivation differs per sub-case because the DH substitution
# happens BEFORE taking the kernel: each of the 4 RRR sub-cases keyed
# in ``TV2_RRR_CASE_KEYS`` produces a different 4-hyperplane system.
# ---------------------------------------------------------------------------


def _tv2_rrr_case_substitution(
    case_key: str,
    a_1: sp.Symbol,
    l_1: sp.Symbol,
    a_2: sp.Symbol,
    l_2: sp.Symbol,
) -> tuple[dict[sp.Symbol, sp.Integer], tuple[sp.Symbol, ...]]:
    """Map a Capco sub-case key to (substitution dict, remaining free DH).

    Returns the symbol-to-zero substitution plus an ordered tuple of
    DH symbols that remain free after the substitution. The caller
    uses the substitution to specialise the 12x16 matrix and the
    remaining-free tuple as ``lambdify`` argument order.
    """
    zero = sp.Integer(0)
    if case_key == "[a_1=0,a_2=0]":
        return {a_1: zero, a_2: zero}, (l_1, l_2)
    if case_key == "[a_1=0,l_2=0]":
        return {a_1: zero, l_2: zero}, (l_1, a_2)
    if case_key == "[l_1=0,a_2=0]":
        return {l_1: zero, a_2: zero}, (a_1, l_2)
    if case_key == "[l_1=0,l_2=0]":
        return {l_1: zero, l_2: zero}, (a_1, a_2)
    raise ValueError(
        f"unknown T(v_2) RRR sub-case {case_key!r}; "
        f"expected one of {TV2_RRR_CASE_KEYS}"
    )


def tv2_rrr_case_for(a_1: float, l_1: float, a_2: float, l_2: float) -> str:
    """Return the appropriate Capco RRR Tv2 sub-case key for given DH.

    Mirrors ``which_case.py:get_Tvd2_key1`` (RRR branch) in Capco's
    reference giac code. Caller is responsible for verifying the
    double-degenerate condition holds (see :func:`tv2_symbolic_in_v2`
    for full preconditions); this function picks among the 4
    sub-cases assuming we already know we need T(v_2).

    :raises ValueError: if no sub-case matches (i.e. the caller
        invoked us when one of T(v_1) or T(v_3) actually applies).
    """
    tol = 1e-9
    a1_zero = abs(a_1) < tol
    l1_zero = abs(l_1) < tol
    a2_zero = abs(a_2) < tol
    l2_zero = abs(l_2) < tol
    if a1_zero and a2_zero:
        return "[a_1=0,a_2=0]"
    if a1_zero and l2_zero:
        return "[a_1=0,l_2=0]"
    if l1_zero and a2_zero:
        return "[l_1=0,a_2=0]"
    if l1_zero and l2_zero:
        return "[l_1=0,l_2=0]"
    raise ValueError(
        f"DH parameters do not match any T(v_2) RRR sub-case: "
        f"a_1={a_1}, l_1={l_1}, a_2={a_2}, l_2={l_2}. "
        f"T(v_2) only applies under (a_1=0 OR l_1=0) AND (a_2=0 OR l_2=0)."
    )


@lru_cache(maxsize=4)
def _build_tv2_rrr_case_chain_symbolic(
    case_key: str,
) -> tuple[sp.Matrix, tuple[sp.Symbol, ...]]:
    """Build the symbolic 8-vec Study DQ chain
    ``t = sigma_1 . sigma_2 . R_z(v_3)`` after applying the
    ``case_key`` DH-degeneracy substitution.

    Returns ``(t_chain, free_syms)`` where ``t_chain`` is a sympy
    8-vec with entries polynomial in (v_1, v_2, v_3) and the
    surviving DH parameters, and ``free_syms`` is the ordered tuple
    ``(v_1, v_2, v_3, *remaining_DH)`` where ``remaining_DH``
    depends on the sub-case (e.g. ``(l_1, l_2, d_2)`` for
    ``[a_1=0,a_2=0]``).

    The numeric runtime helper :func:`tv2_symbolic_in_v2` substitutes
    DH numerically in ``t_chain`` then computes the 4-hyperplane
    kernel via numpy SVD (avoiding sympy nullspace's symbolic-pole
    issues at l_1 = ±1, l_2 = ±1 etc).
    """
    v_1 = _V1_SYM
    v_2 = _V2_SYM
    v_3 = _V3_SYM
    a_1, l_1, d_2 = sp.symbols("a_1 l_1 d_2", real=True)
    a_2, l_2 = sp.symbols("a_2 l_2", real=True)

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

    sigma_1_chain = _dq_mul_sym(rz(v_1), _dq_mul_sym(tx(a_1), rx(l_1)))
    sigma_2_chain = _dq_mul_sym(
        rz(v_2), _dq_mul_sym(tz(d_2), _dq_mul_sym(tx(a_2), rx(l_2)))
    )
    rz_v3 = rz(v_3)

    t_chain = _dq_mul_sym(sigma_1_chain, _dq_mul_sym(sigma_2_chain, rz_v3))

    sub_dict, free_dh = _tv2_rrr_case_substitution(case_key, a_1, l_1, a_2, l_2)
    t_chain = t_chain.subs(sub_dict)

    return t_chain, (v_1, v_2, v_3) + free_dh + (d_2,)


def tv2_symbolic_in_v2(
    case_key: str,
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
) -> sp.Matrix:
    """Return the 4x8 ``T(v_2)`` matrix for one RRR sub-case as a
    sympy ``Matrix`` with ``v_2`` symbolic and DH parameters
    substituted numerically.

    Used by the elimination pipeline (Phase 5c.4 / #176) when both
    ``T(v_1)`` and ``T(v_3)`` are degenerate. The caller is responsible
    for picking the right ``case_key`` via :func:`tv2_rrr_case_for`.

    The runtime free symbol is :data:`_V2_SYM`. Joint-3 DH parameters
    (``a_3``, ``l_3``, ``d_3``) are NOT inputs here -- they get absorbed
    by the Tv2_full change of variables in ``_eliminate.py`` (analogous
    to ``tv1_symbolic_in_v1`` / ``tv3_symbolic_in_v3`` for their
    respective parametrizations).

    Implementation: build the 12x16 monomial-coefficient matrix from
    the symbolic chain (cached per case_key) with DH substituted
    numerically, take its kernel via numpy SVD, and reconstruct the 4
    v_2-linear hyperplane equations from the 4-D kernel basis.
    Numeric kernel avoids sympy nullspace's symbolic-pole artefacts
    at l_1 = ±1, l_2 = ±1 etc.

    :param case_key: one of :data:`TV2_RRR_CASE_KEYS`.
    :param a_1, l_1: joint-1 DH (substituted to 0 if the case demands).
    :param d_2: joint-2 d offset (always free).
    :param a_2, l_2: joint-2 DH (substituted to 0 if the case demands).
    """
    t_chain_sym, (v_1_sym, v_2_sym, v_3_sym, *dh_syms_after_case) = (
        _build_tv2_rrr_case_chain_symbolic(case_key)
    )
    # The symbol table is set up by _build_tv2_rrr_case_chain_symbolic;
    # use sp.symbols(...) lookups to populate the substitution map.
    a1_s, l1_s = sp.symbols("a_1 l_1", real=True)
    a2_s, l2_s = sp.symbols("a_2 l_2", real=True)
    d2_s = sp.symbols("d_2", real=True)
    dh_full = {
        a1_s: sp.Float(a_1),
        l1_s: sp.Float(l_1),
        d2_s: sp.Float(d_2),
        a2_s: sp.Float(a_2),
        l2_s: sp.Float(l_2),
    }
    # Substitute DH numerically; t_chain_num is now an 8-vec of polynomials in
    # (v_1, v_2, v_3) with float coefficients.
    t_chain_num = t_chain_sym.subs(dh_full)

    # Build the 12x16 matrix from monomial coefficients.
    M = np.zeros((12, 16), dtype=np.float64)
    for v1i in range(2):
        for v3i in range(2):
            for v2i in range(3):
                row = v2i + 3 * v3i + 6 * v1i
                for j in range(8):
                    poly = sp.Poly(
                        sp.expand(t_chain_num[j]), v_1_sym, v_2_sym, v_3_sym
                    )
                    coef = poly.coeff_monomial((v1i, v2i, v3i))
                    coef_f = float(coef) if coef is not sp.S.Zero else 0.0
                    # cf basis (constant in v_2): row contributes coef * cf[j]
                    # but we also need (cf+v_2*df)[i] structure -- the row
                    # corresponds to (v_2)^v2i, so the cf[j] enters at v2i=0
                    # and df[j] at v2i=1 (i.e. the v_2-multiplied half).
                    # For our flattened indexing matching Capco's:
                    #   col j in [0..7]   = coeff of cf[j] at this monomial
                    #   col j in [8..15]  = coeff of df[j-8]
                    # The expression eqn = sum_i t_chain[i]*(cf[i] + v_2*df[i])
                    # contributes to monomial (v1i, v2i, v3i) two ways:
                    #   - cf[i] term: t_chain[i]'s (v1i, v2i, v3i) coef -> col i
                    #   - df[i] term: t_chain[i]'s (v1i, v2i-1, v3i) coef * v_2
                    #                 -> col i+8 (only if v2i >= 1)
                    M[row, j] = coef_f
                    if v2i >= 1:
                        # df[j] contribution: comes from t_chain[j]'s
                        # (v1i, v2i-1, v3i) monomial coefficient.
                        poly_lower = sp.Poly(
                            sp.expand(t_chain_num[j]), v_1_sym, v_2_sym, v_3_sym
                        )
                        coef_lower = poly_lower.coeff_monomial(
                            (v1i, v2i - 1, v3i)
                        )
                        coef_lower_f = (
                            float(coef_lower) if coef_lower is not sp.S.Zero else 0.0
                        )
                        M[row, 8 + j] = coef_lower_f

    # Kernel via SVD: pick the 4 right-singular vectors with smallest
    # singular values. For a Capco-valid sub-case these are the
    # 4 hyperplane basis vectors (kernel dim = 4 generically).
    _, sigmas, vh = np.linalg.svd(M, full_matrices=True)
    # vh has shape (16, 16); rows 12..15 are the kernel.
    if sigmas.shape[0] >= 4 and sigmas[-4] > 1e-6:
        # If the 4th-from-last singular value is non-tiny, the kernel
        # isn't 4-dim and our case substitution may be wrong for this DH.
        # Fall back to picking the smallest 4 anyway (best effort).
        pass
    kernel_basis = vh[-4:, :]  # shape (4, 16)

    # Each kernel basis vector v_k of length 16 gives a hyperplane
    #   H_k(xy, v_2) = (v_k[0:8] + v_2 * v_k[8:16]) . xy
    # Build the 4x8 sympy matrix where H[i, j] = v_k[j] + v_2 * v_k[8+j].
    v_2_out = _V2_SYM
    H = sp.zeros(4, 8)
    for i in range(4):
        for j in range(8):
            H[i, j] = sp.Float(float(kernel_basis[i, j])) + v_2_out * sp.Float(
                float(kernel_basis[i, 8 + j])
            )
    return H


def tv2_hyperplanes_rrr(
    case_key: str,
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
    d_3: float,
    a_3: float,
    l_3: float,
    v_2: float,
) -> NDArray[np.float64]:
    """Full 4x8 ``T(v_2)`` coefficient matrix for the RRR case at one ``v_2``.

    Mirrors Capco's ``rrr.py:Tv2_full(left=True)``: the simple-form
    Tv2 hyperplanes (output of :func:`tv2_symbolic_in_v2`) are
    expressed in Study coords ``xy`` at frame ``F_3`` (right after
    joint 3 has rotated by ``v_3``). The full form transforms them to
    coords ``uw`` at frame ``F_4`` (after joint 3's full DH transition
    ``T_z(d_3) T_x(a_3) R_x(l_3)``) by the substitution

        xy = uw . conj(T_z(d_3) T_x(a_3) R_x(l_3))

    Equivalently, the 4x8 hyperplane matrix transforms as
    ``H_full = H_simple @ R_right`` where ``R_right`` is the 8x8
    matrix representation of right-multiplication by
    ``conj(T_z(d_3) T_x(a_3) R_x(l_3))`` (see
    :func:`_dq_right_mult_matrix`).

    For any ``v_1, v_2, v_3 in R``, the 4 hyperplanes returned here
    vanish on the projective Study DQ of the full RRR chain
    ``sigma_1(v_1) sigma_2(v_2) sigma_3(v_3)`` -- the same V_L that
    ``T(v_1)`` and ``T(v_3)`` describe, but parametrised by ``v_2``
    when ``T(v_1)`` and ``T(v_3)`` are both degenerate.

    Argument order matches :func:`tv1_hyperplanes_rrr` for
    consistency: parameters of joints 1, 2, 3 in sequence; ``v_2``
    last (it's the parametrising free variable).
    """
    h_sym = tv2_symbolic_in_v2(case_key, a_1, l_1, d_2, a_2, l_2)
    # Substitute v_2 numerically -> 4x8 sympy float matrix.
    h_at_v2 = h_sym.subs(_V2_SYM, sp.Float(v_2))
    h_simple = np.array(h_at_v2.tolist(), dtype=np.float64)
    if h_simple.shape != (4, 8):  # pragma: no cover -- defensive
        raise RuntimeError(f"tv2 simple form: expected (4, 8), got {h_simple.shape}")

    # Joint-3 DH transition: T_z(d_3) T_x(a_3) R_x(l_3) (no v_3 rotation;
    # v_3 is encoded in xy at F_3 by the simple form already).
    j3_dh = np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * d_3], dtype=np.float64
    )  # T_z(d_3)
    tx_dq = np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.5 * a_3, 0.0, 0.0], dtype=np.float64
    )
    rx_dq = np.array([1.0, l_3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # Compose: J3_DH = T_z(d_3) T_x(a_3) R_x(l_3) via dq_mul.
    from ssik.solvers.husty_pfurner._study import dq_conj, dq_mul as _dq_mul

    j3_dh = _dq_mul(j3_dh, _dq_mul(tx_dq, rx_dq))
    # conj(J3_DH) for the right-mult matrix.
    j3_dh_conj = dq_conj(j3_dh)
    R_right = _dq_right_mult_matrix(j3_dh_conj)

    # H_full = H_simple @ R_right
    h_full = h_simple @ R_right
    return h_full

