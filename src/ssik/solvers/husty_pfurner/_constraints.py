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

import numpy as np
from numpy.typing import NDArray

__all__ = ["hyperplane_residuals", "v1_hyperplanes_rrr"]


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
