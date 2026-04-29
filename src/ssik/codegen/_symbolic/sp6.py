"""Symbolic SP6: two coupled SP4-like equations in two unknowns.

SP6 has substantial numeric machinery (QR decomposition, conic-intersection
quartic, Gauss-Newton refinement) that cannot be reduced to closed-form
trig. The codegen specialises the tractable parts:

* The 2x4 matrix ``A`` (entries are linear forms in inputs ``h, k, p``)
  and the 2-vector ``b`` are CSE'd and inlined per arm.
* The QR decomposition of ``A.T`` stays runtime (``np.linalg.qr``).
* The 2-ellipse intersection (the SP6 quartic) stays runtime
  (``ssik.subproblems._aux.solve_two_ellipse_numeric``).
* The Gauss-Newton refinement stays runtime
  (``ssik.subproblems.sp6._refine_sp6``).

For tier-0 closed-form solvers (three_parallel) the specialisation
mostly speeds up the SP6 *setup* (the ``a_mat``/``b`` build + the
post-SP6 SP1/SP3 chain). The QR+ellipse call stays opaque until
Phase 4 Cython folds them into native code.

This module provides :func:`sp6_a_mat_b_sym`: takes 4-tuples of
sympy 3x1 Matrix inputs (``h``, ``k``, ``p``) plus sympy expressions
``d1``, ``d2``, returns the ``(A_mat, b)`` symbolic forms ready for CSE.
"""

from __future__ import annotations

import sympy as sp

__all__ = ["sp6_a_mat_b_sym"]


def sp6_a_mat_b_sym(
    h: tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix],
    k: tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix],
    p: tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix],
    d1: sp.Expr,
    d2: sp.Expr,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Return the symbolic SP6 ``(A, b)`` so the codegen inlines them.

    Mirrors the SP6 setup in :mod:`ssik.subproblems.sp6`:

        a_cols[i] = column_stack([k[i] x p[i], -k[i] x (k[i] x p[i])])
                    -- a 3x2 matrix per i in 0..3
        h_a[i] = h[i] . a_cols[i]   -- 1x2 vector
        A = [[h_a[0][0], h_a[0][1], h_a[1][0], h_a[1][1]],
             [h_a[2][0], h_a[2][1], h_a[3][0], h_a[3][1]]]   -- 2x4
        b = [d1 - (h[0].k[0])(k[0].p[0]) - (h[1].k[1])(k[1].p[1]),
             d2 - (h[2].k[2])(k[2].p[2]) - (h[3].k[3])(k[3].p[3])]

    :param h, k, p: 4-tuples of sympy 3x1 Matrix inputs.
    :param d1, d2: sympy expressions for the two scalar targets.
    :returns: ``(A, b)`` where ``A`` is a sympy 2x4 Matrix and ``b`` is
        a sympy 2x1 Matrix. Caller substitutes arm constants + T_target,
        runs sympy.cse, emits Python.
    """
    # Build per-i 3x2 a_cols.
    h_a_rows: list[sp.Matrix] = []
    for i in range(4):
        kxp_i = k[i].cross(p[i])
        kx_kxp_i = k[i].cross(kxp_i)
        # 3x2 matrix; column 0 = kxp_i, column 1 = -kx_kxp_i.
        a_col_i = sp.Matrix.hstack(kxp_i, -kx_kxp_i)
        # h[i] . a_col_i -> 1x2 row.
        h_a_i = h[i].T * a_col_i  # shape: 1x2
        h_a_rows.append(h_a_i)

    a_mat = sp.Matrix(
        [
            [h_a_rows[0][0, 0], h_a_rows[0][0, 1], h_a_rows[1][0, 0], h_a_rows[1][0, 1]],
            [h_a_rows[2][0, 0], h_a_rows[2][0, 1], h_a_rows[3][0, 0], h_a_rows[3][0, 1]],
        ]
    )

    b_vec = sp.Matrix(
        [
            [d1 - h[0].dot(k[0]) * k[0].dot(p[0]) - h[1].dot(k[1]) * k[1].dot(p[1])],
            [d2 - h[2].dot(k[2]) * k[2].dot(p[2]) - h[3].dot(k[3]) * k[3].dot(p[3])],
        ]
    )

    return a_mat, b_vec
