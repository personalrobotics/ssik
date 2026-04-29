"""Composer for ``ikgeo.general_6r`` (tier-2 Raghavan-Roth, EAIK-gap path).

Specialisation strategy (#118):

  1. **Build-time** (codegen, runs once per arm):
     - POE -> DH conversion via :func:`ssik.kinematics.poe_to_dh`.
     - AE-3 leftvar selection (cached).
     - Run ``_derive_pq_for_arm`` to get the four sympy Matrices
       (P_sin, P_cos, P_one, Q) -- entries are polynomials in the 12
       T_target symbols T_0..T_11, with arm DH constants substituted.
     - CSE over all 490 matrix entries (4 matrices x 14 rows x ~9 cols).
     - Render each entry as Python source via ``sympy.pycode``.

  2. **Artifact** (per IK call):
     - Destructure ``T_target`` into 12 floats.
     - One inlined builder ``_build_pq_matrices(T_:12)`` computes all 4
       matrices in a single CSE-shared local scope. This is where JACO 2's
       0.866 (cos 60-deg) and similar arm constants multiply T_target
       entries -- explicit ``sin`` / ``cos`` of the arm become visible.
     - Generic linear algebra (Q-rank elimination, Weierstrass, M(x)
       pencil build, 24x24 eigvals + Mobius fallback, back-substitution,
       Newton refinement) stays imported from
       :mod:`ssik.solvers.ikgeo._raghavan_roth`. Those are arm-agnostic;
       Phase 4 Cython compiles them via numpy linalg bindings.
     - Map DH-frame q -> POE q via baked ``theta_offset``.

Output: ``tests/artifacts/jaco2_ik.py`` (and any other non-Pieper 6R
artifact) contains explicit per-arm trig in the matrix builders --
matching the IKFast paradigm we already deliver for tier-0 (Puma, UR5).

This is the substantive Cython prep for tier-2: with the matrix builders
inlined, Phase 4 has visible math to compile. The numpy linalg core
links to LAPACK directly via Cython.
"""

from __future__ import annotations

import textwrap
from typing import cast

import sympy as sp

from ssik._kinbody import KinBody
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import _cached_best_leftvar, _cached_derivation

__all__ = ["compose", "render_constants_header"]


def render_constants_header() -> str:
    """Imports needed by the rendered general_6r artifact."""
    return (
        "import math\n"
        "from ssik.solvers.ikgeo._raghavan_roth import (\n"
        "    eliminate_q0_q1 as _ssik_eliminate_q0_q1,\n"
        "    weierstrass_eliminate_trig as _ssik_weierstrass,\n"
        "    build_m_matrix as _ssik_build_m_matrix,\n"
        "    solve_x2_roots_mobius as _ssik_solve_x2_roots_mobius,\n"
        "    _back_substitute_inner as _ssik_back_substitute_inner,\n"
        "    _fk_dh as _ssik_fk_dh,\n"
        ")\n"
    )


def compose(kb: KinBody) -> str:
    """Render ``_solve_algebraic`` for a tier-2 RR arm with inlined matrices.

    :param kb: a POE-normalised :class:`KinBody` with 6 revolute joints.
    :returns: Python source for ``_solve_algebraic(T_target)`` with:

        - One inlined ``_build_pq_matrices`` function (CSE'd; explicit
          sin/cos of arm constants in cell expressions).
        - Baked DH params + ``theta_offset`` + ``t_pre`` / ``t_post``
          bridges as np.array literals.
        - ``_solve_algebraic`` orchestrating the full RR pipeline.
    """
    if len(kb.joints) != 6:
        raise ValueError(f"general_6r composer requires 6-DOF chain; got {len(kb.joints)}")
    for joint in kb.joints:
        if joint.joint_type != "revolute":
            raise ValueError(f"general_6r requires all-revolute joints; got {joint.joint_type}")

    dh = poe_to_dh(kb)
    alpha = tuple(dh.alpha.tolist())
    a = tuple(dh.a.tolist())
    d = tuple(dh.d.tolist())

    # AE-3: pick the leftvar with best conditioning. Cached per arm.
    linearity = _cached_best_leftvar(alpha, a, d)

    # Get the symbolic matrices from the (cached) per-arm derivation.
    _, _, _, _, meta = _cached_derivation(alpha, a, d, linearity_joint=linearity, apply_so3=False)
    sym_p_sin = cast(sp.Matrix, meta["_sym_p_sin"])
    sym_p_cos = cast(sp.Matrix, meta["_sym_p_cos"])
    sym_p_one = cast(sp.Matrix, meta["_sym_p_one"])
    sym_q = cast(sp.Matrix, meta["_sym_q"])
    t_syms = cast(tuple[sp.Symbol, ...], meta["_sym_t_target"])
    left_bilinear = cast(tuple[int, int], meta["left_bilinear"])
    right_bilinear = cast(tuple[int, int], meta["right_bilinear"])
    drop_joint = cast(int, meta["drop_joint"])

    # Render one combined builder function sharing one CSE pass.
    builder_sources = _render_pq_combined_builder(
        sym_p_sin=sym_p_sin,
        sym_p_cos=sym_p_cos,
        sym_p_one=sym_p_one,
        sym_q=sym_q,
        t_syms=t_syms,
    )

    # Bake DH + bridge constants.
    body = textwrap.dedent(
        f"""\
        # --- baked DH parameters (from poe_to_dh at build time) ---
        _DH_ALPHA = np.array({list(dh.alpha.tolist())!r}, dtype=np.float64)
        _DH_A = np.array({list(dh.a.tolist())!r}, dtype=np.float64)
        _DH_D = np.array({list(dh.d.tolist())!r}, dtype=np.float64)
        _DH_TUPLE = (_DH_ALPHA, _DH_A, _DH_D)
        _DH_THETA_OFFSET = np.array({list(dh.theta_offset.tolist())!r}, dtype=np.float64)
        _T_PRE = np.array({dh.t_pre.tolist()!r}, dtype=np.float64)
        _T_POST = np.array({dh.t_post.tolist()!r}, dtype=np.float64)
        _T_PRE_INV = np.linalg.inv(_T_PRE)
        _T_POST_INV = np.linalg.inv(_T_POST)

        _LINEARITY_JOINT = {linearity}
        _LEFT_BILINEAR = {left_bilinear!r}
        _RIGHT_BILINEAR = {right_bilinear!r}
        _DROP_JOINT = {drop_joint}
        _RR_META = {{
            "linearity_joint": _LINEARITY_JOINT,
            "left_bilinear": _LEFT_BILINEAR,
            "right_bilinear": _RIGHT_BILINEAR,
            "drop_joint": _DROP_JOINT,
            "apply_so3": False,
        }}


        """
    )
    body += builder_sources
    body += textwrap.dedent(
        """\


        def _solve_algebraic(T_target):
            \"\"\"Tier-2 Raghavan-Roth IK candidates with INLINED P/Q matrices.

            Per-arm DH constants substituted into the 4 RR matrix builders
            (CSE'd above); generic linear algebra (eliminate_q0_q1,
            Weierstrass, eigvals, back-substitution) stays imported.
            \"\"\"
            T = np.asarray(T_target, dtype=np.float64)
            T_dh = _T_PRE_INV @ T @ _T_POST_INV

            # Destructure T_dh's free entries (top 3 rows, 4 cols).
            T_0, T_1, T_2, T_3 = T_dh[0, 0], T_dh[0, 1], T_dh[0, 2], T_dh[0, 3]
            T_4, T_5, T_6, T_7 = T_dh[1, 0], T_dh[1, 1], T_dh[1, 2], T_dh[1, 3]
            T_8, T_9, T_10, T_11 = T_dh[2, 0], T_dh[2, 1], T_dh[2, 2], T_dh[2, 3]
            p_sin, p_cos, p_one, q_mat = _build_pq_matrices(
                T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, T_11
            )

            # Generic linear algebra: arm-agnostic, runtime-imported. Phase 4
            # Cython links these to LAPACK directly.
            e_sin, e_cos, e_one = _ssik_eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
            e_quad, e_lin, e_const = _ssik_weierstrass(e_sin, e_cos, e_one)
            m_quad, m_lin, m_const = _ssik_build_m_matrix(e_quad, e_lin, e_const)
            roots, eigvecs = _ssik_solve_x2_roots_mobius(m_quad, m_lin, m_const)

            q_pinv = np.linalg.pinv(q_mat).astype(np.float64)

            # Back-substitute per real root. ``_back_substitute_inner``
            # returns ``(q_dh, fk_err_alg)`` -- we drop fk_err here because
            # the outer ``solve()`` re-runs FK on each candidate (mirroring
            # the wrapper version's behavior, kept for parity).
            inner_qs = []
            for x_lin, eigvec in zip(roots, eigvecs):
                bs_result = _ssik_back_substitute_inner(
                    x_lin, eigvec, p_sin, p_cos, p_one, q_pinv,
                    _DH_TUPLE, T_dh, _RR_META,
                )
                if bs_result is None:
                    continue
                q_dh, _fk_err = bs_result
                inner_qs.append(q_dh)

            # Map DH-frame q back to POE frame.
            return [
                list(np.asarray(q_dh, dtype=np.float64) - _DH_THETA_OFFSET)
                for q_dh in inner_qs
            ]
        """
    )
    return body


def _render_pq_combined_builder(
    *,
    sym_p_sin: sp.Matrix,
    sym_p_cos: sp.Matrix,
    sym_p_one: sp.Matrix,
    sym_q: sp.Matrix,
    t_syms: tuple[sp.Symbol, ...],
) -> str:
    """Render one combined builder ``_build_pq_matrices`` for all 4 matrices.

    Single function so the CSE temporaries stay in one local scope -- no
    globals contamination, no dict-lookup overhead. CSE runs over all 490
    entries together so common subexpressions are shared across the four
    matrices.
    """
    n_rows_p, n_cols_p = sym_p_sin.shape
    n_rows_q, n_cols_q = sym_q.shape

    # Flatten in (matrix, row, col) order so we can recover the structure.
    flat_entries: list[sp.Expr] = []
    for r in range(n_rows_p):
        for cc in range(n_cols_p):
            flat_entries.append(sym_p_sin[r, cc])
    for r in range(n_rows_p):
        for cc in range(n_cols_p):
            flat_entries.append(sym_p_cos[r, cc])
    for r in range(n_rows_p):
        for cc in range(n_cols_p):
            flat_entries.append(sym_p_one[r, cc])
    for r in range(n_rows_q):
        for cc in range(n_cols_q):
            flat_entries.append(sym_q[r, cc])

    cse_subs, finals = sp.cse(flat_entries, symbols=sp.numbered_symbols(prefix="_pq_x"))

    n_entries_p = n_rows_p * n_cols_p
    p_sin_finals = finals[0:n_entries_p]
    p_cos_finals = finals[n_entries_p : 2 * n_entries_p]
    p_one_finals = finals[2 * n_entries_p : 3 * n_entries_p]
    q_finals = finals[3 * n_entries_p :]

    arg_list = ", ".join(str(s) for s in t_syms)
    parts: list[str] = []
    parts.append("# --- inlined per-arm matrix builders (sympy.cse'd) ---")
    parts.append("# Per-arm DH constants combine with T_target here. JACO 2's")
    parts.append("# 0.866 (cos 60-deg) and friends become explicit coefficients.")
    parts.append("# Generic linear algebra (Q-rank, Weierstrass, eigvals,")
    parts.append("# back-substitution) stays imported from ssik.solvers.ikgeo.")
    parts.append("")
    parts.append(f"def _build_pq_matrices({arg_list}):")
    parts.append('    """Build (P_sin, P_cos, P_one, Q) for this T_target. CSE-shared."""')
    if cse_subs:
        for sym, sub in cse_subs:
            parts.append(f"    {sym} = {sp.pycode(sub)}")

    parts.append("")
    parts.append("    p_sin = np.array([")
    parts.extend(_format_matrix_rows(p_sin_finals, n_rows_p, n_cols_p))
    parts.append("    ], dtype=np.float64)")

    parts.append("")
    parts.append("    p_cos = np.array([")
    parts.extend(_format_matrix_rows(p_cos_finals, n_rows_p, n_cols_p))
    parts.append("    ], dtype=np.float64)")

    parts.append("")
    parts.append("    p_one = np.array([")
    parts.extend(_format_matrix_rows(p_one_finals, n_rows_p, n_cols_p))
    parts.append("    ], dtype=np.float64)")

    parts.append("")
    parts.append("    q = np.array([")
    parts.extend(_format_matrix_rows(q_finals, n_rows_q, n_cols_q))
    parts.append("    ], dtype=np.float64)")

    parts.append("")
    parts.append("    return p_sin, p_cos, p_one, q")
    parts.append("")
    return "\n".join(parts) + "\n"


def _format_matrix_rows(final_exprs: list[sp.Expr], n_rows: int, n_cols: int) -> list[str]:
    """Format a flat list of CSE'd finals as nested-list rows for np.array."""
    rows: list[str] = []
    idx = 0
    for _r in range(n_rows):
        cells: list[str] = []
        for _cc in range(n_cols):
            cells.append(sp.pycode(final_exprs[idx]))
            idx += 1
        rows.append("        [" + ", ".join(cells) + "],")
    return rows
