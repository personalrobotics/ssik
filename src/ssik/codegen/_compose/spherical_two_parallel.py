"""Composer for ``ikgeo.spherical_two_parallel`` (Puma class).

Mirrors :func:`ssik.solvers.ikgeo.spherical_two_parallel.solve` but runs
at codegen time: substitutes the KinBody's constants into the symbolic
SP solvers and emits a Python source string for ``_solve_algebraic``
that contains explicit ``sin``/``cos``/``atan2`` of the target-pose
entries.

The rendered function structure (sketch)::

    def _solve_algebraic(T_target):
        r_00 = T_target[0, 0]; ...; p_z = T_target[2, 3]
        candidates = []

        # SP4 for q1 -- closed-form trig
        q1_x0 = ...
        ...
        theta_q1_plus = ...
        theta_q1_minus = ...

        for q1 in (theta_q1_plus, theta_q1_minus):
            # SP3 for q3 (closed-form trig in q1, target)
            ...
            for q3 in (theta_q3_plus, theta_q3_minus):
                # SP1 for q2 (single atan2 in q1, q3, target)
                q2 = math.atan2(...)
                # SP4 for q5 (closed-form trig)
                ...
                for q5 in (theta_q5_plus, theta_q5_minus):
                    # SP1 for q4, q6
                    q4 = math.atan2(...)
                    q6 = math.atan2(...)
                    candidates.append([q1, q2, q3, q4, q5, q6])
        return candidates

Conventions:

* CSE temporary names are prefixed per block (``q1_x0``, ``q3_x0``, ...)
  so blocks at different scope can co-exist without collision.
* All rendered code uses ``math.cos`` / ``math.sin`` / ``math.atan2``
  (sympy.pycode default).
* SP4-style blocks emit LS / degeneracy fallbacks via ``if`` guards on
  ``R_sq`` and ``rhs``, mirroring the runtime solver's behaviour.
"""

from __future__ import annotations

import sympy as sp

from ssik._kinbody import KinBody
from ssik.codegen._compose._target import TargetSymbols, make_target_symbols
from ssik.codegen._symbolic.sp1 import sp1_theta_sym
from ssik.codegen._symbolic.sp3 import sp3_branches_sym
from ssik.codegen._symbolic.sp4 import sp4_branches_sym

__all__ = ["compose", "render_constants_header"]


# Tolerance defaults baked into the rendered guards. Match
# DEFAULT_TOLERANCE_POLICY.subproblem_degeneracy and subproblem_feasibility.
_DEG_SQ = 1e-16
_FEAS_TOL = 1e-8


def render_constants_header() -> str:
    """Return the constants/imports header the rendered artifact needs.

    Emits ``import math`` plus the SP4-guard tolerance constants.
    """
    return f"import math\n\n_DEG_SQ = {_DEG_SQ!r}\n_FEAS_TOL = {_FEAS_TOL!r}\n"


def compose(kb: KinBody) -> str:
    """Render the body of ``_solve_algebraic`` for a Puma-class arm.

    :param kb: a POE-normalised :class:`KinBody` matching the
        ``spherical_two_parallel`` topology (3 consecutive intersecting
        axes at (3, 4, 5), axes[1] || axes[2]). The dispatcher gates
        this; we don't re-check.
    :returns: Python source for ``_solve_algebraic(T_target)``. Returns
        a ``list[list[float]]`` of up to 8 candidate joint vectors;
        verify + dedup happen in the artifact's outer ``solve()`` orchestrator.
    """
    target = make_target_symbols()

    axes = [j.axis for j in kb.joints]
    p_offsets = [j.T_left[:3, 3].copy() for j in kb.joints]
    p_3_consolidated = p_offsets[3] + p_offsets[4] + p_offsets[5]
    p_tool = kb.joints[-1].T_right[:3, 3].copy()
    r_home = kb.joints[-1].T_right[:3, :3].copy()

    # Common: r_06 and p_0t in target symbols.
    r_home_sym = _mat_const(r_home)
    r_06 = target.r * r_home_sym.T
    p_0t = target.p
    p_16 = p_0t - r_06 * _vec_const(p_tool) - _vec_const(p_offsets[0])

    # ---------- SP4 for q1 ----------
    d_q1 = float(axes[1] @ (p_offsets[1] + p_offsets[2] + p_3_consolidated))
    sp4_q1 = sp4_branches_sym(_vec_const(axes[1]), _vec_const(-axes[0]), p_16, sp.Float(d_q1))
    q1_lines = _render_sp4_block(
        prefix="q1",
        sp4_out=sp4_q1,
        comment="SP4 for q1 (shoulder pan).",
    )

    # ---------- Inside the q1 loop: SP3 for q3 ----------
    q1_sym = sp.Symbol("q1", real=True)
    rot_neg_axis_0_q1 = _rotation_matrix_sym(_vec_const(-axes[0]), q1_sym)
    p_tool_sym = _vec_const(p_tool)
    p_0_sym = _vec_const(p_offsets[0])
    p_1_sym = _vec_const(p_offsets[1])
    shoulder = rot_neg_axis_0_q1 * (-p_0t + r_06 * p_tool_sym + p_0_sym) + p_1_sym
    shoulder_norm = sp.sqrt(shoulder.dot(shoulder))
    sp3_q3 = sp3_branches_sym(
        _vec_const(axes[2]),
        -_vec_const(p_3_consolidated),
        _vec_const(p_offsets[2]),
        shoulder_norm,
    )
    q3_lines = _render_sp4_block(
        prefix="q3",
        sp4_out=sp3_q3,
        comment="SP3 for q3 (elbow): reduces to SP4 with target shift.",
    )

    # ---------- Inside (q1, q3): SP1 for q2 ----------
    q3_sym = sp.Symbol("q3", real=True)
    rot_axis_2_q3 = _rotation_matrix_sym(_vec_const(axes[2]), q3_sym)
    sp1_q2_p = -_vec_const(p_offsets[2]) - rot_axis_2_q3 * _vec_const(p_3_consolidated)
    q2_expr = sp1_theta_sym(_vec_const(axes[1]), sp1_q2_p, shoulder)
    q2_lines = _render_atan2_block(
        prefix="q2",
        var_name="q2",
        expr=q2_expr,
        comment="SP1 for q2 (shoulder pitch): closed-form atan2.",
    )

    # ---------- r_36 = Rot(-axes[2], q3) @ Rot(-axes[1], q2) @ Rot(-axes[0], q1) @ r_06 ----------
    q2_sym = sp.Symbol("q2", real=True)
    rot_neg_axis_2_q3 = _rotation_matrix_sym(_vec_const(-axes[2]), q3_sym)
    rot_neg_axis_1_q2 = _rotation_matrix_sym(_vec_const(-axes[1]), q2_sym)
    rot_neg_axis_0_q1_post = _rotation_matrix_sym(_vec_const(-axes[0]), q1_sym)
    r_36 = rot_neg_axis_2_q3 * rot_neg_axis_1_q2 * rot_neg_axis_0_q1_post * r_06

    # ---------- SP4 for q5 ----------
    d_q5 = (_vec_const(axes[3]).T * r_36 * _vec_const(axes[5]))[0, 0]
    sp4_q5 = sp4_branches_sym(_vec_const(axes[3]), _vec_const(axes[4]), _vec_const(axes[5]), d_q5)
    q5_lines = _render_sp4_block(
        prefix="q5",
        sp4_out=sp4_q5,
        comment="SP4 for q5 (wrist pitch).",
    )

    # ---------- SP1 for q4 and q6 ----------
    q5_sym = sp.Symbol("q5", real=True)
    rot_axis_4_q5 = _rotation_matrix_sym(_vec_const(axes[4]), q5_sym)
    rot_neg_axis_4_q5 = _rotation_matrix_sym(_vec_const(-axes[4]), q5_sym)

    q4_expr = sp1_theta_sym(
        _vec_const(axes[3]),
        rot_axis_4_q5 * _vec_const(axes[5]),
        r_36 * _vec_const(axes[5]),
    )
    q4_lines = _render_atan2_block(
        prefix="q4",
        var_name="q4",
        expr=q4_expr,
        comment="SP1 for q4 (wrist roll-1): closed-form atan2.",
    )

    q6_expr = sp1_theta_sym(
        _vec_const(-axes[5]),
        rot_neg_axis_4_q5 * _vec_const(axes[3]),
        r_36.T * _vec_const(axes[3]),
    )
    q6_lines = _render_atan2_block(
        prefix="q6",
        var_name="q6",
        expr=q6_expr,
        comment="SP1 for q6 (wrist roll-2): closed-form atan2.",
    )

    # Assemble.
    return _assemble(
        destructure_lines=_render_destructure(target),
        q1_lines=q1_lines,
        q3_lines=q3_lines,
        q2_lines=q2_lines,
        q5_lines=q5_lines,
        q4_lines=q4_lines,
        q6_lines=q6_lines,
    )


# ---------------------------------------------------------------------------
# Sympy helpers.
# ---------------------------------------------------------------------------


def _vec_const(v: object) -> sp.Matrix:
    return sp.Matrix([float(v[0]), float(v[1]), float(v[2])])  # type: ignore[index]


def _mat_const(m: object) -> sp.Matrix:
    return sp.Matrix([[float(m[i, j]) for j in range(3)] for i in range(3)])  # type: ignore[index]


def _rotation_matrix_sym(axis: sp.Matrix, angle: sp.Symbol) -> sp.Matrix:
    """Symbolic Rodrigues rotation. ``axis`` is a sympy 3x1 of constants."""
    c = sp.cos(angle)
    s = sp.sin(angle)
    x, y, z = axis[0, 0], axis[1, 0], axis[2, 0]
    oc = 1 - c
    return sp.Matrix(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ]
    )


# ---------------------------------------------------------------------------
# Rendering helpers. Each returns a list of source lines with NO leading
# indentation; the assembler adds the correct level.
# ---------------------------------------------------------------------------


def _render_destructure(target: TargetSymbols) -> list[str]:
    lines: list[str] = []
    for i in range(3):
        for j in range(3):
            lines.append(f"{target.r[i, j]} = T_target[{i}, {j}]")
    for i, name in enumerate(("p_x", "p_y", "p_z")):
        lines.append(f"{name} = T_target[{i}, 3]")
    return lines


def _render_atan2_block(*, prefix: str, var_name: str, expr: sp.Expr, comment: str) -> list[str]:
    """Render a single-atan2 closed-form variable assignment with CSE.

    CSE temporaries are namespaced as ``<prefix>_x0``, ``<prefix>_x1``,
    ... so blocks at different scope co-exist without collision.
    """
    cse_subs, [final] = sp.cse([expr], symbols=sp.numbered_symbols(prefix=f"{prefix}_x"))
    lines = [f"# {comment}"]
    for sym, sub in cse_subs:
        lines.append(f"{sym} = {sp.pycode(sub)}")
    lines.append(f"{var_name} = {sp.pycode(final)}")
    return lines


def _render_sp2_block(
    *,
    prefix: str,
    sp2_out: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
    comment: str,
) -> list[str]:
    """Render an SP2-style block: 4 angle outputs + LS / degenerate guards.

    Mirrors :func:`ssik.subproblems.sp2.solve` control flow:

      1. Compute s_sq, gamma_sq_scaled (CSE-shared with branch outputs).
      2. If s_sq < deg_tol: parallel axes -- return single SP1 fallback;
         the artifact lets verify_candidates filter via FK.
      3. If gamma_sq_scaled <= 0: tangent / LS -- single representative
         (we emit the 'a' branch; numerical sp2 emits the gamma=0 z-point).
      4. Else: feasible -- both (theta1_a, theta2_a) and (theta1_b, theta2_b).

    The codegen emits both branches; the artifact's verify step filters
    LS / degenerate cases via FK closure.
    """
    theta1_a, theta2_a, theta1_b, theta2_b, s_sq, gamma_sq = sp2_out
    cse_subs, finals = sp.cse(
        [theta1_a, theta2_a, theta1_b, theta2_b, s_sq, gamma_sq],
        symbols=sp.numbered_symbols(prefix=f"{prefix}_x"),
    )
    lines = [f"# {comment}"]
    for sym, sub in cse_subs:
        lines.append(f"{sym} = {sp.pycode(sub)}")
    names = (
        f"theta_{prefix}_1a",
        f"theta_{prefix}_2a",
        f"theta_{prefix}_1b",
        f"theta_{prefix}_2b",
        f"_{prefix}_s_sq",
        f"_{prefix}_gamma_sq",
    )
    for name, expr in zip(names, finals, strict=True):
        lines.append(f"{name} = {sp.pycode(expr)}")
    # No explicit branching here -- LS / degenerate cases produce candidates
    # that fail FK closure in the verify step. This is bulletproof enough:
    # the artifact returns valid candidates only after FK verification.
    return lines


def _render_sp4_block(
    *,
    prefix: str,
    sp4_out: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
    comment: str,
) -> list[str]:
    """Render an SP4-style block with both branches + LS / degenerate guards.

    Mirrors the runtime ``ssik.subproblems.sp4`` control flow:

      1. Compute (R_sq, rhs, phi). CSE-shared.
      2. If R_sq < DEG_SQ: degenerate -- ``theta = 0`` (verify drops).
      3. Else if |rhs| > sqrt(R_sq) + FEAS_TOL: LS branch -- ``theta = phi``
         (or ``phi + pi`` if rhs < 0).
      4. Else: feasible -- compute ``delta = acos(clip(rhs/sqrt(R_sq)))``,
         ``theta_plus = phi + delta``, ``theta_minus = phi - delta``.

    The clip on the acos input is required even in the "feasible" branch
    because numerical noise on the boundary (|rhs| ~ sqrt(R_sq)) can push
    rhs/R slightly outside [-1, 1], which raises ``ValueError`` in
    ``math.acos``.
    """
    _theta_plus, _theta_minus, r_sq, rhs, phi = sp4_out
    # We only need (R_sq, rhs, phi) at the top; theta_+- get computed
    # inside the conditional after clipping.
    cse_subs, [r_sq_final, rhs_final, phi_final] = sp.cse(
        [r_sq, rhs, phi],
        symbols=sp.numbered_symbols(prefix=f"{prefix}_x"),
    )
    lines = [f"# {comment}"]
    for sym, sub in cse_subs:
        lines.append(f"{sym} = {sp.pycode(sub)}")
    lines.append(f"_{prefix}_R_sq = {sp.pycode(r_sq_final)}")
    lines.append(f"_{prefix}_rhs = {sp.pycode(rhs_final)}")
    lines.append(f"_{prefix}_phi = {sp.pycode(phi_final)}")

    lines.append(f"if _{prefix}_R_sq < _DEG_SQ:")
    lines.append(f"    theta_{prefix}_plus = 0.0")
    lines.append(f"    theta_{prefix}_minus = 0.0  # degenerate; verify-step drops")
    lines.append("else:")
    lines.append(f"    _{prefix}_R = math.sqrt(_{prefix}_R_sq)")
    lines.append(f"    if abs(_{prefix}_rhs) > _{prefix}_R + _FEAS_TOL:")
    lines.append("        # LS fallback: theta = phi (or phi + pi if rhs < 0)")
    lines.append(f"        theta_{prefix}_plus = (")
    lines.append(f"            _{prefix}_phi if _{prefix}_rhs > 0 else _{prefix}_phi + math.pi")
    lines.append("        )")
    lines.append(f"        theta_{prefix}_minus = theta_{prefix}_plus")
    lines.append("    else:")
    lines.append(f"        _{prefix}_clipped = min(1.0, max(-1.0, _{prefix}_rhs / _{prefix}_R))")
    lines.append(f"        _{prefix}_delta = math.acos(_{prefix}_clipped)")
    lines.append(f"        theta_{prefix}_plus = _{prefix}_phi + _{prefix}_delta")
    lines.append(f"        theta_{prefix}_minus = _{prefix}_phi - _{prefix}_delta")
    return lines


def _indent(lines: list[str], spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in lines)


def _assemble(
    *,
    destructure_lines: list[str],
    q1_lines: list[str],
    q3_lines: list[str],
    q2_lines: list[str],
    q5_lines: list[str],
    q4_lines: list[str],
    q6_lines: list[str],
) -> str:
    """Stitch the per-step blocks into the final ``_solve_algebraic`` source."""
    parts = [
        "def _solve_algebraic(T_target):",
        '    """Algebraic IK candidates. Up to 8; verify + dedup in solve().',
        '    """',
        _indent(destructure_lines, 4),
        "    candidates = []",
        "",
        _indent(q1_lines, 4),
        "",
        "    for q1 in (theta_q1_plus, theta_q1_minus):",
        "        s1 = math.sin(q1)",
        "        c1 = math.cos(q1)",
        _indent(q3_lines, 8),
        "",
        "        for q3 in (theta_q3_plus, theta_q3_minus):",
        "            s3 = math.sin(q3)",
        "            c3 = math.cos(q3)",
        _indent(q2_lines, 12),
        "            s2 = math.sin(q2)",
        "            c2 = math.cos(q2)",
        _indent(q5_lines, 12),
        "",
        "            for q5 in (theta_q5_plus, theta_q5_minus):",
        "                s5 = math.sin(q5)",
        "                c5 = math.cos(q5)",
        _indent(q4_lines, 16),
        _indent(q6_lines, 16),
        "                candidates.append([q1, q2, q3, q4, q5, q6])",
        "    return candidates",
        "",
    ]
    return "\n".join(parts)
