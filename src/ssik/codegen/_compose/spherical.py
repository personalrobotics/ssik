"""Composer for ``ikgeo.spherical`` (generic spherical-wrist arm).

Mirrors :func:`ssik.solvers.ikgeo.spherical.solve`:

  1. Call SP5 (runtime) for (q1, q2, q3) shoulder triples. SP5 has
     internal quartic root-finding + Gauss-Newton refinement that don't
     reduce to closed-form trig; we keep them runtime via
     ``ssik.subproblems.sp5.solve``.
  2. For each (q1, q2, q3):
       - r_36 = Rot(-axes[2], q3) @ Rot(-axes[1], q2) @ Rot(-axes[0], q1) @ r_06
       - SP4 for q5 (closed-form, inlined).
       - For each q5: SP1 for q4, SP1 for q6 (inlined).

Same SP-runtime + post-SP1/SP4 closed-form pattern as ``three_parallel``.
"""

from __future__ import annotations

import sympy as sp

from ssik._kinbody import KinBody
from ssik.codegen._compose._target import make_target_symbols
from ssik.codegen._compose.spherical_two_parallel import (
    _mat_const,
    _render_atan2_block,
    _render_destructure,
    _render_sp4_block,
    _rotation_matrix_sym,
    _vec_const,
)
from ssik.codegen._symbolic.sp1 import sp1_theta_sym
from ssik.codegen._symbolic.sp4 import sp4_branches_sym

__all__ = ["compose", "render_constants_header"]


_DEG_SQ = 1e-16
_FEAS_TOL = 1e-8


def render_constants_header() -> str:
    return (
        "import math\n"
        "from ssik.subproblems import sp5 as _sp5_runtime\n"
        "\n"
        f"_DEG_SQ = {_DEG_SQ!r}\n"
        f"_FEAS_TOL = {_FEAS_TOL!r}\n"
    )


def compose(kb: KinBody) -> str:
    """Render ``_solve_algebraic`` for a generic spherical-wrist arm."""
    target = make_target_symbols()

    axes = [j.axis for j in kb.joints]
    p_offsets = [j.T_left[:3, 3].copy() for j in kb.joints]
    p_3_consolidated = p_offsets[3] + p_offsets[4] + p_offsets[5]
    p_tool = kb.joints[-1].T_right[:3, 3].copy()
    r_home = kb.joints[-1].T_right[:3, :3].copy()

    r_home_sym = _mat_const(r_home)
    r_06 = target.r * r_home_sym.T
    p_0t = target.p
    # p_16 components inlined for the SP5 runtime call.
    p_16 = p_0t - _vec_const(p_offsets[0]) - r_06 * _vec_const(p_tool)

    sp5_input_lines: list[str] = []
    sp5_input_lines.append("# T_target-derived SP5 input (p_16).")
    sp5_input_lines.extend(_emit_vec_components("p_16", p_16, prefix="sp5_in"))
    sp5_input_lines.append(_emit_const_array("_NEG_P1", -p_offsets[1]))
    sp5_input_lines.append(_emit_const_array("_P_2_CONST", p_offsets[2]))
    sp5_input_lines.append(_emit_const_array("_P_3_CONS", p_3_consolidated))
    sp5_input_lines.append(_emit_const_array("_NEG_AXIS_0", -axes[0]))
    sp5_input_lines.append(_emit_const_array("_AXIS_1", axes[1]))
    sp5_input_lines.append(_emit_const_array("_AXIS_2", axes[2]))

    # Inside the (q1, q2, q3) loop: r_36 + SP4 for q5 + SP1 for q4, q6.
    q1_sym = sp.Symbol("q1", real=True)
    q2_sym = sp.Symbol("q2", real=True)
    q3_sym = sp.Symbol("q3", real=True)
    rot_neg_axis_0_q1 = _rotation_matrix_sym(_vec_const(-axes[0]), q1_sym)
    rot_neg_axis_1_q2 = _rotation_matrix_sym(_vec_const(-axes[1]), q2_sym)
    rot_neg_axis_2_q3 = _rotation_matrix_sym(_vec_const(-axes[2]), q3_sym)
    r_36 = rot_neg_axis_2_q3 * rot_neg_axis_1_q2 * rot_neg_axis_0_q1 * r_06

    d_q5 = (_vec_const(axes[3]).T * r_36 * _vec_const(axes[5]))[0, 0]
    sp4_q5 = sp4_branches_sym(_vec_const(axes[3]), _vec_const(axes[4]), _vec_const(axes[5]), d_q5)
    q5_lines = _render_sp4_block(
        prefix="q5",
        sp4_out=sp4_q5,
        comment="SP4 for q5 (wrist pitch).",
    )

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
        comment="SP1 for q4 (wrist roll-1).",
    )

    q6_expr = sp1_theta_sym(
        -_vec_const(axes[5]),
        rot_neg_axis_4_q5 * _vec_const(axes[3]),
        r_36.T * _vec_const(axes[3]),
    )
    q6_lines = _render_atan2_block(
        prefix="q6",
        var_name="q6",
        expr=q6_expr,
        comment="SP1 for q6 (wrist roll-2).",
    )

    return _assemble_spherical(
        destructure_lines=_render_destructure(target),
        sp5_input_lines=sp5_input_lines,
        q5_lines=q5_lines,
        q4_lines=q4_lines,
        q6_lines=q6_lines,
    )


def _emit_vec_components(name: str, vec_expr: sp.Matrix, *, prefix: str) -> list[str]:
    cx, cy, cz = vec_expr[0, 0], vec_expr[1, 0], vec_expr[2, 0]
    cse_subs, [final_x, final_y, final_z] = sp.cse(
        [cx, cy, cz], symbols=sp.numbered_symbols(prefix=f"{prefix}_x")
    )
    lines: list[str] = []
    for sym, sub in cse_subs:
        lines.append(f"{sym} = {sp.pycode(sub)}")
    lines.append(f"{name}_x = {sp.pycode(final_x)}")
    lines.append(f"{name}_y = {sp.pycode(final_y)}")
    lines.append(f"{name}_z = {sp.pycode(final_z)}")
    return lines


def _emit_const_array(name: str, v: object) -> str:
    return f"{name} = np.array([{float(v[0])!r}, {float(v[1])!r}, {float(v[2])!r}])"  # type: ignore[index]


def _indent(lines: list[str], spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in lines)


def _assemble_spherical(
    *,
    destructure_lines: list[str],
    sp5_input_lines: list[str],
    q5_lines: list[str],
    q4_lines: list[str],
    q6_lines: list[str],
) -> str:
    parts = [
        "def _solve_algebraic(T_target):",
        '    """Algebraic IK candidates. SP5 runtime; SP4+SP1+SP1 inlined."""',
        _indent(destructure_lines, 4),
        "    candidates = []",
        "",
        _indent(sp5_input_lines, 4),
        "    p_16 = np.array([p_16_x, p_16_y, p_16_z])",
        "    t123_solutions, _ = _sp5_runtime.solve(",
        "        _NEG_P1, p_16, _P_2_CONST, _P_3_CONS, _NEG_AXIS_0, _AXIS_1, _AXIS_2",
        "    )",
        "",
        "    for q1, q2, q3 in t123_solutions:",
        "        s1 = math.sin(q1)",
        "        c1 = math.cos(q1)",
        "        s2 = math.sin(q2)",
        "        c2 = math.cos(q2)",
        "        s3 = math.sin(q3)",
        "        c3 = math.cos(q3)",
        _indent(q5_lines, 8),
        "",
        "        for q5 in (theta_q5_plus, theta_q5_minus):",
        "            s5 = math.sin(q5)",
        "            c5 = math.cos(q5)",
        _indent(q4_lines, 12),
        _indent(q6_lines, 12),
        "            candidates.append([q1, q2, q3, q4, q5, q6])",
        "    return candidates",
        "",
    ]
    return "\n".join(parts)
