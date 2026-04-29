"""Composer for ``ikgeo.spherical_two_intersecting`` (Puma's intersecting-shoulder path).

Mirrors :func:`ssik.solvers.ikgeo.spherical_two_intersecting.solve`:

  1. SP3 for q3 (elbow): closed-form trig.
  2. For each q3 branch:
       - SP2 for (q1, q2): closed-form, 2 branches.
       - For each (q1, q2):
           - r_36 = Rot(-axes[2], q3) @ Rot(-axes[1], q2) @ Rot(-axes[0], q1) @ r_06
           - SP4 for q5 (wrist pitch): 2 branches.
           - For each q5:
               - SP1 for q4 (closed-form atan2)
               - SP1 for q6 (closed-form atan2)

Up to 8 candidates per pose: 2 q3 x 2 (q1,q2) x 2 q5.

Topology preconditions (gated by dispatcher):

  - axes (3, 4, 5) intersect at a common point (spherical wrist).
  - p[1] = 0 (joints 0, 1 share an origin).

All math closed-form; full IKFast-style trig in the rendered artifact.
"""

from __future__ import annotations

import sympy as sp

from ssik._kinbody import KinBody
from ssik.codegen._compose._target import make_target_symbols
from ssik.codegen._compose.spherical_two_parallel import (
    _mat_const,
    _render_atan2_block,
    _render_destructure,
    _render_sp2_block,
    _render_sp4_block,
    _rotation_matrix_sym,
    _vec_const,
)
from ssik.codegen._symbolic.sp1 import sp1_theta_sym
from ssik.codegen._symbolic.sp2 import sp2_branches_sym
from ssik.codegen._symbolic.sp3 import sp3_branches_sym
from ssik.codegen._symbolic.sp4 import sp4_branches_sym

__all__ = ["compose", "render_constants_header"]


_DEG_SQ = 1e-16
_FEAS_TOL = 1e-8


def render_constants_header() -> str:
    return f"import math\n\n_DEG_SQ = {_DEG_SQ!r}\n_FEAS_TOL = {_FEAS_TOL!r}\n"


def compose(kb: KinBody) -> str:
    """Render ``_solve_algebraic`` for a Puma-class arm with intersecting shoulder."""
    target = make_target_symbols()

    axes = [j.axis for j in kb.joints]
    p_offsets = [j.T_left[:3, 3].copy() for j in kb.joints]
    p_3_consolidated = p_offsets[3] + p_offsets[4] + p_offsets[5]
    p_tool = kb.joints[-1].T_right[:3, 3].copy()
    r_home = kb.joints[-1].T_right[:3, :3].copy()

    r_home_sym = _mat_const(r_home)
    r_06 = target.r * r_home_sym.T
    p_0t = target.p
    p_16 = p_0t - r_06 * _vec_const(p_tool) - _vec_const(p_offsets[0])

    # ---------- SP3 for q3 ----------
    # Note: runtime calls sp3.solve(axes[2], p[3], -p[2], ||p_16||) which has
    # arguments in (k, p, q, d) form. Symbolic SP3 wraps SP4 with target shift.
    p_16_norm = sp.sqrt(p_16.dot(p_16))
    sp3_q3 = sp3_branches_sym(
        _vec_const(axes[2]),
        _vec_const(p_3_consolidated),
        -_vec_const(p_offsets[2]),
        p_16_norm,
    )
    q3_lines = _render_sp4_block(
        prefix="q3",
        sp4_out=sp3_q3,
        comment="SP3 for q3 (elbow): reduces to SP4 with |p_16| target.",
    )

    # ---------- SP2 for (q1, q2) ----------
    q3_sym = sp.Symbol("q3", real=True)
    rot_axis_2_q3 = _rotation_matrix_sym(_vec_const(axes[2]), q3_sym)
    sp2_q_target = _vec_const(p_offsets[2]) + rot_axis_2_q3 * _vec_const(p_3_consolidated)
    sp2_q1_q2 = sp2_branches_sym(
        -_vec_const(axes[0]),
        _vec_const(axes[1]),
        p_16,
        sp2_q_target,
    )
    q1q2_lines = _render_sp2_block(
        prefix="q12",
        sp2_out=sp2_q1_q2,
        comment="SP2 for (q1, q2) (shoulder pan + pitch jointly).",
    )

    # ---------- r_36 = Rot(-axes[2], q3) @ Rot(-axes[1], q2) @ Rot(-axes[0], q1) @ r_06 ----------
    q1_sym = sp.Symbol("q1", real=True)
    q2_sym = sp.Symbol("q2", real=True)
    rot_neg_axis_2_q3 = _rotation_matrix_sym(_vec_const(-axes[2]), q3_sym)
    rot_neg_axis_1_q2 = _rotation_matrix_sym(_vec_const(-axes[1]), q2_sym)
    rot_neg_axis_0_q1 = _rotation_matrix_sym(_vec_const(-axes[0]), q1_sym)
    r_36 = rot_neg_axis_2_q3 * rot_neg_axis_1_q2 * rot_neg_axis_0_q1 * r_06

    # ---------- SP4 for q5 (wrist pitch) ----------
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

    return _assemble_intersecting(
        destructure_lines=_render_destructure(target),
        q3_lines=q3_lines,
        q1q2_lines=q1q2_lines,
        q5_lines=q5_lines,
        q4_lines=q4_lines,
        q6_lines=q6_lines,
    )


def _indent(lines: list[str], spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in lines)


def _assemble_intersecting(
    *,
    destructure_lines: list[str],
    q3_lines: list[str],
    q1q2_lines: list[str],
    q5_lines: list[str],
    q4_lines: list[str],
    q6_lines: list[str],
) -> str:
    """Stitch the spherical_two_intersecting blocks into ``_solve_algebraic``."""
    parts = [
        "def _solve_algebraic(T_target):",
        '    """Algebraic IK candidates. Up to 8; verify + dedup in solve()."""',
        _indent(destructure_lines, 4),
        "    candidates = []",
        "",
        _indent(q3_lines, 4),
        "",
        "    for q3 in (theta_q3_plus, theta_q3_minus):",
        "        s3 = math.sin(q3)",
        "        c3 = math.cos(q3)",
        _indent(q1q2_lines, 8),
        "",
        "        for q1, q2 in (",
        "            (theta_q12_1a, theta_q12_2a),",
        "            (theta_q12_1b, theta_q12_2b),",
        "        ):",
        "            s1 = math.sin(q1)",
        "            c1 = math.cos(q1)",
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
