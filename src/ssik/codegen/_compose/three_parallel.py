"""Composer for ``ikgeo.three_parallel`` (UR class).

Mirrors :func:`ssik.solvers.ikgeo.three_parallel.solve` but renders an
inlined per-arm version of the IK chain.

three_parallel structure:

  1. Build SP6 inputs (h_sp, k_sp, p_sp, d1=axes[1].(p[1]+p[2]+p[3]+p[4]),
     d2=0). h_sp/k_sp are entirely constant per arm; p_sp has two
     T_target-dependent entries (p_16 and r_06 @ axes[5]).
  2. Call sp6.solve at runtime to get (theta_0, theta_4) candidates.
     The SP6 internals (QR, ellipse-intersection, Gauss-Newton) stay
     runtime; specialising them is a Phase 4 Cython concern.
  3. For each (q1, q5) candidate:
       - SP1 for theta14 = q1+q2+q3+q4 (closed-form atan2; inlined)
       - SP1 for q6 (inlined)
       - Compute d_inner / d_elbow (inlined)
       - SP3 for q3 (closed-form via SP4; inlined)
       - For each q3 branch:
           - SP1 for q2 (inlined)
           - q4 = wrap_to_pi(theta14 - q2 - q3)

Speedup story: in pure Python, the SP6 call is the dominant cost
(~80% of total). Specialisation only saves the post-SP6 chain (~20%).
The full ~100-1000x story arrives when Cython compiles the SP6
internals + the inlined post-chain into native code.

Bulletproof: the composed artifact must enumerate the same q-set as
the runtime solver across 100+ random poses.
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
from ssik.codegen._symbolic.sp3 import sp3_branches_sym

__all__ = ["compose", "render_constants_header"]


_DEG_SQ = 1e-16
_FEAS_TOL = 1e-8


def render_constants_header() -> str:
    """Return constants/imports the rendered three_parallel artifact needs."""
    return (
        "import math\n"
        "from ssik.subproblems import sp6 as _sp6_runtime\n"
        "\n"
        f"_DEG_SQ = {_DEG_SQ!r}\n"
        f"_FEAS_TOL = {_FEAS_TOL!r}\n"
    )


def compose(kb: KinBody) -> str:
    """Render ``_solve_algebraic`` for a UR-class arm.

    :param kb: a POE-normalised :class:`KinBody` matching the
        ``three_parallel`` topology (3 consecutive parallel axes at
        positions (1, 2, 3)).
    :returns: Python source for ``_solve_algebraic(T_target)``.
    """
    target = make_target_symbols()

    axes = [j.axis for j in kb.joints]
    p_offsets = [j.T_left[:3, 3].copy() for j in kb.joints]
    p_tool = kb.joints[-1].T_right[:3, 3].copy()
    r_home = kb.joints[-1].T_right[:3, :3].copy()

    # ---- T_target-derived symbolic vectors ----
    r_home_sym = _mat_const(r_home)
    r_06 = target.r * r_home_sym.T
    p_0t = target.p
    p_16 = p_0t - _vec_const(p_offsets[0]) - r_06 * _vec_const(p_tool)
    r_06_axes5 = r_06 * _vec_const(axes[5])

    # ---- SP6 input vectors ----
    # h_sp = [axes[1], axes[1], axes[1], axes[1]] (all constant)
    # k_sp = [-axes[0], axes[4], -axes[0], axes[4]] (constants)
    # p_sp = [p_16, -p[5], r_06 @ axes[5], -axes[5]]
    # d1 = axes[1] . (p[1]+p[2]+p[3]+p[4]); d2 = 0
    d1 = float(axes[1] @ (p_offsets[1] + p_offsets[2] + p_offsets[3] + p_offsets[4]))

    # Render: emit the p_16 / r_06_axes5 components inline so the artifact
    # builds the SP6 input arrays from inlined target-pose math.
    sp6_input_lines: list[str] = []
    sp6_input_lines.append("# T_target-derived SP6 inputs.")
    p_16_components = _emit_vec_components("p_16", p_16, prefix="sp6_in")
    sp6_input_lines.extend(p_16_components)
    r06a5_components = _emit_vec_components("r_06_axes5", r_06_axes5, prefix="sp6_in")
    sp6_input_lines.extend(r06a5_components)
    sp6_input_lines.append(_emit_const_array("_H_SP_0", axes[1]))
    sp6_input_lines.append(_emit_const_array("_K_SP_0", -axes[0]))
    sp6_input_lines.append(_emit_const_array("_K_SP_1", axes[4]))
    sp6_input_lines.append(_emit_const_array("_NEG_P_5", -p_offsets[5]))
    sp6_input_lines.append(_emit_const_array("_NEG_AXES_5", -axes[5]))

    # ---- Inside the (q1, q5) loop: theta14 = SP1 of the parallel-trio rotation ----
    q1_sym = sp.Symbol("q1", real=True)
    q5_sym = sp.Symbol("q5", real=True)
    rot_axes_0_q1 = _rotation_matrix_sym(_vec_const(axes[0]), q1_sym)
    rot_axes_4_q5 = _rotation_matrix_sym(_vec_const(axes[4]), q5_sym)

    sp1_theta14_p = rot_axes_4_q5 * _vec_const(axes[5])
    sp1_theta14_q = rot_axes_0_q1.T * r_06 * _vec_const(axes[5])
    theta14_expr = sp1_theta_sym(_vec_const(axes[1]), sp1_theta14_p, sp1_theta14_q)
    theta14_lines = _render_atan2_block(
        prefix="th14",
        var_name="theta14",
        expr=theta14_expr,
        comment="SP1 for theta14 = q1+q2+q3+q4 (sum of parallel-axis rotations).",
    )

    # SP1 for q6
    sp1_q6_p = rot_axes_4_q5.T * _vec_const(axes[1])
    sp1_q6_q = r_06.T * rot_axes_0_q1 * _vec_const(axes[1])
    q6_expr = sp1_theta_sym(-_vec_const(axes[5]), sp1_q6_p, sp1_q6_q)
    q6_lines = _render_atan2_block(
        prefix="q6",
        var_name="q6",
        expr=q6_expr,
        comment="SP1 for q6 (wrist roll-2): closed-form atan2.",
    )

    # ---- d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4] ----
    theta14_sym = sp.Symbol("theta14", real=True)
    rot_axes_1_t14 = _rotation_matrix_sym(_vec_const(axes[1]), theta14_sym)
    d_inner_vec = (
        rot_axes_0_q1.T * p_16
        - _vec_const(p_offsets[1])
        - rot_axes_1_t14 * rot_axes_4_q5 * _vec_const(p_offsets[5])
        - rot_axes_1_t14 * _vec_const(p_offsets[4])
    )
    d_inner_lines: list[str] = []
    d_inner_lines.append("# d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]")
    d_inner_components = _emit_vec_components("d_inner", d_inner_vec, prefix="dinr")
    d_inner_lines.extend(d_inner_components)
    d_inner_lines.append(
        "d_elbow = math.sqrt(d_inner_x*d_inner_x + d_inner_y*d_inner_y + d_inner_z*d_inner_z)"
    )

    # ---- SP3 for q3: reduces to SP4 with target d_elbow ----
    d_elbow_sym = sp.Symbol("d_elbow", real=True)
    sp3_q3 = sp3_branches_sym(
        _vec_const(axes[1]),
        -_vec_const(p_offsets[3]),
        _vec_const(p_offsets[2]),
        d_elbow_sym,
    )
    q3_lines = _render_sp4_block(
        prefix="q3",
        sp4_out=sp3_q3,
        comment="SP3 for q3 (elbow): reduces to SP4 with d_elbow target.",
    )

    # ---- SP1 for q2: p = p[2] + rotate(axes[1], q3, p[3]), q = d_inner ----
    q3_sym = sp.Symbol("q3", real=True)
    rot_axes_1_q3 = _rotation_matrix_sym(_vec_const(axes[1]), q3_sym)
    sp1_q2_p = _vec_const(p_offsets[2]) + rot_axes_1_q3 * _vec_const(p_offsets[3])
    d_inner_x_sym = sp.Symbol("d_inner_x", real=True)
    d_inner_y_sym = sp.Symbol("d_inner_y", real=True)
    d_inner_z_sym = sp.Symbol("d_inner_z", real=True)
    d_inner_vec_sym = sp.Matrix([d_inner_x_sym, d_inner_y_sym, d_inner_z_sym])
    q2_expr = sp1_theta_sym(_vec_const(axes[1]), sp1_q2_p, d_inner_vec_sym)
    q2_lines = _render_atan2_block(
        prefix="q2",
        var_name="q2",
        expr=q2_expr,
        comment="SP1 for q2 (shoulder pitch): closed-form atan2.",
    )

    return _assemble_three_parallel(
        destructure_lines=_render_destructure(target),
        sp6_input_lines=sp6_input_lines,
        d1_const=d1,
        theta14_lines=theta14_lines,
        q6_lines=q6_lines,
        d_inner_lines=d_inner_lines,
        q3_lines=q3_lines,
        q2_lines=q2_lines,
    )


def _emit_vec_components(name: str, vec_expr: sp.Matrix, *, prefix: str) -> list[str]:
    """CSE-and-emit a 3x1 sympy Matrix as ``<name>_x``, ``_y``, ``_z`` lines."""
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
    """Emit ``np.array(...)`` for a constant 3-vector."""
    return f"{name} = np.array([{float(v[0])!r}, {float(v[1])!r}, {float(v[2])!r}])"  # type: ignore[index]


def _indent(lines: list[str], spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in lines)


def _assemble_three_parallel(
    *,
    destructure_lines: list[str],
    sp6_input_lines: list[str],
    d1_const: float,
    theta14_lines: list[str],
    q6_lines: list[str],
    d_inner_lines: list[str],
    q3_lines: list[str],
    q2_lines: list[str],
) -> str:
    """Stitch the three_parallel composer's blocks into ``_solve_algebraic``."""
    parts = [
        "def _solve_algebraic(T_target):",
        '    """Algebraic IK candidates. Calls runtime SP6 for (q1, q5);',
        "    inlines the post-SP6 SP1+SP3+SP1 chain.",
        '    """',
        _indent(destructure_lines, 4),
        "    candidates = []",
        "",
        _indent(sp6_input_lines, 4),
        "    # Build SP6 input arrays. h_sp / k_sp constant per arm; p_sp[0],",
        "    # p_sp[2] depend on T_target via the inlined components above.",
        "    p_16 = np.array([p_16_x, p_16_y, p_16_z])",
        "    r_06_axes5 = np.array([r_06_axes5_x, r_06_axes5_y, r_06_axes5_z])",
        "    h_sp = (_H_SP_0, _H_SP_0, _H_SP_0, _H_SP_0)",
        "    k_sp = (_K_SP_0, _K_SP_1, _K_SP_0, _K_SP_1)",
        "    p_sp = (p_16, _NEG_P_5, r_06_axes5, _NEG_AXES_5)",
        f"    theta15_solutions, _ = _sp6_runtime.solve(h_sp, k_sp, p_sp, {d1_const!r}, 0.0)",
        "",
        "    for q1, q5 in theta15_solutions:",
        "        s1 = math.sin(q1)",
        "        c1 = math.cos(q1)",
        "        s5 = math.sin(q5)",
        "        c5 = math.cos(q5)",
        _indent(theta14_lines, 8),
        _indent(q6_lines, 8),
        "        s14 = math.sin(theta14)",
        "        c14 = math.cos(theta14)",
        _indent(d_inner_lines, 8),
        _indent(q3_lines, 8),
        "",
        "        for q3 in (theta_q3_plus, theta_q3_minus):",
        "            s3 = math.sin(q3)",
        "            c3 = math.cos(q3)",
        _indent(q2_lines, 12),
        "            q4 = ((theta14 - q2 - q3 + math.pi) % (2.0 * math.pi)) - math.pi",
        "            candidates.append([q1, q2, q3, q4, q5, q6])",
        "    return candidates",
        "",
    ]
    return "\n".join(parts)
