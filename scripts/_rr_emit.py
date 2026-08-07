"""sympy -> C++ emitter for the Raghavan-Roth coefficient matrices (#490 slice 4).

The per-arm elimination coefficients P_sin/P_cos/P_one (14x9) and Q (14x8) are
symbolic polynomials in the 12 target-matrix entries (T_0..T_11), with the arm's
numeric DH baked in. This renders them as one CSE'd C++ function feeding the
shared RR runtime (general_6r.hpp) -- the "emit, never hand-port the hard tier"
half of ADR-0001.

The symbolic matrices come straight from the derivation metadata
(_sym_p_sin/... stashed by _derive_pq_for_arm); this module only prints them,
so there is zero re-derivation of the algebra in C++.
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter


class _RrPrinter(CXX11CodePrinter):  # type: ignore[misc]  # sympy is untyped
    """C++ printer that expands small integer powers to multiplication.

    The RR coefficients are low-degree polynomials; after CSE the Pow bases are
    single temporaries, so ``_t5*_t5`` is both faster and cleaner than a
    ``std::pow`` call (which the compiler will not fold without -ffast-math)."""

    def _print_Pow(self, expr: Any) -> str:
        if expr.exp.is_Integer and 2 <= int(expr.exp) <= 4:
            base = self.parenthesize(expr.base, sp.printing.precedence.PRECEDENCE["Mul"])
            return "*".join([str(base)] * int(expr.exp))
        return str(super()._print_Pow(expr))


def render_rr_coeffs(meta: dict[str, object]) -> list[str]:
    """Render the ``rr_coeffs`` C++ function from the derivation metadata.

    CSE runs across all four matrices jointly so subexpressions shared between
    P_sin/P_cos/P_one/Q collapse to one temporary. Only non-zero entries are
    assigned (the matrices are zeroed first), matching the sparse RR structure.
    """
    printer = _RrPrinter({"user_functions": {}})
    t_syms = list(meta["_sym_t_target"])  # type: ignore[call-overload]
    mats = [
        ("p_sin", meta["_sym_p_sin"]),
        ("p_cos", meta["_sym_p_cos"]),
        ("p_one", meta["_sym_p_one"]),
        ("q", meta["_sym_q"]),
    ]

    exprs: list[sp.Expr] = []
    layout: list[tuple[str, int, int]] = []
    for name, mat in mats:
        m = sp.Matrix(mat)
        for r in range(m.rows):
            for c in range(m.cols):
                layout.append((name, r, c))
                exprs.append(m[r, c])

    replacements, reduced = sp.cse(exprs, symbols=sp.numbered_symbols("_t"), optimizations="basic")

    lines = [
        "// Elimination coefficients P_sin/P_cos/P_one (14x9) + Q (14x8) as CSE'd",
        "// polynomials in the 12 target entries. Emitted from the sympy derivation.",
        "inline void rr_coeffs(const double t12[12], rr_detail::Mat14x9& p_sin,",
        "                      rr_detail::Mat14x9& p_cos, rr_detail::Mat14x9& p_one,",
        "                      rr_detail::Mat14x8& q) {",
    ]
    for i, sym in enumerate(t_syms):
        lines.append(f"  const double {sym} = t12[{i}];")
    lines.append("  p_sin.setZero(); p_cos.setZero(); p_one.setZero(); q.setZero();")
    for sym, sub in replacements:
        lines.append(f"  const double {printer.doprint(sym)} = {printer.doprint(sub)};")
    for (name, r, c), expr in zip(layout, reduced, strict=True):
        if expr == 0:
            continue
        lines.append(f"  {name}({r}, {c}) = {printer.doprint(expr)};")
    lines.append("}")
    return lines
