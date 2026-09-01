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


def render_rr_coeffs(
    meta: dict[str, object],
    name: str = "rr_coeffs",
    zero_threshold: float = 0.0,
    use_cse: bool = True,
) -> list[str]:
    """Render the ``<name>`` C++ coefficient function from the derivation metadata.

    CSE runs across all four matrices jointly so subexpressions shared between
    P_sin/P_cos/P_one/Q collapse to one temporary. Only non-zero entries are
    assigned (the matrices are zeroed first), matching the sparse RR structure.

    ``zero_threshold`` (relative to the largest coefficient magnitude across all
    four matrices) drops numerical-noise coefficients before CSE. A well-posed
    arm produces none, but a jointlock arm's DEGENERATE locked sub-chains (near-
    parallel axes at some lock samples) push poe_to_dh products down to ~1e-18 to
    ~1e-49 -- values that are numerically zero but whose exact low bits (and
    whether they land at exactly 0) are BLAS-backend-sensitive. Left in, they make
    the CSE temporary set platform-dependent, so the emitted header structure
    differs macOS vs Linux and trips the --check drift guard (#536). Thresholding
    them to 0 makes the CSE -- and thus the whole header -- byte-deterministic.
    Default 0.0 (off) leaves the general_6r arms exactly as committed.
    """
    printer = _RrPrinter({"user_functions": {}})
    t_syms = list(meta["_sym_t_target"])  # type: ignore[call-overload]
    mats = [
        ("p_sin", meta["_sym_p_sin"]),
        ("p_cos", meta["_sym_p_cos"]),
        ("p_one", meta["_sym_p_one"]),
        ("q", meta["_sym_q"]),
    ]

    sp_mats = [(mat_name, sp.Matrix(mat)) for mat_name, mat in mats]
    if zero_threshold > 0.0:
        scale = max(
            (abs(float(f)) for _, m in sp_mats for e in m for f in e.atoms(sp.Float)),
            default=0.0,
        )
        # Floor the scale at 1.0: RR coefficients are DH products (O(1)), so 1.0 is
        # a physical upper bound on a genuine coefficient's scale. A DEGENERATE
        # locked sub-chain can have its entire coefficient matrix be ~1e-18 noise
        # (scale ~1e-18); a purely relative cutoff would then be ~1e-30 and drop
        # nothing. Flooring keeps the cutoff at zero_threshold (1e-12) so the noise
        # is zeroed regardless -- such a sub-chain's RR unit collapses to all-zero
        # (deterministic), and its (spurious) solutions are dropped by the 7R
        # re-verify anyway.
        cutoff = zero_threshold * max(scale, 1.0)
        if cutoff > 0.0:
            drop = {
                f: sp.Integer(0)
                for _, m in sp_mats
                for e in m
                for f in e.atoms(sp.Float)
                if abs(float(f)) < cutoff
            }
            sp_mats = [(mat_name, m.xreplace(drop)) for mat_name, m in sp_mats]

    exprs: list[sp.Expr] = []
    layout: list[tuple[str, int, int]] = []
    for mat_name, m in sp_mats:
        for r in range(m.rows):
            for c in range(m.cols):
                layout.append((mat_name, r, c))
                exprs.append(m[r, c])

    if use_cse:
        replacements, reduced = sp.cse(
            exprs, symbols=sp.numbered_symbols("_t"), optimizations="basic"
        )
    else:
        # No CSE: emit each non-zero entry's full polynomial inline. sp.cse shares
        # subexpressions by EXACT Float equality, so backend-variant low bits break
        # a share on one platform but not the other -> the temp set (and header
        # structure) is not byte-portable, which no threshold/snap fully fixes
        # (#536). Without sharing the structure is exactly the non-zero-coefficient
        # set, which the zero_threshold makes deterministic. Used on the jointlock
        # path (degenerate sub-chains); general_6r keeps CSE (byte-stable there).
        replacements, reduced = [], exprs

    lines = [
        "// Elimination coefficients P_sin/P_cos/P_one (14x9) + Q (14x8) as CSE'd",
        "// polynomials in the 12 target entries. Emitted from the sympy derivation.",
        f"inline void {name}(const double t12[12], rr_detail::Mat14x9& p_sin,",
        "                      rr_detail::Mat14x9& p_cos, rr_detail::Mat14x9& p_one,",
        "                      rr_detail::Mat14x8& q) {",
    ]
    for i, sym in enumerate(t_syms):
        lines.append(f"  const double {sym} = t12[{i}];")
    lines.append("  p_sin.setZero(); p_cos.setZero(); p_one.setZero(); q.setZero();")
    for sym, sub in replacements:
        lines.append(f"  const double {printer.doprint(sym)} = {printer.doprint(sub)};")
    for (mat_name, r, c), expr in zip(layout, reduced, strict=True):
        if expr == 0:
            continue
        lines.append(f"  {mat_name}({r}, {c}) = {printer.doprint(expr)};")
    lines.append("}")
    return lines
