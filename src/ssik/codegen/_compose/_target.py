"""Sympy symbol pack for the IK target pose ``T_target``.

A 4x4 SE(3) target has 12 free entries (3 translation + 9 rotation,
with the 6 rotation constraints implicit). We expose them as sympy
:class:`Symbol`s so composers can build expressions in terms of the
target pose, then the codegen emits Python that destructures
``T_target[i, j]`` into these names.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

__all__ = ["TargetSymbols", "make_target_symbols"]


@dataclass(frozen=True, kw_only=True)
class TargetSymbols:
    """The 12-symbol pack representing ``T_target``'s independent entries.

    Attributes:
        r: 3x3 sympy Matrix of rotation symbols ``r_00..r_22``.
        p: 3x1 sympy Matrix of translation symbols ``p_x, p_y, p_z``.
        flat: tuple of all 12 symbols in row-major rotation order then
            translation, useful for the codegen's destructure block.
    """

    r: sp.Matrix
    p: sp.Matrix
    flat: tuple[sp.Symbol, ...]


def make_target_symbols() -> TargetSymbols:
    """Allocate fresh symbols for ``T_target``."""
    r_syms = [[sp.Symbol(f"r_{i}{j}", real=True) for j in range(3)] for i in range(3)]
    p_syms = [sp.Symbol(name, real=True) for name in ("p_x", "p_y", "p_z")]
    flat = tuple([r_syms[i][j] for i in range(3) for j in range(3)] + p_syms)
    return TargetSymbols(
        r=sp.Matrix(r_syms),
        p=sp.Matrix(p_syms),
        flat=flat,
    )
