#!/usr/bin/env python
"""Emit the standalone rr_coeffs(...) header for one RR arm (#490 slice 4 helper).

A thin CLI around _rr_emit.render_rr_coeffs, used to validate the emitter through
the runtime harness before it folds into cpp_emit.py (slice 5).

Usage: python scripts/cpp_rr_emit_coeffs.py <arm>
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from _rr_emit import render_rr_coeffs

from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import _cached_best_leftvar, _cached_derivation

_REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    arm = sys.argv[1]
    kb = importlib.import_module(f"ssik.prebuilt.{arm}")._KB
    dh = poe_to_dh(kb)
    alpha, a, d = dh.to_dh_tuple()
    lin = int(_cached_best_leftvar(tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist())))
    _ps, _pc, _po, _q, meta = _cached_derivation(
        tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist()), lin, False
    )
    body = render_rr_coeffs(meta)
    out = [
        "#pragma once",
        '#include "ssik_cpp/solvers/general_6r.hpp"',
        "namespace ssik::rr_emitted {",
        *body,
        "}  // namespace ssik::rr_emitted",
    ]
    dest = _REPO / "cpp" / "gen" / f"{arm}_rr_coeffs.hpp"
    dest.write_text("\n".join(out) + "\n")
    print(f"wrote {dest.relative_to(_REPO)}: {len(body)} lines")


if __name__ == "__main__":
    main()
