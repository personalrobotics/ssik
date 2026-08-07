#!/usr/bin/env python
"""Generate a runtime-parity fixture for the shared C++ RR runtime (#490).

Isolates the C++ general_6r_core pipeline (eliminate -> weierstrass -> build_m ->
RealQZ eigensolve -> back-substitute -> dedup) from the not-yet-built coefficient
emitter: Python evaluates P_sin/P_cos/P_one/Q at each pose and dumps them into the
fixture, so the C++ test feeds those exact matrices and compares its solution SET
against Python's solve_all_ik. Once this passes, the emitter (slice 4) is the only
remaining unknown.

FP results (eigensolve), so this is git-ignored like the other *_parity.hpp and
regenerated from the current oracle before the C++ build.

Usage: python scripts/cpp_rr_parity.py <arm> [n_poses]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import cast

import numpy as np

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import _cached_best_leftvar, build_pq, solve_all_ik

_REPO = Path(__file__).resolve().parent.parent


def _f(x: float) -> str:
    return repr(float(x))


def _mat_literal(m: np.ndarray) -> str:
    """Row-major brace-init list for an Eigen fixed matrix (<< operator feed)."""
    return ", ".join(_f(v) for v in np.asarray(m, dtype=np.float64).reshape(-1))


def _rowmajor_map(cpp_type: str, arr: str) -> str:
    """Eigen::Map a flat row-major double[] as a fixed matrix. Plain-array
    brace-init compiles in seconds where a 126-deep `<<` chain takes minutes."""
    return f"Eigen::Map<const Eigen::Matrix<double, {cpp_type}, Eigen::RowMajor>>({arr})"


def main() -> None:
    arm = sys.argv[1]
    n_poses = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB

    dh = poe_to_dh(kb)
    alpha, a, d = dh.to_dh_tuple()
    theta_offset = dh.theta_offset
    t_pre_inv = np.linalg.inv(dh.t_pre)
    t_post_inv = np.linalg.inv(dh.t_post)

    policy = DEFAULT_TOLERANCE_POLICY
    fk_atol = policy.subproblem_numerical
    dedup_atol = policy.subproblem_dedup

    # Resolve the AE-3 leftvar (as solve_all_ik does) + role metadata once.
    lin = int(_cached_best_leftvar(tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist())))
    _ps, _pc, _po, _q, meta = build_pq(
        dh.to_dh_tuple(), np.eye(4), linearity_joint=lin, return_metadata=True
    )
    lb0, lb1 = cast("tuple[int, int]", meta["left_bilinear"])
    rb0, rb1 = cast("tuple[int, int]", meta["right_bilinear"])
    drop = cast(int, meta["drop_joint"])

    rng = np.random.default_rng(12345)
    ranges = [j.limits if j.limits else (-np.pi, np.pi) for j in kb.joints]

    poses = []
    for _ in range(n_poses):
        q_seed = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t_poe = np.asarray(poe_forward_kinematics(kb, q_seed), dtype=np.float64)
        t_dh = t_pre_inv @ t_poe @ t_post_inv
        p_sin, p_cos, p_one, q_mat = build_pq(
            dh.to_dh_tuple(), t_dh, linearity_joint=lin, return_metadata=False
        )
        inner, _is_ls = solve_all_ik(
            dh.to_dh_tuple(), t_dh, fk_atol=fk_atol, dedup_atol=dedup_atol, linearity_joint=lin
        )
        sols_poe = [s.q - theta_offset for s in inner]
        poses.append((t_poe, p_sin, p_cos, p_one, q_mat, sols_poe))

    out = []
    out.append("// AUTO-GENERATED runtime-parity fixture for the C++ RR runtime (#490).")
    out.append("// FP results (eigensolve) -- git-ignored, regenerated from the oracle.")
    out.append("#pragma once")
    out.append('#include "ssik_cpp/solvers/general_6r.hpp"')
    out.append("#include <vector>")
    out.append("namespace ssik::rr_parity {")
    out.append("struct Pose { Eigen::Matrix4d t; rr_detail::Mat14x9 p_sin, p_cos, p_one;")
    out.append("  rr_detail::Mat14x8 q; std::vector<std::array<double,6>> expected; };")
    out.append(f'inline const char* arm_name() {{ return "{arm}"; }}')
    out.append(f"inline double fk_atol() {{ return {_f(fk_atol)}; }}")
    out.append(f"inline double dedup_atol() {{ return {_f(dedup_atol)}; }}")

    out.append("inline RrConsts consts() {")
    out.append("  RrConsts r;")
    out.append(f"  r.alpha = {{{', '.join(_f(v) for v in alpha)}}};")
    out.append(f"  r.a = {{{', '.join(_f(v) for v in a)}}};")
    out.append(f"  r.d = {{{', '.join(_f(v) for v in d)}}};")
    out.append(f"  r.theta_offset = {{{', '.join(_f(v) for v in theta_offset)}}};")
    out.append(f"  r.t_pre_inv = (Eigen::Matrix4d() << {_mat_literal(t_pre_inv)}).finished();")
    out.append(f"  r.t_post_inv = (Eigen::Matrix4d() << {_mat_literal(t_post_inv)}).finished();")
    out.append(f"  r.linearity_joint = {lin};")
    out.append(f"  r.left_bilinear = {{{lb0}, {lb1}}};")
    out.append(f"  r.right_bilinear = {{{rb0}, {rb1}}};")
    out.append(f"  r.drop_joint = {drop};")
    out.append("  return r;")
    out.append("}")

    out.append("inline std::vector<Pose> poses() {")
    out.append("  std::vector<Pose> v;")
    for t_poe, p_sin, p_cos, p_one, q_mat, sols in poses:
        out.append("  {")
        out.append(f"    static const double t[16] = {{{_mat_literal(t_poe)}}};")
        out.append(f"    static const double ps[126] = {{{_mat_literal(p_sin)}}};")
        out.append(f"    static const double pc[126] = {{{_mat_literal(p_cos)}}};")
        out.append(f"    static const double po[126] = {{{_mat_literal(p_one)}}};")
        out.append(f"    static const double qq[112] = {{{_mat_literal(q_mat)}}};")
        out.append("    Pose p;")
        out.append(f"    p.t = {_rowmajor_map('4, 4', 't')};")
        out.append(f"    p.p_sin = {_rowmajor_map('14, 9', 'ps')};")
        out.append(f"    p.p_cos = {_rowmajor_map('14, 9', 'pc')};")
        out.append(f"    p.p_one = {_rowmajor_map('14, 9', 'po')};")
        out.append(f"    p.q = {_rowmajor_map('14, 8', 'qq')};")
        for s in sols:
            out.append(f"    p.expected.push_back({{{', '.join(_f(v) for v in s)}}});")
        out.append("    v.push_back(p);")
        out.append("  }")
    out.append("  return v;")
    out.append("}")
    out.append("}  // namespace ssik::rr_parity")

    dest = _REPO / "cpp" / "gen" / f"{arm}_rr_parity.hpp"
    dest.write_text("\n".join(out) + "\n")
    total = sum(len(s[5]) for s in poses)
    print(f"wrote {dest.relative_to(_REPO)}: {n_poses} poses, {total} expected solutions")


if __name__ == "__main__":
    main()
