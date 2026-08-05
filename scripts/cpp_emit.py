#!/usr/bin/env python
"""Emit a native C++ artifact header + parity fixtures for one arm (#488).

Phase-0a scope: Band-A constant rendering (the KinBody -> a ``JointConsts``
struct) plus Python-computed FK/solve parity fixtures so the C++ side can be
checked against the reference at machine precision. This is the CppRenderer
precursor for #482; it will fold into ssik.core.codegen behind ``--target cpp``
once the solver bands land.

Two kinds of output, deliberately treated differently (#495):

- ``<arm>.hpp`` (the constants header) is the shippable native artifact. It is
  ``repr()`` of the baked KinBody floats -- no computation, byte-identical
  across platforms -- so it is committed and guarded by ``--check``.
- ``<arm>_fk_parity.hpp`` / ``<arm>_solve_parity.hpp`` are test scaffolding.
  They are floating-point *results* (FK/eigensolve), so they carry BLAS-backend
  ULP variance (#82) and are NOT byte-portable. They are git-ignored and
  regenerated fresh from the current oracle before every build (``--all``),
  then validated by ctest -- which is stronger than diffing stale bytes.

Usage:
  ``python scripts/cpp_emit.py <arm>``    emit one arm
  ``python scripts/cpp_emit.py --all``    re-emit every arm in cpp/gen
  ``python scripts/cpp_emit.py --check``  CI drift guard (constants headers)
"""

from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ssik._kinbody import KinBody
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.prebuilt._manifest import load_manifest

_REPO = Path(__file__).resolve().parent.parent


def _render_limits(joints: list[Any]) -> list[str]:
    """Baked JointLimits<DOF> so the self-contained artifact needs no runtime kb."""
    lo = ", ".join(_f(j.limits[0]) if j.limits else "0.0" for j in joints)
    hi = ", ".join(_f(j.limits[1]) if j.limits else "0.0" for j in joints)
    present = ", ".join("true" if j.limits else "false" for j in joints)
    return [
        "inline JointLimits<DOF> limits() {",
        "  JointLimits<DOF> l;",
        f"  l.lo = {{{lo}}};",
        f"  l.hi = {{{hi}}};",
        f"  l.present = {{{present}}};",
        "  return l;",
        "}",
    ]


# Per-solver self-contained C++ solve(): (include, body-lines). The artifact
# composes the header-only core solver + finalize with its baked constants, so a
# C++ consumer calls ssik::<arm>::solve(T) with zero Python. Families not listed
# here (or arms outside a family's native domain) emit constants only (self-
# contained solve pending their port).
def _render_solve(solver: str, kb: KinBody) -> tuple[str, list[str]] | None:
    if solver == "seven_r.srs":
        return _render_srs_solve(kb)
    fn = {
        "ikgeo.three_parallel": ("three_parallel", "three_parallel_artifact_solve"),
        "ikgeo.spherical_two_parallel": (
            "spherical_two_parallel",
            "spherical_two_parallel_artifact_solve",
        ),
    }.get(solver)
    if fn is None:
        return None
    header, artifact_solve = fn
    return (
        f'#include "ssik_cpp/solvers/{header}.hpp"',
        [
            "// Self-contained IK solve -- header-only C++, no Python runtime.",
            "inline std::vector<Solution<DOF>> solve(",
            "    const Pose& T, const ArtifactParams<DOF>& p = {}) {",
            f"  return {artifact_solve}(consts(), limits(), T, p);",
            "}",
        ],
    )


def _srs_bake(kb: KinBody) -> dict[str, Any] | None:
    """Bake the SrsConsts (base + branch-enumeration extras) for a canonical-ZYZ
    offset-free SRS arm, or None when the arm is outside srs_canonical_solve's
    native domain (offset/tilted wrist -> general #354 path, still Python).

    Mirrors ssik._native._srs_native_args' eligibility gate exactly so the emit
    and the runtime native path agree on which arms are self-contained."""
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY as pol
    from ssik.solvers.seven_r.srs import (  # type: ignore[attr-defined]
        _arm_constants,
        _classify_srs_7r_geometric,
    )

    cls = _classify_srs_7r_geometric(kb, pol)
    if cls is None or len(kb.joints) != 7:
        return None
    l_se, l_ew, ee_offset, origins = _arm_constants(kb, cls)
    j = kb.joints
    upper = origins[cls.elbow_index] - cls.shoulder_pivot
    u_home = upper / np.linalg.norm(upper)
    ez, ey = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    canonical = (
        np.allclose(j[0].axis, ez)
        and np.allclose(j[1].axis, ey)
        and np.allclose(u_home, ez)
        and np.allclose(j[4].axis, ez)
        and np.allclose(j[5].axis, ey)
        and np.allclose(j[6].axis, ez)
    )
    offset_free = np.allclose(origins[5], cls.wrist_pivot, atol=pol.axis_intersect)
    if not (canonical and offset_free):
        return None
    return {
        "l_se": float(l_se),
        "l_ew": float(l_ew),
        "ee_offset_local": np.asarray(ee_offset, dtype=np.float64),
        "shoulder_pivot": np.asarray(cls.shoulder_pivot, dtype=np.float64),
        "r_post_wrist": np.asarray(j[6].T_right[:3, :3], dtype=np.float64),
        "elbow_index": int(cls.elbow_index),
        "upper_home": np.asarray(origins[cls.elbow_index] - cls.shoulder_pivot, dtype=np.float64),
        "forearm_home": np.asarray(cls.wrist_pivot - origins[cls.elbow_index], dtype=np.float64),
    }


def _render_srs_solve(kb: KinBody) -> tuple[str, list[str]] | None:
    """Self-contained SRS solve(): bakes SrsConsts + composes srs_artifact_solve.
    None for arms outside the canonical-ZYZ offset-free native domain."""
    s = _srs_bake(kb)
    if s is None:
        return None
    return (
        '#include "ssik_cpp/solvers/srs.hpp"',
        [
            "// Baked SRS geometry (classifier + _arm_constants), zero runtime Python.",
            "inline SrsConsts srs_consts() {",
            "  SrsConsts s;",
            f"  s.l_se = {_f(s['l_se'])};",
            f"  s.l_ew = {_f(s['l_ew'])};",
            f"  s.ee_offset_local = {_vec3(s['ee_offset_local'])};",
            f"  s.shoulder_pivot = {_vec3(s['shoulder_pivot'])};",
            f"  s.r_post_wrist = {_mat3(s['r_post_wrist'])};",
            f"  s.elbow_index = {s['elbow_index']};",
            f"  s.upper_home = {_vec3(s['upper_home'])};",
            f"  s.forearm_home = {_vec3(s['forearm_home'])};",
            "  return s;",
            "}",
            "",
            "// Self-contained IK solve -- header-only C++, no Python runtime.",
            "inline std::vector<Solution<DOF>> solve(",
            "    const Pose& T, const ArtifactParams<DOF>& p = {}) {",
            "  return srs_artifact_solve(consts(), srs_consts(), limits(), T, p);",
            "}",
        ],
    )


def _consts_kinbody(arm_solver: str, kb: KinBody) -> KinBody:
    """The KinBody whose frames get baked into consts(). Spherical bakes the
    CANONICAL wrist gauge (the runtime canonicalize, moved to build time -- so the
    self-contained artifact has zero runtime preprocessing). Limits stay from the
    original (physical) kb; canonicalization is FK-identical + limit-preserving."""
    if arm_solver == "ikgeo.spherical_two_parallel":
        from ssik._kinbody import canonicalize_spherical_wrist
        from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY

        return canonicalize_spherical_wrist(kb, DEFAULT_TOLERANCE_POLICY)
    return kb


def _f(x: float) -> str:
    """Full-precision C++ double literal (round-trips exactly)."""
    return repr(float(x))


def _mat4(m: np.ndarray) -> str:
    m = np.asarray(m, dtype=np.float64).reshape(4, 4)
    rows = ",\n       ".join(", ".join(_f(v) for v in row) for row in m)
    return f"(Eigen::Matrix4d() <<\n       {rows}).finished()"


def _vec3(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return f"Eigen::Vector3d({_f(v[0])}, {_f(v[1])}, {_f(v[2])})"


def _mat3(m: np.ndarray) -> str:
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    rows = ",\n       ".join(", ".join(_f(v) for v in row) for row in m)
    return f"(Eigen::Matrix3d() <<\n       {rows}).finished()"


def emit(arm: str, out_dir: Path, n_parity: int = 200, seed: int = 0) -> None:
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    joints = kb.joints
    dof = len(joints)
    ns = arm  # e.g. ur5_ik

    # --- Self-contained C++ artifact: baked constants + limits + solve() ----
    solver = load_manifest()[arm].solver
    rendered_solve = _render_solve(solver, kb)
    header_include = rendered_solve[0] if rendered_solve else '#include "ssik_cpp/fk.hpp"'
    title = (
        f"// Self-contained C++ IK artifact for {arm} ({dof}-DOF): baked constants"
        if rendered_solve
        else f"// Native kinematic constants for {arm} ({dof}-DOF), rendered from the KinBody."
    )
    lines: list[str] = [
        "// AUTO-GENERATED by scripts/cpp_emit.py -- do not edit.",
        title,
        "// + solve(); header-only C++, zero Python runtime." if rendered_solve else "",
        "#pragma once",
        "",
        "#include <vector>" if rendered_solve else "",
        header_include,
        "",
        f"namespace ssik::{ns} {{",
        "",
        f"constexpr int DOF = {dof};",
        "",
        "inline JointConsts<DOF> consts() {",
        "  JointConsts<DOF> c;",
    ]
    # Spherical bakes the CANONICAL wrist gauge; three_parallel bakes kb as-is.
    consts_joints = _consts_kinbody(solver, kb).joints
    for i, j in enumerate(consts_joints):
        jt = "Revolute" if j.joint_type == "revolute" else "Prismatic"
        lines.append(f"  c.axis[{i}] = {_vec3(np.asarray(j.axis))};")
        lines.append(f"  c.t_left[{i}] = {_mat4(np.asarray(j.T_left))};")
        lines.append(f"  c.t_right[{i}] = {_mat4(np.asarray(j.T_right))};")
        lines.append(f"  c.type[{i}] = JointType::{jt};")
    lines += ["  return c;", "}"]
    if rendered_solve:
        lines += ["", *_render_limits(joints), "", *rendered_solve[1]]
    lines += ["", f"}}  // namespace ssik::{ns}", ""]
    art = out_dir / f"{arm}.hpp"
    art.write_text("\n".join(lines))

    # --- FK parity fixture (Python reference) -------------------------------
    ranges = [
        (float(j.limits[0]), float(j.limits[1])) if j.limits else (-np.pi, np.pi) for j in joints
    ]
    rng = np.random.default_rng(seed)
    fx: list[str] = [
        "// AUTO-GENERATED by scripts/cpp_emit.py -- do not edit.",
        f"// {n_parity} (q, fk(q)) cases for {arm}, computed by the Python reference.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <vector>",
        "",
        f'#include "{arm}.hpp"',
        "",
        f"namespace ssik::{ns} {{",
        "",
        "struct FkParityCase {",
        "  std::array<double, DOF> q;",
        "  std::array<double, 16> fk;  // row-major 4x4",
        "};",
        "",
        "inline const std::vector<FkParityCase>& fk_parity_cases() {",
        "  static const std::vector<FkParityCase> cases = {",
    ]
    for _ in range(n_parity):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t = poe_forward_kinematics(kb, q)
        qstr = ", ".join(_f(v) for v in q)
        tstr = ", ".join(_f(v) for v in np.asarray(t).reshape(16))
        fx.append(f"    {{{{{qstr}}}, {{{tstr}}}}},")
    fx += ["  };", "  return cases;", "}", "", f"}}  // namespace ssik::{ns}", ""]
    (out_dir / f"{arm}_fk_parity.hpp").write_text("\n".join(fx))

    try:
        art_disp = art.relative_to(_REPO)
    except ValueError:
        art_disp = art  # out_dir outside the repo (e.g. --check temp dir)
    print(f"[cpp_emit] {arm}: DOF={dof} -> {art_disp} + {n_parity} FK parity cases")

    if _is_three_parallel(kb):
        _emit_newton_parity(arm, kb, out_dir, ns, ranges, seed=seed + 1)
    # Self-contained-artifact conformance golden (the gate): the Python artifact
    # solve() output per pose, so a standalone C++ program (which can't call
    # Python) can validate ssik::<arm>::solve(T) against the oracle. Emitted for
    # arms with a self-contained C++ solve().
    if _render_solve(load_manifest()[arm].solver, kb) is not None:
        _emit_solve_parity(arm, out_dir, ns, ranges, seed=seed + 2)


def _emit_solve_parity(
    arm: str,
    out_dir: Path,
    ns: str,
    ranges: list[tuple[float, float]],
    n_cases: int = 120,
    seed: int = 2,
) -> None:
    """Emit the Python artifact solve() output per pose for the standalone C++
    artifact conformance (test_<arm>_artifact.cpp compares ssik::<arm>::solve)."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    rng = np.random.default_rng(seed)
    out: list[str] = [
        "// AUTO-GENERATED by scripts/cpp_emit.py -- do not edit.",
        f"// {n_cases} (target, artifact solve() set) cases for {arm} from the Python oracle.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <vector>",
        "",
        f'#include "{arm}.hpp"',
        "",
        f"namespace ssik::{ns} {{",
        "",
        "struct SolveParityCase {",
        "  std::array<double, 16> target;  // row-major 4x4",
        "  std::vector<std::array<double, DOF>> solutions;",
        "};",
        "",
        "inline const std::vector<SolveParityCase>& solve_parity_cases() {",
        "  static const std::vector<SolveParityCase> cases = {",
    ]
    for _ in range(n_cases):
        seed_q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t = np.asarray(poe_forward_kinematics(kb, seed_q), dtype=np.float64)
        sols = mod.solve(t)
        tstr = ", ".join(_f(v) for v in t.reshape(16))
        sol_strs = ["{" + ", ".join(_f(v) for v in np.asarray(s.q)) + "}" for s in sols]
        out.append(f"    {{{{{tstr}}}, {{{', '.join(sol_strs)}}}}},")
    out += ["  };", "  return cases;", "}", "", f"}}  // namespace ssik::{ns}", ""]
    (out_dir / f"{arm}_solve_parity.hpp").write_text("\n".join(out))
    print(f"[cpp_emit] {arm}: + {n_cases} solve parity cases (artifact gate)")


def _is_three_parallel(kb: KinBody) -> bool:
    """True for a POE arm with three consecutive parallel axes at joints (1,2,3)."""
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.solvers.ikgeo import three_parallel

    try:
        triple = three_parallel.three_consecutive_parallel(  # type: ignore[attr-defined]
            kb.joints, DEFAULT_TOLERANCE_POLICY
        )
    except Exception:
        return False
    return bool(triple == (1, 2, 3))


# Newton-polish parity fixture (#496): perturbed seeds + the Python lm_refine
# outcome, so the C++ lm_refine port is validated directly (convergence + the
# refined joint vector), including the divergence/stall guards on seeds that
# don't converge. fk_atol tight (1e-12) so a match is at machine precision.
_NEWTON_FK_ATOL = 1e-12
_NEWTON_MAX_ITERS = 50


def _emit_newton_parity(
    arm: str,
    kb: KinBody,
    out_dir: Path,
    ns: str,
    ranges: list[tuple[float, float]],
    n_cases: int = 200,
    seed: int = 2,
) -> None:
    from ssik.refinement import kinbody_jacobian, lm_refine

    rng = np.random.default_rng(seed)

    def fk_fn(q: np.ndarray) -> np.ndarray:
        return np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)

    def jac_fn(q: np.ndarray) -> np.ndarray:
        return np.asarray(kinbody_jacobian(kb, q), dtype=np.float64)

    dof = len(kb.joints)
    out: list[str] = [
        "// AUTO-GENERATED by scripts/cpp_emit.py -- do not edit.",
        f"// {n_cases} Newton-polish parity cases for {arm}: a perturbed seed and",
        "// the Python lm_refine outcome (converged flag + refined joint vector).",
        "#pragma once",
        "",
        "#include <array>",
        "#include <vector>",
        "",
        f'#include "{arm}.hpp"',
        "",
        f"namespace ssik::{ns} {{",
        "",
        f"inline constexpr double NEWTON_FK_ATOL = {_f(_NEWTON_FK_ATOL)};",
        f"inline constexpr int NEWTON_MAX_ITERS = {_NEWTON_MAX_ITERS};",
        "",
        "struct NewtonParityCase {",
        "  std::array<double, 16> target;  // row-major 4x4",
        "  std::array<double, DOF> q_seed;",
        "  bool converged;",
        "  std::array<double, DOF> q_refined;  // valid iff converged",
        "};",
        "",
        "inline const std::vector<NewtonParityCase>& newton_parity_cases() {",
        "  static const std::vector<NewtonParityCase> cases = {",
    ]
    for _ in range(n_cases):
        q_star = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t = fk_fn(q_star)
        q_seed = q_star + rng.uniform(-0.15, 0.15, size=dof)
        result = lm_refine(
            q_seed,
            fk_fn,
            t,
            fk_atol=_NEWTON_FK_ATOL,
            max_iters=_NEWTON_MAX_ITERS,
            jacobian_fn=jac_fn,
        )
        tstr = ", ".join(_f(v) for v in t.reshape(16))
        seedstr = ", ".join(_f(v) for v in q_seed)
        if result is None:
            refstr = ", ".join(_f(0.0) for _ in range(dof))
            out.append(f"    {{{{{tstr}}}, {{{seedstr}}}, false, {{{refstr}}}}},")
        else:
            q_ref, _resid, _iters = result
            refstr = ", ".join(_f(v) for v in np.asarray(q_ref))
            out.append(f"    {{{{{tstr}}}, {{{seedstr}}}, true, {{{refstr}}}}},")
    out += ["  };", "  return cases;", "}", "", f"}}  // namespace ssik::{ns}", ""]
    (out_dir / f"{arm}_newton_parity.hpp").write_text("\n".join(out))
    print(f"[cpp_emit] {arm}: + {n_cases} newton parity cases")


def emitted_arms(gen_dir: Path) -> list[str]:
    """Arms with a committed native artifact, derived from ``cpp/gen``.

    The manifest is the generated tree itself: every ``<arm>.hpp`` that is not a
    ``_fk_parity`` / ``_solve_parity`` fixture. Deriving the list this way means
    adding an arm's artifact automatically enrols it in the drift guard -- there
    is no hardcoded list to forget to update.
    """
    arms = []
    for hpp in sorted(gen_dir.glob("*.hpp")):
        stem = hpp.stem
        if stem.endswith(("_fk_parity", "_solve_parity", "_newton_parity")):
            continue
        arms.append(stem)
    return arms


def check_no_drift(gen_dir: Path) -> int:
    """Regenerate each arm's *constants header* and byte-diff against ``gen_dir``.

    Returns 0 when the committed ``<arm>.hpp`` constants headers are
    byte-identical to a fresh emit, non-zero (listing the stale files) otherwise.
    This is the CI gate that makes the Python oracle -> native artifact link
    impossible to silently break (#495): change the KinBody/oracle without
    re-emitting and this turns red.

    Only the constants headers are checked. They are ``repr()`` of the already
    baked KinBody floats -- no computation, so byte-identical across platforms.
    The FK/solve parity fixtures are floating-point *results* (BLAS-backend
    ULP variance, #82), so they are not committed and not byte-comparable; they
    are regenerated fresh before every build and validated by ctest instead.
    """
    arms = emitted_arms(gen_dir)
    if not arms:
        print("[cpp_emit] --check: no emitted arms found in", gen_dir)
        return 0
    stale: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for arm in arms:
            emit(arm, tmp_dir)
            committed = gen_dir / f"{arm}.hpp"
            fresh = tmp_dir / f"{arm}.hpp"
            if not committed.exists():
                stale.append(f"{arm}.hpp (missing from cpp/gen)")
            elif fresh.read_bytes() != committed.read_bytes():
                stale.append(f"{arm}.hpp (differs from a fresh emit)")
    if stale:
        print("[cpp_emit] --check FAILED: native artifacts are stale vs the Python oracle:")
        for s in stale:
            print(f"  - {s}")
        print("Re-run: python scripts/cpp_emit.py --all")
        return 1
    print(f"[cpp_emit] --check OK: {len(arms)} constants header(s) up to date ({', '.join(arms)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", nargs="?", help="prebuilt arm name, e.g. ur5_ik")
    ap.add_argument("--all", action="store_true", help="re-emit every arm already in --out-dir")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify committed artifacts match a fresh emit (CI drift guard); no writes",
    )
    ap.add_argument("--out-dir", type=Path, default=_REPO / "cpp" / "gen")
    args = ap.parse_args()

    if args.check:
        return check_no_drift(args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.all:
        for arm in emitted_arms(args.out_dir):
            emit(arm, args.out_dir)
        return 0
    if not args.arm:
        ap.error("provide an arm name, or --all / --check")
    emit(args.arm, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
