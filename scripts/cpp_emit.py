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
import re
import tempfile
from pathlib import Path
from typing import Any, cast

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
def _render_solve(solver: str, kb: KinBody, arm: str = "") -> tuple[str, list[str]] | None:
    if solver == "seven_r.srs":
        return _render_srs_solve(kb)
    if solver == "ikgeo.general_6r":
        return _render_general_6r_solve(kb)
    if solver == "jointlock.seven_r":
        return _render_jointlock_solve(arm, kb)
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
    """Bake the SrsConsts for a concurrent-axis SRS 7R arm, or None when the arm
    is not SRS-class. Delegates to :func:`ssik._native.srs_native_geometry` -- the
    single source shared with the runtime ``native=True`` path -- so the emitted
    self-contained artifact and the pip native backend always cover exactly the
    same arms and route them the same way (``general_path``)."""
    from ssik._native import srs_native_geometry

    return srs_native_geometry(kb)


def _hp_bake(kb: KinBody) -> dict[str, Any]:
    """Bake the HpConsts fields for a 6R KinBody. Delegates to
    :func:`ssik._native.hp_native_geometry` -- the single source shared with the
    native HP artifact solve -- so the emitted jointlock-HP artifact and the
    runtime path always compute identical baked constants."""
    from ssik._native import hp_native_geometry

    return hp_native_geometry(kb)


def _render_hp_consts(bake: dict[str, Any], suffix: str) -> list[str]:
    """Emit `inline HpConsts hp_consts_<suffix>() {...}` from a _hp_bake dict.
    The DH arrays are 1-indexed in HpConsts (a/l[1..5], d[2..5]); the bake yields
    dh_a/dh_l length-5 and dh_d length-4, placed with the unused low slots left 0."""
    a, ln, dd = bake["dh_a"], bake["dh_l"], bake["dh_d"]
    return [
        f"inline HpConsts hp_consts{suffix}() {{",
        "  HpConsts h;",
        f"  h.t_u = {_mat4x8_pair(bake['t_u'])};",
        f"  h.t_w_pre = {_mat4x8_pair(bake['t_w_pre'])};",
        f"  h.drop_idx = {int(bake['drop_idx'])};",
        f"  h.a = {{0.0, {', '.join(_f(v) for v in a)}}};",
        f"  h.l = {{0.0, {', '.join(_f(v) for v in ln)}}};",
        f"  h.d = {{0.0, 0.0, {', '.join(_f(v) for v in dd)}}};",
        "  h.parametric_var = 0;",
        f"  h.right_parametric_var = {int(bake['right_parametric_var'])};",
        f"  h.t_pre_inv = {_mat4(bake['t_pre_inv'])};",
        f"  h.t_post_inv = {_mat4(bake['t_post_inv'])};",
        f"  h.t_z_neg_d1 = {_mat4(bake['t_z_neg_d1'])};",
        f"  h.t_joint6_offset_inv = {_mat4(bake['t_joint6_offset_inv'])};",
        f"  h.theta_offset = {{{', '.join(_f(v) for v in bake['theta_offset'])}}};",
        "  return h;",
        "}",
    ]


def _render_joint_consts6(joints: list[Any], suffix: str) -> list[str]:
    """Emit `inline JointConsts<6> hp_sub_<suffix>() {...}` (the locked 6R sub-chain
    POE frames hp_core needs for its FK-verify + lm_refine)."""
    out = [f"inline JointConsts<6> hp_sub{suffix}() {{", "  JointConsts<6> c;"]
    for i, j in enumerate(joints):
        jt = "Revolute" if j.joint_type == "revolute" else "Prismatic"
        out.append(f"  c.axis[{i}] = {_vec3(np.asarray(j.axis))};")
        out.append(f"  c.t_left[{i}] = {_mat4(np.asarray(j.T_left))};")
        out.append(f"  c.t_right[{i}] = {_mat4(np.asarray(j.T_right))};")
        out.append(f"  c.type[{i}] = JointType::{jt};")
    out += ["  return c;", "}"]
    return out


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
            f"  s.general_path = {'true' if s['general_path'] else 'false'};",
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


def _snap_dh(vals: np.ndarray, canonicals: tuple[float, ...], atol: float = 1e-6) -> np.ndarray:
    """Snap DH values within ``atol`` of a canonical value to exactly that value.

    A degenerate locked sub-chain (near-parallel axes at some lock samples) has a
    twist that is geometrically exactly 0 or +/-pi and a link length exactly 0,
    but poe_to_dh returns them as canonical +/- ~1e-8 numeric error whose low bits
    differ across BLAS backends. Left raw, that error propagates into the RR
    derivation as ~1e-8..1e-12 noise coefficients that straddle the CSE
    zero-threshold across platforms -> non-deterministic header structure (#536).
    Snapping the SOURCE to exact collapses those products (0.0*x is exactly 0;
    sin(exact pi) is ~1e-16, well below the threshold), so the emit is
    backend-deterministic. atol=1e-6 >> the ~1e-8 numeric error but << any genuine
    DH spacing, so a real near-canonical design value is never touched. Only the
    degenerate samples' spurious solutions are affected, and those are dropped by
    the 7R re-verify regardless."""
    out = np.array(vals, dtype=float)
    for i in range(out.size):
        for c in canonicals:
            if abs(out[i] - c) < atol:
                out[i] = c
                break
    return out


def _render_rr_unit(kb: KinBody, suffix: str = "", zero_threshold: float = 0.0) -> list[str]:
    """Render ``rr_consts<suffix>()`` + ``rr_coeffs<suffix>(...)`` for one 6R
    KinBody: the baked DH bridge + AE-3 leftvar (RrConsts) + the CSE'd coefficient
    function. Shared by the general_6r artifact and the jointlock per-sample fan-
    out (each locked 6R sub-chain is its own RR unit).

    ``zero_threshold`` is passed to render_rr_coeffs to drop numerical-noise
    coefficients (jointlock degenerate sub-chains, #536); general_6r leaves it 0.
    When set, DH values near a canonical (0, +/-pi/2, +/-pi) are also snapped to
    exact at the source, which is what makes the noise land firmly below the
    threshold on every backend rather than straddle it."""
    from _rr_emit import render_rr_coeffs

    from ssik.kinematics.poe_to_dh import poe_to_dh
    from ssik.solvers.ikgeo._raghavan_roth import _cached_best_leftvar, _derive_pq_for_arm

    dh = poe_to_dh(kb)
    alpha, a, d = dh.to_dh_tuple()
    theta_offset = np.array(dh.theta_offset, dtype=float)
    if zero_threshold > 0.0:
        angles = (0.0, np.pi / 2, -np.pi / 2, np.pi, -np.pi)
        alpha = _snap_dh(alpha, angles)
        theta_offset = _snap_dh(theta_offset, angles)
        a = _snap_dh(a, (0.0,))
        d = _snap_dh(d, (0.0,))
    at, bt, dt = tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist())
    lin = int(_cached_best_leftvar(at, bt, dt))
    # Derive directly (not _cached_derivation): importing a jointlock arm primes
    # the derivation cache with AOT-lambdified entries that lack the `_sym_*`
    # symbolic matrices render_rr_coeffs needs. The fresh derivation always
    # carries them (and uses the same raw DH, so the emitted literals are stable).
    *_fns, meta = _derive_pq_for_arm(at, bt, dt, linearity_joint=lin)
    lb0, lb1 = cast("tuple[int, int]", meta["left_bilinear"])
    rb0, rb1 = cast("tuple[int, int]", meta["right_bilinear"])
    drop = cast(int, meta["drop_joint"])
    t_pre_inv = np.linalg.inv(dh.t_pre)
    t_post_inv = np.linalg.inv(dh.t_post)
    return [
        f"inline RrConsts rr_consts{suffix}() {{",
        "  RrConsts r;",
        f"  r.alpha = {{{', '.join(_f(v) for v in alpha)}}};",
        f"  r.a = {{{', '.join(_f(v) for v in a)}}};",
        f"  r.d = {{{', '.join(_f(v) for v in d)}}};",
        f"  r.theta_offset = {{{', '.join(_f(v) for v in theta_offset)}}};",
        f"  r.t_pre_inv = {_mat4(t_pre_inv)};",
        f"  r.t_post_inv = {_mat4(t_post_inv)};",
        f"  r.linearity_joint = {lin};",
        f"  r.left_bilinear = {{{lb0}, {lb1}}};",
        f"  r.right_bilinear = {{{rb0}, {rb1}}};",
        f"  r.drop_joint = {drop};",
        "  return r;",
        "}",
        "",
        # No CSE on the jointlock path (zero_threshold > 0): sympy.cse shares
        # subexpressions by exact float equality, and a degenerate sub-chain's
        # backend-variant coefficient low bits break a share on one platform but
        # not another -> non-portable temp set (#536). Without CSE the emitted
        # structure is exactly the (threshold-stabilised) non-zero-coefficient set,
        # which IS backend-deterministic; and after the threshold the polynomials
        # are short enough that no-CSE is also smaller than the CSE'd form.
        *render_rr_coeffs(
            meta,
            name=f"rr_coeffs{suffix}",
            zero_threshold=zero_threshold,
            use_cse=(zero_threshold == 0.0),
        ),
    ]


def _render_general_6r_solve(kb: KinBody) -> tuple[str, list[str]] | None:
    """Self-contained general-6R (Raghavan-Roth) solve(). None for a non-6R chain."""
    if len(kb.joints) != 6 or any(j.joint_type != "revolute" for j in kb.joints):
        return None
    body = [
        "// Baked RR geometry (poe_to_dh DH bridge + AE-3 leftvar), zero runtime Python.",
        *_render_rr_unit(kb),
        "",
        "// Self-contained IK solve -- header-only C++, no Python runtime.",
        "inline std::vector<Solution<DOF>> solve(",
        "    const Pose& T, const ArtifactParams<DOF>& p = {}) {",
        "  return general_6r_artifact_solve(consts(), rr_consts(), rr_coeffs, limits(), T, p);",
        "}",
    ]
    return ('#include "ssik_cpp/solvers/general_6r.hpp"', body)


def _jointlock_rr_complete(arm: str, kb: KinBody, n_probe: int = 150) -> bool:
    """ONBOARDING TOOL (not called at emit time): decide whether a new jointlock
    7R arm's RR-only lock-sweep covers it, to set its membership in
    ``_HP_DEFERRED_JOINTLOCK_ARMS``. It runs backend-sensitive oracle solves, so
    its result can flip across BLAS backends at a borderline pose -- fine for a
    one-time human-run onboarding decision, but it must NOT drive the per-emit
    header structure (that has to be byte-deterministic for the --check drift
    guard, #536). Run this when adding a jointlock arm; bake the answer.

    True iff a native RR-only lock-sweep (no HP kernel, no rescue) can cover the
    arm, tested two ways over n_probe reachable poses (#535):

    1. HP-fire: if the Python oracle ever dispatches to the HP Study-quaternion
       kernel, an RR-only artifact CANNOT reproduce that pose by definition (HP
       found a root cached-RR could not). This is the sensitive signal -- it fires
       whenever ANY of the 16 lock samples needs HP (~16% of poses for kassow), so
       n_probe=150 catches it with overwhelming probability and returns early.
    2. Direct sweep-vs-oracle: even without HP, the native RR sweep (different
       numerics than Python's cached-RR) must not miss any oracle solution.

    The old HP-fire-only probe used just 20 poses and let kassow through; the rescue
    then MASKED the gap at the gate. Deferred (RR-incomplete) arms stay
    constants-only until the HP kernel lands (#491). The full gate is the
    deterministic backstop if this ever wrongly returns True."""
    from unittest.mock import patch

    import ssik.solvers.husty_pfurner.general_6r as _hp
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY as _pol
    from ssik.refinement import dedup_by_wrap_close
    from ssik.solvers.ikgeo import general_6r as _g6
    from ssik.solvers.jointlock.seven_r import _lock_joint, choose_lock_joint

    lock = int(choose_lock_joint(kb))
    lo0, hi0 = kb.joints[lock].limits or (-np.pi, np.pi)
    samples = np.linspace(lo0, hi0, 16, endpoint=False)
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    fk_atol, dedup_atol = _pol.subproblem_numerical, _pol.subproblem_dedup
    rng = np.random.default_rng(0)
    ranges = [j.limits if j.limits else (-np.pi, np.pi) for j in kb.joints]
    # jointlock imports this module object as hp_general_6r and looks up .solve at
    # call time, so one patch counts every HP-kernel dispatch inside mod.solve.
    fired = {"n": 0}
    orig = _hp.solve

    def counting(*a: Any, **k: Any) -> Any:
        fired["n"] += 1
        return orig(*a, **k)

    with patch.object(_hp, "solve", counting):
        for _ in range(n_probe):
            q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
            t = np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)
            before = fired["n"]
            oracle = list(mod.solve(t, respect_limits=False))
            if fired["n"] > before:
                return False  # oracle needed the HP kernel -> RR-only artifact can't cover it
            # RR-only sweep, mirroring the native runtime: lock -> RR solve -> pad
            # -> 7R re-verify against the actual target -> dedup. _g6 never routes
            # to HP, so this stays a pure RR sweep.
            cands = []
            for q_lock in samples:
                sub = _lock_joint(kb, lock, float(q_lock))
                sols, _ = _g6.solve(sub, t)
                for s in sols:
                    q7 = np.insert(s.q, lock, q_lock)
                    if np.linalg.norm(poe_forward_kinematics(kb, q7) - t) <= fk_atol:
                        cands.append(_Sol(q7))
            sweep = dedup_by_wrap_close(cands, dedup_atol)
            for o in oracle:
                if not any(_wrap_close(o.q, s.q, dedup_atol) for s in sweep):
                    return False  # oracle solution the RR-only sweep misses
    return True


class _Sol:
    """Minimal Solution stand-in for dedup_by_wrap_close in the RR-completeness probe."""

    def __init__(self, q: np.ndarray) -> None:
        self.q = q
        self.fk_residual = 0.0


def _wrap_close(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    return bool(np.max(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)) < atol)


# Jointlock 7R arms whose locked sub-chains genuinely need the HP Study-quaternion
# kernel (their RR-only sweep is incomplete), so they ship constants-only until
# the HP native kernel lands (#491). This is a COMMITTED decision, not a per-emit
# recompute: `_jointlock_rr_complete` runs backend-sensitive oracle solves (SVD /
# eig near singularities), so its True/False can FLIP across BLAS backends
# (Accelerate vs OpenBLAS) at a borderline pose -- which would make the emitted
# header structure platform-dependent and break the --check drift guard (#536).
# The direct check remains the onboarding tool that DECIDES membership here (run
# it when adding a jointlock arm); baking the result keeps every re-emit
# byte-deterministic. Correctness of the set is backstopped by the artifact gate:
# an RR-covered arm listed here would lose coverage (caught by review), and an
# HP-needing arm NOT listed would gate-fail loudly (its sweep misses the oracle).
_HP_DEFERRED_JOINTLOCK_ARMS: set[str] = set()

# Jointlock 7R arms whose locked sub-chains are HP-covered (symmetric-DH, where the
# RR-only sweep is incomplete): the emitter bakes an HP unit per sample and the
# runtime sweeps via jointlock_hp_artifact_solve (Study-quaternion kernel). #491.
_HP_JOINTLOCK_ARMS = {"kassow_kr810_ik"}


def _render_jointlock_solve(arm: str, kb: KinBody) -> tuple[str, list[str]] | None:
    """Self-contained jointlock.seven_r solve(): bake the lock joint + 16-sample
    schedule + each locked 6R sub-chain's inner unit, and compose the sweep. RR
    units + jointlock_artifact_solve for RR-covered arms; HP units +
    jointlock_hp_artifact_solve for HP-covered arms (kassow, #491)."""
    from ssik.solvers.jointlock.seven_r import _lock_joint, choose_lock_joint

    if len(kb.joints) != 7 or arm in _HP_DEFERRED_JOINTLOCK_ARMS:
        return None
    lock = int(choose_lock_joint(kb))
    j = kb.joints[lock]
    lo, hi = j.limits if j.limits else (-np.pi, np.pi)
    samples = np.linspace(lo, hi, 16, endpoint=False)
    inner = "HP" if arm in _HP_JOINTLOCK_ARMS else "RR"

    body = [
        f"// Jointlock 7R: lock joint {lock}, sweep 16 samples, {inner}-solve each 6R sub-chain.",
        "inline JointlockConsts<16> jl_consts() {",
        "  JointlockConsts<16> j;",
        f"  j.lock_idx = {lock};",
        f"  j.q_lock = {{{', '.join(_f(v) for v in samples)}}};",
        "  return j;",
        "}",
        "",
    ]
    if arm in _HP_JOINTLOCK_ARMS:
        return _render_jointlock_hp(kb, lock, samples, body)
    for i, q_lock in enumerate(samples):
        body += ["", f"// --- lock sample {i} (q_lock = {_f(q_lock)}) ---"]
        # zero_threshold: degenerate lock samples (near-parallel axes) push
        # poe_to_dh coefficient products down to ~1e-18 noise whose low bits are
        # BLAS-backend-sensitive; drop them so the emitted header is byte-
        # deterministic across platforms (#536). 1e-12 relative is far below any
        # genuine RR coefficient (DH products, O(1e-3)..O(1)).
        body += _render_rr_unit(_lock_joint(kb, lock, float(q_lock)), f"_{i}", zero_threshold=1e-12)
    body += [
        "",
        "inline std::array<RrConsts, 16> jl_rr() {",
        "  return {" + ", ".join(f"rr_consts_{i}()" for i in range(16)) + "};",
        "}",
        "inline std::array<RrCoeffFn, 16> jl_coeffs() {",
        "  return {" + ", ".join(f"rr_coeffs_{i}" for i in range(16)) + "};",
        "}",
        "",
        "inline std::vector<Solution<DOF>> solve(",
        "    const Pose& T, const ArtifactParams<DOF>& p = {}) {",
        "  return jointlock_artifact_solve<16>(consts(), jl_consts(), jl_rr(), jl_coeffs(),",
        "                                      limits(), T, p);",
        "}",
    ]
    return ('#include "ssik_cpp/solvers/jointlock_seven_r.hpp"', body)


def _render_jointlock_hp(
    kb: KinBody, lock: int, samples: np.ndarray, body: list[str]
) -> tuple[str, list[str]]:
    """HP-inner jointlock solve() (kassow, #491): per lock sample, bake the locked
    6R sub-chain's HpConsts (Study-quaternion kernel) + its JointConsts<6> (hp_core
    needs the sub-chain POE FK for its FK-verify + lm_refine), then compose the
    sweep via jointlock_hp_artifact_solve."""
    from ssik.solvers.jointlock.seven_r import _lock_joint

    for i, q_lock in enumerate(samples):
        sub = _lock_joint(kb, lock, float(q_lock))
        body += ["", f"// --- lock sample {i} (q_lock = {_f(q_lock)}) ---"]
        body += _render_hp_consts(_hp_bake(sub), f"_{i}")
        body += _render_joint_consts6(list(sub.joints), f"_{i}")
    body += [
        "",
        "inline std::array<HpConsts, 16> jl_hp() {",
        "  return {" + ", ".join(f"hp_consts_{i}()" for i in range(16)) + "};",
        "}",
        "inline std::array<JointConsts<6>, 16> jl_hp_sub() {",
        "  return {" + ", ".join(f"hp_sub_{i}()" for i in range(16)) + "};",
        "}",
        "",
        "inline std::vector<Solution<DOF>> solve(",
        "    const Pose& T, const ArtifactParams<DOF>& p = {}) {",
        "  return jointlock_hp_artifact_solve<16>(consts(), jl_consts(), jl_hp(), jl_hp_sub(),",
        "                                         limits(), T, p);",
        "}",
    ]
    return ('#include "ssik_cpp/solvers/jointlock_seven_r.hpp"', body)


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


def _mat4x8(m: np.ndarray) -> str:
    m = np.asarray(m, dtype=np.float64).reshape(4, 8)
    rows = ",\n         ".join(", ".join(_f(v) for v in row) for row in m)
    return f"(Eigen::Matrix<double, 4, 8>() <<\n         {rows}).finished()"


def _mat4x8_pair(tensor: np.ndarray) -> str:
    """Emit a (4,8,2) HP tensor as std::array<Matrix<4,8>,2>{slice0, slice1}."""
    t = np.asarray(tensor, dtype=np.float64).reshape(4, 8, 2)
    return "{\n      " + _mat4x8(t[:, :, 0]) + ",\n      " + _mat4x8(t[:, :, 1]) + "}"


def emit(arm: str, out_dir: Path, n_parity: int = 200, seed: int = 0) -> None:
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    joints = kb.joints
    dof = len(joints)
    ns = arm  # e.g. ur5_ik

    # --- Self-contained C++ artifact: baked constants + limits + solve() ----
    solver = load_manifest()[arm].solver
    rendered_solve = _render_solve(solver, kb, arm)
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
    if _render_solve(load_manifest()[arm].solver, kb, arm) is not None:
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
    artifact conformance (test_<arm>_artifact.cpp compares ssik::<arm>::solve).

    HP jointlock arms (kassow) generate the golden with allow_rescue=False: the
    T-perturbation rescue is STOCHASTIC (different RNG in C++ vs Python recovers
    different, both-sound subsets of the between-sample redundant manifold), so an
    exact set-match gate on rescue output is not a stable cross-platform invariant.
    The golden is thus the DETERMINISTIC lock-sweep; the gate checks native covers
    it (soundness + oracle-subset), and native's own rescue solutions are sound
    extensions beyond it (#491, relative-completeness model)."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    no_rescue = arm in _HP_JOINTLOCK_ARMS
    solve = (lambda t: mod.solve(t, allow_rescue=False)) if no_rescue else mod.solve
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
    # Only WELL-CONDITIONED poses go in the golden: the gate demands exact
    # set-match, which is a meaningful cross-platform invariant only OFF the
    # measure-zero near-singular set. At a branch-boundary pose a pair of
    # solutions is FP-unstable, so C++ and Python can return different (both
    # sound) counts on different platforms (e.g. gp8 case 119: 4 on Linux vs 6 on
    # macOS). Reject any pose whose solution count changes under a tiny random
    # perturbation -- that flags proximity to a boundary far wider (1e-6) than the
    # ~1e-15 C++/Python FP gap, so the survivors are cross-platform reproducible.
    kept = 0
    attempts = 0
    while kept < n_cases and attempts < n_cases * 40:
        attempts += 1
        seed_q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t = np.asarray(poe_forward_kinematics(kb, seed_q), dtype=np.float64)
        sols = solve(t)
        stable = True
        for _ in range(2):
            dq = rng.uniform(-1e-6, 1e-6, len(seed_q))
            tp = np.asarray(poe_forward_kinematics(kb, seed_q + dq), dtype=np.float64)
            if len(solve(tp)) != len(sols):
                stable = False
                break
        if not stable:
            continue
        kept += 1
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
        if stem.endswith(
            ("_fk_parity", "_solve_parity", "_newton_parity", "_rr_parity", "_rr_coeffs")
        ):
            continue
        if stem == "artifact_gate":  # the generated data-driven gate, not an arm
            continue
        arms.append(stem)
    return arms


# Header marker of a self-contained arm (one with an emitted solve()); the gate
# iterates exactly these.
_SOLVE_MARKER = "std::vector<Solution<DOF>> solve("


def _arm_fk_ceiling(arm: str) -> float:
    """The arm's own FK-closure tolerance (its solver's fk_atol), used as the
    gate's worst-FK ceiling. A solution within its solver's tolerance is valid;
    a global 1e-7 is wrong for looser families (RR at 1e-5, where force-refined
    near-double-root solutions settle ~1e-6)."""
    from ssik.core.solver_registry import SOLVERS
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY

    spec = SOLVERS[load_manifest()[arm].solver]
    ns = {"policy": DEFAULT_TOLERANCE_POLICY}
    return float(eval(spec.fk_atol_expr, {"__builtins__": {}}, ns))


# Per-arm artifact-gate allowance for poses where the native solve misses an
# oracle solution (default 0 = must cover the whole oracle). Set >0 ONLY for a
# documented, bounded completeness gap. kassow (HP jointlock): its monic-companion
# eigensolve recovers fewer roots than LAPACK dggev at the degenerate lock samples
# (axes aligned at multiples of pi/2). Every native solution is still sound
# (FK-closes) and the gap is a handful of the 120 golden poses; tracked in #544.
_ARM_MAX_INCOMPLETE = {"kassow_kr810_ik": 4}


def _arm_max_incomplete(arm: str) -> int:
    return _ARM_MAX_INCOMPLETE.get(arm, 0)


def emit_artifact_gate(gen_dir: Path) -> None:
    """Generate the single data-driven artifact gate over EVERY self-contained
    arm (one with an emitted ``solve()``). Adding an arm needs no test/CMake edit
    -- re-emit and the arm auto-enrols, exactly like the ``--check`` drift guard.
    Git-ignored + regenerated before every build (it includes the per-arm solve
    goldens, which are themselves regenerated results)."""
    arms = [a for a in emitted_arms(gen_dir) if _SOLVE_MARKER in (gen_dir / f"{a}.hpp").read_text()]
    lines = [
        "// AUTO-GENERATED by scripts/cpp_emit.py -- do not edit.",
        "// Data-driven self-contained-artifact gate (THE GATE): every arm with an",
        "// emitted solve() is validated against its Python oracle golden, with zero",
        "// pybind/Python. Enrolment is automatic from cpp/gen -- no per-arm test file.",
        "#pragma once",
        "",
        '#include "artifact_conformance.hpp"',
    ]
    for a in arms:
        lines += [f'#include "{a}.hpp"', f'#include "{a}_solve_parity.hpp"']
    lines += ["", "namespace ssik::artifact_test {", "", "inline int run_all() {", "  int rc = 0;"]
    for a in arms:
        ceil = _f(_arm_fk_ceiling(a))
        allow = _arm_max_incomplete(a)
        lines.append(
            f'  rc |= run<{a}::DOF>("{a}", {a}::consts(), {a}::solve_parity_cases(),\n'
            f"                     [](const Pose& T) {{ return {a}::solve(T); }}, {ceil}, {allow});"
        )
    lines += ["  return rc;", "}", "", "}  // namespace ssik::artifact_test", ""]
    (gen_dir / "artifact_gate.hpp").write_text("\n".join(lines))
    print(f"[cpp_emit] artifact gate: {len(arms)} self-contained arm(s) ({', '.join(arms)})")


# A C++ double literal: has a decimal point and/or an exponent. Integers (array
# indices, DOF, sizes) have neither, so they stay in the structural template and
# are compared exactly. Digit runs inside identifiers (e.g. "iiwa14_ik") lack a
# '.'/'e' and are likewise not matched.
_FLOAT_LITERAL_RE = re.compile(
    r"[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)"
)
# Cross-platform tolerance for computed baked geometry (#517/#82). The SRS
# artifact bakes constants from FK + a home-gauge matmul, and the jointlock RR
# units bake poe_to_dh of 16 locked sub-chains -- some at a degenerate lock
# sample where an angle is exactly +/-pi and its low bits (~1e-9) differ between
# BLAS backends (Accelerate vs OpenBLAS, #536). A real un-re-emitted oracle
# change moves geometry by >> this (mm / rad, i.e. >> 1e-6), so the guard stays
# sharp: it only tolerates the last few bits of a computed value, never a change.
_DRIFT_ATOL = 1e-11
_DRIFT_RTOL = 1e-8


def _constants_match(committed: str, fresh: str) -> tuple[bool, float, str]:
    """Whether two constants headers are identical up to cross-platform ULP,
    plus a one-line diagnostic of the worst float delta.

    Byte comparison is too strict for headers that bake *computed* geometry (the
    SRS ``ee_offset_local`` etc. go through FK + an ``R_home.T`` matmul whose low
    bits are BLAS-backend-sensitive, #517/#82; the jointlock RR units bake
    poe_to_dh of 16 locked sub-chains, #536). Instead: the non-float structure
    (identifiers, integers, layout) must match exactly, and every float literal
    must match within ``_DRIFT_ATOL``/``_DRIFT_RTOL`` -- orders of magnitude below
    any real geometry change, so a forgotten re-emit still turns the guard red.
    Pure-``repr`` constants (ur5, irb6700) match exactly and are unaffected.

    :returns: ``(ok, worst_ratio, detail)`` where ``worst_ratio`` is the largest
        delta-over-budget (>1 means a real mismatch) and ``detail`` names it (so a
        near-miss or a pass-with-margin is visible in CI logs).
    """
    if _FLOAT_LITERAL_RE.sub("~", committed) != _FLOAT_LITERAL_RE.sub("~", fresh):
        cl, fl = committed.count("\n"), fresh.count("\n")
        first = next(
            (
                i + 1
                for i, (a, b) in enumerate(
                    zip(committed.splitlines(), fresh.splitlines(), strict=False)
                )
                if _FLOAT_LITERAL_RE.sub("~", a) != _FLOAT_LITERAL_RE.sub("~", b)
            ),
            min(cl, fl) + 1,
        )
        return (
            False,
            float("inf"),
            f"structural mismatch (non-float text differs): committed {cl} lines vs fresh {fl} "
            f"lines, first structural diff at line {first}",
        )
    ca = [float(x) for x in _FLOAT_LITERAL_RE.findall(committed)]
    fa = [float(x) for x in _FLOAT_LITERAL_RE.findall(fresh)]
    if len(ca) != len(fa):
        return False, float("inf"), f"float count differs ({len(ca)} vs {len(fa)})"
    worst_ratio, worst_detail = 0.0, "identical"
    for c, f in zip(ca, fa, strict=True):
        delta = abs(c - f)
        budget = _DRIFT_ATOL + _DRIFT_RTOL * abs(f)
        ratio = delta / budget if budget > 0 else 0.0
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_detail = f"|Δ|={delta:.2e} at value {f:.12g} (budget {budget:.2e}, {ratio:.2f}x)"
    return worst_ratio <= 1.0, worst_ratio, worst_detail


def check_no_drift(gen_dir: Path) -> int:
    """Regenerate each arm's *constants header* and diff against ``gen_dir``.

    Returns 0 when the committed ``<arm>.hpp`` constants headers match a fresh
    emit (structure exactly, float literals within cross-platform ULP), non-zero
    (listing the stale files) otherwise. This is the CI gate that makes the
    Python oracle -> native artifact link impossible to silently break (#495):
    change the KinBody/oracle without re-emitting and this turns red.

    Comparison is structural-exact + numeric-tolerant, not byte-exact: the SRS
    artifact bakes *computed* geometry (FK + a home-gauge matmul) whose low bits
    carry BLAS-backend variance (#517/#82), so a byte diff would false-positive
    across platforms. The tolerance (see :func:`_constants_match`) is far below
    any real geometry change. The FK/solve parity fixtures are floating-point
    *results*, not committed, and validated by ctest instead.
    """
    arms = emitted_arms(gen_dir)
    if not arms:
        print("[cpp_emit] --check: no emitted arms found in", gen_dir)
        return 0
    stale: list[str] = []
    worst_overall = (0.0, "", "")  # (ratio, arm, detail) across all arms, for CI visibility
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for arm in arms:
            emit(arm, tmp_dir)
            committed = gen_dir / f"{arm}.hpp"
            fresh = tmp_dir / f"{arm}.hpp"
            if not committed.exists():
                stale.append(f"{arm}.hpp (missing from cpp/gen)")
                continue
            ok, ratio, detail = _constants_match(committed.read_text(), fresh.read_text())
            if ratio > worst_overall[0]:
                worst_overall = (ratio, arm, detail)
            if not ok:
                stale.append(f"{arm}.hpp (differs from a fresh emit): {detail}")
    if worst_overall[1]:
        print(f"[cpp_emit] --check worst float delta: {worst_overall[1]} -> {worst_overall[2]}")
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
        emit_artifact_gate(args.out_dir)
        return 0
    if not args.arm:
        ap.error("provide an arm name, or --all / --check")
    emit(args.arm, args.out_dir)
    emit_artifact_gate(args.out_dir)  # refresh the gate to include this arm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
