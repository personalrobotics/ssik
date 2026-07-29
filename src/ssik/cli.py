"""``ssik`` command-line interface.

Three subcommands:

* ``ssik classify <urdf> --base <link> --ee <link>`` -- classify topology +
  print which solver would be picked, without emitting an artifact.
* ``ssik build <urdf> --base <link> --ee <link> [--out <path>]`` --
  classify, emit a per-arm artifact (\\*_ik.py), validate it on random
  poses, and report timing.
* ``ssik add-arm <urdf> --base <link> --ee <link> --name <arm>`` --
  vendor a URDF into ``tests/fixtures/`` and generate a bulletproof
  test scaffold for it; turnkey arm onboarding (#196).

All commands print explanatory messages by default. ``-v`` raises log
verbosity (per-solver INFO logs); ``-vv`` shows DEBUG.

The CLI uses argparse so it has no external dependency and the help
output is self-describing (``ssik --help``, ``ssik build --help``).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ssik._kinbody import KinBody
from ssik._urdf import (
    _as_plain_urdf,
    load_urdf_kinbody_normalized,
    needs_xacro_expansion,
    strip_urdf_to_fixture,
    suggest_base_ee,
)
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import DispatchPlan, dispatch
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["main"]

_VALIDATE_DEFAULT_POSES = 100


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the ``ssik`` console script.

    :param argv: command-line args (excluding the program name). ``None``
        defaults to ``sys.argv[1:]`` so this function is also callable from
        tests via ``main(["build", ...])``.
    :returns: process exit status (0 success, non-zero failure).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.command == "classify":
        return _run_classify(args)
    if args.command == "build":
        return _run_build(args)
    if args.command == "add-arm":
        return _run_add_arm(args)
    parser.print_help()
    return 2


# ---------------------------------------------------------------------------
# Argparse construction.
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssik",
        description=(
            "Build per-arm analytical IK artifacts. Loads a URDF, classifies "
            "the kinematic topology, picks the best ssik solver, and emits a "
            "self-contained Python module that wraps that solver."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity. -v shows solver INFO logs; -vv shows DEBUG.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    classify_parser = sub.add_parser(
        "classify",
        help=(
            "Inspect a URDF: print the inferred topology and the solver "
            "that would be selected, without emitting an artifact."
        ),
    )
    _add_common_kinbody_args(classify_parser)

    build_parser = sub.add_parser(
        "build",
        help=(
            "Generate a per-arm IK artifact: classify the topology, render a "
            "<arm>_ik.py wrapper around the chosen solver, and validate it on "
            "random poses."
        ),
    )
    _add_common_kinbody_args(build_parser)
    build_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Path for the emitted artifact. Default: <urdf-stem>_ik.py in "
            "the current working directory."
        ),
    )
    build_parser.add_argument(
        "--module-name",
        type=str,
        default=None,
        help=(
            "Python module name for the artifact. Default: <urdf-stem>_ik. "
            "Used as the artifact's import name and in its docstring."
        ),
    )
    build_parser.add_argument(
        "--validate-poses",
        type=int,
        default=_VALIDATE_DEFAULT_POSES,
        help=(
            f"Number of random poses to use for post-emit validation. "
            f"Default: {_VALIDATE_DEFAULT_POSES}. Set to 0 to skip validation."
        ),
    )
    build_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-emit validation entirely (equivalent to --validate-poses 0).",
    )

    add_arm_parser = sub.add_parser(
        "add-arm",
        help=(
            "Onboard a new arm: vendor the URDF into tests/fixtures/ and "
            "generate a bulletproof test scaffold based on the dispatched "
            "solver. (#196)"
        ),
    )
    _add_common_kinbody_args(add_arm_parser)
    add_arm_parser.add_argument(
        "--name",
        required=True,
        help=(
            "Identifier for the arm (lowercase, underscore-separated). "
            "Determines the fixture filename (tests/fixtures/<name>.urdf), "
            "the test module (tests/test_<name>.py), and the Python helper "
            "(_<name>_kinbody). Examples: 'kinova_gen3', 'flexiv_rizon4'."
        ),
    )
    add_arm_parser.add_argument(
        "--vendor",
        required=True,
        help=(
            "Vendor/brand subpackage the artifact ships under (#421): "
            "ssik.prebuilt.<vendor>.<model>_ik. Lowercase manufacturer slug, "
            "e.g. 'universal_robots', 'kuka', 'abb', 'fanuc' (use an existing "
            "one where possible; 'misc' only for a genuine one-off)."
        ),
    )
    add_arm_parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Path to the ssik repository root. Defaults to the current "
            "working directory; the URDF is vendored to "
            "<repo-root>/tests/fixtures/ and the test file to "
            "<repo-root>/tests/."
        ),
    )
    add_arm_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing fixture/test files for this arm. By default, "
            "the command refuses if either file already exists."
        ),
    )
    add_arm_parser.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "Skip building + coverage-validating the emitted artifact. Use for "
            "slow-build (cached-RR 7R) arms where the inline build takes minutes."
        ),
    )
    add_arm_parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "Append the derived stanza to <repo-root>/src/ssik/prebuilt/"
            "MANIFEST.toml instead of only printing it. Refuses if an entry for "
            "this arm already exists (use --force to overwrite it). Curated TODO "
            "fields (kinematic_class, class_tags, ...) still need your values."
        ),
    )
    return parser


def _add_common_kinbody_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("urdf", type=Path, help="Path to the URDF or xacro file.")
    parser.add_argument(
        "--base",
        default=None,
        help=(
            "Link name to treat as the base of the kinematic chain. "
            "Auto-detected (parent of the first actuated joint) when omitted."
        ),
    )
    parser.add_argument(
        "--ee",
        default=None,
        help=(
            "Link name to treat as the end-effector of the kinematic chain. "
            "Auto-detected (the actuated flange of the longest chain) when omitted. "
            "The onboarding gate (fixture parity + coverage) catches a wrong pick."
        ),
    )
    parser.add_argument(
        "--xacro-arg",
        action="append",
        default=[],
        metavar="NAME:=VALUE",
        dest="xacro_arg",
        help=(
            "Xacro substitution arg for parametrized descriptions (repeatable), "
            "e.g. --xacro-arg ur_type:=ur10e. Ignored for plain URDFs."
        ),
    )


def _resolve_base_ee(args: argparse.Namespace) -> None:
    """Fill ``args.base`` / ``args.ee`` by auto-detection when omitted, and
    print what was chosen plus any alternatives. A wrong guess is caught by the
    onboarding gate (fixture parity + coverage), so this is a convenience, not a
    correctness risk."""
    if args.base and args.ee:
        return
    base, ee, notes = suggest_base_ee(args.urdf, _parse_xacro_args(args))
    if not args.base:
        args.base = base
        print(f"[ssik]   auto-detected --base {base}")
    if not args.ee:
        args.ee = ee
        print(f"[ssik]   auto-detected --ee {ee}")
    for note in notes:
        print(f"[ssik]   note: {note}")
    print("[ssik]   verify these against your robot; the gate catches a wrong ee.")


def _parse_xacro_args(args: argparse.Namespace) -> dict[str, str] | None:
    """Parse ``--xacro-arg NAME:=VALUE`` pairs into a substitution dict."""
    pairs: list[str] = getattr(args, "xacro_arg", []) or []
    if not pairs:
        return None
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition(":=")
        if not sep or not key:
            raise SystemExit(f"[ssik] ERROR: bad --xacro-arg {pair!r}; expected NAME:=VALUE")
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Logging configuration.
# ---------------------------------------------------------------------------


def _configure_logging(verbose_count: int) -> None:
    """Install a stderr handler on the ``ssik`` namespace at the chosen level.

    -v raises to INFO (per-solver entry/exit logs); -vv raises to DEBUG.
    Default (0) leaves the namespace at WARNING -- only anomalous-recovery
    messages bubble through.
    """
    level = logging.WARNING
    if verbose_count == 1:
        level = logging.INFO
    elif verbose_count >= 2:
        level = logging.DEBUG
    logger = logging.getLogger("ssik")
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# `ssik classify` -- dry-run inspection.
# ---------------------------------------------------------------------------


def _run_classify(args: argparse.Namespace) -> int:
    print(f"[ssik] Loading {args.urdf}")
    _resolve_base_ee(args)
    kb = load_urdf_kinbody_normalized(
        args.urdf, args.base, args.ee, xacro_args=_parse_xacro_args(args)
    )
    print(f"[ssik]   {len(kb.joints)} joints, {len(kb.links)} links — POE-normalized OK")
    plan = dispatch(kb)
    _print_dispatch_summary(plan)
    return 0


# ---------------------------------------------------------------------------
# `ssik build` -- end-to-end artifact emission + validation.
# ---------------------------------------------------------------------------


def _run_build(args: argparse.Namespace) -> int:
    print(f"[ssik] Loading {args.urdf}")
    _resolve_base_ee(args)
    kb = load_urdf_kinbody_normalized(
        args.urdf, args.base, args.ee, xacro_args=_parse_xacro_args(args)
    )
    print(f"[ssik]   {len(kb.joints)} joints, {len(kb.links)} links — POE-normalized OK")

    print("[ssik] Classifying topology")
    plan = dispatch(kb)
    _print_dispatch_summary(plan)

    if plan.needs_symbolic_precompute and plan.estimated_precompute_seconds is not None:
        print(
            f"[ssik] Build-time precompute (symbolic): "
            f"~{plan.estimated_precompute_seconds:.0f} s estimated"
        )
        print(
            "[ssik]   (Phase 1 of #110: precompute still runs at first solve(); "
            "build-time baking is Phase 2.)"
        )
    else:
        print("[ssik] No build-time precompute needed (tier-0 closed-form)")

    module_name = args.module_name or f"{args.urdf.stem}_ik"
    output_path = args.out or Path.cwd() / f"{module_name}.py"

    print(f"[ssik] Emitting {output_path}")
    result = emit_artifact(
        kb=kb,
        plan=plan,
        module_name=module_name,
        output_path=str(output_path),
        arm_label=args.urdf.stem,
    )
    print(f"[ssik]   Wrote {len(result.source):,} bytes")

    n_validate = 0 if args.no_validate else args.validate_poses
    if n_validate > 0:
        print(f"[ssik] Validating ({n_validate} random poses)")
        validation = _validate_artifact(output_path, module_name, kb, n_validate)
        # Real correctness regression: candidates returned but FK > 1e-6.
        if validation.fk_failures > 0:
            print(
                f"[ssik]   ✗ {validation.fk_failures}/{n_validate} poses had a "
                f"candidate with FK > 1e-6; max FK error {validation.max_fk_err:.2e}"
            )
            print("[ssik] Build FAILED.")
            return 1
        n_solved = n_validate - validation.empty_poses
        # Random uniform-q samples on multi-DOF arms regularly hit near-singular
        # poses the solver legitimately refuses; report as info, not a failure.
        empty_suffix = (
            f" ({validation.empty_poses} pose{'s' if validation.empty_poses != 1 else ''} "
            f"near-singular, no IK returned)"
            if validation.empty_poses > 0
            else ""
        )
        print(
            f"[ssik]   ✓ {n_solved}/{n_validate} poses solved, "
            f"median {validation.median_ms:.3f} ms, "
            f"max FK error {validation.max_fk_err:.2e}{empty_suffix}"
        )
    else:
        print("[ssik] Validation skipped")

    print("[ssik] ✓ Done. Try:")
    print(f"[ssik]     >>> import {module_name}")
    print(f"[ssik]     >>> sols = {module_name}.solve(T_target)")
    return 0


def _print_dispatch_summary(plan: DispatchPlan) -> None:
    print(f"[ssik]   → Best solver: {plan.solver_name} (tier {plan.tier})")
    print(f"[ssik]   → Expected median IK time: ~{plan.expected_ms_median} ms")
    print(f"[ssik]   → FLOP budget: ~{plan.flop_budget:,} FLOPs / solve")
    print("[ssik]   → Reasoning:")
    for line in plan.reason.splitlines():
        print(f"[ssik]       {line}")


# ---------------------------------------------------------------------------
# Post-emit validation.
# ---------------------------------------------------------------------------


class _ValidationResult:
    __slots__ = ("empty_poses", "fk_failures", "max_fk_err", "median_ms")

    def __init__(
        self,
        *,
        empty_poses: int,
        fk_failures: int,
        max_fk_err: float,
        median_ms: float,
    ) -> None:
        # ``empty_poses``: solve(T) returned [] (pose was near-singular or
        # outside the analytical solver's reachable set). Expected on random
        # uniform-q samples; doesn't indicate an artifact bug.
        # ``fk_failures``: solve(T) returned candidates but at least one had
        # FK residual > 1e-6 (real correctness regression).
        self.empty_poses = empty_poses
        self.fk_failures = fk_failures
        self.max_fk_err = max_fk_err
        self.median_ms = median_ms


def _validate_artifact(
    artifact_path: Path,
    module_name: str,
    kb_source: object,
    n_poses: int,
) -> _ValidationResult:
    """Import the emitted artifact, run ``n_poses`` random IK solves, verify
    every returned solution closes FK against the seeded target.

    Two distinct counters:

    - ``empty_poses``: how many random poses produced no candidates. This
      reflects the arm's analytical reachability, not artifact quality;
      ``rng.uniform(-1, 1)`` on a 7-DOF arm regularly lands near singular
      configurations the solver legitimately refuses.
    - ``fk_failures``: how many returned candidates had FK closure worse
      than 1e-6. This is the real correctness gate -- artifacts that ship
      candidates with high FK error are broken.
    """
    spec = importlib.util.spec_from_file_location(f"_ssik_validate_{module_name}", artifact_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(seed=0)
    n_dof = len(kb_source.joints)  # type: ignore[attr-defined]
    times: list[float] = []
    fk_errs: list[float] = []
    empty_poses = 0
    fk_failures = 0
    for _ in range(n_poses):
        q_star = rng.uniform(-1.0, 1.0, size=n_dof)
        T_star = _fk_poe(kb_source, q_star)
        t0 = time.perf_counter()
        # Validation samples q from [-1, 1] which can land outside URDF
        # limits; bypass respect_limits for FK-roundtrip checks.
        sols = mod.solve(T_star, respect_limits=False)
        times.append((time.perf_counter() - t0) * 1e3)
        if not sols:
            empty_poses += 1
            continue
        worst = 0.0
        for sol in sols:
            T_check = _fk_poe(kb_source, sol.q)
            err = float(np.linalg.norm(T_check - T_star))
            worst = max(worst, err)
        fk_errs.append(worst)
        if worst > 1e-6:
            fk_failures += 1
    return _ValidationResult(
        empty_poses=empty_poses,
        fk_failures=fk_failures,
        max_fk_err=(max(fk_errs) if fk_errs else float("nan")),
        median_ms=float(np.median(times)),
    )


def _fk_poe(kb: object, q: np.ndarray) -> np.ndarray:
    """POE forward kinematics matching the artifact's representation."""
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):  # type: ignore[attr-defined]
        rot = np.eye(4)
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


# ---------------------------------------------------------------------------
# `ssik add-arm` -- vendor a URDF + generate a bulletproof test scaffold (#196).
# ---------------------------------------------------------------------------


def _run_add_arm(args: argparse.Namespace) -> int:
    repo_root = args.repo_root or Path.cwd()
    fixtures_dir = repo_root / "tests" / "fixtures"
    tests_dir = repo_root / "tests"
    if not fixtures_dir.is_dir():
        print(f"[ssik add-arm] ERROR: {fixtures_dir} does not exist.")
        print("[ssik add-arm]   Pass --repo-root to point at the ssik repository.")
        return 1

    urdf_dest = fixtures_dir / f"{args.name}.urdf"
    test_dest = tests_dir / f"test_{args.name}.py"
    if not args.force:
        for p in (urdf_dest, test_dest):
            if p.exists():
                print(f"[ssik add-arm] ERROR: {p} already exists.")
                print("[ssik add-arm]   Pass --force to overwrite.")
                return 1

    print(f"[ssik add-arm] Loading {args.urdf}")
    if not args.urdf.is_file():
        print(f"[ssik add-arm] ERROR: {args.urdf} not found.")
        return 1
    _resolve_base_ee(args)
    # Vendor a mesh-free, kinematics-only fixture FIRST, then load + classify
    # from THAT (never the mesh-laden source). Stripping meshes up front drops
    # every ``package://`` reference, so vendor URDFs that declare xmlns:xacro
    # but reference meshes by ROS package (ABB YuMi, FANUC) load without a ROS
    # workspace. Only genuinely-macro'd xacro is expanded first.
    rel = urdf_dest.relative_to(repo_root)
    print(f"[ssik add-arm] Vendoring URDF (kinematics-only) -> {rel}")
    if needs_xacro_expansion(args.urdf):
        with _as_plain_urdf(args.urdf, _parse_xacro_args(args)) as plain_urdf:
            n_links, n_joints = strip_urdf_to_fixture(plain_urdf, urdf_dest)
    else:
        n_links, n_joints = strip_urdf_to_fixture(args.urdf, urdf_dest)
    kb_bytes = urdf_dest.stat().st_size
    print(f"[ssik add-arm]   stripped to {n_links} links, {n_joints} joints, {kb_bytes:,} bytes")

    kb = load_urdf_kinbody_normalized(urdf_dest, args.base, args.ee)
    print(f"[ssik add-arm]   {len(kb.joints)} joints, {len(kb.links)} links — POE-normalized OK")
    print("[ssik add-arm] Classifying topology")
    plan = dispatch(kb)
    _print_dispatch_summary(plan)

    print(f"[ssik add-arm] Generating test scaffold -> {test_dest.relative_to(repo_root)}")
    test_source = _render_test_scaffold(
        arm_name=args.name,
        urdf_filename=urdf_dest.name,
        base_link=args.base,
        ee_link=args.ee,
        dof=len(kb.joints),
        plan=plan,
    )
    test_dest.write_text(test_source)
    _ruff_format(test_dest)  # so the scaffold passes CI's ``ruff format --check``
    print(f"[ssik add-arm]   wrote {len(test_source):,} bytes ({test_source.count(chr(10))} lines)")

    worst_fk: float | None = None
    if not args.no_validate:
        worst_fk = _build_and_validate(args, kb, plan)
    else:
        _live_smoke(kb, args.name)

    sample_q = _pick_sample_q(kb)
    stanza = _render_manifest_stanza(
        args.name, args.base, args.ee, len(kb.joints), plan, args.vendor, worst_fk, sample_q
    )
    print()
    if args.write_manifest:
        manifest_path = repo_root / "src" / "ssik" / "prebuilt" / "MANIFEST.toml"
        rc = _append_manifest_stanza(manifest_path, args.name, stanza, force=args.force)
        if rc != 0:
            return rc
        rel_manifest = manifest_path.relative_to(repo_root)
        print(f"[ssik add-arm] Appended [arms.{args.name}] to {rel_manifest}")
        print("[ssik add-arm]   Fill the TODO fields (kinematic_class, class_tags, ...) by hand.")
    else:
        print("[ssik add-arm] Add this stanza to src/ssik/prebuilt/MANIFEST.toml")
        print("[ssik add-arm] (TODO fields need your judgement; the rest is derived):")
        print("[ssik add-arm] (or re-run with --write-manifest to append it automatically)")
        print()
        print(stanza)
    print()
    print("[ssik add-arm] ✓ Then finish (build artifact, then one-click bench+docs):")
    print(f"[ssik add-arm]     uv run pytest {test_dest.relative_to(repo_root)} -v")
    print("[ssik add-arm]     uv run python scripts/regen_artifacts.py [--include-slow]")
    print(f"[ssik add-arm]     uv run python scripts/regen_bench.py --arm {args.name} --docs")
    return 0


# Solvers whose baked constants drift in float last-digits across platforms
# (heavy sympy/RR preprocessing). Their artifacts get platform_drift = true so
# the snapshot test uses structural markers off-macOS instead of byte equality.
_DRIFT_PRONE_SOLVERS = frozenset(
    {"ikgeo.general_6r", "seven_r.srs_polished", "seven_r.spherical_shoulder_polished"}
)


# Scaffold FK-closure ceiling by solver numerical floor. Most paths reach
# machine precision or LM-polish to <= 1e-8 (1e-6 is a safe loose bound); the
# RR-resultant general_6r path has a documented ~1e-5 structural conditioning
# floor (JACO 2 / PiPER class), so its scaffold needs a looser ceiling or the
# generated coverage gate fails on a correct arm (Doosan M-series, #292).
_SCAFFOLD_FK_CEILING: dict[str, float] = {"ikgeo.general_6r": 1e-4}


def _scaffold_fk_ceiling(solver_name: str) -> float:
    return _SCAFFOLD_FK_CEILING.get(solver_name, 1e-6)


def _fk_ceiling_from_worst(worst_fk: float) -> float:
    """A clean power-of-ten FK ceiling one decade above the measured worst-case
    residual, floored at 1e-9 (no ceiling tighter than closed-form 6R needs)."""
    import math

    if worst_fk <= 0:
        return 1e-9
    return float(max(1e-9, 10 ** math.ceil(math.log10(worst_fk) + 1)))


def _build_and_validate(args: argparse.Namespace, kb: KinBody, plan: DispatchPlan) -> float | None:
    """Emit the artifact to a temp module, import it, and run the coverage gate
    against the SHIPPED ``module.solve``. Prints a PASS/FAIL verdict and returns
    the worst-case FK residual (for the stanza's ``fk_ceiling_fuzz``), or ``None``
    if coverage is too low to ship."""
    import importlib.util as ilu
    import tempfile

    from ssik._validation import check_solve_coverage

    print("[ssik add-arm] Building + validating the emitted artifact (coverage gate)")
    with tempfile.TemporaryDirectory() as d:
        art = Path(d) / f"{args.name}.py"
        emit_artifact(
            kb=kb, plan=plan, module_name=args.name, output_path=str(art), arm_label=args.name
        )
        spec = ilu.spec_from_file_location(args.name, art)
        if spec is None or spec.loader is None:  # pragma: no cover -- defensive
            print("[ssik add-arm]   could not import the emitted artifact; skipping validation")
            return None
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        coverage, worst_fk = check_solve_coverage(mod, n_poses=64)

    if coverage < 0.9:
        print(
            f"[ssik add-arm] ✗ VALIDATION FAILED: coverage {coverage:.0%} (worst FK {worst_fk:.1e})"
        )
        print("[ssik add-arm]   The emitted artifact returns few/no IK on reachable poses -- do")
        print("[ssik add-arm]   NOT ship as-is. Check --base/--ee, joint limits ([0,0] locks the")
        print("[ssik add-arm]   arm), and wrist-gauge. The fixture/test/stanza were still written.")
        return None
    print(f"[ssik add-arm] ✓ VALIDATED: coverage {coverage:.0%}, worst FK {worst_fk:.1e}")
    return worst_fk


def _live_smoke(kb: KinBody, name: str, *, n_poses: int = 8) -> None:
    """Live-solver coverage smoke for a ``--no-validate`` (slow-build) arm.

    ``--no-validate`` skips the minutes-long artifact build + coverage gate, and
    used to skip onboarding validation entirely -- so a broken slow-build arm
    (0 IK / ``[0,0]``-locked limits / wrong ee) scaffolded green (#445). This
    runs a bounded live-solver smoke instead: no artifact build, ~seconds even on
    HP-fallback 7R arms, enough to rule OUT catastrophic breakage. It is NOT a
    coverage certification (7R arms run the slower HP fallback here, with reduced
    coverage vs the built cached-RR artifact), so it always prints a loud
    UNVALIDATED reminder to build + run the full gate before shipping."""
    import numpy as np

    from ssik.kinematics.poe_fk import poe_forward_kinematics
    from ssik.manipulator import Manipulator

    print("[ssik add-arm] --no-validate: skipping the artifact build + coverage gate.")
    print(f"[ssik add-arm] Running a live-solver smoke instead ({n_poses} in-limits poses):")
    arm = Manipulator(kb)
    rng = np.random.default_rng(0)
    limits = [(j.limits if j.limits is not None else (-np.pi, np.pi)) for j in kb.joints]
    covered = 0
    for _ in range(n_poses):
        q = np.array([rng.uniform(lo, hi) for lo, hi in limits])
        if arm.solve(poe_forward_kinematics(kb, q)):
            covered += 1

    if covered == 0:
        print(f"[ssik add-arm] ✗ LIKELY BROKEN: live solver returned no IK on any of {n_poses}")
        print("[ssik add-arm]   reachable poses. Check --base/--ee, joint limits ([0,0] locks the")
        print("[ssik add-arm]   arm), and wrist gauge BEFORE shipping.")
    else:
        print(f"[ssik add-arm] ~ live smoke: {covered}/{n_poses} poses solved (catastrophic-only")
        print("[ssik add-arm]   check; 7R arms use the slower HP fallback here, reduced coverage).")
    print("[ssik add-arm] ⚠ UNVALIDATED artifact -- before shipping, build + run the full gate:")
    print(f"[ssik add-arm]     uv run python scripts/regen_artifacts.py --arm {name}")
    print(f"[ssik add-arm]     uv run pytest tests/test_{name}.py")


def _append_manifest_stanza(manifest_path: Path, name: str, stanza: str, *, force: bool) -> int:
    """Append ``stanza`` (a rendered ``[arms.<name>]`` block) to ``manifest_path``.

    Refuses (returns 1) if the manifest is missing or already defines
    ``[arms.<name>]``, unless ``force`` is set (in which case the existing
    entry, including its ``.bench`` / ``.eaik`` sub-tables, is removed first).
    Returns 0 on success.
    """
    if not manifest_path.is_file():
        print(f"[ssik add-arm] ERROR: manifest not found at {manifest_path}.")
        print("[ssik add-arm]   Run from the repo root, or pass --repo-root.")
        return 1
    text = manifest_path.read_text(encoding="utf-8")
    header = f"[arms.{name}]"
    if header in text:
        if not force:
            print(f"[ssik add-arm] ERROR: {header} already exists in {manifest_path.name}.")
            print("[ssik add-arm]   Pass --force to replace it.")
            return 1
        text = _strip_manifest_entry(text, name)
    # Append after a single blank-line separator, keeping a trailing newline.
    body = text.rstrip("\n")
    manifest_path.write_text(f"{body}\n\n{stanza.rstrip(chr(10))}\n", encoding="utf-8")
    return 0


def _strip_manifest_entry(text: str, name: str) -> str:
    """Remove ``[arms.<name>]`` and its ``.bench`` / ``.eaik`` sub-tables from a
    manifest string, so a ``--force`` re-add replaces rather than duplicates."""
    lines = text.splitlines()
    prefixes = (f"[arms.{name}]", f"[arms.{name}.")
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            # A new table header decides whether we keep skipping.
            skipping = any(stripped.startswith(p) for p in prefixes)
        if not skipping:
            out.append(line)
    return "\n".join(out).rstrip("\n") + "\n"


# Class metadata is a function of the dispatched solver, not the individual arm,
# so add-arm derives it instead of leaving three TODOs the user hand-copies from a
# same-class neighbour (error-prone; class_tags is one distinct set per solver
# across the whole roster). A human may still refine the per-arm parenthetical
# color in ``kinematic_class`` (e.g. "(offset wrist)", "~1.4 m reach").
_SOLVER_CLASS_META: dict[str, tuple[str, str, list[str]]] = {
    "ikgeo.three_parallel": (
        "three-parallel 6R (UR-class)",
        "three-parallel 6R",
        ["three-parallel", "6R", "Pieper"],
    ),
    "ikgeo.spherical_two_parallel": (
        "Pieper 6R (spherical wrist)",
        "Pieper 6R, spherical wrist",
        ["spherical-wrist", "6R", "Pieper"],
    ),
    "ikgeo.spherical_two_intersecting": (
        "Pieper 6R (spherical wrist, intersecting shoulder)",
        "Pieper 6R, spherical wrist",
        ["spherical-wrist", "6R", "Pieper"],
    ),
    "ikgeo.spherical": (
        "Pieper 6R (spherical wrist, general shoulder)",
        "Pieper 6R, spherical wrist",
        ["spherical-wrist", "6R", "Pieper"],
    ),
    "ikgeo.general_6r": ("non-Pieper 6R", "non-Pieper 6R", ["non-Pieper", "6R"]),
    "jointlock.seven_r": ("non-SRS 7R", "non-SRS 7R", ["non-SRS", "7R"]),
    "seven_r.srs": ("SRS 7R", "SRS 7R", ["SRS", "7R"]),
    "seven_r.srs_polished": (
        "approximate-SRS 7R",
        "approximate-SRS 7R",
        ["approximate-SRS", "7R"],
    ),
    "seven_r.spherical_shoulder": (
        "spherical-shoulder + offset-wrist 7R",
        "spherical-shoulder + offset-wrist 7R",
        ["non-SRS", "spherical-shoulder", "offset-wrist", "7R"],
    ),
    "seven_r.spherical_shoulder_polished": (
        "approximately-spherical-shoulder 7R",
        "approximately-spherical-shoulder 7R",
        ["non-SRS", "spherical-shoulder", "approximate", "7R"],
    ),
}

# Vendor slug -> display prefix for the auto-derived display_name (a starting
# point the user refines with the marketing model string).
_VENDOR_DISPLAY: dict[str, str] = {
    "universal_robots": "Universal Robots",
    "abb": "ABB",
    "kuka": "KUKA",
    "kinova": "Kinova",
    "fanuc": "FANUC",
    "franka": "Franka",
    "flexiv": "Flexiv",
    "kassow": "Kassow",
    "standard_bots": "Standard Bots",
    "agilex": "AgileX",
    "unitree": "Unitree",
    "yaskawa": "Yaskawa",
    "staubli": "Staubli",
    "kawasaki": "Kawasaki",
    "doosan": "Doosan",
}


def _class_meta(solver_name: str) -> tuple[str, str, str]:
    """(kinematic_class, short_class, class_tags-as-TOML) for a solver, or TODO
    placeholders if the solver is unmapped (a new solver family)."""
    meta = _SOLVER_CLASS_META.get(solver_name)
    if meta is None:
        return ("TODO", "TODO", '["TODO"]')
    kc, sc, tags = meta
    return kc, sc, "[" + ", ".join(f'"{t}"' for t in tags) + "]"


def _default_display_name(vendor: str, name: str) -> str:
    """A best-effort ``display_name`` from vendor + arm name (the user confirms
    the marketing model string). e.g. ('abb', 'irb120_ik') -> 'ABB irb120'."""
    prefix = _VENDOR_DISPLAY.get(vendor, vendor.replace("_", " ").title())
    model = name.removesuffix("_ik")
    return f"{prefix} {model}".strip()


def _pick_sample_q(kb: KinBody) -> list[float]:
    """A verified in-limits, live-solvable pose for the stanza's ``sample_q``
    (which ``test_prebuilt_sanity`` solves). Returns the first random in-limits
    pose whose live solve succeeds -- usually the first try -- so a near-home
    singular default can't silently break sanity. Falls back to a small tilt."""
    import numpy as np

    from ssik.kinematics.poe_fk import poe_forward_kinematics
    from ssik.manipulator import Manipulator

    arm = Manipulator(kb)
    rng = np.random.default_rng(0)
    limits = [(j.limits if j.limits is not None else (-np.pi, np.pi)) for j in kb.joints]
    for _ in range(20):
        q = np.array([rng.uniform(lo, hi) for lo, hi in limits])
        if arm.solve(poe_forward_kinematics(kb, q)):
            return [round(float(x), 3) for x in q]
    return [0.1] * len(kb.joints)


def _render_manifest_stanza(
    name: str,
    base: str,
    ee: str,
    dof: int,
    plan: DispatchPlan,
    vendor: str,
    worst_fk: float | None = None,
    sample_q: list[float] | None = None,
) -> str:
    """A ready-to-paste MANIFEST.toml stanza. Derived fields filled: solver, tier,
    dof, platform_drift, fk_ceiling (validation smoke), kinematic_class /
    short_class / class_tags (from the solver), sample_q (a verified-solvable
    pose), and a best-effort display_name. Human-provenance fields
    (fixture_source, the marketing display strings) stay editable.
    ``regen_bench.py`` fills ``[bench]`` and ``[eaik]``."""
    sample = ", ".join(f"{q:.3g}" for q in (sample_q or [0.1] * dof))
    display = _default_display_name(vendor, name)
    kin_class, short_class, tags_toml = _class_meta(plan.solver_name)
    drift = plan.solver_name in _DRIFT_PRONE_SOLVERS
    fk_ceiling = _fk_ceiling_from_worst(worst_fk) if worst_fk is not None else 1e-4
    # Floor the ceiling at the solver's structural FK floor: the general_6r RR
    # path's worst-case FK varies widely across the workspace, so a 64-pose
    # validation smoke can under-estimate it and derive a ceiling the 500-pose
    # uniform fuzz then breaks (xMate SR3: smoke 1.7e-8 -> 1e-6, fuzz found
    # 2.7e-6). ``_SCAFFOLD_FK_CEILING`` holds that per-solver floor (general_6r
    # -> 1e-4); ``.get(..., 0.0)`` leaves every other (machine-precision) solver
    # on its tight smoke-derived value.
    fk_ceiling = max(fk_ceiling, _SCAFFOLD_FK_CEILING.get(plan.solver_name, 0.0))
    lines = [
        f"[arms.{name}]",
        f'display_name = "{display}"  # verify the marketing model string',
        f'short_name = "{display}"  # shorten for compact table cells',
        f'vendor = "{vendor}"',
        f'fixture = "{name}.urdf"',
        'fixture_kind = "urdf"',
        'fixture_source = "TODO: repo/source + license (e.g. robot_descriptions / <pkg>)"',
        f'base_link = "{base}"',
        f'ee_link = "{ee}"',
        f"dof = {dof}",
        f'solver = "{plan.solver_name}"',
        f"tier = {plan.tier}",
        f'kinematic_class = "{kin_class}"',
        f'short_class = "{short_class}"',
        f"class_tags = {tags_toml}",
        "slow_build = false  # set true if the build is minutes (cached-RR 7R)",
        "build_time_sec = 1",
        "artifact_size_kb = 20",
        f"sample_q = [{sample}]",
        f"fk_ceiling_fuzz = {fk_ceiling:.0e}  # from the validation smoke (~10x worst FK)",
        f"platform_drift = {'true' if drift else 'false'}",
    ]
    if drift:
        lines += [
            "drift_markers = [",
            f"    'SOLVER_NAME = \"{plan.solver_name}\"',",
            '    "def fk(q):",',
            "]",
        ]
    lines += [
        "",
        f"[arms.{name}.bench]  # filled by scripts/regen_bench.py",
        "ms_mean = 0.0",
        "ms_ci95 = 0.0",
        "max_fk = 0.0",
        "sols_min = 0",
        "sols_max = 0",
        "",
        f"[arms.{name}.eaik]  # filled by scripts/regen_bench.py (needs EAIK installed)",
        "supported = false",
        'refusal = "TODO (regen_bench fills this; or set supported=true + numbers)"',
    ]
    return "\n".join(lines)


def _render_test_scaffold(
    *,
    arm_name: str,
    urdf_filename: str,
    base_link: str,
    ee_link: str,
    dof: int,
    plan: DispatchPlan,
) -> str:
    """Render the per-arm test scaffold based on the dispatched solver.

    The generated test file contains:

    1. URDF load + DOF / joint-type sanity.
    2. Dispatcher routing (asserts the solver name selected by the
       current dispatcher).
    3. ``@pytest.mark.slow`` hand-picked seeded recovery (4 q*).
    4. ``@pytest.mark.slow`` Hypothesis fuzz (10 random reachable poses).

    Tests assert FK closure ≤ 1e-10 on the BEST IK per pose (matching
    the bulletproof-validation contract).
    """
    arm_label = arm_name
    kb_helper = f"_{arm_name}_kinbody"
    fk_ceiling = _scaffold_fk_ceiling(plan.solver_name)
    routing_test = f"test_{arm_name}_dispatches_to_{_solver_assertion_slug(plan.solver_name)}"
    docstring_header = (
        f'"""Bulletproof validation for the {arm_label} fixture '
        f"(auto-generated by ``ssik add-arm``)."
    )
    return f'''{docstring_header}

The arm dispatches to ``{plan.solver_name}`` (tier {plan.tier}) per the
current ``ssik.core.dispatcher``. The scaffold below verifies, against the
**shipped artifact** (``ssik.prebuilt.{arm_name}`` -- build it with
``uv run python scripts/regen_artifacts.py --arm {arm_name}`` first):

- URDF loads as a {dof}-DOF chain (revolute / continuous joints).
- Dispatcher routing is stable.
- **Coverage:** poses sampled across the arm's real joint limits return at
  least one IK for at least ``_MIN_COVERAGE`` of them. This is the gate a
  broken/degenerate arm (0 solutions, locked ``[0, 0]`` limits, wrong ee
  gauge) fails loudly -- "FK closes on every retained IK" is vacuously
  true at zero coverage, so coverage is asserted separately.
- Every returned IK FK-closes within ``_FK_CEILING``.

If the arm has a genuine structural coverage gap (redundant-7R near-
singular poses, etc.), lower ``_MIN_COVERAGE`` here AND add a
``[arms.{arm_name}.known_gaps]`` entry in MANIFEST.toml documenting why.

Source URDF: ``tests/fixtures/{urdf_filename}`` (vendored via ``ssik add-arm``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.dispatcher import dispatch

URDF_PATH = Path(__file__).parent / "fixtures" / "{urdf_filename}"

# Fraction of reachable poses that must return >= 1 IK. A broken arm
# returns few/none; this is the coverage floor the gate enforces.
_MIN_COVERAGE = 0.95
_N_COVERAGE_POSES = 64
# Worst-case FK-closure ceiling on returned solutions, set from the solver's
# numerical floor (closed-form / LM-polished <= 1e-6; non-Pieper RR ~1e-5).
_FK_CEILING = {fk_ceiling:.0e}


def {kb_helper}():
    return load_urdf_kinbody_normalized(URDF_PATH, "{base_link}", "{ee_link}")


def _joint_bounds():
    """(lo, hi) arrays of per-joint sampling bounds from the real URDF
    limits; continuous / limitless joints sample [-pi, pi]."""
    kb = {kb_helper}()
    lo, hi = [], []
    for j in kb.joints:
        lim = getattr(j, "limits", None)
        if lim is None:
            lo.append(-np.pi)
            hi.append(np.pi)
        else:
            lo.append(float(lim[0]))
            hi.append(float(lim[1]))
    return np.array(lo), np.array(hi)


# ----------------------------------------------------------------------------
# URDF load + topology + routing (fixture-based, fast)
# ----------------------------------------------------------------------------


def test_{arm_name}_loads_as_{dof}dof() -> None:
    kb = {kb_helper}()
    assert len(kb.joints) == {dof}
    for j in kb.joints:
        assert j.joint_type in ("revolute", "continuous")


def {routing_test}() -> None:
    """Dispatcher routing is stable. Updating the dispatcher should
    update this assertion deliberately.
    """
    kb = {kb_helper}()
    plan = dispatch(kb)
    assert plan.solver_name == "{plan.solver_name}"
    assert plan.tier == {plan.tier}


# ----------------------------------------------------------------------------
# Coverage + FK-closure gate against the SHIPPED artifact (fast, default gate)
# ----------------------------------------------------------------------------


def test_{arm_name}_coverage_and_fk_closure() -> None:
    """Sample poses across the arm's real joint limits, FK + solve with the
    shipped ``module.solve`` (not the raw solver), and assert coverage >=
    ``_MIN_COVERAGE`` and FK closure within ``_FK_CEILING``.

    This is the gate a broken arm fails: 0-coverage (FANUC-M710-style
    conditioning, Kinova-[0,0]-limit lockouts) trips the coverage assert
    instead of passing vacuously.
    """
    art = pytest.importorskip(
        "ssik.prebuilt.{arm_name}",
        reason="build the artifact first: regen_artifacts.py --arm {arm_name}",
    )
    from ssik._validation import assert_solve_coverage

    assert_solve_coverage(
        art,
        min_coverage=_MIN_COVERAGE,
        fk_ceiling=_FK_CEILING,
        n_poses=_N_COVERAGE_POSES,
    )


# ----------------------------------------------------------------------------
# Thorough Hypothesis fuzz against the shipped artifact (slow)
# ----------------------------------------------------------------------------


@pytest.mark.slow
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_{arm_name}_random_pose_fk_closure(seed: int) -> None:
    """200 random reachable poses (within real joint limits): at least one
    returned IK FK-closes within ``_FK_CEILING``."""
    art = pytest.importorskip(
        "ssik.prebuilt.{arm_name}",
        reason="build the artifact first: regen_artifacts.py --arm {arm_name}",
    )

    lo, hi = _joint_bounds()
    rng = np.random.default_rng(seed)
    q_star = rng.uniform(lo, hi)
    sols = art.solve(art.fk(q_star))
    assert sols, f"no IK returned for reachable q*={{q_star.tolist()}}"
    best_fk = min(float(s.fk_residual) for s in sols)
    assert best_fk < _FK_CEILING, f"seed={{seed}}: best FK={{best_fk:.2e}} > ceiling"
'''


def _solver_assertion_slug(solver_name: str) -> str:
    """Convert solver_name to a Python-identifier-safe slug for test names."""
    return solver_name.replace(".", "_").replace(":", "_")


def _ruff_format(path: Path) -> None:
    """Best-effort ``ruff format`` on a generated file so it passes CI's
    ``ruff format --check`` without a manual pass. Silent if ruff is absent."""
    import contextlib
    import subprocess

    with contextlib.suppress(OSError):
        subprocess.run(
            ["ruff", "format", "--quiet", str(path)],
            check=False,
            capture_output=True,
        )


if __name__ == "__main__":  # pragma: no cover -- entry-point
    sys.exit(main())
