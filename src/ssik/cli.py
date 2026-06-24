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

from ssik._urdf import _as_plain_urdf, load_urdf_kinbody_normalized, strip_urdf_to_fixture
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
    return parser


def _add_common_kinbody_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("urdf", type=Path, help="Path to the URDF or xacro file.")
    parser.add_argument(
        "--base",
        required=True,
        help="Link name to treat as the base of the kinematic chain.",
    )
    parser.add_argument(
        "--ee",
        required=True,
        help="Link name to treat as the end-effector of the kinematic chain.",
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
    # Resolve xacro -> plain URDF once, then load + vendor from the same source
    # (vendoring a self-contained, expanded URDF, never the raw xacro).
    with _as_plain_urdf(args.urdf, _parse_xacro_args(args)) as plain_urdf:
        kb = load_urdf_kinbody_normalized(plain_urdf, args.base, args.ee)
        print(
            f"[ssik add-arm]   {len(kb.joints)} joints, {len(kb.links)} links — POE-normalized OK"
        )

        print("[ssik add-arm] Classifying topology")
        plan = dispatch(kb)
        _print_dispatch_summary(plan)

        rel = urdf_dest.relative_to(repo_root)
        print(f"[ssik add-arm] Vendoring URDF (kinematics-only) -> {rel}")
        n_links, n_joints = strip_urdf_to_fixture(plain_urdf, urdf_dest)
    kb_bytes = urdf_dest.stat().st_size
    print(f"[ssik add-arm]   stripped to {n_links} links, {n_joints} joints, {kb_bytes:,} bytes")

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
    print(f"[ssik add-arm]   wrote {len(test_source):,} bytes ({test_source.count(chr(10))} lines)")

    print()
    print("[ssik add-arm] Add this stanza to src/ssik/prebuilt/MANIFEST.toml")
    print("[ssik add-arm] (TODO fields need your judgement; the rest is derived):")
    print()
    print(_render_manifest_stanza(args.name, args.base, args.ee, len(kb.joints), plan))
    print()
    print("[ssik add-arm] ✓ Then finish (build artifact, then one-click bench+docs):")
    print(f"[ssik add-arm]     uv run pytest {test_dest.relative_to(repo_root)} -v")
    print("[ssik add-arm]     uv run python scripts/regen_artifacts.py [--include-slow]")
    print(f"[ssik add-arm]     uv run python scripts/regen_bench.py --arm {args.name} --docs")
    return 0


def _render_manifest_stanza(name: str, base: str, ee: str, dof: int, plan: DispatchPlan) -> str:
    """A ready-to-paste MANIFEST.toml stanza: derived fields filled, curated
    fields left as ``TODO`` for human judgement. ``regen_bench.py`` fills the
    ``[bench]`` block."""
    sample = ", ".join("0.1" for _ in range(dof))
    return "\n".join(
        [
            f"[arms.{name}]",
            'display_name = "TODO"',
            'short_name = "TODO"',
            f'fixture = "{name}.urdf"',
            'fixture_kind = "urdf"',
            'fixture_source = "TODO (e.g. robot_descriptions / <pkg>)"',
            f'base_link = "{base}"',
            f'ee_link = "{ee}"',
            f"dof = {dof}",
            f'solver = "{plan.solver_name}"',
            f"tier = {plan.tier}",
            'kinematic_class = "TODO"',
            'short_class = "TODO"',
            'class_tags = ["TODO"]',
            "slow_build = false  # set true if the build is minutes (cached-RR 7R)",
            "build_time_sec = 0  # update after building",
            "artifact_size_kb = 0  # update after building",
            f"sample_q = [{sample}]",
            "fk_ceiling_fuzz = 1e-4",
            "",
            f"[arms.{name}.bench]  # filled by scripts/regen_bench.py",
            "ms_mean = 0.0",
            "ms_ci95 = 0.0",
            "max_fk = 0.0",
            "sols_min = 0",
            "sols_max = 0",
        ]
    )


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
    n_random_poses = 10  # Hypothesis fuzz size; small for the slow path
    fk_atol = 1e-10
    docstring_header = (
        f'"""Bulletproof validation for the {arm_label} fixture '
        f"(auto-generated by ``ssik add-arm``)."
    )
    return f'''{docstring_header}

The arm dispatches to ``{plan.solver_name}`` (tier {plan.tier}) per the
current ``ssik.core.dispatcher``. The scaffold below verifies:

- URDF loads as a {dof}-DOF revolute chain.
- Dispatcher routing is stable.
- Every retained IK FK-closes ≤ {fk_atol:.0e} on hand-picked + random
  reachable poses.

Edit this file freely after generation -- the scaffold is a starting
point. For 7R arms with URDF axis drift, consider adding a drift-
documentation test (see ``tests/test_kinova_gen3.py`` /
``tests/test_flexiv_rizon4.py`` for examples).

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
from ssik.kinematics.poe_fk import poe_forward_kinematics

URDF_PATH = Path(__file__).parent / "fixtures" / "{urdf_filename}"


def {kb_helper}():
    return load_urdf_kinbody_normalized(URDF_PATH, "{base_link}", "{ee_link}")


# ----------------------------------------------------------------------------
# URDF load + topology
# ----------------------------------------------------------------------------


def test_{arm_name}_loads_as_{dof}r() -> None:
    kb = {kb_helper}()
    assert len(kb.joints) == {dof}
    for j in kb.joints:
        assert j.joint_type == "revolute"


def test_{arm_name}_dispatches_to_{_solver_assertion_slug(plan.solver_name)}() -> None:
    """Dispatcher routing is stable. Updating the dispatcher should
    update this assertion deliberately.
    """
    kb = {kb_helper}()
    plan = dispatch(kb)
    assert plan.solver_name == "{plan.solver_name}"
    assert plan.tier == {plan.tier}


# ----------------------------------------------------------------------------
# Hand-picked seeded recovery (slow -- IK calls)
# ----------------------------------------------------------------------------


_HAND_PICKED_Q = [
    np.array({_hand_picked_q_array(0, dof)}),
    np.array({_hand_picked_q_array(1, dof)}),
    np.array({_hand_picked_q_array(2, dof)}),
    np.array({_hand_picked_q_array(3, dof)}),
]


@pytest.mark.slow
@pytest.mark.parametrize("q_star", _HAND_PICKED_Q)
def test_{arm_name}_hand_picked_fk_closure(q_star: np.ndarray) -> None:
    """Every reachable hand-picked q* yields at least one IK with FK
    closure ≤ {fk_atol:.0e}.
    """
    kb = {kb_helper}()
    T_target = poe_forward_kinematics(kb, q_star)
    {_solver_invocation_block(plan.solver_name, "kb", "T_target")}
    assert sols, f"no IK returned for reachable q*={{q_star.tolist()}}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < {fk_atol}, f"best FK={{best_fk:.2e}} > {fk_atol:.0e}"


# ----------------------------------------------------------------------------
# Hypothesis fuzz: random reachable poses
# ----------------------------------------------------------------------------


@pytest.mark.slow
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples={n_random_poses},
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_{arm_name}_random_pose_fk_closure(seed: int) -> None:
    """{n_random_poses} random q in [-0.8, 0.8] per joint: at least one
    returned IK FK-closes < {fk_atol:.0e}.
    """
    rng = np.random.default_rng(seed)
    q_star = rng.uniform(-0.8, 0.8, size={dof})
    kb = {kb_helper}()
    T_target = poe_forward_kinematics(kb, q_star)
    {_solver_invocation_block(plan.solver_name, "kb", "T_target")}
    assert sols, f"no IK returned for random q*={{q_star.tolist()}}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < {fk_atol}, f"seed={{seed}}: best FK={{best_fk:.2e}} > {fk_atol:.0e}"
'''


def _solver_assertion_slug(solver_name: str) -> str:
    """Convert solver_name to a Python-identifier-safe slug for test names."""
    return solver_name.replace(".", "_").replace(":", "_")


def _solver_invocation_block(solver_name: str, kb_var: str, t_var: str) -> str:
    """Render the per-solver invocation line(s) for the test scaffold.

    Pads the chosen solver's import + ``solve`` call onto the test body.
    Different dispatchers expect slightly different call signatures
    (e.g. ``allow_refinement`` is a no-op for some; passing it
    everywhere is harmless).
    """
    if solver_name == "seven_r.srs":
        return f"from ssik.solvers.seven_r import srs\n    sols, _ = srs.solve({kb_var}, {t_var})"
    if solver_name == "seven_r.srs_polished":
        return (
            "from ssik.solvers.seven_r import srs_polished\n"
            f"    sols, _ = srs_polished.solve({kb_var}, {t_var})"
        )
    if solver_name == "jointlock.seven_r":
        return (
            "from ssik.solvers.jointlock import seven_r as jointlock_seven_r\n"
            f"    sols, _ = jointlock_seven_r.solve({kb_var}, {t_var}, allow_refinement=True)"
        )
    if solver_name.startswith("ikgeo."):
        module = solver_name.split(".", 1)[1]
        return (
            f"from ssik.solvers.ikgeo import {module}\n"
            f"    sols, _ = {module}.solve({kb_var}, {t_var}, allow_refinement=True)"
        )
    if solver_name == "husty_pfurner.general_6r":
        return (
            "from ssik.solvers.husty_pfurner import general_6r as hp_general_6r\n"
            f"    sols, _ = hp_general_6r.solve({kb_var}, {t_var}, allow_refinement=True)"
        )
    raise ValueError(f"add-arm: no scaffold template for solver {solver_name!r}")


def _hand_picked_q_array(seed: int, dof: int) -> str:
    """Deterministic hand-picked q* lists for the scaffold's parametrize.

    Seeded so every run of ``ssik add-arm`` produces byte-identical
    test files for the same ``--name`` (regression-friendly).
    """
    rng = np.random.default_rng(seed * 31 + 17)
    q = rng.uniform(-0.8, 0.8, size=dof)
    return "[" + ", ".join(f"{x:.4f}" for x in q) + "]"


if __name__ == "__main__":  # pragma: no cover -- entry-point
    sys.exit(main())
