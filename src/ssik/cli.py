"""``ssik`` command-line interface.

Two subcommands:

* ``ssik classify <urdf> --base <link> --ee <link>`` -- classify topology +
  print which solver would be picked, without emitting an artifact.
* ``ssik build <urdf> --base <link> --ee <link> [--out <path>]`` --
  classify, emit a per-arm artifact (\\*_ik.py), validate it on random
  poses, and report timing.

Both commands print explanatory messages by default. ``-v`` raises log
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

from ssik._urdf import load_urdf_kinbody_normalized
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
    return parser


def _add_common_kinbody_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("urdf", type=Path, help="Path to the URDF file.")
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
    kb = load_urdf_kinbody_normalized(args.urdf, args.base, args.ee)
    print(f"[ssik]   {len(kb.joints)} joints, {len(kb.links)} links — POE-normalized OK")
    plan = dispatch(kb)
    _print_dispatch_summary(plan)
    return 0


# ---------------------------------------------------------------------------
# `ssik build` -- end-to-end artifact emission + validation.
# ---------------------------------------------------------------------------


def _run_build(args: argparse.Namespace) -> int:
    print(f"[ssik] Loading {args.urdf}")
    kb = load_urdf_kinbody_normalized(args.urdf, args.base, args.ee)
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
        if validation.failures > 0:
            print(
                f"[ssik]   ✗ {validation.failures}/{n_validate} poses failed FK check; "
                f"max FK error {validation.max_fk_err:.2e}"
            )
            print("[ssik] Build FAILED.")
            return 1
        print(
            f"[ssik]   ✓ {n_validate} poses, median {validation.median_ms:.3f} ms, "
            f"max FK error {validation.max_fk_err:.2e}, 0 failures"
        )
    else:
        print("[ssik] Validation skipped")

    print("[ssik] ✓ Done. Try:")
    print(f"[ssik]     >>> import {module_name}")
    print(f"[ssik]     >>> sols, is_ls = {module_name}.solve(T_target)")
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
    __slots__ = ("failures", "max_fk_err", "median_ms")

    def __init__(self, *, failures: int, max_fk_err: float, median_ms: float) -> None:
        self.failures = failures
        self.max_fk_err = max_fk_err
        self.median_ms = median_ms


def _validate_artifact(
    artifact_path: Path,
    module_name: str,
    kb_source: object,
    n_poses: int,
) -> _ValidationResult:
    """Import the emitted artifact, run ``n_poses`` random IK solves, verify
    every returned solution closes FK against the seeded target."""
    spec = importlib.util.spec_from_file_location(f"_ssik_validate_{module_name}", artifact_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(seed=0)
    n_dof = len(kb_source.joints)  # type: ignore[attr-defined]
    times: list[float] = []
    fk_errs: list[float] = []
    failures = 0
    for _ in range(n_poses):
        q_star = rng.uniform(-1.0, 1.0, size=n_dof)
        T_star = _fk_poe(kb_source, q_star)
        t0 = time.perf_counter()
        sols, is_ls = mod.solve(T_star)
        times.append((time.perf_counter() - t0) * 1e3)
        if is_ls or not sols:
            failures += 1
            continue
        worst = 0.0
        for sol in sols:
            T_check = _fk_poe(kb_source, sol.q)
            err = float(np.linalg.norm(T_check - T_star))
            worst = max(worst, err)
        fk_errs.append(worst)
        if worst > 1e-6:
            failures += 1
    return _ValidationResult(
        failures=failures,
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


if __name__ == "__main__":  # pragma: no cover -- entry-point
    sys.exit(main())
