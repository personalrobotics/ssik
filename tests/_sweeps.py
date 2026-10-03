"""Exhaustive sweeps that run a fixed sample on most pull requests.

A few tests sweep hundreds or thousands of seeded cases (random poses, fuzz
draws), and together they cost several minutes of every CI shard. Each sweep
keeps its full size, which runs by default (locally), nightly (slow.yml), on
every push to main, on workflow_dispatch, and on any pull request that changes
the code the sweep covers. Other pull requests run the sweep's PR sample: the
first cases of the same seeded stream, so the sample is identical on every run
and is a subset of the full sweep. No case is removed; the full sweep stays the
contract.

``SSIK_PR_SWEEPS`` selects the sampled sweeps: a comma-separated list of names
from :data:`SWEEPS`, or ``all``. Unset or empty runs every sweep in full. CI's
``Classify changed files`` job (scripts/classify_changes.py) sets it from
:data:`COVERED`, which is the one place a sweep's covered code is listed.
"""

from __future__ import annotations

import os

ENV = "SSIK_PR_SWEEPS"

# Changes under any of these run every sweep in full: the solvers, kinematics
# and native code the sweeps exercise, their shared test helpers and fixtures,
# the build, and the CI wiring that picks the tier. A trailing "/" is a
# directory prefix; anything else is an exact path.
_SHARED = (
    "cpp/",
    "src/ssik/__init__.py",
    "src/ssik/_kinbody.py",
    "src/ssik/_native.py",
    "src/ssik/_solve_inputs.py",
    "src/ssik/_urdf.py",
    "src/ssik/core/",
    "src/ssik/cpp/",
    "src/ssik/kinematics/",
    "src/ssik/postprocess.py",
    "src/ssik/refinement/",
    "src/ssik/solvers/",
    "src/ssik/subproblems/",
    "tests/__init__.py",
    "tests/_cpp_backend.py",
    "tests/_sweeps.py",
    "tests/conftest.py",
    "tests/fixtures/",
    "scripts/build_cpp_ext.py",
    "scripts/classify_changes.py",
    "scripts/fetch_eigen.py",
    "hatch_build.py",
    "pyproject.toml",
    "uv.lock",
    ".github/",
)

# Each sweep's covered code: the shared paths plus its own test files and any
# module only it exercises.
COVERED: dict[str, tuple[str, ...]] = {
    # C++ feasible_arcs / feasible_arcs_bounded vs the Python oracle.
    "feasible_arcs": (*_SHARED, "tests/test_feasible_arcs.py"),
    # C++ SRS resolve_in_limits vs the Python oracle.
    "srs_resolve_cpp": (*_SHARED, "tests/test_srs_resolve_in_limits_cpp.py"),
    # The exact in-limits resolvers return a solution for every in-limits pose.
    "in_limits_resolvers": (
        *_SHARED,
        "tests/test_spherical_shoulder.py",
        "tests/test_swivel_limits.py",
    ),
    # Native self-motion charts vs the Python reference.
    "chart_native": (
        *_SHARED,
        "src/ssik/chart.py",
        "src/ssik/manipulator.py",
        "tests/test_chart_native.py",
    ),
}
SWEEPS = tuple(COVERED)


def covers(sweep: str, path: str) -> bool:
    """Whether a change to ``path`` (repo-relative, ``/``-separated) is covered
    by ``sweep``, so a pull request making it must run the sweep in full."""
    return any(path.startswith(p) if p.endswith("/") else path == p for p in COVERED[sweep])


def sampled() -> frozenset[str]:
    """The sweeps ``SSIK_PR_SWEEPS`` asks to sample; raises on an unknown name."""
    names = {n.strip() for n in os.environ.get(ENV, "").split(",") if n.strip()}
    if names == {"all"}:
        return frozenset(SWEEPS)
    unknown = sorted(names - set(SWEEPS))
    if unknown:
        raise ValueError(f"{ENV}: unknown sweep(s) {unknown}; known: {', '.join(SWEEPS)} or all")
    return frozenset(names)


def cases(sweep: str, *, full: int, pr: int) -> int:
    """How many of the sweep's seeded cases to check: ``pr`` when it is
    sampled, else ``full``."""
    if sweep not in COVERED:
        raise KeyError(sweep)
    return pr if sweep in sampled() else full
