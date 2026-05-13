# Contributing to ssik

Thanks for your interest. ssik is built around a few load-bearing principles:

1. **Bulletproof correctness over cleverness.** Every solver PR ships with N-way cross-solver agreement tests, FK closure ≤ 1e-10 on every retained IK, and 500+ Hypothesis-fuzzed random poses per fixture. Failures don't merge.
2. **Profile-driven optimisation.** "Perf claim X% faster" is meaningless without a profile probe before and after. Negative-result spikes (Cython estimates that miss by 2-5×, codegen-bake on parts that turn out to be 0.3% of runtime) are published as closed issues so the next contributor doesn't repeat them.
3. **No papering over.** No clearing Hypothesis caches to hide flakes. No widening tolerances to hide drift. Every workaround files the underlying-bug issue.

## Repo layout

```
ssik/
├── src/ssik/                # source
│   ├── manipulator.py       # public Manipulator class — the v1.0 entry point
│   ├── _kinbody.py          # KinBody / Joint / Link dataclasses (impl detail)
│   ├── _urdf.py             # urchin → KinBody bridge (impl detail)
│   ├── cli.py               # `ssik build`, `ssik add-arm`
│   ├── core/                # dispatch, tolerances, Solution, codegen
│   ├── kinematics/          # POE-FK, POE→DH, predicates, reverse-chain
│   ├── subproblems/         # SP1-SP6 + _rotation Cython primitives
│   ├── solvers/             # tier-0/1/2 solver modules (see docs/architecture.md)
│   ├── refinement/          # opt-in Newton polish
│   └── codegen/             # `ssik build` artifact emitter
├── tests/                   # 1284 tests
│   └── fixtures/            # URDF + Python-spec arm fixtures
├── docs/                    # arm_coverage.md, architecture.md
├── scripts/                 # bench, profile, regen artifacts
└── pyproject.toml
```

## Dev setup

```bash
git clone https://github.com/personalrobotics/ssik.git
cd ssik
uv sync                                 # install dev deps
scripts/install-hooks.sh                # one-time: install pre-push check hook
```

`uv` is the recommended package manager; `pip install -e .[urdf]` works too if you prefer pip.

## Pre-push gate (replaces most of CI)

CI is intentionally minimal — a single Linux wheel-smoke job that catches packaging-class bugs (~5 min per PR). Everything else runs locally before you push:

```bash
scripts/check.sh                        # ruff + format + mypy + pytest (~5 min)
scripts/check.sh --no-tests             # lint + types only (~30 sec)
```

After `scripts/install-hooks.sh`, `git push` runs `scripts/check.sh` automatically. Bypass for WIP pushes with `git push --no-verify`.

If you forget to install the hook, the worst case is a CI failure post-merge that you revert.

## Running tests / lint manually

```bash
# Fast suite (~4 minutes)
uv run pytest

# Slow suite (sympy preprocessing, ~5-10 minutes)
uv run pytest -m slow

# Individual checks
uv run ruff check
uv run ruff format --check
uv run mypy
```

## Pre-release gate

Before tagging a `v*` release, run the local mirror of the cibuildwheel smoke gate:

```bash
scripts/release-precheck.sh             # ~2 min: wheel build + fresh-venv smoke
```

This catches packaging-class bugs (missing runtime deps, broken Cython compile, broken prebuilt imports) that the dev-tree `pytest` misses because dev deps pull everything transitively. Local-green here ≈ rc-tag green on CI.

## Benchmarks

```bash
uv run python scripts/bench_three_parallel.py     # UR5
uv run python scripts/bench_real_jaco2.py         # JACO 2 (RR pipeline)
uv run python scripts/bench_seven_r.py            # synthetic 7R
```

## Pre-built reference artifacts

The `prebuilt/` directory holds committed `.py` artifacts emitted by `ssik build` for popular arms (UR5, Puma 560, JACO 2, iiwa14, Gen3, Franka Panda, Rizon 4, Kassow KR810). They serve as:

1. **User-facing demos** — alpha users can `import prebuilt.ur5_ik` and immediately get a working IK solver.
2. **Codegen-drift snapshot tests** — `tests/test_artifact_snapshots.py` re-emits each artifact and asserts byte-equal against the committed copy.

If you change `ssik.core.codegen` or any solver's dispatch reasoning, the snapshot test will fail. Regenerate with:

```bash
uv run python scripts/regen_artifacts.py                  # fast arms (~30 s total)
uv run python scripts/regen_artifacts.py --include-slow   # also Rizon 4 (~7 min) + Kassow (~20 min)
```

Then commit the updated `prebuilt/*.py` alongside the codegen change so reviewers see the user-facing diff.

## Adding a new arm fixture

```bash
ssik add-arm path/to/arm.urdf --base base_link --ee flange --name my_arm
```

Generates `tests/fixtures/my_arm.urdf` and `tests/test_my_arm.py` with FK-closure assertions on hand-picked + Hypothesis-fuzzed reachable poses. The generated test asserts the dispatcher routing matches the expected solver and that every retained IK FK-closes ≤ 1e-10.

## Adding a new solver

1. New module under `src/ssik/solvers/<family>/<name>.py` with a `solve(kb, T_target, policy, *, max_solutions, ...)` function matching the existing protocol.
2. New dispatcher entry in `ssik.core.dispatcher.dispatch` that classifies which arm topologies route to your solver. Predicate-driven (no per-arm hardcoding); evaluate against a topology test like `is_srs_7r` or `three_consecutive_parallel`.
3. New test file `tests/test_<name>.py` covering:
   - Hand-picked seeded recovery (~5 deliberately-chosen q*, FK to T_target, verify recovery at FK ≤ 1e-10)
   - 500-pose Hypothesis fuzz on a real fixture, FK closure on every retained IK
   - Cross-solver agreement vs an oracle (typically `jointlock + HP` or `ikgeo.general_6r` depending on tier)
4. Update `docs/arm_coverage.md` and `docs/architecture.md`.

## Pull-request guidelines

- Every PR has a clear test plan in the description (which tests demonstrate the change works, what bench numbers look like before/after).
- Profile-driven perf claims only. "X% faster" must come with a benchmark run.
- Negative results are valuable. If you spike something and it doesn't pan out, close the issue with the profile data — don't merge a partial fix that gives 0.3% when the issue advertised 30%.
- No `@pytest.mark.skip` to silence flaky tests. Either fix the test or document the known flake with an issue number.
- Match existing module-docstring style: per-module docstring states the algorithm, the per-arm-constants vs per-call breakdown, and cites the published math.

## License

By contributing, you agree your contributions are released under [BSD-3-Clause](LICENSE).
