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

## The prebuilt-arm manifest (`src/ssik/prebuilt/MANIFEST.toml`)

`MANIFEST.toml` is the **single source of truth** for prebuilt-arm metadata. Every shipped arm has one entry; the schema is documented at the top of the file. Consumers that read from it (so you never hand-edit them per arm):

- `tests/test_prebuilt_sanity.py`, `tests/test_prebuilt_uniform_fuzz.py`, `tests/test_artifact_snapshots.py` — per-arm parametrisation, FK ceilings, known-gap xfails, platform-drift markers
- `scripts/regen_artifacts.py` — which arms to emit + their `--include-slow` gating
- `scripts/regen_bench.py` — measures each prebuilt and writes the `[arms.*.bench]` blocks in place
- `scripts/regen_docs.py` — the AUTOGEN doc tables (README prebuilt + EAIK comparison, docs/quickstart, src/ssik/prebuilt/README)
- `examples/04_compare_vs_eaik.py` — bench fixtures

Doc tables wrapped in `<!-- AUTOGEN:name --> ... <!-- /AUTOGEN -->` are generated from the manifest. **Never edit inside those markers by hand** — CI's drift gate (`scripts/regen_docs.py --check`) will reject it. Edit `MANIFEST.toml`, then:

```bash
uv run python scripts/regen_docs.py        # rewrite the anchored tables
uv run python scripts/regen_docs.py --check # verify in sync (CI runs this)
```

## Pre-built reference artifacts

The `prebuilt/` directory holds committed `.py` artifacts emitted by `ssik build` for the arms in `MANIFEST.toml`. They serve as:

1. **User-facing demos** — alpha users can `import prebuilt.ur5_ik` and immediately get a working IK solver.
2. **Codegen-drift snapshot tests** — `tests/test_artifact_snapshots.py` re-emits each artifact and asserts byte-equal against the committed copy.

If you change `ssik.core.codegen` or any solver's dispatch reasoning, the snapshot test will fail. Regenerate with:

```bash
uv run python scripts/regen_artifacts.py                  # fast arms (~30 s total)
uv run python scripts/regen_artifacts.py --include-slow   # also slow_build arms (Rizon 4 / 10 ~7 min, Kassow ~20 min)
```

Then commit the updated `prebuilt/*.py` alongside the codegen change so reviewers see the user-facing diff.

## Adding a new prebuilt arm

The metadata for a shipped arm lives in `MANIFEST.toml`. The flow is now mostly tooled:

1. **Vendor the fixture + scaffold + get a manifest stanza** with `ssik add-arm`. It strips the source URDF to kinematics-only (`tests/fixtures/<name>.urdf`, no meshes — via `ssik._urdf.strip_urdf_to_fixture`, also exposed as `scripts/strip_urdf_fixture.py`), generates a bulletproof test scaffold, and **prints a ready-to-paste `[arms.<name>_ik]` stanza** with the dispatcher-derived fields (`solver`, `tier`, `dof`) filled and curated fields (`display_name`, `kinematic_class`, …) left as `TODO`:

   ```bash
   ssik add-arm path/to/arm.urdf --base <base> --ee <ee> --name <name>_ik
   ```

   (Spec-transcribed arms still use a Python `*_specs()` builder + `fixture_kind = "specs"`. Xacro: expand to plain URDF first — modular-URDF support is #327.)

2. **Paste the stanza into `MANIFEST.toml`** and fill the `TODO` fields (copy a same-class neighbour for `kinematic_class` / `class_tags`).

3. **Emit the artifact:**

   ```bash
   uv run python scripts/regen_artifacts.py [--include-slow]   # emits src/ssik/prebuilt/<name>_ik.py
   ```

4. **Bench + regenerate all docs — one click** (#341):

   ```bash
   uv run python scripts/regen_bench.py --arm <name>_ik --docs   # fills [bench] + rewrites doc tables
   ```

   `regen_bench.py` measures the prebuilt's `solve()` (time / FK / branch count, examples/04 methodology) and writes the `[arms.<name>_ik.bench]` block in place, then runs `regen_docs.py`. Run it on the reference machine (timing is machine-dependent; FK/sols are not). Omit `--arm` to re-bench every arm.

5. **Set `fk_ceiling_fuzz`** to ~10× the worst FK residual in a 50-pose smoke test. If a pose returns no IK (coverage gap), add a `[arms.<name>_ik.known_gaps]` block with an `xfail_reason` and file an issue. Then run the gates:

   ```bash
   scripts/check.sh        # ruff + format + mypy + regen_docs --check + pytest
   ```

`tests/test_manifest.py` cross-validates every manifest entry against the emitted artifact's baked constants (solver / base / ee / dof), so a typo surfaces immediately.

## Adding a new arm fixture (test scaffold only)

```bash
ssik add-arm path/to/arm.urdf --base base_link --ee flange --name my_arm
```

Strips the source URDF to a kinematics-only `tests/fixtures/my_arm.urdf` (no meshes), generates `tests/test_my_arm.py` with FK-closure assertions on hand-picked + Hypothesis-fuzzed reachable poses (asserting dispatcher routing + FK ≤ 1e-10 on every retained IK), and **prints a ready-to-paste `MANIFEST.toml` stanza**. It does **not** build or bench the arm — to ship it as a prebuilt, paste the stanza and continue from step 3 of "Adding a new prebuilt arm" above.

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
