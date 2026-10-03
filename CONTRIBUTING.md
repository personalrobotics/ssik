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
uv run python scripts/build_cpp_ext.py --out-dir src/ssik   # native backend
scripts/install-hooks.sh                # one-time: install pre-push check hook
```

`uv` is the recommended package manager; `pip install -e .[urdf]` works too if you prefer pip.

**Build the native extension.** `native=True` is the default for every shipped
arm, so without `ssik._ssik_native` you are testing the Python fallback, not
what users get: around thirty native tests skip and the perf gate skips too.
`tests/test_native_coverage.py` fails loudly rather than letting that pass
quietly.

**Rebuild compiled modules after the source changes.** `uv sync` compiles the
Cython modules in `hatch_build.py`'s `CYTHON_TARGETS` beside their sources, and
Python imports a compiled module in place of its `.py`. Nothing rebuilds them
when a pull, checkout or edit changes that source, so the build records the
sha256 of each source next to its extension, and in a checkout `import ssik`
refuses an extension whose record is missing or no longer matches, naming the
files. Rebuild with `uv sync --reinstall-package ssik`, or delete the named
files to run the pure-Python source (handy while bisecting). Installed wheels
are not checked: their sources and extensions are built together. Extensions
left by other Python versions (`*.cpython-310-*.so` under 3.13) are never
imported by this one; each version checks its own.

**Eigen is pinned.** Every native build (this one, the wheels on every platform,
and CI) compiles against one Eigen release, pinned by version and sha256 in
`scripts/fetch_eigen.py`. The build downloads it once into `~/.cache/ssik`
(`SSIK_EIGEN_CACHE` moves it) and never falls back to a system Eigen: if the
download fails, `build_cpp_ext.py` stops, and a wheel build leaves the native
extension out with a warning, which the wheel smoke gates reject. Eigen
releases differ numerically (#599: QZ converges on one and not another), so
tests only speak for the wheels when they ran against the same Eigen. To try
another Eigen on purpose, set `SSIK_EIGEN_INCLUDE_DIR=/path/to/eigen3` for the
Python builds, or pass `-DCMAKE_PREFIX_PATH=<your Eigen prefix>` to CMake. The
extension reports what it compiled against as `ssik._ssik_native.eigen_version`.
For the C++ build against the pin:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python3 scripts/fetch_eigen.py --cmake-prefix)" \
  -DSSIK_EIGEN_VERSION="$(python3 scripts/fetch_eigen.py --version)"
```

**Symbolic derivations are cached on disk.** The general-6R solver's
Raghavan-Roth derivation costs ~10-45 s of sympy per arm and linearity choice.
The first process to run one stores it in `~/.cache/ssik/rr-derivations`
(under `$XDG_CACHE_HOME` when set; `SSIK_DERIVATION_CACHE` moves it and
`SSIK_DERIVATION_CACHE=off` disables it), and every later process, including
each pytest-xdist worker, reloads it in under a second. An entry is keyed on the
exact DH, the linearity choice, the source of `_raghavan_roth.py` and the sympy
and mpmath versions, so editing the derivation or upgrading sympy can only miss.
Deleting the directory is always safe. CI's artifact drift guard runs with the
cache off.

## Pre-push gate

CI (`.github/workflows/ci.yml`, on every PR and push to `main`; doc-only changes skip it) takes ~15-25 min. A PR that changes only `examples/` (plus docs, `*.md`, `LICENSE` or `.gitignore`) runs the examples lane instead: lint, `regen_docs.py --check`, `tests/test_teleop.py` and the native wheel jobs, which run every example. Every other change, and every push to `main`, runs the full suite:

- **Linux, Python 3.10-3.14**: ruff, format, mypy, `regen_docs.py --check`, and the fast pytest suite with the native extension built, split into two shards per version plus a check that the shards together ran every test; then the serial perf gates.
- **C++**: the native artifact drift guard and the conformance build, ctest and external-consumer smoke; plus the Python suite reused against the native backend.
- **Wheels**: a native wheel build and smoke on Linux and macOS.

**Exhaustive sweeps run a fixed sample on most PRs.** A few tests (marked `sweep`) check hundreds or thousands of seeded cases: the C++ `feasible_arcs` fuzz, the C++ SRS `resolve_in_limits` parity, the spherical-shoulder and swivel-limits `test_resolves_every_in_limits_pose`, and the native chart parity. `tests/_sweeps.py` lists each sweep's covered code. A PR that changes none of it runs that sweep's PR sample, the first cases of the same seeded stream, so every run checks the same cases. A PR that changes any of it runs the sweep in full, and so do pushes to `main`, `workflow_dispatch` runs, the nightly slow workflow and local runs. `SSIK_PR_SWEEPS` is the switch: a comma-separated list of sweeps to sample, or `all`. Unset or empty means every sweep runs in full. CI's `Classify changed files` job sets it with `scripts/classify_changes.py`. To run the PR tier locally, use `SSIK_PR_SWEEPS=all uv run pytest`. To run only the full sweeps, use `uv run pytest -m sweep`. When you add a sweep, register it in `tests/_sweeps.py` with the paths it covers.

Slow tests (`-m slow`) don't run on PRs; `.github/workflows/slow.yml` runs them nightly on Linux, together with the full sweeps (`-m "slow or sweep"`), and on demand (`gh workflow run slow.yml --ref <branch>`). The native-parity gate (`tests/test_native_parity.py`) runs a fast tier on PRs and its full tier nightly. Its known gaps are strict xfails, one per (arm, gap class): a PR that fixes a class must delete that class's arms from the gate's `KNOWN_*` tables. Boundary poses resolve differently on Linux and macOS, so the tables record the platforms each cell fails on. After changing a solver, refresh both platforms. On macOS, run `scripts/regen_native_parity.py`. For Linux, run `gh workflow run slow.yml --ref <branch> -f regen=true`, then commit the files from its `native-parity-data` artifact. The script prints the merged tables to paste. Run the same checks locally before you push, so CI is a safety net rather than your test loop:

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

Cut a release as `vX.Y.Zrc1` first (it publishes to TestPyPI only), then tag the final `vX.Y.Z` on a commit that carries **no other `v*` tag**. With an rc tag and the final tag on the same commit, the version the release build derives from git may be the rc's; the publish job then refuses to upload (built version != tag) and nothing is released. If the final release has no code change since the last rc, land any commit on `main` first (a docs change is enough) and tag that.

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

2. **Paste the stanza into `MANIFEST.toml`** (or pass `--write-manifest` to `add-arm` to append it automatically; it refuses to clobber an existing entry unless `--force`). `add-arm` derives most fields: `solver` / `tier` / `dof` / `platform_drift` / `fk_ceiling_fuzz` from dispatch + the validation smoke, `kinematic_class` / `short_class` / `class_tags` from the solver, `sample_q` from a verified-solvable pose, and a best-effort `display_name`. The only fields left to hand-fill are `fixture_source` (repo + license provenance) and confirming the marketing `display_name` / `short_name`. `regen_bench.py` fills `[bench]` / `[eaik]`.

3. **Emit the artifact:**

   ```bash
   uv run python scripts/regen_artifacts.py --arm <name>_ik    # build just this arm (fast; also refreshes the flat-alias map)
   # or: uv run python scripts/regen_artifacts.py [--include-slow]   # rebuild the whole roster
   ```

4. **Bench + regenerate all docs — one click** (#341):

   ```bash
   uv run python scripts/regen_bench.py --arm <name>_ik --docs   # fills [bench] + rewrites doc tables
   ```

   `regen_bench.py` measures the prebuilt's `solve()` (time / FK / branch count, over the same reachable poses it gives EAIK) and writes the `[arms.<name>_ik.bench]` block in place, then runs `regen_docs.py`. Run it on the reference machine (timing is machine-dependent; FK/sols are not). Omit `--arm` to re-bench every arm.

   Then refresh the perf-regression baseline (a new arm without one fails `test_perf_regression.py::test_baseline_covers_benched_arms`):

   ```bash
   uv run python scripts/regen_perf_baseline.py   # writes tests/_perf_baseline.json (solve time relative to ur5)
   ```

   The baseline stores each arm's solve time *relative to ur5* (machine-independent), so the perf gate holds across CI runners. Regenerate it on the reference machine only when a solver change legitimately alters timing, never to silence a gate failure you don't understand.

5. **Set `fk_ceiling_fuzz`** to ~10× the worst FK residual in a 50-pose smoke test. If a pose returns no IK (coverage gap), add a `[arms.<name>_ik.known_gaps]` block with an `xfail_reason` and file an issue. Then run the gates:

   ```bash
   scripts/check.sh        # ruff + format + mypy + regen_docs --check + pytest
   ```

`tests/test_manifest.py` cross-validates every manifest entry against the emitted artifact's baked constants (solver / base / ee / dof), so a typo surfaces immediately.

## Adding a new arm fixture (test scaffold only)

```bash
ssik add-arm path/to/arm.urdf --base base_link --ee flange --name my_arm
```

Strips the source URDF to a kinematics-only `tests/fixtures/my_arm.urdf` (no meshes), and generates a **coverage-gated** `tests/test_my_arm.py` (auto-formatted with `ruff`). The scaffold asserts, against the **shipped artifact** (`module.solve` / `module.fk`, not the raw solver): topology + dispatcher routing, a **coverage floor** (≥ `_MIN_COVERAGE` of poses sampled across the arm's real joint limits return ≥ 1 IK — the gate a broken/degenerate arm fails, since "FK ≤ tol on every retained IK" is vacuously true at zero coverage), and FK closure within `_FK_CEILING`. It also **builds the emitted artifact in a temp dir and runs the coverage gate right there** (unless `--no-validate`), printing a `✓ VALIDATED` / `✗ VALIDATION FAILED` verdict — so a broken arm (0 solutions, `[0,0]`-locked limits, wrong ee) is caught at onboarding, not a CI round-trip later. Use `--no-validate` for slow cached-RR 7R arms. `--base`/`--ee` are auto-detected when omitted. Finally it **prints a ready-to-paste `MANIFEST.toml` stanza** with `solver`/`tier`/`dof`, a derived `fk_ceiling_fuzz` (~10× the measured worst FK), and `platform_drift` set by solver class already filled. To ship, paste the stanza, build with `regen_artifacts.py --arm <name>_ik`, and continue from step 3 of "Adding a new prebuilt arm" above.

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

## Stacked PRs and squash merges

PRs merge by squash, so when a PR is stacked on another and the parent merges, the child branch still carries the parent's original commits. Git sees them as different from the squashed commit on `main`, and the child conflicts even though nothing really changed.

Merge in this order:

```bash
gh pr edit <child-pr> --base main      # 1. retarget the child BEFORE merging the parent
gh pr merge <parent-pr> --squash --delete-branch   # 2. merge the parent
git fetch origin                       # 3. replay only the child's own commits onto main
git rebase --onto origin/main <parent-old-head> <child-branch>   # <parent-old-head>: the parent's last commit before the squash
git push --force-with-lease origin <child-branch>
```

Step 1 matters: if the parent's branch is deleted while the child still targets it, GitHub closes the child PR, and it can't be reopened until that base branch exists again. To recover, push the parent's old head back to its branch name (`git push origin <parent-old-head>:refs/heads/<parent-branch>`), `gh pr reopen <child-pr>`, `gh pr edit <child-pr> --base main`, rebase as in step 3, then delete the parent branch again.

Force-push only your own feature branches, never `main`. Don't merge `main` into the child instead: it leaves a noisy merge commit and triggers an extra full CI run.

## License

By contributing, you agree your contributions are released under [BSD-3-Clause](LICENSE).
