# Semantic versioning policy

ssik follows [SemVer 2.0](https://semver.org/spec/v2.0.0.html): `MAJOR.MINOR.PATCH` where MAJOR is breaking, MINOR is additive, PATCH is bug-fix only.

## What's public, what's not

### Public (covered by semver)

- **Top-level imports**: `ssik.Manipulator`, `ssik.Solution`, `ssik.Diagnostic`, `ssik.TolerancePolicy`, `ssik.DEFAULT_TOLERANCE_POLICY`
- **Postprocess helpers**: `ssik.postprocess.{respect_limits, wrap_to_limits, nearest_to_seed, take_first}`
- **Prebuilt artifacts** (`ssik.prebuilt.*`): their `solve(T, **kwargs)` signature, the four module constants `BASE_LINK / EE_LINK / DOF / T_HOME`, and the per-module `__all__`
- **CLI**: `ssik build / classify / add-arm` argument shapes
- **Wheel manifest**: support for cp311 / cp312 / cp313 × Linux x86_64 / macOS arm64 / macOS x86_64 / Windows x86_64
- **C++ headers for native consumers** (see below): `ssik.get_include()`, `ssik.get_cmake_dir()`, `ssik.cpp.joint_data()` / `ssik.cpp.JointData`, and a named part of the shipped `ssik_cpp` headers

### C++ surface (since 6.2.0)

Every wheel ships the `ssik_cpp` headers. Semver covers only this part of them:

- **Package**: the CMake package `ssik_cpp` (from `ssik.get_cmake_dir()` or `cmake --install` of `cpp/`), its target `ssik::ssik_cpp`, and the include path `ssik_cpp/solvers/three_parallel.hpp`.
- **Entry point**: `ssik::three_parallel_artifact_solve(const JointConsts<6>&, const JointLimits<6>&, const Pose&, const ArtifactParams<6>&)`, for the UR family (`ikgeo.three_parallel`). With the arrays of `ssik.cpp.joint_data()` it returns the set Python's `solve()` returns with the same options, under the rules above.
- **Types it takes and returns**, in namespace `ssik`: `Pose`, `JointConsts<N>`, `JointType`, `JointLimits<N>`, `ArtifactParams<N>`, `SeedMetric`, `Solution<N>`, `Refinement`. Their names, field names, field types and meaning, enumerator values, and the `ArtifactParams` defaults are covered. Adding a field with a default is MINOR.

Not covered: the other families' `*_artifact_solve` functions (each is added here once a consumer has verified it the same way), the defaulted trailing `Tolerances` argument (pass none), every `detail` and `*_detail` namespace, every other function, type and header in `ssik_cpp/`, the generated `cpp/gen/<arm>_ik.hpp` artifacts, and binary compatibility (the headers are compiled into your code, so rebuild when ssik changes). Eigen is the consumer's; ssik tests against the release pinned in `scripts/fetch_eigen.py` (3.4.0).

### Not public (no semver guarantee)

- Anything in `ssik.solvers.*` (use `Manipulator.solve` or a prebuilt)
- Anything in `ssik.core.*` except names explicitly re-exported from the top level
- `ssik.kinematics.*` (POE primitives, predicates, reverse-chain math)
- `ssik.subproblems.*` (SP1–SP6 implementations)
- `ssik.refinement.*` (LM polish internals)
- `ssik.codegen.*` (artifact emission internals)
- Leading-underscore modules: `ssik._kinbody`, `ssik._urdf`, `ssik._pencil`, `ssik._version`
- The `_solve_algebraic`, `_KB`, `_LOCK_IDX`, `_LOCK_SAMPLES`, `_DISPATCH_CACHE` etc. in prebuilt artifacts (private to the codegen). For an artifact's geometry — its `KinBody`, its self-motion charts — the supported route is `Manipulator.from_prebuilt("<arm>")`, whose `solve` is the artifact's own solver (#587)

If a downstream depends on a non-public name, that's a "fragile dependency" the user owns; we may rename or remove it in any release.

## What counts as breaking

A change is **MAJOR** (breaking) when:

- A public symbol is removed, renamed, or moved to a different module
- A public function's signature changes in a non-backward-compatible way (positional arg order, removed kwarg, kwarg name change)
- A returned dataclass loses a field, or an existing field changes type
- A bug fix that produces materially different IK behaviour for valid inputs (e.g. a previously-returned candidate is now filtered, or a previously-empty result is now populated). *Sub-machine-precision FK shifts within the documented tolerance policy are NOT breaking.*
- The CLI's `--flag` semantics change in a way that breaks scripts pinning a major
- Python version support drops (e.g. dropping cp311)
- Wheel-platform support drops in a way that affects existing installs (e.g. dropping macOS arm64)

A change is **MINOR** (additive) when:

- A new top-level symbol is added
- A new optional kwarg with a backward-compatible default is added
- A new `Solution` / `Diagnostic` field is added (existing code that doesn't reference the new field keeps working)
- A new prebuilt arm is added to `ssik.prebuilt`
- A new platform is added (e.g. Linux aarch64 wheels)
- A new CLI subcommand or flag with a backward-compatible default is added

A change is **PATCH** when:

- Bug fixes that don't change documented behaviour
- Numerical precision improvements within the documented tolerance policy
- Performance improvements
- Documentation corrections
- Internal refactors that don't touch the public surface

## Build artifacts

The codegen-emitted artifacts under `ssik.prebuilt` (and any user-built `<arm>_ik.py`) are byte-stable across same-version regenerations (enforced by `tests/test_artifact_snapshots.py`). Across versions:

- **PATCH bump**: artifacts MAY differ byte-wise (e.g. updated docstring text, tighter constants) but the public surface (`solve`, `fk`, `BASE_LINK`, `EE_LINK`, `DOF`, `T_HOME`) stays compatible.
- **MINOR bump**: same, plus new optional `solve()` kwargs may appear.
- **MAJOR bump**: artifacts may need re-running `ssik build` against your URDF.

User-emitted artifacts are frozen at the ssik version that built them. They keep working with future ssik versions — just won't pick up later solver improvements until you re-run `ssik build`.

## Release-tag conventions

- `v1.0.0`, `v1.1.0`, `v1.2.3` etc. are real releases (published to PyPI).
- `v1.1.0rc1`, `v1.1.0rc2` etc. are release candidates published to TestPyPI for validation before promotion to PyPI.
- Annotated tags only (`git tag -a`); the release workflow needs the tag annotation for hatch-vcs version derivation.

## Deprecation policy

- A public symbol marked deprecated in version `X.Y.Z` is removed no earlier than version `X+1.0.0`.
- Deprecations are surfaced via `DeprecationWarning` at the first use of the symbol in a Python session.
- Migration paths are documented in the release notes and in the symbol's docstring.

## Reporting breakage

If a PATCH or MINOR release breaks your code, that's a semver bug — file an issue with a minimal reproducer. A fix or a clarification on what's actually public will follow.
