# ADR-0001: Python is the compiler; native artifacts are the output

- **Status:** Accepted
- **Date:** 2026-08-01
- **Context issues:** [#482](https://github.com/personalrobotics/ssik/issues/482) (native C++ backend), [#110](https://github.com/personalrobotics/ssik/issues/110) / [#322](https://github.com/personalrobotics/ssik/issues/322) (build-time bake), [#399](https://github.com/personalrobotics/ssik/issues/399) (live/composer parity)

## Context

ssik emits a self-contained analytical-IK artifact per arm. The current artifact
is a Python module; there is demand (ROS MoveIt `kinematics::KinematicsBase`,
OMPL, embedded C++ stacks) for a **native artifact with no Python runtime**. That
raises a recurring question: *if we can emit native code, can we retire the
Python?*

This ADR records the boundary. It exists because getting the boundary wrong —
either reimplementing the toolchain in a systems language, or deleting the
reference solvers once a native path exists — would be an expensive mistake.

The key observation is that ssik is really **two programs**:

1. **A compiler** — offline, developer-facing: URDF/MJCF ingestion (urchin,
   mujoco), POE-normalization, topology dispatch, the sympy symbolic derivations
   (Raghavan-Roth resultant for non-Pieper 6R, Husty-Pfurner elimination for
   non-SRS 7R), and the per-arm bake. None of this runs at deploy time.
2. **A runtime** — the per-arm `solve()` math. Everything it needs at runtime is
   numeric (vectors, trig, eigen/pseudo-inverse); the symbolic work is already
   baked to constants at build time (#110 / #322).

Only the runtime is a candidate for native emission. The compiler is inherently
offline tooling where Python is the right tool, and it is also ssik's
differentiator: the EAIK-gap coverage (non-Pieper 6R, non-SRS 7R) *lives* in the
sympy-based derivation.

## Decision

**Python is the compiler and the reference oracle. Native code (C++, #482) is a
build *output*, not a replacement for the codebase.**

Concretely:

- The build/codegen/ingestion toolchain (urchin, mujoco, sympy, the dispatcher,
  `ssik add-arm`, the manifest, the coverage gates) **stays Python**. We do not
  reimplement symbolic elimination or robot-description parsing in a systems
  language.
- The Python reference solvers **stay** as the parity oracle for native
  artifacts (see #482's set-agreement gate). We do not delete them when a native
  backend lands.
- Native artifacts are emitted from the *same* front-end (dispatch → `SolverSpec`
  + baked constants + sympy-derived expressions) via a pluggable renderer, so
  there is no third copy of solver logic to drift.
- "Retiring Python" is scoped to the **deployment boundary only**: a native
  artifact drops into a C++ stack with zero Python/sympy/mujoco dependency. The
  ssik *codebase* keeps its Python compiler and reference.

This is the IKFast model — a Python/sympy generator that emits standalone C++ —
which has never retired its generator, and shouldn't.

## Consequences

**Positive**

- Native consumers (MoveIt/OMPL) get a Python-free artifact.
- The hard, differentiating tiers (RR/HP) are emitted from their single symbolic
  source (sympy → C), not hand-ported, so they cannot drift from the reference.
- FK-closure is self-validating, so the native path is validated for
  *correctness* without requiring bit-parity with Python — and the Python
  reference remains available as a stronger set-agreement oracle.

**Negative / accepted**

- Two runtime implementations of the *geometric* solver compositions exist
  (Python + the small vendored C++ subproblem library). This is bounded to a
  fixed primitive set (SP1-6 + FK + one Newton + the thin compositions) and made
  safe by the parity gate rather than eliminated (#482).
- The Python package remains a required dependency for *building/onboarding*
  arms. Only *deployed* native consumers are Python-free.

## Non-goals

- Reimplementing the compiler (sympy elimination, URDF/MJCF parsing, dispatch) in
  C++/Rust.
- A C++ core with Python bindings that replaces the Python solver
  implementations — that moves the implementation without retiring Python and
  forfeits the reference oracle.
- A solver IR/DSL rendering both numpy and C++ from one hand-written source; the
  solvers' control flow (branch enumeration, swivel sweeps, root-finding, LM
  loops) makes the payoff not worth inventing a language for (#482).
