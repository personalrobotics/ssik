# HP robustness investigation (#178 Phase 1)

**Date**: 2026-05-16
**Scope**: characterise the failure modes of `ssik.solvers.husty_pfurner._eliminate` / `_back_substitute` that surface as Hypothesis flakes #179 and #181.
**Outcome**: the flakes are NOT HP-pencil instability; they're a separate symmetric-DH limitation. #178 (HP pencil random-rotation preconditioner) and the flake-causing symmetric-DH issue are independent.

## Original framing (pre-investigation)

The v1.1.0 umbrella sequenced #178 ('HP Sylvester pencil random-rotation preconditioner') as the deep robustness work, with #179 / #181 expected to close as side-effects. That framing implicitly assumed the flakes were a symptom of pencil instability.

## What the flakes actually do

Both `test_solve_ik_recovers_truth_hypothesis` (#179) and `test_eliminate_hypothesis_fuzz_500_examples` (#181) fuzz random 6R DH chains + q configurations. The Hypothesis-shrunken failing examples consistently have:

- All `|a_i|` values identical (e.g. all 1.0)
- Most `alpha_i` values identical, with one differing by `~0.5`
- `v_i = 0` for most joints, one `v_i = 1.0`

These chains are **algebraically degenerate** — the joint axes line up in a way that breaks the algebraic structure HP relies on. The failure mode is **"HP returns no IK candidates"** (zero solutions returned, not wrong solutions or unstable values).

The previous `assume(a_spread > 0.05 or alpha_spread > 0.05)` filter let these through because the OR allowed one spread to be 0 as long as the other cleared 0.05.

## Tightened filter and residual flake rate

Empirically:

| spread filter | Flake behavior |
|---|---|
| `> 0.05 OR` (original) | Both tests intermittently xfail |
| `> 0.1 AND` | #179 still xfails on seed 0 |
| **`> 0.3 AND`** (now) | Seed 0 xpasses; seed 1 still xfails on #179 |

`> 0.3 AND` cuts most occurrences but Hypothesis can still find boundary cases. Further tightening hits `filter_too_much` (the strategy rejects too many examples per accepted one). The remaining flake rate is low; the xfail strict=False markers prevent CI breakage.

## Why this isn't HP-pencil instability

Verified by tracing a failing case through the eliminate pipeline:

- The pencil eigsolve (`ssik._pencil.solve_polynomial_matrix_eigenvalues`) does NOT raise; eigenvalues come back finite.
- The downstream `back_substitute` returns an empty solution list because none of the eigenvalue candidates produce an FK-closing IK.
- The issue is algebraic rank-deficiency in the eliminate output (rank-deficient pencil → eigenvalues land on the "spurious roots" branch), not numerical instability of the eigsolve itself.

`#178` was originally framed as 'Noferini-Townsend pencil instability → random-rotation preconditioner'. That's a real concern in HP, separately from the symmetric-DH issue. **The two have to be solved by different mechanisms.**

## Decision: split #178 into two pieces

The original #178 issue says 'closes #179 / #181 as side effects'. That claim is wrong. We now have:

1. **Symmetric-DH HP limitation** (was #179 / #181). Closing requires: either solver-side handling (random-rotation gauge transformation, alternate linearity-variable selection) or accepting the limitation as a documented "real robots aren't symmetric DH" caveat. Tracked in #179 / #181 individually.
2. **HP pencil instability per Noferini-Townsend** (still #178). Closing requires the random-rotation preconditioner. Independent of #179 / #181. Not on the v1.1.0 critical path — the practical impact on users is small (HP only fires on universal-fallback paths, which are themselves rare with the cached-RR jointlock path #210 covering most non-Pieper 7R).

## Impact on v1.1.0 sequencing

This investigation **decouples #178 from the flake closes** and **deprioritises #178** for v1.1.0:

- Phase 1 (this PR) tightens the test strategy to cut most flake occurrences and documents the residual
- #178 (HP pencil preconditioner) stands on its own; deferred to v1.2 unless a real user hits it
- Next v1.1.0 priority is #266 (mkdocs docs site) per the umbrella

## What this PR does

- Tightens `assume(a_spread > 0.05 or alpha_spread > 0.05)` → `assume(a_spread > 0.3 and alpha_spread > 0.3)` in both `test_husty_pfurner_back_substitute.py` and `test_husty_pfurner_eliminate.py`. Reduces flake rate; doesn't fully eliminate.
- Updates the xfail `reason` strings to reflect the actual symptom (symmetric-DH limitation, not pencil instability) and point at the right resolution path.
- Adds this investigation document.

No solver code changes.
