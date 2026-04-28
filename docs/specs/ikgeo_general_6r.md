# Spec: `ikgeo.general_6r` — numeric Raghavan-Roth 6R solver

**Status:** draft, pre-implementation. Review before any code lands.
**Strategic role:** the universal-6R closed-form-style solver that closes the EAIK
coverage gap (JACO 2 classical, Agilex Piper, custom non-Pieper 6R; via
`jointlock.seven_r`, also Rizon 4 and other non-SRS 7R arms). Replaces the current
grid-search `gen_six_dof` as the production tier-2 path; the grid version is
retained, vectorized and renamed `gen_six_dof_oracle`, as the cross-check.

## Algorithm — clean-room from Tsai 1999 App. C + Manocha-Canny 1994

References (open-access PDFs cached in `/tmp/rrk/`):
- **Tsai App. C** (METR4202 OCR) — derivation of the 14-equation system, the
  6×9 / 14×8 split, elimination of (q₁, q₂), substitution of half-angles,
  construction of the 12×12 matrix `M(x₃)`.
- **Manocha-Canny 1994** (Berkeley preprint) — Theorem 1 (24×24 companion
  matrix), Theorem 2 (generalized-eigenvalue fallback), Möbius
  reparameterization, eigenvector → (x₄, x₅), Newton refinement.

LGPL note: `vendored/.../ikfast.py:solveDialytically` implements this
construction. Read for *algorithmic existence proof only* — do not copy variable
names, control flow, or comments. Algorithms aren't copyrightable; the
RR algorithm predates IKFast by 17 years.

### Step 1 — DH normalization
Input: POE-normalized `KinBody` with 6 revolute joints. Convert to standard DH
parameters `(αᵢ, aᵢ, dᵢ)` for `i ∈ 1..6` by reading off `T_left`, axes, and
joint origins. (POE → DH conversion already exists in
`ssik.kinematics.poe`; verify it covers the non-orthogonal-twist case used by
JACO 2's 55° offsets.)

### Step 2 — Build the 14×9 / 14×8 system numerically
Substitute target pose `T_target` and DH params into the closed-form expressions
for `P` (14×9, entries linear in `s₃, c₃, 1`) and `Q` (14×8, entries constant per
solve) from Tsai Eq. (C.8) / MC Eq. (4). The 14 rows = 6 from columns 3,4 of
the matrix loop-closure equation + 8 from the (a·a, a·b, a×b, (a·a)b−2(a·b)a)
vector identities. Code this once symbolically (sympy at module import) and
emit a NumPy callable `build_PQ(dh, T_target) -> (P, Q)`.

### Step 3 — Eliminate (q₁, q₂)
SVD-rank `Q` (Manocha-Canny §IV-B). If `rank(Q) = 8`, do Gaussian elimination
with complete pivoting to write the 8 right-side monomials in terms of the 9
left-side monomials, leaving 6 equations in `(s₄, c₄, s₅, c₅)` only. If
`rank(Q) < 8` (degenerate arm — e.g. Puma with `d₆ = 0` collapses to ≤8
solutions), drop ill-conditioned rows; downstream still handles it.

### Step 4 — Substitute half-angles, build `M(x₃)`
Apply Weierstrass `sᵢ = 2xᵢ/(1+xᵢ²), cᵢ = (1−xᵢ²)/(1+xᵢ²)` for `i = 4, 5`,
clear `(1+x₄²)(1+x₅²)`. Apply for `i = 3`, multiply first 4 rows by `(1+x₃²)`.
Stack `[[E'', 0], [0, E''·x₄]]` to get the 12×12 polynomial matrix
`M(x₃) = A·x₃² + B·x₃ + C`. Each of A, B, C is a fully-numeric 12×12 matrix.
The 12-monomial vector is
`v = (x₄²x₅², x₄²x₅, x₄², x₄x₅², x₄x₅, x₄, x₅², x₅, 1, x₄x₅², x₄x₅, x₄)`
(verify exact ordering against Tsai Eq. C.13–C.15 + MC Eq. 18 on first
implementation pass).

### Step 5 — Eigenvalue route (MC Theorem 1)
Compute `cond(A)`. If well-conditioned (rule of thumb: `cond(A) < 1e10`):
build `Σ = [[0, I₁₂], [−A⁻¹C, −A⁻¹B]]` (24×24), call `numpy.linalg.eig(Σ)`.
24 eigenvalues; **drop 8 spurious roots near `±i`** (multiplicity 4 each,
from `(1+x₃²)⁴` factor); the remaining 16 are the candidate `tan(q₃/2)`.
Filter complex roots by `|imag| < max(|real|, 1) · ε` (scale-aware, same
pattern as SP5 cluster filter).

### Step 6 — Conditioning fallback (MC §IV-C, Theorem 2)
If `cond(A) > 1e10`, attempt Möbius reparameterization
`x₃ = (a·x̃₃ + b)/(c·x̃₃ + d)` with random `(a, b, c, d)` (try ≤3
random draws, keep the one with smallest `cond(A_new)`). Rebuild
`A_new, B_new, C_new` via the linear transformation in MC Eq. (17); if
well-conditioned, eigenvalue route on the transformed matrix and apply
inverse Möbius to recover `x₃`. If still ill-conditioned (singular pencil —
extremely rare; only when `(A, B, C)` share a common null space): fall
through to `scipy.linalg.eig(M₁, M₂)` generalized-eigenvalue path
(Theorem 2). 2.5–3× slower; flag in diagnostics.

### Step 7 — Back-substitution (MC §IV-C/D)
Each eigenvector of Σ has structure `V = [v; x₃·v]`. Per root:
- Pick the top half if `|x₃| ≤ 1`, bottom half otherwise (smaller relative
  error — MC Eq. 15 footer).
- Recover `(x₄, x₅)` as ratios of two entries of `v`. Use entries with
  largest magnitudes for numerator/denominator. Cross-check via redundant
  ratios (e.g. `v[5]/v[8] = x₄`, `v[1]/v[5] = x₅`).
- Recover `(q₁, q₂)`: solve the 8-row linear system from Eq. (11) (Tsai
  Eq. C.7's right block) for the 8 monomials `{s₁s₂, s₁c₂, c₁s₂, c₁c₂,
  s₁, c₁, s₂, c₂}`. Two angles via `atan2(s₁c₂, c₁c₂)` etc.
- Recover `q₆` via atan2 on row entries of the original loop-closure equation
  Eq. (3) once `q₁..q₅` are known.

### Step 8 — Newton refinement (MC §V-E)
For each candidate `q ∈ ℝ⁶`, run 1–2 Newton steps on the 14-equation residual
(Tsai Eq. C.8) — this lifts ~6 digit eigenvalue precision to ~10–11 digits.
Step clipped to `π/4`, wrap-to-pi each step (same pattern as SP5/SP6 GN).

### Step 9 — Verification
FK each refined `q` via the existing `_kinbody`-aware FK, compare to
`T_target` Frobenius norm. Drop any `q` failing `‖FK(q) − T_target‖ <
policy.subproblem_numerical`. Dedup with the standard wrap-to-pi
`q_close` predicate.

## Performance budget

- Step 2 (build P, Q): ~50 µs after sympy precompute (per-arm, cached).
- Step 5 (`np.linalg.eig` on 24×24): ~30 µs in NumPy.
- Step 7 (back-sub × 16 roots): ~200 µs (linear solves are tiny).
- Step 8 (Newton × 16): ~500 µs.
- **Total: ~1–3 ms per IK** in pure Python. ~20× faster than the symbolic
  IKFast path. After Phase M codegen: ~100 µs.

## Validation plan

Bulletproof discipline — same standard as `spherical_two_parallel`:

1. **Synthetic 16-solution fixture** — MC's Table I example (DH params + 16
   real solutions). Hand-verified ground truth. Assert `len(solutions) == 16`,
   each FK-matches at `1e-10`.
2. **Real-arm fixtures**: JACO 2 classical (55° non-orthogonal DH), Agilex
   Piper (mujoco_menagerie URDF). Assert FK-closure on 100 random poses.
3. **Cross-check vs vectorized `gen_six_dof_oracle`** on 500 hypothesis poses
   per arm. Solver-set agreement: every solution from one must appear in
   the other (within 1e-6 wrap-to-pi). Catches missing branches.
4. **Pieper-class regression**: run on UR5, Puma 560 (where tier-0 already
   solves). Solution sets must match `three_parallel` /
   `spherical_two_parallel` at machine precision. Catches algorithm bugs
   that only manifest on degenerate `Q` rank.
5. **Hypothesis fuzz**: 500 random POE-normalized 6R chains × random poses.
   FK-closure on every returned solution.
6. **Conditioning stress**: poses near `q₃ = π` (drives `x₃ → ∞`); confirm
   Möbius reparameterization recovers, no NaN/Inf leakage.

## Risks and mitigations

- **Q-rank degeneracy on Pieper arms.** Puma's `Q` has rank ≤7 for some
  poses; rare but real. Mitigation: SVD with explicit rank threshold
  matching MC §IV-B.
- **Möbius reparameterization fails on singular pencils.** MC reports this
  is rare; we fall through to generalized eigenvalue. Worst-case ~10 ms.
- **POE → DH conversion correctness.** Already shipped in `kinematics.poe`,
  but the JACO 2 fixture will be the first non-orthogonal-twist exercise;
  audit before relying on it.
- **Newton non-convergence.** Cap at 5 iterations; if residual still
  `> 1e-6`, drop the candidate and flag in diagnostics.

## Files to create

- `src/ssik/solvers/ikgeo/general_6r.py` — solver module.
- `src/ssik/solvers/ikgeo/_raghavan_roth.py` — private: `build_PQ`,
  `eliminate_q1_q2`, `build_M_matrix`, `extract_x4_x5`, `back_substitute`.
  Sympy used at module import for symbolic `P, Q` derivation; pure NumPy at
  runtime.
- `tests/test_solver_general_6r.py` — full bulletproof suite per validation
  plan above.
- `tests/fixtures/jaco2.py`, `tests/fixtures/agilex_piper.py` — DH /
  mujoco_menagerie URDFs.

## Files to modify

- `src/ssik/solvers/ikgeo/gen_six_dof.py` → rename module to
  `gen_six_dof_oracle.py`. Vectorize the inner SP5 loop (separate PR
  after this one lands; ~10× faster, useful as the validation oracle).
- `SUPPORTED_ROBOTS.md` — promote JACO 2, Piper, Rizon 4 to
  ✅-with-tier-2 / ✅-with-jointlock.

## Out of scope (track as separate issues)

- Per-robot codegen (Phase M). Numeric RR's runtime is the constraint;
  codegen drops it to ~100 µs but isn't blocking v0.1.
- Husty-Pfurner alternative path (paywalled paper; deferred).
- Li-Woernle-Hiller cross-check oracle (Angeles 2014 §9; nice-to-have).

## Estimated effort

5–7 days, broken roughly:
- Day 1–2: sympy derivation of `P, Q` symbolic forms + numerical
  `build_PQ` callable + unit tests against MC Table I.
- Day 3: matrix construction, eigenvalue route, back-substitution.
- Day 4: conditioning fallback (Möbius + generalized-eigenvalue).
- Day 5: Newton refinement, verification, dedup.
- Day 6–7: bulletproof validation suite, JACO 2 + Piper fixtures, cross-solver
  agreement on 500 hypothesis poses, ship PR.
