# 5. Conditioning is the hard part

The Raghavan–Roth pipeline of [Chapter 4](04_raghavan_roth.md) is a sequence of linear-algebra primitives. Run it on a textbook 6R arm with well-separated DH parameters and you get machine-precision IK. Run it on a real arm — Kinova JACO 2 with 60° non-orthogonal twists at joints 4–5 — and the textbook algorithm produces garbage: candidates with FK residual $\sim 10^{0}$ instead of $\sim 10^{-13}$, an algebraic precision floor 12 orders of magnitude looser than expected.

This chapter explains why, and walks through the four independent fixes ssik ships to make the pipeline survive on real arms. The fixes are tagged **AE-1** through **AE-6** in the codebase and issue tracker — analytical-exhaustion measures meant to wring every order of magnitude of conditioning improvement out of the algebraic structure before the algorithm gives up and falls back to numerical refinement.

## The failure mode: a singular pencil

Stage 4 of the pipeline builds the matrix pencil

$$
M(x_2) = A x_2^2 + B x_2 + C, \qquad A, B, C \in \mathbb{R}^{12 \times 12},
$$

and Stage 5 linearises to the $24 \times 24$ companion

$$
\Sigma = \begin{bmatrix} 0 & I \\ -A^{-1} C & -A^{-1} B \end{bmatrix}.
$$

The construction requires $A$ to be **invertible**. If $A$ is rank-deficient, $A^{-1}$ doesn't exist and the companion construction fails. If $A$ is invertible but **ill-conditioned** (large $\kappa(A) = \sigma_{\max}/\sigma_{\min}$), the inverse exists numerically but amplifies floating-point noise: every digit of $\kappa$ above $10^0$ is roughly a digit of precision lost in $A^{-1} B$ and $A^{-1} C$. With double-precision floats giving us 16 digits of machine epsilon, $\kappa(A) > 10^{16}$ means the construction has zero correct digits.

On JACO 2 with the textbook leftvar choice (linearity = $q_2$, the convention in Manocha–Canny 1994), measured values are:

| metric | textbook (linearity $= q_2$) | what we want |
|---|---|---|
| $\kappa(A)$ | $3.75 \times 10^{16}$ | $\le 10^{10}$ |
| algebraic FK error (median) | $\sim 10^{0}$ | $\sim 10^{-13}$ |
| pose-level success rate | $\sim 5\%$ | $100\%$ |

The textbook algorithm is unusable on this arm. The cause is a **singular pencil**: the matrices $A, B, C$ for this geometry happen to share a common kernel direction, so the pencil $(A, B, C)$ as a whole is structurally ill-conditioned regardless of how we factor it.

The next four sections describe the four independent attacks ssik ships against this. Each one peels off some of the conditioning damage; together they bring JACO 2 from $\kappa = 10^{16}$ to $\kappa \approx 100$ and FK error from $10^0$ to $10^{-13}$.

## AE-1: Equilibration of the pencil

[Issue #68.](https://github.com/siddhss5/ikfastpy/issues/68) Implementation: [`_equilibrate_pencil`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py).

**The problem.** Real arms have DH parameters spanning many orders of magnitude — JACO 2 has $a_2 = 0.41$ m alongside $\alpha_4 = 60° = 1.047$ rad. The entries of $A, B, C$ inherit that scale variance: some rows have entries $\sim 10^{-3}$, others $\sim 10^{0}$, and $\kappa(A)$ inflates because LAPACK can't tell row scaling apart from genuine ill-conditioning.

**The fix.** Joint row + column scaling: rescale each row of $(A, B, C)$ jointly so its max-magnitude entry is 1, then rescale each column similarly. The quadratic eigenvalue problem $(A x^2 + B x + C) v = 0$ and the equilibrated $(D_l A D_r) x^2 + (D_l B D_r) x + (D_l C D_r)$ have **the same eigenvalues** $x$; eigenvectors transform as $v = D_r v_{\mathrm{eq}}$. Eigenvalues are scale-invariant, so equilibration costs nothing in correctness.

**The win on JACO 2.** Modest — about $2\times$ reduction in $\kappa(A)$ on most poses. Equilibration is necessary but not sufficient. The structural pencil-singularity needs more than scaling to fix.

IKFast doesn't do this; we put it in because the cost is ~10 µs per pose and the win is consistent.

## AE-3: Leftvar selection — the structural fix

[Issue #70.](https://github.com/siddhss5/ikfastpy/issues/70) Implementation: [`pick_best_leftvar`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py), [`_cached_best_leftvar`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py).

**The intuition.** The Raghavan–Roth loop split — $A_2 A_3 A_4 = A_1^{-1} T A_5^{-1} A_6^{-1}$ — was a **choice**. We chose joint 2 as the linearity variable (the joint whose $\sin / \cos$ enter as linear coefficients of the $v_{\mathrm{left}}$ basis), with $(q_3, q_4)$ as the bilinear pair on the left side and $(q_0, q_1)$ as the bilinear pair on the right. The cycle $\{q_0, q_5\}$ on the ends is fixed by the standard chain ordering, but the choice of which interior joint is the linearity variable — $q_0$, $q_1$, or $q_2$ — is free. Three choices, three different splits, three different $A, B, C$ matrices, three different conditionings.

**The diagnostic.** On JACO 2, the three options give:

| linearity choice | $v_{\mathrm{left}}$ pair | $v_{\mathrm{right}}$ pair | $\kappa(A)$ |
|---|---|---|---|
| $q_0$ (interior, base end) | $(q_1, q_2)$ | $(q_4, q_5)$ | $\sim 10^{14}$ |
| $q_1$ (interior, middle) | $(q_2, q_3)$ | $(q_0, q_5)$ | **127** |
| $q_2$ (textbook MC choice) | $(q_3, q_4)$ | $(q_0, q_1)$ | $3.75 \times 10^{16}$ |

Picking $q_1$ as the leftvar drops $\kappa(A)$ by **14 orders of magnitude**. On UR5 (a Pieper arm we don't actually need this solver for, but useful as a baseline) all three choices give similar conditioning around $10^4$.

**Why this works.** The pencil $(A, B, C)$ becomes ill-conditioned when the 60° pathological joints (joints 4 and 5 on JACO 2) sit in the $v_{\mathrm{left}}$ bilinear pair. With linearity = $q_2$, $v_{\mathrm{left}}$ is $(q_3, q_4)$ — joint 4 is in there. With linearity = $q_1$, $v_{\mathrm{left}}$ is $(q_2, q_3)$ — pathological joints 4 and 5 are now on the $v_{\mathrm{right}}$ side, where they get eliminated by the SVD of $Q$ in Stage 2 and never enter the $A, B, C$ construction. **The structural pathology is moved out of the matrix that needs to be invertible.**

Stated as a heuristic: **pick the leftvar that keeps your arm's structurally-bad joints out of the $v_{\mathrm{left}}$ bilinear pair**.

**The selector.** ssik tries all three leftvar choices at module-import time (or first IK call) for a given DH tuple, builds $A, B, C$ for each, computes $\kappa(A)$, and picks the lowest-$\kappa$ option. The choice is cached per arm via [`_cached_best_leftvar`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py) — geometry-driven, not pose-driven. After AE-3, the JACO 2 algebraic FK error drops from $\sim 10^{0}$ to $\sim 10^{-13}$ on 96.6% of poses.

This is **the** structural fix. AE-1 (equilibration) softens scale variance; AE-3 chooses a fundamentally better problem to solve.

## AE-4: SO(3) reduction

[Issue #71.](https://github.com/siddhss5/ikfastpy/issues/71)

**The problem.** The 6×9 eliminated system $E$ from Stage 2 contains rows that come from rotation-matrix entries (the $A_2 A_3 A_4$ rotation block has 9 entries), and the $\mathrm{SO}(3)$ orthogonality identities ($R^T R = I$, $\det R = 1$) couple them in ways the algebra doesn't exploit. Some rows of $E$ are nearly-redundant scaled copies of others, inflating $\kappa(A)$.

**The fix.** Apply the $\mathrm{SO}(3)$ identities up front: instead of treating the 9 rotation entries as independent, use the row + column orthogonality to reduce. After this reduction the matrix entries are smaller and less correlated.

**The win on JACO 2.** About $3.4\times$ reduction in $\kappa(A)$ — modest, additive on top of AE-3.

## AE-3 + AE-1 + AE-4 together

On JACO 2 with all three applied:

$$
\kappa(A) \approx 1.27 \times 10^{2}, \quad \text{vs textbook } 3.75 \times 10^{16}.
$$

Algebraic FK error median: $2.9 \times 10^{-13}$. Pose-level success: 100%. **No numerical refinement needed.**

## AE-5: Möbius reparameterization (fallback)

For the few arms where AE-1+3+4 don't get $\kappa(A)$ below the cliff threshold ($10^{10}$ in current code), we have a more aggressive fallback: **Möbius reparameterize the eigenvalue variable** to spread roots away from the conditioning cliff.

**The construction.** Substitute

$$
x_2 = \frac{\alpha\, \tilde{x} + \beta}{\gamma\, \tilde{x} + \delta},
$$

clear the denominator $(\gamma \tilde{x} + \delta)^2$, and the new pencil $(A_{\mathrm{new}}, B_{\mathrm{new}}, C_{\mathrm{new}})$ in $\tilde{x}$ has the **same** roots, mapped through the inverse transform. Eigenvalues that were near the conditioning cliff at $x_2 \sim 0$ end up at convenient locations in $\tilde{x}$ space.

**The procedure.** Try a few random $(\alpha, \beta, \gamma, \delta)$ Möbius transforms, build $A_{\mathrm{new}}$ for each, take the one with smallest $\kappa(A_{\mathrm{new}})$. If that's still above threshold, fall through to the **generalised eigenvalue route** (Manocha–Canny Theorem 2): solve the matrix pencil $M_1 - x M_2$ via `scipy.linalg.eig` directly, no inversion of $A$ needed. ~2.5–3× slower but always works (singular pencils included).

In code: [`solve_x2_roots_mobius`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py).

## AE-6: Algebraic-first dispatch

[Issue #74.](https://github.com/siddhss5/ikfastpy/issues/74)

After AE-1/3/4/Möbius, the algebraic precision on most arms is at machine epsilon. But on adversarial poses or under exotic DH, the eigenvalue route still produces candidates with FK residual just above `fk_atol` — close enough to converge under Newton refinement, but not below the gate.

The **algebraic-first contract** (covered in [Chapter 6](06_refinement.md)): by default, candidates that miss `fk_atol` algebraically are **dropped**, not polished. Refinement is opt-in (`allow_refinement=True`) and reports back via `Solution.refinement_used`. This keeps the algebraic / iterative line cleanly drawn and surfaces it to callers — they always know whether the result was pure-algebraic or polished.

## What happens when all six measures still aren't enough

A few classes of pose remain hard even after every conditioning measure:

- **Manocha–Canny Table I synthetic arm** (a paper benchmark, not a real robot): the AE-3 leftvar selection improves it but doesn't reach machine precision algebraically. With `allow_refinement=True`, Newton converges in 4–6 iterations from the algebraic seed; without, candidates get dropped and `is_ls=True` fires for some poses. We track this as [#82](https://github.com/siddhss5/ikfastpy/issues/82) — a known **completeness** gap (the solutions returned are correct; we just can't always find the specific seeded $q^\star$ on this fixture).
- **True kinematic singularities** (wrist-pitch zero, elbow fully extended). Here the $\sin q_4 = 0$ degeneracy collapses two IK branches into one and the eigenvalue route returns two near-equal roots that map to the same configuration. Dedup handles this; reduced-cardinality solution sets are correct, just smaller.
- **Numerically singular pencils** that survive even Möbius. The generalised-eigenvalue path catches these, costing a few ms.

In every case, the user sees an **honest signal**: `fk_residual` is what it is, `refinement_used` reports what fired, `is_ls=True` fires if no candidate met `fk_atol`. No silent papering over.

## Putting it together

The **AE-1/3/4 + Möbius + algebraic-first** combination is what makes the textbook Raghavan–Roth pipeline survive on JACO 2 specifically and the EAIK gap generally. The summary:

- **AE-1** (equilibration): always-on, ~2× $\kappa$ improvement, free.
- **AE-3** (leftvar selection): always-on, **structural** fix, up to 14 orders of magnitude on pathological arms, cached per-arm.
- **AE-4** (SO(3) reduction): optional, ~3× $\kappa$, useful when AE-3 doesn't get all the way home.
- **AE-5** (Möbius reparameterization): triggered by `cond(A) > 1e10` cliff; always-on as a safety net.
- **Generalised eigenvalue fallback**: triggered when Möbius can't recondition; rare but bullet-proofs against truly singular pencils.
- **AE-6** (algebraic-first contract): user-facing — pure algebraic by default, opt-in Newton polish, transparent diagnostics.

The cumulative effect on JACO 2: **median 2.25 ms warm-cache IK, FK error 3.7e-13, 0 failures over 100 random poses.** That's what the next chapters describe how to use.

## A reference for porting RR to a new arm family

If you ever find yourself implementing this pipeline for a new family of arms (or a different parameterisation), the conditioning intuitions transfer:

1. Measure $\kappa(A)$ on a representative pose set with the textbook leftvar choice. If it's below $10^6$, you're done.
2. If $\kappa$ is high, try alternative leftvars and pick the lowest. The "pathological joints out of $v_{\mathrm{left}}$" heuristic is geometric and applies to any chain.
3. Equilibrate. Always.
4. If you still see $\kappa > 10^{10}$, instrument the eigenvector quality with the back-substitution residual and look for which entries of $v_{12}$ are zeroing out — that tells you which monomial is structurally absent and which Möbius transform might decluster the roots.
5. Always have an opt-in Newton polish with transparent diagnostics. The user needs to know when it fired.

Issue [#81](https://github.com/siddhss5/ikfastpy/issues/81) catalogues the IKFast-era tricks for ill-conditioning that we did **not** port (because they were band-aids over the missing AE-3 leftvar choice, or because they obscure failure modes). Worth reading if you're navigating the same conditioning swamp.
