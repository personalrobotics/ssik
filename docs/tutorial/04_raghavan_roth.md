# 4. Raghavan–Roth in 8 stages

The tier-2 numeric solver ssik ships for non-Pieper 6R arms is a clean-room port of the **Raghavan–Roth 1990 / 1993** algebraic-elimination pipeline plus the **Manocha–Canny 1994** companion-matrix eigenvalue route. The math is dense but each stage is a well-defined linear-algebra step. This chapter walks through the algorithm one stage at a time, with code pointers into the implementation in [`src/ssik/solvers/ikgeo/_raghavan_roth.py`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py).

The next chapter, [Conditioning is the hard part](05_conditioning.md), covers the four robustness fixes (AE-1, AE-3, AE-4, Möbius reparameterization) that are necessary for the textbook algorithm to survive on real ill-conditioned arms like JACO 2. Read this chapter first; it sets up the language those fixes use.

## Setup

A 6R serial chain in standard distal-DH form has six joint transforms

$$
A_i(\theta_i) = R_z(\theta_i)\, T_z(d_i)\, T_x(a_i)\, R_x(\alpha_i), \qquad i = 1, \ldots, 6,
$$

where $(\alpha_i, a_i, d_i)$ are the per-link DH parameters and $\theta_i$ is the joint angle for joint $i$. The forward-kinematics map is

$$
\mathrm{FK}(\theta_1, \ldots, \theta_6) = A_1 A_2 A_3 A_4 A_5 A_6.
$$

Given a target pose $T \in \mathrm{SE}(3)$, IK asks: find every $(\theta_1, \ldots, \theta_6)$ with $\mathrm{FK} = T$.

Algebraically the equation $A_1 \cdots A_6 = T$ is six trig equations (12 if you count the redundant rotation entries) in six unknowns. The Raghavan–Roth strategy is **rearrange the loop closure so the unknowns split unevenly across the two sides, then eliminate the cluttered side via linear algebra**.

We split as

$$
A_2 A_3 A_4 = A_1^{-1}\, T\, A_5^{-1} A_6^{-1}.
$$

The left-hand side depends on $\theta_2, \theta_3, \theta_4$. The right-hand side depends on $\theta_1, \theta_5, \theta_6$. Each side is a $4 \times 4$ matrix; equality gives a system of equations in the six unknowns.

## Stage 1 — The 14-equation $(P, Q)$ system

Take three specific column entries and three specific dot products of the matrix equation and you get **14 polynomial equations** that factor as

$$
(P_{\sin}\, s_{q_2} + P_{\cos}\, c_{q_2} + P_{\mathrm{one}})\, v_{\mathrm{left}}(q_3, q_4) = Q\, v_{\mathrm{right}}(q_0, q_1).
$$

where $s_{q_i} = \sin q_i$, $c_{q_i} = \cos q_i$. Here we've renamed so $q_0, \ldots, q_5$ are the joint variables (this is the convention in the code; $\theta_i$ from the setup section maps to $q_{i-1}$). The structure is:

- $P_{\sin}, P_{\cos}, P_{\mathrm{one}}$ are each $14 \times 9$ matrices whose entries are polynomial in the DH parameters and the target pose $T$.
- $v_{\mathrm{left}}(q_3, q_4)$ is the **bilinear monomial vector** in $(s_3, c_3, s_4, c_4)$:

  $$
  v_{\mathrm{left}} = (s_3 s_4,\; s_3 c_4,\; c_3 s_4,\; c_3 c_4,\; s_3,\; c_3,\; s_4,\; c_4,\; 1)^T \in \mathbb{R}^9.
  $$

- $v_{\mathrm{right}}(q_0, q_1) \in \mathbb{R}^8$ is the analogous bilinear-monomial vector in $(s_0, c_0, s_1, c_1)$ but without the constant term.
- $Q$ is a $14 \times 8$ matrix of DH × target-pose entries. **Constant per pose.**

The reason the equations factor this way is the Raghavan–Roth choice of which scalar combinations of the matrix equation to extract — it is engineered so that $q_0$ and $q_1$ live entirely on the right-hand side, $q_3$ and $q_4$ live entirely on the left, and the two are linked only through $q_2$ (which appears as the linear $s_{q_2}, c_{q_2}$ coefficients) and $q_5$ (which has been eliminated entirely from the columns we extracted — that's the role of the loop split $A_2 A_3 A_4 = A_1^{-1} T A_5^{-1} A_6^{-1}$).

In code: [`build_pq`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py) returns $(P_{\sin}, P_{\cos}, P_{\mathrm{one}}, Q, \mathrm{metadata})$. The matrices are derived symbolically once per arm (cached in `_cached_derivation` via `lru_cache` on the DH tuple) and then evaluated at the target pose by lambdified callables — sympy at module-import time, pure numpy at runtime.

## Stage 2 — Eliminate $(q_0, q_1)$ via the left null space of $Q$

$Q$ is $14 \times 8$ with full column rank 8 generically. Its **left null space** has dimension 6: there exist 6 row vectors $n_1, \ldots, n_6 \in \mathbb{R}^{14}$ with $n_i^T Q = 0$.

Multiplying both sides of the (P, Q) system on the left by these null-space rows kills the $Q v_{\mathrm{right}}$ side entirely:

$$
n_i^T (P_{\sin}\, s_{q_2} + P_{\cos}\, c_{q_2} + P_{\mathrm{one}})\, v_{\mathrm{left}} = 0, \qquad i = 1, \ldots, 6.
$$

Stacking the six $n_i$ as rows of $N \in \mathbb{R}^{6 \times 14}$, we get an **eliminated 6×9 system** $E$:

$$
E_{\sin}\, s_{q_2} + E_{\cos}\, c_{q_2} + E_{\mathrm{one}} = N (P_{\sin}\, s + P_{\cos}\, c + P_{\mathrm{one}}),
$$

with $E_{\bullet} \in \mathbb{R}^{6 \times 9}$, all dependent on $q_2$ alone (in the $s_{q_2}, c_{q_2}$ coefficients) and acting on $v_{\mathrm{left}}(q_3, q_4)$.

The left null space is computed via SVD of $Q$: $Q = U \Sigma V^T$, with $U \in \mathbb{R}^{14 \times 14}$. The left null space rows are the rows of $U^T$ corresponding to the eight zero (or near-zero) singular values. In code: [`eliminate_q0_q1`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py) computes this and returns $(E_{\sin}, E_{\cos}, E_{\mathrm{one}})$.

After Stage 2 we have **6 equations in 3 unknowns $(q_2, q_3, q_4)$** — over-determined, exactly what we want.

## Stage 3 — Half-angle (Weierstrass) substitution and basis change

The standard Weierstrass substitution

$$
x = \tan(q/2), \qquad \sin q = \frac{2x}{1 + x^2}, \qquad \cos q = \frac{1 - x^2}{1 + x^2}
$$

turns trig polynomials into rational polynomials. Apply it to all three remaining angles, multiply through by the appropriate $(1 + x_i^2)$ to clear denominators, and the system becomes a **polynomial system in $(x_2, x_3, x_4)$**.

We then reorganise the basis of $v_{\mathrm{left}}$. After Weierstrass on $q_3$ and $q_4$, the original 9-monomial $v_{\mathrm{left}}$ basis becomes a 12-monomial basis in $(x_3, x_4)$:

$$
v_{12} = (x_3^2 x_4^2,\; x_3^2 x_4,\; x_3^2,\; x_3 x_4^2,\; x_3 x_4,\; x_3,\; x_4^2,\; x_4,\; 1,\; x_3 x_4^2,\; x_3 x_4,\; x_3)^T \in \mathbb{R}^{12}.
$$

(The last three rows duplicate three of the earlier rows multiplied by $x_3$. This redundancy is intentional — it gives the next stage the right shape.)

Apply Weierstrass on $q_2$ also: each row that was $E_{\sin}\, s_2 + E_{\cos}\, c_2 + E_{\mathrm{one}}$ becomes a quadratic in $x_2$ after multiplying by $(1 + x_2^2)$:

$$
\text{quadratic-in-}x_2\text{ coefficient}\cdot x_2^2 \;+\; \text{linear-in-}x_2\cdot x_2 \;+\; \text{constant}.
$$

So we now have **6 equations $\times$ 12 monomials, polynomial-in-$x_2$ of degree $\le 2$**. In code: [`weierstrass_eliminate_trig`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py).

## Stage 4 — Build $M(x_2)$, the $12 \times 12$ polynomial matrix pencil

The elimination so far gives 6 equations in 12 monomials. We need 12 to invert. The trick: stack the system with a **shifted copy** that multiplies the same equations by $x_3$:

$$
M(x_2) = \begin{bmatrix} E'' & 0 \\ 0 & E'' \cdot x_3 \end{bmatrix}_{12 \times 12} \cdot v_{12} = 0,
$$

written in matrix form as

$$
M(x_2)\, v_{12} = (A x_2^2 + B x_2 + C)\, v_{12} = 0,
$$

with $A, B, C \in \mathbb{R}^{12 \times 12}$ all numeric (the stage-3 outputs $E_{\mathrm{quad}}, E_{\mathrm{lin}}, E_{\mathrm{const}}$ stacked according to the shift construction). This is a **quadratic eigenvalue problem**: find $x_2$ such that $M(x_2)$ is singular and read off the corresponding kernel vector $v_{12}$.

The polynomial $\det M(x_2)$ has degree 24 in $x_2$. Of those 24 roots, 8 are spurious (clustered near $x_2 = \pm i$ from the $(1 + x_2^2)^4$ factor introduced by Weierstrass on $q_2$). The remaining **16 are the candidate $\tan(q_2/2)$ values** — corresponding to the 16 IK branches a generic 6R chain admits.

In code: [`build_m_matrix`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py).

## Stage 5 — Companion matrix and 24-eigenvalue route

A quadratic eigenvalue problem $(A x^2 + B x + C) v = 0$ linearises into a standard eigenvalue problem on a $24 \times 24$ companion matrix:

$$
\Sigma = \begin{bmatrix} 0_{12} & I_{12} \\ -A^{-1} C & -A^{-1} B \end{bmatrix}, \qquad
\Sigma \begin{bmatrix} v \\ x v \end{bmatrix} = x \begin{bmatrix} v \\ x v \end{bmatrix}.
$$

The 24 eigenvalues of $\Sigma$ are the 24 roots of $\det M(x_2)$. The 24-component eigenvector has block structure $(v;\, x v)$ — the top half is the $v_{12}$ kernel of $M(x_2)$.

We compute $\Sigma$ from $A, B, C$, run `np.linalg.eig`, and filter:

1. Drop eigenvalues clustered near $\pm i$ (spurious from the Weierstrass factor) — controlled by `spurious_tol`.
2. Drop eigenvalues whose imaginary part exceeds `imag_rel_tol * max(|real|, 1)` — they correspond to complex IK solutions, not physical.
3. The surviving real eigenvalues are the candidate $x_2 = \tan(q_2/2)$ values; corresponding eigenvectors are the $v_{12}$ kernels.

In code: [`solve_x2_roots`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py). The fallback path [`solve_x2_roots_mobius`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py) handles ill-conditioned $A$ via reparameterization (covered in [Chapter 5](05_conditioning.md)).

This is the most expensive single operation in the pipeline (~250 µs of LAPACK `dgeev` time on a 24×24 dense matrix; see #86 for ongoing speed work).

## Stage 6 — Back-substitute: eigenvector to $(q_3, q_4)$ and then to $(q_0, q_1, q_5)$

We have $x_2$ (one root) and $v_{12}$ (its 12-component kernel). We need to extract the remaining angles.

**$(q_3, q_4)$ from $v_{12}$:** the monomial structure tells us each entry of $v_{12}$ is one of $\{x_3, x_4, x_3^2, \ldots\}$ times another. So various pairs of entries have ratio equal to $x_3$ or $x_4$:

$$
v_{12}[5] / v_{12}[8] = x_3, \quad v_{12}[7] / v_{12}[8] = x_4, \quad v_{12}[2] / v_{12}[5] = x_3, \ldots
$$

Multiple equivalent ratios exist. Per Manocha–Canny §IV-C, **picking the ratio whose denominator has largest magnitude** is the numerically safe choice — when an eigenvector is unit-normalised, small entries in the denominator amplify error. Implementation: search through a list of redundant ratio pairs, take the one with biggest denominator. From $x_3, x_4$ we recover $q_3 = 2 \arctan x_3$ and $q_4 = 2 \arctan x_4$.

**$(q_0, q_1)$ from $v_{\mathrm{right}}$:** with $q_2, q_3, q_4$ known, the right-hand side $Q v_{\mathrm{right}}$ of the original Stage 1 equation has fully determined left-hand side. Solve

$$
v_{\mathrm{right}} = Q^+\, (P_{\mathrm{eff}}(q_2)\, v_{\mathrm{left}}(q_3, q_4))
$$

where $Q^+$ is the pseudo-inverse and $P_{\mathrm{eff}}(q_2) = P_{\sin} s_2 + P_{\cos} c_2 + P_{\mathrm{one}}$. Read $(\sin q_0, \cos q_0)$ and $(\sin q_1, \cos q_1)$ from the appropriate entries of $v_{\mathrm{right}}$ and recover the angles via `atan2`.

(The pinv of $Q$ is **shared across all eigenvalue branches** — it's identical for every branch of the same target pose. ssik computes it once per pose and reuses; this is one of the [Tier 2 speed wins](https://github.com/siddhss5/ikfastpy/pull/89) on #86.)

**$q_5$ (drop joint) from FK closure:** with $q_0, q_1, q_2, q_3, q_4$ known, the only unknown left is $q_5$. Compute the chain before joint 5 and after joint 5; solve

$$
A_5(q_5) = \mathrm{chain\_before}^{-1}\, T\, \mathrm{chain\_after}^{-1}
$$

The standard DH form $A_5 = R_z(q_5) T_z(d_5) T_x(a_5) R_x(\alpha_5)$ puts $(c_5, s_5)$ in column 0 rows 0,1 of the result. Read off $q_5 = \mathrm{atan2}(A_5[1,0], A_5[0,0])$.

In code: [`back_substitute`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py) (public wrapper) and [`_back_substitute_inner`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py) (hot path with precomputed $Q^+$).

## Stage 7 — FK validation and Newton polish (optional)

Each candidate $q$ from Stage 6 is checked against $T$ by FK closure:

$$
\mathrm{fk\_residual} = \|\mathrm{FK}(q) - T\|_F.
$$

For a **well-conditioned** arm (say UR5), the algebraic precision is at machine epsilon — typical residual $\sim 10^{-13}$. Candidates pass the `fk_atol` gate (default $10^{-5}$) trivially.

For an **ill-conditioned** arm (JACO 2 with the wrong leftvar choice; see Chapter 5), eigenvalue precision degrades and candidates may have residual $\sim 10^{-3}$. The default policy is to **drop those candidates**: ssik enforces an algebraic-first contract (see [Chapter 6](06_refinement.md)) where pure-algebraic precision is required by default. Users who explicitly opt in via `allow_refinement=True` get a Newton-on-spatial-Jacobian polish ([`lm_refine`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/refinement/__init__.py)) that converges 1–5 iterations from a near-miss algebraic seed; the resulting `Solution.refinement_used = "lm"` and `refinement_iters = 3` (or whatever) tell the caller exactly what fired.

## Stage 8 — Deduplicate and return

Different eigenvalue branches can map to the same IK solution mod $2\pi$ (especially on degenerate poses). Final deduplication: collapse solutions whose joint vectors agree wrap-to-π within `dedup_atol` (default $10^{-3}$ rad), keeping the lower-`fk_residual` representative.

The public driver [`solve_all_ik`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py) returns `tuple[list[Solution], bool]`: a deduplicated list of `Solution` dataclasses (each carrying `q`, `fk_residual`, `refinement_used`, `refinement_iters`, `branch_id`, `solver_name`) and an `is_ls` flag indicating whether at least one candidate survived.

## Why this pipeline works

The algebraic miracle is Stage 1's loop split: the choice $A_2 A_3 A_4 = A_1^{-1} T A_5^{-1} A_6^{-1}$ — combined with the choice of which 14 scalar combinations of the matrix equation to extract — produces a system that's:

- **Linear in the joint variables on each side** (after Weierstrass): each joint contributes a $\sin/\cos$ pair, the bilinears are quadratic, and $q_5$ is eliminated entirely from the columns we keep.
- **Decoupled** across the two sides: $(q_0, q_1)$ on the right, $(q_3, q_4)$ on the left, $q_2$ as the bridge.
- **Eliminable** via SVD on $Q$: the right side disappears in a single linear-algebra step.

Manocha–Canny's contribution was recognising that the resulting quadratic eigenvalue problem in $x_2$ linearises cleanly to a 24×24 companion matrix, and the 16 valid roots of $\det M = 0$ are the IK branches.

The whole pipeline is, structurally, a sequence of well-understood linear-algebra primitives: SVD (Stage 2), polynomial substitution (Stage 3), block matrix assembly (Stage 4), companion matrix eigendecomposition (Stage 5), pseudo-inverse and `atan2` (Stage 6). At Python-LAPACK speeds, ssik runs the whole thing in roughly 2.25 ms median per IK on JACO 2 (post-Tier 2.3 of [#86](https://github.com/siddhss5/ikfastpy/issues/86)).

What makes this hard is that several of those primitives become numerically unstable on real arm geometries that produce nearly-singular $A$ in Stage 4. The next chapter covers why, and the four independent attacks ssik ships to handle it.
