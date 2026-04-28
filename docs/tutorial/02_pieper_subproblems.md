# 2. The Pieper class and subproblem composition

!!! warning "Scaffolding"
    Outline below; prose to be filled in. This chapter covers the analytical-IK ground that EAIK / IK-Geo already handle elegantly — necessary background but not the part of ssik that's novel. See [`src/ssik/subproblems/`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/subproblems/) and [`src/ssik/solvers/ikgeo/`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/) for the implementation.

## What this chapter covers

- **Pieper's condition (1968):** a 6-DOF arm with three consecutive intersecting axes admits closed-form IK. Decouples wrist orientation from elbow position.
- **The six canonical subproblems (Paden–Kahan, Elias–Wen):**
  - **SP1** — `axis k`, vector `p` rotated to align with vector `q`. Single angle `θ`.
  - **SP2** — two sequential rotations to align `p` with `q`. Two angles, up to 2 solutions.
  - **SP3** — distance constraint: rotate `p` around `k` to be at distance $d$ from origin. Up to 2 solutions.
  - **SP4** — projection: $k_2 \cdot \mathrm{Rot}(k_1, \theta) p = h$. Up to 2 solutions.
  - **SP5** — three sequential rotations placing `p_0 + R_1 p_1 + R_2 p_2 = -p_3`. Quartic, up to 4 solutions.
  - **SP6** — two pairs of subproblem-4 conditions. Bezout quartic, up to 4 solutions.
- **Compositions:** how the per-family ssik solvers chain SP1–SP6 to produce 8-solution closed-form IK:
  - `ikgeo.three_parallel` (UR5, UR10): SP6 + SP1 × 4 + SP3.
  - `ikgeo.spherical_two_parallel` (Puma 560, Fanuc, KUKA KR): SP4 + SP3 + SP1 + SP4 + SP1 × 2.
  - `ikgeo.spherical_two_intersecting` (Puma 560 alt., compact arms): SP3 + SP2 + SP4 + SP1 × 2.
  - `ikgeo.spherical` (generic spherical wrist): SP5 + SP4 + SP1 × 2.
  - `ikgeo.two_parallel`, `ikgeo.two_intersecting`: tier-1 univariate-search 6R.
- **Cost note:** these solvers run in 50-200 µs warm-cache (with sub-millisecond first-call cost). Pieper-class arms are basically a closed problem.
- **Cross-solver agreement on Puma 560:** Puma satisfies *both* `spherical_two_parallel`'s and `spherical_two_intersecting`'s preconditions; the two algebraically-distinct compositions return the same 8-solution set on every pose. This is the strongest correctness guarantee available — see [Chapter 8](08_bulletproof.md).
- **What Pieper *doesn't* cover:** non-Pieper 6R (no axis triple parallel or intersecting). That's the EAIK gap, addressed in [Chapter 3](03_eaik_gap.md) onward.

## References

- Pieper, D. L. (1968). PhD thesis, Stanford.
- Paden, B. (1986). PhD thesis, UC Berkeley.
- Elias, A. & Wen, J. (2022/2025). "IK-Geo: unified robot inverse kinematics using subproblem decomposition." [arXiv:2211.05737](https://arxiv.org/abs/2211.05737).
- Ostermeier, D. (2024). "EAIK: a Toolbox for Efficient Analytical Inverse Kinematics." [arXiv:2409.14815](https://arxiv.org/abs/2409.14815).
