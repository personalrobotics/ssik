# 6. Algebraic-first, refinement-second

!!! warning "Scaffolding"
    Outline below; prose to be filled in. Implementation: [`src/ssik/refinement/`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/refinement/) and the `Solution` dataclass at [`src/ssik/core/solution.py`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/core/solution.py). Design rationale in [#74](https://github.com/siddhss5/ikfastpy/issues/74) and [#75](https://github.com/siddhss5/ikfastpy/issues/75).

## What this chapter covers

The contract every ssik solver follows: **pure algebraic by default, opt-in Newton polish, transparent diagnostics**.

### The contract (#74)

- Default: `solve(kb, T) -> tuple[list[Solution], bool]`. Candidates that don't reach `fk_atol` algebraically are **dropped**, not silently polished.
- Opt-in: `solve(kb, T, allow_refinement=True)`. Each near-miss candidate goes through one [`lm_refine`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/refinement/__init__.py) pass — Newton on the SE(3) log residual via the spatial Jacobian.
- Per-`Solution` reporting: `refinement_used: Literal["none", "lm"]`, `refinement_iters: int`. The caller always knows what fired.

### Why this matters

Numerical refinement hidden inside an "analytical" solver is one of the worst-of-both-worlds anti-patterns. The user thinks they're getting algebraic precision; in fact they're getting whatever Newton converged to in 100 iterations of LM. ssik's contract makes the algebraic / iterative line *explicit* and surfaces it on every returned `Solution`.

### `lm_refine` — the universal polish primitive

Hand-rolled Newton, no scipy:

- Input: seed `q`, FK callable, target pose, optional analytical Jacobian.
- Inner loop: compute SE(3)-log residual $r = \log(T \cdot \mathrm{FK}(q)^{-1}) \in \mathbb{R}^6$, solve $J_s\, \delta q = r$ via LAPACK, step-clip $\|\delta q\|_\infty \le 0.5$ rad, iterate until $\|r\| < \mathrm{fk\_atol}$ or `max_iters` hit.
- No divergence-abort heuristic: Newton can be non-monotonic near saddles or under step-clipping; aggressive early termination misses real recoveries.
- Single LAPACK `solve` per iter; ~50× faster than the scipy LM wrapper on cases where 1–5 iters suffice.

### `Solution` dataclass

```python
@dataclass(frozen=True)
class Solution:
    q: NDArray[np.float64]
    fk_residual: float
    refinement_used: Literal["none", "lm"]
    refinement_iters: int
    branch_id: int | None = None
    solver_name: str = ""
```

Returned by every solver. Frozen — derived data, callers shouldn't mutate.

### `verify_candidates` — the shared back-half

Every solver runs the same pattern after producing raw joint-angle candidates: FK-verify, optionally Newton-polish, dedup. Factored once into [`ssik.refinement.verify_candidates`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/refinement/__init__.py); ten solvers consume it.

### When refinement actually fires

- Tier-0 closed-form solvers (Pieper-class): **never**, by design. Algebraic precision is at machine epsilon.
- Tier-2 RR solver, well-conditioned arm (JACO 2 post-AE-3): never. 96.6% algebraic_pass at $10^{-13}$ FK error. The refinement plumbing exists but doesn't trigger.
- Tier-2 RR solver, MC Table I synthetic: ~99% of candidates need polish (the textbook fixture is ill-conditioned by design). With `allow_refinement=True` the median-polish-iteration count is 4. Without, candidates get dropped and `is_ls=True` fires. (#82.)

### What this gives users

- A calling pattern that's identical across all 10 solvers in ssik.
- A precision contract that's enforceable: "if `refinement_used == 'none'`, this is pure-algebraic at the reported `fk_residual`".
- A debug surface: when something goes wrong, `Solution.refinement_iters > 5` is a flag that the algebraic path is struggling and you might want to investigate the geometry.

## References

- Memory entry [`feedback_refinement_architecture`](https://github.com/siddhss5/ikfastpy/issues/74).
- GitHub PR #85 (the migration that landed this contract across all solvers).
