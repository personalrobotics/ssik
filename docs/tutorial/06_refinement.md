# 6. Algebraic-first, refinement-second

The Raghavan–Roth pipeline of [Chapter 4](04_raghavan_roth.md) and the conditioning fixes of [Chapter 5](05_conditioning.md) get you to algebraic candidate solutions. For well-conditioned arms with the right leftvar choice, those candidates are at machine precision and you're done. For ill-conditioned poses or adversarial geometries, the candidates are *near misses* — close enough that one Newton step on the FK residual would converge them, but not close enough to call themselves "the answer".

Every analytical-IK library has to decide what to do with near-miss candidates. The choice is more than a tuning decision; it's a **contract with the user** about what "analytical IK" actually means. ssik's contract is: **pure algebraic by default, opt-in Newton polish, transparent diagnostics**.

This chapter walks through that contract, the `Solution` dataclass that surfaces it, and the `lm_refine` primitive that implements the opt-in polish.

## The contract

The shape every solver in ssik exposes:

```python
def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> tuple[list[Solution], bool]:
    ...
```

The contract is in three lines:

- **Default** (`allow_refinement=False`): candidates that miss `policy.subproblem_numerical` algebraically are **dropped**, not silently polished. The solver returns only solutions that the algebraic pipeline produced at the requested tolerance.
- **Opt-in** (`allow_refinement=True`): each near-miss candidate gets one [`lm_refine`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/refinement/__init__.py) pass — Newton on the SE(3) log residual via the spatial Jacobian. If it converges within `refinement_max_iters`, the polished `q` is included; if not, dropped.
- **Per-Solution reporting**: the returned `Solution` dataclass carries `refinement_used: Literal["none", "lm"]` and `refinement_iters: int`. The caller always knows what fired on every solution.

This is GitHub issue [#74](https://github.com/siddhss5/ikfastpy/issues/74)'s design spec. The corresponding dataclass is [#75](https://github.com/siddhss5/ikfastpy/issues/75).

## Why this contract specifically

The obvious alternative — always run Newton inside the solver, hide it from the user — is the worst-of-both-worlds anti-pattern. The user thinks they're getting *analytical* IK; in fact they're getting whatever Newton converged to in 100 iterations of Levenberg–Marquardt. The reported "FK error" is post-LM-polish, which can hide structural problems with the algebraic pipeline (e.g. a leftvar choice that makes most candidates fail by orders of magnitude — the user wouldn't notice because LM cleans it up before they see it).

The Day-4 prototype of ssik's tier-2 RR solver did exactly that: scipy LM was embedded in the candidate-validation loop, up to 100 iterations per branch. It worked — JACO 2 reached FK error 5.7e-9 — but the architecture was wrong:

- Users couldn't tell whether their IK call was deterministic or iterative.
- Iteration counts ballooned (100 LM iters for cases Manocha–Canny §V-E specifies handle in 1–2 Newton steps).
- FK tolerance became a hidden default rather than a user-facing contract.

The user's verbatim direction at the time:

> we can't just paper over LS randomly. it's a last resort and we need to assert exactly when and where we need it. We also need to be super transparent to the user why we're using LS and what the tolerance is and also if they want to do it at all on their robot or just skip it and reject solutions if they don't meet tolerance.

The contract is the codification of that direction.

## The `Solution` dataclass

Every solver in ssik returns `tuple[list[Solution], bool]` where `Solution` is:

```python
@dataclass(frozen=True)
class Solution:
    q: NDArray[np.float64]                       # joint vector
    fk_residual: float                            # ||FK(q) - T_target||_F at return time
    refinement_used: Literal["none", "lm"]        # whether polish fired
    refinement_iters: int                         # iteration count if it did (0 otherwise)
    branch_id: int | None = None                  # IK branch index (0..15 for 6R RR)
    solver_name: str = ""                         # e.g. "ikgeo.general_6r"
```

Frozen — derived data, callers shouldn't mutate. The dataclass lives at [`src/ssik/core/solution.py`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/core/solution.py).

The fields encode the exact provenance and precision of each solution. `fk_residual` is what you'd measure if you computed `np.linalg.norm(FK(q) - T_target)` yourself; the solver's `fk_atol` was a *filter*, not a contract on the value reported here. `refinement_used` is the algebraic-vs-iterative flag — `"none"` means pure algebraic, `"lm"` means Newton-polished. `refinement_iters` says how many iterations consumed (0 if none). `branch_id` is the IK branch index (the back-substitution route assigns a stable index 0..15 per pose for the 16-root RR solver). `solver_name` is useful when results pass through a dispatcher.

This shape has carried through every solver migration in [PR #85](https://github.com/siddhss5/ikfastpy/pull/85): the ten public solvers (six tier-0 closed-form, two tier-1 univariate-search, one tier-2 grid, one tier-2 RR, one jointlock 7R wrapper) all return the same shape with the same semantic guarantees.

## `lm_refine` — the universal polish primitive

The opt-in polish is a hand-rolled Newton method on the SE(3) log residual. Implementation: [`ssik.refinement.lm_refine`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/refinement/__init__.py). The signature:

```python
def lm_refine(
    q_seed: NDArray[np.float64],
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t_target: NDArray[np.float64],
    *,
    fk_atol: float = 1e-9,
    max_iters: int = 15,
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    step_clip: float = 0.5,
) -> tuple[NDArray[np.float64], float, int] | None:
    ...
```

The inner loop in pseudocode:

1. Compute `T_q = fk_fn(q)`.
2. Compute residual $r = \log(T_{\mathrm{target}}\, T_q^{-1}) \in \mathbb{R}^6$ (translation + axis-angle rotation).
3. If $\|r\| < \mathrm{fk\_atol}$, return $(q, \|r\|, \mathrm{iter})$.
4. Compute spatial Jacobian $J_s$ (analytical if `jacobian_fn` given; central differences otherwise — slow).
5. Solve $J_s\, \delta q = r$ via LAPACK `np.linalg.solve` (with Tikhonov-damped fallback on singular Jacobian).
6. Step-clip $\|\delta q\|_\infty \le \mathrm{step\_clip}$ rad to keep the trajectory inside the linearisation regime.
7. $q \leftarrow q + \delta q$, repeat.
8. After `max_iters` if still above tolerance, return `None`.

There is **no divergence-abort heuristic**. Newton can be non-monotonic near saddles or under step-clipping; aggressive early termination misses real recoveries. We trust the `max_iters` cap and a final residual check.

Single LAPACK `solve` per iter, no scipy. Roughly 50× faster than the original scipy-LM wrapper on cases where 1–5 iters suffice (which is virtually all reasonable seeds when the spatial Jacobian is exact).

## `verify_candidates` — the shared back-half

The pattern after producing raw joint-angle candidates — FK-verify, optionally Newton-polish, dedup — is identical across every solver in ssik. It's factored once into [`ssik.refinement.verify_candidates`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/refinement/__init__.py); ten solvers consume it. That's how the contract gets enforced uniformly: the dedup-by-fk-residual rule, the wrap-to-π comparison, the `Solution` wrapping with the right `refinement_used` flag — all live in one place.

The helper takes the raw candidates plus an `fk_fn`, optional `jacobian_fn`, target pose, tolerance policy, and the `allow_refinement` flag, and returns a deduplicated `list[Solution]`. Solvers boil down to: produce raw `q` candidates from the algebraic algorithm, hand them to `verify_candidates`, return the result.

## When refinement actually fires

In practice the refinement plumbing rarely triggers because the algebraic pipeline plus the AE-1/3/4 conditioning fixes get most arms to machine precision:

- **Tier-0 closed-form solvers** (Pieper-class — `spherical_two_parallel`, `three_parallel`, etc.): never. Algebraic precision is at machine epsilon by construction. The `verify_candidates` plumbing exists for uniformity but the lm_refine path doesn't enter.
- **Tier-2 RR solver, well-conditioned arm**: never on real arms with the right leftvar. Real JACO 2 j2n6s200 with AE-3-selected leftvar = q₁ runs at 96.6% algebraic_pass with median FK error 3.7e-13 across 100 random poses (post-Tier 2.3 of [#86](https://github.com/siddhss5/ikfastpy/pull/90)). Refinement off, `is_ls=False`, all branches recovered.
- **Tier-2 RR solver, MC Table I synthetic** (textbook benchmark from Manocha–Canny 1994, deliberately ill-conditioned): ~99% of candidates need polish. With `allow_refinement=True` median Newton iteration count is 4. Without, candidates get dropped and `is_ls=True` fires for some poses. Tracked as [#82](https://github.com/siddhss5/ikfastpy/issues/82) — a known *completeness* gap on a synthetic fixture.

## What this gives users

1. **A uniform calling pattern.** Every solver in ssik takes `(kb, T_target, policy, *, allow_refinement, refinement_max_iters)` and returns `tuple[list[Solution], bool]`. Switching from one solver to another is a one-line import change.

2. **A precision contract that's enforceable.** If `Solution.refinement_used == "none"`, the value is pure-algebraic at the reported `fk_residual`. The user gets to choose whether iterative polish was acceptable for their use case.

3. **A debug surface.** `Solution.refinement_iters > 5` is a flag that the algebraic path is struggling on this specific pose — worth investigating if you see it consistently (it usually means the arm is at or near a singularity, or the leftvar choice isn't ideal).

4. **A mental model**: ssik draws the algebraic / iterative line *explicitly*. There is no "mostly analytical" middle ground — solutions are either reported pure-algebraic (with `refinement_used="none"`) or pure-Newton-polished from a near-miss algebraic seed (with `refinement_used="lm"` and an iteration count).

## The architectural rule

When something is structurally a separate concern, the user-facing API should make it look like a separate concern. Numerical refinement is structurally separate from algebraic IK — it's a different algorithm operating on a different correctness contract — and ssik's API treats it that way. The solver / refinement separation is the same architectural move that distinguishes optional features (`allow_refinement` flag) from the core path (algebraic-only by default).

The corresponding [memory entry](https://github.com/siddhss5/ikfastpy/issues/74) reads, verbatim:

> Numerical refinement is a separate, opt-in, transparent layer. Solvers produce algebraic candidates; LS/Newton lives in `ssik.refinement`, off by default, FK-tolerance-driven, surfaces what fired.

That's what's shipping.
