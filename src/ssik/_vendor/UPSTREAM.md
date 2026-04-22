# Vendored upstream

Files in this directory originate from [rdiankov/openrave](https://github.com/rdiankov/openrave) at the pinned commit below. **Local modifications have been applied** — see the *Local modifications* section. Treat this directory as a vendored fork that we maintain on top of an upstream baseline; never edit these files in ways that aren't tracked here.

## Upstream baseline

- Repository: <https://github.com/rdiankov/openrave>
- Pinned commit: `ec22ecfaf006688cbc5ee0fdd8fa05d2c5676d37` (2024-08-16)

| File | Upstream path | License | Upstream lines | Upstream SHA-256 |
|---|---|---|---|---|
| `ikfast.py` | `python/ikfast.py` | LGPL-3.0-or-later | 9683 | `aa1aeb592c6f82701598f6d6dc33559bec10c225e3054f0fd9c3e969b2bdb4cf` |
| `ikfast_generator_cpp.py` | `python/ikfast_generator_cpp.py` | LGPL-3.0-or-later | 2967 | `b471e0795f12072f3caabceb6de03a5c31d9c09dfc2a1cc10ffbe3afdb6431c4` |
| `ikfast.h` | `python/ikfast.h` | Apache-2.0 | 600 | `bb78d2a3664dee323f165765601d89e84fe7f735b2628fd9245ce2eb043f3442` |

The SHA-256s above are the **upstream** hashes at the pinned commit, not the current vendored content. Use them when re-pinning to confirm the starting baseline.

## Local modifications

Each entry below documents a tracked PR that modifies these files. Re-pinning requires re-applying every entry in order.

| PR | Issue | Files touched | Summary |
|---|---|---|---|
| [#26](https://github.com/siddhss5/ikfastpy/pull/26) | [#4](https://github.com/siddhss5/ikfastpy/issues/4) | `ikfast.py`, `ikfast_generator_cpp.py` | Strip OpenRAVE imports (keep fallback branches inline); rename loggers to `ssik.ikfast` (was `ikfastpy.ikfast` before the #38 package rename); replace `ikfast.py` `__main__` CLI with a `NotImplementedError` stub; remove the docstring section describing the now-removed CLI. |
| [#?](https://github.com/siddhss5/ikfastpy/pulls) | [#6](https://github.com/siddhss5/ikfastpy/issues/6) | `ikfast.py` | Remove the `import six` and `@six.python_2_unicode_compatible` decorator on `CannotSolveError`. Both are Python-2 compat shims that are no-ops on Python ≥3.11. With this single patch the vendored generator imports cleanly on sympy 1.14 (latest stable as of writing). |
| [#?](https://github.com/siddhss5/ikfastpy/pulls) | [#28](https://github.com/siddhss5/ikfastpy/issues/28) | `ikfast.py` | Two sympy-1.14 compat fixes discovered along the UR5 Translation3D solve path: **(a)** oscillation guard on `SimplifyAtan2`'s Add-branch simplify-until-fixed-point loop. `simplify()` can oscillate between structurally-unequal-but-equivalent forms, so the recursion never terminated; a hidden `_add_visited` kwarg tracks expression hashes and breaks out on repeat. **(b)** `isValidSolution` now treats `TypeError` from `isinf`/`isnan` as invalid. `evalf()` can return a complex mpmath number whose repr omits the `I` symbol, so `e.has(I)` misses it and the subsequent `isinf(e)` blows up trying to float-convert. Complex numbers are not valid real solutions. Together these make `generateIkSolver` complete on a UR5 chain. |
| [#?](https://github.com/siddhss5/ikfastpy/pulls) | [#33](https://github.com/siddhss5/ikfastpy/issues/33) | `ikfast.py` | Three bugs discovered while unblocking Transform6D on real arms: **(a)** `Poly.expand()` at `solveLiWoernleHiller` — the method was removed on modern sympy; go through `.as_expr().expand()` instead. **(b)** Typo at `solveVariablesLinearly`: the inner loop's complexity sum referenced the *outer*-loop `i,j` indices (`M2[i,j]`) instead of the inner `i2,j2`, raising `IndexError: a[4]` when the outer loop's value happened to exceed `M2`'s shape. **(c)** `IsDeterminantNonZeroByEval` does `abs(detvalue) > thresh` on a sympy expression that may still contain free symbols (e.g. `new_r00` introduced by Tee remapping). The result is a symbolic `Relational` that can't be cast to bool. Substitute any remaining free symbols with a fixed rational (13/17) before the numeric comparison; as a final safety net, catch `TypeError` and conservatively return non-singular so the solver can proceed. |

## What was deliberately not vendored

The legacy `python/ikfast_sympy0_6.py` and `python/ikfast_generator_cpp_sympy0_6.py` are not vendored. They are sympy-0.6-era backports kept upstream for historical reasons; sympy modernization on the active `ikfast.py` (issue [#6](https://github.com/siddhss5/ikfastpy/issues/6)) targets a current sympy directly, so these files would be dead code here.

## Re-pinning to a newer upstream commit

```bash
SHA=<new-sha-from-rdiankov/openrave>
BASE=https://raw.githubusercontent.com/rdiankov/openrave/$SHA/python
curl -sSf "$BASE/ikfast.py"                  -o src/ssik/_vendor/ikfast.py
curl -sSf "$BASE/ikfast_generator_cpp.py"    -o src/ssik/_vendor/ikfast_generator_cpp.py
curl -sSf "$BASE/ikfast.h"                   -o src/ssik/_vendor/ikfast.h
shasum -a 256 src/ssik/_vendor/{ikfast.py,ikfast_generator_cpp.py,ikfast.h}
```

Then:
1. Update the *Upstream baseline* section above with the new SHA, date, and hashes.
2. Re-apply every entry in the *Local modifications* table (each PR is the canonical reference).
3. Add a row for the re-pin itself if any conflicts had to be resolved.
