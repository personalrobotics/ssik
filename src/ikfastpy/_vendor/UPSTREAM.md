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
| [#?](https://github.com/siddhss5/ikfastpy/pulls) | [#4](https://github.com/siddhss5/ikfastpy/issues/4) | `ikfast.py`, `ikfast_generator_cpp.py` | Strip OpenRAVE imports (keep fallback branches inline); rename loggers to `ikfastpy.ikfast`; replace `ikfast.py` `__main__` CLI with a `NotImplementedError` stub; remove the docstring section describing the now-removed CLI. |

## What was deliberately not vendored

The legacy `python/ikfast_sympy0_6.py` and `python/ikfast_generator_cpp_sympy0_6.py` are not vendored. They are sympy-0.6-era backports kept upstream for historical reasons; sympy modernization on the active `ikfast.py` (issue [#6](https://github.com/siddhss5/ikfastpy/issues/6)) targets a current sympy directly, so these files would be dead code here.

## Re-pinning to a newer upstream commit

```bash
SHA=<new-sha-from-rdiankov/openrave>
BASE=https://raw.githubusercontent.com/rdiankov/openrave/$SHA/python
curl -sSf "$BASE/ikfast.py"                  -o src/ikfastpy/_vendor/ikfast.py
curl -sSf "$BASE/ikfast_generator_cpp.py"    -o src/ikfastpy/_vendor/ikfast_generator_cpp.py
curl -sSf "$BASE/ikfast.h"                   -o src/ikfastpy/_vendor/ikfast.h
shasum -a 256 src/ikfastpy/_vendor/{ikfast.py,ikfast_generator_cpp.py,ikfast.h}
```

Then:
1. Update the *Upstream baseline* section above with the new SHA, date, and hashes.
2. Re-apply every entry in the *Local modifications* table (each PR is the canonical reference).
3. Add a row for the re-pin itself if any conflicts had to be resolved.
