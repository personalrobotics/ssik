# prebuilt/

Pre-built `ssik` IK modules for popular robot arms. Each `.py` file is a
self-contained artifact emitted by `ssik build` — it bakes the per-arm
KinBody constants, dispatched solver choice, and any cached symbolic
preprocessing into a single Python module. **No URDF parsing, no `urchin`
dependency, no cold-cache work at import time** — just a `solve(T)` function.

## Usage

```python
import sys
sys.path.insert(0, "prebuilt")
import ur5_ik

import numpy as np
T_target = np.eye(4)
T_target[:3, 3] = [0.5, 0.1, 0.3]
sols, is_ls = ur5_ik.solve(T_target)
```

Or use them via the top-level `Manipulator` class for a uniform API:

```python
import ssik
arm = ssik.Manipulator.from_urdf("tests/fixtures/ur5.urdf",
                                  base="base_link", ee="ee_link")
sols, is_ls = arm.ik(T_target)
```

The artifact's `solve()` and `Manipulator.ik()` produce identical results;
the artifact is faster on first call (no URDF parsing, no symbolic
preprocessing) and cleaner to ship in production stacks.

## What's included

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `ur5_ik.py` | `ikgeo.three_parallel` | <1 s | ~25 KB |
| `puma560_ik.py` | `ikgeo.spherical_two_parallel` | <1 s | ~28 KB |
| `iiwa14_ik.py` | `seven_r.srs` | <1 s | ~30 KB |
| `gen3_ik.py` | `seven_r.srs_polished` | <1 s | ~30 KB |
| `jaco2_ik.py` | `ikgeo.general_6r` (RR + AE-3) | ~30 s | ~72 KB |
| `franka_panda_ik.py` | `jointlock.seven_r` (tier-0 inner) | ~1 s | ~24 KB |
| `rizon4_ik.py` | `jointlock.seven_r` + cached-RR | ~7 min | ~330 KB |
| `kassow_kr810_ik.py` | `jointlock.seven_r` + cached-RR | ~20 min | ~530 KB |

The slow ones (`rizon4_ik`, `kassow_kr810_ik`) carry the cached
Raghavan-Roth symbolic derivations as base85-encoded zlib-compressed pickle
blobs. Module-init takes ~5 seconds (deserialise + re-`lambdify`), then
every IK call hits warm-cache speed.

## Examples that use these

See [`examples/`](../examples/) for runnable scripts:

- `01_ur5_quickstart.py` — basic API tour using `ur5_ik`
- `02_jaco2_non_pieper.py` — non-Pieper 6R using `jaco2_ik`
- `03_gen3_polished_srs.py` — approximate-SRS using `gen3_ik`
- `04_compare_vs_eaik.py` — measured benchmark vs EAIK over 100 random poses

## Regenerating

These files are committed to the repo and act as **codegen-drift snapshot
tests** — `tests/test_artifact_snapshots.py` re-emits them and asserts
byte-equal against the committed copy.

If you change `ssik.core.codegen` or any solver's dispatch reasoning, the
snapshot test will fail. Regenerate with:

```bash
uv run python scripts/regen_artifacts.py                 # fast arms only (~30 s)
uv run python scripts/regen_artifacts.py --include-slow  # also rebuild rizon4 + kassow (~30 min)
```

Then commit the updated `prebuilt/*.py` alongside your codegen change so
reviewers can see the user-facing diff.
