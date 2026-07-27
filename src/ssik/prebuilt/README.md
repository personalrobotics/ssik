# `ssik.prebuilt`

Pre-built `ssik` IK modules for popular robot arms, shipped with the wheel.
Each `.py` file is a self-contained artifact emitted by `ssik build`. It
bakes the per-arm KinBody constants, dispatched solver choice, and any cached
symbolic preprocessing into a single Python module. **No URDF parsing, no
`urchin` dependency, no cold-cache work at import time**: just a `solve(T)`
function.

## Usage

```python
from ssik.prebuilt import ur5_ik
import numpy as np

T_target = np.eye(4)
T_target[:3, 3] = [0.5, 0.1, 0.3]
sols = ur5_ik.solve(T_target)
```

Or use them via the top-level `Manipulator` class for a uniform API:

```python
import ssik
arm = ssik.Manipulator.from_urdf("tests/fixtures/ur5.urdf",
                                  base="base_link", ee="ee_link")
sols = arm.ik(T_target)
```

The artifact's `solve()` and `Manipulator.ik()` produce identical results;
the artifact is faster on first call (no URDF parsing, no symbolic
preprocessing) and cleaner to ship in production stacks.

## What's included

<!-- AUTOGEN:prebuilt_readme_table -->
<details>
<summary><b>Universal Robots</b>: <code>ssik.prebuilt.universal_robots</code> (11 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `ur5_ik` | `ikgeo.three_parallel` | <1 s | ~26 KB |
| `ur3e_ik` | `ikgeo.three_parallel` | <1 s | ~33 KB |
| `ur5e_ik` | `ikgeo.three_parallel` | <1 s | ~33 KB |
| `ur10e_ik` | `ikgeo.three_parallel` | <1 s | ~33 KB |
| `ur16e_ik` | `ikgeo.three_parallel` | <1 s | ~33 KB |
| `ur20_ik` | `ikgeo.three_parallel` | <1 s | ~33 KB |
| `ur30_ik` | `ikgeo.three_parallel` | <1 s | ~33 KB |
| `ur7e_ik` | `ikgeo.three_parallel` | <1 s | ~10 KB |
| `ur12e_ik` | `ikgeo.three_parallel` | <1 s | ~10 KB |
| `ur15_ik` | `ikgeo.three_parallel` | <1 s | ~10 KB |
| `ur18_ik` | `ikgeo.three_parallel` | <1 s | ~10 KB |

</details>

<details>
<summary><b>Unimation</b>: <code>ssik.prebuilt.unimation</code> (1 arm)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `puma560_ik` | `ikgeo.spherical_two_parallel` | <1 s | ~27 KB |

</details>

<details>
<summary><b>Kinova</b>: <code>ssik.prebuilt.kinova</code> (3 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `jaco2_ik` | `ikgeo.general_6r` | ~25 s | ~73 KB |
| `gen3_ik` | `seven_r.srs_polished` | <1 s | ~10 KB |
| `gen3_lite_ik` | `ikgeo.general_6r` | <1 s | ~10 KB |

</details>

<details>
<summary><b>KUKA</b>: <code>ssik.prebuilt.kuka</code> (1 arm)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `iiwa14_ik` | `seven_r.srs` | <1 s | ~9 KB |

</details>

<details>
<summary><b>Franka</b>: <code>ssik.prebuilt.franka</code> (2 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `panda_ik` | `seven_r.spherical_shoulder` | <1 s | ~22 KB |
| `fr3_ik` | `seven_r.spherical_shoulder` | <1 s | ~22 KB |

</details>

<details>
<summary><b>UFactory</b>: <code>ssik.prebuilt.ufactory</code> (2 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `xarm7_ik` | `seven_r.spherical_shoulder_polished` | <1 s | ~22 KB |
| `xarm6_ik` | `ikgeo.general_6r` | ~15 s | ~70 KB |

</details>

<details>
<summary><b>Unitree</b>: <code>ssik.prebuilt.unitree</code> (1 arm)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `z1_ik` | `ikgeo.three_parallel` | <1 s | ~23 KB |

</details>

<details>
<summary><b>AgileX</b>: <code>ssik.prebuilt.agilex</code> (1 arm)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `piper_ik` | `ikgeo.general_6r` | ~25 s | ~88 KB |

</details>

<details>
<summary><b>Flexiv</b>: <code>ssik.prebuilt.flexiv</code> (2 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `rizon4_ik` | `jointlock.seven_r` + cached-RR | ~8 min | ~1052 KB |
| `rizon10_ik` | `jointlock.seven_r` + cached-RR | ~8 min | ~992 KB |

</details>

<details>
<summary><b>Kassow</b>: <code>ssik.prebuilt.kassow</code> (1 arm)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `kr810_ik` | `jointlock.seven_r` + cached-RR | ~4 min | ~729 KB |

</details>

<details>
<summary><b>FANUC</b>: <code>ssik.prebuilt.fanuc</code> (7 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `crx3ia_ik` | `ikgeo.general_6r` | ~28 s | ~63 KB |
| `crx5ia_ik` | `ikgeo.general_6r` | ~24 s | ~61 KB |
| `crx10ia_ik` | `ikgeo.general_6r` | ~28 s | ~61 KB |
| `crx10ialp_ik` | `ikgeo.general_6r` | ~26 s | ~61 KB |
| `crx20ial_ik` | `ikgeo.general_6r` | ~26 s | ~61 KB |
| `crx30ia_ik` | `ikgeo.general_6r` | ~31 s | ~63 KB |
| `crx10ial_ik` | `ikgeo.general_6r` | ~36 s | ~76 KB |

</details>

<details>
<summary><b>I2RT</b>: <code>ssik.prebuilt.i2rt</code> (2 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `yam_ik` | `ikgeo.general_6r` | ~36 s | ~106 KB |
| `big_yam_ik` | `ikgeo.general_6r` | ~36 s | ~106 KB |

</details>

<details>
<summary><b>Enactic OpenArm</b>: <code>ssik.prebuilt.openarm</code> (2 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `left_ik` | `seven_r.srs` | <1 s | ~9 KB |
| `right_ik` | `seven_r.srs` | <1 s | ~9 KB |

</details>

<details>
<summary><b>Galaxea</b>: <code>ssik.prebuilt.galaxea</code> (2 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `r1pro_left_ik` | `seven_r.srs` | <1 s | ~12 KB |
| `r1pro_right_ik` | `seven_r.srs` | <1 s | ~12 KB |

</details>

<details>
<summary><b>Standard Bots</b>: <code>ssik.prebuilt.standard_bots</code> (3 arms)</summary>

| Arm | Solver | Build time | Artifact size |
|---|---|:---:|:---:|
| `thor_ik` | `ikgeo.three_parallel` | <1 s | ~26 KB |
| `core_ik` | `ikgeo.three_parallel` | <1 s | ~26 KB |
| `spark_ik` | `ikgeo.three_parallel` | <1 s | ~26 KB |

</details>
<!-- /AUTOGEN -->

The slow ones (`rizon4_ik`, `kassow_kr810_ik`) carry the cached
Raghavan-Roth symbolic derivations as base85-encoded zlib-compressed pickle
blobs. Module-init takes ~5 seconds (deserialise + re-`lambdify`), then
every IK call hits warm-cache speed.

## Examples that use these

See [`examples/`](../../../examples/) for runnable scripts:

- `01_ur5_quickstart.py`: basic API tour using `ur5_ik`
- `02_jaco2_non_pieper.py`: non-Pieper 6R using `jaco2_ik`
- `03_gen3_polished_srs.py`: approximate-SRS using `gen3_ik`
- `04_compare_vs_eaik.py`: measured benchmark vs EAIK over 100 random poses

## Regenerating

These files are committed to the repo and act as **codegen-drift snapshot
tests**: `tests/test_artifact_snapshots.py` re-emits them and asserts
byte-equal against the committed copy.

If you change `ssik.core.codegen` or any solver's dispatch reasoning, the
snapshot test will fail. Regenerate with:

```bash
uv run python scripts/regen_artifacts.py                 # fast arms only (~30 s)
uv run python scripts/regen_artifacts.py --include-slow  # also rebuild rizon4 + kassow (~30 min)
```

Then commit the updated `src/ssik/prebuilt/*.py` alongside your codegen
change so reviewers can see the user-facing diff.
