# Solver data, schema 1

This is the normative description of the record that `ssik.cpp.solver_data(arm)`
returns and that `ssik::Solver` (`ssik_cpp/solve.hpp`) reads. Python and C++
hold the same record: `ssik.cpp.SolverData` and `ssik::SolverData`
(`ssik_cpp/solver_data.hpp`). How to use it is in
[`cpp/README.md`](https://github.com/personalrobotics/ssik/blob/main/cpp/README.md#use-it-from-the-python-wheel);
what semver covers is in the [semver policy](../semver_policy.md).

## The record

| field | Python | C++ | meaning |
|---|---|---|---|
| family | `str` | `std::string` | the solver family, `Manipulator.solver_name`; one of `ssik.cpp.NATIVE_FAMILIES` (`ssik::native_families()`) |
| schema | `int` | `int` | the schema number; this page describes **1** (`SOLVER_DATA_SCHEMA`, `ssik::kSolverDataSchema`) |
| dof | `int` | `Solver::dof()` | 6 or 7 (Python only; C++ derives it from the family) |
| arrays | `Mapping[str, ndarray]` | `std::map<std::string, DataArray>` | flat named arrays, below |

Every array is **float64** or **int32**, C order (row-major); a scalar has
shape `()`. In Python the arrays are read-only and C-contiguous. In C++ a
`DataArray` holds its `dtype`, its `shape` and its values in `f64` or `i32`
(the other vector is empty). Every float64 value is finite.

`ssik::Solver` is strict. It accepts only `schema == kSolverDataSchema` and a
family in `native_families()`; otherwise it throws `UnsupportedSolverData`. It
requires every array this page lists for the family, with exactly the listed
dtype and shape and the listed index ranges, and no other array; otherwise it
throws `SolverDataError` naming the array. Both derive from
`std::invalid_argument`.

The record maps one-to-one onto an `.npz`:
`np.savez(path, schema=d.schema, family=d.family, **d.arrays)`. ssik does not
ship a C++ `.npz` reader.

## Arrays

N is the dof. "i32" marks an int32 array; every other array is float64.
Ranges are half-open.

### Common to every family

The arrays of `ssik.cpp.joint_data(arm)`, the fields of `ssik::JointConsts<N>`
and `ssik::JointLimits<N>`. Forward kinematics is
`T = prod_i t_left[i] @ Joint(axis[i], q[i]) @ t_right[i]`.

| name | shape | meaning |
|---|---|---|
| `axis` | (N, 3) | joint axis in its local frame |
| `t_left`, `t_right` | (N, 4, 4) | the fixed transforms before and after each joint. For `ikgeo.spherical_two_parallel` these are in the canonical wrist gauge that solver requires; forward kinematics is the arm's own |
| `joint_type` | (N,) i32 | 0 revolute, 1 prismatic |
| `lo`, `hi` | (N,) | joint limits; `0, 0` when the joint has none |
| `present` | (N,) i32 | 1 when the joint has limits, else 0 |

For `jointlock.seven_r` these describe the full 7-joint arm.

### `ikgeo.three_parallel`, `ikgeo.spherical_two_parallel`

No further arrays.

### `ikgeo.general_6r`

The Raghavan-Roth constants (`ssik::RrConsts`) and the numeric coefficient
tensor (`ssik::rr_detail::RrCoeffTensor`) of the arm.

| name | shape | meaning |
|---|---|---|
| `alpha`, `a`, `d`, `theta_offset` | (6,) | the arm's DH parameters and joint-angle offsets |
| `t_pre_inv`, `t_post_inv` | (4, 4) | inverses of the transforms that map the arm's base and flange onto the DH chain |
| `linearity_joint`, `drop_joint` | () i32 | the AE-3 linearity joint and the eliminated joint |
| `left_bilinear`, `right_bilinear` | (2,) i32 | the bilinear joint pairs |
| `p_sin`, `p_cos` | (14, 9) | the target-independent coefficient blocks |
| `mono_factors` | (M, 3) i32 | the monomial basis: each row is up to three factor indices in [0, 12) into the target's top three rows in the DH frame (row-major), padded with -1 |
| `po_rc`, `po_mono`, `po_coeff` | (P, 2) i32, (P,) i32, (P,) | the target-dependent `p_one` block as COO: row in [0, 14), column in [0, 9), monomial in [0, M), coefficient |
| `q_rc`, `q_mono`, `q_coeff` | (Q, 2) i32, (Q,) i32, (Q,) | the `q` block the same way, column in [0, 8) |

`linearity_joint`, the two `left_bilinear`, the two `right_bilinear` and
`drop_joint` together name each of the joints 0 to 5 exactly once.

### `seven_r.srs`, `seven_r.srs_polished`

The fields of `ssik::SrsConsts`.

| name | shape | meaning |
|---|---|---|
| `l_se`, `l_ew` | () | shoulder-to-elbow and elbow-to-wrist lengths |
| `ee_offset_local` | (3,) | flange position relative to the wrist centre, in the wrist frame |
| `shoulder_pivot` | (3,) | shoulder centre |
| `upper_home`, `forearm_home` | (3,) | elbow minus shoulder and wrist minus elbow, at zero configuration |
| `r_post_wrist` | (3, 3) | rotation of the last joint's `t_right` |
| `elbow_index` | () i32 | the elbow joint, in [0, 7) |
| `general_path` | () i32 | 1 to take the general (non-ZYZ or offset-wrist) path, else 0 |

### `seven_r.spherical_shoulder`, `seven_r.spherical_shoulder_polished`

| name | shape | meaning |
|---|---|---|
| `coef` | (3, 48) | the affine coefficients of `ssik::SphericalShoulderConsts` |

### `jointlock.seven_r`

One joint is locked at 16 sampled values and the remaining 6R sub-chain is
solved for each. Two inner kernels share the family, told apart by
`inner_kernel`.

| name | shape | meaning |
|---|---|---|
| `inner_kernel` | () i32 | 0 Raghavan-Roth, 1 Husty-Pfurner |
| `lock_idx` | () i32 | the locked joint, in [0, 7) |
| `q_lock` | (16,) | the locked joint's sample values |

**Raghavan-Roth kernel** (`inner_kernel == 0`). The `ikgeo.general_6r` arrays
of each sample's sub-chain, with a leading sample axis S = 16:
`alpha`, `a`, `d`, `theta_offset` (16, 6); `t_pre_inv`, `t_post_inv`
(16, 4, 4); `linearity_joint`, `drop_joint` (16,) i32; `left_bilinear`,
`right_bilinear` (16, 2) i32; `p_sin`, `p_cos` (16, 14, 9). The tensors differ
in length between samples, so they are concatenated along their first axis,
with offsets that give each sample's rows:

| name | shape | meaning |
|---|---|---|
| `mono_factors` | (ΣM, 3) i32 | every sample's monomial basis, concatenated |
| `mono_offsets` | (17,) i32 | sample s owns rows `[mono_offsets[s], mono_offsets[s+1])` |
| `po_rc`, `po_mono`, `po_coeff` | (ΣP, 2) i32, (ΣP,) i32, (ΣP,) | every sample's `p_one` COO, concatenated |
| `po_offsets` | (17,) i32 | sample s owns rows `[po_offsets[s], po_offsets[s+1])` |
| `q_rc`, `q_mono`, `q_coeff` | (ΣQ, 2) i32, (ΣQ,) i32, (ΣQ,) | every sample's `q` COO, concatenated |
| `q_offsets` | (17,) i32 | sample s owns rows `[q_offsets[s], q_offsets[s+1])` |

Each offsets array starts at 0, never decreases and ends at its array's row
count. Monomial indices in `po_mono` and `q_mono` are local to their sample's
basis. The joint-role and index rules of `ikgeo.general_6r` hold per sample.

**Husty-Pfurner kernel** (`inner_kernel == 1`). The fields of
`ssik::HpConsts` and the sub-chain's `JointConsts<6>`, per sample:

| name | shape | meaning |
|---|---|---|
| `t_u`, `t_w_pre` | (16, 4, 8, 2) | the Study-quaternion coefficient blocks; the last axis is the `HpConsts` matrix pair |
| `dh_a`, `dh_l` | (16, 5) | `HpConsts::a[1..5]`, `l[1..5]` |
| `dh_d` | (16, 4) | `HpConsts::d[2..5]` |
| `theta_offset` | (16, 6) | joint-angle offsets |
| `t_pre_inv`, `t_post_inv`, `t_z_neg_d1`, `t_joint6_offset_inv` | (16, 4, 4) | the kernel's frame transforms |
| `right_pv` | (16,) i32 | the right parametric variable, 0 or 1 |
| `drop_idx` | (16,) i32 | the dropped Study row, in [0, 8) |
| `sub_axis` | (16, 6, 3) | the sub-chain's `axis` |
| `sub_t_left`, `sub_t_right` | (16, 6, 4, 4) | the sub-chain's `t_left`, `t_right` |
| `sub_joint_type` | (16, 6) i32 | the sub-chain's `joint_type` |

## Which arms have solver data

- Every arm whose family is in `NATIVE_FAMILIES`, prebuilt or built at
  runtime, except a `jointlock.seven_r` arm built at runtime: its inner kernel
  and lock samples are fixed when an arm is onboarded, so only prebuilt
  jointlock arms ship them.
- `ikgeo.spherical_two_intersecting` and `ikgeo.spherical` have no native
  solver.
- `solver_data` raises `ssik.cpp.UnsupportedFamilyError` (a `ValueError`) in
  those cases, and when a family's native geometry check rejects an arm the
  dispatcher put in that family.

A prebuilt arm's `ikgeo.general_6r` and `jointlock.seven_r` data come from the
tensors shipped with its artifact, so they are the data its
`solve(native=True)` uses.

## Versioning

The schema number changes whenever the set, shape, dtype or meaning of an
existing family's arrays changes. Adding a family does not change it: an older
`ssik::Solver` rejects the new family name with `UnsupportedSolverData`. A
consumer compiles one ssik release's headers and may run with another ssik
installed, so the schema check is what stops it from misreading the data.
