# Singh-Kreutz SRS-class 7R analytical IK — design + prototype notes

Implementation reference for #187. The predicate `is_srs_7r` lands in PR #188 as the foundation; this doc captures the algorithmic prototype I validated step-by-step on iiwa14.

## Setup

For an SRS-class 7R chain:

- Shoulder pivot `S` (axes 0, 1, 2 all pass through it).
- Elbow joint at index 3 with axis `u_e`.
- Wrist pivot `W` (axes 4, 5, 6 all pass through it).
- `L_se = ||E_home - S||` (shoulder-to-elbow link length, fixed).
- `L_ew = ||W_home - E_home||` (elbow-to-wrist link length, fixed).
- `ee_offset_local = ee_pos_at_q0 - W_home` (EE position in wrist's tool frame).

These are extracted from the `KinBody` once per arm; they're geometric constants of the chain.

## Algorithm (verified step-by-step on iiwa14)

### Step 1 — target wrist pivot

```python
W_target = T_target.translation - T_target.rotation @ ee_offset_local
```

**Verified on iiwa14** with `q_truth = [0.3, -0.4, 0.7, 0.5, 0.6, -0.5, 0.2]`: `W_target` matches the FK-computed wrist pivot (with `q[4..6] = 0`) to 2.3e-16. ✓

### Step 2 — cosine rule for elbow joint angle q_3

```python
SW = W_target - S
d_sw = ||SW||
cos_interior = (L_se² + L_ew² - d_sw²) / (2 · L_se · L_ew)
q_3_branch_a = π - acos(cos_interior)   # elbow "down"
q_3_branch_b = -(π - acos(cos_interior))  # elbow "up"
```

**Verified on iiwa14**: both `q_3 = ±0.5` candidates produced, truth `q_3 = 0.5` recovered to 4e-16. ✓

The two branches correspond to elbow-up vs elbow-down configurations (mirror symmetry in the plane spanned by SW and the elbow's perpendicular).

### Step 3 — swivel parameterization of elbow position

For each swivel angle θ:

```python
u_sw = SW / d_sw  # unit shoulder-to-wrist direction
x_c = (L_se² - L_ew² + d_sw²) / (2 · d_sw)  # distance from S to circle center along u_sw
r_circle = sqrt(L_se² - x_c²)               # radius of elbow circle

# Two perpendicular unit vectors spanning the plane perpendicular to u_sw.
# Pick u_perp1 as any unit vector perpendicular to u_sw (e.g., world-z minus
# its component along u_sw, normalized; if u_sw IS z, use world-x).
# u_perp2 = u_sw × u_perp1.
E_target(θ) = S + x_c · u_sw + r_circle · (cos(θ) · u_perp1 + sin(θ) · u_perp2)
```

**Verified on iiwa14**: at home (q = 0), `E_home = (0, 0, 0.78)`, `S = (0, 0, 0.36)`, `L_se = 0.42`, `L_ew = 0.40`. ✓

### Step 4 — recover (q_0, q_1) from elbow position

For iiwa14's ZY shoulder (axes z, y at home, before q_2 roll):

```python
d = (E_target - S) / L_se  # unit elbow direction from S
q_1 = ±acos(d_z)                  # 2 branches (+/-)
q_0 = atan2(d_y, d_x)              # adjusted by ±π for the q_1 branch
```

**Verified on iiwa14**: at `E_truth`, recovered `(q_0, q_1) = (-2.84, 0.4)` (the *other* branch from truth `(0.3, -0.4)`). The branch enumeration is necessary — all 2 must be tried.

This step is per-arm: the (q_0, q_1) decomposition formula depends on the canonical axis ordering of the shoulder. For SRS arms with different shoulder axis orderings (e.g., ZX vs ZY), the formula differs. Generic implementation can use `sp1` for the first joint + `sp2` for the (joint 1 + joint 2) composition, OR use `sp5` like the existing IK-Geo `spherical` solver does for its wrist.

### Step 5 — recover q_2 from wrist pivot constraint

The wrist pivot `W(q_0, q_1, q_2, q_3) = E(q_0, q_1) + R_upper_arm(q_0, q_1) · R_z(q_2) · R_elbow(q_3) · (0, 0, L_ew)`.

Given `W_target`, `E_target`, and `q_3`, q_2 is recovered as a single atan2 from the residual:

```python
W_offset = W_target - E_target  # in world frame
# Rotate W_offset back into the body frame at the upper arm.
W_offset_body = R_upper_arm(q_0, q_1)^-1 @ W_offset
# In body frame: lower arm offset at q_2=0, q_3=q_3 is some specific vector.
# Solve q_2 such that R_z(q_2) @ R_elbow(q_3) @ (0, 0, L_ew) = W_offset_body.
# This is SP1 (rotate one vector to another about z-axis).
q_2 = sp1.solve(z_axis, R_elbow(q_3) @ (0, 0, L_ew), W_offset_body)
```

**Not yet verified on iiwa14** — needs implementation + FK closure validation.

### Step 6 — wrist triple from residual rotation

Once `(q_0, q_1, q_2, q_3)` are known, compute the orientation up to the elbow:

```python
R_so_far = R_z(q_0) @ R_y(q_1) @ R_z(q_2) @ R_elbow(q_3)
R_residual = R_so_far^T @ R_target
# R_residual should equal R_z(q_4) @ R_y(q_5) @ R_z(q_6).
# Standard ZYZ Euler decomposition:
q_5 = ±acos(R_residual[2, 2])
q_4 = atan2(R_residual[1, 2], R_residual[0, 2])  # adjusted by π for +/- q_5
q_6 = atan2(R_residual[2, 1], -R_residual[2, 0])
```

**Not yet verified on iiwa14** — needs implementation.

## Branch enumeration

Total candidate IK count per swivel sample:

- q_3: 2 branches (elbow up/down).
- (q_0, q_1): 2 branches per q_3 (sign of q_1).
- q_2: single branch (atan2).
- (q_4, q_5, q_6): 2 branches per (q_0..q_3) (sign of q_5).

Total: 8 candidates per swivel θ. FK closure filters spurious; deduplicate via wrap-to-π on q-vectors.

## Cross-validation strategy

Bulletproof contract:

1. Hand-picked iiwa14 `q_truth` recovery at machine precision (FK closure ≤ 1e-10).
2. 100-pose hypothesis fuzz on iiwa14 with FK roundtrip: every returned q must FK-close.
3. Cross-check vs `jointlock + HP` on iiwa14: same IK set within wrap-to-π at 1e-6.
4. Cross-check vs `gen_six_dof` numerical oracle ... nope, gen_six_dof is gone (#185). Use HP as the oracle.

## Performance target

Sub-millisecond per IK warm cache. Python overhead dominates; subproblem composition is the right abstraction (SP1 + SP2 + SP4 + ZYZ + atan2 = ~10 numerical ops in the hot path).

If sub-ms isn't reachable in pure Python, Cython compile per the #186 pattern (HP precedent).

## Reusable utilities (sharable beyond SRS solver)

1. `axes_meet_at_common_point` — landed in #188.
2. `is_srs_7r` predicate — landed in #188.
3. `swivel_to_elbow(S, W, L_se, L_ew, theta)` — geometric helper. Reusable for any future SRS-style solver and for the redundancy-resolution work (#148).
4. `zyz_euler_decompose(R)` — extract ZYZ Euler angles. Reusable for the existing IK-Geo `spherical` family (currently inlines this).
5. `sp1.solve(axis, p_from, p_to)` — already exists.

## Implementation checklist

- [x] Predicate `is_srs_7r` (#188).
- [ ] `ssik.solvers.seven_r.srs.solve(kb, T_target, lock_samples=16, allow_refinement=True, ...)`
- [ ] Branch enumeration + FK closure filter.
- [ ] Dispatcher tier-0 wiring.
- [ ] Bulletproof tests on iiwa14.
- [ ] Cross-validate against `jointlock + HP`.
- [ ] README perf table update.
