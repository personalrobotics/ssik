# 7. The KinBody-input bridge

The Raghavan–Roth solver of [Chapter 4](04_raghavan_roth.md) speaks **standard distal-DH** — a kinematic representation where each joint contributes $A_i = R_z(\theta_i) T_z(d_i) T_x(a_i) R_x(\alpha_i)$ and frames must be placed with $z_i$ along the joint axis and $x_i$ along the common perpendicular to $z_{i-1}$. Users speak **POE** (Product of Exponentials) — the modern convention where each joint contributes a `T_left @ R_axis(joint.axis, q) @ T_right` factor and frame placement is whatever the URDF or MJCF says it is.

These two representations encode the same kinematic chain, but the algebra in DH form is incompatible with POE form. Asking the user to convert is a footgun (DH parameterisation has six different conventions in use; nobody agrees which is "standard"). The library converts internally.

This chapter walks through the conversion, the bridging transforms `T_pre` and `T_post`, and the load-bearing bug that hid behind UR5's accidental world-z alignment for months until JACO 2 broke it. Implementation: [`src/ssik/kinematics/poe_to_dh.py`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/kinematics/poe_to_dh.py); the public solver wrapper that uses it: [`src/ssik/solvers/ikgeo/general_6r.py`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/general_6r.py). Tracking issue [#79](https://github.com/siddhss5/ikfastpy/issues/79).

## POE versus DH

**Product of Exponentials** is the convention from Murray–Li–Sastry. Each joint $i$ contributes a transform

$$
J_i(q) = T_{\mathrm{left},i} \cdot R_{\mathrm{axis}}(\hat{a}_i, q_i) \cdot T_{\mathrm{right},i}
$$

where $T_{\mathrm{left},i}, T_{\mathrm{right},i} \in \mathrm{SE}(3)$ are arbitrary rigid transforms and $\hat{a}_i$ is the joint axis in the joint's local frame. The forward kinematics is

$$
\mathrm{FK_{POE}}(q) = \prod_{i=1}^{n} J_i(q_i).
$$

POE is convenient because $T_{\mathrm{left}}$ and $T_{\mathrm{right}}$ can be read directly from URDF `<origin>` tags or MJCF `<body pos quat>` attributes. There's no constraint on frame placement; you describe the chain however the upstream model author described it.

**Standard distal-DH** (Spong) is more constrained. The forward kinematics is

$$
\mathrm{FK_{DH}}(\theta_1, \ldots, \theta_n) = \prod_{i=1}^{n} A_i(\theta_i), \qquad A_i = R_z(\theta_i)\, T_z(d_i)\, T_x(a_i)\, R_x(\alpha_i),
$$

with frame placement constraints:

- $z_i$ along joint $(i+1)$'s axis (so $z_0$ along joint 1's axis).
- Origin of frame $i$ at the foot of the common perpendicular between $z_{i-1}$ and $z_i$, on $z_i$'s line.
- $x_i$ along the common perpendicular, from $z_{i-1}$ to $z_i$.

DH is convenient because the algebra closes elegantly — the $A_i$ structure is tight enough that Raghavan–Roth's loop-closure split works. The constraints are the price.

The conversion `poe_to_dh(kb)` walks the POE chain, computes joint axes and origins in world coordinates, and places DH frames satisfying the constraints.

## The `DhWithOffset` shape

The conversion returns a [`DhWithOffset`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/kinematics/poe_to_dh.py) carrying:

```python
@dataclass
class DhWithOffset:
    alpha: NDArray         # length-6, twist angles
    a: NDArray             # length-6, link lengths
    d: NDArray             # length-6, link offsets
    theta_offset: NDArray  # length-6, per-joint angle offset
    t_pre: NDArray         # 4x4, world frame -> DH frame 0
    t_post: NDArray        # 4x4, DH frame n -> user EE
```

with the contract

$$
\mathrm{FK_{POE}}(q) = T_{\mathrm{pre}}\, \mathrm{FK_{DH}}(q + \theta_{\mathrm{offset}})\, T_{\mathrm{post}}
$$

for every $q$. The `(alpha, a, d)` triple is what the Raghavan–Roth pipeline consumes; `theta_offset` accounts for the user's joint zero positions not aligning with DH's zero positions; and `T_pre`, `T_post` are the bridging rotations that translate between the user's world frame, DH frame 0, DH frame n, and the user's end-effector frame.

For commercial arms with conventional URDF base orientation (joint 1 axis = world +z, base at world origin), `T_pre = I` and `theta_offset` is small. For arms with non-trivial base orientations (JACO 2's `link_1.quat = (0, 0, 1, 0)` rotates joint 1's local +z to world −z), `T_pre` absorbs the rotation/translation discrepancy.

## The conversion algorithm

`poe_to_dh(kb)` does the following:

1. **Walk the POE chain at $q = 0$**: extract joint axes $\hat{a}_i^w$ in world frame (rotated by the cumulative `T_left` chain), joint origins $p_i^w$ in world frame, and the home-pose end-effector transform $T_{\mathrm{home}} = \mathrm{FK_{POE}}(0)$.
2. **Place DH frames 0 to $n$** in world coordinates:
   - Frame 0: $z_0 = \hat{a}_0^w$ (joint 1's axis in world). Origin = $p_0^w$ (joint 1's origin in world). $x_0$ = world-x projected onto the plane perpendicular to $z_0$ (or world-y if too parallel).
   - Frame $i$ for $1 \le i \le n - 1$: $z_i = \hat{a}_i^w$. Origin = foot of common perpendicular between $z_{i-1}$ and $z_i$ on $z_i$'s line. $x_i$ direction along the common perpendicular, oriented from frame $i-1$ to frame $i$. For parallel axes use the foot-to-foot direction; for intersecting axes use $(z_{i-1} \times z_i)$ normalised.
   - Frame $n$: $z_n$ = $T_{\mathrm{home}}$'s z-axis. Origin = $T_{\mathrm{home}}$'s translation. $x_n$ = projection of $x_{n-1}$ onto the plane perpendicular to $z_n$.
3. **Read off $(\alpha_i, a_i, d_i, \theta_{\mathrm{offset},i})$** for each transition $i \to i+1$ via signed-angle and projected-distance computations against the placed frames.
4. **Compute $T_{\mathrm{pre}}$** from frame 0's $(x_0, y_0, z_0, \mathrm{origin})$ assembled as a $4 \times 4$ rigid transform.
5. **Compute $T_{\mathrm{post}}$** as $\mathrm{frame}_n^{-1} \cdot T_{\mathrm{home}}$ — absorbs any residual rotation between DH frame $n$ at $q = \theta_{\mathrm{offset}}$ and the actual user-facing EE pose.

The output is cached on the `KinBody` instance as `kb._ssik_dh_with_offset_cache` (the conversion depends only on the chain, not on any IK target) — see the speed work in [PR #88](https://github.com/siddhss5/ikfastpy/pull/88).

## The load-bearing bug

The original implementation hard-coded $z_0 = (0, 0, 1)$ and $T_{\mathrm{pre}} = I$. **This is correct for any arm whose joint-1 axis aligns with world +z and whose joint-1 origin is at the world origin** — UR5, Puma 560, most commercial 6R arms with conventional base orientation. UR5's round-trip test passed for 100 random poses across 3 random seeds at machine precision, so the function looked correct.

JACO 2 broke it.

The MJCF places `link_1` with `pos = (0, 0, 0.15675)` and **`quat = (0, 0, 1, 0)`** — a 180° rotation about world y. Applied to the local +z joint-1 axis, this gives a world-frame axis of $(0, 0, -1)$. Joint 1 rotates around world −z, not world +z.

With $z_0 = +z$ hardcoded, the DH chain's $A_1$ rotates the chain around the wrong axis. The bridge $T_{\mathrm{pre}}^{-1} \cdot T \cdot T_{\mathrm{post}}^{-1}$ doesn't equal $\mathrm{FK_{DH}}(q + \theta_{\mathrm{offset}})$, which means the IK solver receives a target pose in the wrong frame. The Frobenius residual `||bridge - DH(q*+offset)||` measured 0.81 instead of 1e-15. Every IK call on JACO 2 returned solutions that were wildly wrong on the user's POE chain even though they were correct in the (wrong) DH frame.

Worse: this didn't show up as a numerical instability or a flagged failure. The solver returned candidates with low FK residual *against the wrong target*, so `is_ls=False` and the candidates looked valid. They just didn't reproduce the user's target pose. We caught it only by running the full `KinBody → poe_to_dh → solve_all_ik → POE-FK` round-trip and finding `||FK_POE(q) - T*||` = 0.6 on every solution.

## The fix

Replace the hardcoded $z_0$ with the actual world-frame axis of joint 1:

```python
z_0 = axes_world[0] / np.linalg.norm(axes_world[0])
```

Pick a sensible perpendicular as $x_0$ (project world-x onto the plane perpendicular to $z_0$; fall back to world-y if too parallel). Compute origin $\mathrm{origin}_0 = $ `origins_world[0]`. Build $T_{\mathrm{pre}}$ from those vectors as columns of the rotation block + the translation.

For UR5 (joint-1 axis = world +z, joint-1 origin at world origin) this gives $z_0 = (0, 0, 1)$, $x_0 = (1, 0, 0)$, $\mathrm{origin}_0 = (0, 0, 0)$, and $T_{\mathrm{pre}} = I$. The original UR5 round-trip tests still pass at machine precision.

For JACO 2 ($z_0 = (0, 0, -1)$, $\mathrm{origin}_0 = (0, 0, 0.15675)$, $x_0 = $ projection of $(1, 0, 0)$), $T_{\mathrm{pre}}$ becomes a non-trivial rigid transform. The bridge equation now closes at machine precision: `||bridge - DH(q* + offset)||` measures 1.6e-15 instead of 0.81. JACO 2 IK works.

Verifying: the [`tests/test_poe_to_dh.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/test_poe_to_dh.py) round-trip test passes on UR5 over 300 random poses (3 seeds × 100 poses) at 1e-10 atol. The [`tests/test_jaco2_general_6r.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/test_jaco2_general_6r.py) round-trip passes on JACO 2 over the seed pose plus 4 keyframes at 1e-5 atol (the looser tolerance reflects the tier-2 RR algebraic floor, not the bridge — the bridge alone is at machine precision).

## What this lesson is

UR5 and Puma 560 are *too well-behaved* to be the only test fixtures for kinematic-bridge code. Both have base orientations that align with world axes. Both have joint 1 at the world origin. Both have right-handed chains where every quaternion in the chain happens to have its w-component positive. The conventions are so uniform that any of half a dozen subtly-different bridge implementations would pass UR5's round-trip — including the wrong one.

JACO 2's MJCF carries a deliberate 180° rotation in `link_1` because the upstream Kinova URDF defines its base frame that way (the arm's data-sheet "joint 1" is conventionally rotation around its mounting flange's normal, which on the j2n6s200 points downward when the arm is mounted on a workbench). It was the first fixture in our suite where any of the conventions UR5 takes for granted were violated. And it broke immediately.

This is the case for **real-arm fixtures, not synthetic-arm fixtures**. Synthetic fixtures share the convenience-of-construction biases that make testing easy. Real fixtures break those biases, in different ways for different arms, and they break them on day one. ssik tracks adding more real fixtures (Agilex Piper, Flexiv Rizon, KUKA iiwa, Franka Panda) under [#80](https://github.com/siddhss5/ikfastpy/issues/80) — each one will probably surface another similar convention bug.

The right test isn't "did UR5 round-trip pass?" — it's "did UR5 round-trip pass *and* did the round-trip pass on an arm whose base orientation is deliberately not world-aligned?" That's what the JACO 2 fixture is for. That's what real fixtures are for in general.

## Caching

The conversion is geometrically determined — it depends only on the chain's structure, not on any IK target. The expensive parts (foot-of-perpendicular calculations, signed-angle computations) recompute every call against the same input. ssik stores the result on the kb instance as `kb._ssik_dh_with_offset_cache`; subsequent calls return the cached result. Garbage-collected with the kb. Saves ~1.6 ms median per IK call on JACO 2 (the previous-largest single hot-spot in the warm-cache profile; see [PR #88](https://github.com/siddhss5/ikfastpy/pull/88)).

For users who rebuild the `KinBody` per IK call (e.g. URDF-loader inside the loop), the cache misses every time and you pay the conversion cost. The recommended pattern is **build kb once outside the IK loop**, reuse it.
