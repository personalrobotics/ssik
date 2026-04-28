# 7. The KinBody-input bridge

!!! warning "Scaffolding"
    Outline below; prose to be filled in. Implementation: [`src/ssik/kinematics/poe_to_dh.py`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/kinematics/poe_to_dh.py), [`src/ssik/solvers/ikgeo/general_6r.py`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/general_6r.py). Tracking issue [#79](https://github.com/siddhss5/ikfastpy/issues/79).

## What this chapter covers

How POE-form `KinBody` inputs (the canonical ssik / IK-Geo / EAIK convention) get bridged to the standard distal-DH representation that the Raghavan–Roth solver internally uses.

### POE vs DH

- **POE (Product of Exponentials):** every joint contributes a `T_left @ R_axis(joint.axis, q) @ T_right` factor. Conventional in modern robotics (Murray–Li–Sastry); easy to express directly from URDF or MJCF.
- **DH (Denavit–Hartenberg):** every joint contributes $A_i = R_z(\theta_i) T_z(d_i) T_x(a_i) R_x(\alpha_i)$. Frame placement is constrained: $z_i$ must be the joint axis, $x_i$ must lie along the common perpendicular to $z_{i-1}$. Conventional in textbook IK derivations (Spong, Tsai).

The Raghavan–Roth solver's algebra ([Chapter 4](04_raghavan_roth.md)) is in DH form. The user's input is in POE form. We need to bridge.

### `poe_to_dh(kb) -> DhWithOffset`

Returns:

```python
@dataclass
class DhWithOffset:
    alpha: NDArray         # length-6, twist angles
    a: NDArray             # length-6, link lengths
    d: NDArray             # length-6, link offsets
    theta_offset: NDArray  # length-6, per-joint angle offset
    t_pre: NDArray         # 4x4, world to DH frame 0
    t_post: NDArray        # 4x4, DH frame n to user EE
```

Such that for every $q$:

$$
\mathrm{FK_{POE}}(q) = T_{\mathrm{pre}}\, \mathrm{FK_{DH}}(q + \theta_{\mathrm{offset}})\, T_{\mathrm{post}}.
$$

### Algorithm

1. Walk the POE chain at $q = 0$: extract joint axes $z_i$ in world frame, joint origins $p_i$ in world frame, and the home-pose end-effector transform $T_{\mathrm{home}} = \mathrm{FK_{POE}}(0)$.
2. Place DH frames 0 to $n$ in world coords:
   - $z_i$ = joint $(i+1)$'s axis in world (i.e., `axes_world[i]` for $i < n$); $z_n$ = $T_{\mathrm{home}}$'s z-axis.
   - Origin of frame $i$: foot of common perpendicular between $z_{i-1}$ and $z_i$ on $z_i$'s line.
   - $x_i$ direction: along the common perpendicular, from $z_{i-1}$ toward $z_i$. For parallel axes use the foot-to-foot direction; for intersecting axes use $(z_{i-1} \times z_i)$ normalised.
3. Read off $(\alpha_i, a_i, d_i, \theta_{\mathrm{offset},i})$ per transition by computing signed angles + projected distances.
4. Build $T_{\mathrm{pre}}$ from frame 0's $(x_0, y_0, z_0, \mathrm{origin})$ as columns of the rotation block + translation.
5. Build $T_{\mathrm{post}}$ to absorb any residual rotation between DH frame $n$ at $q = \theta_{\mathrm{offset}}$ and the actual $T_{\mathrm{home}}$.

### The load-bearing bug we fixed

The original implementation hard-coded $z_0 = (0, 0, 1)$ and $T_{\mathrm{pre}} = I$. **This is correct for any arm whose joint-1 axis aligns with world +z** — UR5, Puma 560, most commercial arms. UR5's round-trip test passed for 100 random poses across 3 random seeds at machine precision.

JACO 2 broke it. The MJCF places `link_1` with `quat = (0, 0, 1, 0)` — a 180° rotation about world y, which flips the local +z axis to world −z. So `joints[0].axis` in world frame is $(0, 0, -1)$, not $(0, 0, +1)$. With $z_0 = +z$ hardcoded, the DH chain rotates joint 1 around the wrong axis, the bridge $T_{\mathrm{pre}}^{-1} T\, T_{\mathrm{post}}^{-1}$ doesn't match $\mathrm{FK_{DH}}(q + \theta_{\mathrm{offset}})$, and `||bridge_residual||` is 0.81 instead of $10^{-15}$.

### The fix

Set $z_0 = $ `axes_world[0]` (joint 1's actual axis in world frame), $\mathrm{origin}_0 = $ `origins_world[0]` (joint 1's actual origin), $x_0$ chosen as the world-x projection onto the plane perpendicular to $z_0$ (or world-y if $z_0$ is too parallel to x). $T_{\mathrm{pre}}$ then has $(x_0, y_0, z_0, \mathrm{origin}_0)$ as its columns and reduces to identity for commercial-arm conventions while absorbing the rotation/translation discrepancy otherwise. UR5 round-trips still pass; JACO 2 round-trips now pass at machine precision.

### Lesson

UR5 and Puma 560 are *too well-behaved* to be the only test fixture for kinematic-bridge code. They mask convention bugs that real arms with non-trivial base orientations expose immediately. Real-MJCF fixtures matter (#80).

### Caching

`poe_to_dh(kb)` depends only on the chain's geometry, not on any IK target. ssik caches the result on the kb instance (`kb._ssik_dh_with_offset_cache`); subsequent calls are free. ([Tier 1 of #86](https://github.com/siddhss5/ikfastpy/pull/88).)

## References

- Spong, M. W. *Robot Modeling and Control,* §3.2 (DH frame placement).
- Murray–Li–Sastry, *A Mathematical Introduction to Robotic Manipulation* (POE).
