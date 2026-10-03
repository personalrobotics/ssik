# API reference

Auto-generated from docstrings. The public surface is small by design — most users only touch `Manipulator`, `Solution`, and (when relevant) `TolerancePolicy` / `Diagnostic`.

## Entry point: `Manipulator`

::: ssik.Manipulator
    options:
      show_root_heading: false
      members:
        - from_prebuilt
        - from_urdf
        - solve
        - fk
        - self_motion
        - solve_path
        - tracker
        - body_twists
        - free_tail
        - dof
        - solver_name
        - kinbody

## Input validation

Every public entry point that takes a pose or a joint vector checks it once,
before it picks a backend: each prebuilt artifact's `solve` and `fk`,
`Manipulator.solve`, `Manipulator.self_motion`, `Manipulator.solve_path`,
`Tracker` (its `q0`, and the target of every `update`) and the `ssik.teleop`
frame helpers. A malformed call raises the same exception, with the same message, on
`native=True` and `native=False`, and the native extension only ever sees
well-formed input. The implementation is `ssik._solve_inputs`.

**Arrays** (`T_target`, `q_seed`, an artifact's `fk(q)`, `solve_path`'s
`poses` and `q0`). Anything numpy converts to an array of real numbers is
accepted and converted to a C-contiguous `float64` array: `float64`, `float32`
and integer arrays, lists and tuples, Fortran-ordered arrays and strided views,
and object arrays whose elements are all real numbers. The caller's array is
never modified. Rejected:

| Input | Exception |
|---|---|
| `bool`, complex, string, bytes, datetime or structured dtype; an object array with any element that is not a real number (`None`, a string) | `TypeError` |
| Ragged nesting (`[[1, 0, 0, 0], [0, 1, 0]]`) | `ValueError` |
| Wrong shape: `T_target` not `(4, 4)`, `q_seed` or `fk`'s `q` not `(dof,)`, `poses` not `(N, 4, 4)`, `q0` not `(dof,)` or `(k, dof)` | `ValueError` |
| A NaN or infinite entry in `T_target`, `q_seed`, `poses` or `q0` | `ValueError` |

Converting a rejected dtype would invent or drop information: a complex
target's imaginary part, a string's parse. `fk(q)` checks only the shape, so a
NaN joint value gives a NaN pose, as `Manipulator.fk` does.

**Rigid targets.** `T_target` (and each pose of a path) must be a rigid
transform. With `R` its rotation block and `tol =
policy.subproblem_numerical` (`1e-5` by default), it is rejected with a
`ValueError` when

- `||R^T R - I||_F > max(3 * tol, 1e-12)`,
- `det(R) <= 0` (a reflection), or
- its bottom row is further than `max(tol, 1e-12)` (Euclidean) from
  `[0, 0, 0, 1]`.

The tolerance rejects only a target no configuration can reach. Every solver
accepts a configuration only when `||FK(q) - T_target||_F` is within `tol` or
tighter, and `FK(q)` is rigid, so that residual is at least the distance `d`
from `R` to the nearest rotation. For `d < 1`, `||R^T R - I||_F <= 3 d`: each
singular value `s` of `R` is within `d` of 1, and
`|s^2 - 1| = |s - 1| (s + 1) <= 3 |s - 1|`. A target past `3 * tol` therefore
has `d > tol` and could only ever have returned `[]`. The bottom row adds its
distance to the residual directly. The `1e-12` floor applies when a policy is
tighter than round-off: computing `R^T R` itself rounds by about `1e-15`, which
must not reject a rigid target.

A target inside the tolerance is solved as given, never projected onto SO(3),
and `fk_residual` is measured against it. Float64 round-off from upstream pose
arithmetic (around `1e-15`) and a `float32` pose (around `1e-7`) are well
inside. A target off SO(3) by more than one arm's own acceptance gate still
returns no solutions there, as before: the `three_parallel` artifacts accept at
`1e-7` and the exact `seven_r.spherical_shoulder` arms at `1e-10`, so a
`float32` pose can return `[]` on those.

**Options.** These are checked with the arrays:

| Option | Rule | Otherwise |
|---|---|---|
| `max_solutions` | `None` or an integer `>= 0` (Python or numpy) | `TypeError` for a non-integer, `ValueError` below 0 |
| `seed_tolerance` | `None`, or a real number that is not NaN, and only with `q_seed` | `TypeError` for a non-number, `ValueError` for NaN or without `q_seed` |
| `refinement_max_iters` | an integer `>= 0` | `TypeError` for a non-integer, `ValueError` below 0 |
| `seed_metric` | `"wrap_linf"` or `"wrap_l2"` when `q_seed` is given | `ValueError` |

`max_solutions` means "at most this many solutions". A cap of `0` returns `[]`
on every backend without solving, once every other input has passed its checks
(as `heapq.nsmallest(0, ...)` does), so a caller computing a remaining budget
can pass it unchanged. A negative cap used to mean different things on
different backends (all solutions, none, or an error) and is rejected. A cap
larger than any result is no cap on either backend. The
rules for the remaining options (`respect_limits`, `enumerate_windings`,
`allow_rescue`, `policy`, a `bool` given as an integer) are tracked in #575.

**The native extension.** `ssik._ssik_native` is internal and its functions
are not part of the semver contract, but none of them reads memory it was not
given. Each checks every array's shape and every index argument's range, and
the target's and seed's finiteness, before it reads them, and raises
`ValueError` otherwise. The C++ headers do not check their input: their
preconditions are in
[`cpp/README.md`](https://github.com/personalrobotics/ssik/blob/main/cpp/README.md#use-it-from-the-python-wheel).

## Per-call return: `Solution`

::: ssik.Solution
    options:
      show_root_heading: false

### Polished `general_6r` solutions

`general_6r` (Raghavan-Roth, including the jointlock sub-solves of Rizon 4 and
Rizon 10) finds its candidates with an ill-conditioned eigen-solve and accepts
one when its FK residual is within `TolerancePolicy.subproblem_numerical`
(`1e-5`). An accepted angle is then only as accurate as that residual over the
Jacobian's smallest singular value. So, on both backends and by default, every
accepted candidate gets a few Newton steps on the true FK before duplicates are
merged. The polished point replaces the candidate only if its residual drops to
`1e-12` and it stays within twice the first Newton step of the candidate, the
region where Newton's method provably converges to the candidate's own root,
and it keeps every value (or `2*pi` winding) that the candidate had within
round-off (`1e-9` rad, the floor of the [joint-limit band](#joint-limits)) of a
limit. Otherwise the candidate is returned exactly as the solver produced it,
which happens near a singularity or where polishing would cross a limit. Polish
never changes which candidates are accepted, never takes a value within
round-off of a limit out of it, and never moves one to another branch. A
candidate whose value lies further across a limit, but within its own
[error band](#error-band), can be polished to where its root really is, and the
limit pass then judges the polished point by its own, smaller band.
`refinement_used` stays `"none"`. The definition is in
`ssik.refinement.polish`.

This is not `allow_refinement`. That option (off by default) tries to rescue a
candidate that *failed* the acceptance gate, and tags a candidate it rescues
`"lm"`. It stops as soon as the residual is within the gate, so a rescued
solution is only as accurate as the gate.

### Singular continua

At a singular pose a 6R arm can have a one-parameter family of solutions, a
**continuum**, instead of isolated ones. The common case is a wrist whose
outer axes line up (`sin q5 = 0`): only `q4 + q6` (or `q4 - q6`) is fixed, and
every split of it reaches the target. A solver sees such a family only as
whatever samples its arithmetic lands on. On both backends, and for every 6R
family (`three_parallel`, `spherical_two_parallel`, `general_6r`), `solve()`
returns:

- **Seeded** (`q_seed` given): from each continuum, the point nearest the seed
  within the joint limits. "Nearest" is along the continuum: the point where
  the continuum's tangent is orthogonal to the seed offset (every joint
  compared on the circle), or, if that point is outside the limits, the
  in-limit point reached by the shortest walk along the continuum from it.
  The usual seed ranking then orders it among the other solutions, so a
  configuration on a continuum returns itself as the nearest solution.
- **Unseeded**: one representative per continuum, the point whose **free
  joint** is at `0` (modulo `2*pi`). The free joint is `q6` for a locked wrist
  of the closed-form families and, for `general_6r`, the highest-index joint
  the continuum moves (by at least a tenth of its unit tangent). If that point
  is outside the limits, the representative is the in-limit point nearest it
  along the continuum. If the continuum does not reach it (a UR-type arm whose
  elbow cannot follow), the representative is where the walk toward it stops,
  at the turn of the continuum nearest it.

With `respect_limits=False` (or `"wrap"`) the limits play no part: the point
is the seed's nearest, or the one with the free joint at `0`. A continuum with
no in-limit point is dropped by the limit pass, as any out-of-limit solution
is.

Only a solution the solver flags pays for this, and nothing measures the
Jacobian of every solution:

- The closed-form cores measure how close the wrist is to locked: the sine
  between the wrist-roll axis and the pitched outer wrist axis. Within `1e-4`
  they flag the candidate. That covers the precision to which each core
  resolves the wrist pitch at an exact lock, a double root (within `5.4e-8` of
  it through `spherical_two_parallel`'s closed form, `5.6e-6` through
  `three_parallel`'s numerical SP6, over 200 exactly locked poses per arm), so
  both backends flag the same continuum whichever side of the lock their
  arithmetic lands. Both keep their wrist angles (on a lock they are a point of
  the continuum) unless they divide zero by zero (sine within `1e-9`, the
  rule the spherical-shoulder 7R core already uses) or the angles are unusable:
  `spherical_two_parallel`'s miss the target's wrist rotation by more than
  `1e-4`; `three_parallel`'s cannot give a candidate within its `1e-7` FK gate
  (they miss the wrist rotation by more than that, or leave the elbow out of
  reach where another point of the lock reaches it). Then they split the lock.
  `three_parallel`, whose numerical wrist pitch is too coarse to read the
  angles there (near a lock each SP1 angle alone is accurate only to about
  `eps / sine`, though their sum is fixed), makes its shoulder and pitch exact for the lock and sets `q6` to the seed's value
  (`0` unseeded) and to that plus `pi`, each moved to the nearest of 64 values
  around the circle at which the elbow can reach, reading the other wrist
  angle off the target. `spherical_two_parallel`, on the lock, sets `q6` to
  the seed's value (`0` unseeded), and near it returns the two branches on
  either side.
- `general_6r` flags an accepted candidate whose Jacobian may be rank
  deficient: the Frobenius condition number of `J^T J + 1e-9 I`, the matrix
  its polish solves with anyway, is at least `5e7`. That bounds
  `(sigma_max / sigma_min)^2` from above, so it misses none, and the slide
  confirms with an SVD.

A flagged solution moves only along the directions where `J` is rank
deficient (`sigma <= 1e-4 sigma_max`), each step corrected back onto the
solution set by Gauss-Newton steps that never move along them, and the moved
point is kept only if it closes FK to `1e-10`; otherwise the solution stays as
the solver produced it. Every point of an exact continuum closes FK to
round-off, so there the rule's point is returned on both backends. At a pose
only near a continuum, where the family closes FK approximately, an isolated
solution stays where the solver put it, unless the rule's point is itself an
exact solution: a seed that is one is returned. Samples that reach the same
point merge. `fk_residual` is the moved point's. `refinement_used` keeps the
solver's tag. The definition, with every constant, is in `ssik.continuum`; the
native backend runs the same one (`cpp/include/ssik_cpp/continuum.hpp`).
Exactly singular poses get different values from earlier releases, which
returned whatever sample the arithmetic produced, or none.

## Diagnostic record: `Diagnostic`

Returned alongside the solution list when `solve(T, explain=True)`.

::: ssik.Diagnostic
    options:
      show_root_heading: false

## Tuning: `TolerancePolicy`

::: ssik.TolerancePolicy
    options:
      show_root_heading: false

## Partial targets: `Manipulator.free_tail`

A target that fixes less than a pose (a tool axis on a line, a tool plane on a plane, a tool point on a point) is `T0 @ exp(span(G))`, with `G` the body-frame twists that leave the primitive in place. `Manipulator.body_twists()` returns the joint twists `B` with `fk(q) = fk(0) @ se3_exp(B[0] q[0]) @ ... @ se3_exp(B[-1] q[-1])`, and `Manipulator.free_tail(G)` returns how many of the last joints have their twist in `span(G)`. Those joints drop out of the target: any values inside their limits keep it satisfied. A spin about the tool axis is the last joint turning, a plane normal to the last axis frees it wherever the plane sits, and a point at a spherical wrist's centre frees all three wrist joints. Twists are in `(v, w)` order throughout.

::: ssik.kinematics.twists
    options:
      show_root_heading: false

## Self-motion charts: `ssik.chart`

Charts live on `Manipulator`, so the canonical entry point for one of the 72 shipped arms is `Manipulator.from_prebuilt("panda")` — the artifact module exports `solve` and `fk` only. `solve()` on the result is the artifact's own solver, so this is not a slower stand-in for `import panda_ik`; it is that solver with the chart API attached.

`Manipulator.self_motion(T)` returns the self-motion manifold at `T`, a `SelfMotionManifold` of charts. For redundant 7R arms with a closed-form solver (`seven_r.spherical_shoulder`: Franka Panda, FR3; `seven_r.srs`: KUKA iiwa and other exactly-concurrent SRS arms) each chart is one continuous branch `q(t)` with a branch `label` (what it promises across poses, per family, is the label contract in the `ssik.chart` module docstring), its `domain` in the redundancy coordinate, its `in_limits()` arcs under joint limits, its `tangent(t)` as a unit direction and a rate, its `frame(t, metric)` (tangent plus a metric-orthogonal complement), and its `pullback_metric(t, metric)` (the metric pulled back to the chart coordinate, `g(t) = q'(t)^T G q'(t)`, exact from the tangent and divergent at a fold); the manifold object has the inverse map `locate(q)`, `sheets()` (charts glued where they meet at a fold: the connected components a controller can reach by self-motion), `gap(a, b)` and `escape(a, b)` (the task-space twist along which two sheets approach fastest, with `drift_to_merge()` to find where they actually merge), and `continue_from(...)` for continuation along a pose path; `track()` follows a branch through a list of poses by label, and `Manipulator.solve_path(poses)` (`track_all`) follows every branch at once, one manifold build per pose, reporting fold, collision and unreachable events and, on a closed path, the monodromy as `permutation`. `Chart.sample(n, metric)` places points uniformly in arc length rather than in `t` (which bunches up away from folds), and `Chart.length(metric)` is the branch's length; `metric` may be a constant matrix or a batched callable such as the mass matrix. For UR-class 6R arms (`ikgeo.three_parallel`) the charts are the isolated solutions with the geometric (shoulder, elbow, wrist) labels of `three_parallel_label`. `cuspidality_report()` classifies a spherical-shoulder arm's chart slicing (build-time diagnostic). `respect_limits="wrap"` on `Manipulator.solve` returns the full geometric set wrapped into joint ranges without dropping anything.

With the native extension (Linux and macOS wheels) a family builds in about 20 µs on a target move and `q(t)`, `locate(q)` and `tangent(t)` run in a few µs, so all of it fits a 1 kHz control loop; a chart's `domain` and `in_limits()` are computed on first access (well under a millisecond) and cached. `native=False` selects the pure-Python reference.

::: ssik.chart
    options:
      show_root_heading: false
      members:
        - charts
        - track
        - track_all
        - PathTrack
        - drift_to_merge
        - se3_exp
        - cuspidality_report
        - three_parallel_label
        - SelfMotionManifold
        - Chart

## Streaming IK: `Tracker`

`Manipulator.solve_path` tracks a pose list offline. A teleoperated arm sees
its poses one at a time, from a VR controller, a SpaceMouse, a mocap stream or
a transform gizmo. A `Tracker` holds the state that takes: the branch it is
on, the configuration it last commanded, and the time of the last pose.

```python
import ssik

arm = ssik.Manipulator.from_prebuilt("ur5e")       # or from_urdf(...)
tracker = arm.tracker(q_robot, max_joint_speed=2.0)
for T, t in source.poses():                        # any PoseSource
    step = tracker.update(T, t)                    # one pose in
    robot.command(step.q)                          # one configuration out
```

`Tracker` lives on `Manipulator`, so a shipped arm gets it through
`Manipulator.from_prebuilt`, whose `solve` is the artifact's own. It adds no
cost to `solve()`.

**Continuation.** Every update continues the branch the tracker is following,
never a fresh solve that could land on another one:

- On a redundant 7R arm with a closed-form chart (`seven_r.spherical_shoulder`:
  Franka Panda, FR3; `seven_r.srs`: KUKA iiwa and the other exactly concurrent
  SRS arms) the redundancy coordinate is held fixed, as `solve_path` holds it,
  and the candidate is the chart point there nearest the followed one (every
  joint compared on the circle). Where the label contract holds, that is the
  chart `solve_path` continues to by label, so a tracked stretch returns
  `solve_path`'s configurations up to a `2*pi` representative. Near a fold of
  the spherical-shoulder charts the old label can still name a point within
  `max_step` that is not the continuation; the tracker takes the nearer point.
- On every other arm, and on a 7R chart arm when the held point is further
  than `jump_threshold` or outside the joint limits, the candidate is
  `solve(T, q_seed=..., max_solutions=1, allow_rescue=False)` under the
  tracker's `respect_limits`: the nearest configuration in ssik's seed metric.
  At a singular pose that is the point of the continuum nearest the seed
  ([Singular continua](#singular-continua)), so the continuation is defined
  through a singularity. The T-perturbation rescue is off: it is for a
  measure-zero ridge where the analytical path finds nothing, and it costs
  milliseconds native and seconds in Python on every update it runs, which
  would be every update the target is out of reach. At such a ridge the
  tracker holds for one update instead.

The candidate is reported in the representative `solve()` reports, the one
nearest the followed configuration, inside the joint limits when they are
respected.

**Distance.** Every distance is ssik's seed metric (`wrap_linf`): the largest
single-joint move, with continuous joints (no limits) compared on the circle
and every other joint by its coordinate, since a finite joint cannot turn
through its stop.

**Statuses.** With `branch_distance` the distance from the configuration being
followed to the candidate:

| Status | When | `q` |
|---|---|---|
| `OK` | a candidate within `jump_threshold`, reachable this update | the candidate; FK closes to the target within the solver's tolerance |
| `LIMITED` | as `OK`, but some joint would move more than `max_joint_speed * dt` | moved toward the candidate, the step scaled down (direction kept) until no joint exceeds its limit |
| `HELD` | no candidate, or one further than `jump_threshold` with `allow_jump=False` | the previous `q`, unchanged |
| `JUMPED` | a candidate further than `jump_threshold` with `allow_jump=True`, or `next_branch()` | the candidate, rate limited like any move |

`HELD` carries a `reason`:

- `"unreachable"`: no configuration reaches the target, within the limits or
  not (`step.reachable` is `False`);
- `"limits"`: the followed branch continues only outside the joint limits, or
  no in-limit configuration reaches the target;
- `"jump"`: the nearest configuration is further than `jump_threshold`, a
  branch switch.

While held the tracker keeps following the branch it was on, so it resumes
`OK` when the target comes back within reach of that branch.

**Thresholds.** The two limits answer different questions.

- `jump_threshold` (default `0.5` rad, `solve_path`'s `max_step`) decides what
  is the same branch. It is a property of the branch geometry, independent of
  time: a fast hand never triggers it and a slow flip always does.
- `max_joint_speed` (rad/s, or m/s for a prismatic joint; one number or one
  per joint; default none) bounds how fast the arm follows. It applies to an
  update with a timestamp `t` after an earlier timestamp (the constructor's
  `t0`, `reset`'s `t`, or a previous update), with `dt` their difference.
  Timestamps must not decrease. While `LIMITED`, `q` does not close FK to the
  target: `fk_residual` is the honest `||FK(q) - T||_F` and `lag` the
  distance still to go. Later updates keep closing the gap, and the update
  that arrives is `OK`.

**Step fields.** `TrackerStep` is a frozen record: `q` (read-only), `status`,
`reason`, `fk_residual` (against this update's target, whatever the status),
`moved` (from the previous step's `q`), `branch_distance` (`nan` when there
was no candidate), `lag`, and `t`.

**Branches.** `solutions(max_solutions=None)` returns every configuration at
the target the tracker last accepted (`tracker.target`), nearest the followed
branch first, one per geometric branch (`enumerate_windings=False`); on a
redundant 7R arm these are `solve()`'s samples of the self-motion. They are
for rendering the other branches. `next_branch()` switches to the next branch
at that target, cycling in a fixed order (on a 7R chart arm by chart label at
the held coordinate, otherwise by configuration), and returns a `JUMPED`
step, or `None` when there is no other branch. Without a rate limit its `q` is
on the new branch; with one the arm does not move there, the new branch
becomes the one followed, and later timestamped updates carry the arm there
(`LIMITED` until it arrives). `reset(q, t=None)` restarts from a
configuration the arm is at.

::: ssik.Tracker
    options:
      show_root_heading: false
      members:
        - update
        - solutions
        - next_branch
        - reset
        - q
        - target

::: ssik.TrackStatus
    options:
      show_root_heading: false

::: ssik.TrackerStep
    options:
      show_root_heading: false

## Teleoperation frames: `ssik.teleop`

The steps between a device and `Tracker.update` are rigid-transform
compositions, collected in `ssik.teleop`. ssik ships no device code: a device
is anything with a `poses()` method yielding `(T, t)`, the `PoseSource`
protocol. `examples/06_teleop.py` wires a scripted source through every step.

**Conventions.** A pose is a 4x4 homogeneous rigid transform `a_T_b`, the pose
of frame `b` in frame `a`; it maps `b`-coordinates to `a`-coordinates and
composes as `a_T_c = a_T_b @ b_T_c`. Every pose argument is checked as
`solve()` checks `T_target` ([Input validation](#input-validation)), with the
argument's name in the message.

| Helper | Returns |
|---|---|
| `invert(T)` | `[R^T, -R^T p]` |
| `calibration_from(T_device, T_robot)` | `base_T_world = T_robot @ T_device^-1`, the calibration that maps this device reading onto this robot pose |
| `apply_calibration(calibration, T_device)` | `calibration @ T_device`: a reading `world_T_device` in the arm's base frame |
| `tcp_to_flange(T_tcp, flange_T_tcp)` | `T_tcp @ flange_T_tcp^-1`: the flange pose that puts the tool centre point at `T_tcp` (what IK solves for) |
| `flange_to_tcp(T_flange, flange_T_tcp)` | `T_flange @ flange_T_tcp`: the TCP pose at a flange pose |
| `scale_about(T, scale, anchor)` | position `a + scale * (p - a)` about the anchor point `a` (a `(3,)` point, or a pose's position, in `T`'s frame), rotation unchanged |

**Clutch.** Relative teleoperation: the arm follows the device's motion since
the grip was pressed, not its absolute pose. `engage(T_device, T_robot)`
stores the device anchor `A_d = [R_ad, p_ad]` and the arm anchor
`A_r = [R_ar, p_ar]`, both in one frame: the arm's base frame, after
`apply_calibration`. While engaged, a device pose `D = [R_d, p_d]` gives a
target by one of two conventions, chosen by `Clutch(scale=1.0, frame=...)`.

`frame="world"` (default): the device's motion, measured in the base frame,
is applied in the base frame.

```
p = p_ar + scale * (p_d - p_ad)
R = (R_d @ R_ad^T) @ R_ar
```

The rotation since engaging, `R_d @ R_ad^T`, turns the tool about its own
point, with the axis taken in the base frame. A pure device translation `Δ`
moves the tool by `scale * Δ` in the base frame, whatever the device or the
tool is pointing at. Use it when the operator watches the arm and moves a
hand-held device (a VR controller, a mocap marker, a gizmo): moving the hand
toward the arm's +x moves the tool toward +x.

`frame="tool"`: the device's motion, measured in its own frame at engagement,
is applied in the tool's frame at engagement.

```
target(D) = A_r @ S(A_d^-1 @ D)
```

`A_d^-1 @ D` is the device's displacement in the device's frame at engagement;
`S` multiplies its translation by `scale`. With `scale = 1` this is
`(A_r @ A_d^-1) @ D`, a fixed rigid transform of the device's pose, so relative
motion is reproduced exactly and the device's axes act as the tool's axes:
pushing a SpaceMouse cap forward moves the tool along its own approach axis.
Use it for jogging along the tool's axes.

In both conventions:

- `target(A_d) == A_r`: engaging never moves the target.
- Rotation is never scaled, and scaling equals `scale_about(D, scale, A_d)`
  followed by the unscaled clutch.
- `release()` makes `target` return `None` (the caller holds). The next
  `engage` re-anchors both frames, so the operator can reposition the device
  without moving the arm. Engaging again at the current device pose and
  target continues exactly where the clutch was.
- The two agree when the device and arm anchors have the same orientation.

Engage with the arm's actual pose, `flange_to_tcp(arm.fk(tracker.q), tool)`,
so a rate-limited arm that is still catching up is anchored where it is.

::: ssik.teleop
    options:
      show_root_heading: false
      members:
        - PoseSource
        - Clutch
        - calibration_from
        - apply_calibration
        - tcp_to_flange
        - flange_to_tcp
        - scale_about
        - invert

## Postprocess helpers

The `solve()` pipeline already applies these by default (when `respect_limits=True`); they're exposed for callers who want a different order, an extra filter step, or to compose with collision/dexterity scoring.

::: ssik.postprocess.respect_limits
    options:
      show_root_heading: false
      show_root_full_path: false

::: ssik.postprocess.wrap_to_limits
    options:
      show_root_heading: false
      show_root_full_path: false

::: ssik.postprocess.nearest_to_seed
    options:
      show_root_heading: false
      show_root_full_path: false

::: ssik.postprocess.within_seed_tolerance
    options:
      show_root_heading: false
      show_root_full_path: false

::: ssik.postprocess.take_first
    options:
      show_root_heading: false
      show_root_full_path: false

::: ssik.postprocess.rewrap_to_seed
    options:
      show_root_heading: false
      show_root_full_path: false

### Angle representatives

Every `solve()` result, on either backend and under every `respect_limits`
mode, reports each angle in one canonical coordinate. It is chosen before the
limit pass, seed ranking and winding enumeration, so all of them see the same
values:

- A **continuous** revolute joint (`limits=None`) is reported in `(-pi, pi]`.
  An angle within round-off (`1e-9` rad) of the cut is reported as exactly
  `+pi`. Either side of the cut is the same configuration of a continuous
  joint, so compare such joints on the circle.
- A **finite** revolute joint whose limits admit more than one representative
  of `pi` (`[-pi, pi]` to within the `1e-9` [limit band](#joint-limits), or
  the UR family's `[-2*pi, 2*pi]`) reports an angle **at the cut** on the
  `+pi` side: the `2*pi` shift nearest the in-limit representative of `pi`
  closest to `+pi`. If that shift lies beyond a limit that sits at `+pi`, as
  on a `[-pi, pi]` joint, the angle is reported as exactly the limit. An angle
  is at the cut when it lies within its solution's
  [error band](#error-band) of `pi` (modulo `2*pi`), or within `1e-14` rad of
  it.
- Every other angle keeps the solver's value, then `wrap_to_limits`, winding
  enumeration and `rewrap_to_seed` apply as documented below.

The coordinate moves by a multiple of `2*pi`, which is the same configuration.
The only other movements are the `1e-9` snap of a continuous joint and the
clamp of a shifted angle onto a limit at `+pi`, which moves it by at most its
error band, and never more than `1e-6` rad. For an accurate solution that band
is far below round-off, so an accurate angle near `pi` stays where the solver
put it. The band is wider where the solver cannot place the angle better, such
as a folded elbow at the cut, a double root known only to a few `1e-7` rad.
There both backends report the angle on the same side of the cut. The
derivation is at `ssik.postprocess._CUT_SNAP` and `ssik.postprocess._BAND_GAIN`.
A clamp moves the configuration, so the returned `fk_residual` is measured
again at the clamped value (see [Joint limits](#joint-limits)).

### Joint limits

A limit is inclusive up to the solution's own error. With
`respect_limits=True` or `"wrap"`, on either backend:

- an angle outside a joint limit by at most the solution's **limit band** is
  at the limit: the solution is kept and that joint is reported as exactly the
  limit. The limit band is the larger of round-off (`1e-9` rad) and the
  solution's [error band](#error-band);
- an angle inside a limit by at most round-off (`1e-9` rad) is also reported as
  exactly the limit;
- any other angle inside its limits keeps its value, and any angle further
  outside drops the solution.

This applies to the limit pass, the redundant-7R in-limits resolvers, and every
winding representative, so a branch at a limit has the same lifts on both
backends. A configuration exactly at a hard stop is therefore returned on both
backends, and a seeded solve from it returns its own branch, although the
solvers compute its angle on either side of the limit. Every returned angle
lies within its limits exactly. `respect_limits=False` leaves the solver's
values as they are.

The `1e-9` floor covers the closed-form families at a regular or moderately
conditioned pose, which land within `1.2e-10` rad of an exact limit, and
`general_6r`, whose accepted solutions are polished
([Polished `general_6r` solutions](#polished-general_6r-solutions)). Near a
singularity or a fold an angle is a multiple root, known only to about the
square root of round-off, and an angle at an exact limit can land up to a few
`1e-7` rad past it. The error band covers that. The band is capped at `1e-6`
rad, so a clamp never moves a configuration further. A configuration further
than that across a limit is dropped, even when its error estimate is larger.
Such a configuration is typically a different point of a singular family,
which moving one joint cannot bring back without breaking FK. The derivation
is at `ssik.postprocess._LIMIT_BAND` and `ssik.postprocess._BAND_GAIN`.

`Solution.fk_residual` always describes the returned `q`. The solver measures
it, and when this clamp or an [angle-representative](#angle-representatives)
snap or clamp moves a value, it is measured again at the moved value, on both
backends: `||FK(q) - T_target||_F` on the arm's chain, the solvers' own metric.
A `2*pi` shift (a winding representative, a seed rewrap) is the same
configuration and keeps the solver's value. The re-measurement costs one FK per
moved solution, and solutions nothing moved pay nothing.

On a redundant 7R arm the in-limit part of the self-motion can be a single
point or a sliver, typically where two or more joints are at their limits at
once. There every joint's margin touches zero without changing sign, so
bracketing the sign changes finds no in-limits arc. When the in-limits
resolver finds no arc on any branch, it looks for the point where the
worst-case limit violation along each branch is smallest. On a closed-form
chart (`seven_r.srs`, `seven_r.spherical_shoulder`) that point is the
solution. On the polished families it is a starting point, from which the
resolver walks along the arm's true self-motion curve to the minimum. The
point counts as in limits when its worst violation is within its limit band.
Such a point is clamped onto the limits and its `fk_residual` measured again,
as above. Only a contact whose deepest margin is below `1e-4` rad is accepted
this way. A deeper in-limits stretch with no arc around it is left to the
rescue, as before. `Chart.in_limits()` returns such a point as a zero-width
arc `(t, t)`. None of this runs unless the resolver's own search comes back
empty. The `jointlock.seven_r` arms (Rizon 4 and 10, Kassow KR810) have no
in-limits resolver, so this does not apply to them.

### Error band

Whether a value is at a joint limit or at the `+-pi` cut is decided against
that solution's own angular error. For a solution `q` of the target `T`, both
backends compute

```
error band = min(1e-6, 10 * ||FK(q) - T||_F / sigma_min(J(q)))
```

where `J` is the spatial Jacobian of the arm's chain at `q` and `sigma_min`
its smallest singular value. To first order, an error `dq` leaves a residual
of at least `sigma_min * ||dq||`, so the ratio bounds every joint's error.
Across the release pose set (uniform, near-limit and near-singular poses of
every 6R arm, both backends), every actual error up to the `1e-6` cap was at
most 5.9 times that ratio, and at most 2.1 times it for errors below `1e-9`.
The factor 10 covers both. An accurate solution at a regular pose has a
residual near `1e-15` and a band far below round-off. A solution near a
singularity has a small `sigma_min` and a wide band. The `1e-6` cap is a tenth
of the default FK acceptance gate (`TolerancePolicy.subproblem_numerical`), so
a moved solution stays inside that gate. It is fixed: a tighter
`subproblem_numerical` does not narrow it.

The band is computed only for a solution with a value outside a limit, or near
the cut, by between the floor and `1e-6` rad, so an ordinary solve does not pay
for it. The standalone filters (`respect_limits`, `wrap_to_limits`,
`expand_windings`, `rewrap_to_seed`, `count_windings`) use it when given
`T_target`, and otherwise use only the round-off floors.

### Winding representatives

A revolute joint whose limits span more than one turn (the UR family's
`[-2*pi, 2*pi]`) has several in-limit
`q + 2*pi*k` representatives of the *same* geometric branch. They reach the
same pose but are different admissible configurations, at different distances
and with different motions available from where the robot is now. `solve()`
returns all of them by default; pass `enumerate_windings=False` for one
representative per geometric branch.

These are finite-limit **lifts**, not additional geometric branches, and
`Diagnostic.geometric_branches` / `Diagnostic.winding_representatives` report
the two counts separately.

::: ssik.postprocess.winding_joints
    options:
      show_root_heading: false
      show_root_full_path: false

::: ssik.postprocess.expand_windings
    options:
      show_root_heading: false
      show_root_full_path: false

::: ssik.postprocess.count_windings
    options:
      show_root_heading: false
      show_root_full_path: false

## C++ headers: `ssik.cpp`

Every wheel ships the header-only C++ solvers (`ssik_cpp/`) and a CMake package
for them. `ssik.get_include()` and `ssik.get_cmake_dir()` locate them, and
`ssik.cpp.joint_data(arm)` returns the arrays a C++ caller builds
`ssik::JointConsts<N>` and `ssik::JointLimits<N>` from. The consumer guide is
[`cpp/README.md`](https://github.com/personalrobotics/ssik/blob/main/cpp/README.md#use-it-from-the-python-wheel);
the C++ names covered by semver are listed in the [semver policy](semver_policy.md).

::: ssik.cpp
    options:
      show_root_heading: false
      members:
        - get_include
        - get_cmake_dir
        - joint_data
        - JointData

## CLI: `ssik build`

```bash
ssik build <urdf> --base <link> --ee <link> [--out <path>]
ssik classify <urdf> --base <link> --ee <link>
ssik add-arm <urdf> --base <link> --ee <link> --name <arm>
```

Full help: `ssik <command> --help`. See [Setting up your robot](setting_up_your_robot.md) for the full URDF-to-artifact workflow.
