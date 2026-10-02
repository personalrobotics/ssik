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
        - dof
        - solver_name
        - kinbody

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
  rule the spherical-shoulder 7R core already uses) or miss the target's wrist
  rotation by more than `1e-4`. Then they split the lock. `three_parallel`,
  whose numerical wrist pitch is too coarse to read the angles there, makes
  its shoulder and pitch exact for the lock and sets `q6` to the seed's value
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
