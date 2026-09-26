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

## CLI: `ssik build`

```bash
ssik build <urdf> --base <link> --ee <link> [--out <path>]
ssik classify <urdf> --base <link> --ee <link>
ssik add-arm <urdf> --base <link> --ee <link> --name <arm>
```

Full help: `ssik <command> --help`. See [Setting up your robot](setting_up_your_robot.md) for the full URDF-to-artifact workflow.
