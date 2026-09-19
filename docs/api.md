# API reference

Auto-generated from docstrings. The public surface is small by design — most users only touch `Manipulator`, `Solution`, and (when relevant) `TolerancePolicy` / `Diagnostic`.

## Entry point: `Manipulator`

::: ssik.Manipulator
    options:
      show_root_heading: false
      members:
        - from_urdf
        - solve
        - fk
        - charts
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

`Manipulator.charts(T)` returns the manifold of solutions at `T` as charts. For redundant 7R arms with a closed-form solver (`seven_r.spherical_shoulder`: Franka Panda, FR3; `seven_r.srs`: KUKA iiwa and other exactly-concurrent SRS arms) each chart is one continuous branch `q(t)` with a stable branch `label`, its `domain` in the redundancy coordinate, its `in_limits()` arcs under joint limits, its `tangent(t)` as a unit direction and a rate, and its `frame(t, metric)` (tangent plus a metric-orthogonal complement); the family has the inverse map `locate(q)` and `continue_from(...)` for continuation along a pose path, and `track()` follows a branch through a list of poses by label. For UR-class 6R arms (`ikgeo.three_parallel`) the charts are the isolated solutions with the geometric (shoulder, elbow, wrist) labels of `three_parallel_label`. `cuspidality_report()` classifies a spherical-shoulder arm's chart slicing (build-time diagnostic). `respect_limits="wrap"` on `Manipulator.solve` returns the full geometric set wrapped into joint ranges without dropping anything.

With the native extension (Linux and macOS wheels) a family builds in about 20 µs on a target move and `q(t)`, `locate(q)` and `tangent(t)` run in a few µs, so all of it fits a 1 kHz control loop; a chart's `domain` and `in_limits()` are computed on first access (well under a millisecond) and cached. `native=False` selects the pure-Python reference.

::: ssik.chart
    options:
      show_root_heading: false
      members:
        - charts
        - track
        - cuspidality_report
        - three_parallel_label
        - ChartFamily
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
