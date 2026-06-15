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

## CLI: `ssik build`

```bash
ssik build <urdf> --base <link> --ee <link> [--out <path>]
ssik classify <urdf> --base <link> --ee <link>
ssik add-arm <urdf> --base <link> --ee <link> --name <arm>
```

Full help: `ssik <command> --help`. See [Setting up your robot](setting_up_your_robot.md) for the full URDF-to-artifact workflow.
