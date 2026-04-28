# ssik

Analytical inverse kinematics for non-Pieper 6R arms — the **EAIK gap**. The arms whose IK isn't a one-line call against EAIK or IK-Geo (Kinova JACO 2, Agilex Piper, Flexiv Rizon, custom geometries with no parallel or intersecting axis triples).

> **Private repository.** This codebase is proprietary. Source distribution is not authorised; see [`LICENSE`](LICENSE). The user-facing artifact is a per-arm compiled wheel built from this codebase; users do not run this source directly. See [#95](https://github.com/siddhss5/ikfastpy/issues/95) for the distribution model.

> **History.** Originally a port of OpenRAVE's [IKFast](https://www.openrave.org/docs/0.8.2/openravepy/ikfast/) named `ikfastpy`. The IKFast general-solver path turned out unfixable on modern sympy for non-Pieper 6R arms, so the project was rebuilt around (1) tier-0 subproblem-composition solvers ported from BSD-3 [IK-Geo](https://github.com/rpiRobotics/ik-geo) (Elias–Wen 2022/2025) for Pieper-class arms, and (2) a tier-2 numeric Raghavan–Roth + Manocha–Canny pipeline for non-Pieper arms. The vendored LGPL IKFast tree was removed in [#84](https://github.com/siddhss5/ikfastpy/issues/84). The repo was renamed to `ssik` and made private at the same time the proprietary licensing landed.

## What ssik does that the alternatives don't

| arm class | EAIK / IK-Geo | mink / KDL (numeric) | ssik |
|---|---|---|---|
| Pieper-class (UR5, Puma 560, Fanuc, KUKA KR) | ~0.2 ms, all branches | ~20 ms, one solution | ~50-200 µs, all branches |
| Non-Pieper 6R (JACO 2, Piper) | not supported | ~20 ms, one solution | **~2.25 ms median, all branches** |
| Non-SRS 7R (Flexiv Rizon) | not supported | ~30 ms, one solution | ~ms range via joint-locking |

The differentiator is the **non-Pieper 6R analytical solver**. No other library in the ecosystem ships analytical IK for arms whose geometry deliberately violates Pieper's condition for mechanical-design reasons. ssik does, with all branches recovered at machine precision in single-digit milliseconds. See `docs/tutorial/04_raghavan_roth.md` for the math and `docs/tutorial/05_conditioning.md` for the four robustness fixes (AE-1, AE-3, AE-4, Möbius reparameterisation) that make the textbook Raghavan–Roth pipeline survive on real ill-conditioned arms.

## Repository layout

This is the **internal development codebase**. Public users never see this; they receive a per-arm wheel built from this source. Layout:

- `src/ssik/` — solver implementations.
  - `core/` — `Solution` dataclass, tolerance policies.
  - `kinematics/` — POE → DH bridge, predicates.
  - `subproblems/` — SP1–SP6 closed-form primitives.
  - `solvers/ikgeo/` — tier-0/1/2 solver modules.
  - `solvers/jointlock/` — 7R wrapper.
  - `refinement/` — universal opt-in Newton polish layer.
- `tests/` — unit + hypothesis fuzz + cross-solver agreement + slow round-trips.
  - `tests/fixtures/` — UR5, Puma 560, JACO 2 (real MJCF), synthetic arms.
- `scripts/` — bench, profile, diagnostic harnesses.
- `docs/` — internal documentation.
  - `docs/tutorial/` — ten chapters covering the IK problem, the Pieper class, the EAIK gap, the Raghavan–Roth pipeline, conditioning fixes, refinement architecture, KinBody bridge, bulletproof validation, practical guide, roadmap. **Internal only**; the public marketing site (per [#95](https://github.com/siddhss5/ikfastpy/issues/95)) is a stripped subset hosted in a separate public repo.

## Development quick-start

```bash
# Install dev dependencies
uv sync

# Fast tests (excludes slow symbolic-preprocessing tests)
uv run pytest

# Slow tests (sympy preprocessing for tier-2 RR; ~5 min)
uv run pytest -m slow

# Lint, format, typecheck
uv run ruff check
uv run ruff format --check
uv run mypy src tests

# Build internal docs
uv run mkdocs serve
```

## Solver coverage

| solver module | algorithm | fixtures |
|---|---|---|
| `ikgeo.three_parallel` | tier-0 SP6 + SP1 + SP3 | UR5, UR10 (URDF), synthetic |
| `ikgeo.spherical_two_parallel` | tier-0 SP4 + SP3 + SP1 | Puma 560 (URDF), synthetic |
| `ikgeo.spherical_two_intersecting` | tier-0 SP3 + SP2 + SP4 + SP1 | Puma 560 (URDF), synthetic |
| `ikgeo.spherical` | tier-0 SP5 + SP4 + SP1 | synthetic |
| `ikgeo.two_parallel` | tier-1 univariate-search SP6 | synthetic |
| `ikgeo.two_intersecting` | tier-1 univariate-search SP5 | synthetic |
| `ikgeo.gen_six_dof` | tier-2 100×100 grid + Nelder–Mead | synthetic (legacy oracle) |
| `ikgeo.general_6r` | tier-2 numeric Raghavan–Roth + AE-3 | **JACO 2 j2n6s200 (real MJCF)**, UR5 |
| `jointlock.seven_r` | 7R via joint-locking + 6R inner solve | synthetic SRS arm |

## Distribution model

Customers submit a URDF / MJCF via [#95](https://github.com/siddhss5/ikfastpy/issues/95)'s intake mechanism. We run the cold-cache build pipeline (`poe_to_dh` → `pick_best_leftvar` AE-3 selection → symbolic preprocessing → compile to `.so`), produce a per-arm wheel, and deliver. The wheel exposes `solve(T_target) -> tuple[list[Solution], bool]` with the same `Solution` shape used internally. Customer source-code never includes ssik internals — they call into the compiled binary.

## License

Proprietary. See [`LICENSE`](LICENSE) for full terms; in summary: all rights reserved, no public reproduction or distribution without prior written permission. The library incorporates clean-room reimplementations of algorithms from BSD-3-licensed [IK-Geo](https://github.com/rpiRobotics/ik-geo) (Elias–Wen 2022/2025) and from the academic publications of Raghavan–Roth (1990) and Manocha–Canny (1994); the BSD-3 attribution is preserved in `LICENSE` for the algorithmic lineage.

## Tracking

- Strategic distribution model: [#95](https://github.com/siddhss5/ikfastpy/issues/95).
- Speed work across all solver pathways: [#93](https://github.com/siddhss5/ikfastpy/issues/93).
- Tier-2 RR speed (already 2× since baseline): [#86](https://github.com/siddhss5/ikfastpy/issues/86).
- Tutorial / internal docs: [#87](https://github.com/siddhss5/ikfastpy/issues/87).
- Known coverage gap on synthetic MC Table I: [#82](https://github.com/siddhss5/ikfastpy/issues/82).
