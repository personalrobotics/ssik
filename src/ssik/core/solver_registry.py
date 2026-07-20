"""Single source of truth for every analytical solver ssik can dispatch.

Historically the per-solver metadata lived in ~six parallel ``name -> X`` tables
that had to be edited in lockstep whenever a solver was added or renamed
(dispatcher estimates, codegen import paths, manipulator module paths, cli
scaffolds, codegen composers, and the fk-atol / force-refine overrides). Two of
them were byte-identical and fully derivable (``f"ssik.solvers.{name}"``), and
nothing guarded keyset consistency, so drift shipped green and surfaced as a
KeyError in the field.

This module collapses all of that into one :class:`SolverSpec` per solver. Every
consumer reads :data:`SOLVERS`; nothing else holds per-solver metadata.
``tests/test_solver_registry.py`` asserts the table is internally consistent and
that every module/composer path imports with the expected entry point, so a new
solver is wired everywhere from one row or CI goes red.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SOLVERS", "SolverSpec"]


@dataclass(frozen=True)
class SolverSpec:
    """Everything ssik needs to know about one solver, in one place.

    :param name: the dispatch key, e.g. ``"ikgeo.three_parallel"``. It is also
        the module suffix -- :attr:`module_path` is ``"ssik.solvers." + name`` --
        so the (formerly duplicated) import-path and module-path tables are just
        this property.
    :param tier: 0 (closed-form) / 1 (univariate search) / 2 (numeric RR/HP).
    :param expected_ms: rough median solve time, for the dispatch plan + CLI ETA.
    :param flop_budget: rough per-solve FLOP estimate, for the dispatch plan.
    :param needs_symbolic_precompute: True iff ``ssik build`` runs a sympy
        derivation for this solver (tier-2 Raghavan-Roth only today).
    :param composer: dotted module path of the specialised codegen composer
        (its entry point is ``compose``), or ``None`` for solvers emitted via the
        thin-wrapper template.
    :param fk_atol_expr: the FK-closure gate the emitted specialised orchestrator
        verifies candidates at (a Python expression string). Defaults to the
        generic subproblem tolerance; tightened per solver where a boundary
        near-miss would otherwise pass (three_parallel, #362).
    :param force_refine: whether the emitted artifact always runs the LM polish
        (independent of the caller's ``allow_refinement``); artifact-only, the
        live solver still honours the caller.
    :param auto_dispatch: whether :func:`ssik.core.dispatcher.dispatch` can route
        an arm here automatically. False for the tier-1 univariate-search solvers,
        which are documented as explicit-use-only (RR is ~50-200x faster).
    """

    name: str
    tier: int
    expected_ms: float
    flop_budget: int
    needs_symbolic_precompute: bool = False
    composer: str | None = None
    fk_atol_expr: str = "policy.subproblem_numerical"
    force_refine: bool = False
    auto_dispatch: bool = True

    @property
    def module_path(self) -> str:
        """Import path of the live solver module (defines ``solve``)."""
        return f"ssik.solvers.{self.name}"


def _spec(name: str, tier: int, ms: float, flops: int, **kw: object) -> SolverSpec:
    return SolverSpec(name=name, tier=tier, expected_ms=ms, flop_budget=flops, **kw)  # type: ignore[arg-type]


_COMPOSE = "ssik.codegen._compose"

SOLVERS: dict[str, SolverSpec] = {
    s.name: s
    for s in (
        # fk_atol_expr / force_refine: this solver can emit a near-singular SP-clip
        # near-miss (~1e-6 FK, no real IK nearby); the tightened 1e-7 gate drops it
        # and always-polish separates genuine near-singular solutions (converge ->
        # kept, #288) from spurious boundary near-misses (stall -> dropped, #362).
        # 1e-7 == ssik.solvers.ikgeo.three_parallel._FK_VERIFY_ATOL (guard-tested).
        _spec(
            "ikgeo.three_parallel",
            0,
            1.6,
            2_519,
            composer=f"{_COMPOSE}.three_parallel",
            fk_atol_expr="1e-7",
            force_refine=True,
        ),
        _spec(
            "ikgeo.spherical_two_parallel",
            0,
            1.2,
            1_316,
            composer=f"{_COMPOSE}.spherical_two_parallel",
        ),
        _spec(
            "ikgeo.spherical_two_intersecting",
            0,
            1.3,
            1_476,
            composer=f"{_COMPOSE}.spherical_two_intersecting",
        ),
        _spec("ikgeo.spherical", 0, 7.5, 10_312, composer=f"{_COMPOSE}.spherical"),
        _spec("ikgeo.two_parallel", 1, 261.0, 141_569, auto_dispatch=False),
        _spec("ikgeo.two_intersecting", 1, 1184.0, 2_650_681, auto_dispatch=False),
        _spec(
            "ikgeo.general_6r",
            2,
            5.0,
            30_000_000,
            needs_symbolic_precompute=True,
            composer=f"{_COMPOSE}.general_6r",
        ),
        _spec("husty_pfurner.general_6r", 2, 120.0, 50_000_000),
        _spec("seven_r.srs", 0, 8.5, 1_900),
        _spec("seven_r.srs_polished", 0, 56.0, 80_000),
        _spec("seven_r.spherical_shoulder", 0, 17.0, 6_000),
        _spec("seven_r.spherical_shoulder_polished", 0, 8.0, 40_000),
        _spec("jointlock.seven_r", 1, 50.0, 30_274, composer=f"{_COMPOSE}.seven_r"),
    )
}
