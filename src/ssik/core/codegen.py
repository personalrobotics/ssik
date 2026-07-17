"""Per-arm artifact emission. Renders a self-contained Python module that
wraps the dispatched solver around baked KinBody constants.

The emitted artifact is a single ``.py`` file with a stable public API:

    >>> import ur5_ik
    >>> solutions = ur5_ik.solve(T_target, max_solutions=1, q_seed=q_current)
    >>> T = ur5_ik.fk(q)

Internals: the emitted module imports the chosen ssik solver, reconstructs
the POE-normalised :class:`KinBody` from baked numpy literals at import
time, and exports ``solve(T) -> list[Solution]``, ``fk(q) -> NDArray``, plus
``SOLVER_NAME`` and ``DISPATCH_REASON`` constants for diagnostic visibility.

This iteration emits source that has ``ssik`` as a runtime dependency. The
forthcoming Cython port emits the equivalent compiled artifact under the
same import-and-call API; the build CLI gains a flag to switch targets.

For tier-2 ``general_6r`` arms today, the artifact's ``solve()`` triggers
the lazy sympy preprocessing on first call (the existing behaviour). Phase
2 of #110 bakes that preprocessing output into the artifact at build time
so first-call latency is gone.
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from ssik._kinbody import KinBody
    from ssik.core.dispatcher import DispatchPlan

__all__ = ["EmissionResult", "emit_artifact"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class EmissionResult:
    """What :func:`emit_artifact` produced.

    The CLI prints ``output_path`` and uses ``module_name`` for the "to
    use it: ``import <module_name>``" line. Tests assert against
    ``source`` directly (the rendered file content) so they don't have
    to write to disk.
    """

    module_name: str
    """Python module name, e.g. ``ur5_ik`` (no ``.py`` suffix)."""

    output_path: str | None
    """Filesystem path the artifact was written to, or ``None`` if the
    caller asked for source-only emission."""

    source: str
    """Full rendered source of the artifact (also written to disk if
    ``output_path`` is set)."""


def emit_artifact(
    *,
    kb: KinBody,
    plan: DispatchPlan,
    module_name: str,
    output_path: str | None = None,
    arm_label: str | None = None,
) -> EmissionResult:
    """Render a per-arm IK module that wraps the dispatched solver.

    :param kb: a POE-normalised :class:`KinBody` (the one passed to
        :func:`ssik.dispatch`).
    :param plan: dispatch result describing which solver to import and the
        explanatory diagnostic to bake into the artifact's docstring.
    :param module_name: stem for the emitted module (``ur5_ik``,
        ``jaco2_ik``, ...). Used in the artifact's header comment.
    :param output_path: where to write the rendered source. ``None`` means
        return source only without touching disk -- useful in tests and
        when the caller wants to post-process before writing.
    :param arm_label: optional human-readable arm name to embed in the
        artifact header (``"UR5"``, ``"Kinova JACO 2"``). Defaults to
        the module name.
    :returns: :class:`EmissionResult` carrying the rendered source and the
        path where it landed (if any).
    """
    label = arm_label or module_name
    source = _render(kb=kb, plan=plan, module_name=module_name, arm_label=label)
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(source)
        _LOG.info(
            "codegen: emitted %s (%d bytes) for %s -> %s",
            module_name,
            len(source),
            plan.solver_name,
            output_path,
        )
    else:
        _LOG.info(
            "codegen: rendered %s (%d bytes, in-memory) for %s",
            module_name,
            len(source),
            plan.solver_name,
        )
    return EmissionResult(
        module_name=module_name,
        output_path=output_path,
        source=source,
    )


# Maps the dispatcher's solver_name (e.g. ``ikgeo.three_parallel``) onto the
# fully-qualified Python module path under :mod:`ssik.solvers` that the
# emitted artifact will import.
_SOLVER_IMPORT_PATHS: dict[str, str] = {
    "ikgeo.three_parallel": "ssik.solvers.ikgeo.three_parallel",
    "ikgeo.spherical_two_parallel": "ssik.solvers.ikgeo.spherical_two_parallel",
    "ikgeo.spherical_two_intersecting": "ssik.solvers.ikgeo.spherical_two_intersecting",
    "ikgeo.spherical": "ssik.solvers.ikgeo.spherical",
    "ikgeo.two_parallel": "ssik.solvers.ikgeo.two_parallel",
    "ikgeo.two_intersecting": "ssik.solvers.ikgeo.two_intersecting",
    "ikgeo.general_6r": "ssik.solvers.ikgeo.general_6r",
    "husty_pfurner.general_6r": "ssik.solvers.husty_pfurner.general_6r",
    "seven_r.srs": "ssik.solvers.seven_r.srs",
    "seven_r.srs_polished": "ssik.solvers.seven_r.srs_polished",
    "seven_r.spherical_shoulder": "ssik.solvers.seven_r.spherical_shoulder",
    "seven_r.spherical_shoulder_polished": "ssik.solvers.seven_r.spherical_shoulder_polished",
    "jointlock.seven_r": "ssik.solvers.jointlock.seven_r",
}


def _render(*, kb: KinBody, plan: DispatchPlan, module_name: str, arm_label: str) -> str:
    """Render the artifact source as a single string.

    Picks between the **specialised** form (sympy-driven inlined trig with
    arm constants substituted; #112) and the **thin wrapper** form (calls
    into ssik solver at runtime; #110 Phase 1 default). The specialised
    form is preferred when a per-solver composer is registered.
    """
    if plan.solver_name in _SPECIALISED_COMPOSERS:
        return _render_specialised(kb=kb, plan=plan, module_name=module_name, arm_label=arm_label)
    return _render_thin_wrapper(kb=kb, plan=plan, module_name=module_name, arm_label=arm_label)


def _render_thin_wrapper(
    *, kb: KinBody, plan: DispatchPlan, module_name: str, arm_label: str
) -> str:
    """Original Phase-1 emitter: `_solver_solve(_KB, T_target, ...)` wrapper.

    Used for solvers without a registered specialised composer (today:
    every solver except spherical_two_parallel; expand as composers land).
    """
    solver_module = _SOLVER_IMPORT_PATHS[plan.solver_name]
    solver_short = plan.solver_name.split(".")[-1]

    buf = StringIO()
    buf.write(_render_header(module_name, arm_label, plan, kb))
    buf.write("\n\nfrom __future__ import annotations\n\n")
    buf.write("import numpy as np\n\n")
    buf.write("from ssik._kinbody import Joint, KinBody, Link\n")
    buf.write("from ssik.core.solution import Solution\n")
    buf.write("from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy\n")
    buf.write("from ssik.postprocess import (\n")
    buf.write("    nearest_to_seed as _ps_nearest_to_seed,\n")
    buf.write("    respect_limits as _ps_respect_limits,\n")
    buf.write("    within_seed_tolerance as _ps_within_seed_tolerance,\n")
    buf.write("    wrap_to_limits as _ps_wrap_to_limits,\n")
    buf.write(")\n")
    # Bulletproof rescue fallback (#319 / #358): thin-wrapper solvers (the
    # SRS family -- seven_r.srs / srs_polished) get the same T-perturbation
    # rescue the specialised orchestrators have, so a reachable-but-degenerate
    # pose the analytical path empties on is recovered instead of returning [].
    buf.write("import functools as _functools\n")
    buf.write("from ssik.refinement import kinbody_jacobian as _kinbody_jacobian\n")
    # Seeded numerical-tracking fast path (#380): Newton-continue from q_seed for
    # the single-IK trajectory-tracking idiom instead of resolving the whole
    # redundancy. Falls through to the full analytical solve when it can't
    # cleanly continue, so it never trades correctness for speed.
    buf.write("from ssik.refinement import seeded_track as _seeded_track\n")
    buf.write(
        "from ssik.refinement.rescue import "
        "rescue_via_T_perturbation as _rescue_via_T_perturbation\n"
    )
    # Exact joint-limit-aware swivel resolution for SRS-class 7R (#359): recovers
    # in-limits solutions the blind swivel sweep drops. No-op (returns []) for
    # non-SRS chains, so it's safe to import for every thin-wrapper arm.
    buf.write(
        "from ssik.solvers.seven_r._swivel_limits import resolve_in_limits as _resolve_in_limits\n"
    )
    buf.write(f"from {solver_module} import solve as _solver_solve\n\n")
    buf.write(f'SOLVER_NAME = "{plan.solver_name}"\n')
    buf.write(f"SOLVER_TIER = {plan.tier}\n")
    buf.write(f"EXPECTED_MS_MEDIAN = {plan.expected_ms_median!r}\n")
    buf.write(f"FLOP_BUDGET = {plan.flop_budget}\n")
    buf.write(_render_dispatch_reason(plan.reason))
    buf.write(_render_introspection_constants(kb))
    buf.write("\n")
    buf.write(_render_kinbody_constants(kb))
    buf.write("\n")
    buf.write(_render_kinbody_builder())
    buf.write("\n")
    buf.write(_render_solve_function(solver_short))
    buf.write(_render_fk_function_from_kb())
    buf.write(_render_all_export())
    return buf.getvalue()


def _render_specialised(
    *, kb: KinBody, plan: DispatchPlan, module_name: str, arm_label: str
) -> str:
    """Phase 1.5 emitter (#112): inlined per-arm trig + arithmetic.

    Calls the registered composer to produce `_solve_algebraic(T_target)`
    with all of the SP1/SP3/SP4 closed-form math expanded inline + arm
    constants substituted. The artifact's `solve()` orchestrator wraps
    the algebraic candidates with FK verification + dedup, mirroring
    `verify_candidates`.
    """
    composer = _SPECIALISED_COMPOSERS[plan.solver_name]
    composer_module = composer.__module__
    composer_func_name = composer.__name__

    # Local import to avoid a hard dep cycle.
    from importlib import import_module

    comp_mod = import_module(composer_module)
    compose = getattr(comp_mod, composer_func_name)
    render_constants_header = getattr(comp_mod, "render_constants_header")  # noqa: B009

    algebraic_body = compose(kb)

    buf = StringIO()
    buf.write(_render_header(module_name, arm_label, plan, kb))
    buf.write("\n\nfrom __future__ import annotations\n\n")
    buf.write(render_constants_header())
    # ``import cython`` enables the pure-Python-mode Cython annotations
    # in the orchestrator below (``@cython.ccall``, ``@cython.locals``).
    # No-op when the artifact runs as plain Python; takes effect when
    # the artifact is compiled to ``.so`` via ``scripts/build_cython.py``
    # (#137 Slice 3).
    buf.write("\nimport cython\nimport numpy as np\n\n")
    buf.write("from ssik._kinbody import Joint, KinBody, Link\n")
    buf.write("from ssik.core.solution import Solution\n")
    buf.write("from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy\n")
    # _spatial_jacobian is inlined per-arm in the orchestrator (#126); the
    # only refinement primitive imported from runtime is the generic
    # Levenberg-Marquardt step.
    buf.write("from ssik.refinement import lm_refine as _lm_refine\n")
    # Bulletproof rescue fallback (#319): when the analytical path returns no
    # solutions at a reachable-but-degenerate ridge, ``solve()`` recovers via
    # the T-perturbation rescue, gated on a reach-sphere so genuinely
    # out-of-reach targets stay cheap.
    buf.write("import functools as _functools\n")
    buf.write(
        "from ssik.refinement.rescue import "
        "rescue_via_T_perturbation as _rescue_via_T_perturbation\n"
    )
    buf.write("from ssik.postprocess import (\n")
    buf.write("    nearest_to_seed as _ps_nearest_to_seed,\n")
    buf.write("    respect_limits as _ps_respect_limits,\n")
    buf.write("    within_seed_tolerance as _ps_within_seed_tolerance,\n")
    buf.write("    wrap_to_limits as _ps_wrap_to_limits,\n")
    buf.write(")\n")
    buf.write("from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix\n\n")
    buf.write(f'SOLVER_NAME = "{plan.solver_name}"\n')
    buf.write(f"SOLVER_TIER = {plan.tier}\n")
    buf.write(f"EXPECTED_MS_MEDIAN = {plan.expected_ms_median!r}\n")
    buf.write(f"FLOP_BUDGET = {plan.flop_budget}\n")
    buf.write(_render_dispatch_reason(plan.reason))
    buf.write(_render_introspection_constants(kb))
    buf.write("\n")
    buf.write(_render_kinbody_constants(kb))
    buf.write("\n")
    buf.write(_render_kinbody_builder())
    buf.write("\n\n")
    buf.write(algebraic_body)
    buf.write("\n")
    # 7R artifacts get a different orchestrator: their public ``solve()``
    # exposes ``max_solutions`` and ``q_seed`` (forwarded to the underlying
    # lock-sweep for early-exit; see #142). 6R artifacts keep the original
    # exhaustive-only API -- their algebraic path doesn't have a sweep to
    # short-circuit, and these args would just postprocess (which is what
    # ``ssik.postprocess`` is for).
    if plan.solver_name == "jointlock.seven_r":
        buf.write(_render_specialised_solve_orchestrator_7r())
    else:
        buf.write(
            _render_specialised_solve_orchestrator(
                _SPECIALISED_FK_ATOL_EXPR.get(plan.solver_name, "policy.subproblem_numerical"),
                force_refine=plan.solver_name in _SPECIALISED_FORCE_REFINE,
            )
        )
    buf.write(_render_fk_alias())
    buf.write(_render_all_export())
    return buf.getvalue()


# Per-solver FK-verify gate baked into the specialised artifact's solve(). The
# default (``policy.subproblem_numerical`` = 1e-5) is what tier-2 RR arms rely on
# -- their algebraic FK can drift above it and is recovered by refinement. Exact
# Pieper solvers that can emit a near-singular SP-clip near-miss (~1e-6 FK, no
# real IK nearby) tighten the gate to drop it (#362); keep in sync with the live
# solver's own gate.
_SPECIALISED_FK_ATOL_EXPR: dict[str, str] = {
    "ikgeo.three_parallel": "1e-7",  # == ssik.solvers.ikgeo.three_parallel._FK_VERIFY_ATOL
}

# Solvers whose artifact always Newton-polishes near-miss candidates (even when
# the caller passes ``allow_refinement=False``). At a near-singular pose the
# closed form only reaches ~1e-6 FK; polish then separates genuine near-singular
# solutions (converge -> kept, #288) from spurious boundary near-misses (stall ->
# dropped, #362). Fires only for the rare > fk_atol candidate. Mirrors the live
# solver's unconditional ``allow_refinement=True``. Not for tier-2 RR arms --
# their near-misses are common and their default (drop) / opt-in-polish contract
# is deliberate.
_SPECIALISED_FORCE_REFINE: frozenset[str] = frozenset({"ikgeo.three_parallel"})


def _render_specialised_solve_orchestrator(
    fk_atol_expr: str = "policy.subproblem_numerical",
    force_refine: bool = False,
) -> str:
    """Render the public ``solve()`` for specialised artifacts.

    Wraps ``_solve_algebraic`` with FK verification + wrap-to-pi dedup;
    matches the runtime ``verify_candidates`` semantics. Exposes the
    same kwargs as the thin-wrapper version.

    Both ``_fk`` and ``_spatial_jacobian`` are inlined per-arm: they
    iterate over the baked ``_JOINT_AXES`` / ``_JOINT_T_LEFTS`` /
    ``_JOINT_T_RIGHTS`` arrays directly without a ``_KB`` indirection.
    Cython compiles those loops to native code with const-folded
    chain constants -- prerequisite for the Level 3 numerical backstop
    being a clean ``.so``. Math is identical to
    :func:`ssik.refinement.kinbody_jacobian`; only the indirection
    changes.
    """
    template = textwrap.dedent(
        '''\

        # Module-scope ``2*pi`` constant referenced inside the dedup hot
        # loop (Cython compiles ``_TWO_PI`` to a typed C ``double``).
        _TWO_PI: float = 2.0 * math.pi

        # Cached 4x4 identity reused inside ``_fk`` / ``_spatial_jacobian``
        # so each call avoids ``len(_JOINT_AXES)+1`` per-iteration ``np.eye(4)``
        # allocations -- the orchestrator's #1 hotspot per Slice 4 profile
        # (~22% of ``_fk`` cost on Puma 560).
        _FK_EYE4 = np.eye(4, dtype=np.float64)
        _FK_EYE4.flags.writeable = False


        @cython.ccall
        @cython.locals(
            i=cython.int,
            n=cython.int,
            ax=cython.double, ay=cython.double, az=cython.double,
            qi=cython.double, c=cython.double, s=cython.double, oc=cython.double,
            r00=cython.double, r01=cython.double, r02=cython.double,
            r10=cython.double, r11=cython.double, r12=cython.double,
            r20=cython.double, r21=cython.double, r22=cython.double,
            l00=cython.double, l01=cython.double, l02=cython.double, l03=cython.double,
            l10=cython.double, l11=cython.double, l12=cython.double, l13=cython.double,
            l20=cython.double, l21=cython.double, l22=cython.double, l23=cython.double,
            m00=cython.double, m01=cython.double, m02=cython.double, m03=cython.double,
            m10=cython.double, m11=cython.double, m12=cython.double, m13=cython.double,
            m20=cython.double, m21=cython.double, m22=cython.double, m23=cython.double,
            t00=cython.double, t01=cython.double, t02=cython.double, t03=cython.double,
            t10=cython.double, t11=cython.double, t12=cython.double, t13=cython.double,
            t20=cython.double, t21=cython.double, t22=cython.double, t23=cython.double,
            n00=cython.double, n01=cython.double, n02=cython.double, n03=cython.double,
            n10=cython.double, n11=cython.double, n12=cython.double, n13=cython.double,
            n20=cython.double, n21=cython.double, n22=cython.double, n23=cython.double,
            a00=cython.double, a01=cython.double, a02=cython.double, a03=cython.double,
            a10=cython.double, a11=cython.double, a12=cython.double, a13=cython.double,
            a20=cython.double, a21=cython.double, a22=cython.double, a23=cython.double,
            b00=cython.double, b01=cython.double, b02=cython.double, b03=cython.double,
            b10=cython.double, b11=cython.double, b12=cython.double, b13=cython.double,
            b20=cython.double, b21=cython.double, b22=cython.double, b23=cython.double,
        )
        def _fk(q):
            """POE forward kinematics using the baked chain constants.

            Hand-rolled scalar 4x4 matmul + inline Rodrigues -- no per-call
            ``np.eye(4)`` allocations and no per-joint numpy ``@`` dispatch.
            Each numpy ``@`` on a 4x4 has ~3 us of dispatch overhead;
            inlining the ~85 scalar ops per joint turns the inner loop into
            a single native-code chunk under Cython compile.

            Bottom row of the accumulator stays [0, 0, 0, 1] implicitly.
            """
            n = len(_JOINT_AXES)
            # Identity accumulator (the bottom row [0,0,0,1] is implicit).
            a00 = 1.0; a01 = 0.0; a02 = 0.0; a03 = 0.0
            a10 = 0.0; a11 = 1.0; a12 = 0.0; a13 = 0.0
            a20 = 0.0; a21 = 0.0; a22 = 1.0; a23 = 0.0
            for i in range(n):
                # Inline Rodrigues for this joint's axis.
                ax = float(_JOINT_AXES[i][0])
                ay = float(_JOINT_AXES[i][1])
                az = float(_JOINT_AXES[i][2])
                qi = float(q[i])
                c = math.cos(qi); s = math.sin(qi); oc = 1.0 - c
                r00 = c + ax*ax*oc;     r01 = ax*ay*oc - az*s; r02 = ax*az*oc + ay*s
                r10 = ay*ax*oc + az*s;  r11 = c + ay*ay*oc;    r12 = ay*az*oc - ax*s
                r20 = az*ax*oc - ay*s;  r21 = az*ay*oc + ax*s; r22 = c + az*az*oc
                # T_left[i] entries.
                Tl = _JOINT_T_LEFTS[i]
                l00 = float(Tl[0,0]); l01 = float(Tl[0,1])
                l02 = float(Tl[0,2]); l03 = float(Tl[0,3])
                l10 = float(Tl[1,0]); l11 = float(Tl[1,1])
                l12 = float(Tl[1,2]); l13 = float(Tl[1,3])
                l20 = float(Tl[2,0]); l21 = float(Tl[2,1])
                l22 = float(Tl[2,2]); l23 = float(Tl[2,3])
                # M = T_left[i] @ R (R is the homogeneous version of the 3x3
                # rotation above with column 3 = [0,0,0,1]^T).
                m00 = l00*r00 + l01*r10 + l02*r20
                m01 = l00*r01 + l01*r11 + l02*r21
                m02 = l00*r02 + l01*r12 + l02*r22
                m03 = l03
                m10 = l10*r00 + l11*r10 + l12*r20
                m11 = l10*r01 + l11*r11 + l12*r21
                m12 = l10*r02 + l11*r12 + l12*r22
                m13 = l13
                m20 = l20*r00 + l21*r10 + l22*r20
                m21 = l20*r01 + l21*r11 + l22*r21
                m22 = l20*r02 + l21*r12 + l22*r22
                m23 = l23
                # T_right[i] entries.
                Tr = _JOINT_T_RIGHTS[i]
                t00 = float(Tr[0,0]); t01 = float(Tr[0,1])
                t02 = float(Tr[0,2]); t03 = float(Tr[0,3])
                t10 = float(Tr[1,0]); t11 = float(Tr[1,1])
                t12 = float(Tr[1,2]); t13 = float(Tr[1,3])
                t20 = float(Tr[2,0]); t21 = float(Tr[2,1])
                t22 = float(Tr[2,2]); t23 = float(Tr[2,3])
                # N = M @ T_right[i]
                n00 = m00*t00 + m01*t10 + m02*t20
                n01 = m00*t01 + m01*t11 + m02*t21
                n02 = m00*t02 + m01*t12 + m02*t22
                n03 = m00*t03 + m01*t13 + m02*t23 + m03
                n10 = m10*t00 + m11*t10 + m12*t20
                n11 = m10*t01 + m11*t11 + m12*t21
                n12 = m10*t02 + m11*t12 + m12*t22
                n13 = m10*t03 + m11*t13 + m12*t23 + m13
                n20 = m20*t00 + m21*t10 + m22*t20
                n21 = m20*t01 + m21*t11 + m22*t21
                n22 = m20*t02 + m21*t12 + m22*t22
                n23 = m20*t03 + m21*t13 + m22*t23 + m23
                # T_acc = T_acc @ N
                b00 = a00*n00 + a01*n10 + a02*n20
                b01 = a00*n01 + a01*n11 + a02*n21
                b02 = a00*n02 + a01*n12 + a02*n22
                b03 = a00*n03 + a01*n13 + a02*n23 + a03
                b10 = a10*n00 + a11*n10 + a12*n20
                b11 = a10*n01 + a11*n11 + a12*n21
                b12 = a10*n02 + a11*n12 + a12*n22
                b13 = a10*n03 + a11*n13 + a12*n23 + a13
                b20 = a20*n00 + a21*n10 + a22*n20
                b21 = a20*n01 + a21*n11 + a22*n21
                b22 = a20*n02 + a21*n12 + a22*n22
                b23 = a20*n03 + a21*n13 + a22*n23 + a23
                a00, a01, a02, a03 = b00, b01, b02, b03
                a10, a11, a12, a13 = b10, b11, b12, b13
                a20, a21, a22, a23 = b20, b21, b22, b23
            return np.array(
                [[a00, a01, a02, a03],
                 [a10, a11, a12, a13],
                 [a20, a21, a22, a23],
                 [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )


        @cython.ccall
        @cython.locals(i=cython.int, n=cython.int)
        def _spatial_jacobian(q):
            """6 x n_dof spatial Jacobian using the baked chain constants.

            Math identical to ssik.refinement.kinbody_jacobian: column i
            is (p_i x z_i, z_i) where z_i is the i-th joint axis in the
            world frame at q and p_i is the i-th joint origin. This is
            the SPATIAL twist representation -- T(q+dq) @ T(q)^-1 ~
            exp([J @ dq]) -- matching the residual extracted by
            ssik.refinement.se3_log_residual. Per-arm version with
            baked _JOINT_AXES / _JOINT_T_LEFTS / _JOINT_T_RIGHTS so
            there's no KinBody walk at runtime.
            """
            n = len(_JOINT_AXES)
            cum = _FK_EYE4.copy()
            cums = [cum.copy()]
            rot = _FK_EYE4.copy()
            for i in range(n):
                rot[:3, :3] = _rotation_matrix(_JOINT_AXES[i], float(q[i]))
                cum = cum @ _JOINT_T_LEFTS[i] @ rot @ _JOINT_T_RIGHTS[i]
                cums.append(cum.copy())
            J = np.zeros((6, n), dtype=np.float64)
            for i in range(n):
                t_pre = cums[i] @ _JOINT_T_LEFTS[i]
                axis_unit = _JOINT_AXES[i] / np.linalg.norm(_JOINT_AXES[i])
                z_i = t_pre[:3, :3] @ axis_unit
                p_i = t_pre[:3, 3]
                J[:3, i] = np.cross(p_i, z_i)
                J[3:, i] = z_i
            return J


        @cython.ccall
        def _wrap_to_pi(a: float) -> float:
            """Wrap an angle to ``(-pi, pi]``. Called inside the per-IK
            dedup hot loop (235k+ times on Franka 7R)."""
            return ((a + math.pi) % _TWO_PI) - math.pi


        @cython.ccall
        @cython.locals(
            i=cython.int,
            n=cython.int,
            diff=cython.double,
            ai=cython.double,
            bi=cython.double,
        )
        def _q_close_wrap(a, b, tol: float) -> bool:
            """Return ``True`` if joint vectors ``a`` and ``b`` agree (mod 2pi)
            within ``tol`` per element. Replaces the
            ``np.array([_wrap_to_pi(...)]) -> np.all(np.abs(...) < tol)``
            pipeline that allocated a numpy array per dedup-loop iteration --
            a per-element scalar loop avoids the array creation and the
            ``np.all`` reduction overhead, which together dominated the
            artifact's ``solve()`` body at the per-IK level."""
            n = len(a)
            for i in range(n):
                ai = float(a[i])
                bi = float(b[i])
                diff = ((ai - bi + math.pi) % _TWO_PI) - math.pi
                if abs(diff) > tol:
                    return False
            return True


        def solve(
            T_target,
            *,
            max_solutions: int | None = None,
            q_seed=None,
            respect_limits: bool = True,
            allow_refinement: bool = False,
            allow_rescue: bool = True,
            policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
            refinement_max_iters: int = 15,
            seed_metric: str = "wrap_linf",
            seed_tolerance: float | None = None,
        ):
            """Inverse kinematics. Returns ``list[Solution]``.

            :param T_target: 4x4 SE(3) target end-effector pose.
            :param max_solutions: optional cap on returned IKs (post-dedup,
                post-limits filter). ``None`` = full redundancy enumeration.
                Combine with ``q_seed`` for the "give me the IK closest to
                where I am now" trajectory-tracking idiom.
            :param q_seed: optional joint configuration. When provided,
                returned solutions are sorted by distance from ``q_seed``
                (closest first, via ``seed_metric``); with ``max_solutions``
                this returns the nearest ``max_solutions`` to the seed -- the
                trajectory-tracking idiom.
            :param seed_metric: distance used to rank against ``q_seed``.
                ``"wrap_linf"`` (default) minimises the *largest* single-joint
                wrap-to-pi move, which holds the branch during tracking;
                ``"wrap_l2"`` minimises the summed move (can favour a flip
                "paid for" by smaller moves elsewhere). Ignored when
                ``q_seed`` is ``None``.
            :param seed_tolerance: optional max per-joint deviation from
                ``q_seed`` (radians, wrap-to-pi). When set, only solutions with
                *every* joint within ``seed_tolerance`` are returned -- a hard
                tracking guarantee that may return an empty list when no branch
                qualifies. ``None`` (default) keeps the best-effort behaviour.
                Requires ``q_seed``.
            :param respect_limits: when ``True`` (default), solutions
                outside URDF joint limits are dropped. Pass ``False`` for
                the raw geometric set (e.g. analysis / debugging).
            :param allow_refinement: opt into Newton polish for near-miss
                algebraic candidates that don't quite meet ``fk_atol``.
                Default ``False`` -- the algebraic path is already at
                machine precision on tier-0 / SRS arms. On tier-2 RR
                arms (JACO 2, Rizon 4, Kassow), polish can recover
                edge-case candidates whose algebraic FK drifts above
                ``fk_atol``, at ~100-300 us per polished branch.
            :param allow_rescue: when ``True`` (default), if the analytical
                path returns no solutions for a target within the arm's
                reach (a measure-zero rank-deficient RR ridge -- a
                reachable pose the algebraic path can't extract),
                ``solve()`` recovers the IK via the T-perturbation rescue
                (#319), returning machine-precision solutions tagged
                ``refinement_used="lm"``. Set ``False`` for a guaranteed
                purely-analytical result (returns ``[]`` at such ridges).
                Gated by a reach-sphere, so far-field unreachable targets
                stay cheap.
            :param policy: tolerance policy (FK closure + dedup tolerance).
                Rarely customised.
            :param refinement_max_iters: cap on Newton iterations per
                candidate when ``allow_refinement=True``.
            :returns: list of :class:`Solution`; empty list iff no IK
                closed within ``policy.subproblem_numerical`` (or all
                IKs were filtered by ``respect_limits=True``).
            """
            if seed_tolerance is not None and q_seed is None:
                raise ValueError("seed_tolerance requires q_seed")
            T = np.asarray(T_target, dtype=np.float64)
            candidates = _solve_algebraic(T)

            fk_atol = policy.subproblem_numerical
            dedup_atol = policy.subproblem_dedup

            # Three-bucket sort: exact (closes within fk_atol), near-miss
            # (refinable when allow_refinement=True), or drop.
            verified: list[tuple[np.ndarray, float, str, int]] = []
            for cand_q in candidates:
                q = np.asarray(cand_q, dtype=np.float64)
                if not np.all(np.isfinite(q)):
                    continue
                T_check = _fk(q)
                residual = float(np.linalg.norm(T_check - T))
                if residual <= fk_atol:
                    verified.append((q, residual, "none", 0))
                    continue
                if not allow_refinement:
                    continue
                # Newton polish using the per-arm spatial Jacobian.
                refined = _lm_refine(
                    q,
                    _fk,
                    T,
                    fk_atol=fk_atol,
                    max_iters=refinement_max_iters,
                    jacobian_fn=_spatial_jacobian,
                )
                if refined is None:
                    continue
                q_ref, resid_ref, iters = refined
                verified.append((q_ref, resid_ref, "lm", iters))

            # Wrap-to-pi dedup; keep lowest fk_residual on collision.
            # Inner check via ``_q_close_wrap`` -- typed scalar loop, no per-
            # iteration numpy allocation (#137 Slice 3).
            deduped: list[tuple[np.ndarray, float, str, int]] = []
            for cand_q, cand_res, ref_used, ref_iters in verified:
                dup_idx = None
                for j, (existing_q, _, _, _) in enumerate(deduped):
                    if _q_close_wrap(cand_q, existing_q, dedup_atol):
                        dup_idx = j
                        break
                if dup_idx is None:
                    deduped.append((cand_q, cand_res, ref_used, ref_iters))
                elif cand_res < deduped[dup_idx][1]:
                    deduped[dup_idx] = (cand_q, cand_res, ref_used, ref_iters)

            solutions = [
                Solution(
                    q=q,
                    fk_residual=residual,
                    refinement_used=ref_used,
                )
                for q, residual, ref_used, _ref_iters in deduped
            ]

            # Bulletproof fallback (#319): the analytical path found nothing.
            # If the target is within the arm's max reach it may be a
            # measure-zero rank-deficient ridge (a reachable pose the algebraic
            # path can't extract) rather than an unreachable target -- recover
            # via the T-perturbation rescue. The reach-sphere (sum of link
            # lengths; an exact upper bound by the triangle inequality, so it
            # never rejects a reachable pose) is the gate: it is checked only
            # here in the rare empty branch and keeps genuinely far-field
            # targets cheap. (The RR real-root count is NOT used as a gate -- it
            # is an unreliable reachability signal: some reachable ridges, e.g.
            # Rizon 4's, yield only complex roots, so gating on it would
            # silently drop real solutions.) The perturbed re-solves run with
            # allow_rescue=False (recursion guard + analytical-only escape
            # hatch). Rescued sols carry refinement_used="lm", FK-gated to
            # machine precision.
            if not solutions and allow_rescue:
                _reach_radius = sum(
                    float(np.linalg.norm(np.asarray(_t)[:3, 3]))
                    for _t in (*_JOINT_T_LEFTS, *_JOINT_T_RIGHTS)
                )
                if float(np.linalg.norm(T[:3, 3])) <= _reach_radius:
                    solutions = _rescue_via_T_perturbation(
                        _fk,
                        _functools.partial(solve, allow_rescue=False),
                        T,
                        jacobian_fn=_spatial_jacobian,
                    )

            # Post-processing pass (#238 item 4). Order matters:
            #   1. wrap_to_limits tries q +/- 2*pi per joint to bring
            #      candidates into the URDF's limit range
            #   2. respect_limits drops anything still outside
            #   3. nearest_to_seed sorts by distance to q_seed (if given)
            #   4. max_solutions truncates to the first k
            if respect_limits:
                solutions = _ps_wrap_to_limits(solutions, _KB)
                solutions = _ps_respect_limits(solutions, _KB)
            if q_seed is not None:
                if seed_tolerance is not None:
                    solutions = _ps_within_seed_tolerance(solutions, q_seed, seed_tolerance)
                solutions = _ps_nearest_to_seed(solutions, q_seed, metric=seed_metric)
            if max_solutions is not None and len(solutions) > max_solutions:
                solutions = solutions[:max_solutions]
            return solutions
        '''
    ).replace("fk_atol = policy.subproblem_numerical", f"fk_atol = {fk_atol_expr}")
    if force_refine:
        # Always polish near-misses (see _SPECIALISED_FORCE_REFINE). Only rewrite
        # the gate when set, so non-forced artifacts stay byte-identical.
        template = template.replace(
            "if not allow_refinement:", "if not (allow_refinement or True):"
        )
    return template


def _render_specialised_solve_orchestrator_7r() -> str:
    """Render ``solve()`` for 7R artifacts (jointlock.seven_r).

    Extends the 6R orchestrator with ``max_solutions`` and ``q_seed``
    kwargs (forwarded to ``_solve_algebraic`` so the underlying
    lock-sweep can short-circuit -- see #142). Defaults preserve the
    exhaustive-search semantic; the early-exit is opt-in.
    """
    return textwrap.dedent(
        '''\

        # Cached 4x4 identity reused inside ``_fk`` / ``_spatial_jacobian``
        # so each call avoids ``len(_JOINT_AXES)+1`` per-iteration ``np.eye(4)``
        # allocations -- the orchestrator's #1 hotspot per Slice 4 profile
        # (~22% of ``_fk`` cost on Puma 560).
        _FK_EYE4 = np.eye(4, dtype=np.float64)
        _FK_EYE4.flags.writeable = False


        @cython.ccall
        @cython.locals(
            i=cython.int,
            n=cython.int,
            ax=cython.double, ay=cython.double, az=cython.double,
            qi=cython.double, c=cython.double, s=cython.double, oc=cython.double,
            r00=cython.double, r01=cython.double, r02=cython.double,
            r10=cython.double, r11=cython.double, r12=cython.double,
            r20=cython.double, r21=cython.double, r22=cython.double,
            l00=cython.double, l01=cython.double, l02=cython.double, l03=cython.double,
            l10=cython.double, l11=cython.double, l12=cython.double, l13=cython.double,
            l20=cython.double, l21=cython.double, l22=cython.double, l23=cython.double,
            m00=cython.double, m01=cython.double, m02=cython.double, m03=cython.double,
            m10=cython.double, m11=cython.double, m12=cython.double, m13=cython.double,
            m20=cython.double, m21=cython.double, m22=cython.double, m23=cython.double,
            t00=cython.double, t01=cython.double, t02=cython.double, t03=cython.double,
            t10=cython.double, t11=cython.double, t12=cython.double, t13=cython.double,
            t20=cython.double, t21=cython.double, t22=cython.double, t23=cython.double,
            n00=cython.double, n01=cython.double, n02=cython.double, n03=cython.double,
            n10=cython.double, n11=cython.double, n12=cython.double, n13=cython.double,
            n20=cython.double, n21=cython.double, n22=cython.double, n23=cython.double,
            a00=cython.double, a01=cython.double, a02=cython.double, a03=cython.double,
            a10=cython.double, a11=cython.double, a12=cython.double, a13=cython.double,
            a20=cython.double, a21=cython.double, a22=cython.double, a23=cython.double,
            b00=cython.double, b01=cython.double, b02=cython.double, b03=cython.double,
            b10=cython.double, b11=cython.double, b12=cython.double, b13=cython.double,
            b20=cython.double, b21=cython.double, b22=cython.double, b23=cython.double,
        )
        def _fk(q):
            """POE forward kinematics using the baked chain constants.

            Hand-rolled scalar 4x4 matmul + inline Rodrigues -- no per-call
            ``np.eye(4)`` allocations and no per-joint numpy ``@`` dispatch.
            Each numpy ``@`` on a 4x4 has ~3 us of dispatch overhead;
            inlining the ~85 scalar ops per joint turns the inner loop into
            a single native-code chunk under Cython compile.

            Bottom row of the accumulator stays [0, 0, 0, 1] implicitly.
            """
            n = len(_JOINT_AXES)
            # Identity accumulator (the bottom row [0,0,0,1] is implicit).
            a00 = 1.0; a01 = 0.0; a02 = 0.0; a03 = 0.0
            a10 = 0.0; a11 = 1.0; a12 = 0.0; a13 = 0.0
            a20 = 0.0; a21 = 0.0; a22 = 1.0; a23 = 0.0
            for i in range(n):
                # Inline Rodrigues for this joint's axis.
                ax = float(_JOINT_AXES[i][0])
                ay = float(_JOINT_AXES[i][1])
                az = float(_JOINT_AXES[i][2])
                qi = float(q[i])
                c = math.cos(qi); s = math.sin(qi); oc = 1.0 - c
                r00 = c + ax*ax*oc;     r01 = ax*ay*oc - az*s; r02 = ax*az*oc + ay*s
                r10 = ay*ax*oc + az*s;  r11 = c + ay*ay*oc;    r12 = ay*az*oc - ax*s
                r20 = az*ax*oc - ay*s;  r21 = az*ay*oc + ax*s; r22 = c + az*az*oc
                # T_left[i] entries.
                Tl = _JOINT_T_LEFTS[i]
                l00 = float(Tl[0,0]); l01 = float(Tl[0,1])
                l02 = float(Tl[0,2]); l03 = float(Tl[0,3])
                l10 = float(Tl[1,0]); l11 = float(Tl[1,1])
                l12 = float(Tl[1,2]); l13 = float(Tl[1,3])
                l20 = float(Tl[2,0]); l21 = float(Tl[2,1])
                l22 = float(Tl[2,2]); l23 = float(Tl[2,3])
                # M = T_left[i] @ R (R is the homogeneous version of the 3x3
                # rotation above with column 3 = [0,0,0,1]^T).
                m00 = l00*r00 + l01*r10 + l02*r20
                m01 = l00*r01 + l01*r11 + l02*r21
                m02 = l00*r02 + l01*r12 + l02*r22
                m03 = l03
                m10 = l10*r00 + l11*r10 + l12*r20
                m11 = l10*r01 + l11*r11 + l12*r21
                m12 = l10*r02 + l11*r12 + l12*r22
                m13 = l13
                m20 = l20*r00 + l21*r10 + l22*r20
                m21 = l20*r01 + l21*r11 + l22*r21
                m22 = l20*r02 + l21*r12 + l22*r22
                m23 = l23
                # T_right[i] entries.
                Tr = _JOINT_T_RIGHTS[i]
                t00 = float(Tr[0,0]); t01 = float(Tr[0,1])
                t02 = float(Tr[0,2]); t03 = float(Tr[0,3])
                t10 = float(Tr[1,0]); t11 = float(Tr[1,1])
                t12 = float(Tr[1,2]); t13 = float(Tr[1,3])
                t20 = float(Tr[2,0]); t21 = float(Tr[2,1])
                t22 = float(Tr[2,2]); t23 = float(Tr[2,3])
                # N = M @ T_right[i]
                n00 = m00*t00 + m01*t10 + m02*t20
                n01 = m00*t01 + m01*t11 + m02*t21
                n02 = m00*t02 + m01*t12 + m02*t22
                n03 = m00*t03 + m01*t13 + m02*t23 + m03
                n10 = m10*t00 + m11*t10 + m12*t20
                n11 = m10*t01 + m11*t11 + m12*t21
                n12 = m10*t02 + m11*t12 + m12*t22
                n13 = m10*t03 + m11*t13 + m12*t23 + m13
                n20 = m20*t00 + m21*t10 + m22*t20
                n21 = m20*t01 + m21*t11 + m22*t21
                n22 = m20*t02 + m21*t12 + m22*t22
                n23 = m20*t03 + m21*t13 + m22*t23 + m23
                # T_acc = T_acc @ N
                b00 = a00*n00 + a01*n10 + a02*n20
                b01 = a00*n01 + a01*n11 + a02*n21
                b02 = a00*n02 + a01*n12 + a02*n22
                b03 = a00*n03 + a01*n13 + a02*n23 + a03
                b10 = a10*n00 + a11*n10 + a12*n20
                b11 = a10*n01 + a11*n11 + a12*n21
                b12 = a10*n02 + a11*n12 + a12*n22
                b13 = a10*n03 + a11*n13 + a12*n23 + a13
                b20 = a20*n00 + a21*n10 + a22*n20
                b21 = a20*n01 + a21*n11 + a22*n21
                b22 = a20*n02 + a21*n12 + a22*n22
                b23 = a20*n03 + a21*n13 + a22*n23 + a23
                a00, a01, a02, a03 = b00, b01, b02, b03
                a10, a11, a12, a13 = b10, b11, b12, b13
                a20, a21, a22, a23 = b20, b21, b22, b23
            return np.array(
                [[a00, a01, a02, a03],
                 [a10, a11, a12, a13],
                 [a20, a21, a22, a23],
                 [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )


        @cython.ccall
        @cython.locals(i=cython.int, n=cython.int)
        def _spatial_jacobian(q):
            """6 x n_dof spatial Jacobian using the baked chain constants.

            Math identical to ssik.refinement.kinbody_jacobian: column i
            is (p_i x z_i, z_i) where z_i is the i-th joint axis in the
            world frame at q and p_i is the i-th joint origin. This is
            the SPATIAL twist representation -- T(q+dq) @ T(q)^-1 ~
            exp([J @ dq]) -- matching the residual extracted by
            ssik.refinement.se3_log_residual. Per-arm version with
            baked _JOINT_AXES / _JOINT_T_LEFTS / _JOINT_T_RIGHTS so
            there's no KinBody walk at runtime.
            """
            n = len(_JOINT_AXES)
            cum = _FK_EYE4.copy()
            cums = [cum.copy()]
            rot = _FK_EYE4.copy()
            for i in range(n):
                rot[:3, :3] = _rotation_matrix(_JOINT_AXES[i], float(q[i]))
                cum = cum @ _JOINT_T_LEFTS[i] @ rot @ _JOINT_T_RIGHTS[i]
                cums.append(cum.copy())
            J = np.zeros((6, n), dtype=np.float64)
            for i in range(n):
                t_pre = cums[i] @ _JOINT_T_LEFTS[i]
                axis_unit = _JOINT_AXES[i] / np.linalg.norm(_JOINT_AXES[i])
                z_i = t_pre[:3, :3] @ axis_unit
                p_i = t_pre[:3, 3]
                J[:3, i] = np.cross(p_i, z_i)
                J[3:, i] = z_i
            return J


        # Module-scope ``2*pi`` constant referenced inside the dedup hot
        # loop (Cython compiles ``_TWO_PI`` to a typed C ``double``).
        _TWO_PI: float = 2.0 * math.pi


        @cython.ccall
        def _wrap_to_pi(a: float) -> float:
            """Wrap an angle to ``(-pi, pi]``. Called inside the per-IK
            dedup hot loop (235k+ times on Franka 7R)."""
            return ((a + math.pi) % _TWO_PI) - math.pi


        @cython.ccall
        @cython.locals(
            i=cython.int,
            n=cython.int,
            diff=cython.double,
            ai=cython.double,
            bi=cython.double,
        )
        def _q_close_wrap(a, b, tol: float) -> bool:
            """Return ``True`` if joint vectors ``a`` and ``b`` agree (mod 2pi)
            within ``tol`` per element. Replaces the
            ``np.array([_wrap_to_pi(...)]) -> np.all(np.abs(...) < tol)``
            pipeline that allocated a numpy array per dedup-loop iteration --
            a per-element scalar loop avoids the array creation and the
            ``np.all`` reduction overhead, which together dominated the
            artifact's ``solve()`` body at the per-IK level."""
            n = len(a)
            for i in range(n):
                ai = float(a[i])
                bi = float(b[i])
                diff = ((ai - bi + math.pi) % _TWO_PI) - math.pi
                if abs(diff) > tol:
                    return False
            return True


        def solve(
            T_target,
            *,
            max_solutions: int | None = None,
            q_seed=None,
            respect_limits: bool = True,
            allow_refinement: bool = False,
            allow_rescue: bool = True,
            policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
            refinement_max_iters: int = 15,
            seed_metric: str = "wrap_linf",
            seed_tolerance: float | None = None,
        ):
            """Inverse kinematics. Returns ``list[Solution]``.

            :param T_target: 4x4 SE(3) target end-effector pose.
            :param max_solutions: optional early-exit cap on the
                jointlock lock-sweep. ``None`` (default) = exhaustive
                search. ``max_solutions=1`` short-circuits as soon as
                one valid IK is found (~17x faster on this 7R).
            :param q_seed: optional length-7 seed configuration. When
                provided, the lock-joint samples are visited outward from
                ``q_seed[lock_idx]`` and the first yielding slice's full
                branch set is L-infinity-ranked against the seed (see
                ``seed_metric``); with ``max_solutions`` this returns the
                nearest configs to the seed in ~1 sub-solve -- the
                trajectory-tracking fast path (#331), branch-continuous and
                ~20x faster than the exhaustive sweep.
            :param seed_metric: distance used to rank against ``q_seed``.
                ``"wrap_linf"`` (default) minimises the *largest* single-joint
                wrap-to-pi move, holding the branch during tracking;
                ``"wrap_l2"`` minimises the summed move. Ignored when
                ``q_seed`` is ``None``.
            :param seed_tolerance: optional max per-joint deviation from
                ``q_seed`` (radians, wrap-to-pi). When set, only solutions with
                *every* joint within ``seed_tolerance`` are returned -- a hard
                tracking guarantee that may return an empty list when no branch
                qualifies. ``None`` (default) keeps the best-effort behaviour.
                Requires ``q_seed``.
            :param respect_limits: when ``True`` (default), solutions
                outside URDF joint limits are dropped. Pass ``False``
                for the raw geometric set.
            :param allow_refinement: when ``True`` (default), Newton
                polish fires on near-miss algebraic candidates.
            :param allow_rescue: when ``True`` (default), if the analytical
                path returns no solutions for a target within the arm's
                reach (a measure-zero rank-deficient RR ridge -- a
                reachable pose the algebraic path can't extract),
                ``solve()`` recovers the IK via the T-perturbation rescue
                (#319), returning machine-precision solutions tagged
                ``refinement_used="lm"``. Set ``False`` for a guaranteed
                purely-analytical result. Gated by a reach-sphere, so
                far-field unreachable targets stay cheap.
            :param policy: tolerance policy. Rarely customised.
            :param refinement_max_iters: cap on Newton iterations per
                candidate when ``allow_refinement=True``.

            Common idioms::

                # Exhaustive search (default).
                solutions = solve(T_target)

                # "Just give me one IK" -- ~17x faster.
                solutions = solve(T_target, max_solutions=1)

                # Track current config -- ~37x faster.
                solutions = solve(
                    T_target, q_seed=q_current, max_solutions=1,
                )
            """
            if seed_tolerance is not None and q_seed is None:
                raise ValueError("seed_tolerance requires q_seed")
            T = np.asarray(T_target, dtype=np.float64)
            # Lock-sweep filters limits in-flight (#238 review): the
            # short-circuit fires on the first in-limits valid IK, not
            # on a candidate that postprocess would drop. Preserves the
            # max_solutions+q_seed early-exit fast path even with
            # respect_limits=True.
            candidates = _solve_algebraic(
                T, max_solutions=max_solutions, q_seed=q_seed,
                respect_limits=respect_limits,
            )

            fk_atol = policy.subproblem_numerical
            dedup_atol = policy.subproblem_dedup

            # Three-bucket sort: exact (closes within fk_atol), near-miss
            # (refinable when allow_refinement=True), or drop.
            verified: list[tuple[np.ndarray, float, str, int]] = []
            for cand_q in candidates:
                q = np.asarray(cand_q, dtype=np.float64)
                if not np.all(np.isfinite(q)):
                    continue
                T_check = _fk(q)
                residual = float(np.linalg.norm(T_check - T))
                if residual <= fk_atol:
                    verified.append((q, residual, "none", 0))
                    continue
                if not allow_refinement:
                    continue
                # Newton polish using the per-arm spatial Jacobian.
                refined = _lm_refine(
                    q,
                    _fk,
                    T,
                    fk_atol=fk_atol,
                    max_iters=refinement_max_iters,
                    jacobian_fn=_spatial_jacobian,
                )
                if refined is None:
                    continue
                q_ref, resid_ref, iters = refined
                verified.append((q_ref, resid_ref, "lm", iters))

            # Wrap-to-pi dedup; keep lowest fk_residual on collision.
            # Inner check via ``_q_close_wrap`` -- typed scalar loop, no per-
            # iteration numpy allocation (#137 Slice 3).
            deduped: list[tuple[np.ndarray, float, str, int]] = []
            for cand_q, cand_res, ref_used, ref_iters in verified:
                dup_idx = None
                for j, (existing_q, _, _, _) in enumerate(deduped):
                    if _q_close_wrap(cand_q, existing_q, dedup_atol):
                        dup_idx = j
                        break
                if dup_idx is None:
                    deduped.append((cand_q, cand_res, ref_used, ref_iters))
                elif cand_res < deduped[dup_idx][1]:
                    deduped[dup_idx] = (cand_q, cand_res, ref_used, ref_iters)

            solutions = [
                Solution(
                    q=q,
                    fk_residual=residual,
                    refinement_used=ref_used,
                )
                for q, residual, ref_used, _ref_iters in deduped
            ]
            # Bulletproof fallback (#319): the analytical lock-sweep found
            # nothing. If the target is within the arm's max reach it may be a
            # measure-zero rank-deficient ridge (a reachable pose the algebraic
            # path can't extract) rather than an unreachable target -- recover
            # via the T-perturbation rescue. The reach-sphere (sum of link
            # lengths; an exact upper bound by the triangle inequality, so it
            # never rejects a reachable pose) is the gate: it is checked only
            # here in the rare empty branch and keeps far-field targets cheap.
            # (The cached-RR real-root count is NOT used as a gate -- it is an
            # unreliable reachability signal: some reachable ridges, e.g. Rizon
            # 4's, yield only complex roots, so gating on it would silently drop
            # real solutions.) Perturbed re-solves run with allow_rescue=False
            # (recursion guard + analytical-only escape hatch). The rescue calls
            # back with respect_limits=False, so its output gets the limit/seed
            # postprocess here (the analytical path filtered limits in-flight).
            if not solutions and allow_rescue:
                _reach_radius = sum(
                    float(np.linalg.norm(np.asarray(_t)[:3, 3]))
                    for _t in (*_JOINT_T_LEFTS, *_JOINT_T_RIGHTS)
                )
                if float(np.linalg.norm(T[:3, 3])) <= _reach_radius:
                    solutions = _rescue_via_T_perturbation(
                        _fk,
                        _functools.partial(solve, allow_rescue=False),
                        T,
                        jacobian_fn=_spatial_jacobian,
                    )
                    if respect_limits:
                        solutions = _ps_wrap_to_limits(solutions, _KB)
                        solutions = _ps_respect_limits(solutions, _KB)
                    if q_seed is not None:
                        if seed_tolerance is not None:
                            solutions = _ps_within_seed_tolerance(
                                solutions, q_seed, seed_tolerance
                            )
                        solutions = _ps_nearest_to_seed(solutions, q_seed, metric=seed_metric)

            # No orchestrator-level respect_limits pass on the analytical
            # result: the inner ``_solve_algebraic`` already filtered in-flight
            # when respect_limits=True, so candidates here are guaranteed
            # in-limits.
            #
            # Seeded ranking (#331): the lock-sweep returns candidates from the
            # window of lock samples nearest ``q_seed[lock_idx]`` (in seed
            # order), but the genuinely-nearest config -- and the
            # branch-continuous one for tracking -- needs an explicit rank by
            # ``seed_metric`` (default L-infinity) over that window before the
            # cap. Without this the cap would keep the nearest *lock samples*,
            # not the nearest *configs*.
            if q_seed is not None:
                if seed_tolerance is not None:
                    solutions = _ps_within_seed_tolerance(solutions, q_seed, seed_tolerance)
                solutions = _ps_nearest_to_seed(solutions, q_seed, metric=seed_metric)
            if max_solutions is not None and len(solutions) > max_solutions:
                solutions = solutions[:max_solutions]
            return solutions
        '''
    )


# Per-solver registered composers. Solvers absent from this map fall back
# to the thin-wrapper emitter. Add entries as composers land (#112 plan).
from collections.abc import Callable  # noqa: E402

# ``KinBody`` is in TYPE_CHECKING-only scope at module load; use a string
# forward reference inside ``Callable``.
ComposerFn = Callable[["KinBody"], str]


def _import_composer(module_path: str, func_name: str) -> ComposerFn:
    from importlib import import_module

    fn = getattr(import_module(module_path), func_name)
    return fn  # type: ignore[no-any-return]


_SPECIALISED_COMPOSERS: dict[str, ComposerFn] = {
    "ikgeo.spherical_two_parallel": _import_composer(
        "ssik.codegen._compose.spherical_two_parallel", "compose"
    ),
    "ikgeo.three_parallel": _import_composer("ssik.codegen._compose.three_parallel", "compose"),
    "ikgeo.spherical_two_intersecting": _import_composer(
        "ssik.codegen._compose.spherical_two_intersecting", "compose"
    ),
    "ikgeo.spherical": _import_composer("ssik.codegen._compose.spherical", "compose"),
    "ikgeo.general_6r": _import_composer("ssik.codegen._compose.general_6r", "compose"),
    "jointlock.seven_r": _import_composer("ssik.codegen._compose.seven_r", "compose"),
}


def _kb_digest(kb: KinBody) -> str:
    """Deterministic 12-hex-char digest of the KinBody's kinematic structure.

    Hashes the joint axes, ``T_left`` / ``T_right`` matrices, joint types,
    and link names in chain order. Stable across runs and platforms; lets
    a reviewer recognise the same fixture even if the artifact has been
    edited downstream. Identifies fixture changes too: any drift in the
    chain (axis flip, link rename, transform tweak) shifts the digest.
    """
    import hashlib

    # Bumped to v2 when joint limits joined the kinematic spec (#129).
    h = hashlib.sha256()
    h.update(b"ssik-kinbody-v2\n")
    for link in kb.links:
        h.update(b"L:")
        h.update(link.name.encode("utf-8"))
        h.update(b"\n")
    for joint in kb.joints:
        h.update(b"J:")
        h.update(joint.joint_type.encode("utf-8"))
        h.update(b":")
        h.update((joint.name or "").encode("utf-8"))
        h.update(b":axis=")
        for v in joint.axis.tolist():
            h.update(f"{v!r}".encode())
            h.update(b",")
        h.update(b":Tl=")
        for row in joint.T_left.tolist():
            for v in row:
                h.update(f"{v!r}".encode())
                h.update(b",")
        h.update(b":Tr=")
        for row in joint.T_right.tolist():
            for v in row:
                h.update(f"{v!r}".encode())
                h.update(b",")
        h.update(b":lim=")
        if joint.limits is None:
            h.update(b"None")
        else:
            h.update(f"{joint.limits[0]!r},{joint.limits[1]!r}".encode())
        h.update(b"\n")
    return h.hexdigest()[:12]


def _render_header(module_name: str, arm_label: str, plan: DispatchPlan, kb: KinBody) -> str:
    """Top-of-file docstring with provenance + usage.

    Provenance is the ``KinBody hash``: a 12-hex-char sha256 digest of the
    input chain's kinematic structure. Stable across runs and platforms,
    so it does not churn the artifact snapshot; identifies fixture
    changes (axis flip, link rename, transform tweak) when they happen.

    The ssik commit is intentionally NOT baked because
    ``importlib.metadata.version`` reports the install-time pinned value,
    which drifts between local checkouts and CI -- not actually stable.
    The artifact's ssik provenance lives in the parent repo's git history
    (e.g. ``git log -- tests/artifacts/ur5_ik.py``).

    The docstring includes base / end-effector link names + DOF + home FK
    so a downstream user can immediately see which kinematic frame their
    ``T_target`` is expressed in (and whether the baked chain matches
    their own URDF). ``T_target`` is the pose of ``EE_LINK`` in
    ``BASE_LINK``; if the user's URDF disagrees on link selection
    (calibrated geometry, custom tool past the flange, different
    naming), they should run ``ssik build <their.urdf> --base X --ee Y``
    rather than rely on this artifact.
    """
    digest_str = _kb_digest(kb)
    base_link = kb.links[0].name
    ee_link = kb.links[-1].name
    dof = len(kb.joints)
    return textwrap.dedent(
        f'''\
        """Generated IK module for {arm_label}.

        This file was emitted by ``ssik build`` and is the public artifact for
        running analytical inverse kinematics on this specific arm. The
        per-arm KinBody constants are baked in below; you do not need to
        load a URDF or MJCF at runtime.

        Provenance: KinBody hash {digest_str} (sha256/12 of the input chain).
        ``T_target`` is the pose of ``{ee_link}`` (end-effector link) in
        ``{base_link}`` (base link). If your URDF differs (calibrated
        geometry, custom tool past the flange, different link names),
        run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
        produce an artifact correct for your hardware.

        DOF: {dof}    BASE_LINK: "{base_link}"    EE_LINK: "{ee_link}"
        Solver: ``{plan.solver_name}`` (tier {plan.tier})
        Expected median IK time: ~{plan.expected_ms_median} ms on commodity
        single-thread hardware. FLOP budget: {plan.flop_budget:,} per solve.

        Usage:

            import {module_name}
            import numpy as np
            T_target = np.eye(4)  # 4x4 SE(3) pose of {ee_link} in {base_link}
            T_target[:3, 3] = [0.5, 0.1, 0.3]
            solutions = {module_name}.solve(T_target)
            for sol in solutions:
                print(sol.q, sol.fk_residual)

        ``solve(T)`` returns ``list[Solution]``. Empty list iff no
        candidate closed within the solver's FK tolerance -- check
        ``if not solutions:`` for the "unreachable" case.

        Sanity-check the baked geometry: ``{module_name}.T_HOME`` is the
        4x4 home pose (FK at ``q = np.zeros(DOF)``). If it doesn't match
        your robot's home pose, the artifact is for a different URDF.
        """'''
    )


def _render_dispatch_reason(reason: str) -> str:
    """Bake the dispatcher's explanatory diagnostic as a module-level constant."""
    # ``repr`` keeps newlines and quoting safe inside the rendered source.
    return f"DISPATCH_REASON = {reason!r}\n"


def _render_introspection_constants(kb: KinBody) -> str:
    """Emit BASE_LINK / EE_LINK / DOF / T_HOME as public module constants.

    These give downstream callers a way to see -- programmatically -- which
    kinematic frame the artifact's ``T_target`` is expressed in, without
    reading the source. ``T_HOME`` is the forward kinematics at
    ``q = np.zeros(DOF)``: a 4x4 SE(3) pose a user can compare against
    their robot's documented home pose to verify the baked geometry
    matches their hardware.
    """
    from ssik.kinematics.poe_fk import poe_forward_kinematics

    base_link = kb.links[0].name
    ee_link = kb.links[-1].name
    dof = len(kb.joints)
    t_home = poe_forward_kinematics(kb, np.zeros(dof, dtype=np.float64))
    lines: list[str] = []
    lines.append(f'BASE_LINK = "{base_link}"\n')
    lines.append(f'EE_LINK = "{ee_link}"\n')
    lines.append(f"DOF = {dof}\n")
    lines.append(
        "# Home pose: FK at q = np.zeros(DOF). Sanity-check this against\n"
        "# your robot's documented home pose to verify the baked geometry\n"
        "# matches your URDF.\n"
    )
    lines.append(f"T_HOME = np.array({t_home.tolist()!r}, dtype=np.float64)\n")
    return "".join(lines)


def _render_kinbody_constants(kb: KinBody) -> str:
    """Emit the joint axes / T_left / T_right matrices as numpy literals."""
    lines: list[str] = []
    lines.append("# --- baked KinBody constants ---\n")
    lines.append(f"_LINK_NAMES = {[link.name for link in kb.links]!r}\n")
    lines.append("_JOINT_NAMES = [")
    for j in kb.joints:
        lines.append(f"    {j.name!r},")
    lines.append("]\n")
    lines.append("_JOINT_AXES = [")
    for j in kb.joints:
        lines.append(f"    np.array({j.axis.tolist()!r}, dtype=np.float64),")
    lines.append("]\n")
    lines.append("_JOINT_T_LEFTS = [")
    for j in kb.joints:
        lines.append(f"    np.array({j.T_left.tolist()!r}, dtype=np.float64),")
    lines.append("]\n")
    lines.append("_JOINT_T_RIGHTS = [")
    for j in kb.joints:
        lines.append(f"    np.array({j.T_right.tolist()!r}, dtype=np.float64),")
    lines.append("]\n")
    lines.append("_JOINT_TYPES = [")
    for j in kb.joints:
        lines.append(f"    {j.joint_type!r},")
    lines.append("]\n")
    lines.append("_JOINT_LIMITS = [")
    for j in kb.joints:
        lines.append(f"    {j.limits!r},")
    lines.append("]\n")
    return "\n".join(lines)


def _render_kinbody_builder() -> str:
    """Emit the function that reconstructs the baked :class:`KinBody`."""
    return textwrap.dedent(
        """\

        def _build_kb() -> KinBody:
            \"\"\"Reconstruct the baked KinBody. Run once at module import.\"\"\"
            links = [Link(name=n) for n in _LINK_NAMES]
            joints = [
                Joint(
                    name=_JOINT_NAMES[i],
                    dof_index=i,
                    parent_link=links[i],
                    T_left=_JOINT_T_LEFTS[i],
                    T_right=_JOINT_T_RIGHTS[i],
                    axis=_JOINT_AXES[i],
                    joint_type=_JOINT_TYPES[i],
                    limits=_JOINT_LIMITS[i],
                )
                for i in range(len(_JOINT_NAMES))
            ]
            return KinBody(links=links, joints=joints)


        _KB = _build_kb()
        """
    )


def _render_solve_function(solver_short: str) -> str:
    """Emit the public ``solve`` callable wrapping the chosen ssik solver."""
    return textwrap.dedent(
        f"""\

        def solve(
            T_target,
            *,
            max_solutions=None,
            q_seed=None,
            respect_limits: bool = True,
            allow_refinement: bool = False,
            allow_rescue: bool = True,
            policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
            refinement_max_iters: int = 15,
            seed_metric: str = "wrap_linf",
            seed_tolerance: float | None = None,
        ):
            \"\"\"Inverse kinematics. Returns ``list[Solution]``.

            :param T_target: 4x4 SE(3) target end-effector pose, np.float64.
            :param max_solutions: optional cap on returned IKs (post-dedup,
                post-limits filter). ``None`` = full enumeration.
            :param q_seed: optional joint config. When provided, solutions
                are sorted by distance from ``q_seed`` (closest first, via
                ``seed_metric``). Combine with ``max_solutions=1`` for the
                trajectory-tracking idiom.
            :param seed_metric: distance used to rank against ``q_seed``.
                ``"wrap_linf"`` (default, largest single-joint move) holds
                the branch during tracking; ``"wrap_l2"`` uses the summed
                move. Ignored when ``q_seed`` is ``None``.
            :param seed_tolerance: optional max per-joint deviation from
                ``q_seed`` (radians, wrap-to-pi). When set, only solutions with
                *every* joint within ``seed_tolerance`` are returned -- a hard
                tracking guarantee that may return an empty list when no branch
                qualifies. ``None`` (default) keeps the best-effort behaviour.
                Requires ``q_seed``.
            :param respect_limits: when ``True`` (default), solutions
                outside URDF joint limits are dropped. ``False`` returns
                the raw geometric set.
            :param allow_refinement: when ``True`` (default), Newton polish
                fires on near-miss algebraic candidates. Tightens FK
                closure to machine precision.
            :param allow_rescue: when ``True`` (default), if the analytical
                path returns no solutions but the target is within the arm's
                reach-sphere, ``solve()`` recovers the IK via the
                T-perturbation rescue (#319) -- reachable-but-degenerate poses
                (near-singular / near-parallel-axis) return LM-polished
                solutions tagged ``refinement_used="lm"`` instead of ``[]``.
                Set ``False`` for a guaranteed-analytical-or-empty result.
                Gated by the reach-sphere, so far-field unreachable targets
                stay cheap (no rescue fired).
            :param policy: tolerance policy. Rarely customised.
            :param refinement_max_iters: cap on Newton iterations per
                candidate when ``allow_refinement=True``.
            :returns: list of :class:`Solution`, one per analytical IK
                branch (plus any rescued at a degenerate pose). Empty list
                iff the target is unreachable or ``allow_rescue=False`` and
                the analytical path found nothing.

            Solver: {solver_short}.
            \"\"\"
            if seed_tolerance is not None and q_seed is None:
                raise ValueError("seed_tolerance requires q_seed")
            # Seeded numerical-tracking fast path (#380): the caller gave a seed
            # and wants a single IK -- the trajectory-tracking idiom. Newton-
            # continue from the seed (~0.2 ms) instead of resolving the whole
            # redundancy (several ms). On a smooth trajectory the continuation is
            # exactly the seed-nearest solution the full solve would return; it
            # is run through the same limit/tolerance postprocess below so its
            # output is indistinguishable from the full path's. When the seed
            # doesn't continue cleanly (Newton jumped a branch, diverged, or the
            # result fails limits/seed_tolerance) ``_seeded_track`` returns
            # ``None`` / the postprocess empties and we fall through to the full
            # analytical solve -- correctness is never traded for speed.
            if q_seed is not None and max_solutions == 1:
                _tracked = _seeded_track(
                    np.asarray(q_seed, dtype=np.float64),
                    fk,
                    lambda _q: _kinbody_jacobian(_KB, _q),
                    np.asarray(T_target, dtype=np.float64),
                )
                if _tracked is not None:
                    _fast = [_tracked]
                    if respect_limits:
                        _fast = _ps_respect_limits(_ps_wrap_to_limits(_fast, _KB), _KB)
                    if _fast and seed_tolerance is not None:
                        _fast = _ps_within_seed_tolerance(_fast, q_seed, seed_tolerance)
                    if _fast:
                        return _fast[:1]
            sols, _is_ls = _solver_solve(
                _KB,
                T_target,
                policy=policy,
                allow_refinement=allow_refinement,
                refinement_max_iters=refinement_max_iters,
            )
            # Bulletproof fallback (#319 / #358): the analytical path found
            # nothing. If the target is within the arm's max reach it may be a
            # measure-zero degenerate pose (near-singular elbow/gimbal, or a
            # near-parallel-axis spherical joint) the algebraic extraction
            # can't resolve -- rather than an unreachable target. Recover via
            # the T-perturbation rescue. The reach-sphere (sum of link lengths;
            # an exact upper bound by the triangle inequality, so it never
            # rejects a reachable pose) is checked only in this rare empty
            # branch and keeps far-field targets cheap. Perturbed re-solves run
            # with allow_rescue=False (recursion guard + analytical escape
            # hatch); the rescue calls back with respect_limits=False, so the
            # rescued set flows through the same limit/seed postprocess below.
            if not sols and allow_rescue:
                _reach_radius = sum(
                    float(np.linalg.norm(np.asarray(_t)[:3, 3]))
                    for _t in (*_JOINT_T_LEFTS, *_JOINT_T_RIGHTS)
                )
                _T = np.asarray(T_target, dtype=np.float64)
                if float(np.linalg.norm(_T[:3, 3])) <= _reach_radius:
                    sols = _rescue_via_T_perturbation(
                        fk,
                        _functools.partial(solve, allow_rescue=False),
                        _T,
                        jacobian_fn=lambda _q: _kinbody_jacobian(_KB, _q),
                    )
            if respect_limits:
                sols = _ps_wrap_to_limits(sols, _KB)
                sols = _ps_respect_limits(sols, _KB)
                if not sols:
                    # #359: the blind swivel sweep sampled no in-limits candidate
                    # even though a reachable in-limits solution exists (the
                    # in-limits swivel arc was narrower than the sampling). The
                    # feasible-swivel resolver computes the in-limits arcs exactly
                    # and returns solutions directly (no-op for non-SRS chains).
                    sols = _resolve_in_limits(_KB, T_target, policy=policy)
            if q_seed is not None:
                if seed_tolerance is not None:
                    sols = _ps_within_seed_tolerance(sols, q_seed, seed_tolerance)
                sols = _ps_nearest_to_seed(sols, q_seed, metric=seed_metric)
            if max_solutions is not None and len(sols) > max_solutions:
                sols = sols[:max_solutions]
            return sols
        """
    )


def _render_all_export() -> str:
    return (
        "\n__all__ = ["
        '\n    "BASE_LINK",'
        '\n    "DISPATCH_REASON",'
        '\n    "DOF",'
        '\n    "EE_LINK",'
        '\n    "EXPECTED_MS_MEDIAN",'
        '\n    "FLOP_BUDGET",'
        '\n    "SOLVER_NAME",'
        '\n    "SOLVER_TIER",'
        '\n    "T_HOME",'
        '\n    "fk",'
        '\n    "solve",'
        "\n]\n"
    )


def _render_fk_alias() -> str:
    """Specialised templates already define ``_fk``; expose it publicly
    so callers can do ``arm_ik.fk(q)`` instead of the private mangled
    name."""
    return "\nfk = _fk\n"


def _render_fk_function_from_kb() -> str:
    """For thin-wrapper templates that don't bake FK themselves
    (``seven_r.srs``, ``seven_r.srs_polished``): build a public
    ``fk(q)`` from the baked ``_KB`` using runtime POE FK."""
    return textwrap.dedent(
        """\

        from ssik.kinematics.poe_fk import poe_forward_kinematics as _poe_fk


        def fk(q):
            \"\"\"Forward kinematics: returns the 4x4 base->ee pose at ``q``.\"\"\"
            return _poe_fk(_KB, np.asarray(q, dtype=np.float64))
        """
    )
