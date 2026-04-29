"""Per-arm artifact emission. Renders a self-contained Python module that
wraps the dispatched solver around baked KinBody constants.

The emitted artifact is a single ``.py`` file with a stable public API:

    >>> import ur5_ik
    >>> solutions, is_ls = ur5_ik.solve(T_target)

Internals: the emitted module imports the chosen ssik solver, reconstructs
the POE-normalised :class:`KinBody` from baked numpy literals at import
time, and exports ``solve(T) -> (list[Solution], bool)`` plus
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
    "ikgeo.gen_six_dof": "ssik.solvers.ikgeo.gen_six_dof",
}


def _render(*, kb: KinBody, plan: DispatchPlan, module_name: str, arm_label: str) -> str:
    """Render the artifact source as a single string."""
    solver_module = _SOLVER_IMPORT_PATHS[plan.solver_name]
    solver_short = plan.solver_name.split(".")[-1]

    buf = StringIO()
    buf.write(_render_header(module_name, arm_label, plan))
    buf.write("\n\nfrom __future__ import annotations\n\n")
    buf.write("import numpy as np\n\n")
    buf.write("from ssik._kinbody import Joint, KinBody, Link\n")
    buf.write("from ssik.core.solution import Solution\n")
    buf.write("from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy\n")
    buf.write(f"from {solver_module} import solve as _solver_solve\n\n")
    buf.write(f'SOLVER_NAME = "{plan.solver_name}"\n')
    buf.write(f"SOLVER_TIER = {plan.tier}\n")
    buf.write(f"EXPECTED_MS_MEDIAN = {plan.expected_ms_median!r}\n")
    buf.write(f"FLOP_BUDGET = {plan.flop_budget}\n")
    buf.write(_render_dispatch_reason(plan.reason))
    buf.write("\n")
    buf.write(_render_kinbody_constants(kb))
    buf.write("\n")
    buf.write(_render_kinbody_builder())
    buf.write("\n")
    buf.write(_render_solve_function(solver_short))
    buf.write("\n")
    buf.write(_render_all_export())
    return buf.getvalue()


def _render_header(module_name: str, arm_label: str, plan: DispatchPlan) -> str:
    """Top-of-file docstring with provenance + usage."""
    return textwrap.dedent(
        f'''\
        """Generated IK module for {arm_label}.

        This file was emitted by ``ssik build`` and is the public artifact for
        running analytical inverse kinematics on this specific arm. The
        per-arm KinBody constants are baked in below; you do not need to
        load a URDF or MJCF at runtime.

        Solver: ``{plan.solver_name}`` (tier {plan.tier})
        Expected median IK time: ~{plan.expected_ms_median} ms on commodity
        single-thread hardware. FLOP budget: {plan.flop_budget:,} per solve.

        Usage:

            import {module_name}
            import numpy as np
            T_target = np.eye(4)  # 4x4 SE(3) pose
            T_target[:3, 3] = [0.5, 0.1, 0.3]
            solutions, is_ls = {module_name}.solve(T_target)
            for sol in solutions:
                print(sol.q, sol.fk_residual)

        ``solve(T)`` returns ``(list[Solution], is_ls)``. ``is_ls=True``
        signals that no solution closed within the solver's FK tolerance,
        and the returned list is the best-LS approximation (or empty).
        """'''
    )


def _render_dispatch_reason(reason: str) -> str:
    """Bake the dispatcher's explanatory diagnostic as a module-level constant."""
    # ``repr`` keeps newlines and quoting safe inside the rendered source.
    return f"DISPATCH_REASON = {reason!r}\n"


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
            policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
            allow_refinement: bool = False,
            refinement_max_iters: int = 15,
        ):
            \"\"\"Inverse kinematics. Returns ``(list[Solution], is_ls)``.

            :param T_target: 4x4 SE(3) target end-effector pose, np.float64.
            :param policy: tolerance policy. Pass a custom
                :class:`ssik.TolerancePolicy` to tighten or relax the
                FK-closure threshold (``subproblem_numerical``), the
                axis-parallel / axis-intersect predicates, etc. Defaults to
                :data:`ssik.DEFAULT_TOLERANCE_POLICY`.
            :param allow_refinement: opt into Newton-on-spatial-Jacobian
                polish for near-miss algebraic candidates. Default ``False``;
                turn on to recover candidates that don't quite meet
                ``policy.subproblem_numerical`` on their own (e.g. near
                kinematic singularities).
            :param refinement_max_iters: cap on Newton iterations per
                candidate when ``allow_refinement=True``.
            :returns: ``(solutions, is_ls)``. Each ``solution.q`` is a joint
                vector matching the source URDF's joint ordering;
                ``solution.fk_residual`` reports closure against
                ``T_target``. ``is_ls=True`` iff the algebraic path produced
                no candidate meeting the FK tolerance -- callers wanting
                only "exact" solutions check ``is_ls`` and discard.

            Solver: {solver_short}.
            \"\"\"
            return _solver_solve(
                _KB,
                T_target,
                policy=policy,
                allow_refinement=allow_refinement,
                refinement_max_iters=refinement_max_iters,
            )
        """
    )


def _render_all_export() -> str:
    return (
        "\n__all__ = ["
        '\n    "DISPATCH_REASON",'
        '\n    "EXPECTED_MS_MEDIAN",'
        '\n    "FLOP_BUDGET",'
        '\n    "SOLVER_NAME",'
        '\n    "SOLVER_TIER",'
        '\n    "solve",'
        "\n]\n"
    )
