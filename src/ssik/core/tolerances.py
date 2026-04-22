"""Numeric tolerances used across ssik (predicates + subproblem solvers).

Real-world URDFs produce axes and origins that are *nearly* but not exactly
aligned with canonical kinematic structure -- axes at ``(0, 0, 0.99999998)``
from Xacro arithmetic rounding, rpy matrices whose orthogonality check differs
from identity by 1e-12 after accumulation. The predicates that decide "are
these three axes intersecting?" or "parallel?" need explicit tolerances so
that behaviour is predictable and downstream error messages can reference
named fields ("``subproblem_numerical`` tolerance violated") rather than
opaque magic numbers.

:class:`TolerancePolicy` is the single source for every user-tunable threshold
in the library. Structural-predicate fields (axis_parallel, axis_intersect)
sit alongside subproblem-solver fields (feasibility, numerical filtering,
degeneracy detection) so a caller can swap one policy object through the
entire pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_TOLERANCE_POLICY", "TolerancePolicy"]


@dataclass(frozen=True)
class TolerancePolicy:
    """Numeric tolerances for kinematic predicates and subproblem solvers.

    Structural-predicate fields are consumed by
    :mod:`ssik.kinematics.predicates` and
    :func:`ssik.core.topology.describe_topology`.

    Subproblem-solver fields are consumed by SP1-SP6 under
    :mod:`ssik.subproblems`. Their names capture the *reason* each tolerance
    exists so diagnostic messages can say e.g. ``"SP6 sign branch rejected:
    closure distance 1.2e-4 > subproblem_numerical 1e-5"`` rather than citing
    raw numbers.

    Attributes:
        axis_parallel: cross-product magnitude below which two unit-vector
            axes are considered parallel (or anti-parallel). For unit
            vectors ``||a x b|| = sin(theta)`` so the default ``1e-8``
            accepts axes differing by up to ~1 microradian.
        axis_intersect: perpendicular distance below which two lines in
            3D are considered to intersect. Default matches the
            ``axis_parallel`` tolerance in spirit -- metric-scale chains.
        subproblem_feasibility: threshold on residuals that decide whether
            a subproblem input admits an *exact* solution or only a
            least-squares approximation (``is_ls=True``). Used by SP1's
            ``|p_perp| vs |q_perp|`` check, SP4's ``|rhs| - R`` boundary,
            SP2's sphere mismatch, etc.
        subproblem_numerical: threshold for filtering spurious candidates
            produced by quartic / ellipse root-finding inside SP5 and SP6.
            Unit-circle closure is checked to this tolerance; returned
            solutions that fail are dropped.
        subproblem_degeneracy: rank / collinearity threshold. A QR leading
            coefficient or sin-of-angle-between-axes below this value
            marks the input as degenerate and SP6/aux return
            ``([], is_ls=True)`` rather than produce nonsense.
    """

    axis_parallel: float = 1e-8
    axis_intersect: float = 1e-8
    subproblem_feasibility: float = 1e-9
    subproblem_numerical: float = 1e-5
    subproblem_degeneracy: float = 1e-12


DEFAULT_TOLERANCE_POLICY = TolerancePolicy()
