"""Standardized solver return type.

Every ssik solver returns ``list[Solution]`` (paired with an ``is_ls`` flag for
the "no candidate survived" case). The :class:`Solution` dataclass carries
both the joint vector AND the diagnostics callers need to reason about
precision and refinement:

- ``fk_residual`` says what FK closure was actually achieved at return time
  -- not the user's tolerance, the actual measurement.
- ``refinement_used`` says whether numerical refinement fired. Closed-form
  solvers (Pieper, three-parallel, etc.) always have ``"none"``. Numeric
  solvers (Raghavan-Roth, Husty-Pfurner) report ``"lm"`` when
  :func:`ssik.refinement.lm_refine` polished the algebraic candidate.
- ``branch_id`` and ``solver_name`` let callers distinguish parallel
  branches and identify which dispatched solver produced the result.

See GitHub #74 (refinement architecture) and #75 (Solution dataclass) for
the design decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["RefinementMode", "Solution"]

RefinementMode = Literal["none", "lm"]


@dataclass(frozen=True)
class Solution:
    """A single IK solution with provenance and precision metadata.

    :param q: Joint-angle vector. Length matches the chain's DOF.
    :param fk_residual: ``||FK(q) - T_target||_F`` at the moment the solver
        returned the candidate. The caller can compare against any target
        tolerance; the solver's own ``fk_atol`` was a *filter*, not a
        contract on the value reported here.
    :param refinement_used: ``"none"`` if the solution came directly from
        the algebraic / closed-form path; ``"lm"`` if
        :func:`ssik.refinement.lm_refine` polished it.
    :param refinement_iters: number of refinement iterations consumed
        (``0`` if ``refinement_used == "none"``).
    :param branch_id: optional IK-branch identifier for solvers that
        enumerate multiple branches per call (e.g. 0..15 for the 16-root
        Raghavan-Roth route, 0..7 for spherical-wrist + parallel-shoulder).
        ``None`` when the solver doesn't expose a stable branch index.
    :param solver_name: dotted module path (``"ikgeo.general_6r"``,
        ``"ikgeo.spherical_two_parallel"``, ...). Useful when results pass
        through a dispatcher that routes across multiple solver candidates.
    """

    q: NDArray[np.float64]
    fk_residual: float
    refinement_used: RefinementMode = "none"
    refinement_iters: int = 0
    branch_id: int | None = None
    solver_name: str = ""
