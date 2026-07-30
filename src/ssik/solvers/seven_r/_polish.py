"""Shared LM-polish tail for the approximate/polished 7R solver family (#467).

``srs_polished``, ``spherical_shoulder_polished``, and the approximate-SRS
in-limits resolver (``_swivel_limits.resolve_in_limits``) all turn a batch of
candidate seeds into FK-verified, deduplicated :class:`Solution` objects the
same way: build the FK/Jacobian closures, batch-Newton-polish every seed, keep
the rows that FK-close (and, when limits are given, land in-limits *after* the
polish), then cluster-merge. This is that tail, factored out once.

The completeness invariant lives here: seeds are polished first and limit-
filtered second. A near-singular seed that starts a fraction of a radian
out-of-limits but polishes inside is kept -- never discarded pre-polish (#462).
Callers own only seed *generation* (their arm-specific redundancy sweep).

This sits one layer below :func:`ssik.postprocess.finalize_solutions` (which
takes already-computed Solutions through limits -> seed-rank -> truncate).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.refinement import (
    dedup_by_wrap_close,
    kinbody_fk_jacobian_batch,
    kinbody_jacobian,
    lm_refine_batch,
)

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from ssik._kinbody import KinBody

__all__ = ["polish_candidates"]

_LIMIT_SLACK = 1e-9  # shared by every caller's in-limits acceptance test


def _within_limits(q: NDArray[np.float64], limits: list[tuple[float, float]]) -> bool:
    return all(limits[i][0] - _LIMIT_SLACK <= q[i] <= limits[i][1] + _LIMIT_SLACK for i in range(7))


def polish_candidates(
    kb: KinBody,
    seeds: NDArray[np.float64] | Sequence[NDArray[np.float64]],
    t_target: NDArray[np.float64],
    *,
    accept_fk_atol: float,
    dedup_tol: float,
    lm_fk_atol: float = 1e-9,
    lm_max_iters: int = 30,
    limits: list[tuple[float, float]] | None = None,
    batched: bool = False,
    solution_factory: Callable[[int, NDArray[np.float64], float], Solution] | None = None,
    max_solutions: int | None = None,
) -> list[Solution]:
    """LM-polish a batch of candidate seeds into FK-verified, deduped Solutions.

    :param kb: POE-normalized :class:`KinBody`.
    :param seeds: ``(N, dof)`` array (or sequence of ``(dof,)`` vectors) of raw
        candidate joint vectors from the caller's redundancy sweep. Empty in ->
        empty out.
    :param t_target: ``(4, 4)`` target pose.
    :param accept_fk_atol: keep a polished candidate iff its FK residual is
        ``<= accept_fk_atol``.
    :param dedup_tol: cluster-merge gate (``policy.subproblem_dedup``).
    :param lm_fk_atol: convergence threshold passed to :func:`lm_refine_batch`
        (per-seed early stop); distinct from ``accept_fk_atol``, the acceptance
        gate applied afterwards.
    :param lm_max_iters: per-candidate Newton iteration cap.
    :param limits: when given, drop any candidate that is out-of-limits *after*
        the polish (never before). ``None`` skips the limit filter.
    :param batched: use the vectorised :func:`kinbody_fk_jacobian_batch` inside
        the polish (revolute-only; ~3-4x faster). ``False`` uses the scalar
        per-candidate callables.
    :param solution_factory: ``(i, q, res) -> Solution`` builder; defaults to a
        fresh ``Solution(q, res, refinement_used="lm")``. Callers that carry
        per-candidate metadata (e.g. ``srs_polished``'s branch fields) pass a
        ``replace(raw[i], ...)`` factory.
    :param max_solutions: optional cap applied after dedup.
    :returns: FK-verified, in-limits (if requested), cluster-merged solutions.
    """
    q_seeds = np.asarray(seeds, dtype=np.float64)
    if q_seeds.ndim != 2 or q_seeds.shape[0] == 0:
        return []

    def _fk(q: NDArray[np.float64]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = poe_forward_kinematics(kb, q)
        return out

    def _jac(q: NDArray[np.float64]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = kinbody_jacobian(kb, q)
        return out

    batch_fn: (
        Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]] | None
    )
    if batched:

        def batch_fn(q: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return kinbody_fk_jacobian_batch(kb, q)
    else:
        batch_fn = None

    q_pol, res, _iters = lm_refine_batch(
        q_seeds,
        _fk,
        _jac,
        t_target,
        fk_atol=lm_fk_atol,
        max_iters=lm_max_iters,
        fk_jac_batch_fn=batch_fn,
    )

    factory = solution_factory or (
        lambda _i, q, r: Solution(q=q, fk_residual=r, refinement_used="lm")
    )
    out: list[Solution] = []
    for i in range(q_pol.shape[0]):
        if res[i] > accept_fk_atol:
            continue
        if limits is not None and not _within_limits(q_pol[i], limits):
            continue
        out.append(factory(i, q_pol[i], float(res[i])))

    out = dedup_by_wrap_close(out, dedup_tol)
    return out[:max_solutions] if max_solutions is not None else out
