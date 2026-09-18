"""Composable post-processing filters for IK solutions.

ssik's analytical IK kernel returns the full geometric solution set --
every branch the math admits, regardless of joint limits, distance to a
preferred configuration, or any other application-level concern. This
module provides the building blocks the application layer composes on top.

Convention follows IKFast (OpenRAVE wrapper handles limits + collision +
nearest-to-seed in a separate Python layer above the generated kernel) and
EAIK (``mj_manipulator/franka.py`` wraps EAIK's pure IK with limit /
seed / collision logic on the Python side).

Each function takes ``list[Solution]`` (and any extra args) and returns
``list[Solution]`` -- pure transforms, easy to test, easy to compose.
These cover ~95% of real-world post-processing:

  * :func:`respect_limits` -- drop solutions outside any joint's range
  * :func:`wrap_to_limits` -- try ``q ± 2*pi`` per joint to bring solutions in
  * :func:`nearest_to_seed` -- sort by wrap-to-pi distance to a reference q
  * :func:`within_seed_tolerance` -- drop solutions beyond a per-joint
    deviation from a reference q (hard tracking bound; may return empty)
  * :func:`take_first` -- truncate to the first ``k``

Production pipeline pattern::

    from franka_panda_ik import _KB, solve
    from ssik.postprocess import (
        respect_limits, wrap_to_limits, nearest_to_seed, take_first,
    )

    sols = solve(T_target, respect_limits=False)
    sols = wrap_to_limits(sols, _KB)
    sols = respect_limits(sols, _KB)
    sols = nearest_to_seed(sols, q_current)
    sols = take_first(sols, k=4)

Out of scope for v0.1 (separate issues / future work):

  * Collision filtering (needs a collision backend such as FCL).
  * Trajectory-context filters (continuous q-trajectory with smoothness).
  * Reachability / dexterity scoring.

These can be follow-ups; the filters above cover the common case and form a
self-contained module that compiles cleanly to a single shared ``.so``
in Phase 4 (no per-arm specialisation, no symbolic precompute).
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution

__all__ = [
    "count_windings",
    "expand_windings",
    "finalize_solutions",
    "nearest_to_seed",
    "respect_limits",
    "rewrap_to_seed",
    "take_first",
    "winding_joints",
    "within_seed_tolerance",
    "wrap_to_limits",
]


def respect_limits(sols: list[Solution], kb: KinBody) -> list[Solution]:
    """Drop solutions where any joint's q value is outside its reachable range.

    Joints with ``limits=None`` are unconstrained (continuous joints, or
    fixtures that don't supply limits) and never reject a solution. Joints
    with ``limits=(lo, hi)`` reject any solution where ``q[i] < lo`` or
    ``q[i] > hi`` strictly; values exactly on the boundary are accepted.

    :param sols: candidate solutions (e.g. output of an ssik solver's
        ``solve()``).
    :param kb: the same :class:`KinBody` used for the IK call. Joint limits
        come from ``kb.joints[i].limits``.
    :returns: filtered solutions; preserves input order.
    """
    n_joints = len(kb.joints)
    kept: list[Solution] = []
    for sol in sols:
        if len(sol.q) != n_joints:
            raise ValueError(f"solution q-length {len(sol.q)} doesn't match kb DOF {n_joints}")
        within = True
        for i, joint in enumerate(kb.joints):
            if joint.limits is None:
                continue
            lo, hi = joint.limits
            if sol.q[i] < lo or sol.q[i] > hi:
                within = False
                break
        if within:
            kept.append(sol)
    return kept


def wrap_to_limits(sols: list[Solution], kb: KinBody) -> list[Solution]:
    """Try wrapping each joint's q value by ``±2*pi`` integer multiples to
    bring it into the joint's reachable range.

    A revolute joint at ``q = 3.0`` with limits ``[-pi, pi]`` is FK-equivalent
    to ``q - 2*pi ≈ -3.28``, which is *also* outside the range, so neither
    wrap fits and the solution stays at ``q = 3.0`` (and would be dropped by
    :func:`respect_limits` if called next). A joint at ``q = 4.0`` with
    limits ``[-pi, pi]`` wraps to ``q - 2*pi ≈ -2.28`` which is in range:
    we keep the wrapped value.

    Joints with ``limits=None`` are left unchanged (no constraint to wrap
    into). Prismatic joints are left unchanged (no rotational periodicity).

    Search is over ``k ∈ {-2, -1, 0, +1, +2}`` integer multiples of ``2*pi``;
    that covers any joint whose limits span up to ±5*pi (more than enough for
    any commercial arm). The smallest-|k| wrap that lands in range wins,
    biasing toward the original value.

    :param sols: candidate solutions.
    :param kb: the same :class:`KinBody` used for the IK call.
    :returns: solutions with each q-vector adjusted joint-wise; preserves
        input order; returns ``Solution`` instances with the wrapped q
        and other fields unchanged.
    """
    n_joints = len(kb.joints)
    out: list[Solution] = []
    for sol in sols:
        if len(sol.q) != n_joints:
            raise ValueError(f"solution q-length {len(sol.q)} doesn't match kb DOF {n_joints}")
        q_new = np.asarray(sol.q, dtype=np.float64).copy()
        for i, joint in enumerate(kb.joints):
            if joint.limits is None or joint.joint_type != "revolute":
                continue
            lo, hi = joint.limits
            q_i = float(q_new[i])
            if lo <= q_i <= hi:
                continue
            # Try wraps with smallest |k| first.
            best = q_i
            best_in_range = False
            for k in (1, -1, 2, -2):
                candidate = q_i + 2.0 * np.pi * k
                if lo <= candidate <= hi:
                    best = candidate
                    best_in_range = True
                    break
            if best_in_range:
                q_new[i] = best
        out.append(replace(sol, q=q_new))
    return out


def _wrap_to_pi(angle: float) -> float:
    """Wrap a single angle to the canonical ``[-pi, pi]`` representative."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


_TWO_PI = 2.0 * np.pi

# (aggregate, per-joint deviations sorted descending): the part of the seeded
# ordering that rises monotonically with every per-joint deviation, so it can
# drive a best-first walk over a branch's winding lattice.
_LatticeKey = tuple[float, tuple[float, ...]]


# A joint is enumerable only when its limits span *strictly more* than one full
# turn. The tolerance keeps a joint whose span is 2*pi to within round-off (the
# Kassow arms) out of enumeration: its span admits a second representative only
# at the exact boundary, which is the same physical configuration, not a lift.
_SPAN_TOL = 1e-9


def winding_joints(kb: KinBody) -> list[tuple[int, float, float]]:
    """The joints that admit more than one in-limit winding (#562).

    A revolute joint with finite limits spanning more than ``2*pi`` (UR-family
    ``[-2*pi, 2*pi]``) has several distinct
    joint-coordinate representatives of the *same* geometric branch. Those are
    finite-limit lifts, not new IK branches, but they are distinct admissible
    configurations with different distances and feasible motions.

    Excluded, by construction: prismatic joints (no rotational periodicity),
    continuous joints (``limits is None`` -- the lift family is infinite and no
    representative is privileged), and finite joints spanning at most ``2*pi``
    (at most one representative, except at an exact boundary).

    :returns: ``[(joint_index, lo, hi)]``, in joint order.
    """
    out: list[tuple[int, float, float]] = []
    for i, joint in enumerate(kb.joints):
        if joint.joint_type != "revolute" or joint.limits is None:
            continue
        lo, hi = joint.limits
        if hi - lo > _TWO_PI + _SPAN_TOL:
            out.append((i, float(lo), float(hi)))
    return out


def _reps(q_i: float, lo: float, hi: float) -> list[float]:
    """Every ``q_i + 2*pi*k`` inside ``[lo, hi]``, ascending.

    The ``k`` range comes from the limits (so a boundary value like ``0`` under
    ``[-2*pi, 2*pi]`` yields ``{-2*pi, 0, 2*pi}``), then each candidate is
    re-checked against the limits so floating-point error in the ceil/floor can
    never emit an out-of-limit value.
    """
    # math.ceil/floor on plain floats, not np.ceil/np.floor: this runs once per
    # joint per solution on the seeded path, and the numpy scalar round trip
    # dominated it.
    k_lo = math.ceil((lo - q_i) / _TWO_PI) - 1
    k_hi = math.floor((hi - q_i) / _TWO_PI) + 1
    reps = [v for k in range(k_lo, k_hi + 1) if lo <= (v := q_i + _TWO_PI * k) <= hi]
    # A value with no in-limit representative keeps its own, so expansion can
    # only ever add configurations. Expansion runs after the limit filter, so
    # this is unreachable in the wired paths; it is here so that a mistake in
    # that wiring could never silently delete solutions.
    return reps or [q_i]


def count_windings(sols: list[Solution], kb: KinBody) -> int:
    """How many configurations :func:`expand_windings` would produce, without
    building them. Used for diagnostics so a truncated or pruned solve can still
    report the true size of the complete in-limit set.
    """
    wind = winding_joints(kb)
    if not wind:
        return len(sols)
    total = 0
    for sol in sols:
        n = 1
        for i, lo, hi in wind:
            n *= len(_reps(float(sol.q[i]), lo, hi))
        total += n
    return total


def expand_windings(
    sols: list[Solution], kb: KinBody, *, limit: int | None = None
) -> list[Solution]:
    """Expand each solution into every in-limit winding representative (#562).

    Takes the Cartesian product across :func:`winding_joints`, since each
    combination is a distinct point of the bounded joint-coordinate domain. FK
    is identical for every representative; only the coordinates differ. Arms
    with no wide-limit joint are returned unchanged (26 of the 72 shipped arms),
    so they pay nothing.

    Output order is branch-major, and within a branch the ascending-value
    Cartesian product -- the stable order the ranking and truncation rules are
    defined against.

    :param limit: stop once this many configurations exist. Only sound when the
        caller keeps a *prefix* of the expansion (i.e. unseeded truncation);
        a ranked truncation must not pass it.
    """
    wind = winding_joints(kb)
    if not wind:
        return list(sols)
    idxs = [i for i, _, _ in wind]
    out: list[Solution] = []
    for sol in sols:
        q = np.asarray(sol.q, dtype=np.float64)
        opts = [_reps(float(q[i]), lo, hi) for i, lo, hi in wind]
        res, ref = sol.fk_residual, sol.refinement_used
        for combo in itertools.product(*opts):
            q_new = q.copy()
            for t, v in enumerate(combo):
                q_new[idxs[t]] = v
            out.append(Solution(q_new, res, ref))
            if limit is not None and len(out) >= limit:
                return out
    return out


def rewrap_to_seed(
    sols: list[Solution],
    kb: KinBody,
    q_seed: NDArray[np.float64],
    *,
    continuous_only: bool = False,
) -> list[Solution]:
    """Rewrap each revolute joint to the ``q_i + 2*pi*k`` representative nearest
    the seed, staying within the joint's finite limits (#562, step 1).

    A wide-limit joint (limits spanning > 2*pi, e.g. UR ``[-2*pi, 2*pi]``) admits
    several in-limit windings of one geometric branch; a seeded solve should
    return the one nearest ``q_seed`` rather than the principal value, so
    ``solve(T, q_seed=q_current, max_solutions=1)`` never commands a gratuitous
    2*pi turn. Per joint:

    - **finite limits, value in range**: among all ``q_i + 2*pi*k`` in
      ``[lo, hi]``, choose the one nearest ``seed_i`` (ties -> smaller value,
      deterministic). Representatives are derived from the limits, so a
      boundary principal like ``0`` with ``[-2*pi, 2*pi]`` correctly considers
      ``{-2*pi, 0, 2*pi}``. An admissible value never leaves its limits.
    - **finite limits, value out of range**: only reachable via
      ``respect_limits=False``, where the caller wants the raw geometric set.
      The nearest turn to ``seed_i``, unclamped. Forcing such a value *into*
      limits used to return a representative a full turn from the seed --
      exactly the motion this function exists to prevent (shipped in v5.2.0).
    - **continuous** (``limits is None``): the nearest turn to ``seed_i``
      (``k = round((seed_i - q_i) / 2*pi)``), no clamp.
    - **prismatic**: unchanged (no rotational periodicity).

    The choice is made per value rather than from a caller flag, because the
    flag would have to mean "the user wanted limits", which is not what the
    pipeline's own ``respect_limits`` says by the time the ranking pass runs.

    Only the returned coordinate changes; FK is identical. Runs only when a seed
    is supplied (the unseeded path pays nothing).

    :param continuous_only: restrict rewrapping to continuous joints. Set when
        winding enumeration is active: enumeration already emits every in-limit
        representative of a finite joint, so collapsing them onto the
        seed-nearest one would destroy the very set being returned. Continuous
        joints are never enumerated (infinite family), so they still need the
        nearest-turn choice.
    """
    seed = np.asarray(q_seed, dtype=np.float64)
    out: list[Solution] = []
    for sol in sols:
        q_new = np.asarray(sol.q, dtype=np.float64).copy()
        for i, joint in enumerate(kb.joints):
            if joint.joint_type != "revolute":
                continue
            q_i, s_i = float(q_new[i]), float(seed[i])
            if joint.limits is None:
                q_new[i] = q_i + _TWO_PI * round((s_i - q_i) / _TWO_PI)
                continue
            lo, hi = joint.limits
            if not (lo <= q_i <= hi):
                # Out of limits already (respect_limits=False): take the nearest
                # turn without dragging it into limits the caller waived.
                q_new[i] = q_i + _TWO_PI * round((s_i - q_i) / _TWO_PI)
                continue
            if continuous_only:  # enumeration emits this joint's representatives
                continue
            cands = _reps(q_i, lo, hi)
            q_new[i] = min(cands, key=lambda c: (abs(c - s_i), c))
        out.append(replace(sol, q=q_new))
    return out


def _circular_mask(kb: KinBody | None, n: int) -> list[bool]:
    """Per joint: should the seed difference be measured modulo ``2*pi``?

    Only a *continuous* revolute joint may be: it can always take the short way
    round, so ``q`` and ``q + 2*pi`` are the same point of its configuration
    space. A finite revolute joint's configuration space is an interval, not a
    circle: it cannot rotate through a limit, so a joint at ``+3`` really is
    ``6`` radians from a seed at ``-3`` even though the two wrap to within
    ``0.28``. Measuring it modulo ``2*pi`` both understates real motion and
    makes distinct windings tie, which would make the #562 ordering meaningless.
    Prismatic joints have no periodicity at all.

    ``kb is None`` keeps the pre-#562 behaviour (wrap everything) for callers
    using these filters standalone without a :class:`KinBody`.
    """
    if kb is None:
        return [True] * n
    return [j.joint_type == "revolute" and j.limits is None for j in kb.joints]


def _seed_deltas(
    q: NDArray[np.float64], seed: NDArray[np.float64], circular: list[bool]
) -> list[float]:
    """Per-joint signed seed difference, wrapped only where ``circular``."""
    return [
        _wrap_to_pi(float(q[i] - seed[i])) if circular[i] else float(q[i] - seed[i])
        for i in range(len(seed))
    ]


def _aggregate(deltas: list[float], metric: str) -> float:
    if metric == "wrap_l2":
        return float(np.sqrt(sum(d * d for d in deltas)))
    return float(max(abs(d) for d in deltas))  # wrap_linf


# Seeded ordering compares on this grid rather than on raw doubles. Exact ties
# are the norm once windings are enumerated -- under wrap_linf one dominant joint
# fixes the max for every winding of a branch -- and the tie-break then turns on
# the leading element, which the two backends compute to within about 1e-15 of
# each other. Comparing raw doubles let that noise decide the order. A grid far
# below any meaningful joint difference (1e-9 rad is a nanoradian) makes equal
# things compare equal on both backends. Distinct solutions are never this close.
_RANK_QUANTUM = 1e-9


def _snap(x: float) -> float:
    return float(round(x / _RANK_QUANTUM))


def _rank_key(
    deltas: list[float], metric: str, q: NDArray[np.float64]
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """The total, deterministic seeded ordering (#562).

    Ranking on the aggregate alone leaves ties that are neither rare nor
    harmless once windings are enumerated: under ``wrap_linf`` one distant joint
    fixes the max, so every winding of a branch scores identically and the
    "nearest" one is decided by accident of enumeration order. So the aggregate
    is refined by the per-joint deviations sorted descending, compared
    lexicographically -- the leximax refinement. Among configurations whose
    worst joint moves the same, it prefers the one whose next-worst joint moves
    less, which is what a caller asking for the nearest configuration means.
    The joint vector itself is the final tie-break, so the order is total and
    depends only on the solutions, never on the order they arrived in. That
    makes it reproducible across the Python and native backends, which do not
    generate candidates in the same order.
    """
    return (
        _snap(_aggregate(deltas, metric)),
        tuple(_snap(d) for d in sorted((abs(d) for d in deltas), reverse=True)),
        tuple(_snap(float(v)) for v in q),
    )


def nearest_to_seed(
    sols: list[Solution],
    q_seed: NDArray[np.float64],
    *,
    metric: str = "wrap_l2",
    kb: KinBody | None = None,
) -> list[Solution]:
    """Sort solutions by joint-space distance to a reference configuration.

    The "wrap-to-pi" distance treats angle differences modulo ``2*pi``, so
    e.g. ``q=3.0`` and ``q_seed=-3.0`` are at distance
    ``|wrap(3.0 - (-3.0))| = |wrap(6.0)| = |6.0 - 2*pi| ≈ 0.28``, not 6.0.
    This is the right metric for revolute-joint similarity.

    :param sols: candidate solutions.
    :param q_seed: reference joint configuration (length matches the chain's
        DOF).
    :param metric: one of ``"wrap_l2"`` (default; sum-of-squares of
        wrap-to-pi differences) or ``"wrap_linf"`` (max wrap-to-pi
        difference). ``wrap_l2`` is smooth and prefers configurations that
        are uniformly close; ``wrap_linf`` is hard-cap and prefers
        configurations whose worst-joint deviation is small.
    :param kb: when given, differences are wrapped modulo ``2*pi`` only for
        continuous joints; finite revolute and prismatic joints use ordinary
        coordinate distance (see :func:`_circular_mask`). Required for correct
        ranking of winding representatives (#562) -- without it, a branch and
        its ``2*pi`` lift tie. ``None`` wraps every joint (pre-#562 behaviour).
    :returns: solutions sorted by ascending distance to ``q_seed``. Equal
        distances are broken deterministically by :func:`_rank_key`, so the
        order depends only on the solutions themselves.
    """
    if metric not in ("wrap_l2", "wrap_linf"):
        raise ValueError(f"unknown metric {metric!r}; expected 'wrap_l2' or 'wrap_linf'")
    seed = np.asarray(q_seed, dtype=np.float64)
    circular = _circular_mask(kb, len(seed))
    return sorted(sols, key=lambda s: _rank_key(_seed_deltas(s.q, seed, circular), metric, s.q))


def within_seed_tolerance(
    sols: list[Solution],
    q_seed: NDArray[np.float64],
    tolerance: float,
    *,
    kb: KinBody | None = None,
) -> list[Solution]:
    """Keep only solutions within a per-joint deviation of a reference config.

    A solution passes when *every* joint is within ``tolerance`` of the seed in
    wrap-to-pi distance -- ``max_i |wrap(q_i - seed_i)| <= tolerance``, the
    L-infinity (max-joint-move) bound. This is the hard guarantee for
    trajectory tracking ("no joint jumps more than ``tolerance``"), as opposed
    to :func:`nearest_to_seed`, which only ranks. The result may be empty when
    no in-tolerance branch exists -- itself the useful signal that smooth
    continuation is not possible at this pose. Compose with
    :func:`nearest_to_seed` + :func:`take_first` to rank and cap the survivors.

    :param sols: candidate solutions.
    :param q_seed: reference joint configuration (length matches the chain's
        DOF).
    :param tolerance: maximum allowed per-joint deviation, radians.
    :param kb: when given, the bound is measured the way the joint actually
        moves -- modulo ``2*pi`` for continuous joints only, ordinary distance
        for finite revolute and prismatic joints (#562). This makes the
        guarantee honest: a finite joint that must travel 6 radians to reach the
        solution is no longer admitted by a 0.5-radian bound just because the
        endpoints happen to wrap close. ``None`` wraps every joint (pre-#562).
    :returns: the subset of ``sols`` within ``tolerance`` of ``q_seed``, in
        input order.
    """
    seed = np.asarray(q_seed, dtype=np.float64)
    circular = _circular_mask(kb, len(seed))

    def within(sol: Solution) -> bool:
        return all(abs(d) <= tolerance for d in _seed_deltas(sol.q, seed, circular))

    return [s for s in sols if within(s)]


def _windings_topk(
    sols: list[Solution],
    kb: KinBody,
    q_seed: NDArray[np.float64],
    metric: str,
    k: int,
) -> list[Solution]:
    """The globally nearest ``k`` winding representatives, without materializing
    the complete expansion (#562).

    Exactly equivalent to ``expand_windings`` -> ``nearest_to_seed`` ->
    ``take_first(k)``, including tie order, but it never builds the discarded
    configurations. That matters: a UR lifts 8 geometric branches to 256
    configurations, while the tracking idiom asks for one.

    Two facts make the pruning exact. First, a configuration in the global
    top-``k`` is in its own branch's top-``k`` (at most ``k-1`` things precede
    it anywhere, so at most ``k-1`` do within its branch), so per-branch
    top-``k`` then a global merge loses nothing. Second, within a branch the
    per-joint choices are independent and both metrics are non-decreasing in
    every per-joint deviation, so the branch's representatives can be walked in
    ascending distance by best-first search over the product lattice -- pop the
    cheapest rank tuple, push its single-step successors. For ``k = 1`` this
    degenerates to "pick each joint's seed-nearest representative", which is
    exactly :func:`rewrap_to_seed`.
    """
    seed = np.asarray(q_seed, dtype=np.float64)
    if k <= 1:
        # The tracking idiom. Choosing each joint's seed-nearest representative
        # minimises every per-joint deviation at once, so it minimises both the
        # aggregate and the leximax refinement: the branch's best winding is
        # exactly its seed-rewrap, and no lattice search is needed.
        best = rewrap_to_seed(sols, kb, seed)
        return nearest_to_seed(best, seed, metric=metric, kb=kb)[:k]

    wind = winding_joints(kb)
    circular = _circular_mask(kb, len(seed))
    idxs = [i for i, _, _ in wind]
    m = len(wind)
    picked: list[Solution] = []

    for sol in sols:
        q = np.asarray(sol.q, dtype=np.float64)
        base = _seed_deltas(q, seed, circular)
        # Per winding joint: the in-limit representatives ordered by distance to
        # the seed, so rank 0 is the nearest and each step out costs more.
        ladders = [
            sorted((abs(v - float(seed[i])), v) for v in _reps(float(q[i]), lo, hi))
            for i, lo, hi in wind
        ]

        def key(
            ranks: tuple[int, ...],
            base: list[float] = base,
            ladders: list[list[tuple[float, float]]] = ladders,
        ) -> _LatticeKey:
            """The monotone prefix of :func:`_rank_key`: the aggregate, then the
            per-joint deviations sorted descending. Both rise whenever any rank
            rises, which is what makes best-first search able to stop early."""
            deltas = list(base)
            for t, r in enumerate(ranks):
                deltas[idxs[t]] = ladders[t][r][1] - float(seed[idxs[t]])
            return (
                _snap(_aggregate(deltas, metric)),
                tuple(_snap(d) for d in sorted((abs(d) for d in deltas), reverse=True)),
            )

        start = (0,) * m
        heap: list[tuple[_LatticeKey, tuple[int, ...]]] = [(key(start), start)]
        seen = {start}
        taken = 0
        boundary: _LatticeKey | None = None
        while heap:
            # Stop once k are taken and the frontier has moved strictly past the
            # k-th key. Popping through an exact tie keeps the result identical
            # to ranking the complete expansion, whose tie order this cannot see.
            if taken >= k and boundary is not None and heap[0][0] > boundary:
                break
            popped, ranks = heapq.heappop(heap)
            taken += 1
            if taken == k:
                boundary = popped
            q_new = q.copy()
            for t, r in enumerate(ranks):
                q_new[idxs[t]] = ladders[t][r][1]
            picked.append(Solution(q_new, sol.fk_residual, sol.refinement_used))
            for t in range(m):
                nxt = (*ranks[:t], ranks[t] + 1, *ranks[t + 1 :])
                if nxt[t] < len(ladders[t]) and nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (key(nxt), nxt))

    return nearest_to_seed(picked, seed, metric=metric, kb=kb)[:k]


def take_first(sols: list[Solution], k: int) -> list[Solution]:
    """Truncate to the first ``k`` solutions.

    Use after :func:`nearest_to_seed` (or any other ranking) to keep only
    the top-``k`` matches. ``k <= 0`` returns an empty list.

    Renamed from ``max_solutions`` in v1.0 to avoid name collision with
    the ``max_solutions`` kwarg on ``Manipulator.solve`` / artifact
    ``solve()`` -- they have different shapes (kwarg is an int passed in;
    this function takes ``(sols, k)``).

    :param sols: candidate solutions, typically already sorted.
    :param k: maximum number of solutions to keep.
    :returns: ``sols[:max(k, 0)]``.
    """
    return list(sols[: max(k, 0)])


# The ``respect_limits`` function is shadowed by the boolean parameter of the
# same name inside :func:`finalize_solutions`; alias it so the pipeline can still
# call it. (The kwarg name matches the public ``solve()`` API and shouldn't move.)
_apply_limits = respect_limits


def finalize_solutions(
    sols: list[Solution],
    kb: KinBody,
    *,
    respect_limits: bool = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    in_limits_fallback: Callable[[], list[Solution]] | None = None,
    counts: dict[str, int] | None = None,
    enumerate_windings: bool = False,
) -> list[Solution]:
    """The shared IK post-processing pipeline: limits -> seed -> truncate.

    This is the one definition of the tail that ``Manipulator.solve`` and every
    emitted artifact ``solve()`` used to hand-duplicate (four copies, already
    drifted). Order matters and is fixed: ``wrap_to_limits`` (try +/-2pi to bring
    branches into range) then ``respect_limits`` (drop the rest); then, if a seed
    is given, ``within_seed_tolerance`` (hard bound) then ``nearest_to_seed``
    (rank); then truncate to ``max_solutions``.

    :param respect_limits: when ``True`` apply the wrap+drop limit pass.
    :param in_limits_fallback: optional zero-arg callable invoked when the limit
        pass empties the set -- the redundant-7R exact in-limits resolver (#359),
        which recovers a narrow in-limits arc the coarse sweep missed. ``None``
        for callers without one (e.g. ``Manipulator``).
    :param enumerate_windings: when ``True`` (#562), a joint whose
        limits span more than ``2*pi`` contributes every in-limit
        ``q_i + 2*pi*k`` representative, taken as a Cartesian product across
        such joints. These are finite-limit lifts of the same geometric branch,
        not new branches, but they are distinct admissible configurations. Set
        ``False`` for one representative per geometric branch (the pre-6.0
        result set, and the faster path).

        This says "lift in *this* call", which is why it defaults to ``False``
        here while the user-facing default on ``Manipulator.solve`` and the
        artifact ``solve()`` is ``True``: a solve runs this pipeline several
        times (limit pass, rescue pass, then the ranking pass) and the lifts
        must be produced exactly once, by the call that yields the returned
        set. Defaulting off means a pass that forgets to say so cannot
        double-expand. It is likewise the caller's job to pass ``False`` when
        the user asked for the raw geometric set, since the final pass runs
        with ``respect_limits=False`` once the limit filter is behind it.
    :param counts: optional dict; when given, populated with ``dropped_by_limits``
        and ``dropped_by_max_solutions``, plus ``geometric_branches`` and
        ``winding_representatives`` -- reported separately so a caller can tell
        real IK branches from their lifts. ``winding_representatives`` is the
        size of the complete in-limit set even when truncation or top-k pruning
        means it was never built.
    :returns: the post-processed solution list.
    """
    if respect_limits:
        sols = wrap_to_limits(sols, kb)
        pre_limit = len(sols)
        sols = _apply_limits(sols, kb)
        if counts is not None:
            counts["dropped_by_limits"] = pre_limit - len(sols)
        if not sols and in_limits_fallback is not None:
            sols = in_limits_fallback()

    # Whether this call is the one that lifts is the caller's decision (see the
    # parameter docs), not something inferred from respect_limits: an artifact
    # runs this pipeline several times and its final ranking pass passes
    # respect_limits=False because the limit filter already happened.
    expanding = enumerate_windings and bool(winding_joints(kb))
    # The size of the complete set, computed arithmetically so it stays truthful
    # when a prefix cap or top-k pruning means the set is never materialized.
    available = count_windings(sols, kb) if expanding else len(sols)
    if counts is not None:
        counts["geometric_branches"] = len(sols)
        counts["winding_representatives"] = available

    if q_seed is None:
        if expanding:
            # Unseeded output keeps expansion order, so a cap is a prefix and the
            # discarded representatives need never be built.
            sols = expand_windings(sols, kb, limit=max_solutions)
    else:
        # #562 step 1: a seeded solve returns the representative nearest the seed
        # rather than the principal value, so it never commands a gratuitous 2*pi
        # turn. Under enumeration the finite joints are covered by the expansion
        # itself (collapsing them here would destroy the set), leaving only the
        # continuous joints, whose lift family is infinite.
        sols = rewrap_to_seed(sols, kb, q_seed, continuous_only=expanding)
        if expanding:
            if seed_tolerance is None and max_solutions is not None:
                # Ranked truncation: take the globally nearest max_solutions
                # directly. Equivalent to expanding, ranking and truncating.
                sols = _windings_topk(sols, kb, q_seed, seed_metric, max_solutions)
            else:
                sols = expand_windings(sols, kb)
        if seed_tolerance is not None:
            sols = within_seed_tolerance(sols, q_seed, seed_tolerance, kb=kb)
            available = len(sols)
        sols = nearest_to_seed(sols, q_seed, metric=seed_metric, kb=kb)
    if max_solutions is not None:
        # `available` is the size of the complete post-filter set; `sols` may
        # already be shorter than it, because the prefix cap and the top-k prune
        # skip building what the cap would discard.
        if counts is not None and available > max_solutions:
            counts["dropped_by_max_solutions"] = available - max_solutions
        if len(sols) > max_solutions:
            sols = sols[:max_solutions]
    return sols
