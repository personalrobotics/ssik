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

    import ssik
    from ssik.postprocess import (
        respect_limits, wrap_to_limits, nearest_to_seed, take_first,
    )

    arm = ssik.Manipulator.from_prebuilt("panda")
    kb = arm.kinbody

    sols = arm.solve(T_target, respect_limits=False)
    sols = wrap_to_limits(sols, kb)
    sols = respect_limits(sols, kb)
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
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.refinement import kinbody_jacobian

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


# A value within this band of a joint limit is *at* the limit (#624): it is
# accepted and reported as exactly the limit, on either side of it. Without the
# band the limit test decides on round-off: at a configuration exactly on a
# limit (a robot parked at a hard stop) the two backends compute the angle an
# ulp or two apart, one just inside and one just outside, and so return
# different sets.
#
# This is the floor of the band: every solution gets at least this much, and an
# accurate one gets no more (the error band below widens it only for a solution
# whose own error is larger, #651). The band clamps, which moves a
# configuration, so the floor is round-off, like the +-pi snap (_CUT_SNAP): the
# error of a well-conditioned closed-form angle. Measured at poses with one to
# three joints exactly at a limit (150 per arm, every shipped 6R arm, both
# backends), the closed-form families (spherical_two_parallel, three_parallel)
# land within 2.5e-13 of the limit where sigma_min(J) >= 1e-2 and within
# 1.2e-10 where it is >= 1e-4; 1e-9 covers both. general_6r's eigen-solve alone
# is accurate only to its FK gate (angle errors to 1e-5 even at regular poses),
# but its accepted solutions are polished to machine precision
# (ssik.refinement.polish, #636), after which it lands within 1.3e-12 where
# sigma_min >= 1e-2 and 9.3e-11 where it is >= 1e-4.
#
# 1e-9 is the seeded ranking's grid (_RANK_QUANTUM), the +-pi snap (_CUT_SNAP)
# and the 7R in-limits resolvers' acceptance slack (seven_r._polish and
# seven_r.spherical_shoulder, which take this constant), so every "within
# round-off" in the pipeline means the same thing.
# A clamp moves the tool by at most the band x reach per joint, and the moved
# solution's fk_residual is re-measured (_placed, #645). Through _reps it
# also decides which representatives of pi a limit admits, so a limit written
# a few ulps inside pi (Puma 560's +-3.14159265358979) sits on the cut for
# _canonicalize_representatives, like an exact [-pi, pi].
_LIMIT_BAND = 1e-9  # C++ kLimitBand


# The error band (#651). Whether a value is *at* a joint limit, or at the +-pi
# cut, is decided against that solution's own angular error. Its estimate is
#
#     band = min(_BAND_CAP, _BAND_GAIN * r / sigma_min)
#
# where r = ||FK(q) - T_target||_F is the solution's residual (the solvers'
# metric, re-measured here on kb's chain) and sigma_min is the smallest singular
# value of the spatial Jacobian at q (ssik.refinement.kinbody_jacobian). To first
# order an error dq leaves a residual twist J dq, whose norm is at most r (the
# Frobenius norm of a rotation error is sqrt(2) times its angle), so
# ||dq|| <= r / sigma_min bounds every joint's error. A value outside a limit by
# at most max(_LIMIT_BAND, band) is at the limit and is set to it (_onto_limits;
# a value inside moves only within the _LIMIT_BAND round-off), and a finite
# joint's value is at the cut when within max(_CUT_FLOOR, band) of it (see
# _CUT_SNAP). So only a value that would otherwise be dropped or lie outside its
# limits moves, and by at most its own error. One fixed band cannot serve both
# kinds of solution this separates:
#
# - An accurate solution at a regular pose has r ~ 1e-15 and sigma_min >= 1e-2,
#   so its band is far below round-off, and a value 3e-7 from pi stays where the
#   solver put it. A fixed 1e-6 cut band moved such values: the wrist-flipped
#   branch of a Puma 560 pose whose wrist is 3e-7 from 0 lost 4e-7 in FK, and
#   RS007N's shoulder-flipped branch 1.4e-8 (#649's fuzz examples).
# - At a fold or a singularity an angle is a multiple root, known only to about
#   sqrt(round-off), and sigma_min is tiny, so the band widens. There a value at
#   an exact limit comes back 1e-9 to 3e-7 on either side of it (a CRX-10iA/L
#   elbow winding 3.4e-8 past its stop, an M1013 wrist winding 1.3e-9 past), and
#   the fixed 1e-9 limit band made seeded solves jump branches (#644, #632).
#
# _BAND_GAIN: the first-order bound understates the error near a multiple root,
# where the linearisation is poor (from a double root's neighbourhood a Newton
# step goes only half way). Measured against the actual error of the returned
# solution nearest a known root (the release pose set: 40 uniform, 20 near-limit
# and 20 near-singular poses per 6R arm, both backends, 7,199 solutions), the
# error is at most 8.7 x r / sigma_min, and at most 2.1 x for errors below 1e-9
# (down to 1e-16: the estimate tracks round-off too); 10 covers both. An accurate
# solution's r / sigma_min is below 1e-13 at the Puma 560 / RS007N poses above,
# so the gain costs it nothing.
#
# _BAND_CAP bounds the move a clamp or a snap may make: 1e-6 =
# DEFAULT_TOLERANCE_POLICY.subproblem_numerical / 10, so a moved solution stays
# well inside the FK gate (it moves the tool by at most 1e-6 x reach), three
# orders below subproblem_dedup (1e-3), so it cannot confuse distinct branches,
# and above the measured fold uncertainty at the cut (3.6e-7, UR elbows). A
# value beyond the cap is not moved, however uncertain: the only in-limit copy
# of such a configuration is a different point of a singular family (an LR Mate
# 200iD wrist 4.6e-5 along its self-motion), which a single-joint clamp cannot
# reach without breaking FK.
#
# The band is computed only for a solution with a value between the floor and
# _BAND_CAP from a limit or the cut, where it can change the decision, which a
# uniform pose almost never has, so the default path pays nothing. Without a
# target (standalone use of the filters) only the floors apply. C++: kBandCap,
# kBandGain, Remeasure::error_band.
_BAND_CAP = 1e-6  # DEFAULT_TOLERANCE_POLICY.subproblem_numerical / 10
_BAND_GAIN = 10.0


def _error_band(kb: KinBody, q: NDArray[np.float64], T_target: NDArray[np.float64]) -> float:
    """The angular error estimate of the solution at ``q``, capped:
    ``min(_BAND_CAP, _BAND_GAIN * r / sigma_min)`` (see :data:`_BAND_GAIN`).
    Each test adds its own floor."""
    r = float(np.linalg.norm(poe_forward_kinematics(kb, q) - T_target))
    s = float(np.linalg.svd(kinbody_jacobian(kb, q), compute_uv=False)[-1])
    if s <= 0.0:  # exactly singular: nothing bounds the error
        return _BAND_CAP
    return min(_BAND_CAP, _BAND_GAIN * r / s)


class _Band:
    """One solution's error band, evaluated on first use: most solutions never
    need it. Measured at the solution as it entered the current pass."""

    __slots__ = ("_kb", "_q", "_target", "_value")

    def __init__(self, kb: KinBody, q: NDArray[np.float64], T_target: NDArray[np.float64]):
        self._kb = kb
        self._q = q
        self._target = T_target
        self._value: float | None = None

    def __call__(self) -> float:
        if self._value is None:
            self._value = _error_band(self._kb, np.asarray(self._q, dtype=np.float64), self._target)
        return self._value


def _band_of(sol: Solution, kb: KinBody, T_target: NDArray[np.float64] | None) -> _Band | None:
    return None if T_target is None else _Band(kb, sol.q, T_target)


def _onto_limits(v: float, lo: float, hi: float, band: _Band | None = None) -> float | None:
    """``v`` with the limit band applied: ``None`` when ``v`` lies outside the
    limits by more than ``max(_LIMIT_BAND, band())`` (#651; the floor alone
    without ``band``), else the limit itself when ``v`` is outside it or within
    the :data:`_LIMIT_BAND` round-off of it (#624), else ``v``. So a value
    inside its limits moves by round-off at most, however large its error, and
    one outside moves by at most its band. ``band`` is evaluated only for a
    value outside a limit by more than the floor and at most the cap, the one
    range where it can change the answer."""
    if v < lo:
        return lo if _within_band(lo - v, band) else None
    if v > hi:
        return hi if _within_band(v - hi, band) else None
    if v - lo <= _LIMIT_BAND:
        return lo
    if hi - v <= _LIMIT_BAND:
        return hi
    return v


def _within_band(out: float, band: _Band | None) -> bool:
    """Whether a value ``out`` past a limit is within ``max(_LIMIT_BAND,
    band())`` of it."""
    if out <= _LIMIT_BAND:
        return True
    return band is not None and out <= _BAND_CAP and out <= band()


_TWO_PI = 2.0 * np.pi


def _is_move(old: float, new: float) -> bool:
    """Whether ``new`` is anything other than ``old`` shifted by a whole number
    of turns (#645). A ``2*pi`` shift is the same configuration; a snap or a
    clamp is not. Exact comparison against the same ``old + 2*pi*k`` expression
    the pipeline computes a shift with, so a plain shift is never a move and any
    snap or clamp that changed a bit is."""
    return new != old + _TWO_PI * round((new - old) / _TWO_PI)


def _placed(
    sol: Solution,
    q_new: NDArray[np.float64],
    moved: bool,
    kb: KinBody,
    T_target: NDArray[np.float64] | None,
) -> Solution:
    """``sol`` at ``q_new``. When a snap or a clamp ``moved`` it and the target
    is known, ``fk_residual`` is re-measured at ``q_new`` (#645): it must
    describe the configuration returned, not the one the solver measured. The
    solvers' metric, ``||FK(q) - T_target||_F`` on the user's chain."""
    res = sol.fk_residual
    if moved and T_target is not None:
        res = float(np.linalg.norm(poe_forward_kinematics(kb, q_new) - T_target))
    # Constructed directly: dataclasses.replace costs microseconds, and a UR
    # solve places 256 winding lifts.
    return Solution(q_new, res, sol.refinement_used)


def respect_limits(
    sols: list[Solution], kb: KinBody, *, T_target: NDArray[np.float64] | None = None
) -> list[Solution]:
    """Drop solutions where any joint's q value is outside its reachable range.

    Joints with ``limits=None`` are unconstrained (continuous joints, or
    fixtures that don't supply limits) and never reject a solution. Joints
    with ``limits=(lo, hi)`` reject any solution where ``q[i]`` is outside
    ``[lo, hi]`` by more than the limit band: round-off (``1e-9`` rad), widened
    to the solution's own angular error estimate when ``T_target`` is given
    (see ``docs/api.md#joint-limits``). A value within that band of a limit, on
    either side, is at the limit: the solution is kept, with that joint set to
    exactly the limit.

    :param sols: candidate solutions (e.g. output of an ssik solver's
        ``solve()``).
    :param kb: the same :class:`KinBody` used for the IK call. Joint limits
        come from ``kb.joints[i].limits``.
    :param T_target: the IK target. When given, the band is the solution's
        error band (#651), and a solution the clamp moved has its
        ``fk_residual`` re-measured at the returned ``q``; without it the band
        is round-off and the solver's residual is kept.
    :returns: filtered solutions; preserves input order. Every returned value
        lies within its limits exactly.
    """
    n_joints = len(kb.joints)
    kept: list[Solution] = []
    for sol in sols:
        if len(sol.q) != n_joints:
            raise ValueError(f"solution q-length {len(sol.q)} doesn't match kb DOF {n_joints}")
        q_new: NDArray[np.float64] | None = None
        within = True
        band = _band_of(sol, kb, T_target)
        for i, joint in enumerate(kb.joints):
            if joint.limits is None:
                continue
            lo, hi = joint.limits
            q_i = float(sol.q[i])
            v = _onto_limits(q_i, lo, hi, band)
            if v is None:
                within = False
                break
            if v != q_i:
                if q_new is None:
                    q_new = np.asarray(sol.q, dtype=np.float64).copy()
                q_new[i] = v
        if within:
            # Any change here is a clamp: a move.
            kept.append(sol if q_new is None else _placed(sol, q_new, True, kb, T_target))
    return kept


def wrap_to_limits(
    sols: list[Solution], kb: KinBody, *, T_target: NDArray[np.float64] | None = None
) -> list[Solution]:
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
    biasing toward the original value. A wrap within the limit band of a limit,
    on either side, lands exactly on the limit, as in :func:`respect_limits`.

    :param sols: candidate solutions.
    :param kb: the same :class:`KinBody` used for the IK call.
    :param T_target: the IK target. When given, the band is the solution's
        error band (#651), and a solution the band moved has its
        ``fk_residual`` re-measured at the returned ``q``; a ``2*pi`` wrap
        alone keeps it.
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
        moved = False
        band = _band_of(sol, kb, T_target)
        for i, joint in enumerate(kb.joints):
            if joint.limits is None or joint.joint_type != "revolute":
                continue
            lo, hi = joint.limits
            q_i = float(q_new[i])
            # Smallest |k| first; a fit within the band of a limit lands on it.
            for k in (0, 1, -1, 2, -2):
                cand = q_i + 2.0 * np.pi * k
                v = _onto_limits(cand, lo, hi, band)
                if v is not None:
                    q_new[i] = v
                    moved = moved or v != cand
                    break
        out.append(_placed(sol, q_new, moved, kb, T_target))
    return out


def _wrap_to_pi(angle: float) -> float:
    """Wrap a single angle to the canonical ``[-pi, pi]`` representative."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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


def _reps(q_i: float, lo: float, hi: float, band: _Band | None = None) -> list[float]:
    """Every ``q_i + 2*pi*k`` inside ``[lo, hi]``, ascending.

    The ``k`` range comes from the limits (so a boundary value like ``0`` under
    ``[-2*pi, 2*pi]`` yields ``{-2*pi, 0, 2*pi}``), then each candidate goes
    through the limit band (:func:`_onto_limits`, with the solution's ``band``),
    so floating-point error in the ceil/floor can never emit an out-of-limit
    value, and a representative within the band of a limit is that limit.
    Without the band a value an ulp either side of ``0`` would lift to
    ``{-2*pi, 0}`` on one backend and ``{0, 2*pi}`` on the other.
    """
    # math.ceil/floor on plain floats, not np.ceil/np.floor: this runs once per
    # joint per solution on the seeded path, and the numpy scalar round trip
    # dominated it.
    k_lo = math.ceil((lo - q_i) / _TWO_PI) - 1
    k_hi = math.floor((hi - q_i) / _TWO_PI) + 1
    reps = []
    for k in range(k_lo, k_hi + 1):
        v = _onto_limits(q_i + _TWO_PI * k, lo, hi, band)
        if v is not None:
            reps.append(v)
    # A value with no in-limit representative keeps its own, so expansion can
    # only ever add configurations. Expansion runs after the limit filter, so
    # this is unreachable in the wired paths; it is here so that a mistake in
    # that wiring could never silently delete solutions.
    return reps or [q_i]


def count_windings(
    sols: list[Solution], kb: KinBody, *, T_target: NDArray[np.float64] | None = None
) -> int:
    """How many configurations :func:`expand_windings` would produce, without
    building them. Used for diagnostics so a truncated or pruned solve can still
    report the true size of the complete in-limit set. Pass the same
    ``T_target`` as to :func:`expand_windings`: it sets the limit band.
    """
    wind = winding_joints(kb)
    if not wind:
        return len(sols)
    total = 0
    for sol in sols:
        n = 1
        band = _band_of(sol, kb, T_target)
        for i, lo, hi in wind:
            n *= len(_reps(float(sol.q[i]), lo, hi, band))
        total += n
    return total


def expand_windings(
    sols: list[Solution],
    kb: KinBody,
    *,
    limit: int | None = None,
    T_target: NDArray[np.float64] | None = None,
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
    :param T_target: the IK target. When given, the limit band is the
        solution's error band (#651), and a representative the band moved onto
        a limit has its ``fk_residual`` re-measured.
    """
    wind = winding_joints(kb)
    if not wind:
        return list(sols)
    idxs = [i for i, _, _ in wind]
    out: list[Solution] = []
    for sol in sols:
        q = np.asarray(sol.q, dtype=np.float64)
        band = _band_of(sol, kb, T_target)
        opts = [_reps(float(q[i]), lo, hi, band) for i, lo, hi in wind]
        # Per winding joint, the representatives the limit band moved (#645):
        # nearly always none, and then no lift needs checking.
        moves = [
            {v for v in reps if _is_move(float(q[i]), v)}
            for (i, _, _), reps in zip(wind, opts, strict=True)
        ]
        any_move = any(moves)
        res, ref = sol.fk_residual, sol.refinement_used
        for combo in itertools.product(*opts):
            q_new = q.copy()
            for t, v in enumerate(combo):
                q_new[idxs[t]] = v
            if any_move and any(v in moves[t] for t, v in enumerate(combo)):
                out.append(_placed(sol, q_new, True, kb, T_target))
            else:
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
    T_target: NDArray[np.float64] | None = None,
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
    :param T_target: the IK target. When given, the limit band is the
        solution's error band (#651), and a solution whose chosen
        representative the band moved onto a limit has its ``fk_residual``
        re-measured.
    """
    seed = np.asarray(q_seed, dtype=np.float64)
    out: list[Solution] = []
    for sol in sols:
        q_new = np.asarray(sol.q, dtype=np.float64).copy()
        moved = False
        band = _band_of(sol, kb, T_target)
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
            cands = _reps(q_i, lo, hi, band)
            q_new[i] = v = min(cands, key=lambda c: (abs(c - s_i), c))
            moved = moved or _is_move(q_i, v)
        out.append(_placed(sol, q_new, moved, kb, T_target))
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


# Two tolerances decide which coordinate an angle at the +-pi cut gets (#596).
# The solvers pick a side of the cut ad hoc (per atan2 site, per backend, and
# theta_offset moves the cut), so at an exact pi the two backends can return a
# value just either side of it. They differ in what they are allowed to cost.
#
# _CUT_SNAP moves a continuous joint's value onto exactly pi, which moves the
# configuration, so it is held to round-off: the spread between the backends'
# computations of one well-conditioned angle (<= 1e-12 measured at exact-pi
# poses). It is the grid the seeded ranking already uses to make equal things
# compare equal across backends, and a snap this size moves the tool by at most
# 1e-9 x reach. A wider fixed snap is not free: a redundant arm samples its
# self-motion manifold densely enough that accurate values land within 1e-6 of
# pi on ordinary poses (xArm 7 joint 7, a few per pose), and snapping those
# broke FK by ~1e-6. Continuous joints need only this snap: their parity is
# compared on the circle, where values either side of the cut are already the
# same configuration.
#
# A finite joint whose limits admit two or more representatives of pi takes the
# error band (_BAND_GAIN, #651): its value is at the cut when within
# max(_CUT_FLOOR, band) of the pi class. The band is wide only where pi is a
# fold or a singularity: an elbow folded back on itself (UR joint 3 at pi) is a
# double root of its subproblem, so round-off r in its cosine becomes an angle
# error of sqrt(2 r), measured up to 3.6e-7 over 150 exact-fold poses per UR
# arm, and still up to 9e-8 on the Python path after the general_6r polish
# (UR7e; 4e-8 on Thor). The band covers those (a fold's sigma_min is tiny), and
# the UR3e exact-fold parity depends on it. Every other such joint comes back
# within 1e-11 of an exact pi away from a singularity (sigma_min(J) >= 1e-4),
# which the band covers too: it bounds the measured error down to round-off.
#
# _CUT_FLOOR only guards a residual that rounds to zero, where the band would be
# zero while the backends can still differ by an ulp of pi (4.4e-16): 1e-14 is a
# few dozen ulps. It is not the limit floor (1e-9, _LIMIT_BAND), because the cut
# snap runs in every respect_limits mode, including the raw set, and there a
# 1e-9 snap of an accurate value costs up to 1e-9 x reach in FK: more than the
# tightest arm's precision (Puma 560, 1e-12; the uniform fuzz found values 8e-11
# from pi). With limits respected, a value within 1e-9 of a limit on the cut is
# clamped onto it by the limit pass anyway.
#
# Where no limit sits on pi (UR's [-2*pi, 2*pi]), the choice is between
# representatives 2*pi apart, which costs nothing. Where one does (exactly
# [-pi, pi]), the shift to the +pi side can land just beyond the limit (a value
# just above -pi), by at most the band; it is then set to the limit, which moves
# the tool by at most the band x reach, and the moved solution's fk_residual is
# re-measured (#645). A value the shift leaves inside keeps its bits: a fixed
# band set every value within 1e-6 of pi onto it, which moved accurate values
# (and, once the band follows the error, would move an uncertain in-limit value
# by up to 1e-6 for no gain). The seeded choice still sees both ends of the
# range: _reps admits the far end's representative, which lies outside the
# limit by less than the band, as that limit.
_CUT_SNAP = _RANK_QUANTUM  # C++ kCutSnap
_CUT_FLOOR = 1e-14  # C++ kCutFloor


def _pi_class_offset(q_i: float) -> float:
    """Distance from ``q_i`` to the nearest odd multiple of ``pi``."""
    return abs(abs(_wrap_to_pi(q_i)) - math.pi)


def _at_cut(offset: float, band: _Band | None) -> bool:
    """Whether a finite joint's value ``offset`` from the pi class is at the cut:
    within ``max(_CUT_FLOOR, band())`` of it, or the floor alone without
    ``band``. ``band`` is evaluated only between the floor and the cap."""
    if offset <= _CUT_FLOOR:
        return True
    if band is None or offset > _BAND_CAP:
        return False
    return offset <= band()


def _canonicalize_representatives(
    sols: list[Solution], kb: KinBody, T_target: NDArray[np.float64] | None = None
) -> list[Solution]:
    """Pick one deterministic coordinate for every angle that has a choice (#596).

    - **continuous** revolute joints (``limits is None``) are wrapped to
      ``(-pi, pi]``, and a value within :data:`_CUT_SNAP` of the cut is set to
      exactly ``+pi``;
    - a **finite** revolute joint whose limits admit two or more
      representatives of ``pi`` (for example exactly ``[-pi, pi]``, or UR's
      ``[-2*pi, 2*pi]``) takes, when its value is at the cut (within
      ``max(_CUT_FLOOR, band)`` of the ``pi`` class, :func:`_at_cut`), the
      ``2*pi`` shift nearest the in-limit representative closest to ``+pi``;
      if that shift lies beyond a limit sitting on the representative, it is
      set to the limit. Every other finite value is left alone for
      :func:`wrap_to_limits` and winding enumeration, as before;
    - prismatic joints are untouched.

    Values away from the cut keep their exact bits. A coordinate moves only by a
    multiple of ``2*pi``, except for the snap onto ``pi`` of a continuous joint
    (round-off) and the clamp onto a limit at the cut (a value the shift took
    beyond it, by at most its error band). A move re-measures ``fk_residual``
    when ``T_target`` is given (#645).
    Idempotent, so the repeated finalize passes of one solve agree.
    """
    # (joint index, in-limit representative of pi nearest +pi or None for a
    # continuous joint, the joint's limits)
    targets: list[tuple[int, float | None, float, float]] = []
    for i, joint in enumerate(kb.joints):
        if joint.joint_type != "revolute":
            continue
        if joint.limits is None:
            targets.append((i, None, -math.inf, math.inf))
            continue
        lo, hi = joint.limits
        reps = [v for v in _reps(math.pi, lo, hi) if lo <= v <= hi]
        if len(reps) >= 2:
            targets.append((i, min(reps, key=lambda v: abs(v - math.pi)), lo, hi))
    if not targets:
        return sols
    out: list[Solution] = []
    for sol in sols:
        q_new: NDArray[np.float64] | None = None
        moved = False
        band = _band_of(sol, kb, T_target)
        for i, pi_rep, lo, hi in targets:
            q_i = float(sol.q[i])
            if pi_rep is None:
                if -math.pi + _CUT_SNAP < q_i < math.pi - _CUT_SNAP:
                    continue
                if _pi_class_offset(q_i) <= _CUT_SNAP:
                    w = math.pi
                    moved = moved or _is_move(q_i, w)
                else:
                    w = _wrap_to_pi(q_i)
            elif _at_cut(_pi_class_offset(q_i), band):
                w = q_i + _TWO_PI * round((pi_rep - q_i) / _TWO_PI)
                # Only where a limit sits on the representative can the shift
                # land beyond one, and then by at most the band: the value is at
                # that limit. A value the shift leaves inside keeps its bits.
                if w > hi or w < lo:
                    w = hi if w > hi else lo
                    moved = moved or _is_move(q_i, w)
            else:
                continue
            if w != q_i:
                if q_new is None:
                    q_new = np.asarray(sol.q, dtype=np.float64).copy()
                q_new[i] = w
        out.append(sol if q_new is None else _placed(sol, q_new, moved, kb, T_target))
    return out


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

    A solution passes when *every* joint is within ``tolerance`` of the seed --
    ``max_i |d_i| <= tolerance``, the L-infinity (max-joint-move) bound. This is
    the hard guarantee for trajectory tracking ("no joint jumps more than
    ``tolerance``"), as opposed to :func:`nearest_to_seed`, which only ranks. The
    result may be empty when no in-tolerance branch exists -- itself the useful
    signal that smooth continuation is not possible at this pose. Compose with
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
    T_target: NDArray[np.float64] | None = None,
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
        best = rewrap_to_seed(sols, kb, seed, T_target=T_target)
        return nearest_to_seed(best, seed, metric=metric, kb=kb)[:k]

    wind = winding_joints(kb)
    circular = _circular_mask(kb, len(seed))
    idxs = [i for i, _, _ in wind]
    m = len(wind)
    picked: list[Solution] = []

    for sol in sols:
        q = np.asarray(sol.q, dtype=np.float64)
        base = _seed_deltas(q, seed, circular)
        band = _band_of(sol, kb, T_target)
        # Per winding joint: the in-limit representatives ordered by distance to
        # the seed, so rank 0 is the nearest and each step out costs more.
        ladders = [
            sorted((abs(v - float(seed[i])), v) for v in _reps(float(q[i]), lo, hi, band))
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
            moved = False
            for t, r in enumerate(ranks):
                q_new[idxs[t]] = v = ladders[t][r][1]
                moved = moved or _is_move(float(q[idxs[t]]), v)
            picked.append(_placed(sol, q_new, moved, kb, T_target))
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
    respect_limits: bool | Literal["wrap"] = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    in_limits_fallback: Callable[[], list[Solution]] | None = None,
    counts: dict[str, int] | None = None,
    enumerate_windings: bool = False,
    T_target: NDArray[np.float64] | None = None,
) -> list[Solution]:
    """The shared IK post-processing pipeline: limits -> seed -> truncate.

    This is the one definition of the tail that ``Manipulator.solve`` and every
    emitted artifact ``solve()`` used to hand-duplicate (four copies, already
    drifted). Order matters and is fixed: :func:`_canonicalize_representatives`
    (one coordinate per angle at the +-pi cut, every mode), then
    ``wrap_to_limits`` (try +/-2pi to bring
    branches into range) then ``respect_limits`` (drop the rest); then, if a seed
    is given, ``within_seed_tolerance`` (hard bound) then ``nearest_to_seed``
    (rank); then truncate to ``max_solutions``.

    :param respect_limits: when ``True`` apply the wrap+drop limit pass;
        ``"wrap"`` applies only the wrap (each joint's ``q +- 2*pi*k``
        representative inside its range where one exists) and drops nothing,
        so the full geometric set comes back in canonical representatives --
        the raw ``False`` set is unwrapped, and a limit-margin score on it
        reads an in-range branch as a violation; ``False`` leaves the set as
        the solver produced it, apart from the canonical representative.
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
    :param T_target: the IK target. Every ``solve()`` passes it, so that a
        solution this pipeline moves -- a snap onto ``pi`` or a clamp onto a
        limit, not a ``2*pi`` shift -- has its ``fk_residual`` re-measured at the
        returned ``q`` (#645): ``||FK(q) - T_target||_F`` on ``kb``'s chain, the
        solvers' own metric. One FK per moved solution; an unmoved solution keeps
        the solver's value. ``None`` keeps the solver's residuals throughout.
    :returns: the post-processed solution list.
    """
    # One representative per angle before anything else looks at the values
    # (#596): the solvers choose a side of the +-pi cut ad hoc, so without this
    # the two backends disagree by 2*pi on the same configuration, and on a
    # [-pi, pi] joint even the seeded ranking diverges.
    if T_target is not None:
        T_target = np.asarray(T_target, dtype=np.float64)
    sols = _canonicalize_representatives(sols, kb, T_target)
    if respect_limits == "wrap":
        sols = wrap_to_limits(sols, kb, T_target=T_target)
        if counts is not None:
            counts["dropped_by_limits"] = 0
    elif respect_limits:
        sols = wrap_to_limits(sols, kb, T_target=T_target)
        pre_limit = len(sols)
        sols = _apply_limits(sols, kb, T_target=T_target)
        if counts is not None:
            counts["dropped_by_limits"] = pre_limit - len(sols)
        if not sols and in_limits_fallback is not None:
            # The resolvers accept a candidate within _LIMIT_BAND of a limit
            # (seven_r._polish, spherical_shoulder); put it on the limit, as the
            # pass above would.
            sols = wrap_to_limits(in_limits_fallback(), kb, T_target=T_target)

    # Whether this call is the one that lifts is the caller's decision (see the
    # parameter docs), not something inferred from respect_limits: an artifact
    # runs this pipeline several times and its final ranking pass passes
    # respect_limits=False because the limit filter already happened.
    expanding = enumerate_windings and bool(winding_joints(kb))
    # The size of the complete set, computed arithmetically so it stays truthful
    # when a prefix cap or top-k pruning means the set is never materialized.
    available = count_windings(sols, kb, T_target=T_target) if expanding else len(sols)
    if counts is not None:
        counts["geometric_branches"] = len(sols)
        counts["winding_representatives"] = available

    if q_seed is None:
        if expanding:
            # Unseeded output keeps expansion order, so a cap is a prefix and the
            # discarded representatives need never be built.
            sols = expand_windings(sols, kb, limit=max_solutions, T_target=T_target)
    else:
        # #562 step 1: a seeded solve returns the representative nearest the seed
        # rather than the principal value, so it never commands a gratuitous 2*pi
        # turn. Under enumeration the finite joints are covered by the expansion
        # itself (collapsing them here would destroy the set), leaving only the
        # continuous joints, whose lift family is infinite.
        sols = rewrap_to_seed(sols, kb, q_seed, continuous_only=expanding, T_target=T_target)
        if expanding:
            if seed_tolerance is None and max_solutions is not None:
                # Ranked truncation: take the globally nearest max_solutions
                # directly. Equivalent to expanding, ranking and truncating.
                sols = _windings_topk(sols, kb, q_seed, seed_metric, max_solutions, T_target)
            else:
                sols = expand_windings(sols, kb, T_target=T_target)
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
