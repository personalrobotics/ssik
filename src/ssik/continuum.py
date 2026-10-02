"""Singular continua of a 6R solution set: one shared rule for both backends (#662).

At a singular pose a 6R arm can have a one-parameter family of solutions
instead of isolated ones, for example a wrist whose outer axes line up
(``sin q5 = 0``), where only ``q4 + q6`` (or ``q4 - q6``) is fixed. A solver
that treats the family as isolated roots returns arbitrary samples of it:
whichever point its arithmetic lands on, or none at all when a closed form
divides zero by zero. The contract (``docs/api.md#singular-continua``) is

- a **seeded** solve returns the point of the seed's continuum nearest the
  seed, within limits;
- an **unseeded** solve returns one representative per continuum: the point
  whose free joint is at 0, else the in-limit point nearest it along the
  continuum.

This module owns the two halves every backend shares. The native backend
(``cpp/include/ssik_cpp/continuum.hpp``) implements the same definitions; a
change here must be mirrored there.

Detection (the cores)
---------------------

Only a solution a core flags pays anything here; nothing measures the
Jacobian of every solution.

- The closed-form cores (``ikgeo.three_parallel``,
  ``ikgeo.spherical_two_parallel``) measure how close their wrist is to
  locked: the sine between the wrist-roll axis and the point its SP1 rotates,
  ``|k x p| / (|k| |p|)`` (:func:`lock_sine`, the pitched outer wrist axis
  against the roll axis). Within ``RANK_TOL`` they flag the candidate. That
  covers the precision to which each core resolves the wrist pitch at an exact
  lock (a double root: within 5.4e-8 of it through SP4, and within 5.6e-6
  through three_parallel's numerical SP6, over 200 exactly locked poses per
  arm), so both backends flag the same continuum, whichever side of the lock
  their arithmetic lands.
- Both keep their SP1 wrist angles (on a lock they are a point of the
  continuum, whichever side the pitch landed), except within ``LOCK_TOL``,
  where the SP1s divide zero by zero, or where the angles are unusable:
  spherical_two_parallel's miss the target's wrist rotation by more than
  ``RANK_TOL`` (:func:`spherical_wrist_error`: an artifact's expanded SP1
  loses its accuracy near a lock sooner than the native one);
  three_parallel's cannot give a candidate within its 1e-7 FK gate
  (:func:`three_parallel_wrist_misses`: near a lock each SP1 angle alone is
  accurate only to about ``eps / sine``, and the SP6 pitch of an exact lock
  leaves ``sine`` at ~1e-8, #668). There they split the lock.
  three_parallel (:func:`three_parallel_lock`), whose numerical SP6 is too
  coarse there to read the wrist angles or to tell a pose on the lock from one
  just off it, makes ``(q1, q5)`` exact for the lock and sets ``q6`` to the
  seed's value (0 unseeded) and to that plus pi.
- spherical_two_parallel (:func:`spherical_wrist_lock`), which resolves the
  pitch to round-off: on the lock (the target's own wrist
  direction on the axis, within ``AXIS_TOL``), ``q6`` takes the seed's value
  (0 unseeded) and ``q4`` is read off the target; near it, the two branches
  whose pitch leaves the lock on either side.

  A candidate flagged at a pose only near a lock is an isolated solution, or
  closes FK only to the pose's distance from the lock, and the slide leaves it
  where it is (step 5).
- ``ikgeo.general_6r`` has no closed-form subproblem to inspect, and its
  continua are not all at the wrist (a CRX base axis can line up with its
  forearm). It flags an FK-certified candidate whose Jacobian may be rank
  deficient (:func:`rank_deficient`: the Frobenius condition number of the
  normal matrix its accepted-candidate polish solves with anyway). The bound
  has no false negatives; the pass below confirms with an SVD and leaves an
  unconfirmed candidate untouched.

The slide (one pass, after the core and its polish, before finalize)
-------------------------------------------------------------------

For each flagged solution ``x0``, on the chain's FK and spatial Jacobian:

1. Correct (after the first check of step 2): Gauss-Newton steps
   ``dq = J^+ log(T FK(q)^-1)``, the
   pseudo-inverse truncated at ``RANK_TOL sigma_max`` (so a step never moves
   along the continuum), until the residual stops improving.
2. Confirm: some singular value is at most ``RANK_TOL sigma_max`` at the
   solution (checked first, so an unconfirmed flag costs one SVD) and at the
   corrected point, else the solution is not on a continuum and is returned
   unchanged. Those directions span the continuum's null space; its tangent
   ``v`` is the last right singular vector, oriented so that its component on
   the highest-index joint with ``|v_i| >= FREE_MIN`` is positive.
3. Move to the rule's point. Each step is the rule's displacement projected
   onto the null space (projector ``P``), at most ``STEP`` rad long, followed
   by the correction:

   - seeded: ``P wrap(seed - q)``, which stops where the offset to the seed
     (every joint on the circle) is orthogonal to the continuum;
   - unseeded: ``-P e_f wrap(q_f) / P_ff``, the least move that brings the
     free joint ``f`` to 0 (mod 2 pi) to first order. ``f`` is the one the core
     names (``q6`` for the closed-form wrist) or, for a general_6r flag, the
     highest-index joint with ``sqrt(P_ff) >= FREE_MIN``.

   The walk stops when a step falls below ``STOP``, after ``MAX_STEPS``, or
   where the family stops closing FK to ``SLIDE_FK`` (it was only near a
   continuum, or the continuum ends) or the null space vanishes. Seeded, where
   a clipped step leaves ``SLIDE_FK`` the whole step is tried once instead,
   and the walk carries on from it if it closes FK to ``WALK_FK``: a nearly
   aligned wrist (Piper's) has exact solutions along a family that closes FK
   only approximately between them, so a seed that is one of them is reached
   by setting the free joint as the core would, not by walking. A walk off an
   exact continuum therefore costs one or two steps.
4. With ``respect_limits=True``, a point without an in-limit winding of every
   joint (``_LIMIT_BAND``, as the limit filter) is replaced by the nearest
   in-limit point along the continuum: walk both ways along ``v``, ``STEP`` at
   a time for ``ceil(WALK_ARC / STEP)`` steps while it closes FK to
   ``SLIDE_FK``, bisect the first crossing into limits to ``BISECT_STEPS``
   halvings, and keep the side reached in the
   shorter arc (the positive side on a tie). No in-limit point leaves the
   point as it is, for the limit filter to drop.
5. Accept the final point only if it closes FK to ``SLIDE_FK``; otherwise the
   solution is returned exactly as the core produced it. Every point of an
   exact continuum closes FK to round-off, so the rule's point is accepted
   there on both backends. At a pose only near a continuum the family closes
   FK only approximately, and a move along it is accepted only where it lands
   on an exact solution (a seed that is one). A threshold relative to the
   solution's own residual would decide differently on each backend there:
   their samples differ in accuracy (an artifact's to its 1e-5 gate, the
   native core's polished).

The slid solutions are then deduplicated by the same-root rule (#600): two
samples of one continuum slide to the same point and merge.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.postprocess import _LIMIT_BAND
from ssik.refinement import dedup_same_root, same_root_floor, se3_log_residual
from ssik.subproblems import sp1, sp4
from ssik.subproblems._rotation import rotation_matrix

__all__ = [
    "FREE_FROM_KERNEL",
    "LOCK_TOL",
    "NOT_FLAGGED",
    "RANK_TOL",
    "FlaggedSolution",
    "free_of",
    "has_null",
    "lock_sine",
    "rank_deficient",
    "slide_continua",
    "spherical_wrist_error",
    "spherical_wrist_lock",
    "three_parallel_lock",
    "three_parallel_wrist_error",
    "three_parallel_wrist_misses",
    "verify_flagged",
]

# The wrist SP1 that reads a roll angle is degenerate when its point lies within
# this sine of its axis, |k x p| <= LOCK_TOL |k| |p|: there its atan2 divides
# zero by zero and the core splits the lock instead (the free joint set, the
# other read off the target). The rule spherical_shoulder already applies to its
# own wrist (_LOCK_TOL): above it the SP1's angle is accurate to about
# eps / sine, a few 1e-7 rad at worst, below it meaningless. C++ kLockTol.
LOCK_TOL = 1e-9

# A split lock's target direction q (the vector the wrist SP1 rotates toward)
# within this sine of the axis means the pose is on the lock, not only near it.
# q comes from the closed-form shoulder (and, for three_parallel, from q1 made
# exact on the lock), accurate to round-off, while a pose a nudge of 1e-9 off a
# lock (the parity gate's near-limit offsets) leaves it ~1e-9 off. C++ kAxisTol.
AXIS_TOL = 1e-12

# A Jacobian direction with sigma <= RANK_TOL sigma_max is null: the slide's
# confirmation, the truncation of its Gauss-Newton step, the subspace it moves
# in, general_6r's flag (via rank_deficient), and the closed-form cores' flag
# band on the wrist's lock sine (which a pose that far from a lock leaves as
# sigma_min / sigma_max). The cores resolve the wrist pitch at an exact lock to
# 5.6e-6 at worst (three_parallel's numerical SP6, measured over 200 exactly
# locked poses per arm), and some wrists only nearly line up: Piper's at q5 = 0
# leaves sigma_min / sigma_max ~ 7e-6, and its solutions there form a family
# that closes FK to 1e-7 over two radians (#653), which the 1e-5 acceptance
# gate cannot tell from a continuum and the core samples arbitrarily. 1e-4
# covers both. Whatever it admits, the slide moves a solution only to a point
# that closes FK to SLIDE_FK (module docstring, step 5). C++ kRankTol.
RANK_TOL = 1e-4

# The free-joint markers carried with a candidate: not flagged, or flagged with
# the free joint left for the slide to read off the continuum's tangent.
NOT_FLAGGED = -1
FREE_FROM_KERNEL = -2

# The walk (module docstring). C++ mirrors each in continuum.hpp.
STEP = 0.25  # largest predictor step along the tangent, rad (2-norm)
STOP = 1e-12  # a step below this has arrived
MAX_STEPS = 64  # predictor steps toward the rule's point
WALK_ARC = 2.0 * math.pi * math.sqrt(2.0)  # one turn of a wrist continuum, each way
BISECT_STEPS = 40  # halvings of the step that crosses into limits
CORRECT_ITERS = 8  # Gauss-Newton steps per correction
WALK_FK = 1e-6  # a seeded whole step must close FK to this (step 3)
SLIDE_FK = 1e-10  # a slid point closes FK to this (step 5)
FREE_MIN = 0.1  # tangent component a joint needs to serve as free or orient it

_TWO_PI = 2.0 * math.pi


def lock_sine(k: NDArray[np.float64], p: NDArray[np.float64]) -> float:
    """``|k x p| / (|k| |p|)``: how far the wrist SP1 of point ``p`` about axis
    ``k`` is from degenerate. At most :data:`RANK_TOL` the candidate is flagged
    for the slide; at most :data:`LOCK_TOL` the core splits the lock."""
    # Scalar arithmetic: this runs once per wrist branch on every solve.
    k0, k1, k2 = float(k[0]), float(k[1]), float(k[2])
    p0, p1, p2 = float(p[0]), float(p[1]), float(p[2])
    c0, c1, c2 = k1 * p2 - k2 * p1, k2 * p0 - k0 * p2, k0 * p1 - k1 * p0
    return math.sqrt(
        (c0 * c0 + c1 * c1 + c2 * c2)
        / ((k0 * k0 + k1 * k1 + k2 * k2) * (p0 * p0 + p1 * p1 + p2 * p2))
    )


def _one_sided(
    k: NDArray[np.float64],
    a4: NDArray[np.float64],
    p0: NDArray[np.float64],
    q: NDArray[np.float64],
    sign: float,
) -> float:
    """The SP1 angle about ``k`` toward ``q`` of the point ``Rot(a4, t) p0`` as
    ``t -> 0`` from the side ``sign``, where ``p0`` lies on ``k``: there
    ``k x Rot(a4, t) p0 = t (k x (a4 x p0)) + O(t^2)``, and the SP1 angle
    depends only on the direction of ``k x p``."""
    return sp1.angle(k, sign * np.cross(a4, p0), q)


# The free values a locked three_parallel wrist tries when q_free leaves the
# elbow out of reach: q_free + 2 pi k / LOCK_SEARCH, nearest first.
LOCK_SEARCH = 64


def _sp3_feasible(
    k: NDArray[np.float64], p: NDArray[np.float64], q: NDArray[np.float64], d: float
) -> bool:
    """Whether SP3 ``|Rot(k, t) p - q| = d`` has a solution strictly inside its
    range: ``|q . Rot(k, t) p - (|p|^2 + |q|^2 - d^2) / 2|`` reaches 0 with a
    margin of ``LOCK_TOL`` relative to the amplitude ``|k x p| |k x q|``."""
    target = 0.5 * (float(p @ p) + float(q @ q) - d * d)
    axial = float(k @ q) * float(k @ p)
    amp = float(np.linalg.norm(np.cross(k, p)) * np.linalg.norm(np.cross(k, q)))
    return abs(target - axial) <= amp * (1.0 - LOCK_TOL)


def _sp3_miss(
    k: NDArray[np.float64], p: NDArray[np.float64], q: NDArray[np.float64], d: float
) -> float:
    """How far ``d`` lies outside the distances ``|Rot(k, t) p - q|`` reaches
    (0 within them), about the FK position error of an SP3 solved at ``d``."""
    s = float(p @ p) + float(q @ q) - 2.0 * float(k @ q) * float(k @ p)
    amp = 2.0 * float(np.linalg.norm(np.cross(k, p)) * np.linalg.norm(np.cross(k, q)))
    lo, hi = math.sqrt(max(s - amp, 0.0)), math.sqrt(max(s + amp, 0.0))
    return max(lo - d, d - hi, 0.0)


def three_parallel_lock(
    axes: Sequence[ArrayLike],
    r_home: ArrayLike,
    t_target: NDArray[np.float64],
    q1: float,
    q5: float,
    q_free: float,
    offsets: Sequence[ArrayLike],
) -> tuple[float, float, list[tuple[float, float]]]:
    """``(q1, q5, [(theta14, q6), (theta14', q6')])``: the representatives of
    a three_parallel branch whose wrist is within ``RANK_TOL`` of locked
    (``axes[1] || Rot(axes[4], q5) axes[5]``).

    three_parallel reads ``(q1, q5)`` from a numerical SP6, which resolves the
    double root of a lock only to about 1e-5: too coarse to tell a pose on the
    lock from one ``1e-9`` off it, or to read the wrist angles from it, where
    its SP1s fail. So the core represents such a branch the same way on and
    off the lock. ``(q1, q5)`` are first
    made exact for the lock: ``q5`` moves onto it (``Rot(axes[4], q5)
    axes[5]`` onto ``+-axes[1]``), and ``q1`` is solved from SP6's first
    equation with that ``q5``, an SP4, the root nearest SP6's. Then ``q6`` is
    set to ``q_free`` and to ``q_free + pi`` (the wrist-flipped side), and
    ``theta14`` is the angle of ``Rot(axes[0], q1)^T R_06 Rot(axes[5], q6)^T
    Rot(axes[4], q5)^T`` about ``axes[1]``, read off ``axes[4]``. ``theta14``
    moves the elbow's target (``offsets``: the chain's ``p[0..6]``), so where
    ``q6`` leaves the elbow out of reach the next of ``q6 + 2 pi k /
    LOCK_SEARCH`` (``k = 1, -1, 2, ...``) that reaches is taken.

    On the lock these are points of the continuum, exact to round-off, which
    the slide moves to the rule's point. Off it they close FK only to the
    pose's distance from the lock: within the acceptance gate they are kept as
    they are, beyond it the artifact's refinement takes each to its own
    isolated solution (the two sides give the two wrist branches)."""
    q1, q5, out, _ = _three_parallel_lock(axes, r_home, t_target, q1, q5, q_free, offsets)
    return q1, q5, out


def _three_parallel_lock(
    axes: Sequence[ArrayLike],
    r_home: ArrayLike,
    t_target: NDArray[np.float64],
    q1: float,
    q5: float,
    q_free: float,
    offsets: Sequence[ArrayLike],
) -> tuple[float, float, list[tuple[float, float]], bool]:
    """:func:`three_parallel_lock`, and whether some free value it tried
    reaches the elbow."""
    a = [np.asarray(x, dtype=np.float64) for x in axes]
    r_06 = np.asarray(t_target, dtype=np.float64)[:3, :3] @ np.asarray(r_home).T
    p = [np.asarray(x, dtype=np.float64) for x in offsets]
    p_16 = np.asarray(t_target, dtype=np.float64)[:3, 3] - p[0] - r_06 @ p[6]
    a5_mid = rotation_matrix(a[4], q5) @ a[5]
    q5 = q5 + sp1.angle(a[4], a5_mid, math.copysign(1.0, float(a5_mid @ a[1])) * a[1])
    r_45 = rotation_matrix(a[4], q5)
    d1 = float(a[1] @ (p[1] + p[2] + p[3] + p[4])) + float(a[1] @ (r_45 @ p[5]))
    roots, _ = sp4.solve(a[1], -a[0], p_16, d1)
    if roots:
        q1 = min(roots, key=lambda t: abs((t - q1 + math.pi) % _TWO_PI - math.pi))
    r_01 = rotation_matrix(a[0], q1)
    out: list[tuple[float, float]] = []
    reached = False
    for start in (q_free, q_free + math.pi):
        first: tuple[float, float] | None = None
        for k in [0] + [s * j for j in range(1, LOCK_SEARCH // 2 + 1) for s in (1, -1)]:
            v = start + 2.0 * math.pi * k / LOCK_SEARCH
            m = r_01.T @ r_06 @ rotation_matrix(a[5], v).T @ r_45.T
            th = sp1.angle(a[1], a[4], m @ a[4])
            if first is None:
                first = (th, v)
            r_14 = rotation_matrix(a[1], th)
            d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]
            if _sp3_feasible(a[1], -p[3], p[2], float(np.linalg.norm(d_inner))):
                first = (th, v)
                reached = True
                break
        assert first is not None
        out.append(first)
    return q1, q5, out, reached


def spherical_wrist_lock(
    axes: Sequence[ArrayLike],
    r_home: ArrayLike,
    t_target: NDArray[np.float64],
    q1: float,
    q2: float,
    q3: float,
    q5: float,
    q_free: float,
) -> list[tuple[float, float]]:
    """``(q4, q6)`` for a spherical-wrist branch whose wrist SP1s are degenerate
    (``axes[3] || Rot(axes[4], q5) axes[5]``); the two cases of
    :func:`three_parallel_lock`, with ``R_36 = Rot(axes[3], q4) Rot(axes[4],
    q5) Rot(axes[5], q6)`` in place of the trio. ``q = R_36 axes[5]`` depends
    on the closed-form shoulder and elbow only, accurate to round-off. The
    wrist position does not depend on
    ``q4`` and ``q6``, so the free value is always reachable."""
    a = [np.asarray(x, dtype=np.float64) for x in axes]
    r_06 = np.asarray(t_target, dtype=np.float64)[:3, :3] @ np.asarray(r_home).T
    r_36 = (
        rotation_matrix(-a[2], q3) @ rotation_matrix(-a[1], q2) @ rotation_matrix(-a[0], q1) @ r_06
    )
    q_q4 = r_36 @ a[5]
    if lock_sine(a[3], q_q4) <= AXIS_TOL:
        m = r_36 @ rotation_matrix(a[5], q_free).T @ rotation_matrix(a[4], q5).T
        return [(sp1.angle(a[3], a[4], m @ a[4]), q_free)]
    p_q4 = rotation_matrix(a[4], q5) @ a[5]  # on axes[3]
    p_q6 = rotation_matrix(a[4], -q5) @ a[3]  # on axes[5]
    return [
        (
            _one_sided(a[3], a[4], p_q4, q_q4, sign),
            _one_sided(-a[5], -a[4], p_q6, r_36.T @ a[3], sign),
        )
        for sign in (1.0, -1.0)
    ]


@dataclass(frozen=True)
class FlaggedSolution(Solution):
    """A live solver's sample of a singular continuum, carrying its flag to
    :func:`slide_continua` (``ssik.Manipulator.solve``). Internal: the slide
    returns plain :class:`~ssik.core.solution.Solution` objects."""

    free_joint: int = FREE_FROM_KERNEL


def free_of(sols: Sequence[Solution]) -> list[int]:
    """Per solution, its flag (:class:`FlaggedSolution`) or :data:`NOT_FLAGGED`."""
    return [s.free_joint if isinstance(s, FlaggedSolution) else NOT_FLAGGED for s in sols]


def verify_flagged(
    candidates: Sequence[NDArray[np.float64]],
    free: Sequence[int],
    verify: Callable[[list[NDArray[np.float64]], int | None], list[Solution]],
    *,
    fk_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t_target: NDArray[np.float64],
    dedup_atol: float,
    max_solutions: int | None = None,
) -> list[Solution]:
    """A live solver's verify step with its flags kept: ``verify(candidates,
    max_solutions)`` (``ssik.refinement.verify_candidates``) on the unflagged
    and the flagged candidates apart, the flagged survivors marked as
    :class:`FlaggedSolution`, then one same-root dedup over both. With nothing
    flagged it is exactly ``verify(candidates, max_solutions)``."""
    if all(f == NOT_FLAGGED for f in free):
        return verify(list(candidates), max_solutions)
    plain = [c for c, f in zip(candidates, free, strict=True) if f == NOT_FLAGGED]
    out = verify(plain, None) if plain else []
    for f in sorted({f for f in free if f != NOT_FLAGGED}):
        group = [c for c, g in zip(candidates, free, strict=True) if g == f]
        out += [
            FlaggedSolution(s.q, s.fk_residual, s.refinement_used, free_joint=f)
            for s in verify(group, None)
        ]
    out = dedup_same_root(out, dedup_atol, fk_fn, t_target)
    return out if max_solutions is None else out[:max_solutions]


def three_parallel_wrist_error(
    axes: Sequence[ArrayLike],
    r_home: ArrayLike,
    t_target: NDArray[np.float64],
    q1: float,
    q5: float,
    theta14: float,
    q6: float,
) -> float:
    """How far a three_parallel branch's wrist angles miss the target's wrist
    rotation: ``||Rot(axes[0], q1)^T R_06 - Rot(axes[1], theta14) Rot(axes[4],
    q5) Rot(axes[5], q6)||_F`` (see :func:`spherical_wrist_error`)."""
    a = [np.asarray(x, dtype=np.float64) for x in axes]
    r_06 = np.asarray(t_target, dtype=np.float64)[:3, :3] @ np.asarray(r_home).T
    want = rotation_matrix(a[0], q1).T @ r_06
    got = rotation_matrix(a[1], theta14) @ rotation_matrix(a[4], q5) @ rotation_matrix(a[5], q6)
    return float(np.linalg.norm(want - got))


def three_parallel_wrist_misses(
    axes: Sequence[ArrayLike],
    r_home: ArrayLike,
    t_target: NDArray[np.float64],
    q1: float,
    q5: float,
    theta14: float,
    q6: float,
    q_free: float,
    offsets: Sequence[ArrayLike],
    gate: float,
) -> bool:
    """Whether a flagged three_parallel branch is split
    (:func:`three_parallel_lock`) because its SP1 wrist angles cannot give a
    candidate that passes the FK ``gate``. Near a lock the SP1s fix the wrist's
    sum ``theta14 +- q6`` well, but each angle alone only to about ``eps /
    sine``, and the numerical SP6 leaves ``sine`` at ~1e-8 at an exact lock
    (#668). Either the angles miss the target's wrist rotation by more than
    ``gate`` (:func:`three_parallel_wrist_error`, a lower bound on the
    candidate's FK residual), or the ``theta14`` they read leaves the elbow
    more than ``gate`` out of reach while a free value the split tries reaches
    it. Where none does (the elbow is tangent, exactly stretched or folded,
    at a single point of the lock), the SP1 angles are the better start for
    the refinement and are kept."""
    if three_parallel_wrist_error(axes, r_home, t_target, q1, q5, theta14, q6) > gate:
        return True
    a = [np.asarray(x, dtype=np.float64) for x in axes]
    p = [np.asarray(x, dtype=np.float64) for x in offsets]
    t = np.asarray(t_target, dtype=np.float64)
    r_06 = t[:3, :3] @ np.asarray(r_home).T
    p_16 = t[:3, 3] - p[0] - r_06 @ p[6]
    r_14 = rotation_matrix(a[1], theta14)
    d_inner = (
        rotation_matrix(a[0], q1).T @ p_16
        - p[1]
        - r_14 @ rotation_matrix(a[4], q5) @ p[5]
        - r_14 @ p[4]
    )
    if _sp3_miss(a[1], -p[3], p[2], float(np.linalg.norm(d_inner))) <= gate:
        return False
    return _three_parallel_lock(axes, r_home, t_target, q1, q5, q_free, offsets)[3]


def spherical_wrist_error(
    axes: Sequence[ArrayLike],
    r_home: ArrayLike,
    t_target: NDArray[np.float64],
    q1: float,
    q2: float,
    q3: float,
    q4: float,
    q5: float,
    q6: float,
) -> float:
    """How far a spherical wrist's angles miss the target's wrist rotation,
    ``||R_36 - Rot(axes[3], q4) Rot(axes[4], q5) Rot(axes[5], q6)||_F``
    Near a lock the SP1s that read ``q4`` and ``q6`` lose their accuracy (an
    emitted artifact's expanded arithmetic sooner than the native SP1); where
    they miss by more than ``RANK_TOL`` the core splits the lock instead."""
    a = [np.asarray(x, dtype=np.float64) for x in axes]
    r_06 = np.asarray(t_target, dtype=np.float64)[:3, :3] @ np.asarray(r_home).T
    want = (
        rotation_matrix(-a[2], q3) @ rotation_matrix(-a[1], q2) @ rotation_matrix(-a[0], q1) @ r_06
    )
    got = rotation_matrix(a[3], q4) @ rotation_matrix(a[4], q5) @ rotation_matrix(a[5], q6)
    return float(np.linalg.norm(want - got))


# The damping of the normal matrix rank_deficient tests, = the polish's
# (ssik.refinement.polish._DAMPING), so the polish's own solve yields it.
RANK_DAMPING = 1e-9


def rank_deficient_normal(
    normal: NDArray[np.float64], normal_inv: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """:func:`rank_deficient` from the normal matrices ``A = J^T J +
    RANK_DAMPING I`` ``(N, 6, 6)`` and their inverses."""
    kappa = np.sqrt(
        np.einsum("nij,nij->n", normal, normal) * np.einsum("nij,nij->n", normal_inv, normal_inv)
    )
    out: NDArray[np.bool_] = ~(kappa * RANK_TOL * RANK_TOL < 0.5)
    return out


def has_null(jac: NDArray[np.float64]) -> bool:
    """Whether one Jacobian has a null direction, ``sigma_min <= RANK_TOL
    sigma_max`` (an SVD: the slide's own confirmation). general_6r's flag for a
    refined near-miss, which has no polish to read the bound from."""
    s = np.linalg.svd(np.asarray(jac, dtype=np.float64), compute_uv=False)
    return bool(s[-1] <= RANK_TOL * s[0])


def rank_deficient(jac: NDArray[np.float64]) -> NDArray[np.bool_]:
    """``(N,)``: whether each 6x6 Jacobian of ``jac`` ``(N, 6, 6)`` may be rank
    deficient: the Frobenius condition number of ``A = J^T J + RANK_DAMPING
    I`` is at least ``0.5 / RANK_TOL^2``. It bounds ``kappa_2(A) = (sigma_max^2
    + d) / (sigma_min^2 + d)`` from above, which is at least ``0.5 /
    RANK_TOL^2`` whenever ``sigma_min <= RANK_TOL sigma_max`` (the damping
    ``d`` is far below ``RANK_TOL^2 sigma_max^2`` for any arm), so it misses no
    such Jacobian; the slide confirms with an SVD. The polish evaluates ``A``
    and solves with it anyway, so for a candidate it polishes the test costs
    one wider solve. A Jacobian singular in floating point is flagged."""
    jac = np.asarray(jac, dtype=np.float64)
    normal = np.swapaxes(jac, 1, 2) @ jac + RANK_DAMPING * np.eye(jac.shape[2])
    try:
        inv = np.asarray(np.linalg.inv(normal), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.ones(jac.shape[0], dtype=bool)
    return rank_deficient_normal(normal, inv)


def _wrap(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray((a + math.pi) % _TWO_PI - math.pi)


def _inv_rigid(t: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.eye(4)
    rt = t[:3, :3].T
    out[:3, :3] = rt
    out[:3, 3] = -rt @ t[:3, 3]
    return out


class _Chain:
    """FK, Jacobian and the in-limits test the slide works with."""

    def __init__(
        self,
        fk: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        jac: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        t_target: NDArray[np.float64],
        limits: Sequence[tuple[float, float] | None],
    ) -> None:
        self.fk, self.jac, self.t = fk, jac, t_target
        self.floor = same_root_floor(t_target)
        self.limits = list(limits)

    def residual(self, x: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(np.asarray(self.fk(x)) - self.t))

    def correct(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        """Truncated Gauss-Newton onto the solution set; the best point seen."""
        x = np.array(x, dtype=np.float64)
        f = np.asarray(self.fk(x), dtype=np.float64)
        r = float(np.linalg.norm(f - self.t))
        best_x, best_r = x, r
        for _ in range(CORRECT_ITERS):
            if r <= self.floor:
                break
            u, s, vt = np.linalg.svd(np.asarray(self.jac(x), dtype=np.float64))
            keep = s > RANK_TOL * s[0]
            e = se3_log_residual(self.t @ _inv_rigid(f))
            x = x + vt[keep].T @ ((u[:, keep].T @ e) / s[keep])
            f = np.asarray(self.fk(x), dtype=np.float64)
            r = float(np.linalg.norm(f - self.t))
            if not r < best_r:
                break
            best_x, best_r = x, r
        return best_x, best_r

    def tangent(
        self, x: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
        """At ``x``: the continuum's unit tangent (the last right singular
        vector, oriented as in the module docstring), an orthonormal basis of
        the null directions (rows: every right singular vector whose sigma is
        at most ``RANK_TOL sigma_max``), and whether there is one."""
        _, s, vt = np.linalg.svd(np.asarray(self.jac(x), dtype=np.float64))
        v = np.array(vt[-1], dtype=np.float64)
        big = np.flatnonzero(np.abs(v) >= FREE_MIN)
        if big.size and v[big[-1]] < 0.0:
            v = -v
        null = vt[s <= RANK_TOL * s[0]]
        return v, null, bool(null.shape[0])

    def in_limits(self, x: NDArray[np.float64]) -> bool:
        """Whether every joint has a winding inside its limits (within the
        round-off band, as the limit filter)."""
        for i, lim in enumerate(self.limits):
            if lim is None:
                continue
            lo, hi = lim[0] - _LIMIT_BAND, lim[1] + _LIMIT_BAND
            xi = float(x[i])
            # The lowest winding at or above lo, give or take the rounding of
            # the division.
            k = math.ceil((lo - xi) / _TWO_PI)
            if not any(lo <= xi + _TWO_PI * j <= hi for j in (k - 1, k, k + 1)):
                return False
        return True


def _free_joint(free: int, null: NDArray[np.float64]) -> int:
    """The core's free joint, or the highest-index joint the null directions
    move by at least ``FREE_MIN`` (``sqrt(P_ii)``, ``P`` the projector)."""
    if free >= 0:
        return free
    big = np.flatnonzero(np.sqrt(np.einsum("ki,ki->i", null, null)) >= FREE_MIN)
    return int(big[-1]) if big.size else -1


def _walk_to_rule(
    ch: _Chain,
    x: NDArray[np.float64],
    free: int,
    seed: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Step 3: from a corrected point of the continuum to the rule's point.

    Each step is the rule's displacement projected onto the null directions
    (for a one-dimensional null space, along the tangent): seeded, ``P
    wrap(seed - q)``; unseeded, the least move ``P e_f wrap(-q_f) / P_ff`` that
    zeroes the free joint to first order."""
    jumped = False
    for _ in range(MAX_STEPS):
        _, null, ok = ch.tangent(x)
        if not ok:
            break
        if seed is not None:
            d = null.T @ (null @ _wrap(seed - x))
        else:
            f = _free_joint(free, null)
            if f < 0:
                break
            pf = null[:, f]
            g = float(pf @ pf)
            if g < FREE_MIN * FREE_MIN:
                break
            d = (-float(_wrap(np.array([x[f]]))[0]) / g) * (null.T @ pf)
        n = float(np.linalg.norm(d))
        if n <= STOP:
            break
        y, ry = ch.correct(x + (d * (STEP / n) if n > STEP else d))
        if ry > SLIDE_FK:
            if seed is None or jumped or n <= STEP:
                break
            # The family closes FK only approximately between here and the
            # seed's point (a nearly aligned wrist), where the seed may be an
            # exact solution: take the whole step once, as the core would set
            # the free joint, and carry on from there if it is near one.
            jumped = True
            y, ry = ch.correct(x + d)
            if ry > WALK_FK:
                break
        x = y
    return x


def _walk_into_limits(
    ch: _Chain, x: NDArray[np.float64], sign: float
) -> tuple[NDArray[np.float64], float] | None:
    """Step 4, one direction (``sign`` times the oriented tangent): the first
    in-limit point and the arc to it, or None where the continuum ends first."""
    v_prev, _, ok = ch.tangent(x)
    if not ok:
        return None
    v_prev = sign * v_prev
    y, arc = x, 0.0
    for _ in range(math.ceil(WALK_ARC / STEP)):
        v, _, ok = ch.tangent(y)
        if not ok:
            return None
        if float(v @ v_prev) < 0.0:
            v = -v
        z, rz = ch.correct(y + STEP * v)
        if rz > SLIDE_FK:
            return None
        if ch.in_limits(z):
            out, inside = y, z
            for _ in range(BISECT_STEPS):
                m, rm = ch.correct(out + 0.5 * (inside - out))
                if rm > SLIDE_FK:
                    break
                if ch.in_limits(m):
                    inside = m
                else:
                    out = m
            return inside, arc + float(np.linalg.norm(inside - y))
        arc += float(np.linalg.norm(z - y))
        y, v_prev = z, v
    return None


def _slide_one(
    ch: _Chain,
    sol: Solution,
    free: int,
    seed: NDArray[np.float64] | None,
    respect_limits: bool,
) -> Solution:
    sol = Solution(sol.q, sol.fk_residual, sol.refinement_used)
    x0 = np.asarray(sol.q, dtype=np.float64)
    if not ch.tangent(x0)[2]:
        return sol
    x, _ = ch.correct(x0)
    if not ch.tangent(x)[2]:
        return sol
    x = _walk_to_rule(ch, x, free, seed)
    if respect_limits and not ch.in_limits(x):
        best: tuple[NDArray[np.float64], float] | None = None
        for sign in (1.0, -1.0):
            hit = _walk_into_limits(ch, x, sign)
            if hit is not None and (best is None or hit[1] < best[1]):
                best = hit
        if best is not None:
            x = best[0]
    r = ch.residual(x)
    if not r <= SLIDE_FK:
        return sol
    return Solution(q=x, fk_residual=r, refinement_used=sol.refinement_used)


def slide_continua(
    sols: list[Solution],
    free: Sequence[int],
    kb: KinBody,
    t_target: NDArray[np.float64],
    *,
    fk: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jac: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    q_seed: NDArray[np.float64] | None = None,
    respect_limits: bool | str = True,
    dedup_atol: float = 1e-3,
) -> list[Solution]:
    """Move every flagged solution to its continuum's point under the rule
    (module docstring), then merge the samples that met.

    :param sols: the core's solutions, after its polish and before finalize.
    :param free: per solution, :data:`NOT_FLAGGED`, :data:`FREE_FROM_KERNEL`,
        or the index of the free joint the core set.
    :param kb: the chain whose joint limits apply.
    :param fk: ``q -> 4x4`` FK of the chain the residuals are measured on.
    :param jac: ``q -> (6, 6)`` spatial Jacobian of the same chain.
    :param q_seed: the caller's seed, or ``None`` for the unseeded rule.
    :param respect_limits: only ``True`` brings a point into limits.
    :param dedup_atol: the same-root pre-filter radius (``subproblem_dedup``).
    :returns: the solutions in their order, slid ones replaced, merged ones
        removed. Unchanged (the same list) when nothing is flagged.
    """
    if all(f == NOT_FLAGGED for f in free):
        return sols
    t_target = np.asarray(t_target, dtype=np.float64)
    limits = [j.limits if j.joint_type == "revolute" else None for j in kb.joints]
    ch = _Chain(fk, jac, t_target, limits)
    seed = None if q_seed is None else np.asarray(q_seed, dtype=np.float64)
    out = [
        s if f == NOT_FLAGGED else _slide_one(ch, s, f, seed, respect_limits is True)
        for s, f in zip(sols, free, strict=True)
    ]
    return dedup_same_root(out, dedup_atol, fk, t_target)
