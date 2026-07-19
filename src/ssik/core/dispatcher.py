"""Solver dispatcher: classify a KinBody and pick the best ssik solver.

Pure function:

    KinBody -> DispatchPlan

The :class:`DispatchPlan` is the structured handoff between the topology
classifier and the rest of the build pipeline. It carries everything the CLI
needs to print explanatory output and everything the codegen module needs to
emit a per-arm artifact:

* ``solver_name``: dotted module path of the chosen solver (e.g.
  ``ikgeo.three_parallel``).
* ``tier``: 0 closed-form, 1 univariate-search, 2 numeric Raghavan-Roth.
* ``reason``: human-readable explanation of which structural conditions
  matched and why this solver was preferred over its siblings.
* ``expected_ms_median``, ``flop_budget``: rough order-of-magnitude
  estimates from the speed-pass benches (#93). For showing users "expect
  ~X ms / IK on commodity hardware" before they run the artifact.
* ``needs_symbolic_precompute``: True iff the chosen solver runs a sympy
  preprocessing step. Tier-2 RR triggers this; tier-0/1 do not.
* ``estimated_precompute_seconds``: rough cold-cache time when applicable.

Dispatch decision order (best to worst):

1. Three consecutive parallel axes at (1, 2, 3) -> ``three_parallel``.
2. Three consecutive intersecting axes at (3, 4, 5):
   - + axes[1] || axes[2] -> ``spherical_two_parallel``.
   - + p[1] near zero      -> ``spherical_two_intersecting``.
   - else                  -> ``spherical``.
3. otherwise               -> ``general_6r`` (tier-2 Raghavan-Roth).

**Tier-1 univariate-search solvers (``two_parallel``, ``two_intersecting``)
are not auto-dispatched.** They run a 200-sample 1D grid + inner SPx per
sample and benchmark at 100s-of-ms to seconds per IK; the production tier-2
path (Raghavan-Roth + AE-3) handles the same chains in ~5 ms. Tier-1
solvers remain importable for users who want them explicitly, but they're
strictly slower than tier-2 RR on every measured workload.

Concretely: JACO 2 has ``axes[1] || axes[2]`` (parallel shoulder) but no
spherical wrist (60-degree non-orthogonal twist at joints 4-5). The naive
ordering would route it through ``two_parallel`` at ~261 ms median; instead
we route it through ``general_6r`` at ~5 ms median (#85). The 50x-200x
difference is consistent across non-Pieper geometries.

This dispatcher is shared in spirit with
:mod:`ssik.solvers.jointlock.seven_r` (whose internal tier-2 fallback is
``husty_pfurner.general_6r`` for locked sub-chains where no Pieper /
parallel-axis predicate matches). Top-level tier-2 routes to
``ikgeo.general_6r`` (Raghavan-Roth) which is faster on
well-conditioned native 6R arms (e.g. JACO 2 at ~0.6 ms); HP is
preferred for the ill-conditioned post-lock geometries that the
jointlock fallback hits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.predicates import (
    axes_meet_at_common_point,
    axis_parallel,
    three_consecutive_parallel,
)

__all__ = ["DispatchPlan", "dispatch"]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class DispatchPlan:
    """Result of solver-classification on a POE-normalized KinBody.

    Returned by :func:`dispatch`. The ``reason`` is the canonical user-facing
    explanation; tests assert on the structured fields, the CLI renders the
    ``reason`` text. Numeric fields are rough estimates from the #93 speed
    pass on Apple M3 single-thread; treat as "expect this order of
    magnitude," not a contract.
    """

    solver_name: str
    """Dotted path under :mod:`ssik.solvers` (e.g. ``ikgeo.three_parallel``)."""

    tier: int
    """0 closed-form, 1 univariate-search, 2 numeric Raghavan-Roth."""

    reason: str
    """Multi-line human-readable explanation. Suitable for printing as-is."""

    expected_ms_median: float
    """Approximate median wall-clock per IK on commodity x86 / Apple M3."""

    flop_budget: int
    """Approximate machine-invariant FLOPs per IK (#93)."""

    needs_symbolic_precompute: bool
    """True iff the solver runs sympy preprocessing during build/first call."""

    estimated_precompute_seconds: float | None
    """Rough cold-cache time when ``needs_symbolic_precompute=True``."""


# Per-solver order-of-magnitude estimates measured on Apple M3 single-thread,
# Python+numpy, post-#93 speed pass. These are the user-facing "expect ~Xms"
# numbers; codegen also bakes them into the emitted artifact's docstring.
_SOLVER_ESTIMATES: dict[str, tuple[int, float, int]] = {
    # solver_name -> (tier, expected_ms_median, flop_budget)
    "ikgeo.three_parallel": (0, 1.6, 2_519),
    "ikgeo.spherical_two_parallel": (0, 1.2, 1_316),
    "ikgeo.spherical_two_intersecting": (0, 1.3, 1_476),
    "ikgeo.spherical": (0, 7.5, 10_312),
    "ikgeo.two_parallel": (1, 261.0, 141_569),
    "ikgeo.two_intersecting": (1, 1184.0, 2_650_681),
    "ikgeo.general_6r": (2, 5.0, 30_000_000),
    "husty_pfurner.general_6r": (2, 120.0, 50_000_000),
    "seven_r.srs": (0, 8.5, 1_900),  # native SRS-class 7R, full sweep (16 swivel x 8 branches)
    "seven_r.srs_polished": (0, 56.0, 80_000),  # approximate-SRS + batched LM polish (Gen3 et al.)
    "seven_r.spherical_shoulder": (0, 17.0, 6_000),  # exact spherical-shoulder 7R (Franka/FR3)
    "seven_r.spherical_shoulder_polished": (0, 8.0, 40_000),  # near-spherical + LM polish (xArm7)
    "jointlock.seven_r": (1, 50.0, 30_274),  # 7R wrapper around inner 6R
}


def dispatch(kb: KinBody, policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY) -> DispatchPlan:
    """Classify a 6R or 7R chain and pick the best ssik solver.

    :param kb: a POE-normalized :class:`KinBody`. Pass the result of
        :func:`ssik._urdf.load_urdf_kinbody_normalized` (or your MJCF/DH
        equivalent). Non-normalized inputs produce undefined results.
    :param policy: tolerance policy controlling the predicates' axis-parallel
        / axis-intersect thresholds. Defaults to
        :data:`~ssik.core.tolerances.DEFAULT_TOLERANCE_POLICY`.
    :returns: a :class:`DispatchPlan` describing the chosen solver, the
        topology evidence, expected speed, and FLOP budget.
    :raises ValueError: if the chain is not 6R or 7R.

    7R chains route to :mod:`ssik.solvers.jointlock.seven_r`, which locks
    one joint (auto-selected by topology rank of the resulting 6R sub-chain)
    and sweeps it over 16 samples. Covers Franka, iiwa, Rizon, Gen3, xArm7,
    and any other 7R revolute arm.
    """
    if len(kb.joints) == 7:
        # Tier-0 7R: native SRS-class analytical solver (Singh-Kreutz).
        # Detects shoulder-spherical + wrist-spherical topology by exact
        # axis concurrence. Covers KUKA iiwa LBR (the canonical strict-SRS
        # arm).
        from ssik.kinematics.predicates import is_approximately_srs_7r, is_srs_7r

        if is_srs_7r(kb, policy) is not None:
            plan = _make_plan(
                "seven_r.srs",
                reason=(
                    "SRS-class 7R: shoulder axes (joints 0, 1, 2) meet at\n"
                    "one point + wrist axes (joints 4, 5, 6) meet at one\n"
                    "point + joint 3 is the elbow. Closed-form Singh-Kreutz\n"
                    "1989 algorithm, parameterised by elbow swivel angle.\n"
                    "Covers KUKA iiwa LBR (canonical strict-SRS)."
                ),
                needs_symbolic_precompute=False,
            )
            _LOG.info("dispatch: chose %s (tier 0, native SRS-7R)", plan.solver_name)
            return plan

        # Tier-0 7R: exact spherical-shoulder + offset-wrist (Franka Panda / FR3
        # and any arm of the class). Not SRS (the wrist is offset, not spherical),
        # but the last joint's redundancy resolves in closed form -- faster than
        # the jointlock sweep, machine precision, and zero coverage gaps.
        from ssik.solvers.seven_r.spherical_shoulder import is_spherical_shoulder_7r

        if is_spherical_shoulder_7r(kb, policy):
            plan = _make_plan(
                "seven_r.spherical_shoulder",
                reason=(
                    "Spherical-shoulder + offset-wrist 7R: shoulder axes\n"
                    "(joints 0, 1, 2) meet at one point but the wrist is\n"
                    "offset. Treats the last joint as the redundancy and\n"
                    "resolves its reachable/in-limits interval exactly in\n"
                    "closed form (SP3 bracket x feasible arcs on q_i(q6)).\n"
                    "Faster than the jointlock sweep, machine precision, no\n"
                    "coverage gaps. Covers Franka Panda / FR3."
                ),
                needs_symbolic_precompute=False,
            )
            _LOG.info("dispatch: chose %s (tier 0, spherical-shoulder-7R)", plan.solver_name)
            return plan

        # Tier-0 7R: approximately-spherical-shoulder (xArm7) -- the reversed
        # lock-6 wrist triple is concurrent to within a small drift, so the
        # closed-form q_i(q6) seeds + LM polish reach machine precision.
        from ssik.solvers.seven_r.spherical_shoulder_polished import (
            is_approximately_spherical_shoulder_7r,
        )

        if is_approximately_spherical_shoulder_7r(kb, policy=policy):
            plan = _make_plan(
                "seven_r.spherical_shoulder_polished",
                reason=(
                    "Approximately-spherical-shoulder 7R: the reversed\n"
                    "last-joint-locked wrist triple is concurrent to within\n"
                    "a small drift. The closed-form spherical-shoulder recipe\n"
                    "produces excellent seeds; LM polish against the true FK\n"
                    "recovers machine precision. Covers uFactory xArm7."
                ),
                needs_symbolic_precompute=False,
            )
            _LOG.info("dispatch: chose %s (tier 0, approx-spherical-7R)", plan.solver_name)
            return plan

        # Tier-0 7R: approximate-SRS variant for arms whose URDF axes only
        # nearly meet (Kinova Gen3: 12 mm shoulder + 0.4 mm wrist drift).
        # Singh-Kreutz solver as warm-start factory + LM polish to
        # machine precision against the original URDF FK.
        approx = is_approximately_srs_7r(kb, max_drift_m=0.04, policy=policy)
        if approx is not None:
            plan = _make_plan(
                "seven_r.srs_polished",
                reason=(
                    "Approximately-SRS 7R: shoulder axes meet within "
                    f"{approx.shoulder_drift_m * 1000:.1f} mm, wrist axes "
                    f"meet within {approx.wrist_drift_m * 1000:.1f} mm.\n"
                    "Singh-Kreutz on the relaxed pivots produces algebraic\n"
                    "candidates; LM polish recovers machine-precision FK\n"
                    "against the original URDF. 16-30x faster than the\n"
                    "universal jointlock+HP fallback on small-drift arms.\n"
                    "Covers Kinova Gen3 (12 mm / 0.4 mm drift)."
                ),
                needs_symbolic_precompute=False,
            )
            _LOG.info("dispatch: chose %s (tier 0, approximate-SRS-7R)", plan.solver_name)
            return plan

        # Tier-1 7R fallback: joint-lock + dispatch the inner 6R.
        plan = _make_plan(
            "jointlock.seven_r",
            reason=(
                "7R revolute chain (non-SRS). Locking one joint\n"
                "(auto-selected by topology rank of the resulting 6R\n"
                "sub-chain) reduces this to a series of 6R IK problems.\n"
                "Covers Franka Panda, FR3, uFactory xArm7, and any other\n"
                "non-SRS 7R revolute arm."
            ),
            needs_symbolic_precompute=False,
        )
        _LOG.info("dispatch: chose %s (tier 1, 7R wrapper)", plan.solver_name)
        return plan
    if len(kb.joints) != 6:
        raise ValueError(f"dispatch supports 6-DOF and 7-DOF chains; got {len(kb.joints)} joints.")

    parallel_triple = three_consecutive_parallel(kb.joints, policy)
    # Gauge-invariant spherical-wrist test: the wrist axes (3, 4, 5) meeting at a
    # common point is the true Pieper condition, independent of where the joint
    # frame origins sit along those axes. A URDF flange offset (last wrist joint
    # placed along its own axis, e.g. ABB IRB 6700, #377) still routes here; the
    # spherical solver re-gauges it via ``canonicalize_spherical_wrist`` before
    # consolidating. (``three_consecutive_intersecting`` additionally requires
    # origin coincidence -- a consolidation convenience, not a routing condition.)
    wrist_meets = axes_meet_at_common_point(kb.joints, (3, 4, 5), policy) is not None
    j12_parallel = axis_parallel(kb.joints[1].axis, kb.joints[2].axis, policy)
    p1_norm = float(np.linalg.norm(kb.joints[1].T_left[:3, 3]))
    p1_on_axis = p1_norm < policy.axis_intersect

    # Tier 0 -- three consecutive parallel (UR class).
    if parallel_triple == (1, 2, 3):
        plan = _make_plan(
            "ikgeo.three_parallel",
            reason=(
                "Three consecutive parallel axes at joints (1, 2, 3) -- the "
                "UR-class structure (UR3 / UR5 / UR10).\n"
                "Closed-form via SP6 (joints 0+4) + SP1 + SP3."
            ),
            needs_symbolic_precompute=False,
        )
        _LOG.info("dispatch: chose %s (tier 0)", plan.solver_name)
        return plan

    # Tier 0 -- spherical wrist (Pieper class).
    if wrist_meets:
        if j12_parallel and p1_on_axis:
            # Both shoulder specializations match (Puma-560 case). Prefer the
            # parallel-shoulder solver -- typically smaller IK set and slightly
            # tighter conditioning on the SP3 elbow constraint.
            plan = _make_plan(
                "ikgeo.spherical_two_parallel",
                reason=(
                    "Spherical wrist at joints (3, 4, 5) AND axes[1] parallel "
                    "to axes[2] AND ||p[1]|| ~= 0.\n"
                    "Both Pieper specialisations apply (e.g. Puma 560); the "
                    "parallel-shoulder solver is preferred for slightly tighter "
                    "elbow conditioning."
                ),
                needs_symbolic_precompute=False,
            )
        elif j12_parallel:
            plan = _make_plan(
                "ikgeo.spherical_two_parallel",
                reason=(
                    "Spherical wrist at joints (3, 4, 5) AND axes[1] parallel "
                    "to axes[2].\n"
                    "Closed-form via SP4 (shoulder) + SP3 (elbow) + SP1 (wrist). "
                    "Covers most industrial 6R arms (Puma, Fanuc LR/CR, KUKA KR)."
                ),
                needs_symbolic_precompute=False,
            )
        elif p1_on_axis:
            plan = _make_plan(
                "ikgeo.spherical_two_intersecting",
                reason=(
                    "Spherical wrist at joints (3, 4, 5) AND ||p[1]|| ~= 0 "
                    "(joints 0 and 1 share an origin).\n"
                    "Closed-form via SP3 + SP2 + SP4 + SP1. Compact-base arms "
                    "(IRB120-class, lite6/xArm6 subfamilies)."
                ),
                needs_symbolic_precompute=False,
            )
        else:
            plan = _make_plan(
                "ikgeo.spherical",
                reason=(
                    "Spherical wrist at joints (3, 4, 5), no shoulder "
                    "specialisation.\n"
                    "Closed-form via SP5 (shoulder) + SP4 (wrist) + SP1. "
                    "Generic spherical-wrist fallback; rarely matches "
                    "commercial geometry."
                ),
                needs_symbolic_precompute=False,
            )
        _LOG.info("dispatch: chose %s (tier 0)", plan.solver_name)
        return plan

    # Tier 2 -- production Raghavan-Roth path. The EAIK gap.
    #
    # Skips tier-1 univariate-search solvers: those run a 200-sample 1D grid
    # plus an inner SP5/SP6 call per sample and benchmark at 100s-of-ms to
    # seconds, while RR handles the same chains at ~5ms. See module docstring
    # for the JACO 2 example and the speed comparison.
    weak_match_notes = []
    if j12_parallel:
        weak_match_notes.append(
            "axes[1] parallel to axes[2] (would match tier-1 `two_parallel`, "
            "but tier-2 RR is ~50x faster)"
        )
    if float(np.linalg.norm(kb.joints[5].T_left[:3, 3])) < policy.axis_intersect:
        weak_match_notes.append(
            "||p[5]|| ~= 0 (would match tier-1 `two_intersecting`, but tier-2 RR is ~200x faster)"
        )
    weak_block = (
        "\nWeaker structural matches (not used):\n  - " + "\n  - ".join(weak_match_notes)
        if weak_match_notes
        else ""
    )
    plan = _make_plan(
        "ikgeo.general_6r",
        reason=(
            "No tier-0 (Pieper-class) match.\n"
            "Tier-2 numeric Raghavan-Roth + Manocha-Canny pipeline with AE-3 "
            "leftvar selection. Closes the EAIK coverage gap (Kinova JACO 2 "
            "classical, Agilex Piper, custom non-Pieper 6R)." + weak_block
        ),
        needs_symbolic_precompute=True,
    )
    _LOG.info("dispatch: chose %s (tier 2)", plan.solver_name)
    return plan


def _make_plan(solver_name: str, *, reason: str, needs_symbolic_precompute: bool) -> DispatchPlan:
    """Assemble a :class:`DispatchPlan` from per-solver estimates + caller fields."""
    tier, ms, flops = _SOLVER_ESTIMATES[solver_name]
    # Symbolic precompute time estimate -- only applies to tier-2 RR currently.
    # 150-300 s on JACO 2 today; report a single midpoint estimate for the
    # build CLI's ETA. Real per-arm time depends on linearity-joint search
    # hits and sympy version; the build pass measures it precisely.
    precompute_s = 240.0 if needs_symbolic_precompute else None
    return DispatchPlan(
        solver_name=solver_name,
        tier=tier,
        reason=reason,
        expected_ms_median=ms,
        flop_budget=flops,
        needs_symbolic_precompute=needs_symbolic_precompute,
        estimated_precompute_seconds=precompute_s,
    )
