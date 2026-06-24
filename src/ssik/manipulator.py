"""Public Manipulator class -- the v1.0 entry point.

A :class:`Manipulator` is the user-facing handle for a robot arm. Construct
one via :meth:`Manipulator.from_urdf`, then call :meth:`.fk` and :meth:`.ik`
for the common path. The dispatched solver auto-routes based on the arm's
kinematic topology; users do not need to know which solver is firing.

Example::

    import ssik

    arm = ssik.Manipulator.from_urdf(
        "tests/fixtures/ur5.urdf", base="base_link", ee="ee_link"
    )
    T = arm.fk([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    sols = arm.solve(T, max_solutions=1, q_seed=q_prev)
    # sols is a list[Solution]; empty iff no IK closed within tolerance.

The class is intentionally tiny: factory + fk + solve + a handful of
properties. Power users who need solver-specific knobs pass them via
``solver_kwargs``; power users who need the underlying :class:`KinBody`
can reach :attr:`kinbody`.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik._kinbody import KinBody
from ssik.core.diagnostic import Diagnostic
from ssik.core.dispatcher import DispatchPlan, dispatch
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_fk import poe_forward_kinematics

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["Manipulator"]


# Map dispatcher solver names (e.g. "ikgeo.three_parallel") to the dotted
# module path under :mod:`ssik.solvers`. Centralised so the import-path
# convention is stated once.
_SOLVER_MODULE_PATHS: dict[str, str] = {
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
    "jointlock.seven_r": "ssik.solvers.jointlock.seven_r",
}


class Manipulator:
    """Public IK + FK handle for a robot arm.

    Construct via the factory classmethods (e.g. :meth:`from_urdf`), then use
    :meth:`fk` and :meth:`ik` for the common path. The dispatched analytical
    solver auto-routes based on the arm's kinematic topology -- users do not
    need to import or name a specific solver.

    .. note::
        The constructor signature ``Manipulator(kinbody)`` is the escape hatch
        for callers who already have a :class:`KinBody` (e.g. from a custom
        loader). Most users should use :meth:`from_urdf` instead.
    """

    __slots__ = ("_kb", "_plan", "_solver_module", "_warned_cold_coverage")

    def __init__(
        self,
        kinbody: KinBody,
        *,
        policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    ) -> None:
        """Wrap a pre-built :class:`KinBody`.

        :param kinbody: a POE-normalised :class:`KinBody`. If you have a URDF,
            use :meth:`from_urdf` instead -- it builds the KinBody for you.
        :param policy: tolerance policy used by the topology dispatcher.
            Defaults to :data:`~ssik.core.tolerances.DEFAULT_TOLERANCE_POLICY`.
        """
        self._kb: KinBody = kinbody
        self._plan: DispatchPlan = dispatch(kinbody, policy=policy)
        self._solver_module: ModuleType = importlib.import_module(
            _SOLVER_MODULE_PATHS[self._plan.solver_name]
        )
        # Guards the one-time cold-coverage warning (#328).
        self._warned_cold_coverage: bool = False

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_urdf(
        cls,
        path: str | Path,
        *,
        base: str,
        ee: str,
        policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
        xacro_args: dict[str, str] | None = None,
    ) -> Manipulator:
        """Load a URDF (or xacro) and build a :class:`Manipulator` for the chain
        between ``base`` and ``ee``.

        The kinematic chain is POE-normalised internally so the dispatcher
        can match the arm's topology against the solver roster. Mesh loading
        is lazy (the URDF parser does not load STL files unless asked).

        Xacro descriptions (``.xacro`` / ``*.urdf.xacro``, or a ``.urdf`` with a
        xacro namespace) are expanded automatically via ``xacrodoc``
        (``pip install ssik[xacro]``).

        :param path: path to the URDF or xacro file.
        :param base: name of the base link in the URDF.
        :param ee: name of the end-effector link in the URDF.
        :param policy: tolerance policy. Defaults to
            :data:`~ssik.core.tolerances.DEFAULT_TOLERANCE_POLICY`.
        :param xacro_args: substitution args for parametrized xacro descriptions
            (e.g. ``{"ur_type": "ur10e"}``); ignored for plain URDFs.

        :raises FileNotFoundError: if ``path`` doesn't exist.
        :raises ValueError: if ``base`` or ``ee`` are not link names in the URDF.
        :raises ImportError: if the optional dependency ``urchin`` is not
            installed (``pip install ssik[urdf]``).
        """
        # Imported lazily so the urchin dependency is only required when
        # from_urdf is actually called (it's an optional extra).
        from ssik._urdf import load_urdf_kinbody_normalized

        # urchin raises ValueError on missing files; rewrite to the more
        # idiomatic FileNotFoundError for the public API contract.
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"URDF file not found: {path_obj}")

        kb = load_urdf_kinbody_normalized(path, base, ee, xacro_args=xacro_args)
        return cls(kb, policy=policy)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def dof(self) -> int:
        """Number of joints in the chain."""
        return len(self._kb.joints)

    @property
    def joint_limits(self) -> list[tuple[float, float] | None]:
        """Per-joint ``(lower, upper)`` limits, or ``None`` for unconstrained.

        Units are radians for revolute joints, metres for prismatic. Joints
        without explicit limits in the source URDF are ``None`` (typically
        continuous revolute joints).
        """
        return [j.limits for j in self._kb.joints]

    @property
    def dispatch_plan(self) -> DispatchPlan:
        """Diagnostic info on which analytical solver was selected and why.

        Useful for understanding why a given arm dispatches to a particular
        solver (the ``reason`` field is a multi-line user-facing explanation).
        """
        return self._plan

    @property
    def kinbody(self) -> KinBody:
        """The underlying POE-normalised :class:`KinBody`.

        Power-user escape hatch for callers who need to invoke a specific
        solver, run codegen, or interact with the kinematics primitives
        directly. Most users should not need this.
        """
        return self._kb

    @property
    def solver_name(self) -> str:
        """Dotted name of the dispatched solver, e.g. ``"ikgeo.three_parallel"``."""
        return self._plan.solver_name

    def __repr__(self) -> str:
        return (
            f"<Manipulator: {self.dof}-DOF, dispatched to "
            f"{self._plan.solver_name} (tier {self._plan.tier})>"
        )

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def fk(self, q: ArrayLike) -> NDArray[np.float64]:
        """Forward kinematics: return the 4x4 SE(3) pose at config ``q``.

        :param q: joint vector of length :attr:`dof`. Float-castable
            (list, tuple, numpy array all work).

        :returns: 4x4 numpy array. Top-left 3x3 is the end-effector
            rotation matrix; column 3 (rows 0..2) is the position.

        :raises ValueError: if ``len(q) != dof``.
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self.dof,):
            raise ValueError(f"fk expected q of shape ({self.dof},), got {q_arr.shape}")
        result: NDArray[np.float64] = poe_forward_kinematics(self._kb, q_arr)
        return result

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    @overload
    def solve(
        self,
        T_target: ArrayLike,
        *,
        explain: Literal[False] = False,
        max_solutions: int | None = None,
        q_seed: ArrayLike | None = None,
        respect_limits: bool = True,
        allow_refinement: bool = False,
        policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
        refinement_max_iters: int = 15,
        seed_metric: str = "wrap_linf",
        seed_tolerance: float | None = None,
        allow_rescue: bool = True,
        **solver_kwargs: Any,
    ) -> list[Solution]: ...

    @overload
    def solve(
        self,
        T_target: ArrayLike,
        *,
        explain: Literal[True],
        max_solutions: int | None = None,
        q_seed: ArrayLike | None = None,
        respect_limits: bool = True,
        allow_refinement: bool = False,
        policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
        refinement_max_iters: int = 15,
        seed_metric: str = "wrap_linf",
        seed_tolerance: float | None = None,
        allow_rescue: bool = True,
        **solver_kwargs: Any,
    ) -> tuple[list[Solution], Diagnostic]: ...

    def solve(
        self,
        T_target: ArrayLike,
        *,
        explain: bool = False,
        max_solutions: int | None = None,
        q_seed: ArrayLike | None = None,
        respect_limits: bool = True,
        allow_refinement: bool = False,
        policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
        refinement_max_iters: int = 15,
        seed_metric: str = "wrap_linf",
        seed_tolerance: float | None = None,
        allow_rescue: bool = True,
        **solver_kwargs: Any,
    ) -> list[Solution] | tuple[list[Solution], Diagnostic]:
        """Inverse kinematics: find every ``q`` such that ``fk(q) ≈ T_target``.

        :param T_target: 4x4 SE(3) target pose.
        :param explain: when ``True``, returns ``(sols, Diagnostic)``
            instead of just ``sols``. The diagnostic carries dispatch +
            filter attribution for triaging empty-list failures. Default
            ``False`` preserves the v1.0 return signature.
        :param max_solutions: optional cap on returned IKs (post-dedup,
            post-limits filter). ``None`` = full redundancy enumeration.
            ``1`` combined with ``q_seed`` is the trajectory-tracking idiom.
            On 7R jointlock arms the cap also short-circuits the lock-sweep
            internally for a 10-15x speedup.
        :param q_seed: optional length-:attr:`dof` seed. When provided,
            returned solutions are sorted by distance from ``q_seed``
            (closest first, via ``seed_metric``); with ``max_solutions``
            this returns the nearest configs to the seed. On jointlock-7R
            arms it also drives the lock-outward-from-seed fast path (#331).
        :param seed_metric: distance used to rank against ``q_seed``.
            ``"wrap_linf"`` (default) minimises the largest single-joint
            wrap-to-pi move (holds the branch during tracking); ``"wrap_l2"``
            minimises the summed move. Ignored when ``q_seed`` is ``None``.
        :param seed_tolerance: optional max per-joint deviation from ``q_seed``
            (radians, wrap-to-pi). When set, only solutions with every joint
            within ``seed_tolerance`` are returned -- a hard tracking guarantee
            that may return an empty list. ``None`` (default) keeps the
            best-effort behaviour. Requires ``q_seed``.
        :param respect_limits: when ``True`` (default), solutions outside
            URDF joint limits are dropped. Pass ``False`` for the raw
            geometric set (analysis / debugging).
        :param allow_refinement: opt into Newton polish for near-miss
            algebraic candidates. Default ``False`` -- the algebraic
            path is already at machine precision on tier-0 / SRS arms.
            Set ``True`` on tier-2 RR arms to recover edge-case
            candidates whose algebraic FK drifts above ``fk_atol``.
        :param allow_rescue: when ``True`` (default), if the analytical
            path returns no solutions for a target within the arm's reach
            (a measure-zero rank-deficient ridge), recover the IK via the
            T-perturbation rescue (#319) -- matching the baked ``ssik build``
            artifact's coverage so ``from_urdf`` isn't a worse "try before
            you build" path (#328). Set ``False`` for a guaranteed
            analytical-only result.
        :param policy: tolerance policy. Rarely customised. Defaults to
            :data:`~ssik.core.tolerances.DEFAULT_TOLERANCE_POLICY`.
        :param refinement_max_iters: cap on Newton iterations per candidate
            when ``allow_refinement=True``.
        :param solver_kwargs: forwarded verbatim to the dispatched solver
            for power users (``swivel_samples`` for SRS-class,
            ``linearity_joint`` for RR, etc.).

        :returns: list of :class:`Solution`, one per analytical IK branch.
            Empty list iff no candidate FK-closed within
            ``policy.subproblem_numerical`` (or all IKs were filtered by
            ``respect_limits=True``). Check ``if not sols:`` for
            "unreachable target". When ``explain=True``, returns
            ``(sols, Diagnostic)`` -- inspect ``Diagnostic.summary()``
            to attribute empty-list failures.

        :raises ValueError: if ``T_target.shape != (4, 4)`` or
            ``len(q_seed) != dof`` when ``q_seed`` is given.
        """
        from ssik.postprocess import nearest_to_seed as _ps_nearest_to_seed
        from ssik.postprocess import respect_limits as _ps_respect_limits
        from ssik.postprocess import within_seed_tolerance as _ps_within_seed_tolerance
        from ssik.postprocess import wrap_to_limits as _ps_wrap_to_limits

        T = np.asarray(T_target, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"solve expected T_target of shape (4, 4), got {T.shape}")
        if seed_tolerance is not None and q_seed is None:
            raise ValueError("seed_tolerance requires q_seed")
        if q_seed is not None:
            q_seed_arr: NDArray[np.float64] | None = np.asarray(q_seed, dtype=np.float64)
            assert q_seed_arr is not None
            if q_seed_arr.shape != (self.dof,):
                raise ValueError(f"q_seed expected shape ({self.dof},), got {q_seed_arr.shape}")
        else:
            q_seed_arr = None

        # Filter kwargs by the dispatched solver's signature so callers can
        # pass q_seed (or any other not-universally-supported kwarg) without
        # tripping TypeError on solvers that don't accept it. The dispatch
        # is determined at __init__ time, so the signature lookup is per-IK
        # but cheap (~5 us).
        sig = inspect.signature(self._solver_module.solve)
        params = sig.parameters
        kwargs: dict[str, Any] = {"policy": policy}
        if "allow_refinement" in params:
            kwargs["allow_refinement"] = allow_refinement
        if "refinement_max_iters" in params:
            kwargs["refinement_max_iters"] = refinement_max_iters
        # When respect_limits=True, don't short-circuit the inner solver's
        # sweep on max_solutions: the closest-to-seed branch the solver
        # picks first may be out-of-limits and the postprocess pass would
        # drop it, leaving zero results. Force the full sweep, then
        # filter + trim. The user opts out via respect_limits=False.
        if "max_solutions" in params:
            kwargs["max_solutions"] = None if respect_limits else max_solutions
        if q_seed_arr is not None and "q_seed" in params:
            kwargs["q_seed"] = q_seed_arr
        # Power-user kwargs override our defaults.
        kwargs.update(solver_kwargs)

        # One-time coverage warning (#328): the cold jointlock-7R path (no
        # build-time cached-RR prime) dispatches non-tier-0 inner sub-chains to
        # the universal Husty-Pfurner solver, which has coverage gaps the baked
        # ``ssik build`` artifact's cached-RR derivations don't. The
        # T-perturbation rescue below recovers many such poses, but for arms
        # whose cold path is broadly gappy it can't fully match the prebuilt --
        # so surface the difference rather than let it be silent.
        if self._plan.solver_name == "jointlock.seven_r" and not self._warned_cold_coverage:
            self._warned_cold_coverage = True
            warnings.warn(
                f"{self._plan.solver_name}: solving this 7R arm from a URDF uses the "
                "universal Husty-Pfurner inner solver, which can have reduced coverage "
                "(and is slower) vs the cached-RR artifact from `ssik build`. For full "
                "coverage and speed, build the per-arm artifact once.",
                UserWarning,
                stacklevel=2,
            )

        # Internal solver functions still return (sols, is_ls); unwrap the
        # tuple at the public-API boundary. `is_ls` is redundant with
        # `len(sols) == 0` in every shipped solver -- ssik #238 item 1.
        sols, _is_ls = self._solver_module.solve(self._kb, T, **kwargs)

        # Bulletproof rescue (#319 / #328): when the analytical path returns
        # nothing for a target within the arm's reach, recover the IK via the
        # T-perturbation rescue -- the same runtime layer the baked ``ssik
        # build`` artifacts apply. Without this, ``from_urdf(...).solve(T)``
        # silently has worse coverage than the prebuilt at measure-zero
        # rank-deficient ridges. Gated by the reach-sphere (triangle-inequality
        # upper bound on ``|T_pos|``) so far-field unreachable targets stay
        # cheap. ``allow_rescue=False`` is the guaranteed-analytical escape.
        if not sols and allow_rescue:
            reach_radius = sum(
                float(np.linalg.norm(j.T_left[:3, 3])) + float(np.linalg.norm(j.T_right[:3, 3]))
                for j in self._kb.joints
            )
            if float(np.linalg.norm(T[:3, 3])) <= reach_radius:
                from ssik.refinement.rescue import rescue_via_T_perturbation

                def _analytic(T_pert: NDArray[np.float64], **rescue_kwargs: Any) -> list[Solution]:
                    inner: dict[str, Any] = {"policy": policy}
                    inner.update({k: v for k, v in rescue_kwargs.items() if k in params})
                    inner_sols, _ = self._solver_module.solve(self._kb, T_pert, **inner)
                    return list(inner_sols)

                sols = rescue_via_T_perturbation(self.fk, _analytic, T, jacobian_fn=None)

        raw_candidate_count = len(sols)

        # Cross-arm postprocess pass: solvers that didn't honour kwargs
        # natively get them applied here so the public API is uniform.
        # Order: wrap_to_limits first (try +/- 2pi shift to bring branches
        # into the URDF range), THEN respect_limits (drop anything still
        # outside). Without the shift, IKs returned in [-2pi, 0] would
        # erroneously be filtered on arms with limits in [0, 2pi]
        # (the JACO 2 case).
        if respect_limits:
            sols = _ps_wrap_to_limits(sols, self._kb)
            pre_limit_count = len(sols)
            sols = _ps_respect_limits(sols, self._kb)
            dropped_by_limits = pre_limit_count - len(sols)
        else:
            dropped_by_limits = 0
        if q_seed_arr is not None:
            if seed_tolerance is not None:
                sols = _ps_within_seed_tolerance(sols, q_seed_arr, seed_tolerance)
            sols = _ps_nearest_to_seed(sols, q_seed_arr, metric=seed_metric)
        if max_solutions is not None and len(sols) > max_solutions:
            dropped_by_max_solutions = len(sols) - max_solutions
            sols = sols[:max_solutions]
        else:
            dropped_by_max_solutions = 0
        result: list[Solution] = sols
        if not explain:
            return result

        refinement_engaged = sum(1 for s in result if s.refinement_used == "lm")
        max_fk = max((s.fk_residual for s in result), default=float("nan"))
        diag = Diagnostic(
            solver_name=self._plan.solver_name,
            solver_tier=self._plan.tier,
            dispatch_reason=self._plan.reason,
            raw_candidates=raw_candidate_count,
            dropped_by_limits=dropped_by_limits,
            dropped_by_max_solutions=dropped_by_max_solutions,
            final_count=len(result),
            max_fk_residual=max_fk,
            refinement_engaged=refinement_engaged,
            fk_atol=policy.subproblem_numerical,
        )
        return result, diag
