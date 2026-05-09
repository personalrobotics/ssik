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
    sols, is_ls = arm.ik(T)
    # sols is a list[Solution]; is_ls=True signals no candidate FK-closed
    # within tolerance.

The class is intentionally tiny: factory + fk + ik + a handful of properties.
Power users who need solver-specific knobs pass them via ``solver_kwargs``;
power users who need the underlying :class:`KinBody` can reach :attr:`kinbody`.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik._kinbody import KinBody
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

    __slots__ = ("_kb", "_plan", "_solver_module")

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
    ) -> Manipulator:
        """Load a URDF and build a :class:`Manipulator` for the chain
        between ``base`` and ``ee``.

        The kinematic chain is POE-normalised internally so the dispatcher
        can match the arm's topology against the solver roster. Mesh loading
        is lazy (the URDF parser does not load STL files unless asked).

        :param path: path to the URDF file.
        :param base: name of the base link in the URDF.
        :param ee: name of the end-effector link in the URDF.
        :param policy: tolerance policy. Defaults to
            :data:`~ssik.core.tolerances.DEFAULT_TOLERANCE_POLICY`.

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

        kb = load_urdf_kinbody_normalized(path, base, ee)
        return cls(kb, policy=policy)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def dof(self) -> int:
        """Number of joints in the chain."""
        return len(self._kb.joints)

    @property
    def joint_limits(self) -> list[tuple[float, float]]:
        """Per-joint ``(lower, upper)`` limits.

        Units are radians for revolute joints, metres for prismatic. Joints
        without explicit limits in the source URDF default to ``(-pi, +pi)``.
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
        return poe_forward_kinematics(self._kb, q_arr)

    # ------------------------------------------------------------------
    # Inverse kinematics
    # ------------------------------------------------------------------

    def ik(
        self,
        T_target: ArrayLike,
        *,
        max_solutions: int | None = None,
        q_seed: ArrayLike | None = None,
        policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
        allow_refinement: bool = False,
        refinement_max_iters: int = 15,
        **solver_kwargs: Any,
    ) -> tuple[list[Solution], bool]:
        """Inverse kinematics: find ``q`` such that ``fk(q) ≈ T_target``.

        :param T_target: 4x4 SE(3) target pose.
        :param max_solutions: optional cap on returned IKs. ``None`` = full
            redundancy enumeration. ``1`` is the trajectory-tracking
            "give me any IK" mode (typically 10-15x faster than full sweep
            on 7R arms). The cap is honored at every layer of the dispatch
            stack -- the inner solvers stop branch enumeration as soon as
            ``max_solutions`` deduplicated IKs are found.
        :param q_seed: optional length-:attr:`dof` seed configuration. When
            supported by the dispatched solver (currently
            :mod:`ssik.solvers.jointlock.seven_r`), ``q_seed`` reorders the
            internal lock-sample sweep so the closest-to-seed sample fires
            first; combined with ``max_solutions=1`` this is the canonical
            trajectory-tracking idiom. Solvers that don't accept ``q_seed``
            silently ignore it.
        :param policy: tolerance policy. Defaults to
            :data:`~ssik.core.tolerances.DEFAULT_TOLERANCE_POLICY`.
        :param allow_refinement: opt into Newton-on-spatial-Jacobian polish
            for candidates that don't FK-close algebraically. Default off;
            the analytical path is exact for well-conditioned poses.
        :param refinement_max_iters: cap on Newton iterations per candidate
            when ``allow_refinement=True``.
        :param solver_kwargs: forwarded verbatim to the dispatched solver's
            ``solve()`` for power users who need solver-specific knobs
            (``swivel_samples`` for SRS-class, ``linearity_joint`` for RR,
            etc.). Unknown kwargs raise :class:`TypeError` from the
            underlying solver.

        :returns: ``(solutions, is_ls)``. ``is_ls=True`` iff no candidate
            FK-closed within ``policy.subproblem_numerical``; the returned
            list is then either a single best-LS approximation or empty.

        :raises ValueError: if ``T_target.shape != (4, 4)`` or
            ``len(q_seed) != dof`` when ``q_seed`` is given.
        """
        T = np.asarray(T_target, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"ik expected T_target of shape (4, 4), got {T.shape}")
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
        if "max_solutions" in params:
            kwargs["max_solutions"] = max_solutions
        if q_seed_arr is not None and "q_seed" in params:
            kwargs["q_seed"] = q_seed_arr
        # Power-user kwargs override our defaults.
        kwargs.update(solver_kwargs)

        result: tuple[list[Solution], bool] = self._solver_module.solve(self._kb, T, **kwargs)
        return result
