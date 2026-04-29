"""Composer for ``ikgeo.general_6r`` (tier-2 Raghavan-Roth, the EAIK-gap path).

Specialisation strategy for tier-2 RR:

  1. **Build-time** (codegen time, runs once per arm):
     - Convert POE -> DH parameters via :func:`ssik.kinematics.poe_to_dh`.
     - Bake the DH tuple ``(alpha, a, d)``, the ``theta_offset`` vector,
       and the ``t_pre`` / ``t_post`` 4x4 SE(3) bridges as numpy literals.

  2. **Artifact import time** (eager precompute):
     - Trigger :func:`ssik.solvers.ikgeo._raghavan_roth._cached_derivation`
       on the baked DH params. Pays the 30-150 s symbolic-derivation cost
       once at import; subsequent solves are instant.

  3. **Runtime** (per IK call):
     - Map ``T_target`` from POE to DH frame via
       ``T_target_dh = inv(t_pre) @ T_target @ inv(t_post)``.
     - Call the runtime ``solve_all_ik`` with the baked DH tuple. The
       symbolic precompute is already cached so the heavy work is done.
     - Apply ``-theta_offset`` to recover POE-frame q.

The runtime numerical core (24x24 eigenvalue, Mobius reparam fallback,
Newton refinement) stays imported from
:mod:`ssik.solvers.ikgeo._raghavan_roth`. Phase 4 Cython compiles all
of that into native code.

Why this isn't full inlining: the AE-3 leftvar selection and 14x9 / 14x8
Sylvester-resultant build are sympy-derived per arm and produce
sympy.lambdify callables. Those callables ARE per-arm-specialised
already; they just live in process memory, not pickle-able to disk.
We accept the eager-precompute approach in this PR; full source-level
inlining of the lambdify outputs is a follow-up.

Outcome: JACO 2's specialised artifact pays the 150-300 s precompute
cost ONCE at ``import jaco2_ik`` time (or zero if already cached), then
each subsequent ``solve(T)`` runs at the post-precompute hot-path speed
(~5 ms on JACO 2 per #85's tier-2 RR benches). No first-call latency
hit at runtime.
"""

from __future__ import annotations

import textwrap

import numpy as np

from ssik._kinbody import KinBody
from ssik.kinematics.poe_to_dh import poe_to_dh

__all__ = ["compose", "render_constants_header"]


def render_constants_header() -> str:
    """Imports needed by the rendered general_6r artifact."""
    return (
        "import math\n"
        "from ssik.solvers.ikgeo._raghavan_roth import (\n"
        "    _cached_derivation as _ssik_cached_derivation,\n"
        "    solve_all_ik as _ssik_solve_all_ik,\n"
        ")\n"
    )


def compose(kb: KinBody) -> str:
    """Render ``_solve_algebraic`` for a Raghavan-Roth tier-2 6R arm.

    :param kb: a POE-normalised :class:`KinBody` with 6 revolute joints.
    :returns: Python source for ``_solve_algebraic(T_target)`` that:

        - Bakes the DH tuple + t_pre/t_post bridges as np.array literals.
        - Eagerly triggers symbolic precompute at artifact import time.
        - Maps T_target POE -> DH at solve time, calls solve_all_ik with
          the baked DH params, maps q DH -> POE on return.
    """
    if len(kb.joints) != 6:
        raise ValueError(f"general_6r composer requires 6-DOF chain; got {len(kb.joints)}")
    for joint in kb.joints:
        if joint.joint_type != "revolute":
            raise ValueError(f"general_6r requires all-revolute joints; got {joint.joint_type}")

    dh = poe_to_dh(kb)
    alpha = dh.alpha
    a = dh.a
    d = dh.d
    theta_offset = dh.theta_offset
    t_pre = dh.t_pre
    t_post = dh.t_post

    return _render_body(
        alpha=alpha,
        a=a,
        d=d,
        theta_offset=theta_offset,
        t_pre=t_pre,
        t_post=t_post,
    )


def _render_body(
    *,
    alpha: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    theta_offset: np.ndarray,
    t_pre: np.ndarray,
    t_post: np.ndarray,
) -> str:
    """Emit the artifact body for tier-2 RR with DH params baked."""
    alpha_lit = _array_literal(alpha)
    a_lit = _array_literal(a)
    d_lit = _array_literal(d)
    theta_offset_lit = _array_literal(theta_offset)
    t_pre_lit = _matrix_literal(t_pre)
    t_post_lit = _matrix_literal(t_post)

    # Eager-precompute trigger: at module import, populate the
    # _cached_derivation LRU cache so first solve() call doesn't pay the
    # symbolic-derivation cost.
    return textwrap.dedent(
        f"""\
        # --- baked DH parameters (from poe_to_dh at build time) ---
        _DH_ALPHA = {alpha_lit}
        _DH_A = {a_lit}
        _DH_D = {d_lit}
        _DH_THETA_OFFSET = {theta_offset_lit}
        _T_PRE = {t_pre_lit}
        _T_POST = {t_post_lit}
        _T_PRE_INV = np.linalg.inv(_T_PRE)
        _T_POST_INV = np.linalg.inv(_T_POST)

        # Eagerly trigger symbolic precompute at import time so first
        # solve() call has no derivation latency. Returns immediately
        # if already cached.
        _ssik_cached_derivation(
            tuple(_DH_ALPHA.tolist()),
            tuple(_DH_A.tolist()),
            tuple(_DH_D.tolist()),
            linearity_joint=2,
            apply_so3=False,
        )


        def _solve_algebraic(T_target):
            \"\"\"Tier-2 Raghavan-Roth IK candidates for this arm.

            Bakes the DH params; routes to ssik.solvers.ikgeo._raghavan_roth.
            solve_all_ik with linearity_joint='auto' (AE-3 picks per-pose).
            \"\"\"
            T = np.asarray(T_target, dtype=np.float64)
            T_dh = _T_PRE_INV @ T @ _T_POST_INV
            inner_solutions, _is_ls = _ssik_solve_all_ik(
                (_DH_ALPHA, _DH_A, _DH_D),
                T_dh,
                fk_atol=1e-9,
                dedup_atol=1e-3,
                linearity_joint="auto",
                allow_refinement=False,
                refinement_max_iters=15,
                solver_name=SOLVER_NAME,
            )
            # Map DH-frame q back to POE frame.
            return [list(inner.q - _DH_THETA_OFFSET) for inner in inner_solutions]
        """
    )


def _array_literal(arr: np.ndarray) -> str:
    """Render a 1D numpy array as ``np.array([...])``."""
    return f"np.array({arr.tolist()!r}, dtype=np.float64)"


def _matrix_literal(arr: np.ndarray) -> str:
    """Render a 2D numpy array as ``np.array([[...], [...]])``."""
    return f"np.array({arr.tolist()!r}, dtype=np.float64)"
