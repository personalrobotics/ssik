"""Composer for ``jointlock.seven_r`` (7R arms via joint-locking).

Mirrors :func:`ssik.solvers.jointlock.seven_r.solve`:

  1. Auto-select lock joint at codegen time via :func:`choose_lock_joint`
     (deterministic per arm, runs once). Bake ``lock_idx`` as a constant.
  2. At runtime: sweep ``lock_idx`` over 16 samples. Per sample:
       - Build the 6R sub-chain by folding the locked joint out (similarity
         transform on downstream axes/translations + rolling the locked
         rotation into ``T_right[-1]``).
       - Topology-rank the sub-chain to pick the inner ikgeo solver
         (e.g. spherical_two_parallel for Franka after locking joint 3).
       - Dispatch to the inner solver, pad the result back to 7D.
  3. Wrap-to-pi dedup the full 7D candidate set.

What's specialised at codegen time:

  - The original 7R KinBody constants are baked (no KinBody construction
    at runtime; the orchestrator's standard kinbody-bake takes care of this).
  - ``_LOCK_IDX`` is computed once via :func:`choose_lock_joint` and passed
    to the runtime ``solve(..., lock_idx=_LOCK_IDX)``, skipping the per-IK
    topology-rank scan over all 7 lock candidates.

What stays at runtime:

  - The per-sample ``_lock_joint`` similarity transform + ``_topology_rank``
    of the locked sub-chain. These depend on ``q_lock``, so they cannot be
    pre-computed: rotating downstream axes by ``R_lock`` can switch which
    tier-0/1 specialization applies (e.g. near-parallel becoming exactly
    parallel at one sample, exactly intersecting at another).
  - Inner-solver dispatch + per-sample IK.

Phase 4 Cython compiles the ``_lock_joint`` similarity transform and the
inner-solver dispatch into native code, removing the per-sample Python
overhead.

Covered arms: Franka Panda, FR3, KUKA iiwa, Flexiv Rizon, Kinova Gen3,
uFactory xArm7. Each emits its own ``<arm>_ik.py`` artifact via the
existing ``ssik build`` CLI -- no per-arm wiring needed; the dispatcher
recognises 7R and routes here automatically.
"""

from __future__ import annotations

import textwrap

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
from ssik.solvers.jointlock.seven_r import choose_lock_joint

__all__ = ["compose", "render_constants_header"]


def render_constants_header() -> str:
    """Imports needed by the rendered seven_r artifact."""
    return "import math\nfrom ssik.solvers.jointlock import seven_r as _ssik_seven_r\n"


def compose(kb: KinBody) -> str:
    """Render ``_solve_algebraic`` for a 7R arm via joint-locking.

    :param kb: a POE-normalised :class:`KinBody` with 7 revolute joints.
    :returns: Python source for ``_solve_algebraic(T_target)``. Calls the
        runtime ``ssik.solvers.jointlock.seven_r.solve`` with the baked
        7R KinBody and pre-selected ``lock_idx`` to produce the candidates;
        the orchestrator's verify + dedup applies.
    """
    if len(kb.joints) != 7:
        raise ValueError(f"jointlock.seven_r composer requires 7-DOF chain; got {len(kb.joints)}")

    lock_idx = choose_lock_joint(kb, DEFAULT_TOLERANCE_POLICY)

    return textwrap.dedent(
        f"""\
        # --- 7R via joint-lock ---
        # Pre-selected lock_idx via choose_lock_joint at codegen time, passed
        # to the runtime solve() so it skips the per-IK topology-rank scan
        # over all 7 lock candidates. Per-sample inner-solver dispatch still
        # happens at runtime (rotating downstream axes by R_lock can shift
        # which tier-0/1 specialization applies).
        _LOCK_IDX = {lock_idx}


        def _solve_algebraic(T_target):
            \"\"\"7R IK candidates via joint-locking + inner 6R sweep.

            Routes to ssik.solvers.jointlock.seven_r.solve with the baked
            KinBody and pre-selected lock_idx. Returns ``list[list[float]]``
            of length-7 q-vectors.
            \"\"\"
            sub_solutions, _is_ls = _ssik_seven_r.solve(
                _KB, T_target, lock_idx=_LOCK_IDX
            )
            return [list(s.q) for s in sub_solutions]
        """
    )
