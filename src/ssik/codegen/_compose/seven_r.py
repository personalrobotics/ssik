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

import numpy as np

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
from ssik.solvers.jointlock.seven_r import (
    _DEFAULT_SAMPLES,
    _lock_joint,
    _topology_rank,
    choose_lock_joint,
)

__all__ = ["compose", "render_constants_header"]


def render_constants_header() -> str:
    """Imports needed by the rendered seven_r artifact."""
    return (
        "import math\n"
        "import numpy as np\n"
        "from ssik.solvers.jointlock import seven_r as _ssik_seven_r\n"
    )


def compose(kb: KinBody) -> str:
    """Render ``_solve_algebraic`` for a 7R arm via joint-locking.

    :param kb: a POE-normalised :class:`KinBody` with 7 revolute joints.
    :returns: Python source for ``_solve_algebraic(T_target)``. Calls the
        runtime ``ssik.solvers.jointlock.seven_r.solve`` with the baked
        7R KinBody and pre-selected ``lock_idx`` to produce the candidates;
        the orchestrator's verify + dedup applies.

    Codegen-time topology cache (#142 item 4): runs the lock sweep at
    ``_DEFAULT_SAMPLES`` once at codegen time and bakes the resulting
    inner-solver dispatch table. Runtime ``seven_r.solve`` skips its
    per-sample ``_topology_rank`` (~70 us with chain reversal) and
    uses the cached name directly. Saves ~1 ms per IK on a 16-sample
    sweep -- ~3-5% on Franka 7R default.
    """
    if len(kb.joints) != 7:
        raise ValueError(f"jointlock.seven_r composer requires 7-DOF chain; got {len(kb.joints)}")

    policy = DEFAULT_TOLERANCE_POLICY
    lock_idx = choose_lock_joint(kb, policy)

    # Compute the canonical lock-sample schedule + dispatch cache.
    joint_limits = kb.joints[lock_idx].limits
    if joint_limits is None:
        lo, hi = -float(np.pi), float(np.pi)
    else:
        lo, hi = joint_limits
    samples = np.linspace(lo, hi, _DEFAULT_SAMPLES, endpoint=False)
    dispatch: list[str] = []
    for q_lock in samples:
        sub_kb = _lock_joint(kb, lock_idx, float(q_lock))
        _, name = _topology_rank(sub_kb, policy)
        dispatch.append(name)

    samples_repr = ", ".join(repr(float(s)) for s in samples)
    dispatch_repr = ",\n            ".join(repr(name) for name in dispatch)

    return textwrap.dedent(
        f"""\
        # --- 7R via joint-lock ---
        # Pre-selected lock_idx via choose_lock_joint at codegen time, passed
        # to the runtime solve() so it skips the per-IK topology-rank scan
        # over all 7 lock candidates. Per-sample inner-solver dispatch still
        # happens at runtime (rotating downstream axes by R_lock can shift
        # which tier-0/1 specialization applies).
        _LOCK_IDX = {lock_idx}

        # Canonical lock-sample schedule (np.linspace over the locked
        # joint's range, ``_DEFAULT_SAMPLES`` samples, endpoint excluded).
        _LOCK_SAMPLES = np.array(
            [{samples_repr}],
            dtype=np.float64,
        )

        # Codegen-time topology cache (#142 item 4). Pre-computed via
        # ``_lock_joint`` + ``_topology_rank`` at each lock sample; runtime
        # ``seven_r.solve`` uses these directly instead of re-running the
        # topology rank per IK. The cache aligns by sample index with
        # ``_LOCK_SAMPLES``; under ``q_seed`` reordering the runtime
        # permutes the cache alongside the samples.
        _DISPATCH_CACHE = (
            {dispatch_repr},
        )


        def _solve_algebraic(T_target, *, max_solutions=None, q_seed=None):
            \"\"\"7R IK candidates via joint-locking + inner 6R sweep.

            Routes to ssik.solvers.jointlock.seven_r.solve with the baked
            KinBody, lock_idx, lock-sample schedule, and dispatch cache.
            ``max_solutions`` and ``q_seed`` are forwarded so the underlying
            lock-sweep can short-circuit (#142). Returns ``list[list[float]]``
            of length-7 q-vectors.
            \"\"\"
            sub_solutions, _is_ls = _ssik_seven_r.solve(
                _KB, T_target,
                lock_idx=_LOCK_IDX,
                lock_samples=_LOCK_SAMPLES,
                dispatch_cache=_DISPATCH_CACHE,
                max_solutions=max_solutions, q_seed=q_seed,
            )
            return [list(s.q) for s in sub_solutions]
        """
    )
