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

import inspect
import textwrap
import time
from typing import cast

import numpy as np

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import (
    _cached_best_leftvar,
    _cached_derivation,
)
from ssik.solvers.jointlock.seven_r import (
    _DEFAULT_SAMPLES,
    _RR_ELIGIBLE_INNER_SOLVERS,
    _lock_joint,
    _topology_rank,
    choose_lock_joint,
)

__all__ = ["compose", "render_constants_header"]


def render_constants_header() -> str:
    """Imports needed by the rendered seven_r artifact.

    Note: the ``prime_derivation_from_blob`` import + sidecar-blob load
    (cached-RR priming, #210 Phase 2) is added inline in :func:`compose`
    only when the arm has at least one eligible non-tier-0 inner sample.
    Arms with all-tier-0 dispatch (e.g. Franka pre-#219) keep their
    artifact byte-identical to pre-#210.
    """
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
    rr_prime_dhs: list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]] = []
    for q_lock in samples:
        sub_kb = _lock_joint(kb, lock_idx, float(q_lock))
        _, name = _topology_rank(sub_kb, policy)
        dispatch.append(name)
        # If the inner sample would route through a non-tier-0 path,
        # collect its DH so the artifact can prime RR's derivation cache
        # at module-import time (#210). The runtime jointlock dispatch
        # uses cached RR (~1 ms) instead of HP/two_parallel/spherical
        # (~13-260 ms) when the cache is primed.
        bare_name = name[len("reversed:") :] if name.startswith("reversed:") else name
        if bare_name in _RR_ELIGIBLE_INNER_SOLVERS:
            sub_kb_for_dh = sub_kb
            if name.startswith("reversed:"):
                # Reversed dispatch runs RR on the chain-reversed sub_kb.
                from ssik.kinematics.reverse import reverse_kinematic_chain

                sub_kb_for_dh = reverse_kinematic_chain(sub_kb)
            dh = poe_to_dh(sub_kb_for_dh)
            alpha = tuple(float(x) for x in dh.alpha)
            a = tuple(float(x) for x in dh.a)
            d = tuple(float(x) for x in dh.d)
            linearity = _cached_best_leftvar(alpha, a, d)
            rr_prime_dhs.append((alpha, a, d, linearity))

    samples_repr = ", ".join(repr(float(s)) for s in samples)
    dispatch_repr = ",\n            ".join(repr(name) for name in dispatch)
    if rr_prime_dhs:
        # #320 (replaces #210 Phase 2 blob path): pre-compute the symbolic
        # Raghavan-Roth derivation for each (alpha, a, d, linearity) tuple
        # at codegen time, then emit the lambdified callable's Python
        # source verbatim via ``inspect.getsource``. The artifact body
        # becomes ordinary Python -- no sympy at import, no re-lambdify,
        # no base85/zlib unpack. Measured on Kassow KR810: 4.5s blob-prime
        # -> 80ms AOT-prime (~57x faster cold module-import), and the
        # wheel-compressed artifact shrinks (text gzips better than the
        # already-zlib-compressed blob).
        bake_start = time.perf_counter()
        aot_func_sources: list[str] = []
        aot_entries: list[str] = []
        for slot_idx, (alpha, a, d, lin) in enumerate(rr_prime_dhs):
            p_sin_fn, p_cos_fn, p_one_fn, q_fn, meta = _cached_derivation(
                alpha, a, d, linearity_joint=int(lin), apply_so3=False
            )
            slot_names: list[str] = []
            for kind, fn in (
                ("p_sin", p_sin_fn),
                ("p_cos", p_cos_fn),
                ("p_one", p_one_fn),
                ("q", q_fn),
            ):
                uname = f"_aot_dh{slot_idx}_{kind}"
                src = inspect.getsource(fn).replace("_lambdifygenerated", uname, 1)
                aot_func_sources.append(src)
                slot_names.append(uname)
            # meta is typed ``dict[str, object]``; cast the runtime-side
            # fields we actually emit so mypy doesn't complain about the
            # ``int(meta['drop_joint'])`` call.
            drop_joint_val = cast(int, meta["drop_joint"])
            entry = (
                f"            (\n"
                f"                {alpha!r},\n"
                f"                {a!r},\n"
                f"                {d!r},\n"
                f"                {int(lin)!r}, False,\n"
                f"                {meta['left_bilinear']!r},\n"
                f"                {meta['right_bilinear']!r},\n"
                f"                {int(drop_joint_val)!r},\n"
                f"                {', '.join(slot_names)},\n"
                f"            ),"
            )
            aot_entries.append(entry)
        _LOG_TIME = time.perf_counter() - bake_start

        # Each fn source emitted by sp.lambdify is a 2-line snippet:
        #   def _aot_dh{slot}_{kind}(T_0, T_1, ...):
        #       return array([...])
        # Prefix every line with 8 spaces so the surrounding
        # textwrap.dedent leaves us at module-level after stripping.
        def _indent8(src: str) -> str:
            return "\n".join("        " + line if line else line for line in src.splitlines())

        aot_funcs_block = "\n\n".join(_indent8(s) for s in aot_func_sources)
        aot_entries_block = "\n".join(aot_entries)
        rr_prime_block = (
            "\n\n        from numpy import array, cos, sin\n\n"
            "        from ssik.solvers.ikgeo._raghavan_roth import (\n"
            "            _prime_aot as _ssik_rr_prime_aot,\n"
            "        )\n\n"
            "        # AOT-baked Raghavan-Roth derivations (#320). Each function below\n"
            "        # is the verbatim numpy source that ``sp.lambdify`` emitted at build\n"
            "        # time, extracted via inspect.getsource. Module-import only parses\n"
            "        # the source (~80ms total on Kassow KR810) -- no sympy at runtime,\n"
            "        # no re-lambdify. Numerical output is bit-identical to the previous\n"
            "        # blob-prime path. Wheel-compressed artifact also shrinks vs the\n"
            "        # blob path because text gzips better than already-zlib bytes.\n\n"
            f"{aot_funcs_block}\n\n"
            "        _AOT_PRIME_DATA = (\n"
            f"{aot_entries_block}\n"
            "        )\n\n"
            "        for _aot_entry in _AOT_PRIME_DATA:\n"
            "            _ssik_rr_prime_aot(*_aot_entry)\n"
        )
    else:
        # Arms whose dispatch cache is entirely tier-0 don't need priming
        # (e.g. Franka pre-#320 -- all 16 samples route to ``reversed:spherical``).
        # Skip the block entirely to keep artifacts byte-stable.
        rr_prime_block = ""

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
        ){rr_prime_block}


        def _solve_algebraic(
            T_target, *, max_solutions=None, q_seed=None, respect_limits=False,
        ):
            \"\"\"7R IK candidates via joint-locking + inner 6R sweep.

            Routes to ssik.solvers.jointlock.seven_r.solve with the baked
            KinBody, lock_idx, lock-sample schedule, and dispatch cache.
            ``max_solutions``, ``q_seed``, and ``respect_limits`` are
            forwarded so the lock-sweep can short-circuit on the first
            in-limits valid IK (#238 review). Returns
            ``list[list[float]]`` of length-7 q-vectors.
            \"\"\"
            sub_solutions, _is_ls = _ssik_seven_r.solve(
                _KB, T_target,
                lock_idx=_LOCK_IDX,
                lock_samples=_LOCK_SAMPLES,
                dispatch_cache=_DISPATCH_CACHE,
                max_solutions=max_solutions, q_seed=q_seed,
                respect_limits=respect_limits,
            )
            return [list(s.q) for s in sub_solutions]
        """
    )
