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
from ssik.codegen._compose.general_6r import _render_pq_combined_builder
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import _cached_best_leftvar, _cached_derivation
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

    Note: the ``prime_derivation`` import for cached-RR priming (#210)
    is added inline in :func:`compose` only when the arm has at least
    one eligible non-tier-0 inner sample. Arms with all-tier-0 dispatch
    (e.g. Franka) keep their artifact byte-identical to pre-#210.
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
    # Per non-tier-0 sample: collect (alpha, a, d, linearity, builder_source).
    # The CSE'd ``_build_pq_sample_<i>`` source is generated at codegen time
    # so module import is sympy-free (#210 Phase 2).
    rr_bake_entries: list[
        tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int, str]
    ] = []
    for q_lock in samples:
        sub_kb = _lock_joint(kb, lock_idx, float(q_lock))
        _, name = _topology_rank(sub_kb, policy)
        dispatch.append(name)
        # If the inner sample would route through a non-tier-0 path, bake
        # the per-DH RR derivation into the artifact as plain numpy
        # source. Cached at runtime; ~1 ms per call vs HP's 13-35 ms.
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
            # Run the symbolic preprocessing ONCE at codegen time. The
            # output's symbolic matrices feed _render_pq_combined_builder
            # which emits a CSE'd plain-numpy function.
            _, _, _, _, meta = _cached_derivation(
                alpha, a, d, linearity_joint=linearity, apply_so3=False
            )
            import sympy as sp

            sym_p_sin = meta["_sym_p_sin"]
            sym_p_cos = meta["_sym_p_cos"]
            sym_p_one = meta["_sym_p_one"]
            sym_q = meta["_sym_q"]
            t_syms = meta["_sym_t_target"]
            assert isinstance(sym_p_sin, sp.Matrix)
            assert isinstance(sym_p_cos, sp.Matrix)
            assert isinstance(sym_p_one, sp.Matrix)
            assert isinstance(sym_q, sp.Matrix)
            builder_source = _render_pq_combined_builder(
                sym_p_sin=sym_p_sin,
                sym_p_cos=sym_p_cos,
                sym_p_one=sym_p_one,
                sym_q=sym_q,
                t_syms=t_syms,  # type: ignore[arg-type]
            )
            rr_bake_entries.append((alpha, a, d, linearity, builder_source))

    samples_repr = ", ".join(repr(float(s)) for s in samples)
    dispatch_repr = ",\n            ".join(repr(name) for name in dispatch)

    if rr_bake_entries:
        # Per-sample CSE'd builder + insert_derivation registration.
        # Each builder is renamed from ``_build_pq_matrices`` (the default
        # produced by _render_pq_combined_builder for the top-level RR
        # codegen) to ``_build_pq_jointlock_sample_<i>`` so multiple
        # builders can coexist in the same artifact.
        builder_blocks: list[str] = []
        register_lines: list[str] = []
        for i, (alpha, a, d, lin, src) in enumerate(rr_bake_entries):
            renamed = src.replace("def _build_pq_matrices(", f"def _build_pq_jointlock_sample_{i}(")
            # Indent and de-doctstring conflict: the rendered source uses
            # ``# --- inlined ...`` headers and a docstring. Strip the
            # outer header so it doesn't repeat per sample.
            builder_blocks.append(
                f"# --- jointlock sample {i}: alpha={alpha!r}, "
                f"a={a!r}, d={d!r}, linearity={lin} ---\n" + renamed
            )
            register_lines.append(
                f"_ssik_rr_insert(\n"
                f"    {alpha!r},\n    {a!r},\n    {d!r},\n"
                f"    linearity_joint={lin}, apply_so3=False,\n"
                f"    build_pq_combined=_build_pq_jointlock_sample_{i},\n"
                f"    metadata=_ssik_rr_meta_{i},\n"
                f")"
            )

        # Per-sample metadata (lookup keys for jointlock's
        # primed_linearity_for_dh map). Re-derived from the same
        # _cached_derivation call so the artifact's metadata matches the
        # runtime expectation.
        meta_lines: list[str] = []
        for i, (alpha, a, d, lin, _src) in enumerate(rr_bake_entries):
            _, _, _, _, meta = _cached_derivation(alpha, a, d, linearity_joint=lin, apply_so3=False)
            meta_lines.append(
                f"_ssik_rr_meta_{i} = {{\n"
                f"    'linearity_joint': {meta['linearity_joint']!r},\n"
                f"    'left_bilinear': {meta['left_bilinear']!r},\n"
                f"    'right_bilinear': {meta['right_bilinear']!r},\n"
                f"    'drop_joint': {meta['drop_joint']!r},\n"
                f"    'apply_so3': {meta['apply_so3']!r},\n"
                f"}}"
            )

        # Also populate _PRIMED_LINEARITY_MAP so jointlock dispatch can
        # look up the baked linearity choice per DH (avoiding the
        # runtime _cached_best_leftvar AE-3 probe).
        prime_map_lines = []
        for alpha, a, d, lin, _src in rr_bake_entries:
            prime_map_lines.append(
                f"_PRIMED_LINEARITY_MAP[(\n"
                f"    {alpha!r},\n    {a!r},\n    {d!r},\n)] = ({lin}, False)"
            )

        rr_bake_block = (
            "\n\n# --- Cached-RR baked builders (#210 Phase 2) ---\n"
            "# CSE'd plain-numpy matrix builders generated at ssik build time.\n"
            "# At module import we register them into Raghavan-Roth's derivation\n"
            "# cache via insert_derivation; subsequent jointlock dispatches use\n"
            "# them directly (~1 ms per inner IK) instead of HP / two_parallel\n"
            "# (~13-260 ms). 12-25x speedup over the URDF path.\n"
            "from ssik.solvers.ikgeo._raghavan_roth import (\n"
            "    _PRIMED_LINEARITY_MAP,\n"
            "    insert_derivation as _ssik_rr_insert,\n"
            ")\n\n"
            + "\n\n".join(builder_blocks)
            + "\n"
            + "\n".join(meta_lines)
            + "\n\n"
            + "\n".join(register_lines)
            + "\n\n"
            + "\n".join(prime_map_lines)
            + "\n"
        )
    else:
        # Arms whose dispatch cache is entirely tier-0 don't need priming
        # (e.g. Franka -- all 16 samples route to ``reversed:spherical``).
        # Skip the block entirely to keep artifacts byte-stable with
        # pre-#210.
        rr_bake_block = ""

    template = textwrap.dedent(
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
        _RR_BAKE_PLACEHOLDER

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
    # The CSE'd RR builders are emitted at module top-level (no indent),
    # so they need to splice in AFTER textwrap.dedent runs. We use a
    # sentinel ``_RR_BAKE_PLACEHOLDER`` line in the template; replacing
    # it post-dedent injects the unindented block in place. When there
    # is no bake (Franka et al.), we replace the placeholder line with
    # an empty line so the byte layout matches pre-#210 artifacts.
    if rr_bake_block:
        return template.replace("_RR_BAKE_PLACEHOLDER\n", rr_bake_block)
    return template.replace("_RR_BAKE_PLACEHOLDER\n", "\n")
