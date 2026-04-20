"""Numerical evaluator for an IKFast ``SolverIKChainTranslation3D`` chaintree.

Used only by the slow correctness-gate tests in ``test_kinbody_ur5.py``;
not production code. The real consumer of a chaintree is the C++ codegen
(``ikfast_generator_cpp.py`` → compiled via the #10 build pipeline). Pure
sympy evaluation is ~1000x slower but correct-by-construction — perfect
for validating a single roundtrip before we have the compiled pipeline.

Supported node types (added incrementally as UR5's Translation3D chaintree
surfaced them):
    SolverCheckZeros, SolverPolynomialRoots, SolverSolution,
    SolverStoreSolution, SolverSequence.

Any other node type raises ``NotImplementedError`` with the class name;
extend here when a new robot/solve-mode needs it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy
from sympy import Symbol

# Default tolerances mirror the upstream C++ generator's postcheck thresholds.
_ZERO_TOL = 1e-8
_RANGE_TOL = 1e-8


def _eval_float(expr: Any, bindings: dict[Symbol, float]) -> float:
    """Substitute bindings into ``expr`` and return a plain Python float."""
    if isinstance(expr, (int, float)):
        return float(expr)
    result = sympy.sympify(expr).subs(bindings).evalf()
    try:
        return float(result)
    except (TypeError, ValueError) as err:
        raise ValueError(f"could not convert {expr!r} → {result!r} to float") from err


def _trig_bindings(joint_name: str, value: float) -> dict[Symbol, float]:
    """Return ``{j: v, cj: cos(v), sj: sin(v)}`` — the trig pre-computations
    that the generated code keeps as a cache and that solver expressions
    rely on via symbols like ``cj0`` / ``sj0``.
    """
    return {
        Symbol(joint_name): value,
        Symbol("c" + joint_name): float(np.cos(value)),
        Symbol("s" + joint_name): float(np.sin(value)),
    }


def _joint_values_from_solution_node(node: Any, bindings: dict[Symbol, float]) -> list[float]:
    """Enumerate candidate values for the joint that ``node`` solves for.

    Handles the three upstream representations:
    - ``jointeval``: direct value expressions, one candidate each
    - ``jointevalcos`` + ``jointevalsin`` (paired, same length): use
      ``atan2(sin_expr, cos_expr)`` for a unique candidate per index
    - ``jointevalcos`` alone: ``±acos(expr)`` (2 candidates per entry)
    - ``jointevalsin`` alone: ``asin(expr)`` and ``pi - asin(expr)``
    """
    candidates: list[float] = []

    if node.jointeval is not None:
        for e in node.jointeval:
            try:
                candidates.append(_eval_float(e, bindings))
            except ValueError:
                continue

    cos_exprs = node.jointevalcos
    sin_exprs = node.jointevalsin

    if cos_exprs is not None and sin_exprs is not None and len(cos_exprs) == len(sin_exprs):
        for c, s in zip(cos_exprs, sin_exprs, strict=True):
            try:
                cv = _eval_float(c, bindings)
                sv = _eval_float(s, bindings)
            except ValueError:
                continue
            candidates.append(float(np.arctan2(sv, cv)))
    else:
        if cos_exprs is not None:
            for c in cos_exprs:
                try:
                    cv = _eval_float(c, bindings)
                except ValueError:
                    continue
                if -1 - _RANGE_TOL <= cv <= 1 + _RANGE_TOL:
                    cv = max(-1.0, min(1.0, cv))
                    a = float(np.arccos(cv))
                    candidates.extend([a, -a])
        if sin_exprs is not None:
            for s in sin_exprs:
                try:
                    sv = _eval_float(s, bindings)
                except ValueError:
                    continue
                if -1 - _RANGE_TOL <= sv <= 1 + _RANGE_TOL:
                    sv = max(-1.0, min(1.0, sv))
                    a = float(np.arcsin(sv))
                    candidates.extend([a, float(np.pi) - a])

    return candidates


def _poly_roots(node: Any, bindings: dict[Symbol, float]) -> list[float]:
    """Real roots of the numerical polynomial produced by substituting
    bindings into ``node.poly``. Applies upstream's post-range / post-zero
    filters where provided.
    """
    poly_expr = sympy.expand(sympy.sympify(node.poly.as_expr()).subs(bindings))
    gen = node.poly.gens[0]
    try:
        numeric_poly = sympy.Poly(poly_expr, gen)
    except sympy.PolynomialError:
        return []
    coeffs = [float(c) for c in numeric_poly.all_coeffs()]
    if len(coeffs) < 2 or all(abs(c) < _ZERO_TOL for c in coeffs):
        return []
    roots = np.roots(coeffs)
    real_roots: list[float] = []
    for r in roots:
        if abs(r.imag) < _ZERO_TOL:
            real_roots.append(float(r.real))

    # Apply post-checks from the upstream node: reject roots that violate
    # range constraints or zero-checks. These are safety filters IKFast uses
    # to discard spurious solutions.
    if node.postcheckforrange is not None:
        filtered: list[float] = []
        for root in real_roots:
            root_bindings = {**bindings, gen: root}
            ok = True
            for check_expr in node.postcheckforrange:
                try:
                    v = _eval_float(check_expr, root_bindings)
                except ValueError:
                    ok = False
                    break
                if not (-1 - _RANGE_TOL <= v <= 1 + _RANGE_TOL):
                    ok = False
                    break
            if ok:
                filtered.append(root)
        real_roots = filtered

    return real_roots


def _eval_tree(
    tree: list[Any],
    bindings: dict[Symbol, float],
    emit: list[dict[str, float]],
) -> None:
    """Walk ``tree`` sequentially with ``bindings`` accumulating.

    Fan-out happens wherever a node produces multiple candidate values.
    Leaves push a ``{joint_name: value}`` dict into ``emit``.
    """
    if not tree:
        return
    node = tree[0]
    rest = tree[1:]
    cls = type(node).__name__

    if cls == "SolverCheckZeros":
        thresh = node.thresh if node.thresh is not None else _ZERO_TOL
        if node.anycondition:
            is_zero = any(abs(_eval_float(eq, bindings)) <= thresh for eq in node.jointcheckeqs)
        else:
            is_zero = all(abs(_eval_float(eq, bindings)) <= thresh for eq in node.jointcheckeqs)
        branch = node.zerobranch if is_zero else node.nonzerobranch
        _eval_tree(list(branch) + rest, bindings, emit)
        return

    if cls == "SolverBranchConds":
        for check_eqs, branch, _extra in node.jointbranches:
            thresh = node.thresh
            if all(abs(_eval_float(eq, bindings)) <= thresh for eq in check_eqs):
                _eval_tree(list(branch) + rest, bindings, emit)
                return
        # No branch matched — dead end
        return

    if cls == "SolverPolynomialRoots":
        joint = node.jointname
        for root in _poly_roots(node, bindings):
            # The poly's generator is usually `s<joint>` or a half-tan
            # variable. For UR5 Translation3D we typically get `sj<n>` or
            # `cj<n>` style. `node.jointeval` then expresses the joint in
            # terms of that generator.
            gen = node.poly.gens[0]
            root_bindings = {**bindings, gen: root}
            if node.jointeval is not None:
                for joint_expr in node.jointeval:
                    try:
                        jv = _eval_float(joint_expr, root_bindings)
                    except ValueError:
                        continue
                    downstream_bindings = {
                        **bindings,
                        gen: root,
                        **_trig_bindings(joint, jv),
                    }
                    _eval_tree(rest, downstream_bindings, emit)
            else:
                # Generator *is* the joint value (rare for sin/cos style)
                downstream_bindings = {**bindings, **_trig_bindings(joint, root)}
                _eval_tree(rest, downstream_bindings, emit)
        return

    if cls == "SolverSolution":
        for jv in _joint_values_from_solution_node(node, bindings):
            downstream_bindings = {**bindings, **_trig_bindings(node.jointname, jv)}
            _eval_tree(rest, downstream_bindings, emit)
        return

    if cls == "SolverStoreSolution":
        # Emit a solution from the current bindings
        solution: dict[str, float] = {}
        for joint_sym in node.alljointvars:
            val = bindings.get(joint_sym)
            if val is None:
                return  # incomplete solution — skip
            solution[joint_sym.name] = float(val)
        emit.append(solution)
        _eval_tree(rest, bindings, emit)
        return

    if cls == "SolverSequence":
        for subtree in node.jointtrees:
            _eval_tree(list(subtree) + rest, bindings, emit)
        return

    raise NotImplementedError(f"chaintree walker does not support {cls}")


def eval_chaintree(
    chaintree: Any,
    q_free: dict[str, float],
    target_pos: tuple[float, float, float],
) -> list[dict[str, float]]:
    """Numerically evaluate an ``SolverIKChainTranslation3D`` chaintree.

    :param q_free: ``{joint_name: value}`` for the free joints (by name,
        matching the symbols the solver uses, e.g. ``"j3"``, ``"j4"``,
        ``"j5"``).
    :param target_pos: desired end-effector position ``(x, y, z)``.
    :returns: list of candidate solutions, each a full ``{joint_name: value}``
        dict covering both solved and free joints. Empty if no branch yielded
        a solution (either a degenerate target or a solver bug).
    """
    # Seed the bindings with the original EE-position symbols and the free
    # joints (as value + cos + sin).
    px, py, pz = [Symbol(n) for n in ("px", "py", "pz")]
    bindings: dict[Symbol, float] = {
        px: float(target_pos[0]),
        py: float(target_pos[1]),
        pz: float(target_pos[2]),
    }
    for name, val in q_free.items():
        bindings.update(_trig_bindings(name, val))

    # Apply any chain-level dictequations (variable substitutions) first.
    if chaintree.dictequations:
        for sym, expr in chaintree.dictequations:
            bindings[sym] = _eval_float(expr, bindings)

    # The chaintree's ``Pee`` re-expresses the target position in the frame
    # the downstream jointtree expects. Evaluate it with the seeded bindings
    # and write the result back into the px/py/pz symbols.
    new_pee = [_eval_float(chaintree.Pee[i], bindings) for i in range(3)]
    bindings[px] = new_pee[0]
    bindings[py] = new_pee[1]
    bindings[pz] = new_pee[2]

    emit: list[dict[str, float]] = []
    _eval_tree(list(chaintree.jointtree), bindings, emit)
    return emit
