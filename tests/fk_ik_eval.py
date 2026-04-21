"""Numerical evaluator for an IKFast ``SolverIKChainTranslation3D`` chaintree.

Used only by the slow correctness-gate tests in ``test_kinbody_ur5.py``;
not production code. The real consumer of a chaintree is the C++ codegen
(``ikfast_generator_cpp.py`` → compiled via the #10 build pipeline). Pure
sympy evaluation is ~1000x slower but correct-by-construction — perfect
for validating a single roundtrip before we have the compiled pipeline.

Supported node types (added incrementally as real chaintrees surfaced them):
    - SolverCheckZeros, SolverBranchConds — conditional branching.
    - SolverPolynomialRoots — real roots of a polynomial in a joint variable.
    - SolverSolution — direct / cos / sin joint-value expressions.
    - SolverStoreSolution — emit a candidate from current bindings.
    - SolverSequence — chain multiple sub-trees.
    - SolverRotation — rewrite the r_ij target bindings via a 3x3 expression
      matrix before walking a nested tree (used in Pieper-style
      spherical-wrist decompositions).

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
    """Return the trig pre-computations the generated C++ keeps as a cache
    and that solver expressions reference via symbols like ``cj0`` / ``sj0``.

    Transform6D additionally uses the half-tangent (``htj``) and tangent
    (``tj``) forms; Translation3D only ever touches ``cj`` / ``sj`` but
    binding them anyway is harmless (unused bindings are ignored by subs).
    """
    out: dict[Symbol, float] = {
        Symbol(joint_name): value,
        Symbol("c" + joint_name): float(np.cos(value)),
        Symbol("s" + joint_name): float(np.sin(value)),
        Symbol("ht" + joint_name): float(np.tan(value / 2.0)),
        Symbol("t" + joint_name): float(np.tan(value)),
    }
    return out


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

    if cls == "SolverRotation":
        # The Pieper-style decomposition emits this after the first three
        # joints (arm positioning) have been solved: ``node.T`` is a 3x3
        # sympy Matrix re-expressing the remaining orientation problem in a
        # sub-frame where the wrist-joint equations simplify. The C++
        # generator (``ikfast_generator_cpp.generateRotation``) writes the
        # values into **new_r_ij** symbols (NOT ``r_ij``) and leaves the
        # outer r_ij values unchanged. The nested jointtree references
        # ``new_r_ij``. Do NOT touch ``r_ij`` here.
        new_bindings = dict(bindings)
        for i in range(3):
            for j in range(3):
                new_bindings[Symbol(f"new_r{i}{j}")] = _eval_float(node.T[i, j], bindings)
        _eval_tree(list(node.jointtree) + rest, new_bindings, emit)
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


def eval_chaintree_6d(
    chaintree: Any,
    q_free: dict[str, float],
    target_pose: np.ndarray,
) -> list[dict[str, float]]:
    """Numerically evaluate an ``SolverIKChainTransform6D`` chaintree.

    :param q_free: ``{joint_name: value}`` for any free joints (empty for
        standard 6-DOF 6D solve).
    :param target_pose: desired end-effector pose as a 4x4 homogeneous
        transform in the **ikfast frame convention** (i.e., what
        ``chaintree.Tfk`` would produce). See :func:`ee_rest_rotation` for
        translating between this and a URDF-native FK.
    :returns: list of candidate solutions, each a full ``{joint_name: value}``
        dict covering solved + free joints.
    """
    if target_pose.shape != (4, 4):
        raise ValueError(f"target_pose must be 4x4, got {target_pose.shape}")

    # Bind the target as the Tee symbols (r00..r22 + px/py/pz).
    bindings: dict[Symbol, float] = {}
    for i in range(3):
        for j in range(3):
            bindings[Symbol(f"r{i}{j}")] = float(target_pose[i, j])
    bindings[Symbol("px")] = float(target_pose[0, 3])
    bindings[Symbol("py")] = float(target_pose[1, 3])
    bindings[Symbol("pz")] = float(target_pose[2, 3])

    for name, val in q_free.items():
        bindings.update(_trig_bindings(name, val))

    if chaintree.dictequations:
        for sym, expr in chaintree.dictequations:
            bindings[sym] = _eval_float(expr, bindings)

    # The chaintree's Tee is a 4x4 sympy Matrix re-expressing the target in
    # the frame the downstream jointtree expects. Evaluate it numerically
    # and rewrite the r_ij / p_xyz bindings. Pattern mirrors
    # ikfast_generator_cpp.generateChain (ikfast_generator_cpp.py:672).
    new_rotation = [
        [_eval_float(chaintree.Tee[4 * i + j], bindings) for j in range(3)] for i in range(3)
    ]
    new_translation = [_eval_float(chaintree.Tee[4 * i + 3], bindings) for i in range(3)]
    for i in range(3):
        for j in range(3):
            bindings[Symbol(f"r{i}{j}")] = new_rotation[i][j]
    bindings[Symbol("px")] = new_translation[0]
    bindings[Symbol("py")] = new_translation[1]
    bindings[Symbol("pz")] = new_translation[2]

    emit: list[dict[str, float]] = []
    _eval_tree(list(chaintree.jointtree), bindings, emit)
    return emit


def ee_rest_rotation(kinbody: Any) -> np.ndarray:
    """Return the end-effector rotation matrix at q=0 for the given KinBody.

    Used to translate between ikfast's internal FK convention (URDF-native:
    cumulative `rpy` values on joint origins appear in the rest pose) and
    EAIK's convention (FK=I at q=0). For a target pose ``T_ours`` you want
    ikfast to solve, the equivalent target for EAIK is
    ``T_eaik[:3,:3] = T_ours[:3,:3] @ R_rest.T``. Position is unchanged.

    This is purely a convention bridge — the underlying q solutions are the
    same.
    """
    T = np.eye(4, dtype=np.float64)
    for j in kinbody.joints:
        # At q=0 the joint rotation is identity, so the joint contributes
        # T_left @ T_right to the cumulative transform.
        T = T @ j.T_left @ j.T_right
    return T[:3, :3].copy()
