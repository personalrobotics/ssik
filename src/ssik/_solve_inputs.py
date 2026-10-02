"""The ``solve()`` input contract, checked once at the public boundary (#574).

Every public solve entry point -- each artifact's ``solve``, ``Manipulator.solve``
and the chart entry points -- calls :func:`check_solve_inputs` (or
:func:`check_pose`) before it picks a backend, so a malformed call raises the
same exception whichever backend would have run, and the native extension only
ever sees a well-formed target and seed. ``docs/api.md`` ("Input validation")
is the normative statement of the contract; this module is its implementation.

Cost matters here: a seeded native solve takes about 10 us. The common case (a
C-contiguous float64 target and seed, an ``int`` cap) passes on attribute
checks, and the numeric checks on the target and seed run in one native call
(``ssik._ssik_native.input_defect``) when the extension is present.
:func:`_input_defect_py` computes the same quantities with numpy otherwise.
"""

from __future__ import annotations

import math
import numbers
import operator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import TolerancePolicy

_F64 = np.dtype(np.float64)
_REAL_KINDS = frozenset("fiu")

# A target is rejected as non-rigid when ||R^T R - I||_F exceeds this many times
# policy.subproblem_numerical: past it, no rigid FK(q) is within the acceptance
# gate of the target, so it could only ever return []. docs/api.md ("Input
# validation") derives the factor.
_ORTHO_FACTOR = 3.0
# Neither check is tighter than this, whatever the policy: evaluating R^T R
# alone rounds by ~1e-15, so a policy tighter than round-off must not turn a
# rigid target's own rounding into a rejection.
_ROUNDOFF_FLOOR = 1e-12

# Defect codes, shared with the native kernel (ssik._ssik_native.input_defect).
_OK, _NONFINITE, _NOT_ORTHONORMAL, _REFLECTION, _BOTTOM_ROW, _SEED_NONFINITE = range(6)

# The native extension's ``int`` parameters are 32-bit: a cap at or above this
# is no cap, and is passed as -1 (the "no cap" sentinel).
INT32_MAX = 2**31 - 1


def as_real_array(x: Any, name: str) -> NDArray[np.float64]:
    """``x`` as a C-contiguous float64 array, or TypeError if it is not real numbers.

    Real numeric dtypes (float, int, unsigned) are converted exactly as numpy
    does. Bool, complex, string, bytes, datetime and structured data are
    rejected: converting them would silently invent or drop information. An
    object array (a list mixing numpy and Python scalars, a sympy matrix) is
    accepted when every element is a real number.
    """
    if type(x) is np.ndarray and x.dtype is _F64 and x.flags.c_contiguous:
        return x
    try:
        arr = np.asarray(x)
    except ValueError as e:  # ragged nested sequences
        raise ValueError(f"{name} must be a rectangular array of real numbers: {e}") from None
    kind = arr.dtype.kind
    if kind == "O":
        flat = arr.ravel()
        if not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in flat):
            raise TypeError(f"{name} must contain real numbers only, got an object array")
    elif kind not in _REAL_KINDS:
        raise TypeError(f"{name} must be an array of real numbers, got dtype {arr.dtype}")
    return np.ascontiguousarray(arr, dtype=np.float64)


_limits_cache: dict[float, tuple[float, float]] = {}


def _limits(tol: float) -> tuple[float, float]:
    """The ``(rotation, bottom row)`` thresholds for a policy tolerance."""
    lim = _limits_cache.get(tol)
    if lim is None:
        lim = (max(_ORTHO_FACTOR * tol, _ROUNDOFF_FLOOR), max(tol, _ROUNDOFF_FLOOR))
        _limits_cache[tol] = lim
    return lim


def _input_defect_py(
    T: NDArray[np.float64], ortho_lim: float, row_lim: float, q_seed: Any = None
) -> int:
    """Reference implementation of the numeric checks; the native kernel
    computes the same quantities in the same order."""
    if not np.isfinite(T).all():
        return _NONFINITE
    R = T[:3, :3]
    if float(np.linalg.norm(R.T @ R - np.eye(3))) > ortho_lim:
        return _NOT_ORTHONORMAL
    if float(np.linalg.det(R)) <= 0.0:
        return _REFLECTION
    if float(np.linalg.norm(T[3] - (0.0, 0.0, 0.0, 1.0))) > row_lim:
        return _BOTTOM_ROW
    if q_seed is not None and not all(map(math.isfinite, q_seed.tolist())):
        return _SEED_NONFINITE
    return _OK


def _native_kernel() -> Any:
    from ssik._native import _load_ext

    ext = _load_ext()
    return getattr(ext, "input_defect", None) if ext is not None else None


_UNSET: Any = object()
_kernel: Any = _UNSET


def _resolve_kernel() -> Any:
    global _kernel
    _kernel = _native_kernel()
    return _kernel


def _input_defect(
    T: NDArray[np.float64], ortho_lim: float, row_lim: float, q_seed: Any = None
) -> int:
    """The numeric checks: the native kernel when the extension is present,
    else the numpy reference."""
    kernel = _kernel if _kernel is not _UNSET else _resolve_kernel()
    if kernel is None:
        return _input_defect_py(T, ortho_lim, row_lim, q_seed)
    return int(kernel(T, ortho_lim, row_lim, q_seed))


def _input_error(T: NDArray[np.float64], code: int, ortho_lim: float, name: str) -> ValueError:
    if code == _SEED_NONFINITE:
        return ValueError("q_seed must be finite (no NaN or inf)")
    if code == _NONFINITE:
        return ValueError(f"{name} must be finite (no NaN or inf)")
    R = T[:3, :3]
    if code == _NOT_ORTHONORMAL:
        defect = float(np.linalg.norm(R.T @ R - np.eye(3)))
        return ValueError(
            f"{name} is not a rigid transform: its rotation block has "
            f"||R^T R - I||_F = {defect:.3g}, above {ortho_lim:.3g} "
            f"(3 x policy.subproblem_numerical), so no configuration can reach it"
        )
    if code == _REFLECTION:
        det = float(np.linalg.det(R))
        return ValueError(
            f"{name} is not a rigid transform: its rotation block has determinant "
            f"{det:.3g} (a reflection, not a rotation)"
        )
    return ValueError(f"{name} must have bottom row [0, 0, 0, 1], got {T[3].tolist()}")


def _pose_array(T_target: Any, name: str) -> NDArray[np.float64]:
    T = as_real_array(T_target, name)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {T.shape}")
    return T


def _joint_array(q: Any, dof: int, name: str) -> NDArray[np.float64]:
    arr = as_real_array(q, name)
    if arr.shape != (dof,):
        raise ValueError(f"{name} must have shape ({dof},), got {arr.shape}")
    return arr


def check_pose(
    T_target: Any, *, policy: TolerancePolicy, name: str = "T_target"
) -> NDArray[np.float64]:
    """``T_target`` as a C-contiguous float64 ``(4, 4)`` rigid transform.

    :raises TypeError: if it is not an array of real numbers.
    :raises ValueError: if its shape is not ``(4, 4)``, an entry is NaN or inf,
        or it is not a rigid transform within the tolerance documented in
        ``docs/api.md`` ("Input validation").
    """
    T = _pose_array(T_target, name)
    ortho_lim, row_lim = _limits(policy.subproblem_numerical)
    code = _input_defect(T, ortho_lim, row_lim)
    if code:
        raise _input_error(T, code, ortho_lim, name)
    return T


def check_joints(q: Any, dof: int, name: str, *, finite: bool = True) -> NDArray[np.float64]:
    """``q`` as a C-contiguous float64 ``(dof,)`` joint vector, finite unless
    ``finite=False`` (forward kinematics propagates a NaN instead).

    :raises TypeError: if it is not an array of real numbers.
    :raises ValueError: if its shape is not ``(dof,)`` or an entry is NaN or inf.
    """
    arr = _joint_array(q, dof, name)
    if finite and not all(map(math.isfinite, arr.tolist())):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return arr


def check_max_solutions(max_solutions: Any) -> int | None:
    """``None`` or a positive integer (any type with ``__index__``).

    ``0`` and negative caps are rejected: the backends read them differently
    (native took a negative cap as no cap, Python as an empty result, and the
    jointlock solvers raised on ``0``), and an empty list must mean "no
    solution", not "asked for none" (#575).

    :raises TypeError: for a non-integer (``2.5``, ``"3"``).
    :raises ValueError: for an integer below 1.
    """
    if max_solutions is None:
        return None
    try:
        k = operator.index(max_solutions)
    except TypeError:
        raise TypeError(
            f"max_solutions must be None or an integer, got {type(max_solutions).__name__}"
        ) from None
    if k < 1:
        raise ValueError(f"max_solutions must be >= 1 or None, got {k}")
    return k


def _check_refinement_max_iters(n: Any) -> None:
    try:
        iters = operator.index(n)
    except TypeError:
        raise TypeError(
            f"refinement_max_iters must be an integer, got {type(n).__name__}"
        ) from None
    if iters < 0:
        raise ValueError(f"refinement_max_iters must be >= 0, got {iters}")


def _check_seed_tolerance(seed_tolerance: Any, q_seed: Any) -> None:
    if q_seed is None:
        raise ValueError("seed_tolerance requires q_seed")
    if not isinstance(seed_tolerance, numbers.Real) or isinstance(seed_tolerance, bool):
        raise TypeError(
            f"seed_tolerance must be None or a real number, got {type(seed_tolerance).__name__}"
        )
    if seed_tolerance != seed_tolerance:
        raise ValueError("seed_tolerance must not be NaN")


def check_solve_inputs(
    T_target: Any,
    q_seed: Any,
    dof: int,
    *,
    max_solutions: Any,
    seed_tolerance: Any,
    refinement_max_iters: Any,
    policy: TolerancePolicy,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, int | None]:
    """Validate and normalise one ``solve()`` call's inputs.

    Returns ``(T, q_seed, max_solutions)``: the target as a float64 ``(4, 4)``
    rigid transform, the seed as a float64 ``(dof,)`` vector (or ``None``), and
    ``max_solutions`` as an ``int`` (or ``None``). ``seed_tolerance`` and
    ``refinement_max_iters`` are checked for the values the two backends read
    differently (a NaN tolerance, a negative or non-integer iteration count);
    the remaining option rules (``respect_limits``, ``seed_metric`` without a
    seed, booleans as integers, ``policy``) are #575's.
    """
    # Each option's common value is decided by the first test on its line: a
    # seeded native solve takes about 10 us, and this runs before every one.
    if seed_tolerance is not None:
        _check_seed_tolerance(seed_tolerance, q_seed)
    if type(refinement_max_iters) is not int or refinement_max_iters < 0:
        _check_refinement_max_iters(refinement_max_iters)
    if max_solutions is not None and (type(max_solutions) is not int or max_solutions < 1):
        max_solutions = check_max_solutions(max_solutions)
    tol = policy.subproblem_numerical
    lim = _limits_cache.get(tol) or _limits(tol)
    # Fast path: the native kernel itself accepts only a C-contiguous float64
    # (4, 4) target and 1-D seed, so passing it is the dtype, layout and rank
    # check; anything else takes the converting path below.
    kernel = _kernel if _kernel is not _UNSET else _resolve_kernel()
    if (
        kernel is not None
        and type(T_target) is np.ndarray
        and (q_seed is None or (type(q_seed) is np.ndarray and q_seed.shape == (dof,)))
    ):
        try:
            code = kernel(T_target, lim[0], lim[1], q_seed)
        except ValueError:
            code = -1
        if code == 0:
            return T_target, q_seed, max_solutions
    T = _pose_array(T_target, "T_target")
    seed = None if q_seed is None else _joint_array(q_seed, dof, "q_seed")
    code = _input_defect(T, lim[0], lim[1], seed)
    if code:
        raise _input_error(T, code, lim[0], "T_target")
    return T, seed, max_solutions
