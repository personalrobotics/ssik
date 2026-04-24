"""1D zero-finding primitives for IK-Geo's univariate-search solvers.

Private; shared between ``two_intersecting`` and ``two_parallel`` (Round 2
of the solver roster in issue #53).

Ported clean-room from the BSD-3 [ik-geo Rust reference][ikgeo]'s
``auxiliary::search_1d`` (Elias & Wen, arXiv:2211.05737). Retains the
bracketing + false-position structure so the port is auditable against
the source.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/auxiliary.rs
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["search_1d"]

_CROSS_THRESHOLD = 0.1
_FZ_ITERATIONS = 100
_FZ_EPSILON = 1e-5


def search_1d(
    f: Callable[[float], NDArray[np.float64]],
    left: float,
    right: float,
    initial_samples: int,
) -> list[tuple[float, int]]:
    """Find zeros of each component of a vector-valued function ``f`` on
    ``[left, right]``.

    ``f(x)`` returns a length-N vector; ``search_1d`` detects sign changes
    in each component independently over a uniform sampling of
    ``initial_samples`` intervals and refines each bracketed zero via
    false-position iteration.

    :returns: list of ``(x_zero, component_index)`` pairs. Same API as
        the Rust reference: the caller uses ``component_index`` to pick
        which branch of the multi-valued inner solver (e.g. SP5 triple)
        produced the bracketed zero.
    """
    zeros: list[tuple[float, int]] = []
    delta = (right - left) / initial_samples

    last_v = f(left)
    x = left + delta

    for _ in range(initial_samples):
        v = f(x)
        for i, (y, last_y) in enumerate(zip(v, last_v, strict=True)):
            y_f = float(y)
            last_y_f = float(last_y)
            # Sign change AND both bracketed values are below
            # the cross-threshold (IK-Geo's heuristic to avoid chasing
            # ``NAN``/``INFINITY`` discontinuities from SP5 returning no
            # real solutions on a subinterval).
            if (y_f < 0.0) != (last_y_f < 0.0) and (
                abs(y_f) < _CROSS_THRESHOLD and abs(last_y_f) < _CROSS_THRESHOLD
            ):
                z = _find_zero(f, x - delta, x, i)
                if z is not None:
                    zeros.append((z, i))
        last_v = v
        x += delta

    return zeros


def _find_zero(
    f: Callable[[float], NDArray[np.float64]],
    left: float,
    right: float,
    i: int,
) -> float | None:
    """False-position (regula falsi) refinement of a bracketed zero of
    component ``i`` of ``f`` on ``[left, right]``. Returns ``None`` if a
    sampled value is non-finite (indicating the inner solver stopped
    producing real solutions mid-interval)."""
    x_left, x_right = left, right
    y_left = float(f(x_left)[i])
    y_right = float(f(x_right)[i])

    for _ in range(_FZ_ITERATIONS):
        delta = y_right - y_left
        if abs(delta) < _FZ_EPSILON:
            break
        x_0 = x_left - y_left * (x_right - x_left) / delta
        y_0 = float(f(x_0)[i])
        if not np.isfinite(y_0):
            return None
        if (y_left < 0.0) != (y_0 < 0.0):
            x_left = x_0
            y_left = y_0
        else:
            x_right = x_0
            y_right = y_0

    if left <= x_left <= right:
        return x_left
    return None
