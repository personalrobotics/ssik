"""Shared input validation for subproblem solvers.

Subproblems are performance-sensitive (called from every IK solve), so we
keep the validation cheap: only shape, finiteness, and magnitude checks.
No input-normalisation or correction; callers who want unit axes or
scaled vectors apply those themselves.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = ["validate_vec3", "validate_vec3_iterable"]


def validate_vec3(v: NDArray[np.float64], name: str) -> None:
    """Validate a single 3-vector.

    Raises :class:`ValueError` on shape mismatch or non-finite entries.
    """
    if v.shape != (3,):
        raise ValueError(f"{name}: expected shape (3,), got {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name}: non-finite entries {v.tolist()}")


def validate_vec3_iterable(vs: Iterable[NDArray[np.float64]], name: str) -> None:
    """Validate each 3-vector in an iterable; stop at first failure."""
    for i, v in enumerate(vs):
        validate_vec3(v, f"{name}[{i}]")
