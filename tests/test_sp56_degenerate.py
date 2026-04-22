"""Regression tests for known pathological geometries in SP5 and SP6.

These test that the beyond-IK-Geo hardening (upfront degeneracy checks +
post-verification against the original equation) behaves correctly on
configurations that silently break the upstream Rust implementation:

- **SP5**: ``k_1 || k_2`` or ``k_3 || k_2`` (cone reduction ill-defined);
  ``p_1`` collinear with ``k_1`` (angle undetermined).
- **SP6**: ``p_i`` collinear with ``k_i`` (term doesn't depend on its angle);
  rank-deficient stacked ``A`` matrix.

On each case we assert ``is_ls == True`` and ``solutions == []`` -- the
solver detects the degeneracy and refuses cleanly rather than returning
arbitrary numeric noise.

Tracked in #48.
"""

from __future__ import annotations

import numpy as np

from ssik.subproblems import sp5, sp6


def _unit(v: np.ndarray) -> np.ndarray:
    return v / float(np.linalg.norm(v))


# ---------------------------------------------------------------------------
# SP5 pathological geometries
# ---------------------------------------------------------------------------


def test_sp5_k1_parallel_k2_returns_ls() -> None:
    """``k_1 || k_2`` makes the cone-around-k_2 reduction undefined for k_1."""
    k = np.array([0.0, 0.0, 1.0])
    p0 = np.array([0.5, -0.3, 0.8])
    p1 = np.array([0.7, 0.1, -0.2])
    p2 = np.array([-0.2, 0.4, 0.1])
    p3 = np.array([0.1, 0.3, 0.5])
    k3 = _unit(np.array([0.5, 0.5, 0.3]))

    solutions, is_ls = sp5.solve(p0, p1, p2, p3, k, k, k3)
    assert is_ls
    assert solutions == []


def test_sp5_k3_parallel_k2_returns_ls() -> None:
    """``k_3 || k_2`` -- same class."""
    k = np.array([0.0, 1.0, 0.0])
    p0 = np.array([0.5, -0.3, 0.8])
    p1 = np.array([0.7, 0.1, -0.2])
    p2 = np.array([-0.2, 0.4, 0.1])
    p3 = np.array([0.1, 0.3, 0.5])
    k1 = _unit(np.array([0.5, 0.5, 0.3]))

    solutions, is_ls = sp5.solve(p0, p1, p2, p3, k1, k, k)
    assert is_ls
    assert solutions == []


def test_sp5_p1_collinear_with_k1_returns_ls() -> None:
    """``p_1`` along ``k_1`` -- ``theta_1`` undetermined (rotation is identity)."""
    p0 = np.array([0.5, -0.3, 0.8])
    k1 = _unit(np.array([1.0, 0.5, 0.2]))
    p1 = k1 * 1.3  # collinear
    p2 = np.array([-0.2, 0.4, 0.1])
    p3 = np.array([0.1, 0.3, 0.5])
    k2 = _unit(np.array([0.1, 1.0, 0.4]))
    k3 = _unit(np.array([0.5, 0.2, 0.9]))

    solutions, is_ls = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    assert is_ls
    assert solutions == []


def test_sp5_p3_collinear_with_k3_returns_ls() -> None:
    """``p_3`` along ``k_3`` -- ``theta_3`` undetermined."""
    p0 = np.array([0.5, -0.3, 0.8])
    p1 = np.array([0.7, 0.1, -0.2])
    p2 = np.array([-0.2, 0.4, 0.1])
    k3 = _unit(np.array([0.2, 0.9, 0.3]))
    p3 = k3 * 2.1  # collinear
    k1 = _unit(np.array([1.0, 0.5, 0.2]))
    k2 = _unit(np.array([0.1, 1.0, 0.4]))

    solutions, is_ls = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    assert is_ls
    assert solutions == []


# ---------------------------------------------------------------------------
# SP6 pathological geometries
# ---------------------------------------------------------------------------


def test_sp6_p_collinear_with_k_returns_ls() -> None:
    """Any ``p_i`` along ``k_i`` -- the i-th term does not depend on its angle."""
    k1 = _unit(np.array([0.0, 0.0, 1.0]))
    k2 = _unit(np.array([1.0, 0.0, 0.0]))
    k = [k1, k2, k1, k2]
    # Make p[0] collinear with k1.
    p = [
        k1 * 1.5,
        np.array([0.3, 0.4, 0.5]),
        np.array([0.7, -0.2, 0.1]),
        np.array([-0.1, 0.6, 0.4]),
    ]
    h = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
    ]

    solutions, is_ls = sp6.solve(h, k, p, 0.1, 0.2)
    assert is_ls
    assert solutions == []


def test_sp6_rank_deficient_system_returns_ls() -> None:
    """Two equations that collapse to the same constraint -- rank 1."""
    k1 = _unit(np.array([0.0, 0.0, 1.0]))
    k2 = _unit(np.array([1.0, 0.0, 0.0]))
    k = [k1, k2, k1, k2]
    p = [
        np.array([0.5, 0.3, 0.0]),
        np.array([0.0, 0.4, 0.5]),
        np.array([0.5, 0.3, 0.0]),
        np.array([0.0, 0.4, 0.5]),
    ]
    # Equation 2 uses the same (h, k, p) as equation 1 => a_mat rows are
    # parallel and the system's rank drops to 1.
    h = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    solutions, is_ls = sp6.solve(h, k, p, 0.1, 0.1)
    assert is_ls
    assert solutions == []


# ---------------------------------------------------------------------------
# Positive control: post-verification doesn't over-reject well-formed inputs
# ---------------------------------------------------------------------------


def test_sp5_generic_input_still_returns_solutions() -> None:
    """Sanity: the hardening doesn't starve SP5 on generic inputs."""
    rng = np.random.default_rng(42)
    k1 = _unit(rng.standard_normal(3))
    k2 = _unit(rng.standard_normal(3))
    k3 = _unit(rng.standard_normal(3))
    p1 = rng.standard_normal(3)
    p2 = rng.standard_normal(3)
    p3 = rng.standard_normal(3)
    t1, t2, t3 = 0.5, -0.7, 1.2
    # Construct p0 so the equation holds at (t1, t2, t3).
    from ssik.subproblems._rotation import rotate

    rhs = rotate(k2, t2, p2 + rotate(k3, t3, p3))
    p0 = rhs - rotate(k1, t1, p1)

    solutions, is_ls = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    assert not is_ls
    assert len(solutions) >= 1


def test_sp6_generic_input_still_returns_solutions() -> None:
    """Sanity: SP6 hardening leaves generic cases intact."""
    from ssik.subproblems._rotation import rotate

    rng = np.random.default_rng(13)
    k1 = _unit(rng.standard_normal(3))
    k2 = _unit(rng.standard_normal(3))
    k = [k1, k2, k1, k2]
    p = [rng.standard_normal(3) for _ in range(4)]
    h = [rng.standard_normal(3) for _ in range(4)]
    t1, t2 = 0.6, -1.1

    d1 = float(h[0] @ rotate(k[0], t1, p[0])) + float(h[1] @ rotate(k[1], t2, p[1]))
    d2 = float(h[2] @ rotate(k[2], t1, p[2])) + float(h[3] @ rotate(k[3], t2, p[3]))

    solutions, is_ls = sp6.solve(h, k, p, d1, d2)
    assert not is_ls
    assert len(solutions) >= 1
