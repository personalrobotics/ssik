"""The solve input contract (#574): docs/api.md, "Input validation".

Protected failure modes, each one observed on v7.0 before this contract:

- a malformed target or seed was read out of bounds by the native extension
  (a ``(3, 4)`` target or a one-element seed returned "solutions"), while the
  Python backend raised an unrelated ``IndexError``;
- NaN or inf in the target or seed: ``[]`` (or a hang, on the HP jointlock arm)
  on one backend and an exception on the other;
- a negative ``max_solutions`` (and ``0``, which raised on the jointlock
  Python path only), a NaN ``seed_tolerance`` and a negative or non-integer
  ``refinement_max_iters`` meant different things per backend;
- direct calls to ``ssik._ssik_native`` read short arrays and out-of-range
  indices instead of failing.

The malformed calls run in a child process (``scripts/probe_solve_inputs.py``)
so a native crash fails the test instead of ending the run. Each probe test
writes its outcome table to ``<tmp_path>/*.json``; reproduce it, and see every
case, with ``uv run python scripts/probe_solve_inputs.py --json out.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ssik import Solution
from ssik._native import native_available
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import probe_solve_inputs as probe  # type: ignore[import-not-found]

_SHAPE = ("ValueError", r"must have shape")
_FINITE = ("ValueError", r"must be finite")
_RIGID = ("ValueError", r"not a rigid transform")

# Every malformed public case and the exception docs/api.md documents for it.
EXPECTED: dict[str, tuple[str, str]] = {
    "T shape (3, 3)": _SHAPE,
    "T shape (3, 4)": _SHAPE,
    "T shape (16,)": _SHAPE,
    "T shape (4,)": _SHAPE,
    "T shape (1, 4, 4)": _SHAPE,
    "T shape (5, 5)": _SHAPE,
    "T empty": _SHAPE,
    "T scalar": _SHAPE,
    "T None": ("TypeError", r"real numbers"),
    "T ragged list": ("ValueError", r"rectangular"),
    "T strings": ("TypeError", r"real numbers, got dtype <U"),
    "T complex": ("TypeError", r"got dtype complex128"),
    "T bool": ("TypeError", r"got dtype bool"),
    "T NaN translation": _FINITE,
    "T NaN rotation": _FINITE,
    "T +inf translation": _FINITE,
    "T -inf rotation": _FINITE,
    "T rotation scaled 1.01": _RIGID,
    "T rotation off SO(3) by 1e-3": _RIGID,
    "T reflection (det -1)": ("ValueError", r"determinant"),
    "T bottom row [0,0,0,2]": ("ValueError", r"bottom row"),
    "T bottom row [1,0,0,1]": ("ValueError", r"bottom row"),
    "T zero matrix": _RIGID,
    "q_seed length 1": _SHAPE,
    "q_seed length dof-1": _SHAPE,
    "q_seed length dof+1": _SHAPE,
    "q_seed shape (1, dof)": _SHAPE,
    "q_seed scalar": _SHAPE,
    "q_seed 0-d array": _SHAPE,
    "q_seed NaN": _FINITE,
    "q_seed inf": _FINITE,
    "q_seed strings": ("TypeError", r"real numbers"),
    "q_seed complex": ("TypeError", r"real numbers"),
    "q_seed length 1 + max_solutions=1": _SHAPE,
    "max_solutions=-1": ("ValueError", r"max_solutions must be None or >= 0"),
    "max_solutions=-5": ("ValueError", r"max_solutions must be None or >= 0"),
    # A cap of 0 is valid, but only once everything else is.
    "max_solutions=0 + T shape (3, 3)": _SHAPE,
    "max_solutions=0 + T NaN": _FINITE,
    "max_solutions=0 + q_seed length 1": _SHAPE,
    "max_solutions=0 + seed_tolerance without seed": (
        "ValueError",
        r"seed_tolerance requires q_seed",
    ),
    "max_solutions=2.5": ("TypeError", r"max_solutions must be None or an integer"),
    "max_solutions='3'": ("TypeError", r"max_solutions must be None or an integer"),
    "seed_tolerance without seed": ("ValueError", r"seed_tolerance requires q_seed"),
    "seed_tolerance=NaN": ("ValueError", r"seed_tolerance must not be NaN"),
    "refinement_max_iters=-1": ("ValueError", r"refinement_max_iters must be >= 0"),
    "refinement_max_iters=2.5": ("TypeError", r"refinement_max_iters must be an integer"),
    # forward kinematics: shape and dtype only, a NaN propagates
    "fk q length dof-1": _SHAPE,
    "fk q length dof+1": _SHAPE,
    "fk q shape (1, dof)": _SHAPE,
    "fk q strings": ("TypeError", r"real numbers"),
    # the chart entry points
    "self_motion T (3, 3)": _SHAPE,
    "self_motion T NaN": _FINITE,
    "self_motion T inf": _FINITE,
    "self_motion T not rigid": _RIGID,
    "self_motion T strings": ("TypeError", r"real numbers"),
    "solve_path NaN pose": ("ValueError", r"poses\[1\] must be finite"),
    "solve_path (2, 3, 3)": ("ValueError", r"poses must be \(N, 4, 4\)"),
    "solve_path q0 length 1": _SHAPE,
    "solve_path q0 NaN": _FINITE,
}

needs_native = pytest.mark.skipif(not native_available(), reason="needs ssik._ssik_native")


def _write(tmp_path: Path, name: str, rows: list[dict[str, str]]) -> None:
    (tmp_path / name).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


@needs_native
def test_malformed_inputs_raise_the_documented_exception_on_every_backend(
    tmp_path: Path,
) -> None:
    """Every malformed case raises its documented exception on every family's
    artifact (native, Python, Python without the native check kernel), the live
    ``Manipulator``, ``fk`` and the chart entry points, with one message per
    case and arm whichever backend ran. No case crashes or hangs."""
    rows = probe.run_jobs(probe.public_jobs(EXPECTED), timeout_s=120.0)
    _write(tmp_path, "solve_inputs_public.json", rows)
    assert len(rows) > 900
    messages: dict[tuple[str, str], set[str]] = {}
    bad = []
    for r in rows:
        exc, pattern = EXPECTED[r["case"]]
        out = r["outcome"]
        if not (out.startswith(f"{exc}: ") and re.search(pattern, out)):
            bad.append(f"{r['target']} {r['backend']} {r['case']}: {out}")
        if r["backend"] in probe.PUBLIC_BACKENDS:
            messages.setdefault((r["target"], r["case"]), set()).add(out)
    assert not bad, "\n".join(bad)
    split = {k: v for k, v in messages.items() if len(v) > 1}
    assert not split, f"backends disagree: {split}"


# Unusual inputs the contract accepts, and that every backend must then answer
# alike: the same number of solutions.
ACCEPTED = (
    "T object of floats",
    "T rotation off SO(3) by 1e-9",
    "max_solutions=np.int64(2)",
    "max_solutions=2**40",
    "refinement_max_iters=2**40",
    "seed_metric='bogus' (unseeded)",
    "seed_tolerance=inf",
    "fk q NaN",
)
_FAST_ARMS = ("ur5_ik", "irb6700_ik", "iiwa14_ik", "franka_panda_ik")


@needs_native
def test_zero_cap_returns_empty_on_every_backend(tmp_path: Path) -> None:
    """``max_solutions=0`` returns ``[]`` on every family's artifact (native,
    Python, Python without the native kernel) and the live ``Manipulator``,
    including the jointlock arms, whose Python path used to raise on it."""
    rows = probe.run_jobs(probe.public_jobs(["max_solutions=0"]), timeout_s=120.0)
    _write(tmp_path, "solve_inputs_zero_cap.json", rows)
    assert len(rows) == 3 * len(probe.ARMS) + 1
    bad = [r for r in rows if r["outcome"] != "ok:0"]
    assert not bad, bad


@needs_native
def test_accepted_unusual_inputs_agree_across_backends(tmp_path: Path) -> None:
    """Accepted-but-unusual inputs solve, with the same count on every backend.
    (A 2**40 cap once overflowed the native ``int``; an ignored metric once
    raised natively.)"""
    jobs = [j for j in probe.public_jobs(ACCEPTED) if j[0] in _FAST_ARMS]
    rows = probe.run_jobs(jobs, timeout_s=120.0)
    _write(tmp_path, "solve_inputs_accepted.json", rows)
    outcomes: dict[tuple[str, str], set[str]] = {}
    for r in rows:
        assert r["outcome"].startswith("ok:"), r
        outcomes.setdefault((r["target"], r["case"]), set()).add(r["outcome"])
    split = {k: v for k, v in outcomes.items() if len(v) > 1}
    assert not split, f"backends disagree: {split}"


def _binding_ok(name: str, label: str, out: str) -> bool:
    """A malformed binding call raises ValueError naming what was wrong."""
    if name == "input_defect" and (label.endswith(("NaN", "inf")) or label == "q_seed short"):
        return out == "ok:int"  # the check kernel reports a defect, it does not raise
    if not out.startswith("ValueError: "):
        return False
    arg = probe.binding_argument(name, label)
    aliases = {
        "ee_offset": "ee_offset_local",
        "po_rc": "COO",
        "po_mono": "COO",
        "po_coeff": "COO",
        "q_rc": "COO",
        "q_mono": "COO",
        "q_coeff": "COO",
        "linearity_joint": "joint role",
        "drop_joint": "joint role",
    }
    lists = "tensor list" if label.endswith("list short") else None
    return any(s in out for s in (arg, aliases.get(arg), lists) if s)


@needs_native
def test_native_bindings_reject_malformed_arguments(tmp_path: Path) -> None:
    """Every function and class of ``ssik._ssik_native`` (shipped and test-only)
    rejects each malformed argument with a ``ValueError`` that names it, before
    reading it: no crash, no hang, no result computed from out-of-bounds memory."""
    from ssik._native import _load_ext

    ext = _load_ext()
    jobs = probe.binding_jobs()
    covered = {name.split(".")[0] for name, _, _ in jobs}
    assert covered == set(probe.binding_names(ext)), "a binding has no probe baseline"
    rows = probe.run_jobs(jobs, timeout_s=120.0)
    _write(tmp_path, "solve_inputs_bindings.json", rows)
    bad = [
        f"{r['target']} {r['case']}: {r['outcome']}"
        for r in rows
        if not _binding_ok(r["target"], r["case"], r["outcome"])
    ]
    assert not bad, "\n".join(bad)


def _sorted_q(sols: list[Solution]) -> NDArray[np.float64]:
    qs = np.array([s.q for s in sols], dtype=np.float64)
    return qs[np.lexsort(qs.T[::-1])] if len(qs) else qs


# One arm per 6R closed-form family and the exact SRS 7R; the conversions are
# arm-independent, the solvers are not.
_EDGE_ARMS = ("ur5_ik", "irb6700_ik", "iiwa14_ik")


@pytest.mark.parametrize("native", [True, False], ids=["native", "python"])
@pytest.mark.parametrize("arm", _EDGE_ARMS)
def test_valid_unusual_inputs_solve_like_float64(arm: str, native: bool) -> None:
    """Array-likes that convert exactly (a list, a Fortran-ordered or strided
    array, an integer seed) give the float64 result exactly; a float32 target,
    or one 1e-9 off SO(3), gives the same solution set to the precision of the
    perturbation."""
    if native and not native_available():
        pytest.skip("needs ssik._ssik_native")
    mod = probe._module(arm)
    T, q = probe._valid(mod)
    seed = np.round(q)
    ref = _sorted_q(mod.solve(T, q_seed=seed, native=native))
    assert len(ref)
    big = np.zeros((8, 8))
    big[::2, ::2] = T
    exact = {
        "list": T.tolist(),
        "Fortran": np.asfortranarray(T),
        "strided": big[::2, ::2],
    }
    for label, variant in exact.items():
        got = _sorted_q(mod.solve(variant, q_seed=seed.astype(np.int64), native=native))
        np.testing.assert_array_equal(got, ref, err_msg=label)
    near = {
        "float32": T.astype(np.float32),
        "1e-9 off SO(3)": probe._perturb_rot(T, 1e-9),
    }
    for label, variant in near.items():
        got = _sorted_q(mod.solve(variant, q_seed=seed, native=native))
        assert got.shape == ref.shape, label
        # The same set: every solution within 1e-5 rad of one of the other's.
        gap = np.abs(got[:, None, :] - ref[None, :, :]).max(axis=2)
        assert gap.min(axis=0).max() < 1e-5, label
        assert gap.min(axis=1).max() < 1e-5, label


def test_rigidity_tolerance_rejects_only_unreachable_targets() -> None:
    """The bound docs/api.md derives the tolerance from: for a 3x3 matrix ``R``
    at Frobenius distance ``d < 1`` from SO(3), ``||R^T R - I||_F <= 3 d``. So a
    target rejected at ``3 * tol`` is further than ``tol`` from every rigid
    ``FK(q)`` and no solver gate could have accepted a solution for it."""
    rng = np.random.default_rng(574)
    for _ in range(2000):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q *= np.sign(np.linalg.det(Q))
        R = Q + 10.0 ** rng.uniform(-9, 0) * rng.normal(size=(3, 3))
        U, _, Vt = np.linalg.svd(R)
        if np.linalg.det(U @ Vt) < 0:
            continue
        d = float(np.linalg.norm(R - U @ Vt))  # distance to the nearest rotation
        if d >= 1.0:
            continue
        assert float(np.linalg.norm(R.T @ R - np.eye(3))) <= 3.0 * d * (1 + 1e-9) + 1e-15


def test_check_kernels_agree() -> None:
    """The native check kernel and its numpy reference return the same defect
    code on targets clear of the tolerance boundary, so a platform without the
    extension rejects the same targets."""
    from ssik import _solve_inputs as si

    kernel = si._native_kernel()
    if kernel is None:
        pytest.skip("needs ssik._ssik_native")
    ortho_lim, row_lim = si._limits(DEFAULT_TOLERANCE_POLICY.subproblem_numerical)
    rng = np.random.default_rng(0)
    seen = set()
    for k in range(600):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        T = np.eye(4)
        T[:3, :3] = Q * np.sign(np.linalg.det(Q))
        T[:3, 3] = rng.normal(size=3)
        kind = k % 6
        if kind == 1:
            T[:3, :3] *= 1.0 + 10.0 ** rng.uniform(-3, -1)
        elif kind == 2:
            T[:3, 0] *= -1.0
        elif kind == 3:
            T[3, rng.integers(4)] += 10.0 ** rng.uniform(-3, 0)
        elif kind == 4:
            T[rng.integers(4), rng.integers(4)] = rng.choice([np.nan, np.inf, -np.inf])
        elif kind == 5:
            T[:3, :3] += 1e-9 * rng.normal(size=(3, 3))
        seed = None if k % 2 else rng.normal(size=7)
        if seed is not None and k % 12 == 0:
            seed[rng.integers(7)] = np.nan
        code = si._input_defect_py(T, ortho_lim, row_lim, seed)
        assert int(kernel(T, ortho_lim, row_lim, seed)) == code, (k, T, seed)
        seen.add(code)
    assert seen == {0, 1, 2, 3, 4, 5}
