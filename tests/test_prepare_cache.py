"""A prepared general-6R arm is fast in the next process, and returns the same answers.

``Manipulator.prepare()`` writes the arm's Raghavan-Roth derivations and native
tensor to the per-arm preparation cache (docs/api.md, "Per-arm preparation
cache"), so a fresh process loads them instead of re-running seconds of sympy.
Every step here runs in its own process on one temporary cache directory, as
deployments do. Failure modes covered:

- the second process re-derives (the cache is not read across processes);
- a cached entry answers differently from a fresh derivation: a different
  tensor, or different Python or native solutions;
- a damaged entry crashes the solve or is returned, instead of being a miss
  that is re-derived and replaced;
- a different arm is answered from another arm's entry.

Artifact: each process writes its tensor, targets and solutions to
``<basetemp>/test_prepared_arm_*/{cold,warm,repaired,changed}.npz``. To
reproduce, run ``uv run pytest tests/test_prepare_cache.py --basetemp=out``
and compare ``np.load("out/.../cold.npz")`` with ``warm.npz`` and
``repaired.npz`` array by array: every array is bit-identical.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests._cpp_backend import cpp_available

H = math.pi / 2
# The cheapest general_6r arm found (~7 s cold): a PUMA-like arm whose wrist
# offset d5 breaks the spherical wrist, so it dispatches to ikgeo.general_6r.
_DH = [[H, 0, -H, H, -H, 0], [0, 0.43, 0, 0, 0, 0], [0, 0, 0.1, 0.43, 0.05, 0]]
# The same arm with a longer forearm: different geometry, same structure.
_DH_CHANGED = [[H, 0, -H, H, -H, 0], [0, 0.43, 0, 0, 0, 0], [0, 0, 0.1, 0.5, 0.05, 0]]

_WORKER = r"""
import json, sys, time
import numpy as np
import ssik
from ssik._native import native_available, rr_native_geometry, try_native_solve

dh, lin, out = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
arm = ssik.Manipulator.from_dh(*dh)
assert arm.solver_name == "ikgeo.general_6r", arm.solver_name
t0 = time.perf_counter()
g = rr_native_geometry(arm.kinbody, linearity_joint=lin)
t1 = time.perf_counter()
if lin is None:
    arm.prepare()
t2 = time.perf_counter()
arrays = {}
tensors = {"tensor": g}
if lin is None:  # also the tensor for one fixed choice, which a changed arm asks for
    tensors["fixed"] = rr_native_geometry(arm.kinbody, linearity_joint=1)
for prefix, t in tensors.items():
    for k, v in t.items():
        for i, x in enumerate(v if isinstance(v, tuple) else (v,)):
            arrays[f"{prefix}/{k}/{i}"] = np.asarray(x)
rng = np.random.default_rng(7)
for j in range(3):
    T = arm.fk(rng.uniform(-2.0, 2.0, 6))
    arrays[f"target/{j}"] = T
    if lin is None:
        sols = arm.solve(T, respect_limits=False)
        arrays[f"python/{j}"] = np.array([s.q for s in sols])
    if native_available():
        sols = try_native_solve(
            "ikgeo.general_6r", arm.kinbody, T, respect_limits=False, rr_geometry=g
        )
        arrays[f"native/{j}"] = np.array([s.q for s in sols])
        arrays[f"native_fk/{j}"] = np.array([arm.fk(s.q) for s in sols])
np.savez(out, **arrays)
print(json.dumps({"tensor_s": t1 - t0, "prepare_s": t2 - t1}))
"""


def _run(cache: Path, out: Path, dh: list[list[float]], lin: int | None = None) -> dict[str, float]:
    env = {**os.environ, "SSIK_DERIVATION_CACHE": str(cache)}
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, json.dumps(dh), json.dumps(lin), str(out)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    times: dict[str, float] = json.loads(proc.stdout.strip().splitlines()[-1])
    return times


def _assert_identical(a: Path, b: Path) -> None:
    with np.load(a) as za, np.load(b) as zb:
        assert sorted(za.files) == sorted(zb.files)
        for name in za.files:
            x, y = za[name], zb[name]
            assert (x.dtype, x.shape) == (y.dtype, y.shape), name
            assert x.tobytes() == y.tobytes(), f"{name} differs"


def test_prepared_arm_is_fast_and_identical_in_a_fresh_process(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cold = _run(cache, tmp_path / "cold.npz", _DH)
    warm = _run(cache, tmp_path / "warm.npz", _DH)

    # The second process derives nothing: its tensor is a load, and prepare()
    # only re-lambdifies the stored derivations.
    assert warm["tensor_s"] < 0.5, warm
    assert warm["tensor_s"] < cold["tensor_s"] / 10, (cold, warm)
    assert sum(warm.values()) < sum(cold.values()) / 3, (cold, warm)
    _assert_identical(tmp_path / "cold.npz", tmp_path / "warm.npz")
    with np.load(tmp_path / "cold.npz") as z:
        assert any(z[f"python/{j}"].size for j in range(3)), "no Python solutions"
        if cpp_available():
            for j in range(3):  # the cached tensor solves the arm it was built for
                assert z[f"native/{j}"].size
                for fk in z[f"native_fk/{j}"]:
                    np.testing.assert_allclose(fk, z[f"target/{j}"], atol=1e-9)

    # A damaged native-tensor entry (the .npz files, docs/api.md) is a miss: it
    # is re-derived from the stored derivations, answers identically, and is
    # replaced by a readable entry.
    entries = sorted(cache.glob("*.npz"))
    assert entries
    for path in entries:
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    _run(cache, tmp_path / "repaired.npz", _DH)
    _assert_identical(tmp_path / "cold.npz", tmp_path / "repaired.npz")
    for path in entries:
        with np.load(path) as z:
            assert z.files

    # A different arm misses, even when the cache holds an entry for the same
    # linearity choice: its tensor is its own, and solves its own poses.
    _run(cache, tmp_path / "changed.npz", _DH_CHANGED, lin=1)
    with np.load(tmp_path / "cold.npz") as z_old, np.load(tmp_path / "changed.npz") as z_new:
        old, new = z_old["fixed/po_coo/3"], z_new["tensor/po_coo/3"]
        assert old.shape != new.shape or not np.array_equal(old, new)
        if cpp_available():
            for j in range(3):
                assert z_new[f"native/{j}"].size
                for fk in z_new[f"native_fk/{j}"]:
                    np.testing.assert_allclose(fk, z_new[f"target/{j}"], atol=1e-9)
