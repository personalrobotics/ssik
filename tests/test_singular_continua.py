"""The singular-continuum contract, end to end on both backends (#662, #653).

At a singular pose a 6R arm's solutions can form a one-parameter family. The
contract (``docs/api.md#singular-continua``, defined in ``ssik.continuum``):
a seeded solve returns the point of the seed's continuum nearest the seed,
within limits; an unseeded solve returns one representative per continuum,
the point with the free joint at 0, else the in-limit point nearest it along
the continuum. Every check here runs the shipped artifact ``solve()`` on the
native and the Python backend.

Oracles. A configuration that generated the pose is an exact solution, so a
seeded solve from it must return it (it is its own nearest point). The other
checks construct an exactly singular wrist (``q5`` at its lock) and read the
expected point off the continuum's own geometry: the tangent of a wrist
continuum is a pair of joints moving together, found by FK, and the point the
rule names is computed from it independently of the slide.

Artifact. Each test writes what it measured (pose, expected and returned
configurations, residuals) as JSON under pytest's ``tmp_path``; rerun with
``pytest tests/test_singular_continua.py --basetemp=<dir>`` to inspect it.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssik import _native
from ssik.refinement import kinbody_jacobian

_BACKENDS = [False, True]
_SAME = 1e-9  # two configurations are the same point
_EXACT = 1e-12  # FK residual of an exact solution

# #653: seeded solves at a singular wrist that jumped to another branch on
# v7.0.0 (the release comparison's near_singular poses; the configuration is
# the seed and an exact solution). The first is #644's LR Mate pose.
_SEEDED_REPROS = [
    (
        "lrmate200id_ik",
        [-2.4020582823598255, -0.9509135376337363, -0.16809049447981295, 1.648855504254191e-05,
         -6.3703693057169856e-09, -6.283185307179586],
    ),
    (
        "m1013_ik",
        [4.589580489422299, -6.2831853050565485, -2.142642418246868, -6.283185307179586,
         6.283185305493874, 5.469692312484409],
    ),
    (
        "m1013_ik",
        [2.422963892903364, 1.1075975085605758e-08, 3.746400455914818e-09, 3.8845714275775256,
         -6.283135896937257, 1.1037986246801177],
    ),
    (
        "m1013_ik",
        [-3.8804423406692203, 2.4499163053747077, -1.2629266178554306, -1.5462005851103076,
         -6.283185307179586, -6.283185307179586],
    ),
    (
        "m0609_ik",
        [1.570806849726094, 5.7012392037262805, -1.6695825378525883e-07, -3.4951219180413164,
         -6.283185369543124, 1.9794061182095843],
    ),
    (
        "piper_ik",
        [1.5159834099775709, 1.6047548686044288, -9.765352923466948e-05, 1.5707652493169775,
         0.0, -1.0347053998126028],
    ),
    # #668: UR wrists at (ur15) and 5e-9 from (ur3e) a lock, where the SP6
    # pitch leaves the wrist SP1s reading angles that miss the FK gate (ur15)
    # or the elbow's reach (ur3e), so the branch was dropped before the slide.
    (
        "ur15_ik",
        [-0.5626552052533758, 1.6048914870758173, 2.0394662794014966, -2.3270737168086453,
         -3.141592653589793, 4.502876037038774],
    ),
    (
        "ur3e_ik",
        [-5.44384408067801, -4.403323468534377, -0.12735119360821867, 2.6179938889367707,
         3.141592648472754, -1.9550162029918958],
    ),
]  # fmt: skip


def _module(arm: str) -> Any:
    return importlib.import_module(f"ssik.prebuilt.{arm}")


def _skip_unbuilt(native: bool) -> None:
    if native and not _native.native_available():
        pytest.skip("native extension not built")


def _wrap(a: np.ndarray) -> np.ndarray:
    return np.asarray((a + np.pi) % (2.0 * np.pi) - np.pi)


def _residual(m: Any, q: np.ndarray, t: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(m.fk(q)) - t))


def _dump(tmp_path: Path, name: str, record: Any) -> None:
    (tmp_path / f"{name}.json").write_text(json.dumps(record, indent=1, default=float))


@pytest.mark.parametrize("native", _BACKENDS, ids=["python", "native"])
@pytest.mark.parametrize(("arm", "q"), _SEEDED_REPROS)
def test_seeded_solve_at_a_singular_wrist_returns_the_seed(
    arm: str, q: list[float], native: bool, tmp_path: Path
) -> None:
    """#653, #668: a configuration on a singular continuum is its own nearest
    solution, so the seeded solve returns it, at its own accuracy."""
    _skip_unbuilt(native)
    m = _module(arm)
    seed = np.array(q)
    t = np.asarray(m.fk(seed))
    (top,) = m.solve(t, q_seed=seed, max_solutions=1, native=native)
    d = float(np.max(np.abs(_wrap(np.asarray(top.q) - seed))))
    r = _residual(m, np.asarray(top.q), t)
    _dump(tmp_path, f"{arm}_{native}", {"seed": seed.tolist(), "got": list(top.q), "d": d, "r": r})
    assert d <= _SAME, f"{arm}: {d:.2e} rad from the seed, at {list(top.q)}"
    assert r <= _EXACT
    assert abs(top.fk_residual - r) <= 1e-14


def _wrist_sign(m: Any, q: np.ndarray) -> float:
    """The locked wrist's continuum direction (q4, q6) = (1, -s) * t: the sign
    for which a joint-space move along it leaves FK unchanged."""
    t = np.asarray(m.fk(q))
    for s in (1.0, -1.0):
        dq = np.zeros(6)
        dq[3], dq[5] = 0.3, -0.3 * s
        if _residual(m, q + dq, t) <= 1e-12:
            return s
    raise AssertionError("the wrist is not locked at this configuration")


def _solve_both(m: Any, t: np.ndarray, **kw: Any) -> dict[bool, np.ndarray]:
    out = {}
    for native in _BACKENDS:
        if native and not _native.native_available():
            continue
        sols = m.solve(t, native=native, enumerate_windings=False, **kw)
        out[native] = np.array([s.q for s in sols]).reshape(-1, 6)
    return out


def _same_set(a: np.ndarray, b: np.ndarray) -> bool:
    d = np.max(np.abs(_wrap(a[:, None, :] - b[None, :, :])), axis=2)
    return len(a) == len(b) and bool(np.all(d.min(axis=1) <= _SAME))


# Exactly locked wrists (q5 = 0), every other joint arbitrary and in limits.
# Spherical wrists only: no shipped general_6r wrist lines up exactly (the
# Doosan and Piper wrists come within ~1e-5), so general_6r's continua are
# covered by the seeded #653 poses above.
_LOCKED = {
    "puma560_ik": [0.4, -0.7, 0.3, 1.1, 0.0, -0.6],
    "lrmate200id_ik": [-1.2, 0.5, 0.4, 2.2, 0.0, 1.9],
    "vs060_ik": [0.9, -0.4, 1.3, -2.0, 0.0, 0.7],
}


@pytest.mark.parametrize("arm", sorted(_LOCKED))
def test_unseeded_representative_has_the_free_joint_at_zero(arm: str, tmp_path: Path) -> None:
    """Unseeded, the wrist continuum through ``q*`` is returned once, at
    ``q6 = 0``: ``q4`` takes the whole of the fixed ``q4 + s q6``. Both backends
    return the same set."""
    m = _module(arm)
    q = np.array(_LOCKED[arm])
    t = np.asarray(m.fk(q))
    s = _wrist_sign(m, q)
    expected = q.copy()
    expected[3], expected[5] = q[3] + s * q[5], 0.0
    got = _solve_both(m, t, respect_limits=False)
    _dump(tmp_path, arm, {"q": q.tolist(), "expected": expected.tolist(),
                          **{str(k): v.tolist() for k, v in got.items()}})  # fmt: skip
    for native, qs in got.items():
        on = [x for x in qs if np.max(np.abs(_wrap(x - expected))) <= _SAME]
        assert len(on) == 1, f"{arm} native={native}: {qs.tolist()}"
        assert _residual(m, on[0], t) <= _EXACT
        # No other sample of the same continuum survives.
        same_wrist = [
            x for x in qs if np.max(np.abs(_wrap(x[[0, 1, 2, 4]] - q[[0, 1, 2, 4]]))) <= 1e-6
        ]
        assert len(same_wrist) == 1, f"{arm} native={native}: {same_wrist}"
    if len(got) == 2:
        assert _same_set(got[False], got[True])


@pytest.mark.parametrize("arm", sorted(_LOCKED))
def test_seeded_off_the_continuum_returns_its_projection(arm: str) -> None:
    """A seed that is not a solution: the returned point is the one where the
    continuum's tangent is orthogonal to the seed offset (for a straight wrist
    continuum, the seed's orthogonal projection onto it)."""
    m = _module(arm)
    q = np.array(_LOCKED[arm])
    t = np.asarray(m.fk(q))
    s = _wrist_sign(m, q)
    seed = q + np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.1])
    v = np.array([0.0, 0.0, 0.0, 1.0, 0.0, -s]) / np.sqrt(2.0)
    expected = q + v * float(v @ (seed - q))
    for native in _BACKENDS:
        if native and not _native.native_available():
            continue
        (top,) = m.solve(t, q_seed=seed, max_solutions=1, native=native)
        d = float(np.max(np.abs(_wrap(np.asarray(top.q) - expected))))
        assert d <= _SAME, f"{arm} native={native}: {list(top.q)} vs {expected.tolist()}"
        assert _residual(m, np.asarray(top.q), t) <= _EXACT


def test_unseeded_representative_out_of_limits_moves_to_the_nearest_in_limit_point() -> None:
    """IRB 120's q4 is limited to +-2.79 rad, so its windings miss (2.79, 3.49).
    With ``q4 + s q6`` fixed at pi + 0.2, the free joint at 0 would put q4 in
    that gap; the representative is the in-limit point nearest it along the
    continuum, q4 at its lower limit (the upper one is farther)."""
    m = _module("irb120_ik")
    lo, hi = m._KB.joints[3].limits
    q = np.array([0.3, 0.2, -0.4, 2.6, 0.0, 0.0])
    s = _wrist_sign(m, q)
    q[5] = s * (np.pi + 0.2 - q[3])  # q4 + s q6 = pi + 0.2, q* in limits
    t = np.asarray(m.fk(q))
    total = q[3] + s * q[5]
    expected = q.copy()
    expected[3] = lo
    expected[5] = s * (total - (lo + 2.0 * np.pi))  # lo is the winding nearest the gap's top
    got = _solve_both(m, t)
    for native, qs in got.items():
        on = [x for x in qs if np.max(np.abs(_wrap(x - expected))) <= _SAME]
        assert len(on) == 1, f"native={native}: {qs.tolist()} vs {expected.tolist()}"
        assert lo <= on[0][3] <= hi
    if len(got) == 2:
        assert _same_set(got[False], got[True])


@pytest.mark.parametrize("native", _BACKENDS, ids=["python", "native"])
def test_three_parallel_locked_wrist_is_not_dropped(native: bool) -> None:
    """UR5 at q5 = 0 exactly: the wrist SP1s divide zero by zero there, and the
    core used to collapse them to 0 and drop the branch on whichever backend's
    q5 rounded to an exact 0. The representative has q6 = 0, the same q1 and
    q5, and closes FK exactly; at the rule's point the continuum's tangent
    (here bending through q2, q3 and q4 too) is orthogonal to e_6."""
    _skip_unbuilt(native)
    m = _module("ur5_ik")
    q = np.array([0.7, -1.1, 1.4, -0.9, 0.0, 2.3])
    t = np.asarray(m.fk(q))
    sols = m.solve(t, native=native, respect_limits=False, enumerate_windings=False)
    reps = [
        np.asarray(x.q)
        for x in sols
        if abs(_wrap(np.array([x.q[0] - q[0]]))[0]) <= 1e-6
        and abs(_wrap(np.array([x.q[4]]))[0]) <= 1e-6
    ]
    assert reps, [list(x.q) for x in sols]
    for x in reps:
        assert abs(_wrap(np.array([x[5]]))[0]) <= _SAME
        assert _residual(m, x, t) <= _EXACT
        v = np.linalg.svd(kinbody_jacobian(m._KB, x))[2][-1]
        assert abs(v[5]) > 0.1  # q6 moves along it: it is a free joint here
