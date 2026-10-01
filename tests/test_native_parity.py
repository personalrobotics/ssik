"""Native-parity gate: native ``solve()`` against Python and the branch oracle (#626).

Native has been the default solve path since #554. The older parity tests
sample uniform in-limits poses only, drop near-singular, unstable and
near-continuum poses from their goldens, and absorb the rest in per-arm
allowances, so eight classes of native gap (#626's table) went unseen. This
gate runs every shipped arm over the targeted strata of ``tests/_native_parity``
and attributes every violation to one class.

Each test item is one (arm, class) cell: it passes when the arm shows no gap of
that class. A cell with a known gap is a strict xfail naming the class's issue,
so the PR that fixes a class turns its cells into XPASS failures and must
delete them from ``KNOWN_*`` below. A gap no class explains is class ``new``,
never expected on any arm: any new gap fails the gate.

Tiers
    ``fast`` (PR CI): a small prefix of every stratum plus the committed
    witness poses of each known gap (``fast_pins``), so it fails the same
    cells as the full tier in a few seconds per arm. ``full`` (``-m slow``,
    nightly): 200 uniform and about 150 targeted poses per arm.

Platforms
    A pose on a boundary can resolve differently on Linux and macOS, so a
    cell is known per platform: ``KNOWN_*`` maps each known arm to the
    platforms where its cell fails (``ALL`` for both), and the strict xfail
    applies only there. A class whose cells differ even between CI runners of
    one platform (``ROUNDOFF_CLASSES``, none at present) has non-strict known
    cells; a gap on an arm not listed still fails. The few cells of a strict
    class that do the same (``RUNNER_DEPENDENT_CELLS``) are non-strict too,
    each until its named issue is fixed.

Reproduce a failure: the message names the pose id (``<stratum>/<index>``);
``tests._native_parity.poses(arm, "full")`` returns its ``q``, and
``module.fk(q)`` the target. ``scripts/regen_native_parity.py`` regenerates the
committed data and prints these tables.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from ssik import DEFAULT_TOLERANCE_POLICY
from ssik._native import native_available
from tests import _native_parity as npar

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="ssik._ssik_native not built (see scripts/build_cpp_ext.py --out-dir src/ssik)",
)

# Known gaps: class -> {arm: the platforms where that cell fails}. Printed by
# scripts/regen_native_parity.py from the committed per-platform cells.
# hc10_ik's E and J cells are near_singular/9, where the RR pencil is singular
# (det M(x) vanishes for every x): no root or split rule reads its branches,
# and each backend samples the solution continuum at its own points (#662).
ALL = ("darwin", "linux")

KNOWN_FORWARD: dict[str, dict[str, tuple[str, ...]]] = {
    "E": {
        "cr5_ik": ("darwin",),
        "hc10_ik": ALL,
        "irb6700_ik": ("linux",),
        "kr210_r2700_ik": ("darwin",),
        "nova5_ik": ALL,
        "puma560_ik": ALL,
        "rv4fr_ik": ("darwin",),
        "standardbots_core_ik": ALL,
        "standardbots_spark_ik": ("darwin",),
        "standardbots_thor_ik": ALL,
        "ur10e_ik": ALL,
        "ur15_ik": ALL,
        "ur16e_ik": ALL,
        "ur18_ik": ("darwin",),
        "ur20_ik": ALL,
        "ur3e_ik": ("linux",),
        "ur5_ik": ALL,
        "ur5e_ik": ALL,
    },
    "F": {
        "fanuc_crx20ial_ik": ("linux",),
        "fanuc_crx3ia_ik": ALL,
        "openarm_left_ik": ALL,
        "openarm_right_ik": ALL,
    },
    "G": {
        "kassow_kr810_ik": ALL,
    },
    "I": {
        "gen3_ik": ALL,
        "rm75_ik": ALL,
        "yumi_left_ik": ALL,
    },
}

KNOWN_REVERSE: dict[str, dict[str, tuple[str, ...]]] = {
    "F": {
        "fanuc_crx3ia_ik": ("linux",),
        "piper_ik": ALL,
    },
    "J": {
        "cr5_ik": ALL,
        "gp8_ik": ALL,
        "hc10_ik": ("darwin",),
        "irb6700_ik": ("darwin",),
        "kr210_r2700_ik": ALL,
        "lrmate200id_ik": ("darwin",),
        "nova5_ik": ("linux",),
        "r2000ic210l_ik": ALL,
        "rv4fr_ik": ("linux",),
        "standardbots_core_ik": ("linux",),
        "standardbots_spark_ik": ("linux",),
        "standardbots_thor_ik": ALL,
        "ur10e_ik": ALL,
        "ur15_ik": ALL,
        "ur16e_ik": ALL,
        "ur18_ik": ALL,
        "ur20_ik": ALL,
        "ur3e_ik": ALL,
        "ur7e_ik": ALL,
        "vs060_ik": ALL,
    },
}


def _classes(arm: str, direction: str) -> list[str]:
    """The classes whose rules (``_native_parity._Pose``) can fire for this
    arm's solver family, then ``new``."""
    if direction == "reverse":
        return ["D", "F", "J", npar.NEW]
    fam = npar.family(arm)
    out = ["D", "F", "G" if fam == "jointlock.hp" else "E"]
    if fam.startswith("seven_r.spherical_shoulder"):
        out.append("A")
    if fam == "seven_r.srs_polished":
        out += ["B", "I"]
    if fam == "jointlock.rr":
        out.append("C")
    return [*sorted(out), npar.NEW]


def _cells(direction: str, known: dict[str, dict[str, tuple[str, ...]]]) -> list[Any]:
    cells = []
    for arm in npar.arm_names():
        for cls in _classes(arm, direction):
            # One arm's cells share one evaluation (cached per process);
            # ``--dist loadgroup`` keeps them on one worker (slow.yml).
            marks: list[pytest.MarkDecorator] = [pytest.mark.xdist_group(f"parity-{arm}")]
            where = known.get(cls, {}).get(arm, ())
            if where:
                issue, title = npar.CLASSES[cls]
                if (
                    cls in npar.ROUNDOFF_CLASSES
                    or (direction, cls, arm) in npar.RUNNER_DEPENDENT_CELLS
                ):
                    reason = f"#{issue}: {title} (round-off: varies by machine)"
                    marks.append(pytest.mark.xfail(strict=False, reason=reason))
                else:
                    reason = f"#{issue}: {title}"
                    marks.append(
                        pytest.mark.xfail(sys.platform in where, strict=True, reason=reason)
                    )
            cells.append(pytest.param(arm, cls, marks=marks, id=f"{arm}-{cls}"))
    return cells


def _assert_none(gaps: list[npar.Gap], arm: str, cls: str) -> None:
    if gaps:
        head = "\n  ".join(str(g) for g in gaps[:8])
        more = f"\n  ... and {len(gaps) - 8} more" if len(gaps) > 8 else ""
        what = (
            "an unclassified gap" if cls == npar.NEW else f"class {cls} (#{npar.CLASSES[cls][0]})"
        )
        pytest.fail(f"{arm}: {len(gaps)} x {what}:\n  {head}{more}", pytrace=False)


def _forward(arm: str, cls: str, tier: str) -> None:
    _assert_none([g for g in npar.evaluate(arm, tier).forward if g.cls == cls], arm, cls)


def _reverse(arm: str, cls: str, tier: str) -> None:
    _assert_none([g for g in npar.evaluate(arm, tier).reverse if g.cls == cls], arm, cls)


@pytest.mark.parametrize(("arm", "cls"), _cells("forward", KNOWN_FORWARD))
def test_native_parity(arm: str, cls: str) -> None:
    """Soundness, emptiness and coverage of native against Python (fast tier)."""
    _forward(arm, cls, "fast")


@pytest.mark.slow
@pytest.mark.parametrize(("arm", "cls"), _cells("forward", KNOWN_FORWARD))
def test_native_parity_full(arm: str, cls: str) -> None:
    """The same, on the full tier."""
    _forward(arm, cls, "full")


@pytest.mark.parametrize(("arm", "cls"), _cells("reverse", KNOWN_REVERSE))
def test_python_covers_native(arm: str, cls: str) -> None:
    """The reverse direction: what native finds and Python does not (fast tier)."""
    _reverse(arm, cls, "fast")


@pytest.mark.slow
@pytest.mark.parametrize(("arm", "cls"), _cells("reverse", KNOWN_REVERSE))
def test_python_covers_native_full(arm: str, cls: str) -> None:
    """The same, on the full tier."""
    _reverse(arm, cls, "full")


# f. Non-default options, each of which changes the Python result on some
# shipped arm: a closure gate below round-off, a merge radius wider than some
# branch separations, and LM polish.
_POLICY_CASES = {
    "subproblem_numerical=1e-15": (
        replace(DEFAULT_TOLERANCE_POLICY, subproblem_numerical=1e-15),
        False,
    ),
    "subproblem_dedup=0.3": (replace(DEFAULT_TOLERANCE_POLICY, subproblem_dedup=0.3), False),
    "allow_refinement": (DEFAULT_TOLERANCE_POLICY, True),
}
_POLICY_POSES = 3


@pytest.mark.parametrize("arm", npar.arm_names())
def test_native_honours_policy(arm: str) -> None:
    """A caller's ``policy`` and ``allow_refinement`` reach the result: native
    returns what the Python path returns under the same options, whether it
    honours them itself or hands the call to Python (#625, #628). Analytic
    solves only (``allow_rescue=False``), so both sides are deterministic."""
    m = npar.module(arm)
    tol = npar.MATCH_TOL[m.DOF]
    diffs = []
    for pid, q in npar.poses(arm, "full")[:_POLICY_POSES]:
        t = np.asarray(m.fk(q), dtype=np.float64)
        for name, (policy, refine) in _POLICY_CASES.items():
            got = {}
            for native in (True, False):
                sols = m.solve(
                    t,
                    native=native,
                    policy=policy,
                    allow_refinement=refine,
                    allow_rescue=False,
                    enumerate_windings=False,
                )
                got[native] = np.array([s.q for s in sols], dtype=np.float64).reshape(-1, m.DOF)
            nat, py = got[True], got[False]
            if len(nat) != len(py) or npar.unmatched(nat, py, tol) or npar.unmatched(py, nat, tol):
                diffs.append(f"{pid} {name}: native {len(nat)} vs Python {len(py)}")
    if diffs:
        msg = "\n  ".join(diffs)
        pytest.fail(f"{arm}: native ignores the caller's options:\n  {msg}", pytrace=False)
