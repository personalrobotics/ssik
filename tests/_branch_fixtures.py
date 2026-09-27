"""Curated fixtures whose complete branch set is known (#573).

One definition of each fixture, shared by the golden generator
(``scripts/regen_branch_goldens.py``) and the contract tests, so the two cannot
drift. The expected branch sets live in ``tests/data/branch_goldens.json``,
computed offline by the chart-free oracle: finding them takes thousands of LM
solves, which is far too slow to repeat on every pull request.

Each fixture carries a fingerprint of its own numbers. A committed golden
records the fingerprint it was generated from, so editing a fixture without
regenerating turns the tests red instead of silently checking against a stale
expectation.

Cost note, which shapes how to extend this set. A general-6R chain pays its
Raghavan-Roth elimination derivation once, measured at 27 s, and the result is
cached process-wide: every later solve on that chain is about 1 ms, including
from a freshly built ``Manipulator``. So adding poses to an existing chain is
free and adding a chain is not. Prefer several poses per chain, and add a chain
only when its geometry probes something the existing ones cannot.

Note also why these fixtures are solved live rather than through a committed
artifact. An artifact bakes the solver's current behaviour, which is exactly
what M7 is changing, so a baked fixture would keep passing its xfail after the
fix landed. The derivation cost buys the property that these tests see the
solver as it is now.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

GOLDEN_PATH = Path(__file__).parent / "data" / "branch_goldens.json"

PI = np.pi


@dataclass(frozen=True)
class BranchFixture:
    """A chain plus a pose whose full branch set we want pinned."""

    name: str
    description: str
    dh_alpha: tuple[float, ...]
    dh_a: tuple[float, ...]
    dh_d: tuple[float, ...]
    q_star: tuple[float, ...]
    issue: str | None = None
    """Issue this fixture exists for, when it encodes a known defect."""
    prebuilt: str | None = None
    """A shipped arm to use instead of the DH chain (whose fields are then empty)."""
    linearity_choices: tuple[int, ...] = ()
    """RR linearity-joint choices under which both backends must return the
    complete branch set. Empty: only the production solve is checked. A choice
    is left out where the elimination itself is degenerate for the geometry
    (no representation fix can recover branches the pencil does not carry)."""

    def build(self) -> Any:
        import ssik

        if self.prebuilt is not None:
            return ssik.Manipulator.from_prebuilt(self.prebuilt)
        return ssik.Manipulator.from_dh(
            dh_alpha=list(self.dh_alpha), dh_a=list(self.dh_a), dh_d=list(self.dh_d)
        )

    def q_star_array(self) -> NDArray[np.float64]:
        return np.array(self.q_star, dtype=np.float64)

    def fingerprint(self) -> str:
        """Hash of the numbers that define this fixture.

        Changing any of them invalidates the committed branch set, and the
        tests check this rather than trusting the file's name.
        """
        numbers: dict[str, Any] = {
            "alpha": list(self.dh_alpha),
            "a": list(self.dh_a),
            "d": list(self.dh_d),
            "q_star": list(self.q_star),
        }
        if self.prebuilt is not None:
            numbers["prebuilt"] = self.prebuilt
        payload = json.dumps(numbers, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


FIXTURES: tuple[BranchFixture, ...] = (
    BranchFixture(
        name="pi_at_left_bilinear_q2",
        description=(
            "The reproducer from #571, q2 at pi. This chain's auto-selected "
            "leftvar is 0, so q2 is a LEFT-bilinear joint rather than the "
            "polynomial variable: the back-substitution half of the defect. "
            "The root is found; reading it was what failed, because v_12 is a "
            "normalised monomial vector whose low-degree block falls to ~1e-16 "
            "once the variable goes to infinity. Fixed by reading each pair as "
            "a homogeneous coordinate via atan2 instead of dividing. Guards "
            "that: here the old code's own chosen pair already yielded pi, and "
            "the branch was then deleted by an absolute denominator guard."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(0.0, PI / 2, PI, PI / 2, PI / 2, 0.0),
        linearity_choices=(0, 1, 2),
    ),
    BranchFixture(
        name="pi_at_linearity_variable_q0",
        description=(
            "q0 at pi, which IS the polynomial variable under leftvar 0: the "
            "root-finding half of the defect. cond(m_quad) is 1.1e16 here, so "
            "the direct route is skipped and the Mobius search runs -- and it "
            "succeeds, conditioning the pencil down to 34 and recovering the "
            "root as an ordinary finite x_tilde. The inverse map then deleted "
            "it, because mapping back to x_2 divides by a denominator that "
            "vanishes exactly when the joint is at pi. Guards that the map "
            "keeps the root and names it infinity rather than dropping it."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(PI, 0.62, 0.93, -0.44, 0.75, -0.26),
        linearity_choices=(0, 1, 2),
    ),
    BranchFixture(
        name="pi_at_left_bilinear_q1",
        description=(
            "q1 at pi, the other LEFT-bilinear joint, and the cleanest isolation "
            "of the back-substitution half: the linearity joint stays at 0.31 and "
            "its root is found correctly, yet q* was missed by 1.218 rad, so the "
            "loss was provably in the read and not in root finding. Here the "
            "surviving x0^3 block still carries q2 exactly (v[10]/v[11] = 0.50172 "
            "-> 0.9300), while every x0 pair has its denominator in the noise "
            "block -- which is why selecting on the largest denominator failed "
            "and selecting on the largest entry works."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(0.31, PI, 0.93, -0.44, 0.75, -0.26),
        linearity_choices=(0, 1, 2),
    ),
    BranchFixture(
        name="pi_at_dropped_joint_q3",
        description=(
            "q3 at pi, the dropped joint, recovered via atan2 on a genuine "
            "(sin, cos) pair. The control: pi already works here and must keep "
            "working through the projective change. The right-bilinear pair "
            "(q4, q5) works the same way and is the pattern to copy."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(0.31, 0.62, 0.93, PI, 0.75, -0.26),
        issue=None,
        linearity_choices=(0, 1, 2),
    ),
    BranchFixture(
        name="regular_pose_control",
        description=(
            "Same chain, no joint at pi. The other control: whatever the fix does, "
            "this pose must keep the branch set it already has."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(0.3, -0.7, 0.9, -0.4, 0.8, 0.2),
        issue=None,
        linearity_choices=(0, 1, 2),
    ),
    # Repeated roots (#595). Each q* shares its linearity-joint value with a
    # second branch of the same pose, so M(x) has a two-dimensional null space
    # at that root and any single null vector mixes the two branches. q* and
    # its partner were constructed by Gauss-Newton on FK(q_a) = FK(q_b) with
    # q_lin held equal (see the #595 comment for the construction and the
    # chart-free oracle check); linearity 0 is this chain's auto choice.
    BranchFixture(
        name="repeated_root_finite_q0",
        description=(
            "q0 = 0.7 is shared with the branch (0.7, -2.4715, 2.8772, -0.7956, "
            "-2.4356, -0.3428): a finite double root of the linearity variable "
            "under leftvar 0. Both backends read one arbitrary vector of the "
            "2-D null space and dropped one or both branches. Fixed by splitting "
            "the null space with the x_lb0 shift of the monomial vector."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(
            0.7,
            0.896449747236101,
            -2.6490492624428175,
            -1.3012631260053205,
            2.476895205304505,
            -3.1067183864736414,
        ),
        linearity_choices=(0, 1, 2),
    ),
    BranchFixture(
        name="repeated_root_at_pi_q0",
        description=(
            "q0 = pi is shared with the branch (-pi, -0.2269, -2.0328, -1.7930, "
            "1.7203, -0.3324): the double root is the point at infinity, where "
            "native reads A's null space and Python the Mobius-mapped pencil's. "
            "The #571 infinite-root handling alone keeps the root but still "
            "reads one mixed vector for two branches."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(
            PI,
            2.8180583116169835,
            -3.1285452622210848,
            -1.433222632883318,
            -1.8335345562224923,
            1.6653420782413804,
        ),
        linearity_choices=(0, 1, 2),
    ),
    BranchFixture(
        name="xarm6_repeated_root_finite",
        description=(
            "UFactory xArm 6, as shipped (auto linearity 0), q0 = 0.7 shared "
            "with the branch (0.7, -1.7207, 0.6964, 2.0329, 0.9698, 2.1290). "
            "A real arm's production solve lost one branch natively and two in "
            "Python. Linearity 2 is degenerate on this arm (the pencil misses "
            "most branches at every pose), so it is not a completeness claim."
        ),
        dh_alpha=(),
        dh_a=(),
        dh_d=(),
        q_star=(
            0.7,
            1.186123607965735,
            -0.2221316738437733,
            -1.1086668248502818,
            -2.171757479090648,
            -2.7097702948263684,
        ),
        prebuilt="xarm6_ik",
        linearity_choices=(0, 1),
    ),
)

BY_NAME = {f.name: f for f in FIXTURES}


def load_goldens() -> dict[str, Any]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_PATH} is missing. Generate it with: python scripts/regen_branch_goldens.py"
        )
    data: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data
