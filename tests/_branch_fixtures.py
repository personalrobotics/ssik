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

    def build(self) -> Any:
        import ssik

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
        payload = json.dumps(
            {
                "alpha": list(self.dh_alpha),
                "a": list(self.dh_a),
                "d": list(self.dh_d),
                "q_star": list(self.q_star),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


FIXTURES: tuple[BranchFixture, ...] = (
    BranchFixture(
        name="tan_half_infinity",
        description=(
            "Regular pose (rank 6, cond ~123) whose linearity joint sits at pi, "
            "i.e. at tan-half-angle infinity. The tan-half-angle coordinate cannot "
            "represent it, so the Raghavan-Roth path drops it as a nonfinite "
            "generalized eigenvalue and returns seven of the eight branches."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(0.0, PI / 2, PI, PI / 2, PI / 2, 0.0),
        issue="#571",
    ),
    BranchFixture(
        name="pi_at_right_bilinear_q0",
        description=(
            "q0 at pi, every other joint generic. Under the default loop split q0 is "
            "in the right-bilinear pair, whose reconstruction carries its own affine "
            "tan-half representation. Lost for a different reason than the linearity "
            "joint, so a fix aimed only at the eigenvalue at infinity leaves this one "
            "broken (see the diagnosis on #571)."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(PI, 0.62, 0.93, -0.44, 0.75, -0.26),
        issue="#571",
    ),
    BranchFixture(
        name="pi_at_right_bilinear_q1",
        description=(
            "q1 at pi, the second member of the right-bilinear pair. Included "
            "alongside q0 because the two are reconstructed together and a partial "
            "fix could plausibly repair one and not the other."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(0.31, PI, 0.93, -0.44, 0.75, -0.26),
        issue="#571",
    ),
    BranchFixture(
        name="pi_at_left_bilinear_q3",
        description=(
            "q3 at pi. The control for the split: the left-bilinear pair already "
            "handles pi correctly today, so this must stay recovered through any "
            "change to the projective handling. It is also the working pattern the "
            "right-bilinear path can copy."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(0.31, 0.62, 0.93, PI, 0.75, -0.26),
        issue=None,
    ),
    BranchFixture(
        name="tan_half_infinity_regular_pose",
        description=(
            "Same chain as tan_half_infinity at a generic pose, no joint near pi. "
            "The control: whatever the projective fix does, this pose must keep "
            "the branch set it already has."
        ),
        dh_alpha=(PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0),
        dh_a=(1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8),
        dh_d=(1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5),
        q_star=(0.3, -0.7, 0.9, -0.4, 0.8, 0.2),
        issue=None,
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
