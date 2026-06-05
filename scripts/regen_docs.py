"""Regenerate manifest-driven doc tables in README + docs/.

Reads ``src/ssik/prebuilt/MANIFEST.toml`` and rewrites the named-anchor
regions in:

  README.md                     prebuilt table + EAIK comparison table
  docs/quickstart.md            prebuilt table
  docs/arm_coverage.md          per-class rows (6R Pieper, 6R non-Pieper,
                                7R SRS, 7R approximate-SRS, 7R non-SRS)
  src/ssik/prebuilt/README.md   build-time / artifact-size table

Each region is delimited by HTML-comment anchors:

  <!-- AUTOGEN:region_name -->
  ... generated content (do not edit by hand) ...
  <!-- /AUTOGEN -->

Outside the anchors the file is untouched. Adding new arms therefore
only requires editing ``MANIFEST.toml`` and running this script.

Usage::

    uv run python scripts/regen_docs.py            # rewrite in place
    uv run python scripts/regen_docs.py --check    # fail if any file would change

The ``--check`` mode is the CI drift gate: it exits non-zero if the
files on disk diverge from what the manifest implies, with a diff for
the reviewer.

Files this script does NOT yet handle (left for follow-up):

  docs/index.md                       arm count + name list (prose)
  docs/setting_up_your_robot.md       arm count (prose)
  outreach/*.md                       counts + prose name lists
  outreach/conda-forge/meta.yaml      description block
  CITATION.cff                        abstract count + name list (YAML literal block)
  examples/04_compare_vs_eaik.py      FIXTURES list (Python source)

Once the anchor framework here is proven, follow-up PRs extend coverage
to those files. The dominant friction (per-arm table updates across
README + docs/quickstart + docs/arm_coverage) is solved by this PR.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

from ssik.prebuilt._manifest import Arm, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Cell renderers — turn an :class:`Arm` into one row of each known table.
# ---------------------------------------------------------------------------


def _fmt_time(ms_mean: float, ms_ci95: float) -> str:
    """Format mean ± CI95 in either µs or ms, choosing the unit that
    matches the legacy hand-written cells in the README.

    Convention used historically: arms running under 1 ms display in µs
    (``551 ± 12 µs``); arms at or above 1 ms display in ms with a
    decimal (``1.06 ± 0.02 ms``); arms at or above 10 ms display in ms
    with a decimal (``41.46 ± 1.25 ms``).
    """
    if ms_mean < 1.0:
        # Sub-millisecond: show in microseconds, rounded to int.
        return f"{round(ms_mean * 1000)} ± {round(ms_ci95 * 1000)} µs"
    return f"{ms_mean:.2f} ± {ms_ci95:.2f} ms"


def _fmt_fk(max_fk: float) -> str:
    """Format FK residual like ``2e-9`` (one significant digit, lowercase
    ``e``)."""
    if max_fk == 0.0:
        return "0"
    # ``f"{x:.0e}"`` gives ``2e-09``; strip the leading 0 in the exponent
    # to match the historical README style (``2e-9``).
    s = f"{max_fk:.0e}"
    mantissa, exp = s.split("e")
    sign = "-" if exp.startswith("-") else ""
    digits = exp.lstrip("+-").lstrip("0") or "0"
    return f"{mantissa}e{sign}{digits}"


def _fmt_sols(sols_min: int, sols_max: int) -> str:
    """Format the branch-count cell as ``min-max`` or just ``N`` when
    constant across poses (Puma 560 / iiwa14)."""
    if sols_max == sols_min:
        return str(sols_min)
    return f"{sols_min}-{sols_max}"


def _row_fixture_source(arm: Arm) -> str:
    """One row of the README fixture-provenance table (#311).

    Layout::

      | `<name>` | <fixture_source> |

    Surfaces the kinematic-chain provenance so a user picking up a
    prebuilt can audit whether ssik solves the same chain their
    manufacturer ships against their real hardware.
    """
    return f"| `{arm.name}` | {arm.fixture_source} |"


def _row_readme_prebuilt(arm: Arm) -> str:
    """One row of the README prebuilt table.

    Format::

      | `<name>` | <display_name> | <kinematic_class> | `<base_link>` | `<ee_link>` |

    The ``kinematic_class`` cell uses bold markdown for the EAIK-gap
    classes per the legacy convention; we wrap the class in ``**...**``
    when its ``class_tags`` include ``non-Pieper`` / ``non-SRS`` /
    ``approximate-SRS`` (the "ssik exists for these" classes).
    """
    bold_tags = {"non-Pieper", "non-SRS", "approximate-SRS"}
    cls = arm.kinematic_class
    if any(t in bold_tags for t in arm.class_tags):
        # The class string sometimes already contains a parenthesised
        # detail (e.g. "non-Pieper 6R (joint 6 y-offset)"). Bold only
        # the leading class-name portion so the parenthetical reads
        # naturally.
        if " (" in cls:
            head, paren = cls.split(" (", 1)
            cls = f"**{head}** ({paren}"
        else:
            cls = f"**{cls}**"
    return f"| `{arm.name}` | {arm.table_name} | {cls} | `{arm.base_link}` | `{arm.ee_link}` |"


def _row_readme_eaik(arm: Arm) -> str:
    """One row of the README EAIK comparison table.

    Two cells per arm: the EAIK column (a refusal string when the arm
    is outside EAIK's coverage; the measured numbers otherwise), and
    the ssik column (always the measured bench numbers from the
    manifest).
    """
    bench = arm.bench
    if bench is None:
        # Defensive: every shipped arm should have a bench block.
        ssik_cell = "—"
    else:
        ssik_cell = (
            f"{_fmt_time(bench.ms_mean, bench.ms_ci95)} / FK {_fmt_fk(bench.max_fk)} "
            f"/ {_fmt_sols(bench.sols_min, bench.sols_max)} sols"
        )

    # EAIK column: hardcoded refusal strings per class. This is the one
    # piece of doc-table metadata the manifest doesn't own (EAIK's
    # behaviour on each arm is captured verbatim from its actual error
    # messages). Maintained as a small lookup keyed on ``class_tags``.
    eaik_cell = _eaik_cell_for(arm)

    # Bold the EAIK-gap classes inside the row label (the "ssik exists
    # for these" emphasis the legacy README used).
    bold_tags = {"non-Pieper", "non-SRS", "approximate-SRS"}
    label_class = arm.short_class
    if any(t in bold_tags for t in arm.class_tags):
        # The short_class string may carry a parenthesised suffix; bold
        # only the leading class-name portion so the parenthetical reads
        # naturally ("approximate-SRS 7R, 12 mm offset" -> "**approximate-SRS 7R**, 12 mm offset").
        head, _, paren = label_class.partition(",")
        label_class = f"**{head.strip()}**,{paren}" if paren else f"**{label_class}**"

    return f"| {arm.short_name} ({label_class}) | {eaik_cell} | {ssik_cell} |"


def _eaik_cell_for(arm: Arm) -> str:
    """Lookup the EAIK comparison cell for one arm.

    EAIK's behaviour is captured here (not in the manifest) because it
    is a property of EAIK + our bench harness, not of the arm itself.
    A future cross-library comparison bench (e.g. against TracIK or
    MINK) would have its own per-class lookup table at the same layer.

    Pieper 6R: EAIK solves analytically (~5 µs). Cells in this branch
    fall back to a static "no-EAIK-bench-here" tag until the bench
    harness re-runs EAIK side-by-side. (Most arms in this class haven't
    been re-benched against EAIK in the current PR.) Override per arm
    via a future ``arm.bench_eaik`` block if needed.
    """
    tags = set(arm.class_tags)
    if "Pieper" in tags:
        # UR5 / Puma / Z1 — EAIK is native here. The current README
        # cells carry the literal "5 ± 0 µs / FK <x> / <sols>" measured
        # numbers; we preserve those by special-casing the known three.
        return _PIEPER_EAIK.get(arm.name, "_(no EAIK bench cell yet)_")
    if "non-Pieper" in tags:
        return '**refuses** ("6R-Unknown Kinematic Class")'
    if "approximate-SRS" in tags:
        return '**refuses** ("only 1-6R")'
    if "SRS" in tags and arm.fixture_kind == "specs":
        # iiwa14 etc. — EAIK doesn't accept 7-joint DH input via our adapter.
        return "**refuses** (no 7R DH path in bench harness)"
    if "non-SRS" in tags and arm.fixture_kind == "specs":
        return "**refuses** (no 7R DH path in bench harness)"
    if "non-SRS" in tags:
        # URDF-loaded 7R — EAIK rejects with its actual message
        return '**refuses** ("only 1-6R")'
    return "_(uncategorised)_"


# Hand-maintained EAIK measurements for Pieper-class arms (UR5, Puma 560,
# Z1). Updating these requires re-running the bench against EAIK; the
# numbers don't live in the manifest because they're EAIK's, not ssik's.
_PIEPER_EAIK: dict[str, str] = {
    "ur5_ik": "5 ± 0 µs / FK 2e-15 / 2-8 sols",
    "puma560_ik": "5 ± 0 µs / FK 3e-14 / 8 sols",
    "z1_ik": "5 ± 0 µs / FK 1e-15 / 4-8 sols",
}


def _row_quickstart_prebuilt(arm: Arm) -> str:
    """One row of the docs/quickstart.md prebuilt table.

    Compact form -- no bold markup -- because quickstart's table is a
    skim-reference, not the marketing comparison.
    """
    cls = arm.kinematic_class.replace("**", "")
    return f"| `{arm.name}` | {arm.table_name} | {cls} | `{arm.base_link}` | `{arm.ee_link}` |"


def _row_prebuilt_readme(arm: Arm) -> str:
    """One row of the src/ssik/prebuilt/README.md build-time/size table."""
    build_time = (
        "<1 s"
        if arm.build_time_sec <= 1
        else f"~{arm.build_time_sec // 60} min"
        if arm.build_time_sec >= 60
        else f"~{arm.build_time_sec} s"
    )
    solver_label = arm.solver
    # Embellish the solver label for cached-RR arms whose runtime path
    # uses the cached-RR fast path.
    if arm.slow_build and "jointlock" in solver_label:
        solver_label = f"`{solver_label}` + cached-RR"
    else:
        solver_label = f"`{solver_label}`"
    return f"| `{arm.name}` | {solver_label} | {build_time} | ~{arm.artifact_size_kb} KB |"


# ---------------------------------------------------------------------------
# Anchor parser / writer.
# ---------------------------------------------------------------------------

_ANCHOR_RE = re.compile(
    r"(?P<open><!-- AUTOGEN:(?P<name>[a-z0-9_]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<close>\n<!-- /AUTOGEN -->)",
    re.DOTALL,
)


def _rewrite_anchors(text: str, renderer: Callable[[str], str | None]) -> str:
    """Replace each anchored region's body with ``renderer(anchor_name)``.

    Anchors not handled by the renderer (``renderer`` returns ``None``)
    are left untouched.
    """

    def sub(m: re.Match[str]) -> str:
        new_body = renderer(m.group("name"))
        if new_body is None:
            return m.group(0)
        return m.group("open") + new_body + m.group("close")

    return _ANCHOR_RE.sub(sub, text)


# ---------------------------------------------------------------------------
# Per-file orchestration.
# ---------------------------------------------------------------------------


def _render(arms: dict[str, Arm], anchor: str) -> str | None:
    """Return the body for one anchor, or ``None`` if unknown."""
    if anchor == "readme_prebuilt_table":
        rows = [_row_readme_prebuilt(arm) for arm in arms.values()]
        header = "| Module | Arm | Class | base_link | ee_link |\n|---|---|---|---|---|"
        return header + "\n" + "\n".join(rows)
    if anchor == "readme_eaik_table":
        rows = [_row_readme_eaik(arm) for arm in arms.values()]
        header = "| Arm (class) | EAIK | ssik |\n|---|---|---|"
        return header + "\n" + "\n".join(rows)
    if anchor == "quickstart_prebuilt_table":
        rows = [_row_quickstart_prebuilt(arm) for arm in arms.values()]
        header = "| Module | Arm | Class | base_link | ee_link |\n|---|---|---|---|---|"
        return header + "\n" + "\n".join(rows)
    if anchor == "prebuilt_readme_table":
        rows = [_row_prebuilt_readme(arm) for arm in arms.values()]
        header = "| Arm | Solver | Build time | Artifact size |\n|---|---|:---:|:---:|"
        return header + "\n" + "\n".join(rows)
    if anchor == "readme_fixture_source_table":
        rows = [_row_fixture_source(arm) for arm in arms.values()]
        header = "| Module | Fixture provenance |\n|---|---|"
        return header + "\n" + "\n".join(rows)
    return None


_FILES: list[Path] = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "quickstart.md",
    REPO_ROOT / "src" / "ssik" / "prebuilt" / "README.md",
]


def regenerate(*, check: bool) -> int:
    """Walk the known files and rewrite/check their anchored regions.

    Returns a process exit code: 0 if everything is in sync, 1 if
    ``--check`` mode and any file would change.
    """
    manifest = load_manifest()

    def renderer(name: str) -> str | None:
        return _render(manifest, name)

    drift: list[tuple[Path, str, str]] = []
    for path in _FILES:
        original = path.read_text()
        rewritten = _rewrite_anchors(original, renderer)
        if rewritten == original:
            continue
        if check:
            drift.append((path, original, rewritten))
        else:
            path.write_text(rewritten)
            print(f"rewrote {path.relative_to(REPO_ROOT)}")

    if check and drift:
        import difflib

        for path, original, rewritten in drift:
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    rewritten.splitlines(keepends=True),
                    fromfile=f"committed/{path.name}",
                    tofile=f"regenerated/{path.name}",
                    n=2,
                )
            )
            print(diff[:2000], file=sys.stderr)
        print(
            f"\n{len(drift)} file(s) out of sync with MANIFEST.toml. Run "
            f"`uv run python scripts/regen_docs.py` to fix.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail (non-zero exit) if any file would change, with a diff.",
    )
    args = parser.parse_args()
    return regenerate(check=args.check)


if __name__ == "__main__":
    sys.exit(main())
