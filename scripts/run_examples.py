"""Run every example in ``examples/`` and record what each one checked.

Each example asserts its own claims, prints one ``[ok] <claim>`` or
``[FAIL] <claim>`` line per claim, and exits non-zero if any fails. The viewer
(05, in its two scripted headless modes) and the teleoperation loop (06) print
no such lines; they pass on a clean exit, and ``tests/test_teleop.py`` checks
their output. This runs them all (in parallel, each in its own process), writes
each run's output to ``<out>/<run>.log``, and writes ``<out>/summary.json``::

    {"python": ..., "platform": ..., "ssik": {"version": ..., "file": ...},
     "examples": [{"name": "01_quickstart", "command": ["01_quickstart.py"],
                   "status": "passed", "returncode": 0, "seconds": 0.4,
                   "checks": [{"claim": "...", "ok": true}, ...]}, ...]}

``status`` is ``passed`` or ``failed`` (a non-zero exit, a ``[FAIL]`` line, no
check lines from an example that reports them, or a timeout). The checks and
statuses are deterministic for a given ssik build; ``seconds`` and the timings
inside the logs are not, and no check depends on them.

Run it with the Python whose ssik you want to test. CI runs it against the
installed wheel with the ``demo`` extra (the artifact users install)::

    uv build --wheel
    uv venv /tmp/ex && uv pip install --python /tmp/ex/bin/python "$(ls dist/*.whl)[demo]"
    /tmp/ex/bin/python scripts/run_examples.py --out /tmp/ex-out

Exits non-zero if any example failed. Inspect ``summary.json`` and the logs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
CHECK_LINE = re.compile(r"^\[(ok|FAIL)\] (.+)$")


@dataclass(frozen=True)
class Example:
    name: str  # the run's name: its log is <name>.log
    script: str
    args: tuple[str, ...] = ()
    # An example that prints no [ok]/[FAIL] lines passes on a clean exit.
    reports_checks: bool = True


# The viewer's headless modes (05): primitives instead of meshes (no
# downloads), no browser, an ephemeral port, no waits, exit when done.
_VIEWER_HEADLESS = (
    "--no-meshes",
    "--tour-exit",
    "--tour-delay",
    "0",
    "--tour-settle",
    "0",
    "--host",
    "127.0.0.1",
    "--port",
    "0",
)

EXAMPLES = (
    Example("01_quickstart", "01_quickstart.py"),
    Example("02_trajectory_tracking", "02_trajectory_tracking.py"),
    Example("03_your_own_robot", "03_your_own_robot.py"),
    Example("04_redundant_arms", "04_redundant_arms.py"),
    Example(
        "05_viser_tour",
        "05_viser_interactive_ik.py",
        ("--tour", "--tour-per-arm", "0.2", *_VIEWER_HEADLESS),
        reports_checks=False,
    ),
    Example(
        "05_viser_self_motion",
        "05_viser_interactive_ik.py",
        ("--self-motion", "--self-motion-seconds", "0.3", *_VIEWER_HEADLESS),
        reports_checks=False,
    ),
    Example("06_teleop", "06_teleop.py", reports_checks=False),
    Example("07_cpp_from_the_wheel", "07_cpp_from_the_wheel.py"),
)


def _run(ex: Example, python: str, out: Path, timeout: float) -> dict[str, object]:
    record: dict[str, object] = {"name": ex.name, "command": [ex.script, *ex.args]}
    log = out / f"{ex.name}.log"
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"}
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [python, str(EXAMPLES_DIR / ex.script), *ex.args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            env=env,
            cwd=out,  # anything an example writes lands next to its log
        )
        output, returncode = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout
        output = partial.decode(errors="replace") if isinstance(partial, bytes) else partial or ""
        output += f"\n[run_examples] timed out after {timeout:.0f} s\n"
        returncode = None
    seconds = time.perf_counter() - t0
    log.write_text(output)
    checks = [
        {"claim": m.group(2), "ok": m.group(1) == "ok"}
        for m in map(CHECK_LINE.match, output.splitlines())
        if m
    ]
    passed = (
        returncode == 0 and all(c["ok"] for c in checks) and (bool(checks) or not ex.reports_checks)
    )
    record.update(
        status="passed" if passed else "failed",
        returncode=returncode,
        seconds=round(seconds, 2),
        log=log.name,
        checks=checks,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for logs + summary")
    parser.add_argument(
        "--python", default=sys.executable, help="interpreter to run the examples with"
    )
    parser.add_argument(
        "--jobs", type=int, default=os.cpu_count() or 2, help="examples to run at once"
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per example")
    parser.add_argument("--only", nargs="*", help="run only these runs (names, e.g. 01_quickstart)")
    args = parser.parse_args()

    examples = [ex for ex in EXAMPLES if not args.only or ex.name in args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        records = list(pool.map(lambda ex: _run(ex, args.python, out, args.timeout), examples))

    probe = subprocess.run(
        [args.python, "-c", "import ssik; print(ssik.__version__); print(ssik.__file__)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    summary = {
        "python": subprocess.run(
            [args.python, "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "platform": platform.platform(),
        "ssik": {"version": probe[0], "file": probe[1]},
        "examples": records,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    width = max(len(str(r["name"])) for r in records)
    for r in records:
        n = len(r["checks"])  # type: ignore[arg-type]
        print(f"{r['status']!s:7s} {r['name']!s:{width}s}  {n} checks, {r['seconds']} s")
    print(f"summary: {out / 'summary.json'}")
    return 1 if any(r["status"] == "failed" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
