#!/usr/bin/env bash
# Local pre-push check. Runs the same gates CI would, plus the test
# suite (which CI now skips to keep the per-PR Actions cost ~5 min).
#
# Usage:
#   scripts/check.sh
#   scripts/check.sh --no-tests    # skip pytest (lint + types only, ~30s)
#
# To run automatically before every `git push`:
#   scripts/install-hooks.sh
#
# To bypass (e.g. pushing WIP):
#   git push --no-verify
#
# Bug-test gates intentionally use the same invocations CI used to so that
# 'green here' = 'green on CI' modulo platform-specific issues caught by the
# Linux wheel-smoke job remaining in .github/workflows/ci.yml.
set -euo pipefail

cd "$(dirname "$0")/.."

run_tests=1
if [[ "${1:-}" == "--no-tests" ]]; then
    run_tests=0
fi

echo "[check] ruff check"
uv run ruff check

echo "[check] ruff format --check"
uv run ruff format --check

echo "[check] mypy"
uv run mypy

echo "[check] regen_docs --check"
uv run python scripts/regen_docs.py --check

if [[ $run_tests -eq 1 ]]; then
    echo "[check] pytest"
    # Deselects: pre-existing stale xfails / known flakes that gate on
    # optional dev deps (EAIK). The persistent flakes #101/#115/#215/#258
    # were closed in #259; the EAIK-cross-check below only fires when EAIK
    # is locally installed.
    uv run pytest -q \
        --deselect tests/test_husty_pfurner_oracles.py::test_oracle2_eaik_cross_check_ur5
fi

echo "[check] all green"
