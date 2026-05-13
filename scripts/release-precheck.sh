#!/usr/bin/env bash
# Pre-release wheel smoke. Run before tagging v* to catch packaging-class
# bugs (missing runtime deps, broken Cython compile, broken prebuilt imports)
# that fresh-venv install would surface but `uv run` against the dev tree
# would NOT (because the dev environment has every dep installed transitively).
#
# This is the local mirror of the wheel-smoke job in CI; if this is green,
# CI will be too.
#
# Usage:
#   scripts/release-precheck.sh
#
# Lessons baked in:
#   * #252 scipy missing runtime-dep -- would have caught it before rc1/rc2

set -euo pipefail

cd "$(dirname "$0")/.."

VENV=$(mktemp -d -t ssik-release-smoke.XXXXXX)/v
trap "rm -rf $(dirname "$VENV")" EXIT

echo "[precheck] cleaning prior build artifacts"
find src/ssik -name "*.so" -delete
rm -rf build/ dist/

echo "[precheck] building wheel"
uv build --wheel

WHEEL=$(ls dist/ssik-*.whl | head -1)
echo "[precheck] built: $WHEEL"

echo "[precheck] installing in fresh venv: $VENV"
uv venv "$VENV" --python 3.13 >/dev/null
uv pip install --python "$VENV/bin/python" "$WHEEL" >/dev/null

echo "[precheck] smoke 1/3: iiwa14_ik (strict SRS, no scipy in chain)"
"$VENV/bin/python" -c "
from ssik.prebuilt import iiwa14_ik
import numpy as np
q = np.array([0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4])
T = iiwa14_ik.fk(q)
sols = iiwa14_ik.solve(T)
assert sols and max(s.fk_residual for s in sols) < 1e-9, 'iiwa14_ik smoke failed'
print(f'  iiwa14_ik: {len(sols)} sols, maxFK={max(s.fk_residual for s in sols):.1e}')
"

echo "[precheck] smoke 2/3: franka_panda_ik (HP-backed via jointlock)"
"$VENV/bin/python" -c "
from ssik.prebuilt import franka_panda_ik
import numpy as np
T = franka_panda_ik.fk(np.array([0.2, 0.3, -0.4, -1.2, 0.3, 1.4, 0.5]))
sols = franka_panda_ik.solve(T)
assert sols, 'franka_panda_ik returned no IK'
print(f'  franka_panda_ik: {len(sols)} sols, maxFK={max(s.fk_residual for s in sols):.1e}')
"

echo "[precheck] smoke 3/3: ssik --help (full CLI import chain)"
"$VENV/bin/ssik" --help >/dev/null
echo "  ssik --help: ok"

echo "[precheck] all green -- safe to tag"
