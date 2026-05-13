#!/usr/bin/env bash
# Install a pre-push hook that runs `scripts/check.sh` automatically.
# Idempotent -- safe to re-run. Overwrites the existing hook (warns if non-empty).
#
# Usage:
#   scripts/install-hooks.sh
#
# After install, every `git push` runs the local check gate. Bypass with
# `git push --no-verify` if you need to push WIP.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .git ]]; then
    echo "error: not at the repo root (no .git/ directory)" >&2
    exit 1
fi

hook=".git/hooks/pre-push"

if [[ -s "$hook" ]]; then
    if ! grep -q "scripts/check.sh" "$hook"; then
        echo "warning: $hook already exists with non-ssik content; overwriting"
        echo "         backup at $hook.bak"
        cp "$hook" "$hook.bak"
    fi
fi

cat >"$hook" <<'EOF'
#!/usr/bin/env bash
# Installed by ssik's scripts/install-hooks.sh -- runs the local pre-push
# gate. Bypass with `git push --no-verify`.
exec ./scripts/check.sh
EOF

chmod +x "$hook"
echo "[install-hooks] installed $hook"
echo "[install-hooks] every \`git push\` will now run scripts/check.sh"
echo "[install-hooks] bypass for WIP pushes:  git push --no-verify"
