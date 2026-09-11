#!/usr/bin/env bash
# One-command development setup. Safe to re-run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "==> Creating the environment and installing dev dependencies"
uv sync --extra dev

echo "==> Activating the git hooks"
chmod +x .githooks/* scripts/*.sh scripts/*.py
git config core.hooksPath .githooks

cat <<'MSG'

Setup complete.

  hooks      pre-commit  format + lint staged files, block big/binary/debug junk
             commit-msg  Conventional Commits
             pre-push    ruff + mypy + the full pytest suite

  commands   make check      everything the hooks and CI run
             make fmt        auto-format and auto-fix
             make test-fast  skip tests marked slow

  reading    CONTRIBUTING.md          workflow
             docs/conventions/        commits, style, testing

MSG
