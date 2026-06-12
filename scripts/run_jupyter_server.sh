#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JUPYTER_HOST="${JUPYTER_HOST:-127.0.0.1}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

cmd=(
  uv run jupyter lab
  --collaborative
  --ip="${JUPYTER_HOST}"
  --port="${JUPYTER_PORT}"
  --no-browser
  --notebook-dir="${REPO_ROOT}"
)

if [[ -n "${JUPYTER_TOKEN:-}" ]]; then
  cmd+=(--IdentityProvider.token="${JUPYTER_TOKEN}")
fi

cd "${REPO_ROOT}"
exec "${cmd[@]}" "$@"
