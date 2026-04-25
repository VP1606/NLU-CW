#!/usr/bin/env bash
# OracleNLI live demo launcher with a public Cloudflare quick tunnel.
#
# Sets FLARE_TUNNEL=1 so server.py owns the tunnel lifecycle (started in-
# process via flaredantic, captured for the QR panel via /api/public-url,
# torn down via atexit). Anyone with the printed *.trycloudflare.com URL
# can hit /api/predict and consume your local CPU/MPS time, so kill this
# script (Ctrl+C) when you're done.
#
# Usage (from anywhere):
#     ./live_web_demo/start_remote.sh
#
# Prereqs: same as start.sh (venv/, venv310/, model files), plus:
#     venv/bin/pip install flaredantic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PY="$REPO_ROOT/venv/bin/python"
VENV310_PY="$REPO_ROOT/venv310/bin/python"
export PORT="${PORT:-8765}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "error: $VENV_PY not found. Create venv/ and 'pip install -r live_web_demo/requirements-venv.txt'." >&2
  exit 1
fi

if [[ ! -x "$VENV310_PY" ]]; then
  echo "warning: $VENV310_PY not found — OracleNet will fail when invoked. See live_web_demo/server_start.md." >&2
fi

if ! "$VENV_PY" -c "import flaredantic" 2>/dev/null; then
  echo "error: flaredantic is not installed in venv/. Run: $VENV_PY -m pip install flaredantic" >&2
  exit 1
fi

cd "$REPO_ROOT"
export FLARE_TUNNEL=1
exec "$VENV_PY" "$SCRIPT_DIR/server.py"
