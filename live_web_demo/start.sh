#!/usr/bin/env bash
# OracleNLI live demo launcher.
#
# Boots the Flask backend (which serves both the static UI and the
# /api/predict endpoint backed by OracleNet + OracleTF). The OracleNet
# pipeline shells out to the Python 3.10 venv (./venv310) for ELMo, so that
# venv must already exist — it is not created here.
#
# Usage (from anywhere):
#     ./live_web_demo/start.sh
#
# Open http://127.0.0.1:8765/ once the server reports "ready".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PY="$REPO_ROOT/venv/bin/python"
VENV310_PY="$REPO_ROOT/venv310/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "error: $VENV_PY not found. Create venv/ and 'pip install -r live_web_demo/requirements-venv.txt'." >&2
  exit 1
fi

if [[ ! -x "$VENV310_PY" ]]; then
  echo "warning: $VENV310_PY not found — OracleNet will fail when invoked. See live_web_demo/server_start.md." >&2
fi

cd "$REPO_ROOT"
exec "$VENV_PY" "$SCRIPT_DIR/server.py"
