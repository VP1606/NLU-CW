#!/usr/bin/env bash
# OracleNLI live demo launcher with a public Cloudflare quick tunnel.
#
# Boots the Flask backend and a flaredantic-managed `cloudflared` tunnel
# pointing at it. Anyone with the printed *.trycloudflare.com URL can hit
# /api/predict and consume your local CPU/MPS time, so kill this script
# (Ctrl+C) when you're done.
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
PORT="${PORT:-8765}"

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

# Start the Flask server in the background; remember its PID so we can clean
# up if the tunnel script exits.
"$VENV_PY" "$SCRIPT_DIR/server.py" &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Run the tunnel in the foreground; flaredantic prints the URL when ready and
# the tunnel stays open until SIGTERM/Ctrl+C.
exec "$VENV_PY" - "$PORT" <<'PY'
import signal, sys, time
from flaredantic import FlareConfig, FlareTunnel

port = int(sys.argv[1])
tunnel = FlareTunnel(FlareConfig(port=port, verbose=False))
url = tunnel.start()
print()
print("=" * 60)
print(f"  Public URL: {url}")
print(f"  Local URL:  http://127.0.0.1:{port}/")
print("=" * 60)
print("  Press Ctrl+C to stop.")
print(flush=True)

def stop(*_):
    tunnel.stop()
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

while True:
    time.sleep(3600)
PY
