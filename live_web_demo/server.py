"""
Flask backend for the OracleNLI live web demo.

Serves the static prototype (HTML + sibling .jsx files) AND a /api/predict
endpoint that runs OracleNet on a single (premise, hypothesis) pair via
oracle_net.predict.OracleNetPredictor.

The predictor is loaded once at startup (heavy: meta + ELMo subprocess +
OracleNet weights) and reused across requests.

Run from repo root:
    venv/bin/python live_web_demo/server.py

Then open: http://localhost:8765/NLI%20Demo.html
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oracle_net.predict import OracleNetPredictor  # noqa: E402
from oracle_tf.predict import OracleTFPredictor  # noqa: E402

# Pull the file groups straight from the source of truth so the UI mirrors
# whatever hf_connect/file_repo.py declares.
sys.path.insert(0, str(HERE.parent))
from hf_connect.file_repo import (  # noqa: E402
    ELMO_FILES,
    MODERN_BERTA_FILES,
    ORACLE_NET_FILES,
    TASK_SOURCE_FILES,
)

STATUS_GROUPS = [
    ("OracleNet Model Files", ORACLE_NET_FILES),
    ("OracleTF ModernBerta Files", MODERN_BERTA_FILES),
    ("OracleTF TaskSource Files", TASK_SOURCE_FILES),
    ("ELMO Files", ELMO_FILES),
]
REPO_ROOT = HERE.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oraclenli")

app = Flask(__name__, static_folder=None)

log.info("Loading OracleNetPredictor (this can take ~30s on first run)...")
ORACLE_NET = OracleNetPredictor()
log.info("OracleNetPredictor ready on device=%s", ORACLE_NET.device)

log.info("Loading OracleTFPredictor (this loads two ModernBERT-Large checkpoints)...")
try:
    ORACLE_TF = OracleTFPredictor()
    log.info("OracleTFPredictor ready on device=%s", ORACLE_TF.device)
except FileNotFoundError as e:
    ORACLE_TF = None
    log.warning("OracleTFPredictor unavailable: %s", e)

# Stdout redirection inside OracleNet's predict() is process-global, so only
# one prediction can run at a time. A lock keeps concurrent requests serial
# regardless of which model is selected.
_PREDICT_LOCK = threading.Lock()


@app.get("/")
def index():
    return send_from_directory(HERE, "NLI Demo.html")


@app.get("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(HERE, filename)


def _is_present(local_path: str) -> bool:
    """A file counts as present if it exists, is non-empty, and is not an LFS
    pointer stub. The LFS check is a cheap defence so the UI doesn't lie when
    the user has the placeholder file but not the real content."""
    p = Path(local_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        with p.open("rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return False
    except OSError:
        return False
    return True


@app.get("/api/status")
def status():
    groups = []
    for label, files in STATUS_GROUPS:
        missing = [f.local_path for f in files if not _is_present(f.local_path)]
        groups.append({
            "label": label,
            "ok": not missing,
            "missing": missing,
            "total": len(files),
        })
    return jsonify({"groups": groups})


@app.post("/api/predict")
def predict():
    """Stream prediction progress as newline-delimited JSON.

    Each line is one event: {"type": "progress", "stage": ..., "label": ...}
    or {"type": "result", ...} or {"type": "error", "error": ...}.
    """
    payload = request.get_json(silent=True) or {}
    premise = (payload.get("premise") or "").strip()
    hypothesis = (payload.get("hypothesis") or "").strip()
    model = payload.get("model") or "OracleNet"

    if not premise or not hypothesis:
        return jsonify({"error": "premise and hypothesis are required"}), 400

    if model == "OracleNet":
        predictor = ORACLE_NET
    elif model == "OracleTF":
        if ORACLE_TF is None:
            return jsonify(
                {"error": "OracleTF unavailable — model weights are missing on this server"}
            ), 503
        predictor = ORACLE_TF
    else:
        return jsonify({"error": f"unknown model {model!r}"}), 400

    def event_stream():
        events: queue.Queue = queue.Queue()
        SENTINEL = object()
        outcome = {}

        def on_progress(ev: dict) -> None:
            events.put({"type": "progress", **ev})

        def worker() -> None:
            try:
                with _PREDICT_LOCK:
                    result = predictor.predict(
                        premise, hypothesis, progress_callback=on_progress
                    )
                # OracleNet returns a PredictionResult dataclass; OracleTF returns a dict.
                outcome["result"] = result.to_dict() if hasattr(result, "to_dict") else result
            except Exception as e:  # noqa: BLE001
                log.exception("predict failed")
                outcome["error"] = str(e)
            finally:
                events.put(SENTINEL)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            ev = events.get()
            if ev is SENTINEL:
                break
            yield json.dumps(ev) + "\n"

        if "error" in outcome:
            yield json.dumps({"type": "error", "error": outcome["error"]}) + "\n"
        else:
            yield json.dumps(
                {"type": "result", "model": model, **outcome["result"]}
            ) + "\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="application/x-ndjson",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False, use_reloader=False)
