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

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oracle_net.predict import OracleNetPredictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oraclenli")

app = Flask(__name__, static_folder=None)

log.info("Loading OracleNetPredictor (this can take ~30s on first run)...")
PREDICTOR = OracleNetPredictor()
log.info("OracleNetPredictor ready on device=%s", PREDICTOR.device)


@app.get("/")
def index():
    return send_from_directory(HERE, "NLI Demo.html")


@app.get("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(HERE, filename)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    premise = (payload.get("premise") or "").strip()
    hypothesis = (payload.get("hypothesis") or "").strip()
    model = payload.get("model") or "OracleNet"

    if not premise or not hypothesis:
        return jsonify({"error": "premise and hypothesis are required"}), 400

    if model != "OracleNet":
        return jsonify(
            {"error": f"model {model!r} is not wired up yet — only OracleNet is live"}
        ), 501

    try:
        result = PREDICTOR.predict(premise, hypothesis)
    except Exception as e:
        log.exception("predict failed")
        return jsonify({"error": str(e)}), 500

    return jsonify({"model": model, **result.to_dict()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False, use_reloader=False)
