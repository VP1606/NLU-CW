# Manual server start

Use [`start.sh`](start.sh) for the one-shot launcher. This document covers the
underlying steps if you want to run them yourself.

All paths below are relative to the **repo root**.

## 1. Prerequisites

- `venv/` — Python 3.13 virtualenv with the web-demo runtime deps.
- `venv310/` — Python 3.10 virtualenv with `allennlp==2.10.1` for ELMo (used
  only by OracleNet).
- Model + embedding files in place (see [file_requirements.md](file_requirements.md)).

### Create `venv/`

```bash
python3 -m venv venv
venv/bin/pip install -r live_web_demo/requirements-venv.txt
venv/bin/python -m spacy download en_core_web_sm
```

### Create `venv310/` (OracleNet only)

```bash
python3.10 -m venv venv310
venv310/bin/pip install -r live_web_demo/requirements-310.txt
```

## 2. Start the server

```bash
venv/bin/python live_web_demo/server.py
```

The server:
- loads `OracleNetPredictor` (loads `meta.pt`, builds embedding layers, loads
  OracleNet weights — ~10 s),
- loads `OracleTFPredictor` (loads two ModernBERT-Large checkpoints — ~30 s on
  first run, faster on warm cache; logs a warning and continues without
  OracleTF if the weights are missing),
- binds `127.0.0.1:8765`.

Wait for the log line `OracleTFPredictor ready on device=...`, then open
<http://127.0.0.1:8765/>.

## 3. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/` | Serves `NLI Demo.html`. |
| `GET`  | `/<filename>` | Serves any other static file under `live_web_demo/`. |
| `POST` | `/api/predict` | Streams ndjson progress events + a final result for the chosen model. Body: `{"premise": "...", "hypothesis": "...", "model": "OracleNet" \| "OracleTF"}`. |

## 4. Stop the server

`Ctrl+C` in the terminal, or `pkill -f live_web_demo/server.py` if it's
backgrounded.
