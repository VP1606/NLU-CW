# Live Web Demo — Launch Guide

This folder contains the OracleNLI prototype web UI (designed via Claude Design). It is a single-page React app loaded via Babel-standalone from CDN, with a sibling `tweaks-panel.jsx`. Because the HTML loads the `.jsx` file as a sibling, it **must be served over HTTP** — opening `NLI Demo.html` directly via `file://` will fail.

## Files

- `NLI Demo.html` — main page (React app, mock inference logic in `handlePredict`).
- `tweaks-panel.jsx` — reusable tweaks panel (theme, accent hue).
- `design-canvas.jsx` — design-tool artifact, not loaded by the page.
- `NLU-Textual-Entailment.zip` — original archive.

## Launch

From the repo root (`NLU-CW/`):

```bash
cd live_web_demo
../venv/bin/python -m http.server 8765
```

Then open: <http://localhost:8765/NLI%20Demo.html>

To stop the server: `Ctrl+C` (or kill the background process if launched detached).

## Current behaviour (prototype only)

- Predict button runs a fake inference: ~0.8–2 s delay, then a random `Entailment` / `Not Entailment` label with random confidence.
- Model toggle (OracleNet / OracleTF) and example loader are wired into UI state but not yet connected to real models.
- Tweaks panel (theme: dark/light/editorial, accent hue) is fully functional.

The real model wiring will be added in a later pair-programming pass.
