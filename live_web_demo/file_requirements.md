# File requirements

Files needed by each model component, with paths relative to the **repo root**.
Run [`hf_connect/`](../hf_connect/) to download them from Hugging Face, or
provide your own copies at the listed paths.

## OracleNet (Category B — ResESIM + 1477d embeddings)

| File | Path | Notes |
|---|---|---|
| OracleNet weights | `final_model_versions/ff2f02d4/best_model.pt` | ~87 MB. Loaded by `OracleNetPredictor.__init__`. |
| Vocab + GloVe meta | `notebook_data/meta.pt` | ~1 MB. Vocab, char2idx, pos2idx, GloVe matrix. Regenerable via `build_meta.py`. |
| ELMo options | `notebook_data/elmo_model/options.json` | ~600 B. AllenNLP "Original" 5.5B variant. |
| ELMo weights | `notebook_data/elmo_model/weights.hdf5` | ~374 MB. AllenNLP "Original" 5.5B variant. |
| Python 3.10 venv | `venv310/bin/python` | Created manually — see [server_start.md](server_start.md). Used to run AllenNLP ELMo as a subprocess. |

## OracleTF (Category C — ModernBERT-Large ensemble)

| File | Path | Notes |
|---|---|---|
| modernBerta weights | `final_model_versions/modernBerta/model.safetensors` | ~1.47 GB. Fine-tuned ModernBERT-Large. |
| modernBerta config | `final_model_versions/modernBerta/config.json` | Required by `AutoModelForSequenceClassification`. |
| modernBerta tokenizer | `final_model_versions/modernBerta/tokenizer.json` | Required by `AutoTokenizer`. |
| modernBerta tokenizer config | `final_model_versions/modernBerta/tokenizer_config.json` | Required by `AutoTokenizer`. |
| taskSource weights | `final_model_versions/taskSource/model.safetensors` | ~1.47 GB. Fine-tuned ModernBERT-Large (tasksource init). |
| taskSource config | `final_model_versions/taskSource/config.json` | Required by `AutoModelForSequenceClassification`. |
| taskSource tokenizer | `final_model_versions/taskSource/tokenizer.json` | Required by `AutoTokenizer`. |
| taskSource tokenizer config | `final_model_versions/taskSource/tokenizer_config.json` | Required by `AutoTokenizer`. |

## Web demo (live_web_demo/)

| File | Path | Notes |
|---|---|---|
| Page | `live_web_demo/NLI Demo.html` | React app served by Flask at `/`. |
| Tweaks panel | `live_web_demo/tweaks-panel.jsx` | Loaded as a sibling .jsx; must be served, not opened via `file://`. |
| OracleNet wrapper | `live_web_demo/oracle_net/predict.py` | `OracleNetPredictor` class. |
| OracleTF wrapper | `live_web_demo/oracle_tf/predict.py` | `OracleTFPredictor` class. |
| Backend | `live_web_demo/server.py` | Flask app; loads both predictors at startup. |
