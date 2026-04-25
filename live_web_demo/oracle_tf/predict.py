"""
OracleTF (Category C) single-sample inference wrapper.

Mirrors the ensemble pipeline in ../../inference_demo_transformer.ipynb but
takes a single (premise, hypothesis) pair instead of a CSV. Loads both
fine-tuned ModernBERT-Large checkpoints once and runs them per request.

Pipeline:
    1. Tokenize "{premise} [SEP] {hypothesis}" (max_length=256, padding="max_length").
    2. Forward pass through `modernBerta` -> softmax probs.
    3. Forward pass through `taskSource`  -> softmax probs.
    4. Ensemble: 0.4 * modernBerta + 0.6 * taskSource.
    5. Decision: entailment iff ensemble entailment prob >= 0.43.

Designed to be imported by the web backend: instantiate `OracleTFPredictor`
once at startup, then call `.predict(premise, hypothesis)` per request.

CLI usage (from repo root):
    python live_web_demo/oracle_tf/predict.py \\
        --premise "..." --hypothesis "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

# Same convention as OracleNet / data/train.csv.
LABEL_NAMES = {0: "Not Entailment", 1: "Entailment"}

# Calibrated ensemble settings from inference_demo_transformer.ipynb.
W_MODERNBERT = 0.4
W_TASKSOURCE = 0.6
DECISION_THRESHOLD = 0.43
MAX_LENGTH = 256


class OracleTFPredictor:
    """Loads both ModernBERT-Large checkpoints once; reusable per request."""

    def __init__(
        self,
        repo_root: Path | str = REPO_ROOT,
        modernbert_dir: str | None = None,
        tasksource_dir: str | None = None,
        device: torch.device | str | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()

        self.modernbert_dir = modernbert_dir or str(
            self.repo_root / "final_model_versions" / "modernBerta"
        )
        self.tasksource_dir = tasksource_dir or str(
            self.repo_root / "final_model_versions" / "taskSource"
        )

        self._verify_paths()

        self.device = torch.device(device) if device else self._auto_device()

        self.modernbert_tokenizer = AutoTokenizer.from_pretrained(self.modernbert_dir)
        self.modernbert_model = (
            AutoModelForSequenceClassification.from_pretrained(self.modernbert_dir)
            .to(self.device)
            .eval()
        )

        self.tasksource_tokenizer = AutoTokenizer.from_pretrained(self.tasksource_dir)
        self.tasksource_model = (
            AutoModelForSequenceClassification.from_pretrained(self.tasksource_dir)
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _verify_paths(self) -> None:
        missing = []
        for name, path in (
            ("modernBerta", self.modernbert_dir),
            ("taskSource", self.tasksource_dir),
        ):
            p = Path(path)
            if not p.is_dir():
                missing.append((name, path, "directory missing"))
                continue
            weights = p / "model.safetensors"
            if not weights.exists() or weights.stat().st_size < 1024:
                missing.append((name, str(weights), "weights missing or LFS pointer"))
        if missing:
            lines = "\n".join(f"  - {n}: {p}  ({why})" for n, p, why in missing)
            raise FileNotFoundError(
                f"OracleTFPredictor: missing required model files:\n{lines}"
            )

    @torch.no_grad()
    def _model_probs(
        self,
        tokenizer,
        model,
        text: str,
    ) -> torch.Tensor:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return F.softmax(logits, dim=-1).squeeze(0).cpu()

    @torch.no_grad()
    def predict(
        self,
        premise: str,
        hypothesis: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        if not premise or not premise.strip():
            raise ValueError("premise must be a non-empty string")
        if not hypothesis or not hypothesis.strip():
            raise ValueError("hypothesis must be a non-empty string")

        def _emit(event: dict) -> None:
            if progress_callback is not None:
                progress_callback(event)

        text = f"{premise} [SEP] {hypothesis}"

        _emit({"stage": "modernbert", "label": "Running ModernBERT-Large"})
        modern_probs = self._model_probs(
            self.modernbert_tokenizer, self.modernbert_model, text
        )

        _emit({"stage": "tasksource", "label": "Running tasksource ModernBERT"})
        task_probs = self._model_probs(
            self.tasksource_tokenizer, self.tasksource_model, text
        )

        _emit({"stage": "ensemble", "label": "Ensembling predictions"})
        ensemble = W_MODERNBERT * modern_probs + W_TASKSOURCE * task_probs
        prob_entailment = float(ensemble[1])
        label_id = 1 if prob_entailment >= DECISION_THRESHOLD else 0
        confidence = prob_entailment if label_id == 1 else 1.0 - prob_entailment

        return {
            "label": LABEL_NAMES[label_id],
            "label_id": label_id,
            "confidence": confidence,
            "probabilities": {
                LABEL_NAMES[i]: float(ensemble[i]) for i in range(len(ensemble))
            },
            "decision_threshold": DECISION_THRESHOLD,
            "ensemble_weights": {
                "modernBerta": W_MODERNBERT,
                "taskSource": W_TASKSOURCE,
            },
            "per_model": {
                "modernBerta": {
                    LABEL_NAMES[i]: float(modern_probs[i])
                    for i in range(len(modern_probs))
                },
                "taskSource": {
                    LABEL_NAMES[i]: float(task_probs[i])
                    for i in range(len(task_probs))
                },
            },
        }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="OracleTF single-pair inference")
    parser.add_argument("--premise", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    predictor = OracleTFPredictor(device=args.device)
    result = predictor.predict(args.premise, args.hypothesis)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
