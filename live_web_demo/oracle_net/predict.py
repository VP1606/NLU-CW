"""
OracleNet single-sample inference wrapper.

Mirrors the pipeline in ../../demo.ipynb, but accepts a single (premise,
hypothesis) pair as Python strings instead of a bulk CSV. Internally it writes
a temporary 1-row CSV and reuses the existing EmbeddingPrecomputer +
ResESIM_Dataset + OracleNet pipeline so behaviour matches the notebook exactly.

Designed to be imported by the web backend: instantiate `OracleNetPredictor`
once at server startup, then call `.predict(premise, hypothesis)` per request.

CLI usage (from repo root):
    python live_web_demo/oracle_net/predict.py \
        --premise "A person on a horse jumps over a broken down airplane." \
        --hypothesis "A person is outdoors, on a horse."
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Resolve repo root (parents: oracle_net -> live_web_demo -> repo root) and put
# it on sys.path so `res_esim` and `precomputeClasses` resolve the same way as
# the notebook.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precomputeClasses import EmbeddingPrecomputer  # noqa: E402
from res_esim.loader.res_esim_dataset import ResESIM_Dataset  # noqa: E402
from res_esim.model_layers.oracle_net import OracleNet  # noqa: E402

# Convention from data/train.csv: 1 = entailment, 0 = not entailment.
LABEL_NAMES = {0: "Not Entailment", 1: "Entailment"}

# Maps "[N/5]" prefix in precomputeClasses stdout to a short, user-facing label.
_PRECOMPUTE_STAGE_LABELS = {
    "1/5": "Loading meta",
    "2/5": "Building embedding layers",
    "3/5": "Pre-computing ELMo (Peters et al., 2018)",
    "4/5": "Pre-computing 1477d embeddings",
    "5/5": "Saving embeddings",
}


class _StdoutTap:
    """Line-buffered sink that forwards parsed precompute stages to a callback.

    Mirrors writes to the original stdout so the existing logs still surface
    in the server console.
    """

    def __init__(self, callback: Callable[[dict], None], passthrough):
        self._cb = callback
        self._passthrough = passthrough
        self._buf = ""

    def write(self, s: str) -> int:
        if self._passthrough is not None:
            self._passthrough.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, _, self._buf = self._buf.partition("\n")
            line = line.strip()
            if not line:
                continue
            for key, label in _PRECOMPUTE_STAGE_LABELS.items():
                if line.startswith(f"[{key}]"):
                    self._cb({"stage": key, "label": label})
                    break
        return len(s)

    def flush(self) -> None:
        if self._passthrough is not None:
            self._passthrough.flush()


@dataclass
class PredictionResult:
    label: str
    label_id: int
    confidence: float
    probabilities: dict  # {"Entailment": float, "Not Entailment": float}

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class OracleNetPredictor:
    """Loads OracleNet + the embedding precomputer once; reusable per request."""

    def __init__(
        self,
        repo_root: Path | str = REPO_ROOT,
        meta_path: str | None = None,
        elmo_options: str | None = None,
        elmo_weights: str | None = None,
        elmo_venv_python: str | None = None,
        model_weights: str | None = None,
        device: torch.device | str | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()

        self.meta_path = meta_path or str(self.repo_root / "notebook_data" / "meta.pt")
        self.elmo_options = elmo_options or str(
            self.repo_root / "notebook_data" / "elmo_model" / "options.json"
        )
        self.elmo_weights = elmo_weights or str(
            self.repo_root / "notebook_data" / "elmo_model" / "weights.hdf5"
        )
        self.elmo_venv_python = elmo_venv_python or str(
            self.repo_root / "venv310" / "bin" / "python"
        )
        self.model_weights = model_weights or str(
            self.repo_root / "final_model_versions" / "ff2f02d4" / "best_model.pt"
        )

        self._verify_paths()

        self.device = torch.device(device) if device else self._auto_device()

        # EmbeddingPrecomputer is stateful (loads meta, builds embedders) so we
        # construct it once and reuse it across predict() calls.
        self.precomputer = EmbeddingPrecomputer(
            meta_path=self.meta_path,
            elmo_options=self.elmo_options,
            elmo_weights=self.elmo_weights,
            elmo_venv=self.elmo_venv_python,
        )

        self.model = OracleNet(
            input_dim=1477,
            hidden_dim=512,
            num_blocks=3,
            num_classes=2,
            dropout_rate=0.2,
            num_heads=8,
        )
        self.model.load_state_dict(
            torch.load(self.model_weights, map_location=self.device, weights_only=False)
        )
        self.model.to(self.device).eval()

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _verify_paths(self) -> None:
        required = {
            "meta": self.meta_path,
            "elmo_options": self.elmo_options,
            "elmo_weights": self.elmo_weights,
            "elmo_venv_python": self.elmo_venv_python,
            "model_weights": self.model_weights,
        }
        missing = [(k, v) for k, v in required.items() if not Path(v).exists()]
        if missing:
            lines = "\n".join(f"  - {k}: {v}" for k, v in missing)
            raise FileNotFoundError(
                f"OracleNetPredictor: missing required files:\n{lines}"
            )

    @torch.no_grad()
    def predict(
        self,
        premise: str,
        hypothesis: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> PredictionResult:
        if not premise or not premise.strip():
            raise ValueError("premise must be a non-empty string")
        if not hypothesis or not hypothesis.strip():
            raise ValueError("hypothesis must be a non-empty string")

        def _emit(event: dict) -> None:
            if progress_callback is not None:
                progress_callback(event)

        # Reuse the exact bulk pipeline by writing a 1-row CSV and pointing
        # EmbeddingPrecomputer at it.
        with tempfile.TemporaryDirectory(prefix="oraclenet_") as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "input.csv"
            npz_path = tmp / "embeddings.npz"

            _emit({"stage": "input", "label": "Preparing input"})
            pd.DataFrame(
                [{"premise": premise, "hypothesis": hypothesis}]
            ).to_csv(csv_path, index=False)

            _emit({"stage": "precompute", "label": "Pre-computing embeddings"})
            if progress_callback is not None:
                tap = _StdoutTap(progress_callback, sys.__stdout__)
                with contextlib.redirect_stdout(tap):
                    self.precomputer.run(
                        csv_path=str(csv_path), output_npz=str(npz_path)
                    )
            else:
                self.precomputer.run(
                    csv_path=str(csv_path), output_npz=str(npz_path)
                )

            _emit({"stage": "inference", "label": "Running OracleNet inference"})
            dataset = ResESIM_Dataset(str(npz_path))
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            batch = next(iter(loader))

            logits = self.model(
                batch["premise_embedding"].to(self.device),
                batch["hyp_embedding"].to(self.device),
                batch["premise_length"].to(self.device),
                batch["hyp_length"].to(self.device),
            )

        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        label_id = int(max(range(len(probs)), key=probs.__getitem__))
        return PredictionResult(
            label=LABEL_NAMES[label_id],
            label_id=label_id,
            confidence=float(probs[label_id]),
            probabilities={LABEL_NAMES[i]: float(p) for i, p in enumerate(probs)},
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="OracleNet single-pair inference")
    parser.add_argument("--premise", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device override (e.g. cpu, mps, cuda). Default: auto.",
    )
    args = parser.parse_args()

    predictor = OracleNetPredictor(device=args.device)
    result = predictor.predict(args.premise, args.hypothesis)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
