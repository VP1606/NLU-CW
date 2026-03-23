from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ResESIM_Dataset(Dataset):
    """
    Loads precomputed ELMo embeddings and CSV labels for OracleNet.
    Optionally appends negation flags as an extra feature dimension.

    Args:
        prem_npy      : path to .npy of shape (N, max_len, 1024)
        hyp_npy       : path to .npy of shape (N, max_len, 1024)
        csv_path      : path to CSV with a 'label' column
        negation_path : (optional) path to .pt with keys
                        'premise_negation' and 'hypothesis_negation'
                        (lists of N variable-length int lists)
    """

    def __init__(
        self,
        prem_npy: Path,
        hyp_npy: Path,
        csv_path: Path,
        negation_path: Path = None,
    ):
        self.prem = np.load(prem_npy)  # (N, max_len, 1024)
        self.hyp = np.load(hyp_npy)  # (N, max_len, 1024)

        labels = pd.read_csv(csv_path)["label"].tolist()
        self.labels = labels

        # Sequence length = number of non-zero token positions
        self.prem_lengths = (self.prem.any(axis=-1)).sum(axis=-1)  # (N,)
        self.hyp_lengths = (self.hyp.any(axis=-1)).sum(axis=-1)  # (N,)

        if negation_path is not None:
            neg = torch.load(negation_path)
            self.prem = self._append_flags(self.prem, neg["premise_negation"])
            self.hyp = self._append_flags(self.hyp, neg["hypothesis_negation"])

    def _append_flags(self, embeddings: np.ndarray, flag_lists: list) -> np.ndarray:
        """
        Pads each flag list to max_len and concatenates as a single
        extra feature column.

        embeddings : (N, max_len, dim)
        flag_lists : list of N variable-length int lists
        returns    : (N, max_len, dim + 1)
        """
        N, max_len, _ = embeddings.shape
        flags = np.zeros((N, max_len, 1), dtype=np.float32)
        for i, f in enumerate(flag_lists):
            l = min(len(f), max_len)
            flags[i, :l, 0] = f[:l]
        return np.concatenate([embeddings, flags], axis=-1)  # (N, max_len, dim+1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "premise_embedding": torch.tensor(self.prem[idx], dtype=torch.float32),
            "hyp_embedding": torch.tensor(self.hyp[idx], dtype=torch.float32),
            "premise_length": torch.tensor(self.prem_lengths[idx], dtype=torch.long),
            "hyp_length": torch.tensor(self.hyp_lengths[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
