from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ResESIM_Dataset(Dataset):
    """
    Loads precomputed ELMo embeddings and CSV labels for OracleNet.

    Args:
        prem_npy  : path to .npy of shape (N, max_len, 1024)
        hyp_npy   : path to .npy of shape (N, max_len, 1024)
        csv_path  : path to CSV with a 'label' column
    """

    def __init__(self, prem_npy: Path, hyp_npy: Path, csv_path: Path):
        self.prem = np.load(prem_npy)  # (N, max_len, 1024)
        self.hyp = np.load(hyp_npy)  # (N, max_len, 1024)

        labels = pd.read_csv(csv_path)["label"].tolist()
        self.labels = labels

        # Sequence length = number of non-zero token positions
        self.prem_lengths = (self.prem.any(axis=-1)).sum(axis=-1)  # (N,)
        self.hyp_lengths = (self.hyp.any(axis=-1)).sum(axis=-1)  # (N,)

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
