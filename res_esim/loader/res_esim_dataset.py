from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from util.tokenization import tokenise
from embeddings.negation import get_negation_flags
from embeddings.char_cnn import words_to_char_tensor
from embeddings.pos_embedding import get_pos_ids

MAX_LEN = 64
MAX_WORD_LEN = 20


class ResESIM_Dataset(Dataset):
    """
    Dataset with pre-computed embeddings (ELMo, GloVe, CharCNN, POS, negation).

    Args:
        csv_path  : path to CSV with 'premise', 'hypothesis', 'label' columns
        elmo_path : path to .npy with pre-computed ELMo [N, 64, 1024]
        vocab     : Vocabulary object (word2idx mapping)
        char2idx  : dict mapping characters to indices
        pos2idx   : dict mapping POS tags to indices
    """

    def __init__(
        self,
        csv_path: Path,
        elmo_path: Path,
        vocab,
        char2idx: dict,
        pos2idx: dict,
    ):
        # Load pre-computed ELMo
        self.elmo_embeddings = np.load(elmo_path)  # (N, 64, 1024)

        # Load CSV
        df = pd.read_csv(csv_path)
        self.labels = df["label"].astype(int).tolist()
        self.prem_tokens = [tokenise(s) for s in df["premise"]]
        self.hyp_tokens = [tokenise(s) for s in df["hypothesis"]]

        # Vocab lookups
        self.vocab = vocab
        self.char2idx = char2idx
        self.pos2idx = pos2idx

    def __len__(self):
        return len(self.labels)

    def _get_token_ids(self, tokens):
        ids = self.vocab.encode(tokens)[: MAX_LEN]
        return ids + [0] * (MAX_LEN - len(ids))

    def _get_mask(self, tokens):
        real_len = min(len(tokens), MAX_LEN)
        return [1] * real_len + [0] * (MAX_LEN - real_len)

    def __getitem__(self, idx):
        prem_tok = self.prem_tokens[idx]
        hyp_tok = self.hyp_tokens[idx]

        return {
            # Pre-computed ELMo
            "elmo_embedding": torch.tensor(self.elmo_embeddings[idx], dtype=torch.float32),
            # GloVe indices
            "premise_ids": torch.tensor(self._get_token_ids(prem_tok), dtype=torch.long),
            "hypothesis_ids": torch.tensor(self._get_token_ids(hyp_tok), dtype=torch.long),
            # CharCNN
            "premise_char": words_to_char_tensor(
                prem_tok, self.char2idx, MAX_WORD_LEN, MAX_LEN
            ),
            "hypothesis_char": words_to_char_tensor(
                hyp_tok, self.char2idx, MAX_WORD_LEN, MAX_LEN
            ),
            # POS
            "premise_pos": torch.tensor(
                get_pos_ids(prem_tok, self.pos2idx, MAX_LEN), dtype=torch.long
            ),
            "hypothesis_pos": torch.tensor(
                get_pos_ids(hyp_tok, self.pos2idx, MAX_LEN), dtype=torch.long
            ),
            # Negation 3d
            "premise_neg": get_negation_flags(prem_tok, MAX_LEN),
            "hypothesis_neg": get_negation_flags(hyp_tok, MAX_LEN),
            # Masks
            "premise_mask": torch.tensor(self._get_mask(prem_tok), dtype=torch.long),
            "hypothesis_mask": torch.tensor(self._get_mask(hyp_tok), dtype=torch.long),
            # Label
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
