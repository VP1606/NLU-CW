"""POS tag embedding module."""

from typing import Dict, List
import spacy
import torch
import torch.nn as nn

nlp = spacy.load("en_core_web_sm")


def build_pos_vocab(token_lists: List[List[str]]) -> Dict[str, int]:
    """
    Build POS tag vocabulary from token lists using spaCy.

    Args:
        token_lists: List of token sequences

    Returns:
        Dictionary mapping POS tags to indices
    """
    pos_tags = set()

    for tokens in token_lists:
        doc = nlp(" ".join(tokens))
        for token in doc:
            pos_tags.add(token.pos_)

    pos2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, pos in enumerate(sorted(pos_tags)):
        if pos not in pos2idx:
            pos2idx[pos] = len(pos2idx)

    return pos2idx


def get_pos_ids(
    token_list: List[str], pos2idx: Dict[str, int], max_len: int = 64
) -> List[int]:
    """
    Get POS tag indices for a token sequence.

    Args:
        token_list: List of word tokens
        pos2idx: POS tag to index mapping
        max_len: Maximum sequence length

    Returns:
        List of POS tag indices padded to max_len
    """
    unk = pos2idx.get("<UNK>", 1)
    pad = pos2idx.get("<PAD>", 0)

    doc = nlp(" ".join(token_list))
    ids = [pos2idx.get(token.pos_, unk) for token in doc]

    # Pad to max_len
    ids = ids[:max_len] + [pad] * (max_len - len(ids))

    return ids


class POSEmbedding(nn.Module):
    """
    Learned POS tag embedding layer.

    Embedding dimension: 50
    """

    def __init__(self, pos_vocab_size: int, pos_embed_dim: int = 50):
        super().__init__()
        self.embedding = nn.Embedding(pos_vocab_size, pos_embed_dim, padding_idx=0)
        self.output_dim = pos_embed_dim

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_ids: Tensor of shape (batch, seq_len) with POS tag indices

        Returns:
            Tensor of shape (batch, seq_len, 50)
        """
        return self.embedding(pos_ids)
