"""Vocabulary builder and encoder."""

from collections import Counter
from typing import List, Dict


class Vocabulary:
    """Builds vocabulary from token lists with min frequency threshold."""

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_freq = Counter()

    def build(self, token_lists: List[List[str]]) -> None:
        """
        Build vocabulary from token lists.

        Args:
            token_lists: List of token sequences
        """
        # Count word frequencies
        for tokens in token_lists:
            self.word_freq.update(tokens)

        # Add words that meet minimum frequency threshold
        idx = 2  # Start after PAD and UNK
        for word, freq in self.word_freq.most_common():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Encode a sequence of tokens to indices.

        Args:
            tokens: List of token strings

        Returns:
            List of token indices, using UNK for unknown words
        """
        unk_idx = self.word2idx["<UNK>"]
        return [self.word2idx.get(token, unk_idx) for token in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Decode a sequence of indices to tokens.

        Args:
            indices: List of token indices

        Returns:
            List of token strings
        """
        return [self.idx2word.get(idx, "<UNK>") for idx in indices]

    def __len__(self) -> int:
        return len(self.word2idx)
