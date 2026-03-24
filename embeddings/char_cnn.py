"""Character-level CNN for word embeddings."""

from typing import Dict, List
import torch
import torch.nn as nn


def build_char_vocab(token_lists: List[List[str]]) -> Dict[str, int]:
    """
    Build character vocabulary from token lists.

    Args:
        token_lists: List of token sequences

    Returns:
        Dictionary mapping characters to indices
    """
    chars = set()
    for tokens in token_lists:
        for token in tokens:
            chars.update(token)

    char2idx = {"<PAD>": 0, "<UNK>": 1}
    for i, char in enumerate(sorted(chars)):
        if char not in char2idx:
            char2idx[char] = len(char2idx)

    return char2idx


def words_to_char_tensor(
    token_list: List[str],
    char2idx: Dict[str, int],
    max_word_len: int = 20,
    max_seq_len: int = 64,
) -> torch.Tensor:
    """
    Convert word tokens to character index tensor.

    Args:
        token_list: List of word tokens
        char2idx: Character to index mapping
        max_word_len: Maximum characters per word
        max_seq_len: Maximum sequence length

    Returns:
        Tensor of shape (max_seq_len, max_word_len) with character indices
    """
    pad = char2idx.get("<PAD>", 0)
    unk = char2idx.get("<UNK>", 1)

    result = []
    for i in range(max_seq_len):
        if i < len(token_list):
            word = token_list[i][: max_word_len]
            ids = [char2idx.get(c, unk) for c in word]
            ids = ids + [pad] * (max_word_len - len(ids))
        else:
            ids = [pad] * max_word_len
        result.append(ids)

    return torch.tensor(result, dtype=torch.long)


class CharCNN(nn.Module):
    """
    Character-level CNN for generating word embeddings.

    Architecture:
    - Char embedding: 30d
    - Conv1d: kernels (2,3,4) with filters (33,33,34)
    - Max pooling + concatenation → 100d output per word
    """

    def __init__(self, char_vocab_size: int, char_embed_dim: int = 30):
        super().__init__()

        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)

        # Three convolutional layers with different kernel sizes
        self.conv2 = nn.Conv1d(char_embed_dim, 33, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(char_embed_dim, 33, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(char_embed_dim, 34, kernel_size=4, padding=0)

        self.dropout = nn.Dropout(p=0.3)

        # Output: 33 + 33 + 34 = 100
        self.output_dim = 100

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: Tensor of shape (batch, seq_len, max_word_len)

        Returns:
            Tensor of shape (batch, seq_len, 100)
        """
        batch_size, seq_len, max_word_len = char_ids.size()

        # Reshape for embedding: (batch*seq_len, max_word_len)
        char_ids_flat = char_ids.view(-1, max_word_len)

        # Embed characters: (batch*seq_len, max_word_len, 30)
        char_embeds = self.char_embed(char_ids_flat)

        # Transpose for Conv1d: (batch*seq_len, 30, max_word_len)
        char_embeds = char_embeds.transpose(1, 2)

        # Apply convolutions and max pooling
        conv2_out = torch.max_pool1d(
            torch.relu(self.conv2(char_embeds)), kernel_size=max_word_len - 2 + 1
        )
        conv3_out = torch.max_pool1d(
            torch.relu(self.conv3(char_embeds)), kernel_size=max_word_len - 3 + 1
        )
        conv4_out = torch.max_pool1d(
            torch.relu(self.conv4(char_embeds)), kernel_size=max_word_len - 4 + 1
        )

        # Concatenate: (batch*seq_len, 100)
        cnn_out = torch.cat([conv2_out, conv3_out, conv4_out], dim=1).squeeze(-1)

        # Apply dropout
        cnn_out = self.dropout(cnn_out)

        # Reshape back to (batch, seq_len, 100)
        cnn_out = cnn_out.view(batch_size, seq_len, self.output_dim)

        return cnn_out
