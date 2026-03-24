"""GloVe embedding loading and layer building."""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .vocab import Vocabulary


def load_glove(
    glove_path: Path, vocab: Vocabulary, embed_dim: int = 300
) -> np.ndarray:
    """
    Load GloVe vectors and build embedding matrix for vocabulary.

    Args:
        glove_path: Path to GloVe file (bin/glove/glove.6B.300d.txt)
        vocab: Vocabulary object
        embed_dim: Embedding dimension (default 300)

    Returns:
        Embedding matrix of shape (vocab_size, embed_dim)
        - PAD (idx 0): zeros
        - UNK (idx 1): mean of all vectors
        - Unknown words: random U(-0.1, 0.1)
    """
    # Read GloVe file
    glove_vectors = {}
    all_vectors = []

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            glove_vectors[word] = vector
            all_vectors.append(vector)

    # Compute mean vector for UNK
    all_vectors = np.array(all_vectors)
    mean_vector = all_vectors.mean(axis=0)

    # Build embedding matrix with random initialization
    vocab_size = len(vocab)
    embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim)).astype(
        np.float32
    )

    # PAD = zeros
    embedding_matrix[0] = np.zeros(embed_dim, dtype=np.float32)

    # UNK = mean vector
    embedding_matrix[1] = mean_vector

    # Fill in known words
    for word, idx in vocab.word2idx.items():
        if word in glove_vectors and word not in ["<PAD>", "<UNK>"]:
            embedding_matrix[idx] = glove_vectors[word]

    return embedding_matrix


def build_glove_layer(embedding_matrix: np.ndarray, freeze: bool = True) -> nn.Embedding:
    """
    Build a frozen GloVe embedding layer.

    Args:
        embedding_matrix: Numpy array of shape (vocab_size, embed_dim)
        freeze: Whether to freeze the layer (default True)

    Returns:
        nn.Embedding layer with requires_grad=False
    """
    vocab_size, embed_dim = embedding_matrix.shape
    embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    embedding.weight.data = torch.from_numpy(embedding_matrix)

    if freeze:
        embedding.weight.requires_grad = False

    return embedding
