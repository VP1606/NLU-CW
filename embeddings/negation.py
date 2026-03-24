"""Negation feature extraction combining multiple approaches."""

from typing import List
import spacy
import torch

nlp = spacy.load("en_core_web_sm")

# Negation trigger words
TRIGGERS = {
    "not",
    "no",
    "never",
    "n't",
    "neither",
    "nor",
    "without",
    "lack",
    "lacking",
    "fail",
    "fails",
    "failed",
    "hardly",
    "barely",
    "scarcely",
    "cannot",
}

# Boundary words that reset negation scope
BOUNDARIES = {",", ".", "but", "and", "or", "though", "although"}


def get_negation_flags(token_list: List[str], max_len: int = 64) -> torch.Tensor:
    """
    Compute combined negation features for a token sequence.

    Returns [max_len, 3] tensor where each dimension represents:
    - dim 0: is_neg - token is a negation trigger word
    - dim 1: in_scope - token is head verb of spaCy neg dependency
    - dim 2: boundary_scope - negation scope active via boundary tracking

    Args:
        token_list: List of word tokens
        max_len: Maximum sequence length

    Returns:
        Tensor of shape (max_len, 3) with float32 negation flags
    """
    tokens = [t.lower() for t in token_list[:max_len]]

    # Dimension 2: Boundary-based scope tracking
    boundary_flags = []
    in_scope = False
    for tok in tokens:
        if tok in TRIGGERS:
            in_scope = True
        if tok in BOUNDARIES:
            in_scope = False
        boundary_flags.append(1.0 if in_scope else 0.0)

    # Dimension 1: spaCy dependency-based scope
    doc = nlp(" ".join(tokens))
    negated_heads = {t.head.i for t in doc if t.dep_ == "neg"}
    dep_scope = [1.0 if t.i in negated_heads else 0.0 for t in doc]

    # Dimension 0: Is trigger word
    is_neg = [1.0 if t in TRIGGERS else 0.0 for t in tokens]

    # Combine dimensions
    flags = []
    for i in range(max_len):
        if i < len(tokens):
            flags.append(
                [
                    is_neg[i],
                    dep_scope[i] if i < len(dep_scope) else 0.0,
                    boundary_flags[i],
                ]
            )
        else:
            flags.append([0.0, 0.0, 0.0])

    return torch.tensor(flags, dtype=torch.float32)
