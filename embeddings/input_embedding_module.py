"""Combined input embedding module with pre-computed embeddings."""

import torch
import torch.nn as nn

from .glove import build_glove_layer
from .char_cnn import CharCNN
from .pos_embedding import POSEmbedding


class InputEmbeddingModule(nn.Module):
    """
    Combines GloVe, pre-computed ELMo, CharCNN, POS, and negation features.

    All embeddings (ELMo, GloVe, CharCNN, POS, negation) are pre-computed and loaded
    from dataset. This module just concatenates them.

    Output dimension: 300 (GloVe) + 1024 (ELMo) + 100 (CharCNN) + 50 (POS) + 3 (negation) = 1477
    """

    def __init__(
        self,
        glove_layer,
        char_cnn,
        pos_embedding,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        # Pre-built embedding layers
        self.glove_layer = glove_layer  # frozen GloVe
        self.char_cnn = char_cnn  # learned CharCNN
        self.pos_embed = pos_embedding  # learned POS

        # Output dimension
        self.output_dim = 300 + 1024 + 100 + 50 + 3  # 1477

        # Final dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        token_ids: torch.Tensor,
        char_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_flags: torch.Tensor,
        elmo_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine all embeddings.

        Args:
            token_ids: Tensor of shape (batch, 64) — GloVe indices
            char_ids: Tensor of shape (batch, 64, 20) — character indices
            pos_ids: Tensor of shape (batch, 64) — POS tag indices
            neg_flags: Tensor of shape (batch, 64, 3) — negation features
            elmo_embeddings: Tensor of shape (batch, 64, 1024) — pre-computed ELMo

        Returns:
            Tensor of shape (batch, 64, 1477)
        """
        # GloVe embeddings: (batch, 64, 300)
        glove_out = self.glove_layer(token_ids)

        # CharCNN: (batch, 64, 100)
        char_out = self.char_cnn(char_ids)

        # POS embeddings: (batch, 64, 50)
        pos_out = self.pos_embed(pos_ids)

        # Concatenate all: (batch, 64, 1477)
        combined = torch.cat([glove_out, elmo_embeddings, char_out, pos_out, neg_flags], dim=-1)

        # Apply dropout
        output = self.dropout(combined)

        return output
