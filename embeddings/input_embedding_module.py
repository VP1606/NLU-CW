"""Combined input embedding module with real-time ELMo inference."""

import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids

from .glove import build_glove_layer
from .char_cnn import CharCNN
from .pos_embedding import POSEmbedding


class InputEmbeddingModule(nn.Module):
    """
    Combines GloVe, real-time ELMo, CharCNN, POS, and negation features.

    ELMo is run on-the-fly for each batch using AllenNLP's Elmo module.
    Output dimension: 300 (GloVe) + 1024 (ELMo) + 100 (CharCNN) + 50 (POS) + 3 (negation) = 1477
    """

    def __init__(
        self,
        elmo_options,
        elmo_weights,
        glove_layer,
        char_cnn,
        pos_embedding,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        # ELMo — real-time contextual embeddings (Peters et al., 2018)
        # num_output_representations=1 returns mean across 3 layers
        self.elmo = Elmo(
            elmo_options, elmo_weights, num_output_representations=1, dropout=0
        )

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
        raw_tokens,
    ) -> torch.Tensor:
        """
        Combine all embeddings with real-time ELMo inference.

        Args:
            token_ids: Tensor of shape (batch, 64) — GloVe indices
            char_ids: Tensor of shape (batch, 64, 20) — character indices
            pos_ids: Tensor of shape (batch, 64) — POS tag indices
            neg_flags: Tensor of shape (batch, 64, 3) — negation features
            raw_tokens: list[list[str]] — token strings for ELMo batch_to_ids

        Returns:
            Tensor of shape (batch, 64, 1477)
        """
        device = token_ids.device

        # ELMo real-time inference
        char_ids_elmo = batch_to_ids(raw_tokens).to(device)
        elmo_out = self.elmo(char_ids_elmo)["elmo_representations"][0]
        # elmo_out: [batch, seq_len, 1024]

        # Pad to MAX_LEN=64 if needed
        if elmo_out.shape[1] < 64:
            pad = torch.zeros(
                elmo_out.shape[0], 64 - elmo_out.shape[1], 1024, device=device
            )
            elmo_out = torch.cat([elmo_out, pad], dim=1)

        # Truncate to MAX_LEN=64
        elmo_out = elmo_out[:, :64, :]

        # GloVe embeddings: (batch, 64, 300)
        glove_out = self.glove_layer(token_ids)

        # CharCNN: (batch, 64, 100)
        char_out = self.char_cnn(char_ids)

        # POS embeddings: (batch, 64, 50)
        pos_out = self.pos_embed(pos_ids)

        # Concatenate all: (batch, 64, 1477)
        combined = torch.cat([glove_out, elmo_out, char_out, pos_out, neg_flags], dim=-1)

        # Apply dropout
        output = self.dropout(combined)

        return output
