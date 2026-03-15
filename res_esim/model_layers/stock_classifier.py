"""
classifier.py
-------------
Aggregation and classification module for res-ESIM.

Implements Sections II-B-5 and II-B-6 of:
  Li et al. (2019) "Residual Connected Enhanced Sequential Inference Model
  for Natural Language Inference." IEEE.

Sits downstream of ResESIM, which outputs refined token-level representations.
This module aggregates those representations into sentence-level vectors and
produces class logits.

Pipeline (per the paper):
  h_p, h_h  (from ResESIM)
      │
  Aggregation BiLSTM           ← eq. 20–21: vp = BiLSTM(h_p), vh = BiLSTM(h_h)
      │
  Masked Max Pooling            ← eq. 22:   vp_max = max(vp),  vh_max = max(vh)
      │
  Concatenation                 ← eq. 23:   v = [vp_max; vh_max; vp_max−vh_max; vp_max*vh_max]
      │
  FFN: ReLU(vW4 + b4)W5 + b5   ← eq. 24
      │
  Softmax → class probabilities
"""

import torch
import torch.nn as nn

class StockClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim  : int,
        n_classes   : int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- Aggregation BILSTM (Independent for Premise & Hypothesis) ---------
        self._bilstm_premise = nn.LSTM(
            input_size = hidden_dim,
            hidden_size = hidden_dim // 2,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        self._bilstm_hyp = nn.LSTM(
            input_size = hidden_dim,
            hidden_size = hidden_dim // 2,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        # --- Classification FFN ------------------------------------------------
        self.dropout = nn.Dropout(dropout_rate)

        self.W4_linear = nn.Linear(4 * hidden_dim, hidden_dim)
        self.W5_linear = nn.Linear(hidden_dim, n_classes)

    # --- Utilities ---------------------------------------------------------------
    @staticmethod
    def _masked_max_pool(tensor, mask):
        """
        Max Pool over sequence dimension. Implements eq. 22 in the paper.
        """
        tensor = tensor.masked_fill(mask.unsqueeze(-1), float('-inf'))
        return tensor.max(dim=1).values

    # --- Forward Pass -----------------------------------------------------------
    def forward(self, h_p, h_h, mask_p, mask_h):
        # Returns: LOGITS: (batch, n_classes)

        # --- Aggregation BILSTM ------------------------------------------------
        v_p, _ = self._bilstm_premise(h_p)  # (batch, seq_len, hidden_dim)
        v_h, _ = self._bilstm_hyp(h_h)      # (batch, seq_len, hidden_dim)

        # --- Masked Max Pooling ------------------------------------------------
        v_p_max = self._masked_max_pool(v_p, mask_p)  # (batch, hidden_dim)
        v_h_max = self._masked_max_pool(v_h, mask_h)  # (batch, hidden_dim)

        # --- Concatenation (equation 23)----------------------------------------
        # v = [vp,max; vh,max; vp,max − vh,max; vp,max ∗ vh,max]
        v = torch.cat([
            v_p_max,
            v_h_max,
            v_p_max - v_h_max,
            v_p_max * v_h_max
        ], dim=-1)  # (batch, 4*hidden_dim)

        # --- FFN (equation 24) ------------------------------------------------
        # ypred = softmax(ReLU (vW4 + b4)W5) + b5)
        # == softmax(W5[ReLU(W4[v])])
        # Note: The paper's equation 24 omits the final softmax, but in PyTorch we typically return raw logits and apply softmax in the loss function (e.g., CrossEntropyLoss).
        y_pred = self.W5_linear(
            torch.relu(self.W4_linear(self.dropout(v)))
        )

        return y_pred
