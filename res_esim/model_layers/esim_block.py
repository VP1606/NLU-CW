import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Look into NaN values appearing in simulation testing: may not appear in real data, but this must be checked and solved.

class ESIMBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared BI-LSTM for Premise & Hypothesis as per paper.
        # Use //2 for hidden dimension - doubles for both directions.
        self.shared_bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers = 1,
            bidirectional=True,
            batch_first=True
        )

        # FFN: 4*hidden_dim -> hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # LayerNorm per stream
        self.layer_norm_p = nn.LayerNorm(hidden_dim)
        self.layer_norm_h = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

    # ── Attention ─────────────────────────────────────────────────────────────
    def _soft_dot_attention(self, h_p, h_h, mask_p, mask_h):
        """
        Soft cross-sentence dot-product attention (equations 8–10 in paper).

        Each premise token attends over all hypothesis tokens (→ h'_p),
        and each hypothesis token attends over all premise tokens (→ h'_h).

        Padding positions are masked to -inf before softmax so they
        contribute zero weight.

        Args:
            h_p    : (batch, len_p, hidden_dim)
            h_h    : (batch, len_h, hidden_dim)
            mask_p : (batch, len_p)  True = padding position in premise
            mask_h : (batch, len_h)  True = padding position in hypothesis

        Returns:
            h_p_att : (batch, len_p, hidden_dim)  attended premise
            h_h_att : (batch, len_h, hidden_dim)  attended hypothesis
        """

        # Similarity Matrix: E_ij = h_p[b, i] * h_h[b, j]
        e = torch.bmm(h_p, h_h.transpose(1, 2))  # (batch, len_p, len_h)

        # Mask Padding: in hypothesis - affects premise attending to hypothesis.
        if mask_h is not None:
            e = e.masked_fill(mask_h.unsqueeze(1), float('-inf'))  # (batch, len_p, len_h)

        # Mask Padding: in premise - affects hypothesis attending to premise.
        e_t = e.clone()
        if mask_p is not None:
            e_t = e_t.masked_fill(mask_p.unsqueeze(2), float('-inf'))  # (batch, len_p, len_h)

        # Attending: Premise attends to hypothesis - softmax over len_h.
        alpha = F.softmax(e, dim=2) # (batch, len_p, len_h)

        # DEBUG: Check for NaN in alpha
        if torch.isnan(alpha).any():
            print(f"[DEBUG] NaN detected in alpha (premise attending to hypothesis)!")
            print(f"        mask_h shape: {mask_h.shape if mask_h is not None else None}")
            print(f"        mask_h sum per sequence: {mask_h.sum(dim=1) if mask_h is not None else None}")
            print(f"        e shape: {e.shape}, e min: {e.min():.4f}, e max: {e.max():.4f}")
            print(f"        Number of all-inf rows in e: {(e == float('-inf')).all(dim=2).sum()}")

        h_p_att = torch.bmm(alpha, h_h)  # (batch, len_p, hidden_dim)

        # Attending: Hypothesis attends to premise - softmax over len_p.
        beta = F.softmax(e_t, dim=1)  # (batch, len_p, len_h)

        # DEBUG: Check for NaN in beta
        if torch.isnan(beta).any():
            print(f"[DEBUG] NaN detected in beta (hypothesis attending to premise)!")
            print(f"        mask_p shape: {mask_p.shape if mask_p is not None else None}")
            print(f"        mask_p sum per sequence: {mask_p.sum(dim=1) if mask_p is not None else None}")
            print(f"        e_t shape: {e_t.shape}, e_t min: {e_t.min():.4f}, e_t max: {e_t.max():.4f}")
            print(f"        Number of all-inf columns in e_t: {(e_t == float('-inf')).all(dim=1).sum()}")

        h_h_att = torch.bmm(beta.transpose(1, 2), h_p)  # (batch, len_h, hidden_dim)

        return h_p_att, h_h_att

    # ── Enhancement ───────────────────────────────────────────────────────────
    @staticmethod
    def _enhance(h, h_att):
        """
        Compute the enhancement vector (equations 11–12 in paper):
            m = [h, h_att, h − h_att, h * h_att]

        Difference captures what is mismatched; product captures similarity.

        Args:
            h     : (batch, len, hidden_dim)   original BiLSTM output
            h_att : (batch, len, hidden_dim)   attended representation

        Returns:
            m : (batch, len, 4 * hidden_dim)
        """
        return torch.cat([h, h_att, h - h_att, h * h_att], dim=-1)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, h_p, h_h, mask_p=None, mask_h=None):
        """
        Args:
            h_p    : (batch, len_p, hidden_dim)
            h_h    : (batch, len_h, hidden_dim)
            mask_p : (batch, len_p)  True = padding
            mask_h : (batch, len_h)  True = padding

        Returns:
            out_p : (batch, len_p, hidden_dim)  refined premise representations
            out_h : (batch, len_h, hidden_dim)  refined hypothesis representations
        """
        # Save inputs for residual connection
        residual_p = h_p
        residual_h = h_h

        # ── 1. BiLSTM encoding ───────────────────────────────────────────────
        enc_p, _ = self.shared_bilstm(self.dropout(h_p))  # (batch, len_p, hidden_dim)
        enc_h, _ = self.shared_bilstm(self.dropout(h_h))  # (batch, len_h, hidden_dim)

        # ── 2. Cross-sentence attention ──────────────────────────────────────
        att_p, att_h = self._soft_dot_attention(enc_p, enc_h, mask_p, mask_h)

        # ── 3. Enhancement ───────────────────────────────────────────────────
        m_p = self._enhance(enc_p, att_p)          # (batch, len_p, 4*hidden_dim)
        m_h = self._enhance(enc_h, att_h)          # (batch, len_h, 4*hidden_dim)

        # ── 4. FFN ───────────────────────────────────────────────────────────
        n_p = self.ffn(m_p)                        # (batch, len_p, hidden_dim)
        n_h = self.ffn(m_h)                        # (batch, len_h, hidden_dim)

        # ── 5. Residual connection + LayerNorm (equation 19 in paper) ────────
        out_p = self.layer_norm_p(residual_p + n_p)
        out_h = self.layer_norm_h(residual_h + n_h)

        return out_p, out_h
