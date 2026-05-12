"""Phase E.2 — Forecaster Transformer model (2026-05-08).

Plan: docs/plans/phase_e_forecaster_buffer.md.
CLAUDE.md §8: f_ψ standalone module, PPO loss와 분리.

Architecture (사용자 권장 — 작은 모델):
  - Input projection: per-step (s_a + s_d + action) → d_model=64
  - Learnable positional encoding (history length K=10)
  - TransformerEncoder: 2 layers, 64 d_model, 2 heads, 256 ffn
  - Aggregate: last token of encoder output → (B, d_model)
  - Linear head: (B, d_model) → (B, H * s_d_dim) → reshape (B, H, s_d_dim)

Forward:
  Input:  s_a_history (B, K, s_a_dim) + s_d_history (B, K, s_d_dim) + action_history (B, K, num_actions)
  Output: s_d_pred (B, H, s_d_dim)   — H-step defender state prediction

NOTE on history shape: ReplayBuffer stores K+1 length (t-K..t inclusive). Model
uses last K (most recent) for input sequence. CLAUDE.md §8 spec is K-step history
ending at current step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForecasterModel(nn.Module):
    """Small Transformer encoder for defender state prediction.

    Args:
        s_a_dim: attacker state dim (default 18 — pos+vel+rot_mat+ang_vel).
        s_d_dim: defender state dim (default 18).
        num_actions: action dim (default 4).
        history_K: input sequence length.
        horizon_H: output prediction length (multi-step).
        d_model: transformer hidden dim (사용자 권장 64).
        nhead: attention heads (2).
        num_layers: encoder layer count (2).
        ffn_dim: feed-forward dim (4 × d_model).
        dropout: dropout rate (0.1).
    """

    def __init__(
        self,
        s_a_dim: int = 18,
        s_d_dim: int = 18,
        num_actions: int = 4,
        history_K: int = 10,
        horizon_H: int = 5,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * d_model

        self.s_a_dim = int(s_a_dim)
        self.s_d_dim = int(s_d_dim)
        self.num_actions = int(num_actions)
        self.history_K = int(history_K)
        self.horizon_H = int(horizon_H)
        self.d_model = int(d_model)

        # per-step input dim (s_a + s_d + action)
        in_dim = self.s_a_dim + self.s_d_dim + self.num_actions

        # Input projection
        self.input_proj = nn.Linear(in_dim, d_model)

        # Learnable positional encoding (history_K positions)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.history_K, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: last token → (H * s_d_dim)
        self.output_head = nn.Linear(d_model, self.horizon_H * self.s_d_dim)

        # Init last layer with small weights to start near zero predictions.
        nn.init.zeros_(self.output_head.bias)
        nn.init.xavier_uniform_(self.output_head.weight, gain=0.1)

    def forward(
        self,
        s_a_history: torch.Tensor,    # (B, K_or_K1, s_a_dim) — last K used
        s_d_history: torch.Tensor,    # (B, K_or_K1, s_d_dim) — last K used
        action_history: torch.Tensor, # (B, K, num_actions)
    ) -> torch.Tensor:
        """Forward pass.

        Returns: predicted defender state (B, H, s_d_dim).
        """
        B = s_a_history.shape[0]
        K = self.history_K

        # ReplayBuffer는 K+1 길이로 저장 (t-K..t). 모델은 last K (t-K+1..t) 사용.
        # action_history는 K 길이 (action_{t-K..t-1}).
        if s_a_history.shape[1] == K + 1:
            s_a = s_a_history[:, -K:, :]
            s_d = s_d_history[:, -K:, :]
        elif s_a_history.shape[1] == K:
            s_a, s_d = s_a_history, s_d_history
        else:
            raise ValueError(
                f"s_a_history seq_len {s_a_history.shape[1]} not in {{K={K}, K+1={K+1}}}"
            )
        assert s_d.shape == (B, K, self.s_d_dim)
        assert action_history.shape == (B, K, self.num_actions)

        # Concat per step → (B, K, in_dim)
        x = torch.cat([s_a, s_d, action_history], dim=-1)
        # Input projection + pos encoding
        x = self.input_proj(x) + self.pos_emb   # (B, K, d_model)
        # Encoder
        h = self.encoder(x)                     # (B, K, d_model)
        # Aggregate: last token (most recent t)
        last = h[:, -1, :]                      # (B, d_model)
        # Output head
        pred_flat = self.output_head(last)      # (B, H * s_d_dim)
        pred = pred_flat.view(B, self.horizon_H, self.s_d_dim)
        return pred

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
