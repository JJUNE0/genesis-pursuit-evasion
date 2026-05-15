"""Phase 3f — τ-Unaware Flow Matching forecaster (TODO §3f, 2026-05-14).

Encoder-decoder transformer that predicts the **current defender state** from
fixed-length attacker/action history + a single last-observed defender snapshot,
**without ever seeing the actual delay τ**.

Critical design — τ leak prevention (CLAUDE.md §1 + plan §2):
  - No τ embedding in encoder context.
  - No τ-dependent positional offset (every defender_last token is at the same
    fixed position).
  - Sequence length is τ-independent (K + K + 1 = 2K + 1 always).
  - The only τ-dependent quantity in the input is the *value* of
    ``defender_last`` (older snapshot for larger τ); shape stays (B, 1, 6).

Output is a single-step ``s^d_t ∈ R^6`` velocity field (Flow Matching).

NOT cherry-picked from CleanDiffuser. Uses ``torch.nn`` primitives directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from agents.forecaster.model import SinusoidalEmbedding   # reuse
from agents.forecaster.tau_unaware import constants as C


class TauUnawareForecasterModel(nn.Module):
    """τ-unaware flow-matching velocity field for current defender estimate.

    Forward signature::
        v_pred = model(x_t, t, context)
    where
        x_t      : (B, target_dim)         noised target at flow time t
        t        : (B, 1)                  flow time ∈ [0, 1]
        context  : dict
            attacker_history : (B, K, attacker_dim)   K = history_K
            defender_last    : (B, 1, defender_dim)   single snapshot, staleness HIDDEN
            action_history   : (B, K, action_dim)

    Output: ``(B, target_dim)`` velocity prediction (same shape as ``x_t``).

    NOTE: ``forward`` does **not** accept ``tau``. Compare with Phase 3c
    ``ForecasterModel.forward(..., tau)`` which explicitly takes τ.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        attacker_dim: int = C.ATTACKER_DIM,
        defender_dim: int = C.DEFENDER_DIM,
        action_dim: int = C.ACTION_DIM,
        target_dim: int = C.TARGET_DIM,
        history_K: int = C.DEFAULT_HISTORY_K,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.history_K = int(history_K)
        self.target_dim = int(target_dim)
        self.attacker_dim = int(attacker_dim)
        self.defender_dim = int(defender_dim)
        self.action_dim = int(action_dim)

        # Per-modality projections.
        self.attacker_proj = nn.Linear(attacker_dim, d_model)
        self.defender_proj = nn.Linear(defender_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.trajectory_proj = nn.Linear(target_dim, d_model)

        # Positional encodings — sequence-position only, τ-independent.
        # ``pos_attacker`` and ``pos_action`` index 0..K-1 (history step index).
        # ``pos_defender_last`` is a single learnable token (always position 0).
        self.pos_attacker = nn.Parameter(torch.zeros(1, history_K, d_model))
        self.pos_action = nn.Parameter(torch.zeros(1, history_K, d_model))
        self.pos_defender_last = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pos_attacker, std=0.02)
        nn.init.trunc_normal_(self.pos_action, std=0.02)
        nn.init.trunc_normal_(self.pos_defender_last, std=0.02)

        # Sinusoidal embedding for flow time t (∈ [0, 1]). NO τ embedding.
        self.t_embed = SinusoidalEmbedding(d_model)

        # Encoder (self-attention only).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_ratio * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        # Decoder (self + cross attention). Single decoder token (current target).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_ratio * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        # Output projection (init small for stable RF training).
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, target_dim)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)

    def forward(
        self,
        x_t: torch.Tensor,                              # (B, target_dim)
        t: torch.Tensor,                                # (B, 1)
        context: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        attacker_h = context["attacker_history"]        # (B, K, A)
        defender_l = context["defender_last"]           # (B, 1, D)
        action_h = context["action_history"]            # (B, K, C)
        B, K, _ = attacker_h.shape
        if K != self.history_K:
            raise ValueError(f"context K={K} != history_K={self.history_K}")

        # ---- Encoder context (NO τ embedding, fixed sequence length) ----
        A = self.attacker_proj(attacker_h) + self.pos_attacker[:, :K]      # (B, K, d)
        D = self.defender_proj(defender_l) + self.pos_defender_last        # (B, 1, d)
        C_ = self.action_proj(action_h) + self.pos_action[:, :K]           # (B, K, d)
        encoder_input = torch.cat([A, C_, D], dim=1)                       # (B, 2K+1, d)
        memory = self.encoder(encoder_input)                               # (B, 2K+1, d)

        # ---- Decoder (single-token target) ----
        x_emb = self.trajectory_proj(x_t).unsqueeze(1)                     # (B, 1, d)
        t_emb = self.t_embed(t.reshape(-1)).unsqueeze(1)                   # (B, 1, d)
        decoder_input = x_emb + t_emb                                      # (B, 1, d)
        decoded = self.decoder(tgt=decoder_input, memory=memory)           # (B, 1, d)
        decoded = self.output_norm(decoded)
        v_pred = self.output_proj(decoded).squeeze(1)                      # (B, target_dim)
        return v_pred

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Deterministic regression baseline (Comparison row #6) — flow matching ablation
# ---------------------------------------------------------------------------

class TauUnawareDeterministicModel(nn.Module):
    """Deterministic regression baseline — **same encoder + same decoder** as
    ``TauUnawareForecasterModel`` (flow). The ONLY differences are:

      1. Decoder query token is a single **learned constant** (no x_t, no t input)
         vs flow's ``trajectory_proj(x_t) + sinusoidal(t_flow)``.
      2. Loss is direct MSE on ``model(context) → target`` vs CFM velocity loss.

    Architecture parity ensures the ablation row #6 cleanly isolates "flow
    matching contribution" from "decoder architecture contribution"
    (peer review 2026-05-14).

    Forward signature::
        s_d_pred = model(context)           # NO x_t, NO t
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        attacker_dim: int = C.ATTACKER_DIM,
        defender_dim: int = C.DEFENDER_DIM,
        action_dim: int = C.ACTION_DIM,
        target_dim: int = C.TARGET_DIM,
        history_K: int = C.DEFAULT_HISTORY_K,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.history_K = int(history_K)
        self.target_dim = int(target_dim)

        self.attacker_proj = nn.Linear(attacker_dim, d_model)
        self.defender_proj = nn.Linear(defender_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)

        self.pos_attacker = nn.Parameter(torch.zeros(1, history_K, d_model))
        self.pos_action = nn.Parameter(torch.zeros(1, history_K, d_model))
        self.pos_defender_last = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pos_attacker, std=0.02)
        nn.init.trunc_normal_(self.pos_action, std=0.02)
        nn.init.trunc_normal_(self.pos_defender_last, std=0.02)

        # Encoder — identical config to flow model.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_ratio * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        # Decoder — identical config to flow model.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_ratio * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        # Single learned constant query (vs flow's trajectory_proj(x_t) + t_emb).
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query_token, std=0.02)

        # Output projection (matches flow model layout for parity).
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, target_dim)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)

    def forward(self, context: dict[str, torch.Tensor]) -> torch.Tensor:
        attacker_h = context["attacker_history"]                           # (B, K, A)
        defender_l = context["defender_last"]                              # (B, 1, D)
        action_h = context["action_history"]                               # (B, K, C)
        B, K, _ = attacker_h.shape

        A = self.attacker_proj(attacker_h) + self.pos_attacker[:, :K]
        D = self.defender_proj(defender_l) + self.pos_defender_last
        C_ = self.action_proj(action_h) + self.pos_action[:, :K]
        encoder_input = torch.cat([A, C_, D], dim=1)                       # (B, 2K+1, d)
        memory = self.encoder(encoder_input)                               # (B, 2K+1, d)

        # Single learned query → cross-attend to memory (architecture parity with flow).
        query = self.query_token.expand(B, 1, self.d_model)                # (B, 1, d)
        decoded = self.decoder(tgt=query, memory=memory)                   # (B, 1, d)
        decoded = self.output_norm(decoded)
        return self.output_proj(decoded).squeeze(1)                        # (B, target_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
