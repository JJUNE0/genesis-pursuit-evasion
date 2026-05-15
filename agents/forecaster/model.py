"""Phase 3c — Chi_Transformer encoder-decoder forecaster (TODO §3c, 2026-05-13).

Architecture (spec §3.4 v10):
- Encoder: multi-modal context (attacker history + actions + delayed defender +
  τ embedding) → self-attention transformer (4 layers, pre-norm, GELU).
- Decoder: noised future trajectory + flow-time t embedding → self+cross
  attention (4 layers) → 6D velocity field per future step.
- Heads = 8, d_model = 256, FFN = 4 × d_model, dropout 0.1.
- Conditioning: cross-attention from decoder to encoder context (no causal mask).
- τ (delay) and t (flow time) carried as sinusoidal embeddings.

CLAUDE.md §8: standalone module; no PPO gradient leakage.

NOT cherry-picked from CleanDiffuser ChiTransformer — the reference targets
single-modality action prediction with causal masks, our task is multi-modal
trajectory forecasting. We use ``torch.nn`` primitives directly to keep the
repo standalone (no external diffusion library dependency).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal scalar embedding (τ, flow time t)
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Map a scalar (B,) → (B, dim) via fixed sinusoid basis.

    Standard ``cos/sin`` pair; ``dim`` must be even. Frequencies pre-computed
    as a buffer to avoid per-call device transfers.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalEmbedding dim must be even, got {dim}")
        self.dim = int(dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        )                                                              # (half,)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) or (B, 1) scalar values (any dtype)
        x = x.float().reshape(-1, 1)                                   # (B, 1)
        args = x * self.freqs.to(x.device).unsqueeze(0)                # (B, half)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # (B, dim)


# ---------------------------------------------------------------------------
# Forecaster (encoder-decoder)
# ---------------------------------------------------------------------------

class ForecasterModel(nn.Module):
    """Rectified-flow velocity field for defender future trajectory.

    Forward signature::
        v_pred = model(x_t, t, context, tau)
    where
        x_t      : (B, τ, target_dim)  noised target at flow time t
        t        : (B, 1, 1)            flow time ∈ [0, 1]
        context  : dict
            attacker_history : (B, τ+1, attacker_dim)
            defender_delayed : (B, 1,   defender_dim)
            action_history   : (B, τ,   action_dim)
        tau      : int                  τ (same for all batch items — Pattern 2)

    Output: ``(B, τ, target_dim)`` velocity prediction.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        attacker_dim: int = 18,
        defender_dim: int = 6,
        action_dim: int = 4,
        target_dim: int = 6,
        tau_max: int = 50,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.tau_max = int(tau_max)
        self.target_dim = int(target_dim)

        # Per-modality projections.
        self.attacker_proj = nn.Linear(attacker_dim, d_model)
        self.defender_proj = nn.Linear(defender_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.trajectory_proj = nn.Linear(target_dim, d_model)

        # Learnable positional encodings (sliced per-batch by current τ).
        self.pos_attacker = nn.Parameter(torch.zeros(1, tau_max + 1, d_model))
        self.pos_action = nn.Parameter(torch.zeros(1, tau_max, d_model))
        self.pos_target = nn.Parameter(torch.zeros(1, tau_max, d_model))
        nn.init.trunc_normal_(self.pos_attacker, std=0.02)
        nn.init.trunc_normal_(self.pos_action, std=0.02)
        nn.init.trunc_normal_(self.pos_target, std=0.02)

        # Sinusoidal scalar embeddings for τ (delay) and t (flow time).
        self.tau_embed = SinusoidalEmbedding(d_model)
        self.t_embed = SinusoidalEmbedding(d_model)

        # Encoder (self-attention only, pre-norm, GELU).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_ratio * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        # Decoder (self + cross attention, pre-norm, GELU).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_ratio * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)

        # Output projection (init small so initial v_pred ≈ 0 → stable RF init).
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, target_dim)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,                               # (B, τ, target_dim)
        t: torch.Tensor,                                 # (B, 1, 1)
        context: dict[str, torch.Tensor],
        tau: int,
    ) -> torch.Tensor:
        if int(tau) < 1 or int(tau) > self.tau_max:
            raise ValueError(f"tau out of [1, {self.tau_max}]: {tau}")
        T = int(tau)
        B = x_t.shape[0]

        # ---- Encoder context ----
        A = self.attacker_proj(context["attacker_history"]) + self.pos_attacker[:, : T + 1]  # (B, τ+1, d)
        C = self.action_proj(context["action_history"]) + self.pos_action[:, :T]              # (B, τ, d)
        D = self.defender_proj(context["defender_delayed"])                                   # (B, 1, d)

        tau_scalar = torch.full((B,), float(T), device=x_t.device, dtype=torch.float32)
        tau_emb = self.tau_embed(tau_scalar).unsqueeze(1)                                     # (B, 1, d)

        encoder_input = torch.cat([A, C, D, tau_emb], dim=1)                                   # (B, 2τ+3, d)
        memory = self.encoder(encoder_input)                                                   # (B, 2τ+3, d)

        # ---- Decoder ----
        x_emb = self.trajectory_proj(x_t) + self.pos_target[:, :T]                             # (B, τ, d)
        # Flow time t is a per-sample scalar; broadcast its embedding across τ positions.
        t_emb = self.t_embed(t.reshape(-1)).unsqueeze(1)                                       # (B, 1, d)
        decoder_input = x_emb + t_emb                                                          # (B, τ, d) (broadcast on dim=1)

        decoded = self.decoder(tgt=decoder_input, memory=memory)                               # (B, τ, d)
        decoded = self.output_norm(decoded)
        v_pred = self.output_proj(decoded)                                                     # (B, τ, target_dim)
        return v_pred

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
