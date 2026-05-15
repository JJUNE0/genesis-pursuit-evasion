"""Phase 3c — Rectified Flow loss + Euler sampling utilities (TODO §3c, 2026-05-13).

Straight-line probability path: ``x_t = (1-t) x_0 + t x_1`` with ``x_0 ∼ N(0,I)``
and ``x_1`` = target. Velocity loss::

    L = E[ ||v_ψ(x_t, t, c) - (x_1 - x_0)||² ]

Inference: Euler integration over [0, 1] in ``num_steps`` (default 20) starting
from ``x_0 ∼ N(0,I)`` → integrated ``x ≈ x_1``.

CLAUDE.md §8: forecaster gradient stays inside this module / its trainer; no
PPO loss path touches model parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Training loss (single forward)
# ---------------------------------------------------------------------------

def rf_loss(model, batch: dict[str, torch.Tensor], tau: int) -> torch.Tensor:
    """One RF training step. Returns scalar loss tensor (mean over batch)."""
    target = batch["target"]                                            # (B, τ, target_dim)
    B = target.shape[0]
    device = target.device

    noise = torch.randn_like(target)                                    # (B, τ, target_dim) — x_0
    t = torch.rand(B, 1, 1, device=device)                              # (B, 1, 1) ∼ U(0,1)

    x_t = (1.0 - t) * noise + t * target                                # (B, τ, target_dim)
    v_target = target - noise                                           # (B, τ, target_dim)

    context = {
        "attacker_history": batch["attacker_history"],                  # (B, τ+1, A)
        "defender_delayed": batch["defender_delayed"],                  # (B, 1,   D)
        "action_history":   batch["action_history"],                    # (B, τ,   C)
    }
    v_pred = model(x_t=x_t, t=t, context=context, tau=tau)              # (B, τ, target_dim)
    return F.mse_loss(v_pred, v_target)


# ---------------------------------------------------------------------------
# Inference (Euler 20-step by default)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_via_euler(
    model,
    context: dict[str, torch.Tensor],
    tau: int,
    num_steps: int = 20,
    target_dim: int = 6,
) -> torch.Tensor:
    """Euler ODE integration: x_0 ∼ N(0,I) → x_1 ≈ target via velocity field.

    Returns ``(B, τ, target_dim)``.
    """
    if int(num_steps) < 1:
        raise ValueError(f"num_steps must be ≥ 1, got {num_steps}")
    B = context["attacker_history"].shape[0]
    device = context["attacker_history"].device
    T = int(tau)
    N = int(num_steps)

    x = torch.randn(B, T, int(target_dim), device=device)               # (B, τ, target_dim)
    dt = 1.0 / float(N)

    for step in range(N):
        t_val = step / float(N)
        t = torch.full((B, 1, 1), t_val, device=device)                 # (B, 1, 1)
        v = model(x_t=x, t=t, context=context, tau=T)                   # (B, τ, target_dim)
        x = x + v * dt                                                  # Euler update

    return x
