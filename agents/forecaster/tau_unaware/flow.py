"""Phase 3f — Conditional Flow Matching loss + Euler sampling (τ-unaware, TODO §3f).

Single-step target ``s^d_t ∈ R^6``. Same straight-line probability path as
Phase 3c rectified flow, but ``forward()`` does **not** take ``tau`` (model is
τ-unaware by construction).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cfm_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """One CFM training step. Returns scalar loss tensor (mean over batch)."""
    target = batch["target"]                                            # (B, target_dim)
    B = target.shape[0]
    device = target.device

    noise = torch.randn_like(target)                                    # (B, target_dim) — x_0
    t = torch.rand(B, 1, device=device)                                 # (B, 1) ∼ U(0,1)

    x_t = (1.0 - t) * noise + t * target                                # (B, target_dim)
    v_target = target - noise                                           # (B, target_dim)

    context = {
        "attacker_history": batch["attacker_history"],                  # (B, K, A)
        "defender_last":    batch["defender_last"],                     # (B, 1, D)
        "action_history":   batch["action_history"],                    # (B, K, C)
    }
    v_pred = model(x_t=x_t, t=t, context=context)                       # (B, target_dim)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_via_euler(
    model,
    context: dict[str, torch.Tensor],
    num_steps: int = 10,
    target_dim: int = 6,
) -> torch.Tensor:
    """Euler ODE integration: x_0 ∼ N(0,I) → x_1 ≈ target.

    Returns ``(B, target_dim)``. Default 10 steps (fewer than Phase 3c's 20
    since target is single-step + paper deployment-realism focus).
    """
    if int(num_steps) < 1:
        raise ValueError(f"num_steps must be ≥ 1, got {num_steps}")
    B = context["attacker_history"].shape[0]
    device = context["attacker_history"].device
    N = int(num_steps)

    x = torch.randn(B, int(target_dim), device=device)                  # (B, target_dim)
    dt = 1.0 / float(N)
    for step in range(N):
        t_val = step / float(N)
        t = torch.full((B, 1), t_val, device=device)                    # (B, 1)
        v = model(x_t=x, t=t, context=context)                          # (B, target_dim)
        x = x + v * dt
    return x


# ---------------------------------------------------------------------------
# Deterministic baseline loss (Comparison row #6)
# ---------------------------------------------------------------------------

def deterministic_mse_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Plain MSE regression for the deterministic-head ablation model."""
    target = batch["target"]                                            # (B, target_dim)
    context = {
        "attacker_history": batch["attacker_history"],
        "defender_last":    batch["defender_last"],
        "action_history":   batch["action_history"],
    }
    pred = model(context)                                               # (B, target_dim)
    return F.mse_loss(pred, target)
