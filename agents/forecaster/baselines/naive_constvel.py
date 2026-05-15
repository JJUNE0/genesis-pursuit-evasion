"""Naive constant-velocity forecaster baseline (Comparison row #2, paper §5).

Pure math, NO learnable parameters. Given the last-observed defender state
(``rel_pos`` + ``rel_vel`` in attacker body frame at the last-observation
time), extrapolates ``τ`` steps forward assuming constant velocity::

    pos_pred = pos_last + (τ * dt) * vel_last
    vel_pred = vel_last

Critical — frame transform parity (CLAUDE.md §3 / plan §6):
The output is in the **same body frame and scaled** as the trained-forecaster
output (Phase 3c τ-aware / Phase 3f τ-unaware), so it can be plug-in swapped
into the same eval pipeline without re-implementing the obs adapter.

Two callable signatures provided:

  1. ``naive_constvel_predict(defender_last, tau, dt, pos_scale, vel_scale)``
     — ``defender_last`` is the body-frame scaled tensor (B, 1, 6) as the
     forecaster receives it; pos/vel scales used to unscale → extrapolate
     → re-scale.

  2. ``NaiveConstVelForecaster`` — class with the same ``forward(...)``-like
     interface as ``ForecasterModel`` for drop-in plug-in eval. Returns
     ``(B, target_dim=6)`` matching τ-unaware single-step shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def naive_constvel_predict(
    defender_last: torch.Tensor,    # shape: (B, 1, 6) body-frame scaled (rel_pos*pos_scale, rel_vel*vel_scale)
    tau: int,
    dt: float,
    pos_scale: float,
    vel_scale: float,
) -> torch.Tensor:
    """Constant-velocity extrapolation in body frame (no learning).

    Returns ``(B, 6)`` — single-step prediction at time ``t = (last_obs_time + τ)``.

    Implementation::
        # Unscale (body-frame) → physical (m, m/s)
        rel_pos_m   = defender_last[:, 0, 0:3] / pos_scale       # (B, 3)
        rel_vel_mps = defender_last[:, 0, 3:6] / vel_scale       # (B, 3)
        # Extrapolate in body frame (assumes attacker body frame stationary —
        # same approximation made by the trained forecasters).
        rel_pos_new_m = rel_pos_m + (τ * dt) * rel_vel_mps
        # Velocity unchanged (constant-velocity model)
        rel_vel_new_mps = rel_vel_mps
        # Re-scale to forecaster-output convention
        return cat([rel_pos_new_m * pos_scale, rel_vel_new_mps * vel_scale])
    """
    if defender_last.dim() != 3 or defender_last.shape[-1] != 6 or defender_last.shape[-2] != 1:
        raise ValueError(
            f"defender_last must be (B, 1, 6); got {tuple(defender_last.shape)}"
        )
    if int(tau) < 1:
        raise ValueError(f"tau must be ≥ 1; got {tau}")

    rel_pos_scaled = defender_last[:, 0, 0:3]                       # (B, 3) body * pos_scale
    rel_vel_scaled = defender_last[:, 0, 3:6]                       # (B, 3) body * vel_scale
    # Unscale → m / (m/s)
    rel_pos_m = rel_pos_scaled / float(pos_scale)
    rel_vel_mps = rel_vel_scaled / float(vel_scale)
    # Constant-velocity extrapolation
    rel_pos_new_m = rel_pos_m + float(tau) * float(dt) * rel_vel_mps
    rel_vel_new_mps = rel_vel_mps
    # Re-scale to forecaster output convention
    return torch.cat([
        rel_pos_new_m * float(pos_scale),
        rel_vel_new_mps * float(vel_scale),
    ], dim=-1)                                                       # (B, 6)


# ---------------------------------------------------------------------------
# Drop-in callable for eval pipeline (mimics trained-forecaster interface)
# ---------------------------------------------------------------------------

class NaiveConstVelForecaster(nn.Module):
    """Stateless wrapper exposing a ``forward(context, tau)`` API.

    For comparable plug-in into ``eval_5forecasters`` (E2). Has zero parameters.
    Frame and scale conventions match Phase 3c τ-aware + Phase 3f τ-unaware
    forecasters (body-frame, scaled output).
    """

    def __init__(self, dt: float, pos_scale: float, vel_scale: float):
        super().__init__()
        self.dt = float(dt)
        self.pos_scale = float(pos_scale)
        self.vel_scale = float(vel_scale)

    @torch.no_grad()
    def forward(self, context: dict[str, torch.Tensor], tau: int) -> torch.Tensor:
        """``context`` must contain ``defender_last`` (B, 1, 6)."""
        return naive_constvel_predict(
            defender_last=context["defender_last"],
            tau=int(tau), dt=self.dt,
            pos_scale=self.pos_scale, vel_scale=self.vel_scale,
        )

    def num_parameters(self) -> int:
        return 0
