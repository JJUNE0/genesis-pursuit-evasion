"""PR-O — Squashed Gaussian distribution for rsl_rl 5.x PPO.

Drop-in replacement for ``rsl_rl.modules.distribution.GaussianDistribution`` that
applies SAC-style ``tanh`` squashing to the actor output. Solves the policy
distribution collapse seen across ams_s2_v[1-4] (deterministic eval shows raw
mean ≈ 25 vs boundary [-1, 1], i.e. all 4 action dims always saturate after
``clamp(-1, 1)``; cf. logs/.../diag_def_v10/, diag_rpm_trace stochastic vs
deterministic both show ``saturation_frac = 1.000``).

Mechanism — pre-squash Gaussian carries learnable params (μ_raw from the MLP,
σ from log_std/std_param). All emitted actions are ``tanh(μ_raw)`` (deterministic)
or ``tanh(μ_raw + σ·ε)`` (stochastic). Construction-side properties (``mean``,
``std``, ``entropy``, ``params``, ``kl_divergence``) operate on the **pre-squash**
Gaussian — this is the standard PPO regime where KL on z = μ + σ·ε is
well-defined; post-squash KL is intractable in closed form. The squashing only
affects what reaches the environment + log_prob (with Jacobian correction).

Usage — drop-in via train cfg:

    actor:
      class_name: MLPModel
      hidden_dims: [256, 256, 256]
      activation: tanh
      distribution_cfg:
        class_name: agents.squashed_gaussian.SquashedGaussianDistribution
        init_std: 1.0
        std_type: scalar

CLAUDE.md §1 — additive new class, no modification to existing rsl_rl code.
Old (Phase 1.0/1.5/A/B/C) ckpts using GaussianDistribution remain valid; the
new squashed cfg requires fresh restart (μ_raw scale is different so transfer
is meaningless).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules.distribution import GaussianDistribution


class SquashedGaussianDistribution(GaussianDistribution):
    """Tanh-squashed Gaussian. ``__init__`` signature unchanged (parent's)."""

    EPS: float = 1.0e-2   # numerical safety for atanh and Jacobian.
    # 1e-6 → 1e-3 → 1e-2 누적 강화 (cycle 28 NaN, cycle 11 NaN 모두 atanh 분산 폭발).
    # atanh(1-1e-2) = 2.65, atanh(1-1e-3) = 3.8, atanh(1-1e-6) = 7.25.
    # 1e-2로 30% 더 안전 — log_prob 분산 폭발 차단.

    # ------------------------------------------------------------------
    # action emission (sample / deterministic) — squashed
    # ------------------------------------------------------------------

    def sample(self) -> torch.Tensor:
        """Stochastic sample squashed to (-1, 1)."""
        u = self._distribution.sample()                                 # (B, D) pre-squash
        return torch.tanh(u)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Deterministic action = tanh(mean). Used by MLPModel.forward()
        when stochastic_output=False (PPO eval, frozen-actor inference)."""
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Export-friendly nn.Module wrapper for ONNX/JIT — applies tanh."""
        return _TanhDeterministicOutput()

    # ------------------------------------------------------------------
    # log probability with tanh Jacobian
    # ------------------------------------------------------------------

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute ``log p(a)`` for ``a = tanh(u)``.

        Standard SAC change-of-variables:
            log p(a) = log p(u) - sum_i log(1 - tanh²(u_i))
        We invert ``a = tanh(u)`` via ``atanh`` (clamping to (-1+ε, 1-ε) for
        numerical safety) and use the stable form
            log(1 - tanh²(u)) = 2 (log 2 - u - softplus(-2u))
        which avoids subtractive cancellation for large |u|.
        """
        outputs = outputs.clamp(-1.0 + self.EPS, 1.0 - self.EPS)
        # Fix E 되돌림 (2026-05-02) — u.clamp(-3, 3)이 ams_s2_v13 cycle 4에서
        # def_crash 100% collapse 유발. log_prob gradient 잘려 학습 quality
        # 파괴. EPS=1e-2 + max_grad_norm=0.5 (Fix A+D)만 유지.
        u = torch.atanh(outputs)                                         # (B, D)
        log_p_pre = self._distribution.log_prob(u).sum(dim=-1)           # (B,)
        log_jac = (
            2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        ).sum(dim=-1)                                                    # (B,)
        return log_p_pre - log_jac


class _TanhDeterministicOutput(nn.Module):
    """Trivial export wrapper for the squashed deterministic head."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
