"""PR-Q — TanhMean Gaussian distribution for rsl_rl 5.x PPO.

Squashed Gaussian (PR-O)을 대체. PPO 안정성 우선:
  - mean = tanh(mlp_output) 구조적 [-1, 1] 강제 (distribution collapse 차단)
  - sample = Normal(tanh(μ), σ).sample().clamp(-1+ε, 1-ε)
  - log_prob = Normal.log_prob (atanh 없음 — PPO ratio 폭발 차단)

vs SquashedGaussianDistribution (PR-O):
  - PR-O: action = tanh(N(μ, σ).sample()), log_prob에 atanh + Jacobian
  - PR-Q: action = clamp(N(tanh(μ), σ).sample()), log_prob 표준 Normal
  - PR-Q는 수치적으로 안정 (ams_s2_v[10-13] NaN crash 회피)
  - distribution collapse도 차단 (mean이 tanh로 [-1, 1] 잠김)

CLAUDE.md §1 — 새 파일, 기존 코드 무수정.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.distribution import GaussianDistribution


class TanhMeanGaussianDistribution(GaussianDistribution):
    """`mean = tanh(mlp_output)` + standard Gaussian log_prob.

    rsl_rl GaussianDistribution의 update_distribution 패턴을 따르지 않고
    update 메서드 자체를 override (parent에 update_distribution 없음 — update만).
    """

    EPS: float = 1.0e-4   # sample clamp 영역. atanh 안 쓰므로 작은 값 OK.

    def update(self, mlp_output: torch.Tensor) -> None:
        """tanh(mlp_output) → mean. parent update에 변환된 mean 전달."""
        super().update(torch.tanh(mlp_output))

    def sample(self) -> torch.Tensor:
        """Normal sample 후 clamp((-1+ε, 1-ε)). env clamp 없어도 안전."""
        s = self._distribution.sample()
        return s.clamp(-1.0 + self.EPS, 1.0 - self.EPS)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Deterministic action = tanh(mean). Used by MLPModel.forward()
        when stochastic_output=False (PPO eval, frozen-actor inference)."""
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Export-friendly nn.Module wrapper for ONNX/JIT — applies tanh."""
        return _TanhDeterministicOutput()

    # log_prob, mean, std, entropy, params, kl_divergence, init_mlp_weights
    # 모두 parent (GaussianDistribution) 그대로 사용. atanh 없음.


class _TanhDeterministicOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
