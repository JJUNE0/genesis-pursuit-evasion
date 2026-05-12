"""Phase E.2 — ForecasterTrainer (2026-05-08).

Plan: docs/plans/phase_e_forecaster_buffer.md.
CLAUDE.md §8: forecaster gradient는 PPO loss로 흐르지 않음. 별도 optimizer.

Responsibilities:
  1. collect_from_rollout(s_a, s_d, action): rollout tensor에서 sliding windows
     추출 후 ReplayBuffer push.
  2. train_step(batch_size): buffer sample → model forward → MSE loss → backward.
  3. eval_mse: 검증용 (학습 모드 영향 없음).
  4. predict(...): inference (env가 actor obs 구성 시 호출, Phase E.3).

Loss: per-step squared error mean over (B, H, s_d_dim).
Gradient clipping: max_norm=1.0 (transformer 안정).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from agents.forecaster.model import ForecasterModel
from agents.forecaster.replay_buffer import ForecasterReplayBuffer


class ForecasterTrainer:
    """Forecaster 학습 orchestrator.

    Args:
        model:          ForecasterModel instance
        buffer:         ForecasterReplayBuffer instance
        lr:             Adam learning rate (default 3e-4)
        weight_decay:   Adam weight decay (default 1e-4)
        grad_clip:      gradient norm clip (default 1.0)
        device:         "cuda" / "cpu". Model은 이미 device 옮겨졌다고 가정.
    """

    def __init__(
        self,
        model: ForecasterModel,
        buffer: ForecasterReplayBuffer,
        lr: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        grad_clip: float = 1.0,
        device: torch.device | str = "cuda",
    ):
        self.model = model
        self.buffer = buffer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.grad_clip = float(grad_clip)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=float(lr), weight_decay=float(weight_decay),
        )
        self.last_loss = float("nan")
        self.train_step_count = 0

    # ------------------------------------------------------------------
    # Buffer push — sliding windows from rollout
    # ------------------------------------------------------------------

    def collect_from_rollout(
        self,
        s_a: torch.Tensor,    # (T, B, s_a_dim)
        s_d: torch.Tensor,    # (T, B, s_d_dim)
        action: torch.Tensor, # (T, B, num_actions)
    ) -> int:
        """Extract sliding windows + push to buffer.

        Window structure (per timestep t in [K, T-H-1]):
          - s_a_history: s_a[t-K:t+1]   (K+1 length)
          - s_d_history: s_d[t-K:t+1]
          - action_history: action[t-K:t]  (K length)
          - s_d_future: s_d[t+1:t+H+1]  (H length)

        Returns: number of windows pushed (n_windows × B).
        """
        if s_a.dim() != 3 or s_d.dim() != 3 or action.dim() != 3:
            raise ValueError(
                f"Expected 3D tensors (T, B, dim). Got s_a={s_a.shape}, "
                f"s_d={s_d.shape}, action={action.shape}"
            )
        T, B, _ = s_a.shape
        if s_d.shape[:2] != (T, B) or action.shape[:2] != (T, B):
            raise ValueError(
                f"Inconsistent (T, B): s_a={s_a.shape}, s_d={s_d.shape}, action={action.shape}"
            )

        K = self.model.history_K
        H = self.model.horizon_H
        if T < K + H + 1:
            # Rollout too short to extract any window. No-op.
            return 0

        # Move to buffer device.
        s_a = s_a.to(self.device)
        s_d = s_d.to(self.device)
        action = action.to(self.device)

        # Vectorized window extraction:
        # For t in [K, T-H-1] (inclusive of T-H-1) → n_windows = T - K - H
        n_windows = T - K - H

        # Build via list (memory-efficient enough for typical T~1000).
        # Each iteration: O(B × (K+1) × dim) memory, n_windows total.
        all_s_a_h = []
        all_s_d_h = []
        all_a_h = []
        all_s_d_f = []
        for t in range(K, T - H):
            # s_a[t-K:t+1] — shape (K+1, B, dim) → permute to (B, K+1, dim)
            all_s_a_h.append(s_a[t - K:t + 1].transpose(0, 1).contiguous())
            all_s_d_h.append(s_d[t - K:t + 1].transpose(0, 1).contiguous())
            all_a_h.append(action[t - K:t].transpose(0, 1).contiguous())
            all_s_d_f.append(s_d[t + 1:t + H + 1].transpose(0, 1).contiguous())

        s_a_h_cat = torch.cat(all_s_a_h, dim=0)    # (n_windows × B, K+1, s_a_dim)
        s_d_h_cat = torch.cat(all_s_d_h, dim=0)
        a_h_cat = torch.cat(all_a_h, dim=0)        # (n_windows × B, K, num_actions)
        s_d_f_cat = torch.cat(all_s_d_f, dim=0)    # (n_windows × B, H, s_d_dim)

        self.buffer.add(s_a_h_cat, s_d_h_cat, a_h_cat, s_d_f_cat)
        return n_windows * B

    # ------------------------------------------------------------------
    # Train / eval
    # ------------------------------------------------------------------

    def train_step(self, batch_size: int = 256) -> float:
        """One supervised gradient step. Returns loss.item() (NaN if buffer underfilled)."""
        if len(self.buffer) < batch_size:
            return float("nan")
        batch = self.buffer.sample(batch_size)
        self.model.train()
        pred = self.model(
            batch.s_a_history, batch.s_d_history, batch.action_history,
        )
        loss = F.mse_loss(pred, batch.s_d_future)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip,
            )
        self.optimizer.step()
        self.last_loss = float(loss.item())
        self.train_step_count += 1
        return self.last_loss

    @torch.no_grad()
    def eval_mse(self, batch_size: int = 256) -> float:
        """Eval-mode forward + MSE on a fresh sample. No gradient."""
        if len(self.buffer) < batch_size:
            return float("nan")
        batch = self.buffer.sample(batch_size)
        self.model.eval()
        pred = self.model(
            batch.s_a_history, batch.s_d_history, batch.action_history,
        )
        loss = F.mse_loss(pred, batch.s_d_future)
        return float(loss.item())

    # ------------------------------------------------------------------
    # Inference (Phase E.3 will use this for actor obs)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        s_a_history: torch.Tensor,
        s_d_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Forward in eval mode. Returns (B, H, s_d_dim).

        Used by env._build_attacker_obs() in Phase E.3 to build forecast obs.
        Inputs must match buffer's K (or K+1) and num_actions dims.
        """
        self.model.eval()
        return self.model(s_a_history, s_d_history, action_history)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def state_summary(self) -> dict:
        """Compact dict for logging."""
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.capacity,
            "train_steps": self.train_step_count,
            "last_loss": self.last_loss if not math.isnan(self.last_loss) else None,
        }
