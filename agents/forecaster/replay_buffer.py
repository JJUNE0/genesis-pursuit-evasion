"""Phase E.2 — Forecaster용 Replay Buffer (2026-05-08).

Plan: docs/plans/phase_e_forecaster_buffer.md.

FIFO replay buffer for supervised forecaster learning.
- 매 PPO rollout 후 transition windows를 push.
- 매 forecaster train step에서 random batch sample.
- Cycle 누적 → catastrophic forgetting 방지 (alternating cycles 데이터 모두 보존).

CLAUDE.md §8: forecaster gradient는 PPO loss로 흐르지 않음. buffer는 ndarray 기반.

Stored window structure (per transition entry):
  - s_a_history: (K+1, attacker_state_dim)  — history s^a_{t-K..t}
  - s_d_history: (K+1, defender_state_dim)  — history s^d_{t-K..t}
  - action_history: (K, num_actions)        — history actions_{t-K..t-1}
  - s_d_future: (H, defender_state_dim)     — target ŝ^d_{t+1..t+H}
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForecasterBatch:
    """One sampled batch from ForecasterReplayBuffer."""
    s_a_history: torch.Tensor    # (B, K+1, s_a_dim)
    s_d_history: torch.Tensor    # (B, K+1, s_d_dim)
    action_history: torch.Tensor # (B, K, num_actions)
    s_d_future: torch.Tensor     # (B, H, s_d_dim)


class ForecasterReplayBuffer:
    """FIFO replay buffer (preallocated tensors).

    Stored on ``device`` (typically GPU). Capacity in #transitions.
    Buffer wraps around (FIFO) when full.

    Usage:
      buffer = ForecasterReplayBuffer(capacity, K, H, s_a_dim, s_d_dim, num_actions, device)
      buffer.add(s_a_history, s_d_history, action_history, s_d_future)  # (B, ...)
      batch = buffer.sample(batch_size)
    """

    def __init__(
        self,
        capacity: int,
        history_K: int,
        horizon_H: int,
        s_a_dim: int,
        s_d_dim: int,
        num_actions: int,
        device: torch.device | str = "cuda",
        seed: int = 0,
    ):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if history_K <= 0:
            raise ValueError(f"history_K must be > 0, got {history_K}")
        if horizon_H <= 0:
            raise ValueError(f"horizon_H must be > 0, got {horizon_H}")

        self.capacity = int(capacity)
        self.history_K = int(history_K)
        self.horizon_H = int(horizon_H)
        self.s_a_dim = int(s_a_dim)
        self.s_d_dim = int(s_d_dim)
        self.num_actions = int(num_actions)
        self.device = torch.device(device) if isinstance(device, str) else device

        # Preallocated storage tensors.
        K1 = self.history_K + 1
        H = self.horizon_H
        self._s_a_history = torch.zeros(
            (self.capacity, K1, self.s_a_dim), device=self.device, dtype=torch.float32,
        )
        self._s_d_history = torch.zeros(
            (self.capacity, K1, self.s_d_dim), device=self.device, dtype=torch.float32,
        )
        self._action_history = torch.zeros(
            (self.capacity, self.history_K, self.num_actions),
            device=self.device, dtype=torch.float32,
        )
        self._s_d_future = torch.zeros(
            (self.capacity, H, self.s_d_dim), device=self.device, dtype=torch.float32,
        )

        self.size = 0   # 채워진 개수 (≤ capacity)
        self.ptr = 0    # 다음 push 위치 (FIFO wrap-around)

        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.size

    def is_full(self) -> bool:
        return self.size >= self.capacity

    def add(
        self,
        s_a_history: torch.Tensor,    # (B, K+1, s_a_dim)
        s_d_history: torch.Tensor,    # (B, K+1, s_d_dim)
        action_history: torch.Tensor, # (B, K, num_actions)
        s_d_future: torch.Tensor,     # (B, H, s_d_dim)
    ) -> None:
        """Push a batch of B transitions. FIFO wrap if buffer full."""
        B = s_a_history.shape[0]
        if B == 0:
            return
        # Shape sanity
        assert s_a_history.shape == (B, self.history_K + 1, self.s_a_dim), \
            f"s_a_history shape {s_a_history.shape} mismatch"
        assert s_d_history.shape == (B, self.history_K + 1, self.s_d_dim)
        assert action_history.shape == (B, self.history_K, self.num_actions)
        assert s_d_future.shape == (B, self.horizon_H, self.s_d_dim)

        # Write — FIFO wrap-around. If B + ptr > capacity, split.
        end = self.ptr + B
        if end <= self.capacity:
            self._s_a_history[self.ptr:end] = s_a_history
            self._s_d_history[self.ptr:end] = s_d_history
            self._action_history[self.ptr:end] = action_history
            self._s_d_future[self.ptr:end] = s_d_future
        else:
            # Split: first chunk to end of buffer, second chunk wraps to start.
            first = self.capacity - self.ptr
            self._s_a_history[self.ptr:] = s_a_history[:first]
            self._s_d_history[self.ptr:] = s_d_history[:first]
            self._action_history[self.ptr:] = action_history[:first]
            self._s_d_future[self.ptr:] = s_d_future[:first]
            second = B - first
            self._s_a_history[:second] = s_a_history[first:]
            self._s_d_history[:second] = s_d_history[first:]
            self._action_history[:second] = action_history[first:]
            self._s_d_future[:second] = s_d_future[first:]
        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int) -> ForecasterBatch:
        """Random uniform sample from filled portion."""
        if self.size == 0:
            raise RuntimeError("ForecasterReplayBuffer is empty — push transitions first.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        idx = torch.randint(
            0, self.size, (batch_size,), device=self.device, generator=self._gen,
        )
        return ForecasterBatch(
            s_a_history=self._s_a_history[idx],
            s_d_history=self._s_d_history[idx],
            action_history=self._action_history[idx],
            s_d_future=self._s_d_future[idx],
        )
