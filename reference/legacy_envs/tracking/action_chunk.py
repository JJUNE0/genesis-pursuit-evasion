"""K-step action chunk executor for TrackingEnv Phase 3.

The RL policy outputs 4*K dimensional actions (K steps of 4D body-rate commands).
This executor maintains a per-env buffer and feeds one action per simulation step.

New chunks are accepted only when signaled (e.g., when a fresh observation arrives
after communication delay resolves). During blackout, the current chunk continues
draining. When exhausted, the last action is held.

All operations vectorized over (B,) batch dimension.
"""

import torch


class ActionChunkExecutor:
    def __init__(self, num_envs: int, action_dim: int, chunk_size: int, device: str):
        """
        Args:
            num_envs: batch size B
            action_dim: single-step action dimension (4 for body-rate commands)
            chunk_size: K — number of steps per chunk
            device: torch device
        """
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.K = chunk_size
        self.device = device

        # Buffers
        self.chunk_buf = torch.zeros((num_envs, chunk_size, action_dim), device=device)
        self.chunk_ptr = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.last_action = torch.zeros((num_envs, action_dim), device=device)

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Zero out chunk buffer and pointer for specified envs."""
        if len(envs_idx) == 0:
            return
        self.chunk_buf[envs_idx] = 0.0
        self.chunk_ptr[envs_idx] = 0
        self.last_action[envs_idx] = 0.0

    def update(self, raw_actions: torch.Tensor, accept_new: torch.Tensor) -> torch.Tensor:
        """Get single-step action from chunk buffer.

        Args:
            raw_actions: (B, action_dim * K) from policy
            accept_new: (B,) bool — True where new chunk should be loaded

        Returns:
            action: (B, action_dim) single-step action to execute
        """
        # 1. Where accept_new is True, load new chunk and reset pointer
        if accept_new.any():
            new_chunk = raw_actions[accept_new].reshape(-1, self.K, self.action_dim)
            self.chunk_buf[accept_new] = new_chunk
            self.chunk_ptr[accept_new] = 0

        # 2. Read current action from chunk buffer
        # Clamp pointer to valid range [0, K-1] for indexing
        safe_ptr = self.chunk_ptr.clamp(max=self.K - 1)
        arange = torch.arange(self.num_envs, device=self.device)
        current_action = self.chunk_buf[arange, safe_ptr]  # (B, action_dim)

        # 3. If pointer < K, use current_action; otherwise hold last_action
        within_chunk = (self.chunk_ptr < self.K).unsqueeze(1)  # (B, 1)
        action = torch.where(within_chunk, current_action, self.last_action)

        # 4. Advance pointer (won't go past K since we clamp on read)
        self.chunk_ptr = (self.chunk_ptr + 1).clamp(max=self.K)

        # 5. Update last_action for hold-last fallback
        self.last_action[:] = action

        return action
