"""Communication delay buffer with blackout support for TrackingEnv Phase 3.

Simulates realistic communication delay between ground station and ego drone.
Each env has a per-episode tau_comm delay (in simulation steps). During blackout
periods, observations freeze at the last pre-blackout value.

All operations vectorized over (B,) batch dimension. No Python for-loops over envs.
"""

from __future__ import annotations

import torch


class CommDelayBuffer:
    """Ring-buffer based communication delay model with blackout support.

    Args:
        num_envs:               Batch size B.
        obs_dim:                Dimension of delayed observation (6 for rel_pos + rel_vel).
        device:                 Torch device string.
        tau_max:                Maximum communication delay in steps (0.4s at 50 Hz).
        blackout_prob:          Probability of scheduling a blackout episode per reset.
        blackout_duration_range: (min_steps, max_steps) for blackout duration.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        device: str,
        tau_max: int = 20,
        blackout_prob: float = 0.1,
        blackout_duration_range: tuple = (50, 150),
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = device
        self.tau_max = tau_max
        self.blackout_prob = blackout_prob
        self.blackout_dur_lo, self.blackout_dur_hi = blackout_duration_range
        self.buf_len = tau_max + 1  # ring buffer length

        # Ring buffer: (B, buf_len, obs_dim)
        self.ring_buf = torch.zeros((num_envs, self.buf_len, obs_dim), device=device)
        # Points to the slot where the NEXT write will go
        self.write_idx = torch.zeros(num_envs, device=device, dtype=torch.long)
        # Per-env communication delay in steps
        self.tau_comm = torch.zeros(num_envs, device=device, dtype=torch.long)

        # Blackout state
        self.in_blackout = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.blackout_counter = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.blackout_scheduled = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.blackout_start_step = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.blackout_duration = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.frozen_obs = torch.zeros((num_envs, obs_dim), device=device)

        # Freshness tracking
        self.last_read_idx = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.step_count = torch.zeros(num_envs, device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Reset specified envs: resample tau_comm, clear ring buffer, schedule blackout.

        Args:
            envs_idx: 1-D integer tensor of environment indices to reset.
        """
        n = len(envs_idx)
        if n == 0:
            return

        self.ring_buf[envs_idx] = 0.0
        self.write_idx[envs_idx] = 0
        self.step_count[envs_idx] = 0
        self.last_read_idx[envs_idx] = -1

        # Sample per-env delay: uniform integer in [0, tau_max]
        self.tau_comm[envs_idx] = torch.randint(
            0, self.tau_max + 1, (n,), device=self.device
        )

        # Clear blackout state
        self.in_blackout[envs_idx] = False
        self.blackout_counter[envs_idx] = 0
        self.frozen_obs[envs_idx] = 0.0

        # Schedule blackout with probability blackout_prob
        schedule_mask = torch.rand(n, device=self.device) < self.blackout_prob
        self.blackout_scheduled[envs_idx] = schedule_mask

        # Sample blackout start step (skip first 100 steps for ring-buffer warmup)
        # and blackout duration
        self.blackout_start_step[envs_idx] = torch.randint(
            100, 800, (n,), device=self.device
        )
        self.blackout_duration[envs_idx] = torch.randint(
            self.blackout_dur_lo, self.blackout_dur_hi + 1, (n,), device=self.device
        )

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push(self, obs: torch.Tensor) -> None:
        """Write current observation into the ring buffer and advance counters.

        Must be called once per simulation step BEFORE ``get()``.

        Args:
            obs: (B, obs_dim) current (ground-truth) observation tensor.
        """
        arange = torch.arange(self.num_envs, device=self.device)

        # Write obs at current write position
        self.ring_buf[arange, self.write_idx] = obs

        # Advance write index (circular)
        self.write_idx = (self.write_idx + 1) % self.buf_len
        self.step_count += 1

        # --- Blackout trigger check ---
        # Trigger when: scheduled, not already in blackout, step threshold reached,
        # and blackout counter is 0 (first and only blackout per episode).
        trigger = (
            self.blackout_scheduled
            & ~self.in_blackout
            & (self.step_count >= self.blackout_start_step)
            & (self.blackout_counter == 0)
        )

        if trigger.any():
            trigger_arange = arange[trigger]
            # Freeze the delayed obs value at the moment of blackout onset
            read_idx = (
                self.write_idx[trigger] - 1 - self.tau_comm[trigger]
            ) % self.buf_len
            self.frozen_obs[trigger] = self.ring_buf[trigger_arange, read_idx]
            self.in_blackout[trigger] = True
            self.blackout_counter[trigger] = self.blackout_duration[trigger]

        # Decrement counter for active blackouts
        active = self.in_blackout & (self.blackout_counter > 0)
        self.blackout_counter[active] -= 1

        # End blackout when counter reaches 0
        ended = self.in_blackout & (self.blackout_counter == 0)
        self.in_blackout[ended] = False

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve delayed observation and freshness flag.

        Must be called after ``push()`` within the same step.

        Returns:
            delayed_obs: (B, obs_dim) — delayed or frozen observation.
            obs_is_new:  (B,) bool   — True if a fresh (new) observation is available.
        """
        arange = torch.arange(self.num_envs, device=self.device)

        # write_idx points to the NEXT write slot, so the most-recently written
        # slot is write_idx - 1.  Subtract tau_comm to get the delayed slot.
        read_idx = (self.write_idx - 1 - self.tau_comm) % self.buf_len

        normal_obs = self.ring_buf[arange, read_idx]

        # Freshness: read_idx changed since last get() AND not in blackout
        obs_is_new = (read_idx != self.last_read_idx) & ~self.in_blackout
        self.last_read_idx = read_idx.clone()

        # Blackout override: return frozen obs and mark stale
        delayed_obs = torch.where(
            self.in_blackout.unsqueeze(1),
            self.frozen_obs,
            normal_obs,
        )
        obs_is_new = obs_is_new & ~self.in_blackout

        return delayed_obs, obs_is_new
