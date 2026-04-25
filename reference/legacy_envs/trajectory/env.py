"""
Waypoint trajectory tracking environment.

The task is represented as a fixed-size waypoint horizon in the observation.
Waypoints are generated per-episode on the GPU unless `fixed_waypoints` is
provided in `command_cfg`.

Agile Flight extensions (opt-in via env_cfg):
  - ATS (Adaptive Temporal Scaling): per-env alpha that scales the waypoint
    reach threshold.  Close tracking → alpha grows (advance sooner, fly faster).
    Poor tracking → alpha shrinks (stay precise, fly slower).
  - Residual Dynamics: online MLP that learns (s_t, a_t) → Δs_{t+1} inside
    step(), without any changes to the PPO runner.
"""

import torch

import genesis as gs

from envs.base_drone_env import BaseDroneEnv, gs_rand_float


class TrajectoryEnv(BaseDroneEnv):
    def __init__(
        self,
        num_envs,
        params,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer: bool = False,
    ):
        self.obs_waypoint_count = int(command_cfg.get("obs_waypoint_count", 3))
        fixed_waypoints = command_cfg.get("fixed_waypoints")
        if fixed_waypoints is not None:
            self.min_waypoints = len(fixed_waypoints)
            self.max_waypoints = len(fixed_waypoints)
        else:
            default_count = int(command_cfg.get("num_waypoints", 4))
            self.min_waypoints = int(command_cfg.get("min_waypoints", default_count))
            self.max_waypoints = int(command_cfg.get("max_waypoints", default_count))
        super().__init__(
            num_envs=num_envs,
            params=params,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def _init_task_buffers(self):
        # --- Waypoint tracking (existing) ---
        self.waypoints = torch.zeros((self.num_envs, self.max_waypoints, 3), device=self.device, dtype=gs.tc_float)
        self.waypoint_counts = torch.full((self.num_envs,), self.max_waypoints, device=self.device, dtype=gs.tc_int)
        self.current_wp_idx = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.current_targets = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.rel_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
        self.waypoint_reached_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.trajectory_completed_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        # --- ATS: Adaptive Temporal Scaling ---
        self.ats_enabled = bool(self.env_cfg.get("ats_enabled", False))
        self.ats_easy_dist = float(self.env_cfg.get("ats_easy_dist", 0.10))
        self.ats_hard_dist = float(self.env_cfg.get("ats_hard_dist", 0.30))
        self.ats_rate = float(self.env_cfg.get("ats_rate", 0.001))
        self.ats_alpha_min = float(self.env_cfg.get("ats_alpha_min", 0.5))
        self.ats_alpha_max = float(self.env_cfg.get("ats_alpha_max", 2.0))
        # alpha=1: normal speed, >1: faster (larger threshold), <1: slower
        self.alpha_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # --- Residual Dynamics: online MLP learning ---
        self.use_residual = bool(self.env_cfg.get("use_residual", False))
        if self.use_residual:
            from models.residual_dynamics import ResidualDynamics
            self.residual_model = ResidualDynamics(
                state_dim=18,
                action_dim=int(self.env_cfg.get("num_actions", 4)),
                hidden_dim=int(self.env_cfg.get("residual_hidden_dim", 64)),
                buffer_size=int(self.env_cfg.get("residual_buffer_size", 100_000)),
                device=self.device,
            )
            self.residual_batch_size = int(self.env_cfg.get("residual_batch_size", 256))
            # s_t stored at end of each metrics call for next step's push
            self._prev_residual_state = torch.zeros(
                (self.num_envs, 18), device=self.device, dtype=gs.tc_float
            )
            # False after reset so we skip push on the very first step
            self._residual_has_prev = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.bool
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_task(self, envs_idx):
        self.current_wp_idx[envs_idx] = 0
        self.waypoint_reached_buf[envs_idx] = False
        self.trajectory_completed_buf[envs_idx] = False
        self.alpha_buf[envs_idx] = 1.0
        if self.use_residual:
            self._residual_has_prev[envs_idx] = False
        self._sample_waypoints(envs_idx)

    # ------------------------------------------------------------------
    # Task state update (unchanged)
    # ------------------------------------------------------------------

    def _update_task_state(self):
        clamped_idx = torch.minimum(self.current_wp_idx, self.waypoint_counts - 1)
        gather_idx = clamped_idx.view(-1, 1, 1).expand(-1, 1, 3)
        self.current_targets[:] = self.waypoints.gather(1, gather_idx).squeeze(1)
        self.rel_pos[:] = self.current_targets - self.base_pos
        self.last_rel_pos[:] = self.current_targets - self.last_base_pos

    # ------------------------------------------------------------------
    # Task metrics — ATS + Residual learning hooks
    # ------------------------------------------------------------------

    def _compute_task_metrics(self):
        dist = torch.norm(self.rel_pos, dim=1)  # (num_envs,)

        # ATS: update per-env alpha and derive effective threshold
        if self.ats_enabled:
            speed_up_mask = dist < self.ats_easy_dist
            slow_down_mask = dist > self.ats_hard_dist
            self.alpha_buf = torch.where(
                speed_up_mask,
                self.alpha_buf * (1.0 + self.ats_rate),
                self.alpha_buf,
            )
            self.alpha_buf = torch.where(
                slow_down_mask,
                self.alpha_buf * (1.0 - self.ats_rate),
                self.alpha_buf,
            )
            self.alpha_buf.clamp_(self.ats_alpha_min, self.ats_alpha_max)
            effective_threshold = self.env_cfg["at_target_threshold"] * self.alpha_buf
        else:
            effective_threshold = self.env_cfg["at_target_threshold"]

        self.waypoint_reached_buf = dist < effective_threshold
        last_wp = self.current_wp_idx >= (self.waypoint_counts - 1)
        self.trajectory_completed_buf = self.waypoint_reached_buf & last_wp

        # Residual dynamics: online one-step learning
        # push (s_t, a_t, s_{t+1}), then one SGD step
        if self.use_residual:
            s_next = self.residual_model.encode_state(
                self.base_pos, self.base_quat, self.base_lin_vel, self.base_ang_vel
            )
            valid_mask = self._residual_has_prev
            if valid_mask.any():
                valid_idx = valid_mask.nonzero(as_tuple=False).reshape(-1)
                self.residual_model.push(
                    self._prev_residual_state[valid_idx],
                    self.actions[valid_idx],
                    s_next[valid_idx],
                )
                self.residual_model.update(batch_size=self.residual_batch_size)
            self._prev_residual_state[:] = s_next
            self._residual_has_prev[:] = True

    # ------------------------------------------------------------------
    # Remaining task hooks (unchanged)
    # ------------------------------------------------------------------

    def _compute_task_reset(self):
        return self.trajectory_completed_buf

    def _advance_task(self, reset_envs):
        advance_mask = self.waypoint_reached_buf & ~self.trajectory_completed_buf
        if len(reset_envs) > 0:
            advance_mask[reset_envs] = False
        envs_idx = advance_mask.nonzero(as_tuple=False).reshape(-1)
        if len(envs_idx) == 0:
            return
        self.current_wp_idx[envs_idx] += 1
        self._update_task_state()
        self._compute_task_metrics()

    def _compute_termination(self):
        margin = float(self.env_cfg.get("workspace_margin", 1.0))
        x_low, x_high = self.command_cfg["pos_x_range"]
        y_low, y_high = self.command_cfg["pos_y_range"]
        z_low, z_high = self.command_cfg["pos_z_range"]
        return (
            (self.base_pos[:, 0] < x_low - margin)
            | (self.base_pos[:, 0] > x_high + margin)
            | (self.base_pos[:, 1] < y_low - margin)
            | (self.base_pos[:, 1] > y_high + margin)
            | (self.base_pos[:, 2] < z_low - margin)
            | (self.base_pos[:, 2] > z_high + margin)
        )

    def _get_task_target_positions(self):
        return self.current_targets

    def _get_task_obs(self):
        horizon = torch.arange(self.obs_waypoint_count, device=self.device, dtype=gs.tc_int).view(1, -1)
        future_idx = self.current_wp_idx.view(-1, 1) + horizon
        max_idx = (self.waypoint_counts - 1).view(-1, 1)
        future_idx = torch.minimum(future_idx, max_idx)
        gather_idx = future_idx.unsqueeze(-1).expand(-1, -1, 3)
        future_targets = self.waypoints.gather(1, gather_idx)
        future_rel = future_targets - self.base_pos.unsqueeze(1)
        return torch.clip(future_rel * self.obs_scales["rel_pos"], -1, 1).reshape(self.num_envs, -1)

    # ------------------------------------------------------------------
    # Waypoint sampling (unchanged)
    # ------------------------------------------------------------------

    def _sample_waypoints(self, envs_idx):
        if len(envs_idx) == 0:
            return

        fixed_waypoints = self.command_cfg.get("fixed_waypoints")
        if fixed_waypoints is not None:
            points = torch.tensor(fixed_waypoints, device=self.device, dtype=gs.tc_float)
            count = points.shape[0]
            self.waypoints[envs_idx, :count] = points.unsqueeze(0)
            if count < self.max_waypoints:
                self.waypoints[envs_idx, count:] = points[-1]
            self.waypoint_counts[envs_idx] = count
            return

        num_resets = len(envs_idx)
        self.waypoint_counts[envs_idx] = torch.randint(
            low=self.min_waypoints,
            high=self.max_waypoints + 1,
            size=(num_resets,),
            device=self.device,
            dtype=gs.tc_int,
        )

        x_low, x_high = self.command_cfg["pos_x_range"]
        y_low, y_high = self.command_cfg["pos_y_range"]
        z_low, z_high = self.command_cfg["pos_z_range"]
        min_dist = float(self.command_cfg.get("min_waypoint_distance", 0.4))

        samples = torch.zeros((num_resets, self.max_waypoints, 3), device=self.device, dtype=gs.tc_float)
        prev = self.base_init_pos.expand(num_resets, 3)

        for wp_idx in range(self.max_waypoints):
            candidate = torch.stack(
                [
                    gs_rand_float(x_low, x_high, (num_resets,), self.device),
                    gs_rand_float(y_low, y_high, (num_resets,), self.device),
                    gs_rand_float(z_low, z_high, (num_resets,), self.device),
                ],
                dim=-1,
            )
            delta = candidate - prev
            dist = torch.norm(delta, dim=1, keepdim=True).clamp_min(1e-6)
            short_mask = dist.squeeze(-1) < min_dist
            if torch.any(short_mask):
                candidate[short_mask] = prev[short_mask] + delta[short_mask] / dist[short_mask] * min_dist
            candidate[:, 0].clamp_(x_low, x_high)
            candidate[:, 1].clamp_(y_low, y_high)
            candidate[:, 2].clamp_(z_low, z_high)
            samples[:, wp_idx] = candidate
            prev = candidate

        wp_ids = torch.arange(self.max_waypoints, device=self.device, dtype=gs.tc_int).view(1, -1)
        last_valid = (self.waypoint_counts[envs_idx] - 1).view(-1, 1)
        tail_mask = wp_ids > last_valid
        last_points = samples.gather(1, last_valid.unsqueeze(-1).expand(-1, 1, 3)).expand(-1, self.max_waypoints, -1)
        samples = torch.where(tail_mask.unsqueeze(-1), last_points, samples)
        self.waypoints[envs_idx] = samples

    # ------------------------------------------------------------------
    # Rewards (unchanged)
    # ------------------------------------------------------------------

    def _reward_target(self):
        return torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)

    def _reward_waypoint(self):
        return self.waypoint_reached_buf.to(dtype=gs.tc_float)

    def _reward_completion(self):
        return self.trajectory_completed_buf.to(dtype=gs.tc_float)
