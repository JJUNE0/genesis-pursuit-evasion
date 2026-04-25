"""
Hover RL environment built on top of BaseDroneEnv.

obs (17,):
    rel_pos    (3,)   target-relative position
    base_quat  (4,)   attitude quaternion
    lin_vel    (3,)   linear velocity in body frame
    ang_vel    (3,)   angular velocity in body frame
    last_act   (4,)   previous action
"""

import math

import torch

import genesis as gs

from envs.base_drone_env import BaseDroneEnv, gs_rand_float


class HoverEnv(BaseDroneEnv):
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
        self.min_command_steps = math.ceil(env_cfg.get("resampling_time_s", 3.0) / float(env_cfg.get("dt", 0.01)))
        super().__init__(
            num_envs=num_envs,
            params=params,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )

    def _init_task_buffers(self):
        self.num_commands = self.command_cfg["num_commands"]
        self.command_time_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.rel_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
        self.target_reached_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def _reset_task(self, envs_idx):
        self.command_time_buf[envs_idx] = 0
        self._resample_commands(envs_idx)

    def _increment_task_counters(self):
        self.command_time_buf += 1

    def _update_task_state(self):
        self.rel_pos[:] = self.commands - self.base_pos
        self.last_rel_pos[:] = self.commands - self.last_base_pos

    def _compute_task_metrics(self):
        self.target_reached_buf = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            & (self.command_time_buf >= self.min_command_steps)
        )

    def _advance_task(self, reset_envs):
        resample_mask = self.target_reached_buf.clone()
        if len(reset_envs) > 0:
            resample_mask[reset_envs] = False
        envs_idx = resample_mask.nonzero(as_tuple=False).reshape(-1)
        self._resample_commands(envs_idx)
        if len(envs_idx) > 0:
            self._update_task_state()
            self._compute_task_metrics()

    def _compute_termination(self):
        return (
            (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
        )

    def _get_task_target_positions(self):
        return self.commands

    def _get_task_obs(self):
        return torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)
        self.command_time_buf[envs_idx] = 0

    def _reward_target(self):
        # Gaussian distance reward: sharper curve → stronger pull toward target
        dist = torch.norm(self.rel_pos, dim=1)
        return torch.exp(-5.0 * dist)

    def _reward_lin_vel(self):
        return torch.norm(self.base_lin_vel, dim=1)

    def _reward_hover(self):
        # Bonus for being near target AND moving slowly (both conditions required)
        dist = torch.norm(self.rel_pos, dim=1)
        vel = torch.norm(self.base_lin_vel, dim=1)
        return torch.exp(-2.0 * dist) * torch.exp(-5.0 * vel)
