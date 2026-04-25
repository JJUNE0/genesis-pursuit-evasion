"""
Hover task with body-rate actions that are mixed to per-motor RPM commands.

action (4,):
    c_thrust  (1,)   collective thrust command around hover thrust
    nx        (1,)   normalized roll rate command in [-1, 1]
    ny        (1,)   normalized pitch rate command in [-1, 1]
    nz        (1,)   normalized yaw rate command in [-1, 1]
"""

import torch

import genesis as gs

from envs.hover.env import HoverEnv


class AttitudeRatesEnv(HoverEnv):
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
        self.max_body_rate = float(env_cfg.get("max_body_rate", 6.0))
        self.rate_kp = torch.tensor(env_cfg.get("rate_kp", [4.0, 4.0, 1.0]), device=gs.device, dtype=gs.tc_float)
        self.max_thrust_per_motor = float(params.max_thrust_per_motor)
        self.kf = float(params.kf)
        self.hover_thrust_per_motor = float(params.hover_thrust_per_motor)
        self.mixer_inv = self._build_mixer_inverse(params)
        super().__init__(
            num_envs=num_envs,
            params=params,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )

    def _build_mixer_inverse(self, params):
        if params.n_propellers != 4:
            raise ValueError(f"AttitudeRatesEnv expects 4 propellers, got {params.n_propellers}")

        allocation = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                params.prop_positions[:, 1].tolist(),
                (-params.prop_positions[:, 0]).tolist(),
                (params.spin_directions * params.torque_thrust_ratio).tolist(),
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        return torch.linalg.inv(allocation)

    def _init_task_buffers(self):
        super()._init_task_buffers()
        self.kf_buf = torch.full((self.num_envs,), self.kf, device=self.device, dtype=gs.tc_float)

    def _apply_domain_rand(self, envs_idx, kf_mult: torch.Tensor):
        self.kf_buf[envs_idx] = self.kf * kf_mult

    def _get_exec_actions(self, actions):
        delayed_actions = super()._get_exec_actions(actions)

        collective = delayed_actions[:, 0]
        omega_des = delayed_actions[:, 1:] * self.max_body_rate
        omega_err = torch.clamp(
            omega_des - self.base_ang_vel,
            -self.max_body_rate * 2,
             self.max_body_rate * 2,
        )
        torques = self.rate_kp.unsqueeze(0) * omega_err

        total_thrust = (1.0 + collective * 0.5) * (4.0 * self.hover_thrust_per_motor)
        wrench = torch.cat([total_thrust.unsqueeze(-1), torques], dim=-1)

        motor_forces = wrench @ self.mixer_inv.transpose(0, 1)
        motor_forces = torch.clamp(motor_forces, min=0.0, max=self.max_thrust_per_motor)
        return self._forces_to_rpms(motor_forces, kf=self.kf_buf)
