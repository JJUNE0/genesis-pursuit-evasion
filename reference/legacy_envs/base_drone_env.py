"""
Base drone RL environment for Genesis AI.

This class owns the Genesis scene, drone state buffers, common step/reset flow,
and shared rewards. Task-specific environments implement only the task hooks.
"""

from __future__ import annotations

import abc
import copy
import math

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)

from utils.drone_params import DroneParams


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class BaseDroneEnv(abc.ABC):
    def __init__(
        self,
        num_envs: int,
        params: DroneParams,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        show_viewer: bool = False,
    ):
        self.num_envs = num_envs
        self.params = params
        self.cfg = env_cfg        # alias expected by rsl_rl OnPolicyRunner
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.device = gs.device
        self.rendered_env_num = min(env_cfg.get("num_rendered_envs", 10), self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = copy.deepcopy(reward_cfg.get("reward_scales", {}))
        self.domain_rand = env_cfg.get("domain_rand", False)
        self.dr_kf_scale = float(env_cfg.get("dr_kf_scale", 0.3))

        hover_thrust_per_motor = params.mass * params.gravity / 4.0
        self.hover_rpm = float(np.sqrt(hover_thrust_per_motor / params.kf))

        # --- Actuator mode ---
        self.actuator_mode = env_cfg.get("actuator_mode", "rpm")
        if self.actuator_mode == "pwm":
            self._init_pwm_lookup()

        self.simulate_action_latency = env_cfg["simulate_action_latency"]
        self.action_latency_steps = int(env_cfg.get("action_latency_steps", 1))
        self.dt = float(env_cfg.get("dt", 0.01))
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device, dtype=gs.tc_float)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device, dtype=gs.tc_float)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.scene, self.drone, self.target, self.cam = self._build_scene(show_viewer)
        self.scene.build(n_envs=num_envs)

        self._init_common_buffers()
        self._init_task_buffers()
        self._init_rewards()
        self.reset()

    def _build_scene(self, show_viewer: bool):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.env_cfg.get("substeps", 4)),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(self.rendered_env_num))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        scene.add_entity(gs.morphs.Plane())

        target = None
        if self.env_cfg.get("visualize_target", False):
            target = scene.add_entity(
                morph=gs.morphs.Sphere(radius=0.05, collision=False),
                surface=gs.surfaces.Rough(color=(1.0, 0.5, 0.5, 0.8)),
            )

        cam = None
        if self.env_cfg.get("visualize_camera", False):
            cam = scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0.0, 0.0, 0.5),
                fov=30,
                GUI=True,
            )

        drone = scene.add_entity(
            gs.morphs.Drone(
                file=self.params.urdf_path,
                propellers_link_name=self.params.prop_link_names,
                propellers_spin=[int(s) for s in self.params.spin_directions],
            )
        )
        return scene, drone, target, cam

    def _init_common_buffers(self):
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.action_buf = torch.zeros(
            (max(self.action_latency_steps, 1), self.num_envs, self.num_actions),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.crash_condition = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.critic_obs_buf = torch.zeros((self.num_envs, 1), device=self.device, dtype=gs.tc_float)
        self.hover_rpm_buf = torch.full((self.num_envs,), self.hover_rpm, device=self.device, dtype=gs.tc_float)
        self.extras = {}

        if self.actuator_mode == "pwm":
            self.hover_pwm_buf = torch.full(
                (self.num_envs,), self.hover_pwm_scalar,
                device=self.device, dtype=gs.tc_float,
            )

    def _init_rewards(self):
        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self._get_exec_actions(self.actions)

        if self.env_cfg.get("raw_rpm_output", False):
            rpms = exec_actions
        elif self.actuator_mode == "pwm":
            pwm = self.hover_pwm_buf.unsqueeze(1) + exec_actions * (
                self.pwm_max - self.hover_pwm_buf.unsqueeze(1)
            )
            pwm = torch.clamp(pwm, self.pwm_min, self.pwm_max)
            thrusts = self._pwm_to_thrust_torch(pwm)
            rpms = torch.sqrt(torch.clamp(thrusts, min=1e-6) / self.params.kf)
        else:
            rpms = (1.0 + exec_actions * 0.5) * self.hover_rpm_buf.unsqueeze(1)
        self.drone.set_propellers_rpm(rpms)

        target_positions = self._get_task_target_positions()
        if self.target is not None and target_positions is not None:
            self.target.set_pos(target_positions, zero_velocity=True)

        self.scene.step()

        self.episode_length_buf += 1
        self._increment_task_counters()
        self.last_base_pos[:] = self.base_pos[:]
        self._update_base_state()
        self._update_task_state()
        self._compute_task_metrics()

        time_out = self.episode_length_buf > self.max_episode_length
        self.crash_condition = self._compute_crash_condition()
        task_reset = self._compute_task_reset()
        self.reset_buf = time_out | self.crash_condition | task_reset

        self.extras["time_outs"] = time_out.to(dtype=gs.tc_float)

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        reset_envs = self.reset_buf.nonzero(as_tuple=False).reshape(-1)
        self._advance_task(reset_envs)
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        self._update_observation()
        self.last_actions[:] = self.actions[:]

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _get_exec_actions(self, actions):
        if not self.simulate_action_latency:
            return actions
        self.action_buf = torch.roll(self.action_buf, shifts=1, dims=0)
        self.action_buf[0] = actions
        return self.action_buf[-1]

    def _update_base_state(self):
        self.base_pos[:] = self.drone.get_pos()
        self.base_quat[:] = self.drone.get_quat()
        self.base_euler[:] = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.drone.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.drone.get_ang(), inv_base_quat)

    def _quat_to_rotmat_flat(self, q: torch.Tensor) -> torch.Tensor:
        """quaternion [w,x,y,z] → rotation matrix flattened to (N, 9)."""
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x * x + z * z),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        )

    def _update_observation(self):
        # obs (22,): task_obs | rot_mat(9) | lin_vel(3) | ang_vel(3) | last_act(4)
        rot_mat = self._quat_to_rotmat_flat(self.base_quat)
        self.obs_buf = torch.cat(
            [
                self._get_task_obs(),
                rot_mat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            dim=-1,
        )
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=1.0, neginf=-1.0)

        # critic-only: time fraction t/T ∈ [0,1] (SimpleFlight Factor 2)
        time_frac = (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(1)
        self.critic_obs_buf = time_frac

    def get_observations(self):
        return TensorDict(
            {"policy": self.obs_buf, "critic_obs": self.critic_obs_buf},
            batch_size=[self.num_envs],
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.base_euler[envs_idx] = 0.0

        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.zero_all_dofs_velocity(envs_idx)

        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.action_buf[:, envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        if self.domain_rand:
            kf_mult = gs_rand_float(
                1.0 - self.dr_kf_scale, 1.0 + self.dr_kf_scale, (len(envs_idx),), self.device
            )
            if self.actuator_mode == "pwm":
                thrust_hover_rand = self.params.hover_thrust_per_motor / kf_mult
                self.hover_pwm_buf[envs_idx] = self._thrust_to_pwm_torch(thrust_hover_rand)
            else:
                self.hover_rpm_buf[envs_idx] = self.hover_rpm / torch.sqrt(kf_mult)
            self._apply_domain_rand(envs_idx, kf_mult)

        self._reset_task(envs_idx)
        self._update_task_state()
        self._compute_task_metrics()

    def reset(self):
        envs_idx = torch.arange(self.num_envs, device=self.device)
        self.reset_buf[:] = True
        self.reset_idx(envs_idx)
        self._update_observation()
        return self.get_observations()

    def _compute_crash_condition(self):
        crash = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        return crash | self._compute_termination()

    # ------------------------------------------------------------------
    # Actuator model: PWM ↔ Thrust (pure torch, GPU batch)
    # ------------------------------------------------------------------

    def _init_pwm_lookup(self):
        """PWM→Thrust lookup 테이블을 torch 텐서로 변환 (GPU 배치 보간용)."""
        table = self.params._cfg["drone"]["pwm_thrust_table"]
        table_t = torch.tensor(table, device=self.device, dtype=gs.tc_float)
        self._pwm_table_pwm = table_t[:, 0].contiguous()       # (N,)
        self._pwm_table_thrust = table_t[:, 1].contiguous()     # (N,)

        self.hover_pwm_scalar = float(self.params.hover_pwm)
        self.pwm_min = float(self.params.pwm_min)
        self.pwm_max = float(self.params.pwm_max)

    def _pwm_to_thrust_torch(self, pwm: torch.Tensor) -> torch.Tensor:
        """PWM [µs] → Thrust [N], pure torch 선형 보간. Shape: (B, 4) → (B, 4)."""
        xp = self._pwm_table_pwm
        fp = self._pwm_table_thrust
        pwm_c = torch.clamp(pwm, xp[0], xp[-1])
        idx = torch.searchsorted(xp, pwm_c.contiguous()) - 1
        idx = torch.clamp(idx, 0, len(xp) - 2)
        x0, x1 = xp[idx], xp[idx + 1]
        y0, y1 = fp[idx], fp[idx + 1]
        t = (pwm_c - x0) / (x1 - x0 + 1e-8)
        return y0 + t * (y1 - y0)

    def _thrust_to_pwm_torch(self, thrust: torch.Tensor) -> torch.Tensor:
        """Thrust [N] → PWM [µs], pure torch 선형 보간 (역산). Shape: any → same."""
        xp = self._pwm_table_thrust
        fp = self._pwm_table_pwm
        thrust_c = torch.clamp(thrust, xp[0], xp[-1])
        idx = torch.searchsorted(xp, thrust_c.contiguous()) - 1
        idx = torch.clamp(idx, 0, len(xp) - 2)
        x0, x1 = xp[idx], xp[idx + 1]
        y0, y1 = fp[idx], fp[idx + 1]
        t = (thrust_c - x0) / (x1 - x0 + 1e-8)
        return y0 + t * (y1 - y0)

    def _forces_to_rpms(self, forces: torch.Tensor, kf=None) -> torch.Tensor:
        """Per-motor forces [N] → RPMs, actuator_mode에 따라 변환 경로 분기.

        Args:
            forces: (B, 4) per-motor desired thrust [N]
            kf: (B,) or scalar. None이면 self.params.kf 사용.
        """
        if kf is None:
            kf = self.params.kf

        if self.actuator_mode == "pwm":
            pwm = self._thrust_to_pwm_torch(forces)
            pwm = torch.clamp(pwm, self.pwm_min, self.pwm_max)
            thrusts = self._pwm_to_thrust_torch(pwm)
            return torch.sqrt(torch.clamp(thrusts, min=1e-6) / self.params.kf)
        else:
            if isinstance(kf, torch.Tensor) and kf.dim() == 1:
                kf = kf.unsqueeze(1)
            return torch.sqrt(torch.clamp(forces / kf, min=0.0))

    def _apply_domain_rand(self, envs_idx, kf_mult: torch.Tensor):
        """Override in subclasses to randomize additional DR parameters per env."""
        pass

    def _increment_task_counters(self):
        pass

    def _compute_task_metrics(self):
        pass

    def _advance_task(self, reset_envs):
        del reset_envs

    def _compute_task_reset(self):
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def _compute_termination(self):
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

    def _get_task_target_positions(self):
        return None

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180.0 * math.pi
        return torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))

    def _reward_angular(self):
        ang_vel_safe = torch.clamp(self.base_ang_vel, -100.0, 100.0)
        return torch.norm(ang_vel_safe / math.pi, dim=1)

    def _reward_crash(self):
        return self.crash_condition.to(dtype=gs.tc_float)

    @abc.abstractmethod
    def _init_task_buffers(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _reset_task(self, envs_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def _update_task_state(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_task_obs(self):
        raise NotImplementedError
