"""HoverEnv — single-drone hover task with body-rate action.

Goal: validate that the BodyRateController (PR-H/I) actually works for RL by
training a hover policy. action ∈ [-1, 1]^4 = [T_norm, wx_ref, wy_ref, wz_ref]
goes through BodyRateController → motor RPMs → Genesis sim.

obs (22):
    rel_pos     (3)   target_pos - drone_pos (world frame, scaled)
    rot_mat     (9)   flattened body→world rotation
    lin_vel     (3)   body-frame linear velocity (scaled)
    ang_vel     (3)   body-frame angular velocity (scaled)
    last_action (4)   previous action (rate-mode)

reward:
    target  : exp(-α · ‖rel_pos‖)            attraction
    hover   : exp(-α₁·dist) · exp(-α₂·v)     near-target + slow bonus
    lin_vel : -‖v‖                           velocity penalty
    smooth  : -‖a_t - a_{t-1}‖²              action smoothness
    yaw     : exp(-λ·|yaw|)                  keep yaw aligned
    angular : -‖ω/π‖                         body-rate penalty
    crash   : -1 on terminate                crash penalty

action_mode (yaml):
    "rate"  → BodyRateController (default; PR-K validation 후 RL 학습 진입)
    "rpm"   → 기존 (1+0.5·a)·hover_rpm fallback (비교용)

CLAUDE.md §10 reproducibility: env owns ``obs_groups``, ``cfg`` (CfgDict).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs

from utils.cfg_dict import CfgDict
from utils.drone_params import DroneParams
from utils.rate_controller import BodyRateController


def _uniform(low: float, high: float, n: int, device) -> torch.Tensor:
    return torch.rand((n,), device=device, dtype=gs.tc_float) * (high - low) + low


def _quat_to_rotmat_flat(q: torch.Tensor) -> torch.Tensor:
    """quat (w,x,y,z) (B, 4) → flattened body→world rotation matrix (B, 9)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    """unit quaternion inverse: (w, -x, -y, -z)."""
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by q (Hamilton convention, q = (w,x,y,z))."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    # v' = q v q*
    t2 = 2 * (qy * vz - qz * vy)
    t3 = 2 * (qz * vx - qx * vz)
    t4 = 2 * (qx * vy - qy * vx)
    rx = vx + qw * t2 + (qy * t4 - qz * t3)
    ry = vy + qw * t3 + (qz * t2 - qx * t4)
    rz = vz + qw * t4 + (qx * t3 - qy * t2)
    return torch.stack([rx, ry, rz], dim=-1)


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rad) from quat (w,x,y,z) using ZYX intrinsic Euler."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


class HoverEnv:
    """Single-drone hover task with rate-mode action.

    rsl-rl 5.x interface: ``cfg``, ``obs_groups``, ``num_actions``, ``num_envs``,
    ``device``, ``reset()``, ``step(actions)``, ``get_observations()``.
    """

    def __init__(
        self,
        num_envs: int,
        params: DroneParams,
        env_cfg: dict[str, Any],
        obs_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        command_cfg: dict[str, Any],
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.params = params
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.cfg: dict[str, Any] = CfgDict(env_cfg)

        self.dt = float(env_cfg["dt"])
        self.substeps = int(env_cfg.get("substeps", 16))
        self.episode_length_s = float(env_cfg["episode_length_s"])
        self.max_episode_length = max(1, math.ceil(self.episode_length_s / self.dt))
        self.num_actions = 4
        self.clip_actions = float(env_cfg.get("clip_actions", 1.0))

        self.action_mode = str(env_cfg.get("action_mode", "rate"))
        if self.action_mode not in ("rate", "rpm"):
            raise ValueError(
                f"action_mode must be 'rate' or 'rpm', got {self.action_mode!r}"
            )

        self.device = gs.device
        self.show_viewer = bool(show_viewer)
        self.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        self.cfg["obs_groups"] = self.obs_groups

        # obs scaling
        scales = obs_cfg.get("obs_scales", {})
        self.scale_pos = float(scales.get("rel_pos", 1.0 / 3.0))
        self.scale_vel = float(scales.get("lin_vel", 1.0 / 3.0))
        self.scale_ang = float(scales.get("ang_vel", 1.0 / math.pi))

        # reward scales (multiplied by dt to match per-step accumulation)
        self.reward_scales = {
            k: float(v) * self.dt for k, v in reward_cfg.get("reward_scales", {}).items()
        }
        self.yaw_lambda = float(reward_cfg.get("yaw_lambda", -10.0))
        self.target_alpha = float(reward_cfg.get("target_alpha", 5.0))
        self.hover_alpha_dist = float(reward_cfg.get("hover_alpha_dist", 2.0))
        self.hover_alpha_vel = float(reward_cfg.get("hover_alpha_vel", 5.0))

        # spawn / target ranges
        self.spawn_pos_range = command_cfg["spawn_pos_range"]   # dict with x,y,z
        self.target_pos_range = command_cfg["target_pos_range"]

        # termination thresholds
        self.term_roll_deg = float(env_cfg["termination_if_roll_greater_than"])
        self.term_pitch_deg = float(env_cfg["termination_if_pitch_greater_than"])
        self.term_z_floor = float(env_cfg["termination_if_close_to_ground"])
        self.term_xy_max = float(env_cfg["termination_if_xy_greater_than"])
        self.term_z_max = float(env_cfg["termination_if_z_greater_than"])

        # build scene
        self._build_scene(show_viewer)
        self.scene.build(n_envs=self.num_envs)

        # hover_rpm precompute (legacy fallback path)
        n_props = float(params.n_propellers)
        self.hover_rpm_scalar = float(np.sqrt(params.mass * params.gravity / (n_props * params.kf)))

        # rate controller
        if self.action_mode == "rate":
            rcfg = env_cfg.get("rate_controller", {}) or {}
            kp = tuple(float(v) for v in rcfg.get("kp_rate", (0.05, 0.10, 0.10)))
            kd = tuple(float(v) for v in rcfg.get("kd_rate", (1.0e-4, 5.0e-5, 1.0e-4)))
            max_rate = float(rcfg.get("max_body_rate", math.pi))
            self.rate_ctrl = BodyRateController(
                num_envs=self.num_envs, dt=self.dt, params=params,
                device=self.device, kp_rate=kp, kd_rate=kd, max_body_rate=max_rate,
            )
        else:
            self.rate_ctrl = None

        # buffers
        self._init_buffers()
        self.reset()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _build_scene(self, show_viewer: bool) -> None:
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(self.env_cfg.get("max_visualize_FPS", 60)),
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(4, self.num_envs)))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())

        urdf_path = self.params.urdf_path
        if not Path(urdf_path).is_absolute():
            urdf_path = str(Path(__file__).resolve().parents[2] / urdf_path)

        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file=urdf_path,
                propellers_link_name=list(self.params.prop_link_names),
                propellers_spin=[int(s) for s in self.params.spin_directions],
            )
        )

        self.target_marker = None
        if show_viewer or self.env_cfg.get("visualize_target", False):
            self.target_marker = self.scene.add_entity(
                morph=gs.morphs.Sphere(radius=0.05, collision=False),
                surface=gs.surfaces.Rough(color=(1.0, 0.5, 0.5, 0.8)),
            )

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _init_buffers(self) -> None:
        B = self.num_envs
        f = gs.tc_float
        i = gs.tc_int
        d = self.device

        self.actions = torch.zeros((B, self.num_actions), device=d, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((B, 3), device=d, dtype=f)
        self.base_quat = torch.zeros((B, 4), device=d, dtype=f)
        self.base_quat[:, 0] = 1.0
        self.base_lin_vel = torch.zeros((B, 3), device=d, dtype=f)         # body frame
        self.base_ang_vel = torch.zeros((B, 3), device=d, dtype=f)         # body frame
        self.base_lin_vel_world = torch.zeros((B, 3), device=d, dtype=f)
        self.base_yaw = torch.zeros((B,), device=d, dtype=f)

        self.target_pos = torch.zeros((B, 3), device=d, dtype=f)
        self.rel_pos = torch.zeros((B, 3), device=d, dtype=f)              # world frame target - drone

        self.episode_length_buf = torch.zeros((B,), device=d, dtype=i)
        self.reset_buf = torch.ones((B,), device=d, dtype=torch.bool)
        self.crash_buf = torch.zeros((B,), device=d, dtype=torch.bool)
        self.rew_buf = torch.zeros((B,), device=d, dtype=f)

        self.episode_sums = {
            k: torch.zeros((B,), device=d, dtype=f) for k in self.reward_scales
        }
        self.extras: dict[str, Any] = {"episode": {}, "time_outs": torch.zeros((B,), device=d, dtype=f)}

        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=d, dtype=f).unsqueeze(0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        self.actions[:] = actions

        rpms = self._action_to_rpm(actions)
        self.drone.set_propellers_rpm(rpms)
        if self.target_marker is not None:
            self.target_marker.set_pos(self.target_pos, zero_velocity=True)
        self.scene.step()

        self.episode_length_buf += 1
        self._update_state()

        # crash + termination
        roll, pitch = self._roll_pitch_deg()
        crash = (
            (roll.abs() > self.term_roll_deg)
            | (pitch.abs() > self.term_pitch_deg)
            | (self.base_pos[:, 2] < self.term_z_floor)
            | (self.base_pos[:, :2].abs().amax(dim=1) > self.term_xy_max)
            | (self.base_pos[:, 2] > self.term_z_max)
        )
        self.crash_buf[:] = crash
        timeout = self.episode_length_buf >= self.max_episode_length

        # reward
        self.rew_buf.zero_()
        for name, scale in self.reward_scales.items():
            r = self._reward(name) * scale
            self.rew_buf += r
            self.episode_sums[name] += r
        self.rew_buf = torch.nan_to_num(self.rew_buf, nan=0.0, posinf=1.0, neginf=-1.0)

        done = crash | timeout
        self.reset_buf[:] = done
        self.extras["time_outs"] = timeout.to(dtype=gs.tc_float)

        self.last_actions[:] = self.actions
        reset_envs = done.nonzero(as_tuple=False).reshape(-1)
        if reset_envs.numel() > 0:
            self._update_episode_extras(reset_envs)
            self.reset_idx(reset_envs)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> TensorDict:
        envs_idx = torch.arange(self.num_envs, device=self.device)
        self.reset_buf[:] = True
        self.reset_idx(envs_idx)
        return self.get_observations()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if envs_idx.numel() == 0:
            return
        n = int(envs_idx.numel())
        d = self.device
        f = gs.tc_float

        # spawn
        sp = self.spawn_pos_range
        spawn = torch.zeros((n, 3), device=d, dtype=f)
        spawn[:, 0] = _uniform(*sp["x"], n=n, device=d)
        spawn[:, 1] = _uniform(*sp["y"], n=n, device=d)
        spawn[:, 2] = _uniform(*sp["z"], n=n, device=d)
        self.base_pos[envs_idx] = spawn

        identity_q = self._identity_quat.expand(n, 4).contiguous()
        self.base_quat[envs_idx] = identity_q
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.base_lin_vel_world[envs_idx] = 0.0
        self.base_yaw[envs_idx] = 0.0

        self.drone.set_pos(spawn, zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(identity_q, zero_velocity=True, envs_idx=envs_idx)

        # target
        tp = self.target_pos_range
        tgt = torch.zeros((n, 3), device=d, dtype=f)
        tgt[:, 0] = _uniform(*tp["x"], n=n, device=d)
        tgt[:, 1] = _uniform(*tp["y"], n=n, device=d)
        tgt[:, 2] = _uniform(*tp["z"], n=n, device=d)
        self.target_pos[envs_idx] = tgt

        self.episode_length_buf[envs_idx] = 0
        self.crash_buf[envs_idx] = False
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0

        if self.rate_ctrl is not None:
            self.rate_ctrl.reset(envs_idx)

        # refresh state
        self._update_state()

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def _update_state(self) -> None:
        self.base_pos[:] = self.drone.get_pos()
        self.base_quat[:] = self.drone.get_quat()
        vel_world = self.drone.get_vel()
        self.base_lin_vel_world[:] = vel_world
        # body-frame: v_body = R^T · v_world = quat_inv ⊗ v ⊗ quat
        q_inv = _quat_inv(self.base_quat)
        self.base_lin_vel[:] = _quat_apply(q_inv, vel_world)
        self.base_ang_vel[:] = self.drone.get_ang()
        self.base_yaw[:] = _quat_to_yaw(self.base_quat)
        self.rel_pos[:] = self.target_pos - self.base_pos

    def _roll_pitch_deg(self) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.base_quat
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = torch.asin(sinp)
        return roll * (180.0 / math.pi), pitch * (180.0 / math.pi)

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _action_to_rpm(self, action: torch.Tensor) -> torch.Tensor:
        if self.rate_ctrl is not None:
            return self.rate_ctrl.step(action, self.base_ang_vel)
        return (1.0 + 0.5 * action) * self.hover_rpm_scalar

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"policy": self._build_obs()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _build_obs(self) -> torch.Tensor:
        rel = torch.clip(self.rel_pos * self.scale_pos, -1.0, 1.0)               # (B, 3)
        rot = _quat_to_rotmat_flat(self.base_quat)                               # (B, 9)
        lin = torch.clip(self.base_lin_vel * self.scale_vel, -1.0, 1.0)          # (B, 3)
        ang = torch.clip(self.base_ang_vel * self.scale_ang, -1.0, 1.0)          # (B, 3)
        last_a = self.last_actions                                               # (B, 4)
        obs = torch.cat([rel, rot, lin, ang, last_a], dim=1)                     # (B, 22)
        return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------
    # Reward terms
    # ------------------------------------------------------------------

    def _reward(self, name: str) -> torch.Tensor:
        fn = getattr(self, f"_reward_{name}")
        return fn()

    def _reward_target(self) -> torch.Tensor:
        dist = self.rel_pos.norm(dim=1).clamp_min(1e-6)
        return torch.exp(-self.target_alpha * dist)

    def _reward_hover(self) -> torch.Tensor:
        dist = self.rel_pos.norm(dim=1).clamp_min(1e-6)
        v = self.base_lin_vel.norm(dim=1)
        return torch.exp(-self.hover_alpha_dist * dist) * torch.exp(-self.hover_alpha_vel * v)

    def _reward_lin_vel(self) -> torch.Tensor:
        return self.base_lin_vel.norm(dim=1)

    def _reward_smooth(self) -> torch.Tensor:
        diff = self.actions - self.last_actions
        return (diff * diff).sum(dim=1)

    def _reward_yaw(self) -> torch.Tensor:
        # exp(λ·|yaw|) with λ = yaw_lambda (negative) → ~1 at yaw=0, → 0 at large |yaw|.
        return torch.exp(self.yaw_lambda * self.base_yaw.abs())

    def _reward_angular(self) -> torch.Tensor:
        ang = torch.clamp(self.base_ang_vel, -100.0, 100.0)
        return (ang / math.pi).norm(dim=1)

    def _reward_crash(self) -> torch.Tensor:
        return self.crash_buf.to(dtype=gs.tc_float)

    # ------------------------------------------------------------------
    # Episode logging
    # ------------------------------------------------------------------

    def _update_episode_extras(self, reset_envs: torch.Tensor) -> None:
        ep = {}
        T = float(self.episode_length_s)
        for k, sums in self.episode_sums.items():
            ep[f"rew_{k}"] = float(sums[reset_envs].mean().item()) / T
            sums[reset_envs] = 0.0
        ep["dist_to_target_mean"] = float(self.rel_pos[reset_envs].norm(dim=1).mean().item())
        ep["crash_rate"] = float(self.crash_buf[reset_envs].to(gs.tc_float).mean().item())
        self.extras["episode"] = ep
