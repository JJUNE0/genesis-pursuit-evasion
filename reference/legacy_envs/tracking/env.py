"""
Counter-UAS Tracking Environment — Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5.

The ego drone (Nova) pursues a scripted evader drone whose motion is
governed by a jerk-limited kinematic model with random waypoints.

Phase 2 additions:
  - BatteryModel: voltage-dependent thrust scaling + motor lag
  - Residual history buffer: command-acceleration residuals for latent dynamics

Phase 3 additions:
  - CommDelayBuffer: per-env observation delay + blackout
  - ActionChunkExecutor: K-step action chunk buffer
  - IMU observation (always local, bypasses comm delay)

Phase 4 additions:
  - Lead pursuit reward: intercept point prediction (oracle, not in obs)
  - Energy penalty: acceleration magnitude cost
  - 3-stage curriculum: straight-line → random heading → reactive evasion
  - Per-env target motion params for curriculum support

Phase 5 additions:
  - Onboard camera sensor (RasterizerCameraOptions) attached to ego drone base_link
  - 64×64 grayscale render → 2-frame stack → CNN encoder (via rsl-rl CNNModel)
  - Replaces task_obs(6D) with visual features; proprioceptive obs (19D) always available
  - Asymmetric actor-critic: actor uses images, critic uses privileged state

Action space  (4*K,): K-step body-rate command chunk
    Each step is 4D: [c_thrust, nx, ny, nz] normalised ∈ [-1, 1]
    When action chunking is disabled, action space is (4,).

Task obs (6,): relative state in ego **body** frame (DELAYED by τ_comm)
    rel_pos_body (3,)  target position relative to ego, in body frame (scaled)
    rel_vel_body (3,)  target velocity relative to ego, in body frame (scaled)

Full obs layout (25D, Phase 3 mode):
    delayed_task_obs(6) | rot_mat(9) | ang_vel(3) | imu_accel(3) | last_exec_act(4)

Phase 5 obs layout (19D proprioceptive + visual):
    policy: rot_mat(9) | ang_vel(3) | imu_accel(3) | last_exec_act(4)
    visual_obs: (B, 2, 64, 64) grayscale frame stack (2D, for CNNModel)
    critic_task: delayed_task_obs(6) — privileged, critic only

With dynamics encoder, get_observations() adds TensorDict group
"residual_hist" of shape (B, H*4).
"""

from __future__ import annotations

import numpy as np
import torch
import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

from envs.base_drone_env import BaseDroneEnv, gs_rand_float


class TrackingEnv(BaseDroneEnv):
    def __init__(
        self,
        num_envs: int,
        params,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        show_viewer: bool = False,
    ):
        # Body-rate mixer params must be ready before super().__init__
        # because super().__init__ → _build_scene → scene.build → reset → step hooks
        self.max_body_rate = float(env_cfg.get("max_body_rate", 3.0))
        self.rate_kp = torch.tensor(
            env_cfg.get("rate_kp", [2.0, 2.0, 0.5]),
            device=gs.device,
            dtype=gs.tc_float,
        )
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

    # ------------------------------------------------------------------
    # Body-rate mixer  (mirrors AttitudeRatesEnv)
    # ------------------------------------------------------------------

    def _build_mixer_inverse(self, params) -> torch.Tensor:
        if params.n_propellers != 4:
            raise ValueError(f"TrackingEnv requires 4 propellers, got {params.n_propellers}")
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

    # ------------------------------------------------------------------
    # Scene: add evader drone entity as a side-effect before returning
    # ------------------------------------------------------------------

    def _build_scene(self, show_viewer: bool):
        scene, drone, target, cam = super()._build_scene(show_viewer)
        # Second drone entity — kinematically driven (position overridden each step)
        self.target_drone = scene.add_entity(
            gs.morphs.Drone(
                file=self.params.urdf_path,
                propellers_link_name=self.params.prop_link_names,
                propellers_spin=[int(s) for s in self.params.spin_directions],
            )
        )

        # ── Phase 5: onboard camera sensor ───────────────────────────────
        self.use_visual_obs = self.env_cfg.get("use_visual_obs", False)
        if self.use_visual_obs:
            from genesis.options.sensors import RasterizerCameraOptions

            vis_cfg = self.env_cfg.get("visual_cfg", {})
            res = tuple(vis_cfg.get("res", (64, 64)))
            fov = float(vis_cfg.get("fov", 90.0))
            cam_offset_pos = vis_cfg.get("camera_offset_pos", [0.05, 0.0, 0.02])

            # 4x4 offset transform: position offset from base_link
            offset_T = np.eye(4, dtype=np.float32)
            offset_T[:3, 3] = cam_offset_pos

            self.onboard_cam = scene.add_sensor(
                RasterizerCameraOptions(
                    res=res,
                    pos=(0.0, 0.0, 1.0),     # initial world pos (overridden by attachment)
                    lookat=(1.0, 0.0, 1.0),
                    fov=fov,
                    near=0.05,
                    far=20.0,
                    entity_idx=drone.idx,
                    link_idx_local=0,         # base_link
                    offset_T=offset_T,
                )
            )

        return scene, drone, target, cam

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def _init_task_buffers(self):
        # Per-env kf for domain-rand (same pattern as AttitudeRatesEnv)
        self.kf_buf = torch.full(
            (self.num_envs,), self.kf, device=self.device, dtype=gs.tc_float
        )

        # Evader kinematic state (world frame)
        self.tgt_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.tgt_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.tgt_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        # Waypoint + countdown timer
        self.tgt_waypoint = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.tgt_wp_timer = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        # Relative state cache (updated each step in _update_task_state)
        self.rel_pos_body = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.rel_vel_body = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.dist_to_target = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.last_dist_to_target = torch.zeros_like(self.dist_to_target)

        # Target motion parameters — per-env (B, 1) tensors for curriculum support
        self.tgt_max_speed_base = float(self.command_cfg.get("tgt_max_speed", 2.0))
        self.tgt_max_acc_base   = float(self.command_cfg.get("tgt_max_acc",   2.0))
        self.tgt_max_jerk_base  = float(self.command_cfg.get("tgt_max_jerk",  5.0))
        self.tgt_max_speed = torch.full(
            (self.num_envs, 1), self.tgt_max_speed_base, device=self.device, dtype=gs.tc_float
        )
        self.tgt_max_acc = torch.full(
            (self.num_envs, 1), self.tgt_max_acc_base, device=self.device, dtype=gs.tc_float
        )
        self.tgt_max_jerk = torch.full(
            (self.num_envs, 1), self.tgt_max_jerk_base, device=self.device, dtype=gs.tc_float
        )
        self.wp_min_steps  = int(self.command_cfg.get("wp_min_steps", 100))
        self.wp_max_steps  = int(self.command_cfg.get("wp_max_steps", 300))

        # ── Phase 2a: Battery model ──────────────────────────────────────
        self.use_battery = self.env_cfg.get("use_battery", False)
        if self.use_battery:
            from envs.tracking.battery_model import BatteryModel
            self.battery = BatteryModel(
                num_envs=self.num_envs,
                device=self.device,
                dt=self.dt,
                **self.env_cfg.get("battery_cfg", {}),
            )
        # Motor lag filter state — persistent across steps
        self.motor_forces_actual = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float,
        )

        # ── Phase 2b: Residual history for latent dynamics encoder ───────
        self.use_dynamics_encoder = self.env_cfg.get("use_dynamics_encoder", False)
        if self.use_dynamics_encoder:
            self.residual_H = int(self.env_cfg.get("residual_history_len", 16))
            self.residual_hist = torch.zeros(
                (self.num_envs, self.residual_H, 4),
                device=self.device, dtype=gs.tc_float,
            )
            self.prev_lin_vel = torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=gs.tc_float,
            )
            self.prev_ang_vel = torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=gs.tc_float,
            )

        # ── Phase 3a: Communication delay buffer ─────────────────────────
        self.use_comm_delay = self.env_cfg.get("use_comm_delay", False)
        if self.use_comm_delay:
            from envs.tracking.comm_delay import CommDelayBuffer
            self.comm_delay = CommDelayBuffer(
                num_envs=self.num_envs,
                obs_dim=6,  # rel_pos_body(3) + rel_vel_body(3)
                device=self.device,
                **self.env_cfg.get("comm_delay_cfg", {}),
            )

        # ── Phase 3b: Action chunk executor ──────────────────────────────
        self.use_action_chunking = self.env_cfg.get("use_action_chunking", False)
        if self.use_action_chunking:
            from envs.tracking.action_chunk import ActionChunkExecutor
            self.action_chunk_K = int(self.env_cfg.get("action_chunk_K", 8))
            self.action_chunk = ActionChunkExecutor(
                num_envs=self.num_envs,
                action_dim=4,
                chunk_size=self.action_chunk_K,
                device=self.device,
            )

        # ── Phase 3c: IMU + execution tracking buffers ───────────────────
        self.imu_accel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float,
        )
        self.prev_lin_vel_imu = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float,
        )
        self.last_exec_action = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float,
        )
        self.prev_exec_action = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float,
        )
        self.obs_fresh_flag = torch.ones(
            (self.num_envs,), device=self.device, dtype=torch.bool,
        )
        self.delayed_task_obs = torch.zeros(
            (self.num_envs, 6), device=self.device, dtype=gs.tc_float,
        )

        # ── Phase 4: Lead pursuit curriculum ─────────────────────────────
        self.use_curriculum = self.env_cfg.get("use_curriculum", False)
        self.curriculum_stage = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.int32,
        )
        self.curriculum_success_ema = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float,
        )
        self.curriculum_ema_alpha = float(self.env_cfg.get("curriculum_ema_alpha", 0.05))
        self.curriculum_advance_threshold = float(
            self.env_cfg.get("curriculum_advance_threshold", 0.7)
        )
        self.intercept_threshold = float(self.env_cfg.get("intercept_threshold", 1.0))

        # Stage 1: heading change timer + direction per-env
        self.tgt_heading_timer = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.int32,
        )
        self.tgt_heading_dir = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float,
        )
        self.heading_change_min_steps = int(self.command_cfg.get("heading_change_min_steps", 150))
        self.heading_change_max_steps = int(self.command_cfg.get("heading_change_max_steps", 250))

        # Stage 2: reactive evasion config
        self.evasion_gain = float(self.command_cfg.get("evasion_gain", 1.5))
        self.evasion_blend = float(self.command_cfg.get("evasion_blend", 0.5))

        # Curriculum stage → target speed scaling
        self.curriculum_speed_scales = torch.tensor(
            self.command_cfg.get("curriculum_speed_scales", [1.0, 1.2, 1.5]),
            device=self.device, dtype=gs.tc_float,
        )

        # ── Phase 5: visual observation buffers ─────────────────────────
        if self.use_visual_obs:
            vis_cfg = self.env_cfg.get("visual_cfg", {})
            self.frame_stack_n = int(vis_cfg.get("frame_stack", 2))
            res = tuple(vis_cfg.get("res", (64, 64)))
            self.visual_w, self.visual_h = res
            self.render_every_n = int(vis_cfg.get("render_every_n", 1))
            self.render_counter = 0

            # Frame buffer: (B, frame_stack, H, W) grayscale float32 [0, 1]
            self.frame_buf = torch.zeros(
                (self.num_envs, self.frame_stack_n, self.visual_h, self.visual_w),
                device=self.device, dtype=gs.tc_float,
            )

    def _apply_domain_rand(self, envs_idx, kf_mult: torch.Tensor):
        self.kf_buf[envs_idx] = self.kf * kf_mult

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_task(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)

        # Sample initial evader position (randomised offset from origin)
        x_lo, x_hi = self.command_cfg.get("tgt_init_x_range", [-2.0, 2.0])
        y_lo, y_hi = self.command_cfg.get("tgt_init_y_range", [-2.0, 2.0])
        z_lo, z_hi = self.command_cfg.get("tgt_init_z_range", [1.0,  3.0])

        self.tgt_pos[envs_idx, 0] = gs_rand_float(x_lo, x_hi, (n,), self.device)
        self.tgt_pos[envs_idx, 1] = gs_rand_float(y_lo, y_hi, (n,), self.device)
        self.tgt_pos[envs_idx, 2] = gs_rand_float(z_lo, z_hi, (n,), self.device)
        self.tgt_vel[envs_idx]    = 0.0
        self.tgt_acc[envs_idx]    = 0.0
        self.dist_to_target[envs_idx] = 0.0
        self.last_dist_to_target[envs_idx] = 0.0

        self._sample_target_waypoints(envs_idx)

        # Place evader drone in scene immediately so the first render is correct
        self.target_drone.set_pos(
            self.tgt_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx
        )
        self.target_drone.set_quat(
            self.base_init_quat.unsqueeze(0).expand(n, -1),
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # ── Phase 2a: reset battery state ────────────────────────────────
        if self.use_battery:
            self.battery.reset(envs_idx)
        self.motor_forces_actual[envs_idx] = 0.0

        # ── Phase 2b: reset residual history ─────────────────────────────
        if self.use_dynamics_encoder:
            self.residual_hist[envs_idx] = 0.0
            self.prev_lin_vel[envs_idx] = 0.0
            self.prev_ang_vel[envs_idx] = 0.0

        # ── Phase 3: reset comm delay + action chunk ─────────────────────
        if self.use_comm_delay:
            self.comm_delay.reset(envs_idx)
        if self.use_action_chunking:
            self.action_chunk.reset(envs_idx)
        self.last_exec_action[envs_idx] = 0.0
        self.prev_exec_action[envs_idx] = 0.0
        self.obs_fresh_flag[envs_idx] = True
        self.imu_accel[envs_idx] = 0.0
        self.prev_lin_vel_imu[envs_idx] = 0.0
        self.delayed_task_obs[envs_idx] = 0.0

        # ── Phase 4: curriculum-dependent target params ──────────────────
        if self.use_curriculum:
            stages = self.curriculum_stage[envs_idx]                  # (n,)
            speed_scale = self.curriculum_speed_scales[stages].unsqueeze(1)  # (n, 1)
            self.tgt_max_speed[envs_idx] = self.tgt_max_speed_base * speed_scale
            self.tgt_max_acc[envs_idx]   = self.tgt_max_acc_base * speed_scale
            self.tgt_max_jerk[envs_idx]  = self.tgt_max_jerk_base * speed_scale

            # Reset heading timer for Stage 1 envs
            self.tgt_heading_timer[envs_idx] = torch.randint(
                self.heading_change_min_steps, self.heading_change_max_steps,
                (n,), device=self.device, dtype=torch.int32,
            )
            rand_dir = torch.randn((n, 3), device=self.device)
            rand_dir[:, 2] = 0.0
            rand_dir = rand_dir / rand_dir.norm(dim=1, keepdim=True).clamp_min(1e-6)
            self.tgt_heading_dir[envs_idx] = rand_dir

        # ── Phase 5: reset frame buffer ──────────────────────────────────
        if self.use_visual_obs:
            self.frame_buf[envs_idx] = 0.0

    def _advance_task(self, reset_envs):
        """Phase 4: update curriculum EMA and advance stage on episode end."""
        if not self.use_curriculum or len(reset_envs) == 0:
            return

        # Success: ego was within intercept distance at episode end
        success = (self.dist_to_target[reset_envs] < self.intercept_threshold).float()

        # Exponential moving average of success
        alpha = self.curriculum_ema_alpha
        self.curriculum_success_ema[reset_envs] = (
            (1.0 - alpha) * self.curriculum_success_ema[reset_envs] + alpha * success
        )

        # Stage advancement: advance if EMA > threshold AND not at max stage (2)
        can_advance = (
            (self.curriculum_success_ema[reset_envs] > self.curriculum_advance_threshold)
            & (self.curriculum_stage[reset_envs] < 2)
        )
        if can_advance.any():
            advance_idx = reset_envs[can_advance]
            self.curriculum_stage[advance_idx] += 1
            self.curriculum_success_ema[advance_idx] = 0.0  # reset EMA on promotion

        # Log curriculum stats
        self.extras["curriculum_stage_mean"] = self.curriculum_stage.float().mean().item()
        self.extras["curriculum_ema_mean"] = self.curriculum_success_ema.mean().item()

    def _sample_target_waypoints(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        x_lo, x_hi = self.command_cfg.get("wp_x_range", [-3.0, 3.0])
        y_lo, y_hi = self.command_cfg.get("wp_y_range", [-3.0, 3.0])
        z_lo, z_hi = self.command_cfg.get("wp_z_range", [0.5,  4.0])
        self.tgt_waypoint[envs_idx, 0] = gs_rand_float(x_lo, x_hi, (n,), self.device)
        self.tgt_waypoint[envs_idx, 1] = gs_rand_float(y_lo, y_hi, (n,), self.device)
        self.tgt_waypoint[envs_idx, 2] = gs_rand_float(z_lo, z_hi, (n,), self.device)
        self.tgt_wp_timer[envs_idx] = torch.randint(
            self.wp_min_steps,
            self.wp_max_steps,
            (n,),
            device=self.device,
            dtype=torch.int32,
        )

    # ------------------------------------------------------------------
    # Target kinematics — called BEFORE scene.step() via the hook below
    # ------------------------------------------------------------------

    def _get_task_target_positions(self):
        """Advance evader kinematics and push the scripted position to Genesis."""
        self._advance_target_kinematics()

        # Kinematic override: set evader position in physics (visual only)
        self.target_drone.set_pos(self.tgt_pos, zero_velocity=True)

        # Keep evader rotors spinning at hover RPM to avoid physics NaN
        hover_rpms = self.hover_rpm_buf.unsqueeze(1).expand(-1, 4)  # (B, 4)
        self.target_drone.set_propellers_rpm(hover_rpms)

        return None  # BaseDroneEnv sphere target unused

    def _advance_target_kinematics(self):
        """Jerk-limited pursuit toward current waypoint (fully vectorised).

        All motion params (tgt_max_speed, tgt_max_acc, tgt_max_jerk) are
        (B, 1) tensors to support per-env curriculum stages.
        """
        dt = self.dt

        # ── Waypoint refresh ──────────────────────────────────────────
        self.tgt_wp_timer = self.tgt_wp_timer - 1
        wp_dist  = (self.tgt_waypoint - self.tgt_pos).norm(dim=1)
        need_new = (self.tgt_wp_timer <= 0) | (wp_dist < 0.5)
        if need_new.any():
            self._sample_target_waypoints(need_new.nonzero(as_tuple=False).reshape(-1))

        # ── Desired acceleration toward waypoint ──────────────────────
        to_wp    = self.tgt_waypoint - self.tgt_pos                  # (B, 3)
        dist     = to_wp.norm(dim=1, keepdim=True).clamp_min(1e-6)
        direction = to_wp / dist                                     # (B, 3)

        # Brake when close: reduce commanded acceleration near waypoint
        brake_dist = self.tgt_max_speed ** 2 / (2.0 * self.tgt_max_acc + 1e-6)  # (B, 1)
        speed_frac = (dist / brake_dist).clamp(0.0, 1.0)            # (B, 1)
        acc_des    = direction * self.tgt_max_acc * speed_frac       # (B, 3)

        # ── Phase 4: curriculum-dependent target behavior ─────────────
        if self.use_curriculum:
            stage = self.curriculum_stage                             # (B,)

            # Stage 1 (idx 1): random heading changes every 3-5s
            stage1_mask = (stage == 1)
            if stage1_mask.any():
                self.tgt_heading_timer -= stage1_mask.int()
                need_heading = stage1_mask & (self.tgt_heading_timer <= 0)
                if need_heading.any():
                    idx = need_heading.nonzero(as_tuple=False).reshape(-1)
                    nn = len(idx)
                    rand_dir = torch.randn((nn, 3), device=self.device)
                    rand_dir[:, 2] *= 0.3        # reduce vertical component
                    rand_dir = rand_dir / rand_dir.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    self.tgt_heading_dir[idx] = rand_dir
                    self.tgt_heading_timer[idx] = torch.randint(
                        self.heading_change_min_steps, self.heading_change_max_steps,
                        (nn,), device=self.device, dtype=torch.int32,
                    )
                heading_acc = self.tgt_heading_dir * self.tgt_max_acc   # (B, 3)
                acc_des = torch.where(stage1_mask.unsqueeze(1), heading_acc, acc_des)

            # Stage 2 (idx 2): reactive evasion — turn away from pursuer
            stage2_mask = (stage == 2)
            if stage2_mask.any():
                away_vec = self.tgt_pos - self.base_pos              # (B, 3)
                away_dist = away_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)
                away_dir = away_vec / away_dist
                acc_evasion = away_dir * self.evasion_gain
                blended = (1.0 - self.evasion_blend) * acc_des + self.evasion_blend * acc_evasion
                acc_des = torch.where(stage2_mask.unsqueeze(1), blended, acc_des)

        # ── Jerk limit: clamp rate of change of acceleration ──────────
        jerk_limit = self.tgt_max_jerk * dt                          # (B, 1)
        d_acc = torch.clamp(acc_des - self.tgt_acc, min=-jerk_limit, max=jerk_limit)
        self.tgt_acc = self.tgt_acc + d_acc

        # ── Euler integration ─────────────────────────────────────────
        self.tgt_vel = self.tgt_vel + self.tgt_acc * dt

        # Speed clamp — tensor comparison (B, 1)
        speed = self.tgt_vel.norm(dim=1, keepdim=True).clamp_min(1e-6)
        over  = speed > self.tgt_max_speed                           # (B, 1)
        if over.any():
            scale = torch.where(over, self.tgt_max_speed / speed, torch.ones_like(speed))
            self.tgt_vel = self.tgt_vel * scale

        self.tgt_pos = self.tgt_pos + self.tgt_vel * dt

        # Altitude floor
        self.tgt_pos[:, 2] = self.tgt_pos[:, 2].clamp_min(0.3)

    # ------------------------------------------------------------------
    # Phase 5: visual rendering
    # ------------------------------------------------------------------

    def _render_and_update_frames(self):
        """Render onboard camera and update frame stack buffer."""
        self.render_counter += 1
        if self.render_counter % self.render_every_n != 0:
            return

        # camera.read() auto-updates pose for attached sensors
        data = self.onboard_cam.read()              # CameraData(rgb=Tensor)
        rgb = data.rgb.float()                      # (N, H, W, 3) or (H, W, 3)
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)                  # ensure batch dim
        gray = rgb.mean(dim=-1) / 255.0             # (N, H, W) normalised [0, 1]

        # Shift frame stack: oldest removed, newest appended
        self.frame_buf = torch.roll(self.frame_buf, -1, dims=1)
        self.frame_buf[:, -1, :, :] = gray

    # ------------------------------------------------------------------
    # Task state (called after scene.step + base state update)
    # ------------------------------------------------------------------

    def _update_task_state(self):
        self.last_dist_to_target[:] = self.dist_to_target[:]

        inv_q = inv_quat(self.base_quat)                            # (B, 4)

        # Relative position in world → rotate to ego body frame
        rel_pos_world = self.tgt_pos - self.base_pos                # (B, 3)
        self.rel_pos_body[:] = transform_by_quat(rel_pos_world, inv_q)

        # Relative velocity in body frame:
        #   tgt_vel is world-frame; base_lin_vel is already body-frame
        tgt_vel_body = transform_by_quat(self.tgt_vel, inv_q)       # (B, 3)
        self.rel_vel_body[:] = tgt_vel_body - self.base_lin_vel

        self.dist_to_target[:] = rel_pos_world.norm(dim=1)

        # ── Phase 3: IMU linear acceleration (local sensor, always current) ──
        self.imu_accel[:] = (self.base_lin_vel - self.prev_lin_vel_imu) / self.dt
        self.prev_lin_vel_imu[:] = self.base_lin_vel

        # ── Phase 3: push current task obs to comm delay, retrieve delayed ───
        scale_pos = self.obs_scales["rel_pos"]
        scale_vel = self.obs_scales.get("rel_vel", 1.0 / 3.0)
        current_task_obs = torch.cat([
            (self.rel_pos_body * scale_pos).clamp(-1.0, 1.0),
            (self.rel_vel_body * scale_vel).clamp(-1.0, 1.0),
        ], dim=-1)                                                       # (B, 6)

        if self.use_comm_delay:
            self.comm_delay.push(current_task_obs)
            self.delayed_task_obs[:], self.obs_fresh_flag[:] = self.comm_delay.get()
        else:
            self.delayed_task_obs[:] = current_task_obs
            self.obs_fresh_flag[:] = True

        # ── Phase 5: render onboard camera and update frame stack ────────
        if self.use_visual_obs:
            self._render_and_update_frames()

        # ── Phase 2b: compute and store command-acceleration residual ────
        if self.use_dynamics_encoder:
            # Response proxy (4D): [z_accel / g, angular_accel / max_rate]
            acc_z = (self.base_lin_vel[:, 2] - self.prev_lin_vel[:, 2]) / self.dt
            ang_acc = (self.base_ang_vel - self.prev_ang_vel) / self.dt  # (B, 3)
            response = torch.cat([
                (acc_z / 9.81).unsqueeze(1),
                ang_acc / self.max_body_rate,
            ], dim=1)                                                    # (B, 4)

            # Use last_exec_action (4D) instead of last_actions (may be 4*K)
            residual = self.last_exec_action - response                  # (B, 4)

            # Shift history left and append new residual at the end
            self.residual_hist = torch.roll(self.residual_hist, -1, dims=1)
            self.residual_hist[:, -1, :] = residual

            self.prev_lin_vel[:] = self.base_lin_vel
            self.prev_ang_vel[:] = self.base_ang_vel

    # ------------------------------------------------------------------
    # Crash / termination
    # ------------------------------------------------------------------

    def _compute_termination(self):
        margin = float(self.env_cfg.get("workspace_margin", 6.0))
        z_max  = float(self.env_cfg.get("max_altitude", 8.0))
        return (
            (self.base_pos[:, 0].abs() > margin)
            | (self.base_pos[:, 1].abs() > margin)
            | (self.base_pos[:, 2] > z_max)
        )

    # ------------------------------------------------------------------
    # Observation: 6D task obs
    # ------------------------------------------------------------------

    def _get_task_obs(self) -> torch.Tensor:
        if self.use_comm_delay:
            return self.delayed_task_obs                             # (B, 6) already scaled+clamped
        scale_pos = self.obs_scales["rel_pos"]
        scale_vel = self.obs_scales.get("rel_vel", 1.0 / 3.0)
        rel_p = (self.rel_pos_body * scale_pos).clamp(-1.0, 1.0)
        rel_v = (self.rel_vel_body * scale_vel).clamp(-1.0, 1.0)
        return torch.cat([rel_p, rel_v], dim=-1)                    # (B, 6)

    # ------------------------------------------------------------------
    # Action conversion: body-rate → per-motor RPM
    # ------------------------------------------------------------------

    def _get_exec_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # ── Phase 3: extract single-step action from chunk ───────────────
        if self.use_action_chunking:
            single_action = self.action_chunk.update(actions, self.obs_fresh_flag)
        else:
            single_action = actions                                      # (B, 4)

        self.prev_exec_action[:] = self.last_exec_action
        self.last_exec_action[:] = single_action

        # ── Body-rate mixer (uses single_action, same logic as before) ───
        collective = single_action[:, 0]
        omega_des  = single_action[:, 1:] * self.max_body_rate
        omega_err  = (omega_des - self.base_ang_vel).clamp(
            -self.max_body_rate * 2, self.max_body_rate * 2
        )
        torques = self.rate_kp.unsqueeze(0) * omega_err

        total_thrust = (1.0 + collective * 0.5) * (4.0 * self.hover_thrust_per_motor)
        wrench       = torch.cat([total_thrust.unsqueeze(-1), torques], dim=-1)

        motor_forces = (wrench @ self.mixer_inv.T).clamp(0.0, self.max_thrust_per_motor)

        # ── Phase 2a: battery-dependent thrust scaling + motor lag ────────
        if self.use_battery:
            self.battery.step()
            scale = self.battery.thrust_scale().unsqueeze(1)             # (B, 1)
            motor_forces = motor_forces * scale
            tau   = self.battery.tau_mech().unsqueeze(1)                 # (B, 1)
            alpha = (self.dt / tau).clamp(0.0, 1.0)                     # (B, 1)
            self.motor_forces_actual += alpha * (motor_forces - self.motor_forces_actual)
            motor_forces = self.motor_forces_actual

        return self._forces_to_rpms(motor_forces, kf=self.kf_buf)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _reward_tracking(self) -> torch.Tensor:
        """Exponential proximity reward — always ≥ 0, peaks when ego reaches target."""
        sigma = float(self.env_cfg.get("tracking_sigma", 2.0))
        return torch.exp(-self.dist_to_target / sigma)

    def _reward_closing(self) -> torch.Tensor:
        """Bonus for decreasing distance; penalty for increasing it."""
        return self.last_dist_to_target - self.dist_to_target

    def _reward_intercept(self) -> torch.Tensor:
        """Exponential proximity to predicted intercept point (oracle, NOT in obs).

        t_intercept = dist / closing_speed, clamped to [0.2, 3.0]s.
        p_intercept = p_tgt + v_tgt * t_intercept.
        """
        closing_speed = (self.last_dist_to_target - self.dist_to_target) / self.dt  # (B,)
        closing_speed_safe = closing_speed.clamp(min=0.5)            # floor 0.5 m/s
        t_intercept = (self.dist_to_target / closing_speed_safe).clamp(0.2, 3.0)  # (B,)

        p_intercept = self.tgt_pos + self.tgt_vel * t_intercept.unsqueeze(1)  # (B, 3)
        dist_to_intercept = (self.base_pos - p_intercept).norm(dim=1)         # (B,)

        sigma = float(self.env_cfg.get("intercept_sigma", 2.0))
        return torch.exp(-dist_to_intercept / sigma)

    def _reward_energy(self) -> torch.Tensor:
        """Penalize acceleration magnitude (encourages efficient trajectories)."""
        return self.imu_accel.square().sum(dim=1) / (9.81 ** 2)

    # ------------------------------------------------------------------
    # Reward overrides
    # ------------------------------------------------------------------

    def _reward_smooth(self) -> torch.Tensor:
        """Action smoothness on executed single-step actions (not full chunk)."""
        return torch.sum(
            torch.square(self.last_exec_action - self.prev_exec_action), dim=1
        )

    # _reward_angular, _reward_crash inherited from BaseDroneEnv

    # ------------------------------------------------------------------
    # Observation override (Phase 3 + Phase 2b)
    # ------------------------------------------------------------------

    def _update_observation(self):
        if self.use_visual_obs:
            # Phase 5: proprioceptive-only obs (19D) — no task_obs (replaced by camera)
            rot_mat = self._quat_to_rotmat_flat(self.base_quat)
            imu_accel_scaled = (self.imu_accel / 9.81).clamp(-1.0, 1.0)
            self.obs_buf = torch.cat([
                rot_mat,                                                          # (B, 9)
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1), # (B, 3)
                imu_accel_scaled,                                                  # (B, 3)
                self.last_exec_action,                                             # (B, 4)
            ], dim=-1)  # 19D total
            self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=1.0, neginf=-1.0)
            time_frac = (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(1)
            self.critic_obs_buf = time_frac
            return

        if not self.use_comm_delay:
            super()._update_observation()
            return

        # Phase 3 obs layout (25D):
        # [delayed_task_obs(6) | rot_mat(9) | ang_vel(3) | imu_accel(3) | last_exec_act(4)]
        rot_mat = self._quat_to_rotmat_flat(self.base_quat)
        imu_accel_scaled = (self.imu_accel / 9.81).clamp(-1.0, 1.0)

        self.obs_buf = torch.cat([
            self._get_task_obs(),                                                # (B, 6) delayed
            rot_mat,                                                             # (B, 9) local
            torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),  # (B, 3) local
            imu_accel_scaled,                                                    # (B, 3) local
            self.last_exec_action,                                               # (B, 4) local
        ], dim=-1)

        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=1.0, neginf=-1.0)

        # Critic obs: time fraction (same as base)
        time_frac = (self.episode_length_buf.float() / self.max_episode_length).unsqueeze(1)
        self.critic_obs_buf = time_frac

    def get_observations(self):
        if self.use_visual_obs:
            from tensordict import TensorDict
            obs = TensorDict(
                {
                    "policy": self.obs_buf,               # (B, 19) proprioceptive
                    "visual_obs": self.frame_buf,          # (B, 2, 64, 64) for CNNModel
                    "critic_obs": self.critic_obs_buf,     # (B, 1) time fraction
                    "critic_task": self.delayed_task_obs,  # (B, 6) privileged state
                },
                batch_size=[self.num_envs],
            )
            if self.use_dynamics_encoder:
                obs["residual_hist"] = self.residual_hist.reshape(self.num_envs, -1)
            return obs

        obs = super().get_observations()
        if self.use_dynamics_encoder:
            obs["residual_hist"] = self.residual_hist.reshape(self.num_envs, -1)
        return obs
