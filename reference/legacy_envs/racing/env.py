"""
Gate racing environment.

The drone must fly through a sequence of gates in order.
Gate passage is detected by checking when the drone crosses the gate plane
(sign change of the dot product with the gate normal).

obs (task part, 6D):
    rel_pos_next  (3,)  position relative to next gate center (scaled)
    rel_pos_after (3,)  position relative to gate-after-next center (scaled)

Total obs (22D via BaseDroneEnv):
    task_obs (6) + rot_mat (9) + lin_vel (3) + ang_vel (3) + last_act (4) - 1
    = actually: task_obs replaces the rel_pos slot (3→6), so 22D → 25D
    BaseDroneEnv concatenates _get_task_obs() + rot_mat + ... dynamically,
    so obs size = len(_get_task_obs()) + 9 + 3 + 3 + 4

Gate layout: gates are generated per-episode as a sequence of 3D positions.
Each gate has:
  - center (3,): world-frame center position
  - normal (3,): unit vector pointing in the expected approach direction

Gate visualization (when show_viewer=True and visualize_target=True):
  4 Box bars per gate form a hollow rectangular frame:
    top bar, bottom bar, left bar, right bar
  Each bar's position is set per-env via batched set_pos()/set_quat().
  All rendered envs show their own gate layouts simultaneously.
"""

import torch

import genesis as gs

from envs.base_drone_env import BaseDroneEnv, gs_rand_float


def _mat3_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.

    R: (B, 3, 3) — columns are the target frame axes
    Returns: (B, 4) quaternion [w, x, y, z]
    """
    B = R.shape[0]
    r00 = R[:, 0, 0]; r01 = R[:, 0, 1]; r02 = R[:, 0, 2]
    r10 = R[:, 1, 0]; r11 = R[:, 1, 1]; r12 = R[:, 1, 2]
    r20 = R[:, 2, 0]; r21 = R[:, 2, 1]; r22 = R[:, 2, 2]
    trace = r00 + r11 + r22

    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    m1 = trace > 0
    if m1.any():
        s = torch.sqrt(torch.clamp(trace[m1] + 1.0, min=1e-10)) * 2
        q[m1, 0] = 0.25 * s
        q[m1, 1] = (r21[m1] - r12[m1]) / s
        q[m1, 2] = (r02[m1] - r20[m1]) / s
        q[m1, 3] = (r10[m1] - r01[m1]) / s

    m2 = (~m1) & (r00 > r11) & (r00 > r22)
    if m2.any():
        s = torch.sqrt(torch.clamp(1.0 + r00[m2] - r11[m2] - r22[m2], min=1e-10)) * 2
        q[m2, 0] = (r21[m2] - r12[m2]) / s
        q[m2, 1] = 0.25 * s
        q[m2, 2] = (r01[m2] + r10[m2]) / s
        q[m2, 3] = (r02[m2] + r20[m2]) / s

    m3 = (~m1) & (~m2) & (r11 > r22)
    if m3.any():
        s = torch.sqrt(torch.clamp(1.0 + r11[m3] - r00[m3] - r22[m3], min=1e-10)) * 2
        q[m3, 0] = (r02[m3] - r20[m3]) / s
        q[m3, 1] = (r01[m3] + r10[m3]) / s
        q[m3, 2] = 0.25 * s
        q[m3, 3] = (r12[m3] + r21[m3]) / s

    m4 = (~m1) & (~m2) & (~m3)
    if m4.any():
        s = torch.sqrt(torch.clamp(1.0 + r22[m4] - r00[m4] - r11[m4], min=1e-10)) * 2
        q[m4, 0] = (r10[m4] - r01[m4]) / s
        q[m4, 1] = (r02[m4] + r20[m4]) / s
        q[m4, 2] = (r12[m4] + r21[m4]) / s
        q[m4, 3] = 0.25 * s

    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-10)
    return q


class RacingEnv(BaseDroneEnv):
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
        fixed_gates = command_cfg.get("fixed_gates")
        if fixed_gates is not None:
            self.num_gates = len(fixed_gates)
        else:
            self.num_gates = int(command_cfg.get("num_gates", 4))

        # Stored early so _build_scene() can use them before _init_task_buffers()
        self._gate_half_w_cfg = float(env_cfg.get("gate_half_width", 0.6))
        self._gate_half_h_cfg = float(env_cfg.get("gate_half_height", 0.6))

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
    # Scene construction: add gate frame bars
    # ------------------------------------------------------------------

    _GATE_URDF = "assets/urdf/gate_frame.urdf"

    def _build_scene(self, show_viewer: bool):
        scene, drone, target, cam = super()._build_scene(show_viewer)

        self._gate_entities = []
        if show_viewer and self.env_cfg.get("visualize_target", False):
            for _ in range(self.num_gates):
                self._gate_entities.append(scene.add_entity(
                    morph=gs.morphs.URDF(file=self._GATE_URDF, collision=False),
                ))

        return scene, drone, target, cam

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def _init_task_buffers(self):
        # Gate positions and normals: (num_envs, num_gates, 3)
        self.gate_centers = torch.zeros(
            (self.num_envs, self.num_gates, 3), device=self.device, dtype=gs.tc_float
        )
        self.gate_normals = torch.zeros(
            (self.num_envs, self.num_gates, 3), device=self.device, dtype=gs.tc_float
        )
        # Half-sizes for passage detection
        self.gate_half_w = float(self.env_cfg.get("gate_half_width", 0.6))
        self.gate_half_h = float(self.env_cfg.get("gate_half_height", 0.6))
        self.gate_pass_margin = float(self.env_cfg.get("gate_pass_margin", 0.3))

        # Current gate index per env (which gate to pass next)
        self.current_gate_idx = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        # Sign of (pos - gate_center) · gate_normal at previous step (for crossing detection)
        self._prev_side = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        # Gate-pass event buffer (set True in step where gate is passed)
        self.gate_passed_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        # Whether the full course is completed
        self.course_completed_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_task(self, envs_idx):
        self.current_gate_idx[envs_idx] = 0
        self.gate_passed_buf[envs_idx] = False
        self.course_completed_buf[envs_idx] = False
        self._sample_gates(envs_idx)
        # Initialise prev_side for current gate
        self._refresh_prev_side(envs_idx)

    def _refresh_prev_side(self, envs_idx):
        """Compute initial side sign for the current target gate.

        Uses base_init_pos instead of base_pos to avoid stale values
        immediately after reset (base_pos is updated after _reset_task).
        """
        if len(envs_idx) == 0:
            return
        idx = self.current_gate_idx[envs_idx]
        centers = self.gate_centers[envs_idx].gather(
            1, idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        normals = self.gate_normals[envs_idx].gather(
            1, idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        # Use base_init_pos (known, fixed) instead of stale base_pos
        init_pos = self.base_init_pos.unsqueeze(0).expand(len(envs_idx), -1)
        diff = init_pos - centers
        self._prev_side[envs_idx] = torch.sum(diff * normals, dim=1)

    # ------------------------------------------------------------------
    # Task state update + gate visualization
    # ------------------------------------------------------------------

    def _update_task_state(self):
        self._update_gate_visuals()

    def _update_gate_visuals(self):
        """Update gate URDF entity positions/orientations for all envs (batched)."""
        if not self._gate_entities:
            return

        N = self.num_envs
        G = self.num_gates
        normals = self.gate_normals  # (N, G, 3)

        # Gate local frame: X=normal, Y=gate_right, Z=gate_up
        world_up = torch.tensor([0., 0., 1.], device=self.device, dtype=gs.tc_float)
        world_x  = torch.tensor([1., 0., 0.], device=self.device, dtype=gs.tc_float)
        world_up_exp = world_up.view(1, 1, 3).expand(N, G, 3)
        world_x_exp  = world_x.view(1, 1, 3).expand(N, G, 3)

        right = torch.cross(normals, world_up_exp, dim=2)
        right_norm = right.norm(dim=2, keepdim=True).clamp_min(1e-6)
        degenerate = (right_norm.squeeze(-1) < 0.1).unsqueeze(-1)
        fallback = torch.cross(normals, world_x_exp, dim=2)
        right = torch.where(degenerate, fallback / fallback.norm(dim=2, keepdim=True).clamp_min(1e-6),
                            right / right_norm)

        gate_up = torch.cross(right, normals, dim=2)
        gate_up = gate_up / gate_up.norm(dim=2, keepdim=True).clamp_min(1e-6)

        R = torch.stack([normals, right, gate_up], dim=3)          # (N, G, 3, 3)
        quats = _mat3_to_quat(R.reshape(N * G, 3, 3)).reshape(N, G, 4)

        for g, entity in enumerate(self._gate_entities):
            entity.set_pos(self.gate_centers[:, g], zero_velocity=True)
            entity.set_quat(quats[:, g], zero_velocity=True)

    # ------------------------------------------------------------------
    # Task metrics — gate crossing detection
    # ------------------------------------------------------------------

    def _compute_task_metrics(self):
        self.gate_passed_buf[:] = False

        clamped_idx = torch.clamp(self.current_gate_idx, 0, self.num_gates - 1)
        gi = clamped_idx.view(-1, 1, 1).expand(-1, 1, 3)
        centers = self.gate_centers.gather(1, gi).squeeze(1)   # (N, 3)
        normals = self.gate_normals.gather(1, gi).squeeze(1)   # (N, 3)

        diff = self.base_pos - centers                          # (N, 3)
        cur_side = torch.sum(diff * normals, dim=1)             # (N,)

        # Plane crossing: sign changed from negative to positive
        crossed = (self._prev_side < 0) & (cur_side >= 0)

        # Must be within gate bounds when crossing.
        # Project diff onto gate plane (perpendicular to normal) to get the
        # in-plane displacement vector, then decompose into a horizontal
        # component (world Z excluded) and a vertical component (world Z).
        along_normal = cur_side.unsqueeze(1) * normals
        in_plane = diff - along_normal                         # (N, 3)

        # Horizontal span: magnitude of in-plane XY projection
        in_plane_horiz = torch.norm(in_plane[:, :2], dim=1)   # (N,)
        # Vertical span: in-plane Z offset
        in_plane_vert = torch.abs(in_plane[:, 2])              # (N,)

        in_bounds = (
            (in_plane_horiz < self.gate_half_w)
            & (in_plane_vert < self.gate_half_h)
        )

        passed = crossed & in_bounds
        last_gate = clamped_idx >= (self.num_gates - 1)

        self.gate_passed_buf[:] = passed
        self.course_completed_buf[:] = passed & last_gate
        self._prev_side[:] = cur_side

    # ------------------------------------------------------------------
    # Advance: move to next gate
    # ------------------------------------------------------------------

    def _compute_task_reset(self):
        return self.course_completed_buf

    def _advance_task(self, reset_envs):
        advance_mask = self.gate_passed_buf & ~self.course_completed_buf
        if len(reset_envs) > 0:
            advance_mask[reset_envs] = False
        envs_idx = advance_mask.nonzero(as_tuple=False).reshape(-1)
        if len(envs_idx) == 0:
            return
        self.current_gate_idx[envs_idx] += 1
        self._refresh_prev_side(envs_idx)

    # ------------------------------------------------------------------
    # Crash
    # ------------------------------------------------------------------

    def _compute_termination(self):
        margin = float(self.env_cfg.get("workspace_margin", 2.0))
        x_lo, x_hi = self.command_cfg["gate_x_range"]
        y_lo, y_hi = self.command_cfg["gate_y_range"]
        z_lo = 0.0
        z_hi = self.command_cfg["gate_z_range"][1]
        return (
            (self.base_pos[:, 0] < x_lo - margin)
            | (self.base_pos[:, 0] > x_hi + margin)
            | (self.base_pos[:, 1] < y_lo - margin)
            | (self.base_pos[:, 1] > y_hi + margin)
            | (self.base_pos[:, 2] < z_lo)
            | (self.base_pos[:, 2] > z_hi + margin)
        )

    # ------------------------------------------------------------------
    # Observation: next gate + gate-after-next relative positions
    # ------------------------------------------------------------------

    def _get_task_target_positions(self):
        """Return current target gate center for visualisation."""
        clamped = torch.clamp(self.current_gate_idx, 0, self.num_gates - 1)
        gi = clamped.view(-1, 1, 1).expand(-1, 1, 3)
        return self.gate_centers.gather(1, gi).squeeze(1)

    def _get_task_obs(self):
        # Next gate relative position
        idx0 = torch.clamp(self.current_gate_idx, 0, self.num_gates - 1)
        idx1 = torch.clamp(self.current_gate_idx + 1, 0, self.num_gates - 1)

        gi0 = idx0.view(-1, 1, 1).expand(-1, 1, 3)
        gi1 = idx1.view(-1, 1, 1).expand(-1, 1, 3)

        c0 = self.gate_centers.gather(1, gi0).squeeze(1)  # (N, 3)
        c1 = self.gate_centers.gather(1, gi1).squeeze(1)  # (N, 3)

        rel0 = torch.clip((c0 - self.base_pos) * self.obs_scales["rel_pos"], -1, 1)
        rel1 = torch.clip((c1 - self.base_pos) * self.obs_scales["rel_pos"], -1, 1)

        return torch.cat([rel0, rel1], dim=-1)  # (N, 6)

    # ------------------------------------------------------------------
    # Gate sampling
    # ------------------------------------------------------------------

    def _sample_gates(self, envs_idx):
        if len(envs_idx) == 0:
            return

        fixed_gates = self.command_cfg.get("fixed_gates")
        if fixed_gates is not None:
            centers = torch.tensor(
                [g["center"] for g in fixed_gates],
                device=self.device, dtype=gs.tc_float,
            )  # (num_gates, 3)
            normals_raw = torch.tensor(
                [g.get("normal", [1.0, 0.0, 0.0]) for g in fixed_gates],
                device=self.device, dtype=gs.tc_float,
            )
            normals = normals_raw / normals_raw.norm(dim=1, keepdim=True).clamp_min(1e-6)
            self.gate_centers[envs_idx] = centers.unsqueeze(0)
            self.gate_normals[envs_idx] = normals.unsqueeze(0)
            return

        n = len(envs_idx)
        x_lo, x_hi = self.command_cfg["gate_x_range"]
        y_lo, y_hi = self.command_cfg["gate_y_range"]
        z_lo, z_hi = self.command_cfg["gate_z_range"]
        min_dist = float(self.command_cfg.get("min_gate_distance", 1.5))

        centers = torch.zeros((n, self.num_gates, 3), device=self.device, dtype=gs.tc_float)
        prev = self.base_init_pos.expand(n, 3).clone()

        for g in range(self.num_gates):
            candidate = torch.stack([
                gs_rand_float(x_lo, x_hi, (n,), self.device),
                gs_rand_float(y_lo, y_hi, (n,), self.device),
                gs_rand_float(z_lo, z_hi, (n,), self.device),
            ], dim=-1)
            # Enforce minimum distance from previous
            delta = candidate - prev
            dist = delta.norm(dim=1, keepdim=True).clamp_min(1e-6)
            short = dist.squeeze(-1) < min_dist
            if short.any():
                candidate[short] = prev[short] + delta[short] / dist[short] * min_dist
            candidate[:, 0].clamp_(x_lo, x_hi)
            candidate[:, 1].clamp_(y_lo, y_hi)
            candidate[:, 2].clamp_(z_lo, z_hi)
            centers[:, g] = candidate
            prev = candidate

        # Gate normal = unit vector from previous point to gate center
        first_prev = self.base_init_pos.expand(n, 3)
        all_prev = torch.cat([first_prev.unsqueeze(1), centers[:, :-1]], dim=1)  # (n, num_gates, 3)
        raw_normals = centers - all_prev
        normals = raw_normals / raw_normals.norm(dim=2, keepdim=True).clamp_min(1e-6)

        self.gate_centers[envs_idx] = centers
        self.gate_normals[envs_idx] = normals

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _reward_gate_pass(self):
        return self.gate_passed_buf.to(dtype=gs.tc_float)

    def _reward_progress(self):
        """Exponential proximity reward: always positive, peaks at gate center.

        Replaces delta-distance to avoid the "crash-fast" collapse:
        with delta rewards the expected value under random actions is 0,
        making accumulated penalty > expected progress → policy learns to crash.
        Exponential shaping gives a continuous positive gradient toward each gate.
        """
        clamped = torch.clamp(self.current_gate_idx, 0, self.num_gates - 1)
        gi = clamped.view(-1, 1, 1).expand(-1, 1, 3)
        next_gate = self.gate_centers.gather(1, gi).squeeze(1)
        dist = torch.norm(next_gate - self.base_pos, dim=1)
        sigma = float(self.env_cfg.get("progress_sigma", 2.0))
        return torch.exp(-dist / sigma)

    # _reward_crash is inherited from BaseDroneEnv
