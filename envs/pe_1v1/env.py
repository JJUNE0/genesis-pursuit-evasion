"""PursuitEvasion1v1Env — concrete 1v1 PE env for Phase 1 (perfect info).

Fills the abstract hooks of ``BasePursuitEvasionEnv``:

- spawn distributions (attacker box, defender at distance [d_min, d_max]).
- g_mission sampling (uniform in goal box; rejection sampling against
  defender deferred to PR-C tuning, see plan §14).
- termination: mission_reached / captured / crash / oob (priority resolved
  by base step()).
- reward (TODO.md §1.2 표):
    attacker = R_progress(potential-based, clamp) + R_action_smooth + R_existential + terminal
    defender = -R_progress + R_action_smooth + terminal
  Terminal zero-sum is enforced via ``R_*_win + R_*_failed = 0`` magnitudes
  in the YAML; ``tests/sanity/test_reward_zero_sum.py`` asserts the contract.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import genesis as gs

from envs.base_pe_env import (
    BasePursuitEvasionEnv,
    ScriptedPolicy,
    quat_to_euler_deg,
    quat_to_rotmat_flat,
    rotate_to_body_frame,
)
from envs.pe_1v1.cfgs import (
    ATTACKER_OBS_DIM,
    CRITIC_PRIV_DIM,
    DEFAULT_OBS_SCALES,
    DEFENDER_OBS_DIM,
    G_MISSION_DIM,
    LAST_ACTION_DIM,
    S_FULL_DIM,
    S_KIN_DIM,
)
from utils.drone_params import DroneParams


def _uniform(low: float, high: float, n: int, device: Any) -> torch.Tensor:
    """Batched ``U(low, high)`` shape (n,) on device."""
    return torch.rand((n,), device=device, dtype=gs.tc_float) * (high - low) + low


class PursuitEvasion1v1Env(BasePursuitEvasionEnv):

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_envs: int,
        params: DroneParams,
        env_cfg: dict[str, Any],
        obs_cfg: dict[str, Any],
        reward_cfg: dict[str, Any],
        command_cfg: dict[str, Any],
        attacker_policy: ScriptedPolicy | None = None,
        defender_policy: ScriptedPolicy | None = None,
        show_viewer: bool = False,
    ):
        super().__init__(
            num_envs=num_envs,
            params=params,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            attacker_policy=attacker_policy,
            defender_policy=defender_policy,
            show_viewer=show_viewer,
        )

    # ------------------------------------------------------------------
    # PE-specific buffers / cfg unpack
    # ------------------------------------------------------------------

    def _init_pe_buffers(self) -> None:
        # termination thresholds (env_cfg)
        self.d_capture = float(self.env_cfg["d_capture"])
        self.g_mission_radius = float(self.env_cfg["g_mission_radius"])
        self.pitch_max_deg = float(self.env_cfg["pitch_max_deg"])
        self.roll_max_deg = float(self.env_cfg["roll_max_deg"])
        self.z_floor = float(self.env_cfg["z_floor"])
        self.v_max = float(self.env_cfg["v_max"])
        self.bound_xyz = float(self.env_cfg["bound_xyz"])
        # per-axis bound: defaults to bound_xyz when not overridden in cfg.
        self.bound_x = float(self.env_cfg.get("bound_x", self.bound_xyz))
        self.bound_y = float(self.env_cfg.get("bound_y", self.bound_xyz))
        self.bound_z = float(self.env_cfg.get("bound_z", self.bound_xyz))
        # 2026-05-04 — disable attitude (roll/pitch) as crash trigger when
        # ``crash_on_attitude=False``. attitude_penalty handles regulation;
        # hard episode termination on roll/pitch removed. Default True for
        # backward compat — every existing yaml keeps current behavior.
        self.crash_on_attitude = bool(self.env_cfg.get("crash_on_attitude", True))
        # 2026-05-04 — disable speed (|v|>v_max) as crash trigger when
        # ``crash_on_speed=False``. velocity_penalty (safety_penalty) handles
        # regulation. Default True for backward compat.
        self.crash_on_speed = bool(self.env_cfg.get("crash_on_speed", True))

        # reward scalars
        self.progress_gamma = float(self.reward_cfg.get("progress_gamma", 0.99))
        self.progress_max = float(self.reward_cfg["progress_max"])
        # Phase B (defender fragility diag) — action_smooth per-side. Legacy
        # `action_smooth` 단일 키는 양쪽 default로 fallback (Phase 1.0/1.5/A yaml
        # 무영향). Phase B yaml은 두 신규 키로 명시.
        _legacy_action_smooth = float(self.reward_cfg.get("action_smooth", 1.0e-4))
        self.action_smooth_attacker = float(self.reward_cfg.get(
            "action_smooth_attacker", _legacy_action_smooth
        ))
        self.action_smooth_defender = float(self.reward_cfg.get(
            "action_smooth_defender", _legacy_action_smooth
        ))
        self.existential_w1 = float(self.reward_cfg["existential_w1"])
        # PR-F (2026-04-28) — angular velocity penalty per-side.
        # R_omega = -coef · ‖ω‖². attacker/defender 자기 ang_vel만 페널티.
        # P0/P1 진단 (logs/pe1v1_v0/attacker_v0/...) 결과: attacker 학습 후기에
        # |ω| 1.65 → 10.06 rad/s 단조 증가 (자세 발산 trap). defender |ω|는 5.5
        # rad/s = chase 기동 자연 자세 (95% capture, ep_len 1초). 따라서
        # attacker엔 5e-5 enable, defender는 0 유지 — chase 기동 차단 방지.
        # legacy `omega_coef` 키는 silent fallback 위험 차단 위해 제거 (KeyError 강제).
        self.omega_coef_attacker = float(self.reward_cfg["omega_coef_attacker"])
        self.omega_coef_defender = float(self.reward_cfg["omega_coef_defender"])
        # Phase F v6 (2026-05-11) — defender chase reward attenuation.
        # default 1.0 = 기존 동작 (pure_pe r_chase = -r_progress_buf). 사용자 critique
        # on F.1c v3: defender chase가 너무 강해 평균 episode 1.4s, timeout 0%.
        # attenuation<1 → defender chase 약화 → attacker timeout 학습 기회 확보.
        self.defender_chase_attenuation = float(
            self.reward_cfg.get("defender_chase_attenuation", 1.0)
        )
        # Phase B — |ω|² clamp. self_crash incentive 차단 + 정상 chase 신호 보존.
        # default ∞ → no clamp (Phase 1.0/1.5/A 무영향). Phase B yaml은 25.0 명시
        # (즉 |ω| > 5 rad/s 영역에서 페널티 saturate).
        self.omega_clamp_max_sq = float(self.reward_cfg.get(
            "omega_clamp_max_sq", float("inf")
        ))
        # 2026-05-03 — Attitude penalty: quadratic above ``attitude_pad_deg``.
        # Triggered by inspect_def_crash_timing diagnosis (100% roll_bad at 60°
        # threshold). Adds soft pressure in [pad, threshold] band so chase
        # banking self-regulates before crash. coef=0 (default) → backward
        # compatible (no penalty, legacy yaml unaffected).
        self.attitude_pad_deg = float(self.reward_cfg.get("attitude_pad_deg", 60.0))
        self.attitude_coef = float(self.reward_cfg.get("attitude_coef", 0.0))
        # 2026-05-04 — Soft boundary penalties (quadratic, symmetric per side).
        # Diagnosed in inspect: relaxed crash 80°에서 def 자살이 z_below(83.6%)
        # + oob(39.8%)로 전이 — over-commit chase가 boundary 부드러운 압력
        # 부재로 단번에 hard threshold 터치. attitude penalty와 같은 패턴.
        # All defaults 0 (no penalty) → backward compat.
        self.velocity_pad = float(self.reward_cfg.get("velocity_pad", float("inf")))
        self.velocity_coef = float(self.reward_cfg.get("velocity_coef", 0.0))
        self.z_safety_pad = float(self.reward_cfg.get("z_safety_pad", 0.0))
        self.z_safety_coef = float(self.reward_cfg.get("z_safety_coef", 0.0))
        self.bound_safety_pad = float(self.reward_cfg.get("bound_safety_pad", 0.0))
        self.bound_safety_coef = float(self.reward_cfg.get("bound_safety_coef", 0.0))
        # PR-D — Gaussian dense rewards (hover env 차용; coef · exp(-α · d)).
        # attacker가 정지(StationaryAttacker) 상대일 때 R_progress=0이라
        # defender가 학습 신호를 못 받는 문제를 해결.
        # PR-D3 — dense → sparse linear decay schedule (TODO §6.2 ablation):
        #   coef(step) = init + (final − init) · min(step / decay_steps, 1)
        #   decay_steps=0 (default) → constant coef (= init = chase_coef yaml).
        # Phase E (TODO §5.6) — chase_coef per-side. Legacy single-key fallback
        # 그대로 (Phase 1.0/1.5/A/B/C/D yaml 무영향). 새 yaml에선
        # chase_coef_attacker_*, chase_coef_defender_*, chase_alpha_attacker,
        # chase_alpha_defender 명시.
        self.chase_coef_init = float(self.reward_cfg.get(
            "chase_coef_init", self.reward_cfg.get("chase_coef", 2.0e-4)
        ))
        self.chase_coef_final = float(self.reward_cfg.get(
            "chase_coef_final", self.chase_coef_init
        ))
        self.chase_alpha = float(self.reward_cfg.get("chase_alpha", 1.0))
        self.chase_coef_attacker_init = float(self.reward_cfg.get(
            "chase_coef_attacker_init", self.chase_coef_init
        ))
        self.chase_coef_attacker_final = float(self.reward_cfg.get(
            "chase_coef_attacker_final", self.chase_coef_final
        ))
        self.chase_coef_defender_init = float(self.reward_cfg.get(
            "chase_coef_defender_init", self.chase_coef_init
        ))
        self.chase_coef_defender_final = float(self.reward_cfg.get(
            "chase_coef_defender_final", self.chase_coef_final
        ))
        self.chase_alpha_attacker = float(self.reward_cfg.get(
            "chase_alpha_attacker", self.chase_alpha
        ))
        self.chase_alpha_defender = float(self.reward_cfg.get(
            "chase_alpha_defender", self.chase_alpha
        ))
        self.goal_dense_coef_init = float(self.reward_cfg.get(
            "goal_dense_coef_init", self.reward_cfg.get("goal_dense_coef", 2.0e-4)
        ))
        self.goal_dense_coef_final = float(self.reward_cfg.get(
            "goal_dense_coef_final", self.goal_dense_coef_init
        ))
        self.goal_dense_alpha = float(self.reward_cfg.get("goal_dense_alpha", 1.0))
        self.decay_steps = int(self.reward_cfg.get("decay_steps", 0))
        # current coefs (mutated each env step by _update_dense_schedule)
        self.chase_coef = self.chase_coef_init                          # legacy alias (deprecated)
        self.chase_coef_attacker = self.chase_coef_attacker_init        # Phase E
        self.chase_coef_defender = self.chase_coef_defender_init        # Phase E
        self.goal_dense_coef = self.goal_dense_coef_init
        self._dense_global_step = 0
        self.R_mission_win = float(self.reward_cfg["R_mission_win"])
        self.R_captured = float(self.reward_cfg["R_captured"])
        self.R_timeout = float(self.reward_cfg["R_timeout"])
        self.R_capture_win = float(self.reward_cfg["R_capture_win"])
        self.R_mission_failed = float(self.reward_cfg["R_mission_failed"])
        self.R_timeout_win = float(self.reward_cfg["R_timeout_win"])
        # crash/oob per-side reward magnitudes — minimax 부호 보정.
        # 부등식 가드는 자살/cheese 익스플로잇 차단:
        #   - R_self_crash > R_captured        → attacker가 capture 회피용 자살 학습
        #   - R_self_crash > R_mission_failed  → defender가 mission 회피용 자살 학습
        #   - R_opp_crash > R_capture_win      → defender가 정공 capture보다 harassment 선호
        #   - R_opp_crash > R_mission_win      → attacker가 mission보다 defender lure 선호
        self.R_self_crash = float(self.reward_cfg["R_self_crash"])
        self.R_opp_crash = float(self.reward_cfg["R_opp_crash"])
        assert self.R_self_crash <= self.R_captured, (
            f"R_self_crash({self.R_self_crash}) > R_captured({self.R_captured}): "
            "attacker가 capture 회피 위해 자살 학습 가능"
        )
        assert self.R_self_crash <= self.R_mission_failed, (
            f"R_self_crash({self.R_self_crash}) > R_mission_failed({self.R_mission_failed}): "
            "defender가 mission 실패 회피 위해 자살 학습 가능"
        )
        assert self.R_opp_crash <= self.R_capture_win, (
            f"R_opp_crash({self.R_opp_crash}) > R_capture_win({self.R_capture_win}): "
            "defender가 정공 capture보다 harassment 선호 학습 가능"
        )
        assert self.R_opp_crash <= self.R_mission_win, (
            f"R_opp_crash({self.R_opp_crash}) > R_mission_win({self.R_mission_win}): "
            "attacker가 mission보다 defender lure 선호 학습 가능"
        )

        # obs scaling (PR-D — hover env 차용; 입력 분포 ~[-1,1])
        scales = self.obs_cfg.get("obs_scales", DEFAULT_OBS_SCALES)
        self.pos_scale = float(scales.get("pos", DEFAULT_OBS_SCALES["pos"]))
        self.vel_scale = float(scales.get("vel", DEFAULT_OBS_SCALES["vel"]))
        self.ang_vel_scale = float(scales.get("ang_vel", DEFAULT_OBS_SCALES["ang_vel"]))

        # spawn / goal ranges (command_cfg).
        # PR-D: defender spawn is now a 3D box (x/y/z) instead of a polar
        # distance-from-attacker — phase별로 attacker/defender가 분리된 영역에
        # spawn하도록 조정하기 쉽고 (e.g., attacker 우측, defender 좌측),
        # collapse 방지는 box들이 겹치지 않게 yaml에서 보장.
        spawn = self.command_cfg["spawn"]
        att_box = spawn["attacker_init_pos"]
        self._attacker_x_range = tuple(att_box["x"])
        self._attacker_y_range = tuple(att_box["y"])
        self._attacker_z_range = tuple(att_box["z"])

        d_box = spawn["defender_init_pos"]
        self._defender_x_range = tuple(d_box["x"])
        self._defender_y_range = tuple(d_box["y"])
        self._defender_z_range = tuple(d_box["z"])

        goal = self.command_cfg["g_mission"]["box"]
        self._goal_x_range = tuple(goal["x"])
        self._goal_y_range = tuple(goal["y"])
        self._goal_z_range = tuple(goal["z"])

        # PE state buffers
        B = self.num_envs
        f = gs.tc_float
        d = self.device
        self.dist_to_attacker_defender = torch.zeros((B,), device=d, dtype=f)   # shape: (B,)
        # Phase F (pure_pe) — attacker-defender 거리의 progress shaping 위해 t-1 값 추적.
        self.last_dist_to_attacker_defender = torch.zeros((B,), device=d, dtype=f)  # shape: (B,)
        self.r_progress_buf = torch.zeros((B,), device=d, dtype=f)              # shape: (B,)

        # constants
        self._identity_quat = torch.zeros((1, 4), device=d, dtype=f)
        self._identity_quat[0, 0] = 1.0
        self._existential_per_step = -self.existential_w1 / float(self.max_episode_length)

    # ------------------------------------------------------------------
    # Spawn / reset
    # ------------------------------------------------------------------

    def _spawn(self, envs_idx: torch.Tensor) -> None:
        n = int(envs_idx.numel())
        if n == 0:
            return
        d = self.device
        f = gs.tc_float

        # attacker pose (box).
        a_pos = torch.zeros((n, 3), device=d, dtype=f)
        a_pos[:, 0] = _uniform(*self._attacker_x_range, n=n, device=d)
        a_pos[:, 1] = _uniform(*self._attacker_y_range, n=n, device=d)
        a_pos[:, 2] = _uniform(*self._attacker_z_range, n=n, device=d)

        # defender pose (box). PR-D — yaml에서 attacker box와 분리된 영역으로
        # 잡으면 collapse 자체가 발생 불가.
        d_pos = torch.zeros((n, 3), device=d, dtype=f)
        d_pos[:, 0] = _uniform(*self._defender_x_range, n=n, device=d)
        d_pos[:, 1] = _uniform(*self._defender_y_range, n=n, device=d)
        d_pos[:, 2] = _uniform(*self._defender_z_range, n=n, device=d)

        # write to mirror buffers + Genesis sim state
        self.attacker_pos[envs_idx] = a_pos
        self.attacker_vel[envs_idx] = 0.0
        self.attacker_quat[envs_idx] = self._identity_quat
        self.attacker_ang_vel[envs_idx] = 0.0

        self.defender_pos[envs_idx] = d_pos
        self.defender_vel[envs_idx] = 0.0
        self.defender_quat[envs_idx] = self._identity_quat
        self.defender_ang_vel[envs_idx] = 0.0

        identity_q = self._identity_quat.expand(n, 4).contiguous()
        self.attacker_drone.set_pos(a_pos, zero_velocity=True, envs_idx=envs_idx)
        self.attacker_drone.set_quat(identity_q, zero_velocity=True, envs_idx=envs_idx)
        self.defender_drone.set_pos(d_pos, zero_velocity=True, envs_idx=envs_idx)
        self.defender_drone.set_quat(identity_q, zero_velocity=True, envs_idx=envs_idx)

    def _reset_pe(self, envs_idx: torch.Tensor) -> None:
        n = int(envs_idx.numel())
        if n == 0:
            return
        d = self.device
        f = gs.tc_float
        # uniform g_mission sampling. Rejection sampling vs defender distance
        # is a tuning concern deferred to PR-C (plan §14 open question).
        g = torch.zeros((n, 3), device=d, dtype=f)
        g[:, 0] = _uniform(*self._goal_x_range, n=n, device=d)
        g[:, 1] = _uniform(*self._goal_y_range, n=n, device=d)
        g[:, 2] = _uniform(*self._goal_z_range, n=n, device=d)
        self.g_mission[envs_idx] = g

    # ------------------------------------------------------------------
    # Per-step state update
    # ------------------------------------------------------------------

    def _update_pe_state(self) -> None:
        # dist_to_goal — clamp_min for safe norm + division (CLAUDE.md §4)
        delta_g = self.attacker_pos - self.g_mission
        self.dist_to_goal = delta_g.norm(dim=1).clamp_min(1e-8)

        # dist between drones
        delta_ad = self.attacker_pos - self.defender_pos
        self.dist_to_attacker_defender = delta_ad.norm(dim=1).clamp_min(1e-8)

        # potential-based progress (clamped). Φ = -dist; r = γΦ_{t+1} - Φ_t.
        # Phase F (2026-05-10) — game_mode 분기:
        #   - "mission" (기존): attacker→g_mission 거리 변화 (last_d_g - γ·d_g).
        #   - "pure_pe": attacker-defender 거리 변화 — attacker는 거리 증가(+) = 도망,
        #     defender는 -r_progress_buf로 거리 감소(+) = chase.
        if self.game_mode == "pure_pe":
            # attacker 도망 reward = κa·(d_t - d_{t-1})
            raw = self.progress_gamma * self.dist_to_attacker_defender - self.last_dist_to_attacker_defender
        else:  # mission
            raw = self.last_dist_to_goal - self.progress_gamma * self.dist_to_goal
        self.r_progress_buf = torch.clamp(raw, -self.progress_max, self.progress_max)

        # PR-D3 — Gaussian dense coef schedule (linear decay init → final).
        if self.decay_steps > 0:
            self._dense_global_step += 1
            frac = min(self._dense_global_step / float(self.decay_steps), 1.0)
            self.chase_coef = (
                self.chase_coef_init
                + (self.chase_coef_final - self.chase_coef_init) * frac
            )
            self.chase_coef_attacker = (
                self.chase_coef_attacker_init
                + (self.chase_coef_attacker_final - self.chase_coef_attacker_init) * frac
            )
            self.chase_coef_defender = (
                self.chase_coef_defender_init
                + (self.chase_coef_defender_final - self.chase_coef_defender_init) * frac
            )
            self.goal_dense_coef = (
                self.goal_dense_coef_init
                + (self.goal_dense_coef_final - self.goal_dense_coef_init) * frac
            )

    # ------------------------------------------------------------------
    # Observations (Phase 1: g_mission visible to attacker only)
    # ------------------------------------------------------------------

    def _attacker_obs(self) -> torch.Tensor:
        """Attacker actor obs (PR-M, 2026-04-29 + Phase 3a, 2026-05-12).

        Self pos: world frame (env 내부 spatial reward 계산과 일치).
        Self vel: body frame (action도 body frame이라 representation 일관성).
        상대 drone + g_mission: attacker body frame (rotation/translation invariance).

        Phase 3a — defender state is read from the τ-delay ring buffer (index 0
        = τ steps ago). Body-frame rotation uses the CURRENT attacker pose, so
        the actor sees "where defender WAS, relative to where I am NOW". For
        τ=0 the buffer holds the current state, so behavior reduces to legacy.
        """
        pos_a = self.attacker_pos * self.pos_scale                        # shape: (B, 3) world
        vel_a = rotate_to_body_frame(
            self.attacker_vel, self.attacker_quat,
        ) * self.vel_scale                                                # shape: (B, 3) body
        rot_a = quat_to_rotmat_flat(self.attacker_quat)                   # shape: (B, 9) body→world
        ang_a = self.attacker_ang_vel * self.ang_vel_scale                # shape: (B, 3) body

        # τ-delayed defender world-frame state (Phase 3a). τ=0 → current.
        delayed_kin = self.defender_state_history[0]                      # shape: (B, 6)
        delayed_def_pos = delayed_kin[:, :3]                              # shape: (B, 3) world
        delayed_def_vel = delayed_kin[:, 3:]                              # shape: (B, 3) world

        # 상대 위치/속도 → attacker body frame (delayed defender vs current attacker)
        rel_pos_d = rotate_to_body_frame(
            delayed_def_pos - self.attacker_pos, self.attacker_quat,
        ) * self.pos_scale                                                # shape: (B, 3)
        rel_vel_d = rotate_to_body_frame(
            delayed_def_vel - self.attacker_vel, self.attacker_quat,
        ) * self.vel_scale                                                # shape: (B, 3)

        # g_mission 도 body frame 상대값
        rel_g = rotate_to_body_frame(
            self.g_mission - self.attacker_pos, self.attacker_quat,
        ) * self.pos_scale                                                # shape: (B, 3)

        last_a = self.last_attacker_action                                # shape: (B, 4)

        s_a = torch.cat([pos_a, vel_a, rot_a, ang_a], dim=1)              # shape: (B, 18)
        s_d_kin = torch.cat([rel_pos_d, rel_vel_d], dim=1)                # shape: (B, 6) body
        obs = torch.cat([s_a, s_d_kin, rel_g, last_a], dim=1)             # shape: (B, 31)
        assert obs.shape[1] == ATTACKER_OBS_DIM == \
            S_FULL_DIM + S_KIN_DIM + G_MISSION_DIM + LAST_ACTION_DIM
        return obs

    def _defender_obs(self) -> torch.Tensor:
        """Defender actor obs (PR-M, 2026-04-29).

        Self pos: world frame. Self vel: body frame.
        상대 drone (attacker): defender body frame.
        """
        pos_d = self.defender_pos * self.pos_scale                        # world
        vel_d = rotate_to_body_frame(
            self.defender_vel, self.defender_quat,
        ) * self.vel_scale                                                # body
        rot_d = quat_to_rotmat_flat(self.defender_quat)                   # body→world
        ang_d = self.defender_ang_vel * self.ang_vel_scale                # body

        # 상대 위치/속도 → defender body frame
        rel_pos_a = rotate_to_body_frame(
            self.attacker_pos - self.defender_pos, self.defender_quat,
        ) * self.pos_scale
        rel_vel_a = rotate_to_body_frame(
            self.attacker_vel - self.defender_vel, self.defender_quat,
        ) * self.vel_scale

        last_d = self.last_defender_action

        s_d = torch.cat([pos_d, vel_d, rot_d, ang_d], dim=1)              # shape: (B, 18)
        s_a_kin = torch.cat([rel_pos_a, rel_vel_a], dim=1)                # shape: (B, 6) body
        obs = torch.cat([s_d, s_a_kin, last_d], dim=1)                    # shape: (B, 28)
        assert obs.shape[1] == DEFENDER_OBS_DIM == \
            S_FULL_DIM + S_KIN_DIM + LAST_ACTION_DIM
        return obs

    # ------------------------------------------------------------------
    # Phase 3d (TODO §3d, 2026-05-12) — BPQL-style asymmetric critic.
    # [Kim et al. 2023, NeurIPS] Belief projection: critic Q_β is defined over
    # the original (non-augmented) state space — here, the opponent's current
    # 6D kinematics in ego body frame, no τ-delay. Same representation as the
    # actor's s_d_kin slot so the critic only differs by removing the delay.
    # Supersedes PR-E.1 10D (vel+ang_vel+quat) which was extra-dynamics, not
    # belief projection.
    # ------------------------------------------------------------------

    def _critic_priv_obs(self) -> torch.Tensor:
        """BPQL privileged obs — opponent current 6D kinematics, ego body frame.

        Returns ``rel_pos_GT(3) + rel_vel_GT(3) = 6D`` matching the actor's
        s_d_kin slice exactly except for the τ-delay removal. Body-frame
        rotation uses the ego's current quaternion, identical to the actor
        path — so when τ=0 this slice equals the actor's s_d_kin slot
        bit-for-bit.
        """
        if self.ego == "attacker":
            rel_pos_GT = rotate_to_body_frame(
                self.defender_pos - self.attacker_pos, self.attacker_quat,
            ) * self.pos_scale                                    # (B, 3)
            rel_vel_GT = rotate_to_body_frame(
                self.defender_vel - self.attacker_vel, self.attacker_quat,
            ) * self.vel_scale                                    # (B, 3)
        else:
            rel_pos_GT = rotate_to_body_frame(
                self.attacker_pos - self.defender_pos, self.defender_quat,
            ) * self.pos_scale
            rel_vel_GT = rotate_to_body_frame(
                self.attacker_vel - self.defender_vel, self.defender_quat,
            ) * self.vel_scale
        obs = torch.cat([rel_pos_GT, rel_vel_GT], dim=1)          # shape: (B, 6)
        assert obs.shape[1] == CRITIC_PRIV_DIM
        return obs

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _compute_mission_reached(self) -> torch.Tensor:
        return self.dist_to_goal < self.g_mission_radius

    def _compute_capture(self) -> torch.Tensor:
        return self.dist_to_attacker_defender < self.d_capture

    def _compute_crash(self) -> tuple[torch.Tensor, torch.Tensor]:
        a_crash, a_subs = self._drone_crash(
            self.attacker_pos, self.attacker_vel, self.attacker_quat,
        )
        d_crash, d_subs = self._drone_crash(
            self.defender_pos, self.defender_vel, self.defender_quat,
        )
        # 2026-05-04 — expose per-condition flags so base_pe_env can mask + log
        # them to wandb. Diagnostic for "which trigger fires most" without
        # offline inspect script. Mask is applied in base_pe_env.step().
        self.crash_subs = {
            "att_crash_z":      a_subs["z_below"],
            "att_crash_speed":  a_subs["speed_high"],
            "att_crash_roll":   a_subs["roll_bad"],
            "att_crash_pitch":  a_subs["pitch_bad"],
            "def_crash_z":      d_subs["z_below"],
            "def_crash_speed":  d_subs["speed_high"],
            "def_crash_roll":   d_subs["roll_bad"],
            "def_crash_pitch":  d_subs["pitch_bad"],
        }
        return a_crash, d_crash

    def _drone_crash(
        self, pos: torch.Tensor, vel: torch.Tensor, quat: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_below = pos[:, 2] < self.z_floor
        speed_high = vel.norm(dim=1) > self.v_max
        eul = quat_to_euler_deg(quat)
        roll_bad = eul[:, 0].abs() > self.roll_max_deg
        pitch_bad = eul[:, 1].abs() > self.pitch_max_deg
        # Build hard-crash mask incrementally based on enabled triggers.
        # Sub-condition flags (z_below/speed_high/roll_bad/pitch_bad) always
        # populated for diagnostic logging (crash_subs).
        any_crash = z_below
        if self.crash_on_speed:
            any_crash = any_crash | speed_high
        if self.crash_on_attitude:
            any_crash = any_crash | roll_bad | pitch_bad
        subs = {
            "z_below": z_below,
            "speed_high": speed_high,
            "roll_bad": roll_bad,
            "pitch_bad": pitch_bad,
        }
        return any_crash, subs

    def _compute_oob(self) -> tuple[torch.Tensor, torch.Tensor]:
        def _oob(pos: torch.Tensor) -> torch.Tensor:
            return (
                (pos[:, 0].abs() > self.bound_x)
                | (pos[:, 1].abs() > self.bound_y)
                | (pos[:, 2].abs() > self.bound_z)
            )
        return _oob(self.attacker_pos), _oob(self.defender_pos)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _attitude_penalty(self, quat: torch.Tensor) -> torch.Tensor:
        """Quadratic soft penalty for |roll| or |pitch| above ``attitude_pad_deg``.

        Zero inside ``[-pad, pad]`` band, quadratic outside. Designed to
        self-regulate banking before crash threshold (60°). Diagnosed root
        cause for 100% def_crash via roll_bad in phase_m wide-spawn ams.
        """
        eul = quat_to_euler_deg(quat)
        over = (
            torch.maximum(eul[:, 0].abs(), eul[:, 1].abs()) - self.attitude_pad_deg
        ).clamp_min(0.0)
        return -self.attitude_coef * over * over

    def _safety_penalty(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        """Quadratic soft penalty for proximity to crash/oob boundaries.

        Three independent components, all summed:
          1. velocity:     |v| > velocity_pad     → -coef · over²
          2. z_safety:     z   < z_safety_pad     → -coef · (pad - z)²  (지면 가까이)
          3. bound_safety: |x|>bound_x-pad / |y|>bound_y-pad / |z|>bound_z-pad → -coef · sum(over²)

        Each coef defaults 0 → no penalty (backward compatible).
        Diagnosed: phase_m_ams (relaxed 80°)에서 attitude penalty가 roll banking은
        regulate 했지만 over-commit chase가 z_floor / bound 단번에 터치 (z_below
        83.6%, def_oob 39.8%). 부드러운 사전 압력 추가.
        """
        out = torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
        if self.velocity_coef > 0.0:
            speed = vel.norm(dim=1)
            over = (speed - self.velocity_pad).clamp_min(0.0)
            out = out - self.velocity_coef * over * over
        if self.z_safety_coef > 0.0:
            over = (self.z_safety_pad - pos[:, 2]).clamp_min(0.0)
            out = out - self.z_safety_coef * over * over
        if self.bound_safety_coef > 0.0:
            pad = self.bound_safety_pad
            bx = max(0.0, self.bound_x - pad)
            by = max(0.0, self.bound_y - pad)
            bz = max(0.0, self.bound_z - pad)
            ox = (pos[:, 0].abs() - bx).clamp_min(0.0)
            oy = (pos[:, 1].abs() - by).clamp_min(0.0)
            oz = (pos[:, 2].abs() - bz).clamp_min(0.0)
            out = out - self.bound_safety_coef * (ox * ox + oy * oy + oz * oz)
        return out

    def _attacker_reward(self, attacker_action: torch.Tensor) -> torch.Tensor:
        # Phase F (2026-05-10) — game_mode 분기.
        if self.game_mode == "pure_pe":
            return self._attacker_reward_pure_pe(attacker_action)
        r_progress = self.r_progress_buf
        action_diff = attacker_action - self.last_attacker_action
        r_smooth = -self.action_smooth_attacker * (action_diff * action_diff).sum(dim=1)
        r_existential = torch.full(
            (self.num_envs,), self._existential_per_step, device=self.device, dtype=gs.tc_float,
        )
        r_goal_dense = self.goal_dense_coef * torch.exp(-self.goal_dense_alpha * self.dist_to_goal)
        # Phase E — attacker r_evade에 attacker-side coef 사용 (defender chase 강화가 attacker mission 학습 차단하지 않도록)
        r_evade = -self.chase_coef_attacker * torch.exp(-self.chase_alpha_attacker * self.dist_to_attacker_defender)
        # PR-F — angular velocity penalty (attacker 자세 발산 dense 차단).
        # Phase B — clamp |ω|² to omega_clamp_max_sq before scaling.
        omega_sq = (self.attacker_ang_vel * self.attacker_ang_vel).sum(dim=1)
        omega_sq = omega_sq.clamp(max=self.omega_clamp_max_sq)
        r_omega = -self.omega_coef_attacker * omega_sq
        # 2026-05-03 — quadratic attitude penalty (symmetric on both sides).
        r_attitude = self._attitude_penalty(self.attacker_quat)
        # 2026-05-04 — soft boundary penalty (velocity / z floor / bound).
        r_safety = self._safety_penalty(self.attacker_pos, self.attacker_vel)
        r_terminal = self._terminal_reward_attacker()
        return r_progress + r_smooth + r_existential + r_goal_dense + r_evade + r_omega + r_attitude + r_safety + r_terminal

    def _defender_reward(self, defender_action: torch.Tensor) -> torch.Tensor:
        # Phase F — game_mode 분기.
        if self.game_mode == "pure_pe":
            return self._defender_reward_pure_pe(defender_action)
        # Dense base: R_block = -R_progress.
        # zero-sum 의도지만, attacker가 정지(StationaryAttacker)일 때는 양쪽 모두 0이라
        # defender 학습 신호가 사라짐 → 별도 chase reward로 보강.
        r_block = -self.r_progress_buf
        action_diff = defender_action - self.last_defender_action
        r_smooth = -self.action_smooth_defender * (action_diff * action_diff).sum(dim=1)
        # PR-D — defender chase dense (Gaussian on attacker-defender distance).
        # Phase E — defender r_chase에 defender-side coef 사용 (대칭화 의도 유지)
        r_chase = self.chase_coef_defender * torch.exp(-self.chase_alpha_defender * self.dist_to_attacker_defender)
        # TODO(phase15): r_goal_protect = -goal_dense_coef·exp(-α·dist_to_goal).
        # attacker의 r_goal_dense와 dense level minimax mirror 후보. 보류 사유:
        # defender obs에 g_mission이 없어 actor가 reward 인과 추론 불가 (free
        # energy). Phase 1.5 attacker 학습 진입 시 obs schema 확장과 함께 도입
        # 결정 (CLAUDE.md §1 — obs 변경은 사용자 명시 승인 필요).
        # PR-F — angular velocity penalty (defender 자기 자세만). 0 유지 default —
        # chase 기동 시 자연 |ω|≈5.5 rad/s 차단 방지 (P1 진단 결과).
        # Phase B — defender omega 0→2e-5 + clamp |ω|²→25. ams_s2_v2 c10 진단:
        # stationary 상대 def_crash 82%, action sat_frac 0.99 (bang-bang 확정).
        omega_sq = (self.defender_ang_vel * self.defender_ang_vel).sum(dim=1)
        omega_sq = omega_sq.clamp(max=self.omega_clamp_max_sq)
        r_omega = -self.omega_coef_defender * omega_sq
        # 2026-05-03 — quadratic attitude penalty (symmetric on both sides).
        r_attitude = self._attitude_penalty(self.defender_quat)
        # 2026-05-04 — soft boundary penalty (velocity / z floor / bound).
        r_safety = self._safety_penalty(self.defender_pos, self.defender_vel)
        r_terminal = self._terminal_reward_defender()
        return r_block + r_smooth + r_chase + r_omega + r_attitude + r_safety + r_terminal

    def _terminal_reward_attacker(self) -> torch.Tensor:
        # mission/captured/timeout buckets are priority-resolved
        # (mutually exclusive). crash/oob are per-side: a single env can have
        # both attacker_crash AND defender_crash set in the crash bucket, so
        # self_term and opp_term are evaluated separately and sum.
        z = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        z = torch.where(self.mission_reached_buf, z + self.R_mission_win, z)
        z = torch.where(self.captured_buf, z + self.R_captured, z)
        z = torch.where(self.timeout_buf, z + self.R_timeout, z)
        # 자기(attacker) 추락/oob → R_self_crash 페널티.
        self_term = self.attacker_crash_buf | self.attacker_oob_buf
        z = torch.where(self_term, z + self.R_self_crash, z)
        # 상대(defender) 추락/oob → R_opp_crash 보너스 (반사이익).
        opp_term = self.defender_crash_buf | self.defender_oob_buf
        z = torch.where(opp_term, z + self.R_opp_crash, z)
        return z

    def _terminal_reward_defender(self) -> torch.Tensor:
        # defender 입장: 자기 = defender, 상대 = attacker (attacker 측과 대칭).
        z = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        z = torch.where(self.mission_reached_buf, z + self.R_mission_failed, z)
        z = torch.where(self.captured_buf, z + self.R_capture_win, z)
        z = torch.where(self.timeout_buf, z + self.R_timeout_win, z)
        self_term = self.defender_crash_buf | self.defender_oob_buf
        z = torch.where(self_term, z + self.R_self_crash, z)
        opp_term = self.attacker_crash_buf | self.attacker_oob_buf
        z = torch.where(opp_term, z + self.R_opp_crash, z)
        return z

    # ------------------------------------------------------------------
    # Phase F (2026-05-10) — Pure PE reward functions (AMSPB-style).
    # 5~6항만 (소프트 페널티 모두 제거: action_smooth, attitude, velocity_safety,
    # z_safety, bound_safety, chase, goal_dense). Hard crash 90°/v_max/z_floor가
    # 자세/속도/지면 안전 규제. terminal: capture/escape ±κc, timeout ±κt, self_crash.
    # ------------------------------------------------------------------

    def _attacker_reward_pure_pe(self, attacker_action: torch.Tensor) -> torch.Tensor:
        del attacker_action  # action_smooth 제거 — 사용 안 함
        # r_progress = κa·(d_atk_def_t - d_atk_def_{t-1}) clamped (game_mode 분기로 r_progress_buf에 저장됨)
        r_progress = self.r_progress_buf
        # κs (survival per step) — attacker만
        r_existential = torch.full(
            (self.num_envs,), self._existential_per_step, device=self.device, dtype=gs.tc_float,
        )
        # -κbr·‖ω‖² (L2)
        omega_sq = (self.attacker_ang_vel * self.attacker_ang_vel).sum(dim=1)
        omega_sq = omega_sq.clamp(max=self.omega_clamp_max_sq)
        r_omega = -self.omega_coef_attacker * omega_sq
        r_terminal = self._terminal_reward_attacker_pure_pe()
        return r_progress + r_existential + r_omega + r_terminal

    def _defender_reward_pure_pe(self, defender_action: torch.Tensor) -> torch.Tensor:
        del defender_action
        # r_chase = κa·(d_atk_def_{t-1} - d_atk_def_t) — attacker reward의 부호 반전.
        # Phase F v6: × defender_chase_attenuation (yaml, default 1.0).
        r_chase = -self.r_progress_buf * self.defender_chase_attenuation
        # ★ κs 없음 (사용자 spec — defender 생존 보상은 chase 약화 요인)
        omega_sq = (self.defender_ang_vel * self.defender_ang_vel).sum(dim=1)
        omega_sq = omega_sq.clamp(max=self.omega_clamp_max_sq)
        r_omega = -self.omega_coef_defender * omega_sq
        r_terminal = self._terminal_reward_defender_pure_pe()
        return r_chase + r_omega + r_terminal

    def _terminal_reward_attacker_pure_pe(self) -> torch.Tensor:
        """Pure PE attacker terminal: mission 무시, escape(timeout)=+, captured=-, self_crash=-.

        yaml의 R_timeout_win을 +κt로 사용 (도망 성공 보상). yaml R_timeout(-5)은 mission 모드용.
        Phase F yaml에서 R_timeout_win=+5, R_timeout=-5 둘 다 ±κt로 설정.
        """
        z = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        # mission_reached 무시 (pure_pe라 안 일어남, 안전망 차원에서 0)
        z = torch.where(self.captured_buf, z + self.R_captured, z)         # -κc (=-10)
        z = torch.where(self.timeout_buf, z + self.R_timeout_win, z)        # +κt (=+5, escape)
        self_term = self.attacker_crash_buf | self.attacker_oob_buf
        z = torch.where(self_term, z + self.R_self_crash, z)                # -10
        # opponent crash → 별도 보상 X (R_opp_crash=0 합의)
        return z

    def _terminal_reward_defender_pure_pe(self) -> torch.Tensor:
        """Pure PE defender terminal: capture=+, timeout(miss)=-, self_crash=-."""
        z = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        z = torch.where(self.captured_buf, z + self.R_capture_win, z)       # +κc (=+10)
        z = torch.where(self.timeout_buf, z + self.R_timeout, z)            # -κt (=-5, miss)
        self_term = self.defender_crash_buf | self.defender_oob_buf
        z = torch.where(self_term, z + self.R_self_crash, z)                # -10
        return z
