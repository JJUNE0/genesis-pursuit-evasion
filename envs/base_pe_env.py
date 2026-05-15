"""BasePursuitEvasionEnv — 1v1 PE env ABC for Phase 1 (perfect-info baseline).

Owns the Genesis Scene with two Drone entities (attacker + defender), runs
both drones inside a single batched ``scene.step()`` (V1.5 pattern), exposes
the rsl-rl 5.2 contract (``cfg``, ``obs_groups``, ``num_actions``,
``num_envs``, ``device``, ``reset()``, ``step(actions)``, ``get_observations()``).

Phase 1 design constraints:

- delay buffer / forecaster are **out of scope** (Phase 2 / 3).
- ``obs_groups = {"actor": ["policy"], "critic": ["policy"]}`` — Phase 4 will
  add ``critic_priv`` without touching this base.
- Action space is per-prop scaled-hover RPM (TODO.md §1.2):
  ``rpms = (1 + 0.5 * a) * hover_rpm`` with ``a ∈ [-1, 1]^4``.
- Termination priority (TODO.md §1.2):
  mission_reached > captured > crash > oob > timeout.
  Same-step ``mission_reached AND captured`` → attacker wins.
- Reward magnitudes / zero-sum contract live in subclass — this base only
  computes terminal flags + houses the dispatching machinery.

Caller contract (CLAUDE.md §10):
    Run ``gs.init(backend=..., seed=..., logging_level=...)`` BEFORE
    constructing this env. Seeding torch / numpy / random is the script's
    responsibility — this base does not re-seed.
"""

from __future__ import annotations

import abc
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs

from utils.cfg_dict import CfgDict
from utils.drone_params import DroneParams
from utils.rate_controller import BodyRateController


# ---------------------------------------------------------------------------
# Scripted-policy interface (used by env when one side is non-learning)
# ---------------------------------------------------------------------------

class ScriptedPolicy(Protocol):
    """Either side's scripted controller (see envs/pe_1v1/scripted_*.py).

    Implementations are pure-tensor, batched on ``num_envs``.  No for-loop
    over envs (CLAUDE.md §14).
    """

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Per-env reset hook (e.g., random-walk RNG state)."""

    def step(
        self,
        ego_state: dict[str, torch.Tensor],
        opponent_state: dict[str, torch.Tensor],
        g_mission: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return action shape (B, 4) in [-1, 1]; env converts to RPM."""


# ---------------------------------------------------------------------------
# Quaternion helpers — Genesis convention (w, x, y, z) per V1 sanity.
# ---------------------------------------------------------------------------

def quat_to_rotmat_flat(q: torch.Tensor) -> torch.Tensor:
    """quat (w,x,y,z) shape (B, 4) → flattened rotation matrix (B, 9)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )


def rotate_to_body_frame(
    vec_world: torch.Tensor,    # shape: (B, 3) world frame vector
    quat: torch.Tensor,         # shape: (B, 4) ego quaternion (w, x, y, z)
) -> torch.Tensor:
    """World frame vector → ego body frame: ``R^T · vec``. shape: (B, 3).

    PR-M (2026-04-29): obs schema 변경 — 상대 drone 위치/속도와 g_mission을
    ego body frame 으로 변환해 yaw rotation invariance + sim-to-real friendly.
    """
    R = quat_to_rotmat_flat(quat).view(-1, 3, 3)            # (B, 3, 3) body→world
    return torch.bmm(R.transpose(1, 2), vec_world.unsqueeze(-1)).squeeze(-1)


def quat_to_euler_deg(q: torch.Tensor) -> torch.Tensor:
    """quat (w,x,y,z) → (roll, pitch, yaw) deg, shape (B, 3). ZYX intrinsic."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1) * (180.0 / math.pi)


# ---------------------------------------------------------------------------
# BasePursuitEvasionEnv
# ---------------------------------------------------------------------------

class BasePursuitEvasionEnv(abc.ABC):
    """Abstract base for 1v1 PE envs (Phase 1).

    Subclasses fill in spawn / observation / reward bodies via the abstract
    hooks at the bottom of this class.
    """

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
        self.num_envs = int(num_envs)
        self.params = params
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        # rsl-rl 5.2 reads cfg directly from the env. Wandb logger calls
        # ``env.cfg.to_dict()`` — wrap in CfgDict so plain-dict cfg satisfies
        # both wandb and tensorboard paths.
        self.cfg: dict[str, Any] = CfgDict(env_cfg)

        self.ego = env_cfg["ego"]
        if self.ego not in ("attacker", "defender"):
            raise ValueError(
                f"env_cfg['ego'] must be 'attacker' or 'defender', got {self.ego!r}"
            )
        if self.ego == "attacker" and defender_policy is None:
            raise ValueError("defender_policy is required when ego == 'attacker'")
        if self.ego == "defender" and attacker_policy is None:
            raise ValueError("attacker_policy is required when ego == 'defender'")

        self._attacker_policy = attacker_policy
        self._defender_policy = defender_policy

        self.dt = float(env_cfg["dt"])
        self.substeps = int(env_cfg["substeps"])
        self.episode_length_s = float(env_cfg["episode_length_s"])
        self.max_episode_length = max(1, math.ceil(self.episode_length_s / self.dt))
        self.num_actions = int(env_cfg["num_actions"])
        self.clip_actions = float(env_cfg["clip_actions"])

        # Action mode: "rpm" (legacy per-prop scaled-hover) or "rate"
        # ([T_norm, wx_ref, wy_ref, wz_ref] body-rate PD inner loop).
        # Defaulting to "rpm" preserves existing test fixtures; flip via yaml
        # ``action.action_mode: rate`` once PD gains are tuned (plan §15).
        self.action_mode = str(env_cfg.get("action_mode", "rpm"))
        if self.action_mode not in ("rpm", "rate"):
            raise ValueError(
                f"action_mode must be 'rpm' or 'rate', got {self.action_mode!r}"
            )

        # Phase F (2026-05-10) — game mode: "mission" (기존) | "pure_pe" (AMSPB-style).
        # Reward / terminal 함수 분기 (env.py에서 self.game_mode 사용).
        self.game_mode = str(env_cfg.get("game_mode", "mission"))
        if self.game_mode not in ("mission", "pure_pe"):
            raise ValueError(f"game_mode must be 'mission' or 'pure_pe', got {self.game_mode!r}")

        # rsl-rl 5.2 obs_groups. Phase E.1 (CLAUDE.md §7) — env_cfg에 use_asymmetric_critic
        # 활성 시 critic은 policy + critic_priv 사용. 비활성이면 v16 그대로 (backward compat).
        self.use_asymmetric_critic = bool(env_cfg.get("use_asymmetric_critic", False))
        if self.use_asymmetric_critic:
            self.obs_groups = {"actor": ["policy"], "critic": ["policy", "critic_priv"]}
        else:
            self.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        self.cfg["obs_groups"] = self.obs_groups

        self.device = gs.device
        self.show_viewer = bool(show_viewer)

        self._build_scene(show_viewer)
        self.scene.build(n_envs=self.num_envs)

        self._assert_plan_a()
        self._init_common_buffers()
        self._build_rate_controllers(env_cfg)
        self._init_pe_buffers()

        self.reset()

    # ------------------------------------------------------------------
    # Rate controllers (one per drone). Constructed once; only used when
    # ``action_mode == "rate"`` — kept ``None`` otherwise.
    # ------------------------------------------------------------------

    def _build_rate_controllers(self, env_cfg: dict[str, Any]) -> None:
        if self.action_mode != "rate":
            self._attacker_rate_ctrl: BodyRateController | None = None
            self._defender_rate_ctrl: BodyRateController | None = None
            return
        rcfg = env_cfg.get("rate_controller", {}) or {}
        kp = tuple(float(x) for x in rcfg.get("kp_rate", (0.05, 0.05, 0.05)))
        kd = tuple(float(x) for x in rcfg.get("kd_rate", (0.001, 0.001, 0.001)))
        max_rate = float(rcfg.get("max_body_rate", math.pi))
        # PR-N (Phase C) — opt-in PX4-style wrench saturation. default off →
        # Phase 1.0/1.5/A/B yaml 무영향. Phase C yaml은 명시 true.
        enable_sat = bool(rcfg.get("enable_wrench_saturation", False))
        if len(kp) != 3 or len(kd) != 3:
            raise ValueError(
                f"rate_controller.kp_rate / kd_rate must be length-3, got "
                f"kp={kp!r}, kd={kd!r}"
            )
        self._attacker_rate_ctrl = BodyRateController(
            num_envs=self.num_envs, dt=self.dt, params=self.params,
            device=self.device, kp_rate=kp, kd_rate=kd, max_body_rate=max_rate,
            enable_wrench_saturation=enable_sat,
        )
        self._defender_rate_ctrl = BodyRateController(
            num_envs=self.num_envs, dt=self.dt, params=self.params,
            device=self.device, kp_rate=kp, kd_rate=kd, max_body_rate=max_rate,
            enable_wrench_saturation=enable_sat,
        )

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self, show_viewer: bool) -> None:
        n_rendered = max(1, min(4, self.num_envs))
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(self.env_cfg.get("max_visualize_fps", 60)),
                camera_pos=(3.5, -1.5, 2.5),
                camera_lookat=(0.5, 0.0, 1.0),
                camera_fov=45,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(n_rendered))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                # collision=True로 복귀 — False는 멀티링크 드론의 self-collision까지
                # 꺼서 propeller link들이 분리·발산 (random init 즉시 NaN). 두 드론
                # contact 안정화는 substeps↑로 처리 (cfg sim.substeps).
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())

        urdf_path = self.params.urdf_path
        if not Path(urdf_path).is_absolute():
            urdf_path = str(Path(__file__).resolve().parents[1] / urdf_path)

        prop_links = list(self.params.prop_link_names)
        prop_spins = [int(s) for s in self.params.spin_directions]

        self.attacker_drone = self.scene.add_entity(
            gs.morphs.Drone(file=urdf_path, propellers_link_name=prop_links, propellers_spin=prop_spins)
        )
        self.defender_drone = self.scene.add_entity(
            gs.morphs.Drone(file=urdf_path, propellers_link_name=prop_links, propellers_spin=prop_spins)
        )

        # show_viewer 마커 — opponent (학습 중인 ego의 상대편) 빨간 투명구,
        # g_mission 초록 투명구. collision=False 라 물리에 영향 없음.
        # ego="attacker" → opponent = defender, ego="defender" → opponent = attacker.
        self.opponent_marker = None
        self.goal_marker = None
        self.centerline_marker = None
        if show_viewer:
            opp_radius = float(self.env_cfg.get("d_capture", 0.3))
            goal_radius = float(self.env_cfg.get("g_mission_radius", 0.3))
            self.opponent_marker = self.scene.add_entity(
                morph=gs.morphs.Sphere(radius=opp_radius, collision=False),
                surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0, 0.35)),
            )
            self.goal_marker = self.scene.add_entity(
                morph=gs.morphs.Sphere(radius=goal_radius, collision=False),
                surface=gs.surfaces.Rough(color=(0.0, 1.0, 0.0, 0.35)),
            )
            # Phase G — x=0 centerline (attacker spawn 우측 / defender 좌측 분리 표시)
            # 길이 10m (y=-5~5), width/height 0.3m, ground 근처 z=0.15. collision=False.
            self.centerline_marker = self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=(0.3, 10.0, 0.3),
                    pos=(0.0, 0.0, 0.15),
                    collision=False,
                    fixed=True,                  # gravity 영향 차단 (정적 marker)
                ),
                surface=gs.surfaces.Rough(color=(1.0, 0.0, 0.0, 1.0)),
            )

        n_props = float(self.params.n_propellers)
        hover = float(np.sqrt(self.params.mass * self.params.gravity / (n_props * self.params.kf)))
        self.hover_rpm_scalar = hover

    def _assert_plan_a(self) -> None:
        """V1 / V1.5 inline guards: kf convention + quat order."""
        if not (1000.0 <= self.hover_rpm_scalar <= 50000.0):
            raise RuntimeError(
                f"hover_rpm = {self.hover_rpm_scalar:.1f} outside [1000, 50000]. "
                "kf likely uses rad/s² convention (V1 sanity precondition failed)."
            )
        q0 = self.attacker_drone.get_quat()
        q_first_abs = float(q0[0, 0].abs())
        q_last_abs = float(q0[0, 3].abs())
        if q_first_abs < 0.99 or q_last_abs > 0.1:
            raise RuntimeError(
                f"Genesis quat is not (w,x,y,z) at identity: q[0]={q0[0].tolist()}. "
                "V1 sanity quat guard failed."
            )

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _init_common_buffers(self) -> None:
        B = self.num_envs
        f = gs.tc_float
        i = gs.tc_int
        d = self.device

        self.actions = torch.zeros((B, self.num_actions), device=d, dtype=f)        # shape: (B, 4)
        self.last_attacker_action = torch.zeros((B, self.num_actions), device=d, dtype=f)
        self.last_defender_action = torch.zeros((B, self.num_actions), device=d, dtype=f)

        # attacker state
        self.attacker_pos = torch.zeros((B, 3), device=d, dtype=f)                  # shape: (B, 3) ENU m
        self.attacker_vel = torch.zeros((B, 3), device=d, dtype=f)                  # shape: (B, 3) m/s world
        self.attacker_quat = torch.zeros((B, 4), device=d, dtype=f)                 # shape: (B, 4) (w,x,y,z)
        self.attacker_quat[:, 0] = 1.0
        self.attacker_ang_vel = torch.zeros((B, 3), device=d, dtype=f)              # shape: (B, 3) rad/s

        # defender state (mirror of attacker)
        self.defender_pos = torch.zeros((B, 3), device=d, dtype=f)
        self.defender_vel = torch.zeros((B, 3), device=d, dtype=f)
        self.defender_quat = torch.zeros((B, 4), device=d, dtype=f)
        self.defender_quat[:, 0] = 1.0
        self.defender_ang_vel = torch.zeros((B, 3), device=d, dtype=f)

        # Phase 3a (TODO §3a, 2026-05-12) — τ-delay ring buffer for actor obs.
        # Stores last (τ+1) steps of defender kinematics (pos+vel, world frame).
        # Body-frame rotation is performed at obs construction with CURRENT
        # attacker pose, so the buffer holds raw world-frame state. Layout
        # along dim 0: [oldest, ..., newest]; actor obs reads index 0 = state
        # τ steps ago. τ=0 → buf_size=1, history[0] = current (backward compat).
        self.tau_delay = int(self.env_cfg.get("tau_delay", 0))
        if self.tau_delay < 0:
            raise ValueError(f"tau_delay must be >= 0, got {self.tau_delay}")
        buf_size = max(1, self.tau_delay + 1)
        self.defender_state_history = torch.zeros((buf_size, B, 6), device=d, dtype=f)

        # context
        self.g_mission = torch.zeros((B, 3), device=d, dtype=f)                     # shape: (B, 3)

        self.episode_length_buf = torch.zeros((B,), device=d, dtype=i)
        self.reset_buf = torch.ones((B,), device=d, dtype=torch.bool)
        self.rew_buf = torch.zeros((B,), device=d, dtype=f)

        # termination flags (priority-resolved after step).
        # crash/oob는 per-side: 자기 추락(self) vs 상대 추락(opp) 보상 신호 분리.
        # `last_term`/`extras["episode"]`엔 기존 합산 ``crash``/``oob`` 키도 함께
        # 노출해서 eval.py / 기존 sanity 테스트의 backward-compat 유지.
        self.mission_reached_buf = torch.zeros((B,), device=d, dtype=torch.bool)
        self.captured_buf = torch.zeros((B,), device=d, dtype=torch.bool)
        self.attacker_crash_buf = torch.zeros((B,), device=d, dtype=torch.bool)
        self.defender_crash_buf = torch.zeros((B,), device=d, dtype=torch.bool)
        self.attacker_oob_buf = torch.zeros((B,), device=d, dtype=torch.bool)
        self.defender_oob_buf = torch.zeros((B,), device=d, dtype=torch.bool)
        self.timeout_buf = torch.zeros((B,), device=d, dtype=torch.bool)

        # progress potential (potential-based progress reward)
        self.dist_to_goal = torch.zeros((B,), device=d, dtype=f)
        self.last_dist_to_goal = torch.zeros((B,), device=d, dtype=f)

        self.extras: dict[str, Any] = {"episode": {}, "time_outs": torch.zeros((B,), device=d, dtype=f)}

    @abc.abstractmethod
    def _init_pe_buffers(self) -> None:
        """Subclass: spawn ranges resolved from cfg, etc."""

    # ------------------------------------------------------------------
    # rsl-rl 5.2 API
    # ------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        # Phase E.1 — asymmetric critic 활성 시 "critic_priv" key 추가 (CLAUDE.md §7).
        # actor는 obs_groups["actor"] 따라 "policy"만 사용; critic은 "policy"+"critic_priv".
        out = {"policy": self._build_policy_obs()}
        if self.use_asymmetric_critic:
            out["critic_priv"] = self._build_critic_priv_obs()
        return TensorDict(out, batch_size=[self.num_envs], device=self.device)

    def _build_policy_obs(self) -> torch.Tensor:
        if self.ego == "attacker":
            obs = self._attacker_obs()
        else:
            obs = self._defender_obs()
        return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _build_critic_priv_obs(self) -> torch.Tensor:
        """BPQL privileged critic obs (Phase 3d, TODO §3d) — 6D body frame.

        Opponent current kinematics (no delay) projected into ego body frame:
        rel_pos_GT(3) + rel_vel_GT(3) = 6D. Same representation as actor's
        s_d_kin slot — only difference is the τ-delay removal. Implementation
        lives in the subclass (``_critic_priv_obs``); base wraps in nan_to_num.
        """
        obs = self._critic_priv_obs()
        return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    @abc.abstractmethod
    def _attacker_obs(self) -> torch.Tensor:
        """Return shape (B, ATTACKER_OBS_DIM) actor input for attacker role."""

    @abc.abstractmethod
    def _defender_obs(self) -> torch.Tensor:
        """Return shape (B, DEFENDER_OBS_DIM) actor input for defender role."""

    def _critic_priv_obs(self) -> torch.Tensor:
        """Phase E.1 default — opponent ground truth (vel/ang_vel/quat).

        Subclass(env.py) overrides if needed. base 구현으로 일반 케이스 처리.
        """
        raise NotImplementedError(
            "use_asymmetric_critic=True requires subclass to implement _critic_priv_obs()"
        )

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(
        self, ego_actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        ego_actions = torch.nan_to_num(ego_actions, nan=0.0, posinf=1.0, neginf=-1.0)
        ego_actions = torch.clip(ego_actions, -self.clip_actions, self.clip_actions)
        self.actions[:] = ego_actions

        if self.ego == "attacker":
            attacker_action = ego_actions
            assert self._defender_policy is not None
            defender_action = self._defender_policy.step(
                ego_state=self._defender_state_dict(),
                opponent_state=self._attacker_state_dict(),
                g_mission=None,
            )
        else:
            defender_action = ego_actions
            assert self._attacker_policy is not None
            attacker_action = self._attacker_policy.step(
                ego_state=self._attacker_state_dict(),
                opponent_state=self._defender_state_dict(),
                g_mission=self.g_mission,
            )

        attacker_action = torch.clip(
            torch.nan_to_num(attacker_action, nan=0.0, posinf=1.0, neginf=-1.0),
            -self.clip_actions, self.clip_actions,
        )
        defender_action = torch.clip(
            torch.nan_to_num(defender_action, nan=0.0, posinf=1.0, neginf=-1.0),
            -self.clip_actions, self.clip_actions,
        )

        self.attacker_drone.set_propellers_rpm(self._attacker_action_to_rpm(attacker_action))
        self.defender_drone.set_propellers_rpm(self._defender_action_to_rpm(defender_action))
        self.scene.step()

        self.episode_length_buf += 1
        self._update_state()
        # Phase 3a — push fresh defender state into τ-delay ring buffer
        # AFTER _update_state (so we capture t+1 state) and BEFORE obs build.
        # For done envs, reset_idx() below refills the entire buffer with the
        # new spawn state — this push for those envs is benign overwritten work.
        self._push_defender_history()
        self._update_pe_state()

        # Raw termination flags (per-side for crash/oob).
        mission = self._compute_mission_reached()
        captured_raw = self._compute_capture()
        a_crash, d_crash = self._compute_crash()
        a_oob, d_oob = self._compute_oob()
        crash_raw = a_crash | d_crash
        oob_raw = a_oob | d_oob
        timeout_raw = self.episode_length_buf >= self.max_episode_length

        # Priority resolution: mission > captured > crash > oob > timeout.
        captured = captured_raw & ~mission
        crash_any = crash_raw & ~mission & ~captured_raw
        oob_any = oob_raw & ~mission & ~captured_raw & ~crash_raw
        timeout = timeout_raw & ~mission & ~captured_raw & ~crash_raw & ~oob_raw

        # Per-side flags share the same priority gating: a side flag fires only
        # when the env reaches the crash/oob bucket (i.e. mission/captured did
        # not win). Within the crash bucket either or both sides can be set;
        # within oob only sides that did NOT crash this step are set.
        mask_alive = ~mission & ~captured_raw
        mask_no_crash = mask_alive & ~crash_raw
        self.attacker_crash_buf[:] = a_crash & mask_alive
        self.defender_crash_buf[:] = d_crash & mask_alive
        self.attacker_oob_buf[:] = a_oob & mask_no_crash
        self.defender_oob_buf[:] = d_oob & mask_no_crash
        # Subclass-set per-condition crash flags (pe_1v1 stores roll_bad /
        # pitch_bad / speed_high / z_below per side). Gate by the per-side
        # crash_buf so counts reflect actual TERMINATING crash events, not
        # per-step threshold violations during normal play. Required after
        # crash_on_attitude/crash_on_speed=False introduced — without this
        # gate, roll_bad fires every step a drone is past pad even when env
        # never terminates, blowing up sub-condition counts beyond n_episodes.
        if hasattr(self, "crash_subs"):
            for k in list(self.crash_subs):
                if k.startswith("att_"):
                    self.crash_subs[k] = self.crash_subs[k] & self.attacker_crash_buf
                elif k.startswith("def_"):
                    self.crash_subs[k] = self.crash_subs[k] & self.defender_crash_buf
                else:
                    # Unknown prefix: fall back to mask_alive (legacy contract).
                    self.crash_subs[k] = self.crash_subs[k] & mask_alive

        self.mission_reached_buf[:] = mission
        self.captured_buf[:] = captured
        self.timeout_buf[:] = timeout

        # Snapshot for diagnostic / external readers. reset_idx() below
        # wipes the per-env *_buf so callers that inspect the env AFTER
        # step() returns would otherwise see all-zero termination flags
        # on done envs. These references survive the wipe.
        # Aggregate ``crash``/``oob`` keys retained for eval.py / legacy
        # sanity tests; per-side keys expose minimax breakdown.
        self.last_term = {
            "mission":  mission,
            "captured": captured,
            "crash":    crash_any,
            "oob":      oob_any,
            "att_crash": self.attacker_crash_buf.clone(),
            "def_crash": self.defender_crash_buf.clone(),
            "att_oob":   self.attacker_oob_buf.clone(),
            "def_oob":   self.defender_oob_buf.clone(),
            "timeout":  timeout,
        }
        # 2026-05-04 — Per-condition crash flags survive reset wipe so h2h
        # eval / external readers can count {z, speed, roll, pitch} breakdown
        # alongside the aggregate att_crash/def_crash. Subclass-defined.
        if hasattr(self, "crash_subs"):
            for k, v in self.crash_subs.items():
                self.last_term[k] = v.clone()
        # Diagnostic snapshot of post-step pre-reset state. ``reset_idx`` below
        # respawns done envs, wiping pos/vel/quat. External readers wanting to
        # inspect the state that *triggered* the termination (e.g., defender
        # speed at crash) should read this dict, not env.{attacker,defender}_*.
        self.last_alive_state = {
            "attacker_pos":  self.attacker_pos.clone(),
            "attacker_vel":  self.attacker_vel.clone(),
            "attacker_quat": self.attacker_quat.clone(),
            "defender_pos":  self.defender_pos.clone(),
            "defender_vel":  self.defender_vel.clone(),
            "defender_quat": self.defender_quat.clone(),
        }

        rew_attacker = self._attacker_reward(attacker_action)
        rew_defender = self._defender_reward(defender_action)
        rew_attacker = torch.nan_to_num(rew_attacker, nan=0.0, posinf=1.0, neginf=-1.0)
        rew_defender = torch.nan_to_num(rew_defender, nan=0.0, posinf=1.0, neginf=-1.0)

        # Exposed for tests (terminal + dense zero-sum assertions). Not part
        # of the rsl-rl contract — the runner only consumes ``self.rew_buf``.
        self.rew_attacker_buf = rew_attacker
        self.rew_defender_buf = rew_defender

        self.rew_buf[:] = rew_attacker if self.ego == "attacker" else rew_defender

        done = mission | captured | crash_any | oob_any | timeout
        self.reset_buf[:] = done
        self.extras["time_outs"] = timeout.to(dtype=gs.tc_float)

        self.last_attacker_action[:] = attacker_action
        self.last_defender_action[:] = defender_action
        # Snapshot post-step distance for next step's potential-based progress.
        # ``reset_idx`` will overwrite this for any env about to be reset, so
        # the snapshot is correct for both surviving and re-spawning envs.
        self.last_dist_to_goal[:] = self.dist_to_goal
        # Phase F (pure_pe) — attacker-defender 거리도 동일하게 snapshot.
        if hasattr(self, "last_dist_to_attacker_defender"):
            self.last_dist_to_attacker_defender[:] = self.dist_to_attacker_defender

        self._update_episode_extras(rew_attacker, rew_defender)

        reset_envs = done.nonzero(as_tuple=False).reshape(-1)
        if reset_envs.numel() > 0:
            self.reset_idx(reset_envs)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def reset(self) -> TensorDict:
        envs_idx = torch.arange(self.num_envs, device=self.device)
        self.reset_buf[:] = True
        self.reset_idx(envs_idx)
        return self.get_observations()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if envs_idx.numel() == 0:
            return
        self._spawn(envs_idx)
        self._reset_pe(envs_idx)
        self.episode_length_buf[envs_idx] = 0
        self.last_attacker_action[envs_idx] = 0.0
        self.last_defender_action[envs_idx] = 0.0
        self.mission_reached_buf[envs_idx] = False
        self.captured_buf[envs_idx] = False
        self.attacker_crash_buf[envs_idx] = False
        self.defender_crash_buf[envs_idx] = False
        self.attacker_oob_buf[envs_idx] = False
        self.defender_oob_buf[envs_idx] = False
        self.timeout_buf[envs_idx] = False
        # Refresh state buffers from Genesis after the spawn writes; ensures
        # ``last_dist_to_goal`` snapshots the post-reset geometry.
        self._update_state()
        self._update_pe_state()
        self.last_dist_to_goal[envs_idx] = self.dist_to_goal[envs_idx]
        # Phase F (pure_pe) — atk-def 거리도 reset 시 snapshot.
        if hasattr(self, "last_dist_to_attacker_defender"):
            self.last_dist_to_attacker_defender[envs_idx] = self.dist_to_attacker_defender[envs_idx]
        # Phase 3a — warmup-fill the τ-delay ring buffer with the new spawn
        # state for reset envs, so history[0] = current (not zero garbage)
        # for the first τ steps of the next episode.
        # CRITICAL backward compat (2026-05-13): skip entirely when tau_delay
        # == 0. The cat + expand + indexed assignment adds GPU ops even when
        # the buffer is unused, perturbing cuDNN ordering and breaking seed
        # reproducibility for ams_v6 baseline. See
        # docs/notes/prompt_for_tau_delay_claude.md.
        if self.tau_delay > 0:
            n = int(envs_idx.numel())
            fresh = torch.cat(
                [self.defender_pos[envs_idx], self.defender_vel[envs_idx]], dim=1,
            )                                                   # shape: (n, 6)
            self.defender_state_history[:, envs_idx] = fresh.unsqueeze(0).expand(
                self.defender_state_history.shape[0], n, 6,
            )

        if self._attacker_policy is not None:
            self._attacker_policy.reset(envs_idx)
        if self._defender_policy is not None:
            self._defender_policy.reset(envs_idx)
        # Rate-controller per-env state (last_ang_vel) must clear too so the
        # PD derivative term doesn't bleed across episodes.
        if self._attacker_rate_ctrl is not None:
            self._attacker_rate_ctrl.reset(envs_idx)
        if self._defender_rate_ctrl is not None:
            self._defender_rate_ctrl.reset(envs_idx)

    @abc.abstractmethod
    def _spawn(self, envs_idx: torch.Tensor) -> None:
        """Sample attacker / defender pose, write via drone.set_pos / set_quat."""

    @abc.abstractmethod
    def _reset_pe(self, envs_idx: torch.Tensor) -> None:
        """Sample g_mission etc. for the given env slots."""

    # ------------------------------------------------------------------
    # State refresh
    # ------------------------------------------------------------------

    def _push_defender_history(self) -> None:
        """Roll the τ-delay ring buffer and write current defender state.

        CRITICAL backward compat (2026-05-13): when ``tau_delay == 0`` this is
        a no-op. Doing the roll+index even on a size-1 buffer added GPU ops
        that perturbed cuDNN nondeterministic ordering — same seed produced
        different cycle 6-8 trajectories (a_win 60-72% → 3-5% on ams_v6).
        See docs/notes/prompt_for_tau_delay_claude.md for the incident.

        Layout along dim 0: [oldest, ..., newest]. ``roll(-1, dims=0)`` wraps
        element 0 to the end; the next line overwrites that wrapped slot with
        the fresh state, so the buffer is always [oldest..., newest] after
        both lines. Actor obs reads ``[0]`` = state τ steps ago.
        """
        if self.tau_delay == 0:
            return
        self.defender_state_history = torch.roll(
            self.defender_state_history, shifts=-1, dims=0,
        )
        self.defender_state_history[-1, :, :3] = self.defender_pos
        self.defender_state_history[-1, :, 3:] = self.defender_vel

    def _update_state(self) -> None:
        self.attacker_pos[:] = self.attacker_drone.get_pos()
        self.attacker_vel[:] = self.attacker_drone.get_vel()
        self.attacker_quat[:] = self.attacker_drone.get_quat()
        
        world_ang_vel_att = self.attacker_drone.get_ang()
        self.attacker_ang_vel[:] = rotate_to_body_frame(world_ang_vel_att, self.attacker_quat)

        self.defender_pos[:] = self.defender_drone.get_pos()
        self.defender_vel[:] = self.defender_drone.get_vel()
        self.defender_quat[:] = self.defender_drone.get_quat()
        
        world_ang_vel_def = self.defender_drone.get_ang()
        self.defender_ang_vel[:] = rotate_to_body_frame(world_ang_vel_def, self.defender_quat)

        if self.show_viewer:
            self._update_markers()

    def _update_markers(self) -> None:
        """show_viewer 마커 위치 갱신. collision=False 마커이므로 물리 영향 없음."""
        if self.opponent_marker is not None:
            opp_pos = self.defender_pos if self.ego == "attacker" else self.attacker_pos
            self.opponent_marker.set_pos(opp_pos, zero_velocity=True)
        if self.goal_marker is not None:
            self.goal_marker.set_pos(self.g_mission, zero_velocity=True)

    @abc.abstractmethod
    def _update_pe_state(self) -> None:
        """Refresh dist-to-goal / dist-to-defender etc."""

    def _attacker_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "pos": self.attacker_pos,
            "vel": self.attacker_vel,
            "quat": self.attacker_quat,
            "ang_vel": self.attacker_ang_vel,
        }

    def _defender_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "pos": self.defender_pos,
            "vel": self.defender_vel,
            "quat": self.defender_quat,
            "ang_vel": self.defender_ang_vel,
        }

    # ------------------------------------------------------------------
    # Action conversion (TODO §1.2)
    # action_mode == "rpm"   : per-prop scaled-hover (legacy fallback).
    # action_mode == "rate"  : [T_norm, w*_ref] → BodyRateController → RPM.
    # ------------------------------------------------------------------

    def _action_to_rpm(self, action: torch.Tensor) -> torch.Tensor:
        """Legacy per-prop scaled-hover mapping. ``a ∈ [-1,1]^4`` →
        ``rpms = (1 + 0.5·a) · hover_rpm``. shape: (B, 4)."""
        return (1.0 + 0.5 * action) * self.hover_rpm_scalar

    def _attacker_action_to_rpm(self, action: torch.Tensor) -> torch.Tensor:
        if self._attacker_rate_ctrl is not None:
            return self._attacker_rate_ctrl.step(action, self.attacker_ang_vel)
        return self._action_to_rpm(action)

    def _defender_action_to_rpm(self, action: torch.Tensor) -> torch.Tensor:
        if self._defender_rate_ctrl is not None:
            return self._defender_rate_ctrl.step(action, self.defender_ang_vel)
        return self._action_to_rpm(action)

    # ------------------------------------------------------------------
    # Termination hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _compute_mission_reached(self) -> torch.Tensor:
        """shape: (B,) bool."""

    @abc.abstractmethod
    def _compute_capture(self) -> torch.Tensor:
        """shape: (B,) bool."""

    @abc.abstractmethod
    def _compute_crash(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(attacker_crash, defender_crash)``, each shape (B,) bool.

        z<floor / |roll/pitch|>limit / |v|>v_max evaluated per side.
        """

    @abc.abstractmethod
    def _compute_oob(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(attacker_oob, defender_oob)``, each shape (B,) bool."""

    # ------------------------------------------------------------------
    # Reward hooks (each must be priority-aware via the resolved buffers).
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _attacker_reward(self, attacker_action: torch.Tensor) -> torch.Tensor:
        """shape: (B,)."""

    @abc.abstractmethod
    def _defender_reward(self, defender_action: torch.Tensor) -> torch.Tensor:
        """shape: (B,)."""

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _update_episode_extras(
        self, rew_attacker: torch.Tensor, rew_defender: torch.Tensor
    ) -> None:
        # Priority-resolved buckets; mission/captured/timeout are mutually
        # exclusive across envs. crash/oob are the union of per-side flags
        # (att+def can both fire within the same env in the crash bucket),
        # so the aggregate rates use the per-side OR for the "any side
        # crashed" denominator.
        a_cr = self.attacker_crash_buf
        d_cr = self.defender_crash_buf
        a_ob = self.attacker_oob_buf
        d_ob = self.defender_oob_buf
        crash_any_buf = a_cr | d_cr
        oob_any_buf = a_ob | d_ob
        done = (
            self.mission_reached_buf | self.captured_buf
            | crash_any_buf | oob_any_buf | self.timeout_buf
        )
        n_done = float(done.sum().item())
        inv = (1.0 / n_done) if n_done > 0.0 else 0.0
        # Aggregate keys (eval.py / legacy sanity tests) + per-side
        # breakdown (minimax 신호 진단). Per-side counters are normalized by
        # the same n_done so they sum (att+def) ≈ aggregate within each row.
        self.extras["episode"] = {
            "attacker_win_rate": float(self.mission_reached_buf.sum().item()) * inv,
            "captured_rate":     float(self.captured_buf.sum().item()) * inv,
            "crash_rate":        float(crash_any_buf.sum().item()) * inv,
            "oob_rate":          float(oob_any_buf.sum().item()) * inv,
            "att_crash_rate":    float(a_cr.sum().item()) * inv,
            "def_crash_rate":    float(d_cr.sum().item()) * inv,
            "att_oob_rate":      float(a_ob.sum().item()) * inv,
            "def_oob_rate":      float(d_ob.sum().item()) * inv,
            "timeout_rate":      float(self.timeout_buf.sum().item()) * inv,
            "n_done_this_step":  n_done,
            "rew_attacker": float(rew_attacker.mean().item()),
            "rew_defender": float(rew_defender.mean().item()),
        }
        # PR-D3 — current dense coef (decay schedule monitoring). Subclass-defined.
        if hasattr(self, "chase_coef"):
            self.extras["episode"]["chase_coef"] = float(self.chase_coef)
        if hasattr(self, "goal_dense_coef"):
            self.extras["episode"]["goal_dense_coef"] = float(self.goal_dense_coef)
        # 2026-05-04 — per-condition crash breakdown (pe_1v1 sets crash_subs
        # in _compute_crash). 8 keys: att_crash_{z,speed,roll,pitch}_rate +
        # def_crash_{...}. Lets diagnostics during training identify which
        # trigger dominates without an offline inspect script.
        if hasattr(self, "crash_subs"):
            for k, buf in self.crash_subs.items():
                self.extras["episode"][f"{k}_rate"] = float(buf.sum().item()) * inv
