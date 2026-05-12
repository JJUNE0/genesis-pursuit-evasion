"""Scripted attacker policies for Phase 1.

PR-D (2026-04-26): Phase 1.0 default가 ``RandomWalkAttacker`` → ``StationaryAttacker``로
바뀜. 사용자 의도: defender의 첫 학습 task는 "랜덤 위치에 정지한 표적을 추적해
intercept하는 호버링 정책". random-walk는 chaotic torque 폭주로 학습 신호를 망가뜨림.

PR-N (2026-04-29): Phase 5 alternating co-training용 ``PretrainedAttacker`` 추가.
defender 학습 시 frozen attacker actor를 상대로 사용. obs는 env._attacker_obs와
parity (body frame, 31-dim).

Interface matches ``envs.base_pe_env.ScriptedPolicy``: env converts the
returned ``[-1, 1]^4`` action to per-prop scaled-hover RPM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

import genesis as gs

from envs.base_pe_env import quat_to_rotmat_flat, rotate_to_body_frame
from envs.pe_1v1.cfgs import ATTACKER_OBS_DIM, DEFAULT_OBS_SCALES


class StationaryAttacker:
    """Random spawn pose에 정지(hover) 유지.

    매 step ``action = zeros(B, 4)`` → ``RPM = hover_rpm`` → drone weight를
    완전 상쇄해 spawn 위치에서 정지. defender의 chase/intercept 학습용 정적
    표적.
    """

    def __init__(self, num_envs: int, num_actions: int = 4, device: Any = None):
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.device = device if device is not None else gs.device
        self._zero = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float32
        )

    def reset(self, envs_idx: torch.Tensor) -> None:
        del envs_idx

    def step(
        self,
        ego_state: dict[str, torch.Tensor],
        opponent_state: dict[str, torch.Tensor],
        g_mission: torch.Tensor | None,
    ) -> torch.Tensor:
        del ego_state, opponent_state, g_mission
        return self._zero


class RandomWalkAttacker:
    """Uniform-random per-step action. CLAUDE.md §14: no for-loop over envs.

    PR-D 이후로는 *옵션*으로만 보존 (chaotic baseline; 학습 친화적이지 않음).
    Phase 1.0 default는 StationaryAttacker.
    """

    def __init__(self, num_envs: int, num_actions: int = 4, device: Any = None):
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.device = device if device is not None else gs.device

    def reset(self, envs_idx: torch.Tensor) -> None:
        # Stateless — uniform sampling each step.
        del envs_idx

    def step(
        self,
        ego_state: dict[str, torch.Tensor],
        opponent_state: dict[str, torch.Tensor],
        g_mission: torch.Tensor | None,
    ) -> torch.Tensor:
        del ego_state, opponent_state, g_mission
        action = torch.rand(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float32
        ) * 2.0 - 1.0
        return action


# ---------------------------------------------------------------------------
# PretrainedAttacker — PR-N (Phase 5)
# ---------------------------------------------------------------------------

def _build_attacker_policy_obs(
    ego_state: dict[str, torch.Tensor],
    opponent_state: dict[str, torch.Tensor],
    g_mission: torch.Tensor,
    last_action: torch.Tensor,
    pos_scale: float,
    vel_scale: float,
    ang_vel_scale: float,
) -> torch.Tensor:
    """Mirror ``PursuitEvasion1v1Env._attacker_obs`` (PR-M body frame, PR-N parity).

    Layout (31-dim):
        ``[pos_a(world, scaled), vel_a(body), rot_a_flat, ang_vel_a(body),
           rel_pos_d(body), rel_vel_d(body), rel_g(body), last_action]``

    Self pos = world frame. Self vel + ang_vel = ego body frame.
    상대 drone + g_mission = ego body frame relative.
    """
    quat = ego_state["quat"]
    pos_a = ego_state["pos"] * pos_scale                                  # shape: (B, 3) world
    vel_a = rotate_to_body_frame(ego_state["vel"], quat) * vel_scale      # shape: (B, 3) body
    rot_a = quat_to_rotmat_flat(quat)                                     # shape: (B, 9)
    ang_a = ego_state["ang_vel"] * ang_vel_scale                          # shape: (B, 3) body
    rel_pos_d = rotate_to_body_frame(
        opponent_state["pos"] - ego_state["pos"], quat,
    ) * pos_scale                                                         # shape: (B, 3) body
    rel_vel_d = rotate_to_body_frame(
        opponent_state["vel"] - ego_state["vel"], quat,
    ) * vel_scale                                                         # shape: (B, 3) body
    rel_g = rotate_to_body_frame(
        g_mission - ego_state["pos"], quat,
    ) * pos_scale                                                         # shape: (B, 3) body

    s_a = torch.cat([pos_a, vel_a, rot_a, ang_a], dim=1)                  # shape: (B, 18)
    s_d_kin = torch.cat([rel_pos_d, rel_vel_d], dim=1)                    # shape: (B, 6)
    obs = torch.cat([s_a, s_d_kin, rel_g, last_action], dim=1)            # shape: (B, 31)
    return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)


class PretrainedAttacker:
    """Load Phase 1.5 산출물 ``attacker_v*.pt`` and run frozen actor.

    Mirror of ``PretrainedDefender`` for the attacker side. The actor expects
    the attacker obs layout ``[s_a, s_d_kin, rel_g, last_action]`` (31-dim,
    PR-M body frame). Construction loads weights and sets
    ``requires_grad=False`` + ``eval()``.

    ``ckpt_path`` accepts either:
      - a specific ``attacker_v*.pt`` file, OR
      - a directory containing ``attacker_v*.pt`` (latest by mtime is picked,
        useful for AMS-DRL stage progression v0 → v1 → v2 ...).

    The attacker obs **requires** ``g_mission`` (rel_g 컴포넌트). ``step()``
    raises ``ValueError`` when called with ``g_mission=None``.
    """

    def __init__(
        self,
        num_envs: int,
        ckpt_path: str | Path,
        num_actions: int = 4,
        device: Any = None,
        obs_scales: dict[str, float] | None = None,
        deterministic: bool = True,
    ):
        from agents.ppo_runner import resolve_actor_ckpt

        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.device = device if device is not None else gs.device
        self.deterministic = bool(deterministic)
        scales = dict(DEFAULT_OBS_SCALES)
        if obs_scales:
            scales.update({k: float(v) for k, v in obs_scales.items()})
        self.pos_scale = float(scales["pos"])
        self.vel_scale = float(scales["vel"])
        self.ang_vel_scale = float(scales["ang_vel"])

        requested = Path(ckpt_path)
        self.ckpt_path = resolve_actor_ckpt(ckpt_path, prefix="attacker")
        try:
            mtime = self.ckpt_path.stat().st_mtime
            from datetime import datetime as _dt
            mtime_str = _dt.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            mtime_str = "?"
        print(
            f"[PretrainedAttacker] requested: {requested}\n"
            f"[PretrainedAttacker] resolved → {self.ckpt_path}  (mtime {mtime_str})"
        )

        try:
            import rsl_rl  # noqa: F401  — needed for unpickling the actor module
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                "PretrainedAttacker requires rsl-rl to be importable for "
                f"unpickling the actor module saved at {self.ckpt_path}."
            ) from e

        actor = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)

        actor_dim = int(getattr(actor, "obs_dim", -1))
        if actor_dim != -1 and actor_dim != ATTACKER_OBS_DIM:
            raise ValueError(
                f"PretrainedAttacker ckpt actor.obs_dim={actor_dim} does not match "
                f"ATTACKER_OBS_DIM={ATTACKER_OBS_DIM}. ckpt is incompatible — "
                "regenerate via scripts/train_attacker.py with the current obs layout."
            )
        self.actor = actor
        print(
            f"[PretrainedAttacker] actor obs_dim={actor_dim} num_actions={self.num_actions}  "
            f"frozen, eval()=on"
        )
        self.last_action = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float32
        )

    def reset(self, envs_idx: torch.Tensor) -> None:
        if envs_idx is None or envs_idx.numel() == 0:
            return
        self.last_action[envs_idx] = 0.0

    @torch.no_grad()
    def step(
        self,
        ego_state: dict[str, torch.Tensor],
        opponent_state: dict[str, torch.Tensor],
        g_mission: torch.Tensor | None,
    ) -> torch.Tensor:
        if g_mission is None:
            raise ValueError(
                "PretrainedAttacker.step() requires g_mission — attacker obs "
                "includes rel_g (mission relative). env.py must pass self.g_mission."
            )
        obs = _build_attacker_policy_obs(
            ego_state, opponent_state, g_mission, self.last_action,
            self.pos_scale, self.vel_scale, self.ang_vel_scale,
        )
        td = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        mean = self.actor(td)                                            # (B, 4) raw mean
        if self.deterministic:
            action = mean
        else:
            std = self.actor.distribution.std                            # may be (B_train, 4)
            if std.dim() == 2:
                std = std[0]                                              # (4,)
            action = mean + std * torch.randn_like(mean)                  # broadcast (4,)→(B,4)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self.last_action[:] = action
        return action


# ---------------------------------------------------------------------------
# PretrainedAttackerPool — Fictitious Play 다중 ckpt mixing (Phase D, 2026-05-07)
# ---------------------------------------------------------------------------

class PretrainedAttackerPool:
    """Multi-ckpt attacker pool for Fictitious Play (FP).

    Mirror of ``PretrainedDefenderPool`` for the attacker side. Used by
    ``train_defender.py --attacker_ckpt_pool ...`` for FP-based AMS-DRL
    co-training stage 4 (Phase D). Drop-in compatible with
    ``PretrainedAttacker`` (same step / reset interface).

    Attacker obs requires ``g_mission`` (rel_g component); ``step()`` raises
    if g_mission is None. Deterministic only (mean of policy distribution).
    """

    def __init__(
        self,
        num_envs: int,
        ckpt_paths: list[str | Path],
        num_actions: int = 4,
        device: Any = None,
        obs_scales: dict[str, float] | None = None,
        sample: str = "uniform",
        seed: int = 0,
        current_best_ckpt: str | Path | None = None,   # Phase F (2026-05-10)
        pold: float = 1.0,
    ):
        from agents.ppo_runner import resolve_actor_ckpt

        if not ckpt_paths and current_best_ckpt is None:
            raise ValueError(
                "PretrainedAttackerPool requires at least 1 ckpt path or current_best_ckpt."
            )
        if sample != "uniform":
            raise ValueError(f"sample={sample!r} not supported (only 'uniform' in v1)")
        if not (0.0 <= pold <= 1.0):
            raise ValueError(f"pold must be in [0, 1], got {pold}")

        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.device = device if device is not None else gs.device
        self.sample = sample
        scales = dict(DEFAULT_OBS_SCALES)
        if obs_scales:
            scales.update({k: float(v) for k, v in obs_scales.items()})
        self.pos_scale = float(scales["pos"])
        self.vel_scale = float(scales["vel"])
        self.ang_vel_scale = float(scales["ang_vel"])

        try:
            import rsl_rl  # noqa: F401
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                "PretrainedAttackerPool requires rsl-rl to be importable."
            ) from e

        resolved: list[Path] = []
        actors: list[Any] = []
        for cp in ckpt_paths:
            r = resolve_actor_ckpt(cp, prefix="attacker")
            actor = torch.load(r, map_location=self.device, weights_only=False)
            actor.eval()
            for p in actor.parameters():
                p.requires_grad_(False)
            actor_dim = int(getattr(actor, "obs_dim", -1))
            if actor_dim != -1 and actor_dim != ATTACKER_OBS_DIM:
                raise ValueError(
                    f"PretrainedAttackerPool ckpt {r} actor.obs_dim={actor_dim} "
                    f"does not match ATTACKER_OBS_DIM={ATTACKER_OBS_DIM}."
                )
            resolved.append(r)
            actors.append(actor)
        self.ckpt_paths: list[Path] = resolved
        self.actors: list[Any] = actors
        K = len(self.actors)
        # Phase F — current_best (별도 ckpt for pold mixing)
        self.current_best_actor: Any | None = None
        if current_best_ckpt is not None:
            cb_path = resolve_actor_ckpt(current_best_ckpt, prefix="attacker")
            cb_actor = torch.load(cb_path, map_location=self.device, weights_only=False)
            cb_actor.eval()
            for p in cb_actor.parameters():
                p.requires_grad_(False)
            self.current_best_actor = cb_actor
            self.current_best_path = cb_path
        self.pold = float(pold)
        eff_pold = self.pold if (K > 0 and self.current_best_actor is not None) else (1.0 if K > 0 else 0.0)
        print(
            f"[PretrainedAttackerPool] loaded K={K} pool ckpts + current_best={'yes' if self.current_best_actor else 'no'} "
            f"(pold={self.pold} → effective={eff_pold:.2f}, num_envs={self.num_envs})"
        )
        for i, p in enumerate(self.ckpt_paths):
            print(f"  [{i}] {p}")
        if self.current_best_actor is not None:
            print(f"  [current_best] {self.current_best_path}")

        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))
        self.env_to_ckpt = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._reassign_all()

        self.last_action = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float32
        )

    def _reassign_all(self) -> None:
        all_idx = torch.arange(self.num_envs, device=self.device)
        self._reassign(all_idx)

    def _reassign(self, envs_idx: torch.Tensor) -> None:
        K = len(self.actors)
        n = envs_idx.numel()
        if K == 0:
            self.env_to_ckpt[envs_idx] = -1
            return
        if self.current_best_actor is None:
            self.env_to_ckpt[envs_idx] = torch.randint(
                0, K, (n,), device=self.device, generator=self._gen,
            )
            return
        rand_uniform = torch.rand(n, device=self.device, generator=self._gen)
        use_pool = rand_uniform < self.pold
        pool_idx = torch.randint(0, K, (n,), device=self.device, generator=self._gen)
        new_assignment = torch.where(use_pool, pool_idx, torch.full_like(pool_idx, -1))
        self.env_to_ckpt[envs_idx] = new_assignment

    def reset(self, envs_idx: torch.Tensor) -> None:
        if envs_idx is None or envs_idx.numel() == 0:
            return
        self.last_action[envs_idx] = 0.0
        self._reassign(envs_idx)

    @torch.no_grad()
    def step(
        self,
        ego_state: dict[str, torch.Tensor],
        opponent_state: dict[str, torch.Tensor],
        g_mission: torch.Tensor | None,
    ) -> torch.Tensor:
        if g_mission is None:
            raise ValueError(
                "PretrainedAttackerPool.step() requires g_mission — attacker obs "
                "includes rel_g (mission relative)."
            )
        obs = _build_attacker_policy_obs(
            ego_state, opponent_state, g_mission, self.last_action,
            self.pos_scale, self.vel_scale, self.ang_vel_scale,
        )
        action = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float32
        )
        for k, actor in enumerate(self.actors):
            mask = self.env_to_ckpt == k
            n_k = int(mask.sum().item())
            if n_k == 0:
                continue
            sub = obs[mask]
            td = TensorDict({"policy": sub}, batch_size=[n_k], device=self.device)
            action[mask] = actor(td)   # deterministic (mean) only
        # Phase F — current_best path (env_to_ckpt == -1)
        if self.current_best_actor is not None:
            mask = self.env_to_ckpt == -1
            n_cb = int(mask.sum().item())
            if n_cb > 0:
                sub = obs[mask]
                td = TensorDict({"policy": sub}, batch_size=[n_cb], device=self.device)
                action[mask] = self.current_best_actor(td)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self.last_action[:] = action
        return action
