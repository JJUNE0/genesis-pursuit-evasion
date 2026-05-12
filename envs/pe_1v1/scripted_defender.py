"""Scripted / loaded defender policies for Phase 1.

Three options selected via cfg (TODO.md §1.3):

- ``StationaryDefender``  — action = 0 → hover_rpm.
- ``RandomWalkDefender``  — uniform U(-1, 1)^4 each step.
- ``PretrainedDefender``  — Phase 1.0 산출물 ``actor_v0.pt`` (frozen rsl-rl
  actor module pickled by ``scripts/train_defender.py``) 로드. Actor의
  defender obs layout 은 ``[s_d, s_a_kin]`` 24-dim (g_mission 미노출,
  논문 §3.6.3 K6).

All three share the ``envs.base_pe_env.ScriptedPolicy`` interface — return
shape ``(B, 4)`` action in [-1, 1]; the env handles RPM conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

import genesis as gs

from envs.base_pe_env import quat_to_rotmat_flat, rotate_to_body_frame
from envs.pe_1v1.cfgs import DEFAULT_OBS_SCALES, DEFENDER_OBS_DIM


# ---------------------------------------------------------------------------
# StationaryDefender
# ---------------------------------------------------------------------------

class StationaryDefender:
    """Hover in place: action=0 every step → RPM = hover_rpm."""

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


# ---------------------------------------------------------------------------
# RandomWalkDefender
# ---------------------------------------------------------------------------

class RandomWalkDefender:
    """Uniform-random per-step action ∈ [-1, 1]^4."""

    def __init__(self, num_envs: int, num_actions: int = 4, device: Any = None):
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.device = device if device is not None else gs.device

    def reset(self, envs_idx: torch.Tensor) -> None:
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
# ChaseDefender (Phase F.1.5, 2026-05-11) — AMSPB Alg.1 line 1 analog.
# ---------------------------------------------------------------------------

class ChaseDefender:
    """Simple PD chase via tilt-acceleration.

    Action layout (rate controller mode, ``action_mode="rate"``):
        ``[T_norm, ωx_ref, ωy_ref, ωz_ref]`` ∈ [-1, 1]^4
        ωx = roll-rate, ωy = pitch-rate, ωz = yaw-rate (body frame).
    Strategy:
        1. err_world = attacker_pos - defender_pos.
        2. Rotate to defender body frame → (ex, ey, ez_body).
        3. Forward (x_body): pitch nose down → ωy_ref = -kp · ex.
        4. Lateral (y_body): roll left → ωx_ref = +kr · ey.
        5. Vertical (z_world): thrust adjust → T_norm = +kt · ez_world.
    Used by ``train_attacker.py --defender chase`` for AMSPB-style attacker
    pretraining (Phase F.1.5, 사용자 critique on F.1c 진동).
    """

    def __init__(
        self,
        num_envs: int,
        num_actions: int = 4,
        device: Any = None,
        gain_pitch: float = 0.5,
        gain_roll: float = 0.5,
        gain_thrust: float = 0.3,
    ):
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.device = device if device is not None else gs.device
        self.gain_pitch = float(gain_pitch)
        self.gain_roll = float(gain_roll)
        self.gain_thrust = float(gain_thrust)

    def reset(self, envs_idx: torch.Tensor) -> None:
        del envs_idx

    def step(
        self,
        ego_state: dict[str, torch.Tensor],
        opponent_state: dict[str, torch.Tensor],
        g_mission: torch.Tensor | None,
    ) -> torch.Tensor:
        del g_mission
        err_world = opponent_state["pos"] - ego_state["pos"]                  # shape: (B, 3)
        err_body = rotate_to_body_frame(err_world, ego_state["quat"])         # shape: (B, 3)
        ex_body = err_body[:, 0]                                              # forward
        ey_body = err_body[:, 1]                                              # left
        ez_world = err_world[:, 2]                                            # up (thrust frame)
        omega_y = torch.clamp(-self.gain_pitch * ex_body, -1.0, 1.0)          # nose down for forward accel
        omega_x = torch.clamp(self.gain_roll * ey_body, -1.0, 1.0)            # roll for lateral accel
        omega_z = torch.zeros_like(omega_x)                                   # yaw unused
        thrust = torch.clamp(self.gain_thrust * ez_world, -1.0, 1.0)          # T_norm hover-relative
        action = torch.stack([thrust, omega_x, omega_y, omega_z], dim=1)      # shape: (B, 4)
        return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)


# ---------------------------------------------------------------------------
# PretrainedDefender — activated in PR-C (Phase 1.5)
# ---------------------------------------------------------------------------

def _build_defender_policy_obs(
    ego_state: dict[str, torch.Tensor],
    opponent_state: dict[str, torch.Tensor],
    last_action: torch.Tensor,
    pos_scale: float,
    vel_scale: float,
    ang_vel_scale: float,
) -> torch.Tensor:
    """Mirror ``PursuitEvasion1v1Env._defender_obs`` (PR-M body frame, PR-N parity).

    Layout (28-dim):
        ``[pos_d(world, scaled), vel_d(body), rot_d_flat, ang_vel_d(body),
           rel_pos_a(body), rel_vel_a(body), last_action]``

    Self pos = world frame (env spatial reward 계산과 일치).
    Self vel + ang_vel = ego body frame (action도 body frame이라 representation 일관성).
    상대 drone (attacker) pos/vel = ego body frame relative
    (yaw rotation invariance, sim-to-real friendly). g_mission은 의도적으로 부재
    (논문 §3.6.3 K6 — defender obs는 mission 정보 없음).
    """
    quat = ego_state["quat"]
    pos_d = ego_state["pos"] * pos_scale                                  # shape: (B, 3) world
    vel_d = rotate_to_body_frame(ego_state["vel"], quat) * vel_scale      # shape: (B, 3) body
    rot_d = quat_to_rotmat_flat(quat)                                     # shape: (B, 9) already [-1,1]
    ang_d = ego_state["ang_vel"] * ang_vel_scale                          # shape: (B, 3) body
    rel_pos_a = rotate_to_body_frame(
        opponent_state["pos"] - ego_state["pos"], quat,
    ) * pos_scale                                                         # shape: (B, 3) body
    rel_vel_a = rotate_to_body_frame(
        opponent_state["vel"] - ego_state["vel"], quat,
    ) * vel_scale                                                         # shape: (B, 3) body

    s_d = torch.cat([pos_d, vel_d, rot_d, ang_d], dim=1)                  # shape: (B, 18)
    s_a_kin = torch.cat([rel_pos_a, rel_vel_a], dim=1)                    # shape: (B, 6) body
    obs = torch.cat([s_d, s_a_kin, last_action], dim=1)                   # shape: (B, 28)
    return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)


class PretrainedDefender:
    """Load Phase 1.0 산출물 ``defender_v*.pt`` and run frozen actor.

    The actor expects the defender obs layout ``[s_d, s_a_kin, last_action]``
    (28-dim, PR-D).  Construction loads weights and sets ``requires_grad=False``
    + ``eval()``.

    ``ckpt_path`` accepts either:
      - a specific ``defender_v*.pt`` file, OR
      - a directory containing ``defender_v*.pt`` (latest by mtime is picked,
        useful for AMS-DRL stage progression v0 → v1 → v2 ...).

    Implementation detail: ``scripts/train_defender.py`` does
    ``torch.save(runner.alg.actor, ...)``, so loading reconstructs the rsl-rl
    actor module directly.  This requires ``rsl_rl`` to be importable in the
    process that loads it; raise a clear error otherwise.
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

        # File or directory both accepted; resolver raises FileNotFoundError
        # with an informative message if no defender_v*.pt is found.
        requested = Path(ckpt_path)
        self.ckpt_path = resolve_actor_ckpt(ckpt_path, prefix="defender")
        # 사용자가 directory 또는 symlink 줬을 때 어떤 파일이 실제 picking됐는지
        # 즉시 stdout으로 명시 — 의도한 ckpt를 쓰는지 확인용.
        try:
            mtime = self.ckpt_path.stat().st_mtime
            from datetime import datetime as _dt
            mtime_str = _dt.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            mtime_str = "?"
        print(
            f"[PretrainedDefender] requested: {requested}\n"
            f"[PretrainedDefender] resolved → {self.ckpt_path}  (mtime {mtime_str})"
        )

        try:
            import rsl_rl  # noqa: F401  — needed for unpickling the actor module
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                "PretrainedDefender requires rsl-rl to be importable for "
                f"unpickling the actor module saved at {self.ckpt_path}."
            ) from e

        actor = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)

        actor_dim = int(getattr(actor, "obs_dim", -1))
        if actor_dim != -1 and actor_dim != DEFENDER_OBS_DIM:
            raise ValueError(
                f"PretrainedDefender ckpt actor.obs_dim={actor_dim} does not match "
                f"DEFENDER_OBS_DIM={DEFENDER_OBS_DIM}. ckpt is incompatible — "
                "regenerate via scripts/train_defender.py with the current obs layout."
            )
        self.actor = actor
        print(
            f"[PretrainedDefender] actor obs_dim={actor_dim} num_actions={self.num_actions}  "
            f"frozen, eval()=on"
        )
        # PR-D — last_action is part of the obs vector. Initialize to zero
        # and update each step (env doesn't pass it through the ScriptedPolicy
        # interface, so the policy tracks its own history).
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
        del g_mission
        obs = _build_defender_policy_obs(
            ego_state, opponent_state, self.last_action,
            self.pos_scale, self.vel_scale, self.ang_vel_scale,
        )
        td = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        action = self.actor(td)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        self.last_action[:] = action
        return action


# ---------------------------------------------------------------------------
# PretrainedDefenderPool — Fictitious Play 다중 ckpt mixing (Phase D, 2026-05-07)
# ---------------------------------------------------------------------------

class PretrainedDefenderPool:
    """Multi-ckpt defender pool for Fictitious Play (FP).

    Loads K frozen actors from given ckpt paths. Each env is assigned a ckpt
    id; on reset (episode-level), envs_idx are reassigned new random ckpts
    (Heinrich & Silver 2016 NFSP style). Per-step inference dispatches obs to
    the actor for each env's assigned ckpt via mask grouping.

    Drop-in compatible with PretrainedDefender (same step / reset interface).
    Used by ``train_attacker.py --defender_ckpt_pool ...`` for FP-based
    AMS-DRL co-training stage 4 (Phase D).

    Args:
        num_envs:    parallel envs (Genesis batched).
        ckpt_paths:  list of attacker_v_K.pt or directory paths (≥1).
        sample:      "uniform" — random uniform over pool; only mode in v1.
        seed:        rng seed for env_to_ckpt assignment (reproducibility).
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
        current_best_ckpt: str | Path | None = None,   # Phase F (2026-05-10) AMSPB pold
        pold: float = 1.0,                              # default 1.0 = old behavior (pool only)
    ):
        from agents.ppo_runner import resolve_actor_ckpt

        # Phase F — current_best 없으면 ckpt_paths 비어있어도 OK (cycle 1 fallback).
        if not ckpt_paths and current_best_ckpt is None:
            raise ValueError(
                "PretrainedDefenderPool requires at least 1 ckpt path or current_best_ckpt."
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
            import rsl_rl  # noqa: F401  — needed for unpickling actor modules
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                "PretrainedDefenderPool requires rsl-rl to be importable for "
                "unpickling actor modules."
            ) from e

        # Resolve all ckpts upfront (FileNotFoundError early).
        resolved: list[Path] = []
        actors: list[Any] = []
        for cp in ckpt_paths:
            r = resolve_actor_ckpt(cp, prefix="defender")
            actor = torch.load(r, map_location=self.device, weights_only=False)
            actor.eval()
            for p in actor.parameters():
                p.requires_grad_(False)
            actor_dim = int(getattr(actor, "obs_dim", -1))
            if actor_dim != -1 and actor_dim != DEFENDER_OBS_DIM:
                raise ValueError(
                    f"PretrainedDefenderPool ckpt {r} actor.obs_dim={actor_dim} "
                    f"does not match DEFENDER_OBS_DIM={DEFENDER_OBS_DIM}."
                )
            resolved.append(r)
            actors.append(actor)
        self.ckpt_paths: list[Path] = resolved
        self.actors: list[Any] = actors
        K = len(self.actors)
        # Phase F — current_best (별도 ckpt for pold mixing)
        self.current_best_actor: Any | None = None
        if current_best_ckpt is not None:
            cb_path = resolve_actor_ckpt(current_best_ckpt, prefix="defender")
            cb_actor = torch.load(cb_path, map_location=self.device, weights_only=False)
            cb_actor.eval()
            for p in cb_actor.parameters():
                p.requires_grad_(False)
            self.current_best_actor = cb_actor
            self.current_best_path = cb_path
        self.pold = float(pold)
        eff_pold = self.pold if (K > 0 and self.current_best_actor is not None) else (1.0 if K > 0 else 0.0)
        print(
            f"[PretrainedDefenderPool] loaded K={K} pool ckpts + current_best={'yes' if self.current_best_actor else 'no'} "
            f"(pold={self.pold} → effective={eff_pold:.2f}, num_envs={self.num_envs})"
        )
        for i, p in enumerate(self.ckpt_paths):
            print(f"  [{i}] {p}")
        if self.current_best_actor is not None:
            print(f"  [current_best] {self.current_best_path}")

        # env_to_ckpt mapping: -1 = current_best, 0..K-1 = pool index.
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(seed))
        self.env_to_ckpt = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._reassign_all()

        self.last_action = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float32
        )

    def _reassign_all(self) -> None:
        """Initial assignment for all envs based on pold mixing."""
        all_idx = torch.arange(self.num_envs, device=self.device)
        self._reassign(all_idx)

    def _reassign(self, envs_idx: torch.Tensor) -> None:
        """Phase F pold mixing — pold prob → pool, (1-pold) prob → current_best."""
        K = len(self.actors)
        n = envs_idx.numel()
        if K == 0:
            # No pool — 모두 current_best (-1 sentinel)
            self.env_to_ckpt[envs_idx] = -1
            return
        if self.current_best_actor is None:
            # No current_best — 모두 pool (pold=1.0 effective)
            self.env_to_ckpt[envs_idx] = torch.randint(
                0, K, (n,), device=self.device, generator=self._gen,
            )
            return
        # Mix: pold 확률로 pool, else current_best
        rand_uniform = torch.rand(n, device=self.device, generator=self._gen)
        use_pool = rand_uniform < self.pold
        # pool indices for use_pool=True
        pool_idx = torch.randint(0, K, (n,), device=self.device, generator=self._gen)
        # -1 sentinel for use_pool=False (current_best)
        new_assignment = torch.where(use_pool, pool_idx, torch.full_like(pool_idx, -1))
        self.env_to_ckpt[envs_idx] = new_assignment

    def reset(self, envs_idx: torch.Tensor) -> None:
        """Reset envs_idx: zero last_action + reassign new ckpt ids (pold mixing)."""
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
        del g_mission
        obs = _build_defender_policy_obs(
            ego_state, opponent_state, self.last_action,
            self.pos_scale, self.vel_scale, self.ang_vel_scale,
        )
        action = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float32
        )
        # Dispatch — pool actors (env_to_ckpt 0..K-1) + current_best (env_to_ckpt = -1).
        for k, actor in enumerate(self.actors):
            mask = self.env_to_ckpt == k
            n_k = int(mask.sum().item())
            if n_k == 0:
                continue
            sub = obs[mask]
            td = TensorDict({"policy": sub}, batch_size=[n_k], device=self.device)
            action[mask] = actor(td)
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
