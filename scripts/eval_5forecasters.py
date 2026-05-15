"""Phase 3f / Paper §5 — 5-way forecaster comparison eval (E2 main result).

Runs the same attacker-defender rollout setup with τ-step delay, swapping in
five different forecaster strategies for the obs's defender slot:

    1. no_fc            — env's raw τ-delayed defender obs (no correction)
    2. naive_constvel   — constant-velocity extrapolation (math, no learning)
    3. tau_aware_flow   — Phase 3c τ-aware flow (ams_v9_multi_v1)
    4. tau_unaware_flow — Phase 3f τ-unaware flow (tau_unaware_v1)
    5. tau_unaware_det  — Phase 3f τ-unaware deterministic head (ablation)

For each (pair, τ, forecaster) cell, runs N episodes and records mission/
capture/oob_crash/timeout rates + mean episode length.

Usage::

    python scripts/eval_5forecasters.py \\
        --pairs a5_d6 a7_d8 a8_d9 a9_d10 \\
        --tau_values 10 25 50 \\
        --num_episodes 500 --num_envs 64 \\
        --ckpt_aware    logs/forecaster/ams_v9_multi_v1/best.pt \\
        --ckpt_unaware  logs/forecaster/tau_unaware_v1/best.pt \\
        --ckpt_det      logs/forecaster/tau_unaware_det_v1/best.pt \\
        --out_dir logs/forecaster/comparisons/eval_5fc
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from agents.forecaster.baselines.naive_constvel import NaiveConstVelForecaster   # noqa: E402
from agents.forecaster.dataset import load_scales_from_dataset   # noqa: E402
from agents.forecaster.model import ForecasterModel   # noqa: E402
from agents.forecaster.rectified_flow import sample_via_euler as sample_aware   # noqa: E402
from agents.forecaster.tau_unaware import (   # noqa: E402
    TauUnawareDeterministicModel,
    TauUnawareForecasterModel,
    sample_via_euler as sample_unaware,
)
from agents.forecaster.tau_unaware import constants as TUC   # noqa: E402
from agents.forecaster.tau_unaware.dataset import build_sample_unaware as _build_unaware_sample   # noqa: E402
from agents.forecaster.trainer import load_ckpt as load_aware_ckpt   # noqa: E402
from agents.forecaster.tau_unaware.trainer import load_ckpt as load_unaware_ckpt   # noqa: E402
from envs.base_pe_env import quat_to_rotmat_flat   # noqa: E402


FC_TYPES = ("no_fc", "naive_constvel", "tau_aware_flow", "tau_unaware_flow", "tau_unaware_det")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-forecaster comparison eval")
    p.add_argument("--pairs", type=str, nargs="+", default=["a5_d6", "a7_d8", "a8_d9", "a9_d10"],
                   help="Pair labels in 'a{i}_d{j}' form. ckpts under logs/phasef/ams_v9/.")
    p.add_argument("--ams_dir", type=str, default="logs/phasef/ams_v9",
                   help="Directory containing attacker_v*.pt + defender_v*.pt + cfgs.pkl")
    p.add_argument("--ckpt_aware", type=str, default="logs/forecaster/ams_v9_multi_v1/best.pt")
    p.add_argument("--ckpt_unaware", type=str, default="logs/forecaster/tau_unaware_v1/best.pt")
    p.add_argument("--ckpt_det", type=str, default="logs/forecaster/tau_unaware_det_v1/best.pt")
    p.add_argument("--forecaster_data_dir", type=str,
                   default="data/forecaster_data_ams_v9_c8",
                   help="Used to load (pos, vel, ang_vel) scales for frame transform.")
    p.add_argument("--tau_values", type=int, nargs="+", default=[10, 25, 50])
    p.add_argument("--tau_max", type=int, default=50)
    p.add_argument("--history_K", type=int, default=TUC.DEFAULT_HISTORY_K)
    p.add_argument("--fc_types", type=str, nargs="+", default=list(FC_TYPES),
                   choices=list(FC_TYPES))
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_euler_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Multi-seed override (overrides --seed). Each seed runs full sweep.")
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_steps_safety_factor", type=int, default=4)
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# External history buffer (mirrors eval_forecaster_compare; 18D world raw)
# ---------------------------------------------------------------------------

class HistoryBuffer:
    """Per-env attacker (18D world) + action (4D) ring buffer."""

    def __init__(self, num_envs: int, max_len: int, device: torch.device,
                 attacker_dim: int = 18, action_dim: int = 4):
        self.num_envs = int(num_envs)
        self.max_len = int(max_len)
        self.device = device
        self.attacker = torch.zeros((max_len + 1, num_envs, attacker_dim),
                                    device=device, dtype=torch.float32)
        self.action = torch.zeros((max_len, num_envs, action_dim),
                                  device=device, dtype=torch.float32)

    def reset_idx(self, envs_idx: torch.Tensor, fresh_attacker: torch.Tensor) -> None:
        if envs_idx.numel() == 0:
            return
        n = envs_idx.numel()
        self.attacker[:, envs_idx] = fresh_attacker.unsqueeze(0).expand(
            self.max_len + 1, n, fresh_attacker.shape[-1],
        )
        self.action[:, envs_idx] = 0.0

    def push(self, current_attacker: torch.Tensor, current_action: torch.Tensor) -> None:
        self.attacker = torch.roll(self.attacker, shifts=-1, dims=0)
        self.attacker[-1] = current_attacker
        self.action = torch.roll(self.action, shifts=-1, dims=0)
        self.action[-1] = current_action

    def window_aware(self, tau: int) -> tuple[torch.Tensor, torch.Tensor]:
        """τ-aware window: (B, τ+1, 18) attacker, (B, τ, 4) action."""
        a = self.attacker[-(tau + 1):].transpose(0, 1).contiguous()
        c = self.action[-tau:].transpose(0, 1).contiguous()
        return a, c

    def window_unaware(self, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        """τ-unaware window: (B, K, 18) attacker (last K), (B, K, 4) action (last K)."""
        a = self.attacker[-K:].transpose(0, 1).contiguous()
        c = self.action[-K:].transpose(0, 1).contiguous()
        return a, c


# ---------------------------------------------------------------------------
# Frame transform (batched) — reused from Phase 3c eval_forecaster_compare
# ---------------------------------------------------------------------------

def transform_aware(s_a_world, s_d_world_delayed, pos_scale, vel_scale, ang_vel_scale):
    """For τ-aware forecaster: (B, τ+1, 18) attacker + (B, 1, 18) defender.

    Returns (attacker_history (B, τ+1, 18) body+scaled, defender_delayed (B, 1, 6))
    in t-attacker body frame (reference = attacker pose at τ+1-th step = current).
    """
    B, K, _ = s_a_world.shape
    pos_k = s_a_world[..., 0:3] * pos_scale
    vel_world_k = s_a_world[..., 3:6]
    rot_k = s_a_world[..., 6:15].reshape(B, K, 3, 3)
    ang_k = s_a_world[..., 15:18] * ang_vel_scale
    vel_body_k = torch.einsum("bkji,bkj->bki", rot_k, vel_world_k)
    vel_k = vel_body_k * vel_scale
    rot_flat_k = s_a_world[..., 6:15]
    attacker_history = torch.cat([pos_k, vel_k, rot_flat_k, ang_k], dim=-1)

    attacker_pos_t = s_a_world[:, -1, 0:3]
    attacker_vel_t = s_a_world[:, -1, 3:6]
    attacker_rot_t = s_a_world[:, -1, 6:15].reshape(B, 3, 3)
    pos_d = s_d_world_delayed[..., 0:3]
    vel_d = s_d_world_delayed[..., 3:6]
    rel_pos_b = torch.einsum("bji,bnj->bni", attacker_rot_t,
                             pos_d - attacker_pos_t.unsqueeze(1)) * pos_scale
    rel_vel_b = torch.einsum("bji,bnj->bni", attacker_rot_t,
                             vel_d - attacker_vel_t.unsqueeze(1)) * vel_scale
    defender_delayed = torch.cat([rel_pos_b, rel_vel_b], dim=-1)
    return attacker_history, defender_delayed


def transform_unaware(s_a_world, s_d_world_last, pos_scale, vel_scale, ang_vel_scale):
    """For τ-unaware forecaster: (B, K, 18) attacker (NO +1), (B, 1, 18) defender_last.

    Reference = attacker pose at K-th step (=last real-time step) — matches
    Phase 3f build_sample_unaware convention.
    """
    B, K, _ = s_a_world.shape
    pos_k = s_a_world[..., 0:3] * pos_scale
    vel_world_k = s_a_world[..., 3:6]
    rot_k = s_a_world[..., 6:15].reshape(B, K, 3, 3)
    ang_k = s_a_world[..., 15:18] * ang_vel_scale
    vel_body_k = torch.einsum("bkji,bkj->bki", rot_k, vel_world_k)
    vel_k = vel_body_k * vel_scale
    rot_flat_k = s_a_world[..., 6:15]
    attacker_history = torch.cat([pos_k, vel_k, rot_flat_k, ang_k], dim=-1)

    attacker_pos_ref = s_a_world[:, -1, 0:3]
    attacker_vel_ref = s_a_world[:, -1, 3:6]
    attacker_rot_ref = s_a_world[:, -1, 6:15].reshape(B, 3, 3)
    pos_d = s_d_world_last[..., 0:3]
    vel_d = s_d_world_last[..., 3:6]
    rel_pos_b = torch.einsum("bji,bnj->bni", attacker_rot_ref,
                             pos_d - attacker_pos_ref.unsqueeze(1)) * pos_scale
    rel_vel_b = torch.einsum("bji,bnj->bni", attacker_rot_ref,
                             vel_d - attacker_vel_ref.unsqueeze(1)) * vel_scale
    defender_last = torch.cat([rel_pos_b, rel_vel_b], dim=-1)
    return attacker_history, defender_last


# ---------------------------------------------------------------------------
# Env utilities
# ---------------------------------------------------------------------------

def get_attacker_world_state(env) -> torch.Tensor:
    pos = env.attacker_pos                                              # (B, 3)
    vel = env.attacker_vel                                              # (B, 3) world
    rot = quat_to_rotmat_flat(env.attacker_quat)                        # (B, 9)
    ang = env.attacker_ang_vel                                          # (B, 3) body
    return torch.cat([pos, vel, rot, ang], dim=1)                       # (B, 18)


def get_defender_delayed_world_18d(env) -> torch.Tensor:
    """env.defender_state_history[0] (or current if τ=0) → (B, 1, 18) world raw."""
    B = env.num_envs
    device = env.device
    if int(env.tau_delay) > 0:
        delayed_kin = env.defender_state_history[0]                     # (B, 6) world
    else:
        delayed_kin = torch.cat([env.defender_pos, env.defender_vel], dim=1)
    pos = delayed_kin[:, 0:3]
    vel = delayed_kin[:, 3:6]
    rot_id = torch.zeros((B, 9), device=device, dtype=torch.float32)
    rot_id[:, [0, 4, 8]] = 1.0
    ang = torch.zeros((B, 3), device=device, dtype=torch.float32)
    s_d = torch.cat([pos, vel, rot_id, ang], dim=1)
    return s_d.unsqueeze(1)


def build_env(cfgs, num_envs, defender_policy, tau_delay):
    from envs.pe_1v1.env import PursuitEvasion1v1Env   # noqa: PLC0415
    from utils.drone_params import DroneParams         # noqa: PLC0415
    env_cfg = dict(cfgs["env_cfg"])
    env_cfg["ego"] = "attacker"
    env_cfg["tau_delay"] = int(tau_delay)
    drone_yaml = cfgs.get("drone_yaml") or str(ROOT / "configs" / "drones" / "nova.yaml")
    params = DroneParams(drone_yaml)
    return PursuitEvasion1v1Env(
        num_envs=num_envs, params=params,
        env_cfg=env_cfg, obs_cfg=cfgs["obs_cfg"], reward_cfg=cfgs["reward_cfg"],
        command_cfg=cfgs["command_cfg"], defender_policy=defender_policy,
        show_viewer=False,
    )


# ---------------------------------------------------------------------------
# Eval loop (fc_type-dispatched)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    env, actor, num_episodes: int,
    *,
    fc_type: str,
    forecasters: dict[str, Any],   # name → model (or None for no_fc)
    tau: int, history_K: int,
    scales: tuple[float, float, float],
    num_euler_steps: int,
    max_steps_safety_factor: int = 4,
) -> dict[str, Any]:
    device = env.device
    num_envs = env.num_envs
    pos_scale, vel_scale, ang_vel_scale = scales

    obs = env.reset()
    history = HistoryBuffer(num_envs, max_len=max(history_K, env.tau_delay), device=device)
    if fc_type != "no_fc":
        all_idx = torch.arange(num_envs, device=device)
        history.reset_idx(all_idx, get_attacker_world_state(env))

    counts = {"mission": 0, "capture": 0, "oob_crash": 0, "timeout": 0}
    episode_lengths: list[int] = []
    per_env_step = torch.zeros((num_envs,), device=device, dtype=torch.long)
    n_collected = 0
    step_idx = 0
    safety_steps = (
        int(env.max_episode_length)
        * (max(1, num_episodes // num_envs + 4))
        * int(max_steps_safety_factor)
    )

    while n_collected < num_episodes and step_idx < safety_steps:
        obs_td = env.get_observations()
        policy_obs = obs_td["policy"]                                   # (B, 31)

        if fc_type == "no_fc":
            corrected = policy_obs
        else:
            # Build forecaster context using env history
            defender_d_18 = get_defender_delayed_world_18d(env)         # (B, 1, 18)
            if fc_type == "tau_aware_flow":
                a_w, act_w = history.window_aware(tau)                  # (B, τ+1, 18), (B, τ, 4)
                a_b, d_b = transform_aware(a_w, defender_d_18,
                                            pos_scale, vel_scale, ang_vel_scale)
                ctx = {"attacker_history": a_b, "defender_delayed": d_b,
                       "action_history": act_w}
                pred_seq = sample_aware(model=forecasters["tau_aware_flow"],
                                         context=ctx, tau=tau,
                                         num_steps=num_euler_steps, target_dim=6)
                pred_last = pred_seq[:, -1, :]                          # (B, 6)
            elif fc_type == "tau_unaware_flow":
                a_w, act_w = history.window_unaware(history_K)          # (B, K, 18), (B, K, 4)
                a_b, d_b = transform_unaware(a_w, defender_d_18,
                                              pos_scale, vel_scale, ang_vel_scale)
                ctx = {"attacker_history": a_b, "defender_last": d_b,
                       "action_history": act_w}
                pred_last = sample_unaware(model=forecasters["tau_unaware_flow"],
                                            context=ctx, num_steps=num_euler_steps,
                                            target_dim=6)               # (B, 6)
            elif fc_type == "tau_unaware_det":
                a_w, act_w = history.window_unaware(history_K)
                a_b, d_b = transform_unaware(a_w, defender_d_18,
                                              pos_scale, vel_scale, ang_vel_scale)
                ctx = {"attacker_history": a_b, "defender_last": d_b,
                       "action_history": act_w}
                pred_last = forecasters["tau_unaware_det"](ctx)         # (B, 6) deterministic
            elif fc_type == "naive_constvel":
                # Use unaware-style transform for defender_last only (history not needed).
                a_w, _ = history.window_unaware(history_K)
                _, d_b = transform_unaware(a_w, defender_d_18,
                                            pos_scale, vel_scale, ang_vel_scale)
                ctx = {"defender_last": d_b}
                pred_last = forecasters["naive_constvel"](ctx, tau=tau)  # (B, 6)
            else:
                raise ValueError(f"unknown fc_type {fc_type!r}")
            corrected = policy_obs.clone()
            corrected[:, 18:24] = pred_last

        td = TensorDict({"policy": corrected}, batch_size=[num_envs], device=device)
        action = actor(td)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        per_env_step += 1
        _next_obs, _rew, done, _extras = env.step(action)

        # Termination counts
        m = env.last_term["mission"]
        c = env.last_term["captured"]
        oo = (env.last_term["att_crash"] | env.last_term["def_crash"]
              | env.last_term["att_oob"] | env.last_term["def_oob"])
        to = env.last_term["timeout"]
        counts["mission"] += int(m.sum().item())
        counts["capture"] += int(c.sum().item())
        counts["oob_crash"] += int(oo.sum().item())
        counts["timeout"] += int(to.sum().item())

        done_idx = done.nonzero(as_tuple=False).reshape(-1)
        if done_idx.numel() > 0:
            episode_lengths.extend(per_env_step[done_idx].cpu().tolist())
            n_collected += int(done_idx.numel())
            per_env_step[done_idx] = 0

        if fc_type != "no_fc":
            current = get_attacker_world_state(env)
            history.push(current, action)
            if done_idx.numel() > 0:
                history.reset_idx(done_idx, current[done_idx])

        step_idx += 1

    return {
        "counts": counts,
        "n_collected": n_collected,
        "episode_lengths": episode_lengths,
        "step_idx": step_idx,
    }


def _rates(stats):
    n = max(1, stats["n_collected"])
    out = {f"{k}_rate": stats["counts"][k] / n for k in stats["counts"]}
    if stats["episode_lengths"]:
        out["mean_ep_length"] = float(np.mean(stats["episode_lengths"]))
    else:
        out["mean_ep_length"] = 0.0
    out["n_collected"] = stats["n_collected"]
    return out


# ---------------------------------------------------------------------------
# Forecaster loaders
# ---------------------------------------------------------------------------

def load_all_forecasters(args, device, scales) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pos_scale, vel_scale, _ = scales

    if "naive_constvel" in args.fc_types:
        # dt from cfgs.pkl loaded later; pass via setattr after env build (closure).
        # Here just instantiate; dt set later in main.
        out["naive_constvel"] = None    # placeholder; fill in main with dt

    if "tau_aware_flow" in args.fc_types:
        m = ForecasterModel(tau_max=int(args.tau_max)).to(device)
        load_aware_ckpt(args.ckpt_aware, model=m, optimizer=None, map_location=device)
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        out["tau_aware_flow"] = m
        print(f"[eval-5fc] τ-aware flow loaded: {args.ckpt_aware}", flush=True)

    if "tau_unaware_flow" in args.fc_types:
        m = TauUnawareForecasterModel(history_K=int(args.history_K)).to(device)
        load_unaware_ckpt(args.ckpt_unaware, model=m, optimizer=None, map_location=device)
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        out["tau_unaware_flow"] = m
        print(f"[eval-5fc] τ-unaware flow loaded: {args.ckpt_unaware}", flush=True)

    if "tau_unaware_det" in args.fc_types:
        m = TauUnawareDeterministicModel(history_K=int(args.history_K)).to(device)
        load_unaware_ckpt(args.ckpt_det, model=m, optimizer=None, map_location=device)
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        out["tau_unaware_det"] = m
        print(f"[eval-5fc] τ-unaware det loaded: {args.ckpt_det}", flush=True)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu

    cfgs_pkl = Path(args.ams_dir) / "cfgs.pkl"
    if not cfgs_pkl.is_file():
        raise SystemExit(f"cfgs_pkl not found: {cfgs_pkl}")
    with open(cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)
    dt_env = float(cfgs["env_cfg"]["dt"])

    fc_data_dir = Path(args.forecaster_data_dir)
    scales = load_scales_from_dataset(fc_data_dir)
    print(f"[eval-5fc] scales={scales}  dt={dt_env}", flush=True)

    all_results: dict[str, Any] = {
        "args": {k: getattr(args, k) for k in vars(args)},
        "scales": list(scales),
        "dt": dt_env,
        "cells": [],   # list of {pair, tau, fc_type, seed, rates}
    }

    # ---- Seed loop ----
    for seed in seeds:
        seed = int(seed)
        _seed_all(seed)
        gs.init(backend=backend, seed=seed, logging_level="warning")
        device = gs.device

        forecasters = load_all_forecasters(args, device, scales)
        if "naive_constvel" in forecasters:
            forecasters["naive_constvel"] = NaiveConstVelForecaster(
                dt=dt_env, pos_scale=scales[0], vel_scale=scales[1],
            ).to(device)

        from envs.pe_1v1.scripted_defender import PretrainedDefender   # noqa: PLC0415

        # ---- Pair loop ----
        for pair in args.pairs:
            # 'a5_d6' → attacker_v5.pt, defender_v6.pt
            try:
                a_idx, d_idx = [int(x[1:]) for x in pair.split("_")]
            except Exception:
                raise SystemExit(f"pair label {pair!r} not in 'a{{i}}_d{{j}}' form")
            attacker_ckpt = Path(args.ams_dir) / f"attacker_v{a_idx}.pt"
            defender_ckpt = Path(args.ams_dir) / f"defender_v{d_idx}.pt"
            if not attacker_ckpt.is_file() or not defender_ckpt.is_file():
                raise SystemExit(f"missing ckpt(s) for pair {pair}: {attacker_ckpt}, {defender_ckpt}")

            actor = torch.load(str(attacker_ckpt), weights_only=False, map_location=device)
            actor.eval()
            for p in actor.parameters(): p.requires_grad_(False)

            for tau in args.tau_values:
                tau = int(tau)
                if tau < 1 or tau > int(args.tau_max):
                    raise SystemExit(f"tau {tau} out of [1, {args.tau_max}]")

                for fc_type in args.fc_types:
                    print(f"\n[eval-5fc] === seed={seed} pair={pair} τ={tau} fc={fc_type} ===",
                          flush=True)
                    defender_policy = PretrainedDefender(
                        num_envs=int(args.num_envs), ckpt_path=str(defender_ckpt), device=device,
                    )
                    env = build_env(cfgs, num_envs=int(args.num_envs),
                                    defender_policy=defender_policy, tau_delay=tau)
                    t0 = time.time()
                    stats = run_eval(
                        env=env, actor=actor, num_episodes=int(args.num_episodes),
                        fc_type=fc_type, forecasters=forecasters,
                        tau=tau, history_K=int(args.history_K), scales=scales,
                        num_euler_steps=int(args.num_euler_steps),
                        max_steps_safety_factor=int(args.max_steps_safety_factor),
                    )
                    elapsed = time.time() - t0
                    r = _rates(stats)
                    print(
                        f"  collected {stats['n_collected']} eps in {elapsed:.1f}s — "
                        f"mission={r['mission_rate']:.3f}  capture={r['capture_rate']:.3f}  "
                        f"oob={r['oob_crash_rate']:.3f}  to={r['timeout_rate']:.3f}  "
                        f"mean_len={r['mean_ep_length']:.1f}",
                        flush=True,
                    )
                    all_results["cells"].append({
                        "seed": seed, "pair": pair, "tau": tau, "fc_type": fc_type,
                        "rates": r, "counts": stats["counts"],
                        "n_collected": stats["n_collected"],
                        "elapsed_s": round(elapsed, 1),
                    })
                    # Incremental save (crash safety)
                    (out_dir / "results.json").write_text(
                        json.dumps(all_results, indent=2)
                    )
                    del env, defender_policy

    print(f"\n[eval-5fc] DONE — {len(all_results['cells'])} cells. out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
