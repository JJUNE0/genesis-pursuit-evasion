"""Phase 3f / Paper §5 — Random-delay deployment eval (E3).

Tests robustness under realistic delay variability:

  - **fixed_tau50**: τ ≡ 50 (sanity / cellular-relay worst-case)
  - **random_per_episode**: τ ∼ U[1, 50] sampled once at episode reset
  - **random_per_step**: τ ∼ U[1, 50] sampled fresh every env step
                         (packet jitter / retransmit storm scenario)

For each (pair, scenario, fc_type), runs N episodes and records mission/
capture/oob_crash/timeout rates + mean episode length. Used for Figure 6
("τ-unaware degrades the least under increasing delay variability").

Implementation note (CLAUDE.md §1: env code unchanged):
The env is always built with ``tau_delay=tau_max`` so that
``env.defender_state_history`` retains τ_max+1 step lookback. We then **stub
into the attacker obs's defender slot externally**, picking the slot
``defender_state_history[τ_max - effective_τ]`` for the current effective_τ.
no_fc baselines also use this external override to keep the comparison
apples-to-apples (same obs except for forecaster correction).

Usage::

    python scripts/eval_random_delay.py \\
        --pairs a5_d6 a7_d8 a8_d9 a9_d10 \\
        --scenarios fixed_tau50 random_per_episode random_per_step \\
        --fc_types no_fc naive_constvel tau_aware_flow tau_unaware_flow \\
        --num_episodes 500 --num_envs 64 \\
        --out_dir logs/forecaster/comparisons/eval_random_delay
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
from agents.forecaster.trainer import load_ckpt as load_aware_ckpt   # noqa: E402
from agents.forecaster.tau_unaware.trainer import load_ckpt as load_unaware_ckpt   # noqa: E402
from envs.base_pe_env import quat_to_rotmat_flat   # noqa: E402

from scripts.eval_5forecasters import (   # noqa: E402
    HistoryBuffer, build_env, get_attacker_world_state,
    transform_aware, transform_unaware,
)


SCENARIOS = ("fixed_tau50", "random_per_episode", "random_per_step")
FC_TYPES = ("no_fc", "naive_constvel", "tau_aware_flow", "tau_unaware_flow", "tau_unaware_det")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-delay deployment eval (E3)")
    p.add_argument("--pairs", type=str, nargs="+",
                   default=["a5_d6", "a7_d8", "a8_d9", "a9_d10"])
    p.add_argument("--scenarios", type=str, nargs="+", default=list(SCENARIOS),
                   choices=list(SCENARIOS))
    p.add_argument("--fc_types", type=str, nargs="+",
                   default=["no_fc", "naive_constvel", "tau_aware_flow", "tau_unaware_flow"],
                   choices=list(FC_TYPES))
    p.add_argument("--ams_dir", type=str, default="logs/phasef/ams_v9")
    p.add_argument("--ckpt_aware", type=str, default="logs/forecaster/ams_v9_multi_v1/best.pt")
    p.add_argument("--ckpt_unaware", type=str, default="logs/forecaster/tau_unaware_v1/best.pt")
    p.add_argument("--ckpt_det", type=str, default="logs/forecaster/tau_unaware_det_v1/best.pt")
    p.add_argument("--forecaster_data_dir", type=str,
                   default="data/forecaster_data_ams_v9_c8")
    p.add_argument("--tau_max", type=int, default=50)
    p.add_argument("--tau_min", type=int, default=1)
    p.add_argument("--history_K", type=int, default=TUC.DEFAULT_HISTORY_K)
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_euler_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Multi-seed override for headline cells.")
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_steps_safety_factor", type=int, default=4)
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Per-env effective_τ tracker (per-episode / per-step / fixed)
# ---------------------------------------------------------------------------

class EffectiveTauScheduler:
    """Decides per-env, per-step effective τ value."""

    def __init__(self, scenario: str, num_envs: int, tau_min: int, tau_max: int,
                 device: torch.device, gen_seed: int = 0):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")
        self.scenario = scenario
        self.num_envs = int(num_envs)
        self.tau_min = int(tau_min)
        self.tau_max = int(tau_max)
        self.device = device
        self._gen = torch.Generator(device=device)
        self._gen.manual_seed(int(gen_seed))
        # Initial assignment
        self.current = self._sample_all()                              # (B,) int

    def _sample_all(self) -> torch.Tensor:
        if self.scenario == "fixed_tau50":
            return torch.full((self.num_envs,), self.tau_max,
                              device=self.device, dtype=torch.long)
        return torch.randint(self.tau_min, self.tau_max + 1, (self.num_envs,),
                             device=self.device, generator=self._gen)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Called when env auto-resets done envs (per-episode τ refresh)."""
        if envs_idx.numel() == 0:
            return
        if self.scenario == "fixed_tau50":
            return
        # Both random_per_episode and random_per_step refresh on reset.
        n = envs_idx.numel()
        new = torch.randint(self.tau_min, self.tau_max + 1, (n,),
                            device=self.device, generator=self._gen)
        self.current[envs_idx] = new

    def step_advance(self) -> None:
        """Per-step τ refresh (only for random_per_step)."""
        if self.scenario == "random_per_step":
            self.current = self._sample_all()


# ---------------------------------------------------------------------------
# Helper: extract defender state @ effective_τ-step ago, world raw 18D
# ---------------------------------------------------------------------------

def get_defender_at_eff_tau(env, eff_tau: torch.Tensor) -> torch.Tensor:
    """Per-env lookup into env.defender_state_history at eff_tau-step ago.

    env.defender_state_history layout: (env.tau_delay+1, B, 6) — index 0 = oldest,
    index -1 = newest (= current). For env.tau_delay=tau_max, index
    ``tau_max - eff_tau`` corresponds to "eff_tau-step ago".

    Returns ``(B, 1, 18)`` world raw layout (rot=identity, ang_vel=0 padded —
    only pos+vel slots are used by forecaster + frame transform).
    """
    B = env.num_envs
    device = env.device
    if int(env.tau_delay) == 0:
        # Fallback: only current available
        delayed_kin = torch.cat([env.defender_pos, env.defender_vel], dim=1)
    else:
        # Index: tau_max - eff_tau (per-env). tau_max = env.tau_delay.
        idx = int(env.tau_delay) - eff_tau.clamp(min=0, max=int(env.tau_delay))  # (B,)
        # Gather along dim 0
        # env.defender_state_history shape: (tau_max+1, B, 6)
        # We need delayed_kin[b] = env.defender_state_history[idx[b], b]
        idx_3d = idx.view(1, B, 1).expand(1, B, 6)                  # (1, B, 6)
        gathered = torch.gather(env.defender_state_history, 0, idx_3d)  # (1, B, 6)
        delayed_kin = gathered.squeeze(0)                           # (B, 6)
    pos = delayed_kin[:, 0:3]
    vel = delayed_kin[:, 3:6]
    rot_id = torch.zeros((B, 9), device=device, dtype=torch.float32)
    rot_id[:, [0, 4, 8]] = 1.0
    ang = torch.zeros((B, 3), device=device, dtype=torch.float32)
    s_d = torch.cat([pos, vel, rot_id, ang], dim=1)
    return s_d.unsqueeze(1)                                          # (B, 1, 18)


# ---------------------------------------------------------------------------
# Eval loop (random-delay, fc-dispatched)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    env, actor, num_episodes: int,
    *,
    fc_type: str,
    forecasters: dict[str, Any],
    scheduler: EffectiveTauScheduler,
    history_K: int,
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
        # Per-step τ refresh (no-op except random_per_step)
        scheduler.step_advance()
        eff_tau = scheduler.current                                       # (B,) — per-env effective τ

        obs_td = env.get_observations()
        policy_obs = obs_td["policy"]                                     # (B, 31)

        # External override: replace obs's defender slot using effective_tau lookup
        # (this affects no_fc too — the env's auto-built obs uses tau_delay=tau_max
        # which would always be 50-step delay; we want effective_tau).
        defender_eff_18 = get_defender_at_eff_tau(env, eff_tau)           # (B, 1, 18)

        if fc_type == "no_fc":
            # Just transform the eff_tau-step-ago defender into body frame and stub-in.
            # Use last-1 (= current) attacker pose as reference (matches env's normal obs convention).
            # Reference frame computation: same as transform_unaware reference (last step).
            B = num_envs
            # Build a single-step "attacker_history" (just current attacker world state) for transform.
            current_world = get_attacker_world_state(env).unsqueeze(1)    # (B, 1, 18)
            _, defender_b = transform_unaware(current_world, defender_eff_18,
                                              pos_scale, vel_scale, ang_vel_scale)
            corrected = policy_obs.clone()
            corrected[:, 18:24] = defender_b.squeeze(1)
        else:
            # FC paths: use history buffer + correct context for each fc_type.
            # For random τ scenarios, we still pass eff_tau (per-batch scalar approx).
            # Since Pattern 2 collator-style models expect a single τ, use per-env mode:
            # call model with the median or majority τ (approx). For best accuracy, run
            # per-unique-τ subbatches — but that's expensive. Approximation: use mode τ.
            # In fixed_tau50 scenario, all envs share τ=50 → no approximation.
            tau_modal = int(eff_tau.float().median().item())    # batch-level τ approximation

            if fc_type == "tau_aware_flow":
                a_w, act_w = history.window_aware(tau_modal)
                a_b, d_b = transform_aware(a_w, defender_eff_18,
                                            pos_scale, vel_scale, ang_vel_scale)
                ctx = {"attacker_history": a_b, "defender_delayed": d_b,
                       "action_history": act_w}
                pred_seq = sample_aware(model=forecasters["tau_aware_flow"],
                                         context=ctx, tau=tau_modal,
                                         num_steps=num_euler_steps, target_dim=6)
                pred_last = pred_seq[:, -1, :]
            elif fc_type == "tau_unaware_flow":
                a_w, act_w = history.window_unaware(history_K)
                a_b, d_b = transform_unaware(a_w, defender_eff_18,
                                              pos_scale, vel_scale, ang_vel_scale)
                ctx = {"attacker_history": a_b, "defender_last": d_b,
                       "action_history": act_w}
                pred_last = sample_unaware(model=forecasters["tau_unaware_flow"],
                                            context=ctx, num_steps=num_euler_steps,
                                            target_dim=6)
            elif fc_type == "tau_unaware_det":
                a_w, act_w = history.window_unaware(history_K)
                a_b, d_b = transform_unaware(a_w, defender_eff_18,
                                              pos_scale, vel_scale, ang_vel_scale)
                ctx = {"attacker_history": a_b, "defender_last": d_b,
                       "action_history": act_w}
                pred_last = forecasters["tau_unaware_det"](ctx)
            elif fc_type == "naive_constvel":
                a_w, _ = history.window_unaware(history_K)
                _, d_b = transform_unaware(a_w, defender_eff_18,
                                            pos_scale, vel_scale, ang_vel_scale)
                ctx = {"defender_last": d_b}
                pred_last = forecasters["naive_constvel"](ctx, tau=tau_modal)
            else:
                raise ValueError(f"unknown fc_type {fc_type!r}")
            corrected = policy_obs.clone()
            corrected[:, 18:24] = pred_last

        td = TensorDict({"policy": corrected}, batch_size=[num_envs], device=device)
        action = actor(td)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        per_env_step += 1
        _next_obs, _rew, done, _extras = env.step(action)

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
            scheduler.reset_idx(done_idx)

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
    out["mean_ep_length"] = float(np.mean(stats["episode_lengths"])) if stats["episode_lengths"] else 0.0
    out["n_collected"] = stats["n_collected"]
    return out


# ---------------------------------------------------------------------------
# Forecaster loaders (reuse pattern from eval_5forecasters)
# ---------------------------------------------------------------------------

def load_all_forecasters(args, device, scales) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if "naive_constvel" in args.fc_types:
        out["naive_constvel"] = None    # filled in main with dt

    if "tau_aware_flow" in args.fc_types:
        m = ForecasterModel(tau_max=int(args.tau_max)).to(device)
        load_aware_ckpt(args.ckpt_aware, model=m, optimizer=None, map_location=device)
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        out["tau_aware_flow"] = m
        print(f"[eval-rd] τ-aware flow loaded: {args.ckpt_aware}", flush=True)

    if "tau_unaware_flow" in args.fc_types:
        m = TauUnawareForecasterModel(history_K=int(args.history_K)).to(device)
        load_unaware_ckpt(args.ckpt_unaware, model=m, optimizer=None, map_location=device)
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        out["tau_unaware_flow"] = m
        print(f"[eval-rd] τ-unaware flow loaded: {args.ckpt_unaware}", flush=True)

    if "tau_unaware_det" in args.fc_types:
        m = TauUnawareDeterministicModel(history_K=int(args.history_K)).to(device)
        load_unaware_ckpt(args.ckpt_det, model=m, optimizer=None, map_location=device)
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        out["tau_unaware_det"] = m
        print(f"[eval-rd] τ-unaware det loaded: {args.ckpt_det}", flush=True)

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
    with open(cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)
    dt_env = float(cfgs["env_cfg"]["dt"])

    fc_data_dir = Path(args.forecaster_data_dir)
    scales = load_scales_from_dataset(fc_data_dir)
    print(f"[eval-rd] scales={scales}  dt={dt_env}", flush=True)

    all_results: dict[str, Any] = {
        "args": {k: getattr(args, k) for k in vars(args)},
        "scales": list(scales),
        "dt": dt_env,
        "cells": [],
    }

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

        for pair in args.pairs:
            try:
                a_idx, d_idx = [int(x[1:]) for x in pair.split("_")]
            except Exception:
                raise SystemExit(f"pair {pair!r} not in 'a{{i}}_d{{j}}'")
            attacker_ckpt = Path(args.ams_dir) / f"attacker_v{a_idx}.pt"
            defender_ckpt = Path(args.ams_dir) / f"defender_v{d_idx}.pt"

            actor = torch.load(str(attacker_ckpt), weights_only=False, map_location=device)
            actor.eval()
            for p in actor.parameters(): p.requires_grad_(False)

            for scenario in args.scenarios:
                for fc_type in args.fc_types:
                    print(f"\n[eval-rd] === seed={seed} pair={pair} scenario={scenario} fc={fc_type} ===",
                          flush=True)
                    defender_policy = PretrainedDefender(
                        num_envs=int(args.num_envs), ckpt_path=str(defender_ckpt), device=device,
                    )
                    # Always build env with tau_delay = tau_max so history holds full lookback
                    env = build_env(cfgs, num_envs=int(args.num_envs),
                                    defender_policy=defender_policy,
                                    tau_delay=int(args.tau_max))
                    scheduler = EffectiveTauScheduler(
                        scenario=scenario, num_envs=int(args.num_envs),
                        tau_min=int(args.tau_min), tau_max=int(args.tau_max),
                        device=device, gen_seed=seed,
                    )
                    t0 = time.time()
                    stats = run_eval(
                        env=env, actor=actor, num_episodes=int(args.num_episodes),
                        fc_type=fc_type, forecasters=forecasters,
                        scheduler=scheduler,
                        history_K=int(args.history_K), scales=scales,
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
                        "seed": seed, "pair": pair, "scenario": scenario, "fc_type": fc_type,
                        "rates": r, "counts": stats["counts"],
                        "n_collected": stats["n_collected"],
                        "elapsed_s": round(elapsed, 1),
                    })
                    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
                    del env, defender_policy

    print(f"\n[eval-rd] DONE — {len(all_results['cells'])} cells. out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
