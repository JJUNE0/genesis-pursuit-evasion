"""Phase 3f / Paper §5 — τ-aware misspecification grid (E4).

Tests how the **τ-aware** forecaster degrades when its declared τ_input does
not match the actual delay τ_actual. Heatmap of mission/capture rates over
``(τ_actual, τ_input) ∈ {10, 25, 50}²``.

Key claim (paper §5):
  - On the diagonal (τ_input = τ_actual): forecaster is "well-specified"
  - Off-diagonal: declared τ wrong → degradation
  - The τ-unaware variant (single point) achieves competitive performance
    at all τ_actual **without needing τ_input** → misspec-immune by design

For Figure 7 — 3×3 heatmap (mission rate per cell) + adjacent τ-unaware bar.

Usage::

    python scripts/eval_misspec_grid.py \\
        --pairs a5_d6 a7_d8 a8_d9 a9_d10 \\
        --tau_values 10 25 50 \\
        --num_episodes 500 --num_envs 64 \\
        --out_dir logs/forecaster/comparisons/eval_misspec
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

from agents.forecaster.dataset import load_scales_from_dataset   # noqa: E402
from agents.forecaster.model import ForecasterModel   # noqa: E402
from agents.forecaster.rectified_flow import sample_via_euler as sample_aware   # noqa: E402
from agents.forecaster.tau_unaware import (   # noqa: E402
    TauUnawareForecasterModel,
    sample_via_euler as sample_unaware,
)
from agents.forecaster.tau_unaware import constants as TUC   # noqa: E402
from agents.forecaster.trainer import load_ckpt as load_aware_ckpt   # noqa: E402
from agents.forecaster.tau_unaware.trainer import load_ckpt as load_unaware_ckpt   # noqa: E402

from scripts.eval_5forecasters import (   # noqa: E402
    HistoryBuffer, build_env, get_attacker_world_state,
    transform_aware, transform_unaware,
)
from scripts.eval_random_delay import get_defender_at_eff_tau   # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="τ-aware misspecification grid (E4)")
    p.add_argument("--pairs", type=str, nargs="+",
                   default=["a5_d6", "a7_d8", "a8_d9", "a9_d10"])
    p.add_argument("--ams_dir", type=str, default="logs/phasef/ams_v9")
    p.add_argument("--ckpt_aware", type=str, default="logs/forecaster/ams_v9_multi_v1/best.pt")
    p.add_argument("--ckpt_unaware", type=str, default="logs/forecaster/tau_unaware_v1/best.pt",
                   help="Adjacent baseline (single-cell, misspec-immune by design)")
    p.add_argument("--forecaster_data_dir", type=str,
                   default="data/forecaster_data_ams_v9_c8")
    p.add_argument("--tau_values", type=int, nargs="+", default=[10, 25, 50],
                   help="Both τ_actual and τ_input grid values.")
    p.add_argument("--tau_max", type=int, default=50)
    p.add_argument("--history_K", type=int, default=TUC.DEFAULT_HISTORY_K)
    p.add_argument("--include_unaware_baseline", action="store_true", default=True,
                   help="Also eval τ-unaware at each τ_actual for comparison column.")
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_euler_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_steps_safety_factor", type=int, default=4)
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Eval loop — fixed (τ_actual, τ_input) per cell
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(
    env, actor, num_episodes: int,
    *,
    fc_type: str,
    forecaster: torch.nn.Module,
    tau_actual: int, tau_input: int,
    history_K: int,
    scales: tuple[float, float, float],
    num_euler_steps: int,
    max_steps_safety_factor: int = 4,
) -> dict[str, Any]:
    """fc_type ∈ {'tau_aware_flow', 'tau_unaware_flow'}.

    For 'tau_aware_flow': forecaster called with ``tau=tau_input``; defender_last
    sourced from ``tau_actual``-step ago (env history).
    For 'tau_unaware_flow': forecaster takes K-step history; defender_last from
    ``tau_actual``-step ago (no τ_input concept — diagonal-only).
    """
    device = env.device
    num_envs = env.num_envs
    pos_scale, vel_scale, ang_vel_scale = scales

    obs = env.reset()
    history = HistoryBuffer(num_envs, max_len=max(history_K, env.tau_delay), device=device)
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

    eff_tau_tensor = torch.full((num_envs,), int(tau_actual),
                                device=device, dtype=torch.long)

    while n_collected < num_episodes and step_idx < safety_steps:
        obs_td = env.get_observations()
        policy_obs = obs_td["policy"]
        defender_actual_18 = get_defender_at_eff_tau(env, eff_tau_tensor)   # (B, 1, 18)

        if fc_type == "tau_aware_flow":
            # Window: τ_input + 1 attacker history (forecaster expects τ_input)
            a_w, act_w = history.window_aware(int(tau_input))
            a_b, d_b = transform_aware(a_w, defender_actual_18,
                                        pos_scale, vel_scale, ang_vel_scale)
            ctx = {"attacker_history": a_b, "defender_delayed": d_b,
                   "action_history": act_w}
            pred_seq = sample_aware(model=forecaster, context=ctx,
                                     tau=int(tau_input),
                                     num_steps=num_euler_steps, target_dim=6)
            pred_last = pred_seq[:, -1, :]
        elif fc_type == "tau_unaware_flow":
            a_w, act_w = history.window_unaware(history_K)
            a_b, d_b = transform_unaware(a_w, defender_actual_18,
                                          pos_scale, vel_scale, ang_vel_scale)
            ctx = {"attacker_history": a_b, "defender_last": d_b,
                   "action_history": act_w}
            pred_last = sample_unaware(model=forecaster, context=ctx,
                                        num_steps=num_euler_steps, target_dim=6)
        else:
            raise ValueError(f"unsupported fc_type {fc_type!r}")

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

    fc_data_dir = Path(args.forecaster_data_dir)
    scales = load_scales_from_dataset(fc_data_dir)
    print(f"[eval-misspec] scales={scales}", flush=True)

    all_results: dict[str, Any] = {
        "args": {k: getattr(args, k) for k in vars(args)},
        "scales": list(scales),
        "cells": [],
    }

    for seed in seeds:
        seed = int(seed)
        _seed_all(seed)
        gs.init(backend=backend, seed=seed, logging_level="warning")
        device = gs.device

        # Load both forecasters (τ-aware for grid, τ-unaware for adjacent column)
        m_aware = ForecasterModel(tau_max=int(args.tau_max)).to(device)
        load_aware_ckpt(args.ckpt_aware, model=m_aware, optimizer=None, map_location=device)
        m_aware.eval()
        for p in m_aware.parameters(): p.requires_grad_(False)

        m_unaware = None
        if args.include_unaware_baseline:
            m_unaware = TauUnawareForecasterModel(history_K=int(args.history_K)).to(device)
            load_unaware_ckpt(args.ckpt_unaware, model=m_unaware,
                              optimizer=None, map_location=device)
            m_unaware.eval()
            for p in m_unaware.parameters(): p.requires_grad_(False)

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

            for tau_actual in args.tau_values:
                tau_actual = int(tau_actual)

                # --- τ-aware grid: τ_input ∈ args.tau_values ---
                for tau_input in args.tau_values:
                    tau_input = int(tau_input)
                    print(f"\n[eval-misspec] === seed={seed} pair={pair} "
                          f"τ_actual={tau_actual} τ_input={tau_input} fc=τ-aware ===",
                          flush=True)
                    defender_policy = PretrainedDefender(
                        num_envs=int(args.num_envs), ckpt_path=str(defender_ckpt), device=device,
                    )
                    env = build_env(cfgs, num_envs=int(args.num_envs),
                                    defender_policy=defender_policy,
                                    tau_delay=int(args.tau_max))
                    t0 = time.time()
                    stats = run_eval(
                        env=env, actor=actor, num_episodes=int(args.num_episodes),
                        fc_type="tau_aware_flow", forecaster=m_aware,
                        tau_actual=tau_actual, tau_input=tau_input,
                        history_K=int(args.history_K), scales=scales,
                        num_euler_steps=int(args.num_euler_steps),
                        max_steps_safety_factor=int(args.max_steps_safety_factor),
                    )
                    elapsed = time.time() - t0
                    r = _rates(stats)
                    print(f"  collected {stats['n_collected']} eps in {elapsed:.1f}s — "
                          f"mission={r['mission_rate']:.3f}  capture={r['capture_rate']:.3f}",
                          flush=True)
                    all_results["cells"].append({
                        "seed": seed, "pair": pair,
                        "tau_actual": tau_actual, "tau_input": tau_input,
                        "fc_type": "tau_aware_flow",
                        "rates": r, "counts": stats["counts"],
                        "n_collected": stats["n_collected"],
                        "elapsed_s": round(elapsed, 1),
                    })
                    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
                    del env, defender_policy

                # --- τ-unaware adjacent column (no τ_input dependence) ---
                if m_unaware is not None:
                    print(f"\n[eval-misspec] === seed={seed} pair={pair} "
                          f"τ_actual={tau_actual} fc=τ-unaware ===", flush=True)
                    defender_policy = PretrainedDefender(
                        num_envs=int(args.num_envs), ckpt_path=str(defender_ckpt), device=device,
                    )
                    env = build_env(cfgs, num_envs=int(args.num_envs),
                                    defender_policy=defender_policy,
                                    tau_delay=int(args.tau_max))
                    t0 = time.time()
                    stats = run_eval(
                        env=env, actor=actor, num_episodes=int(args.num_episodes),
                        fc_type="tau_unaware_flow", forecaster=m_unaware,
                        tau_actual=tau_actual, tau_input=tau_actual,
                        history_K=int(args.history_K), scales=scales,
                        num_euler_steps=int(args.num_euler_steps),
                        max_steps_safety_factor=int(args.max_steps_safety_factor),
                    )
                    elapsed = time.time() - t0
                    r = _rates(stats)
                    print(f"  collected {stats['n_collected']} eps in {elapsed:.1f}s — "
                          f"mission={r['mission_rate']:.3f}  capture={r['capture_rate']:.3f}",
                          flush=True)
                    all_results["cells"].append({
                        "seed": seed, "pair": pair,
                        "tau_actual": tau_actual, "tau_input": None,   # τ-unaware: no input
                        "fc_type": "tau_unaware_flow",
                        "rates": r, "counts": stats["counts"],
                        "n_collected": stats["n_collected"],
                        "elapsed_s": round(elapsed, 1),
                    })
                    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2))
                    del env, defender_policy

    print(f"\n[eval-misspec] DONE — {len(all_results['cells'])} cells. out={out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
