"""Phase 3c — Compare attacker performance with vs without forecaster correction.

Holds (attacker, defender) ckpts + a single env (with τ-step communication
delay) constant; sweeps ``τ ∈ {10, 20, 30, 40, 50}`` and for each runs two
parallel rollouts:

  - **NO-FC**:  env auto-builds attacker obs with τ-step delayed defender
                slot (default behavior). Attacker actor consumes that obs.
  - **WITH-FC**: same env / same setup, but each step we run the trained
                 forecaster on the current (attacker-history, action-history,
                 delayed-defender) context, take its **last-step prediction**
                 (forecaster's estimate of the current defender), and stub it
                 into the obs's defender slot before passing to the actor.

Counts mission/captured/oob/timeout terminations and average episode length
across N episodes per (τ, fc/no-fc) cell.

CLAUDE.md §1: env code unchanged. We manage attacker + action history in an
external ``ForecasterHistoryBuffer`` and stub-in obs externally — env stays
read-only.

Usage::

    python scripts/eval_forecaster_compare.py \\
        --attacker_ckpt logs/phasef/ams_v9/attacker_v8.pt \\
        --defender_ckpt logs/phasef/ams_v9/defender_v8.pt \\
        --cfgs_pkl     logs/phasef/ams_v9/cfgs.pkl \\
        --forecaster_ckpt logs/forecaster/ams_v9_multi_v1/best.pt \\
        --tau_values 10 20 30 40 50 \\
        --num_episodes 500 --num_envs 64 \\
        --out_dir logs/forecaster/comparisons/eval_c8
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from agents.forecaster.dataset import load_scales_from_dataset   # noqa: E402
from agents.forecaster.model import ForecasterModel   # noqa: E402
from agents.forecaster.rectified_flow import sample_via_euler   # noqa: E402
from agents.forecaster.trainer import load_ckpt as load_fc_ckpt   # noqa: E402
from envs.base_pe_env import quat_to_rotmat_flat   # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecaster on/off comparison eval")
    p.add_argument("--attacker_ckpt", type=str, required=True)
    p.add_argument("--defender_ckpt", type=str, required=True)
    p.add_argument("--cfgs_pkl", type=str, required=True)
    p.add_argument("--forecaster_ckpt", type=str, required=True)
    p.add_argument("--forecaster_data_dir", type=str, default=None,
                   help="Used only to load (pos, vel, ang_vel) scales. Default: "
                        "data/forecaster_data_ams_v9_c8.")
    p.add_argument("--tau_values", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    p.add_argument("--tau_max", type=int, default=50)
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_euler_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_steps_safety_factor", type=int, default=4,
                   help="Outer-loop safety multiplier on max_episode_length × ceil(target/num_envs).")
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# External history buffer (env code unchanged)
# ---------------------------------------------------------------------------

class ForecasterHistoryBuffer:
    """Per-env attacker (18D world) + action (4D) ring buffer for forecaster context.

    Mirrors env.defender_state_history layout: ``[oldest, ..., newest]`` along
    dim 0. ``push`` rolls and writes the newest slot; ``reset`` fills all
    slots with a single fresh state (mimicking env's tau-buffer reset).
    """

    def __init__(self, num_envs: int, tau_max: int, device: torch.device,
                 attacker_dim: int = 18, action_dim: int = 4):
        self.num_envs = int(num_envs)
        self.tau_max = int(tau_max)
        self.device = device
        # +1 length to hold (t-τ_max .. t) inclusive.
        self.attacker_history = torch.zeros(
            (tau_max + 1, num_envs, attacker_dim), device=device, dtype=torch.float32,
        )
        # τ length: actions a_{t-τ_max..t-1}
        self.action_history = torch.zeros(
            (tau_max, num_envs, action_dim), device=device, dtype=torch.float32,
        )

    def reset_idx(self, envs_idx: torch.Tensor, fresh_attacker_state: torch.Tensor) -> None:
        """Refill buffers for given envs (fresh state for attacker_history; zero for actions)."""
        if envs_idx.numel() == 0:
            return
        n = envs_idx.numel()
        # (τ_max+1, n, 18) ← broadcast fresh_attacker_state (n, 18)
        self.attacker_history[:, envs_idx] = fresh_attacker_state.unsqueeze(0).expand(
            self.tau_max + 1, n, fresh_attacker_state.shape[-1],
        )
        self.action_history[:, envs_idx] = 0.0

    def push(self, current_attacker_state: torch.Tensor, current_action: torch.Tensor) -> None:
        """Roll oldest off, append newest. Both inputs (B, *)."""
        self.attacker_history = torch.roll(self.attacker_history, shifts=-1, dims=0)
        self.attacker_history[-1] = current_attacker_state
        self.action_history = torch.roll(self.action_history, shifts=-1, dims=0)
        self.action_history[-1] = current_action

    def get_window(self, tau: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (attacker_history (B, τ+1, 18), action_history (B, τ, 4))."""
        if tau < 1 or tau > self.tau_max:
            raise ValueError(f"tau out of [1, {self.tau_max}]: {tau}")
        # Slice last (τ+1) attacker steps and last τ actions, transpose to batch-first.
        attacker = self.attacker_history[-(tau + 1):].transpose(0, 1).contiguous()
        actions = self.action_history[-tau:].transpose(0, 1).contiguous()
        return attacker, actions


# ---------------------------------------------------------------------------
# World → t-attacker body frame batched transform (parity with dataset.build_sample)
# ---------------------------------------------------------------------------

def transform_to_body_batched(
    s_a_world: torch.Tensor,           # (B, τ+1, 18) world raw
    s_d_world_delayed: torch.Tensor,   # (B, 1, 18) world raw (defender at t-τ)
    pos_scale: float, vel_scale: float, ang_vel_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (attacker_history_body (B, τ+1, 18), defender_delayed_body (B, 1, 6))."""
    B, K, _ = s_a_world.shape

    pos_k = s_a_world[..., 0:3] * pos_scale                                  # (B, K, 3) world
    vel_world_k = s_a_world[..., 3:6]                                        # (B, K, 3) world
    rot_k = s_a_world[..., 6:15].reshape(B, K, 3, 3)                         # (B, K, 3, 3)
    ang_k = s_a_world[..., 15:18] * ang_vel_scale                            # (B, K, 3) body
    # body vel at step k = R_k^T @ world_vel_k
    vel_body_k = torch.einsum("bkji,bkj->bki", rot_k, vel_world_k)
    vel_k = vel_body_k * vel_scale
    rot_flat_k = s_a_world[..., 6:15]
    attacker_history = torch.cat([pos_k, vel_k, rot_flat_k, ang_k], dim=-1)  # (B, K, 18)

    # t-attacker body frame for delayed defender
    attacker_pos_t = s_a_world[:, -1, 0:3]                                   # (B, 3)
    attacker_vel_t = s_a_world[:, -1, 3:6]                                   # (B, 3)
    attacker_rot_t = s_a_world[:, -1, 6:15].reshape(B, 3, 3)                 # (B, 3, 3)

    pos_d = s_d_world_delayed[..., 0:3]                                      # (B, 1, 3)
    vel_d = s_d_world_delayed[..., 3:6]                                      # (B, 1, 3)
    # einsum 'bji,bnj->bni' = R_t^T @ each row vector
    rel_pos_b = torch.einsum(
        "bji,bnj->bni", attacker_rot_t, pos_d - attacker_pos_t.unsqueeze(1),
    ) * pos_scale
    rel_vel_b = torch.einsum(
        "bji,bnj->bni", attacker_rot_t, vel_d - attacker_vel_t.unsqueeze(1),
    ) * vel_scale
    defender_delayed = torch.cat([rel_pos_b, rel_vel_b], dim=-1)             # (B, 1, 6)

    return attacker_history, defender_delayed


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def build_env(cfgs: dict[str, Any], num_envs: int, defender_policy, tau_delay: int):
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


def get_attacker_world_state_batched(env) -> torch.Tensor:
    """env attribute → 18D world raw [pos(3)+vel_world(3)+rot_flat(9)+ang_vel(3)]."""
    pos = env.attacker_pos                                              # (B, 3)
    vel = env.attacker_vel                                              # (B, 3)
    rot = quat_to_rotmat_flat(env.attacker_quat)                        # (B, 9)
    ang = env.attacker_ang_vel                                          # (B, 3)
    return torch.cat([pos, vel, rot, ang], dim=1)                       # (B, 18)


def get_defender_delayed_world_18d(env) -> torch.Tensor:
    """env.defender_state_history[0] gives (B, 6) delayed pos+vel.

    Pad to 18D (rot=identity flat, ang_vel=0) so the body-frame transform
    helper can consume the same layout. Only the pos+vel slots are used by
    the forecaster (which projects to 6D).
    """
    B = env.num_envs
    device = env.device
    if int(env.tau_delay) > 0:
        delayed_kin = env.defender_state_history[0]                     # (B, 6) world
    else:
        delayed_kin = torch.cat([env.defender_pos, env.defender_vel], dim=1)  # (B, 6)
    pos = delayed_kin[:, 0:3]                                           # (B, 3)
    vel = delayed_kin[:, 3:6]                                           # (B, 3)
    # Pad rot+ang to match 18D layout (these slots are unused downstream).
    rot_identity = torch.zeros((B, 9), device=device, dtype=torch.float32)
    rot_identity[:, [0, 4, 8]] = 1.0   # eye(3) flat
    ang = torch.zeros((B, 3), device=device, dtype=torch.float32)
    s_d = torch.cat([pos, vel, rot_identity, ang], dim=1)               # (B, 18)
    return s_d.unsqueeze(1)                                             # (B, 1, 18)


# ---------------------------------------------------------------------------
# Eval loops
# ---------------------------------------------------------------------------

def _classify_terminations_batch(env, alive_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-env termination booleans (post-step, last_term snapshot)."""
    return {
        "mission":  env.last_term["mission"] & alive_mask,
        "capture":  env.last_term["captured"] & alive_mask,
        "oob_crash": (
            env.last_term["att_crash"] | env.last_term["def_crash"]
            | env.last_term["att_oob"] | env.last_term["def_oob"]
        ) & alive_mask,
        "timeout": env.last_term["timeout"] & alive_mask,
    }


@torch.no_grad()
def run_eval(
    env, actor, num_episodes: int,
    *,
    forecaster: torch.nn.Module | None,
    tau: int,
    scales: tuple[float, float, float],
    num_euler_steps: int,
    max_steps_safety_factor: int = 4,
) -> dict[str, Any]:
    """Single rollout. ``forecaster=None`` → no correction (delayed obs as-is).

    Returns dict with termination counts, episode lengths, n_episodes_collected.
    """
    device = env.device
    num_envs = env.num_envs

    # Reset env + history buffer
    obs = env.reset()
    history: ForecasterHistoryBuffer | None = None
    if forecaster is not None:
        history = ForecasterHistoryBuffer(num_envs, env.tau_delay, device)
        # All envs are freshly reset → fill history with current attacker state.
        all_idx = torch.arange(num_envs, device=device)
        history.reset_idx(all_idx, get_attacker_world_state_batched(env))

    counts = {"mission": 0, "capture": 0, "oob_crash": 0, "timeout": 0}
    episode_lengths: list[int] = []
    # Track per-env step count since reset for episode_length recording.
    per_env_step = torch.zeros((num_envs,), device=device, dtype=torch.long)

    n_collected = 0
    step_idx = 0
    safety_steps = (
        int(env.max_episode_length)
        * (max(1, num_episodes // num_envs + 4))
        * int(max_steps_safety_factor)
    )

    pos_scale, vel_scale, ang_vel_scale = scales

    while n_collected < num_episodes and step_idx < safety_steps:
        # ---- Build actor input ----
        # env.get_observations() returns TensorDict({"policy": (B, 31)}). For
        # ego="attacker", the defender slot [18:24] is already τ-delayed.
        obs_td = env.get_observations()
        policy_obs = obs_td["policy"]                                   # (B, 31)

        if forecaster is not None and tau > 0:
            # Build forecaster context.
            attacker_w, action_h = history.get_window(tau)              # (B, τ+1, 18), (B, τ, 4)
            defender_d_18 = get_defender_delayed_world_18d(env)         # (B, 1, 18)
            attacker_b, defender_b = transform_to_body_batched(
                attacker_w, defender_d_18,
                pos_scale=pos_scale, vel_scale=vel_scale, ang_vel_scale=ang_vel_scale,
            )                                                           # (B, τ+1, 18), (B, 1, 6)
            ctx = {
                "attacker_history": attacker_b,
                "defender_delayed": defender_b,
                "action_history":   action_h,
            }
            pred = sample_via_euler(
                model=forecaster, context=ctx, tau=tau,
                num_steps=int(num_euler_steps), target_dim=6,
            )                                                           # (B, τ, 6) body+scaled
            # Stub-in last-step prediction into obs's defender slot.
            policy_obs = policy_obs.clone()
            policy_obs[:, 18:24] = pred[:, -1, :]

        # ---- Actor forward ----
        td = TensorDict({"policy": policy_obs}, batch_size=[num_envs], device=device)
        action = actor(td)                                              # (B, 4) raw mean
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

        # ---- Step ----
        per_env_step += 1
        _next_obs, _rew, done, _extras = env.step(action)

        # ---- Record terminations BEFORE env-side reset wipes per-env state ----
        # last_term snapshot survives reset_idx (env stores .clone()).
        alive_mask = torch.ones((num_envs,), device=device, dtype=torch.bool)  # all "alive" — done envs are valid
        term = _classify_terminations_batch(env, alive_mask)
        for key in counts:
            counts[key] += int(term[key].sum().item())

        # Episode length record for done envs
        done_idx_local = done.nonzero(as_tuple=False).reshape(-1)
        if done_idx_local.numel() > 0:
            episode_lengths.extend(per_env_step[done_idx_local].cpu().tolist())
            n_collected += int(done_idx_local.numel())
            per_env_step[done_idx_local] = 0

        # ---- Update history buffer (push current state + action, reset done envs) ----
        if forecaster is not None:
            # First push (newest = post-step state).
            current_state = get_attacker_world_state_batched(env)
            history.push(current_state, action)
            # Then refill done envs with the freshly-spawned state (idempotent overwrite).
            if done_idx_local.numel() > 0:
                history.reset_idx(done_idx_local, current_state[done_idx_local])

        step_idx += 1

    return {
        "counts": counts,
        "n_collected": n_collected,
        "episode_lengths": episode_lengths,
        "step_idx": step_idx,
    }


# ---------------------------------------------------------------------------
# Reporting + plotting
# ---------------------------------------------------------------------------

def _rates(stats: dict[str, Any]) -> dict[str, float]:
    n = max(1, stats["n_collected"])
    out = {f"{k}_rate": stats["counts"][k] / n for k in stats["counts"]}
    if stats["episode_lengths"]:
        out["mean_ep_length"] = float(np.mean(stats["episode_lengths"]))
    else:
        out["mean_ep_length"] = 0.0
    out["n_collected"] = stats["n_collected"]
    return out


def render_table(results: dict[int, dict[str, dict[str, Any]]]) -> str:
    """Plain-text table for console + report file."""
    rows = ["", "Forecaster on/off comparison — per-τ results"]
    rows.append("=" * 105)
    rows.append(f"{'τ':>4} | {'mode':<8} | {'mission':>8} | {'capture':>8} | "
                f"{'oob/crash':>9} | {'timeout':>8} | {'mean_len':>9} | {'n_eps':>6}")
    rows.append("-" * 105)
    for tau, by_mode in sorted(results.items()):
        for mode in ("no_fc", "with_fc"):
            r = _rates(by_mode[mode])
            rows.append(
                f"{tau:>4} | {mode:<8} | {r['mission_rate']:>8.3f} | "
                f"{r['capture_rate']:>8.3f} | {r['oob_crash_rate']:>9.3f} | "
                f"{r['timeout_rate']:>8.3f} | {r['mean_ep_length']:>9.1f} | "
                f"{r['n_collected']:>6}"
            )
        # Δ row (with - no)
        no = _rates(by_mode["no_fc"]); wf = _rates(by_mode["with_fc"])
        rows.append(
            f"{tau:>4} | {'Δ (w-n)':<8} | "
            f"{wf['mission_rate'] - no['mission_rate']:>+8.3f} | "
            f"{wf['capture_rate'] - no['capture_rate']:>+8.3f} | "
            f"{wf['oob_crash_rate'] - no['oob_crash_rate']:>+9.3f} | "
            f"{wf['timeout_rate'] - no['timeout_rate']:>+8.3f} | "
            f"{wf['mean_ep_length'] - no['mean_ep_length']:>+9.1f} |"
        )
        rows.append("-" * 105)
    return "\n".join(rows)


def plot_comparison(results: dict[int, dict[str, dict[str, Any]]], out_path: Path) -> None:
    taus = sorted(results.keys())
    metrics = ["mission_rate", "capture_rate", "oob_crash_rate", "timeout_rate"]
    titles = ["mission_success", "captured", "oob/crash", "timeout"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    width = 0.35
    x = np.arange(len(taus))
    for ax, metric, title in zip(axes, metrics, titles):
        no_vals = [_rates(results[t]["no_fc"])[metric] for t in taus]
        wf_vals = [_rates(results[t]["with_fc"])[metric] for t in taus]
        ax.bar(x - width / 2, no_vals, width, label="no FC", color="C3", alpha=0.85)
        ax.bar(x + width / 2, wf_vals, width, label="with FC", color="C0", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"τ={t}" for t in taus])
        ax.set_ylim(0.0, max(0.05, max(no_vals + wf_vals) * 1.15))
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if ax is axes[0]:
            ax.set_ylabel("rate")
            ax.legend(fontsize=8)
    fig.suptitle("Attacker termination rates: forecaster on vs off", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"[plot] saved {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    _seed_all(args.seed)
    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")
    device = gs.device

    # Resolve & load cfgs / ckpts
    cfgs_pkl = Path(args.cfgs_pkl).resolve()
    if not cfgs_pkl.is_file():
        raise SystemExit(f"cfgs_pkl not found: {cfgs_pkl}")
    with open(cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)

    attacker_ckpt = Path(args.attacker_ckpt).resolve()
    defender_ckpt = Path(args.defender_ckpt).resolve()
    forecaster_ckpt = Path(args.forecaster_ckpt).resolve()
    for p in (attacker_ckpt, defender_ckpt, forecaster_ckpt):
        if not p.is_file():
            raise SystemExit(f"ckpt not found: {p}")

    fc_data_dir = (
        Path(args.forecaster_data_dir) if args.forecaster_data_dir
        else ROOT / "data" / "forecaster_data_ams_v9_c8"
    )
    scales = load_scales_from_dataset(fc_data_dir)
    print(f"[eval-fc] scales (pos/vel/ang_vel) = {scales}", flush=True)

    # Defender (env-injected) is rebuilt per-τ since env shares it.
    from envs.pe_1v1.scripted_defender import PretrainedDefender   # noqa: PLC0415

    # Actor (raw torch.save'd module from train_attacker)
    actor = torch.load(str(attacker_ckpt), weights_only=False, map_location=device)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    print(f"[eval-fc] attacker loaded: {attacker_ckpt}", flush=True)

    # Forecaster
    forecaster = ForecasterModel(tau_max=int(args.tau_max)).to(device)
    payload = load_fc_ckpt(str(forecaster_ckpt), model=forecaster, optimizer=None,
                           map_location=device)
    forecaster.eval()
    for p in forecaster.parameters():
        p.requires_grad_(False)
    fc_step = payload.get("step", "?")
    print(f"[eval-fc] forecaster loaded: {forecaster_ckpt}  (step={fc_step})", flush=True)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict[str, dict[str, Any]]] = {}
    overall_t0 = time.time()
    for tau in args.tau_values:
        tau = int(tau)
        if tau < 1 or tau > int(args.tau_max):
            raise SystemExit(f"tau {tau} out of [1, {args.tau_max}]")
        results.setdefault(tau, {})
        for mode in ("no_fc", "with_fc"):
            print(f"\n[eval-fc] === τ={tau}  mode={mode} ===", flush=True)
            # Build defender + env fresh per cell so ckpt caches/seeds are clean.
            defender_policy = PretrainedDefender(
                num_envs=int(args.num_envs), ckpt_path=str(defender_ckpt), device=device,
            )
            env = build_env(cfgs, num_envs=int(args.num_envs),
                            defender_policy=defender_policy, tau_delay=tau)
            t0 = time.time()
            stats = run_eval(
                env=env, actor=actor, num_episodes=int(args.num_episodes),
                forecaster=(forecaster if mode == "with_fc" else None),
                tau=tau, scales=scales,
                num_euler_steps=int(args.num_euler_steps),
                max_steps_safety_factor=int(args.max_steps_safety_factor),
            )
            elapsed = time.time() - t0
            r = _rates(stats)
            print(
                f"  collected {stats['n_collected']} eps in {elapsed:.1f}s — "
                f"mission={r['mission_rate']:.3f}  capture={r['capture_rate']:.3f}  "
                f"oob/crash={r['oob_crash_rate']:.3f}  timeout={r['timeout_rate']:.3f}  "
                f"mean_len={r['mean_ep_length']:.1f}",
                flush=True,
            )
            results[tau][mode] = stats
            # Free env to release GPU memory between cells.
            del env, defender_policy

    table = render_table(results)
    print("\n" + table)
    (out_dir / "results.txt").write_text(table)
    (out_dir / "results.json").write_text(json.dumps({
        str(tau): {mode: {
            "rates": _rates(s), "counts": s["counts"], "n_collected": s["n_collected"],
        } for mode, s in by_mode.items()}
        for tau, by_mode in results.items()
    }, indent=2))
    plot_comparison(results, out_dir / "comparison.png")
    print(f"\n[eval-fc] DONE — total {time.time() - overall_t0:.1f}s. out={out_dir}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
