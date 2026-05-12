"""S2 diagnostic: roll out an attacker/defender pair, record per-step trajectories,
and classify att_crash terminations into sub-causes (z_floor / pitch_max / roll_max
/ v_max / other).

Forked from scripts/eval_head_to_head.py — read-only env access, no env/config
changes (CLAUDE.md §1). Phase A baseline-vs-retrain comparison artifact.

Outputs (under --output_dir):
  trajectories.npz   — per-step (env, step) tensors: a_pos, a_vel, a_quat,
                       d_pos, dist_pair, done flags
  episodes.json      — per-episode summary: termination type, sub-cause,
                       episode_length, final state
  summary.json       — aggregate counts + sub-cause breakdown
  histogram.png      — sub-cause histogram + ep_length distribution
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from envs.pe_1v1.env import PursuitEvasion1v1Env   # noqa: E402
from envs.pe_1v1.scripted_attacker import PretrainedAttacker   # noqa: E402
from envs.pe_1v1.scripted_defender import PretrainedDefender   # noqa: E402
from utils.drone_params import DroneParams   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2 trajectory dump + att_crash sub-cause classifier")
    p.add_argument("--attacker_ckpt", type=str, required=True)
    p.add_argument("--defender_ckpt", type=str, required=True)
    p.add_argument("--cfgs_pkl", type=str, default="",
                   help="Default = cfgs.pkl next to attacker_ckpt.")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--drone_yaml", type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _quat_to_roll_pitch_deg(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q shape: (B, 4) [w, x, y, z]. Returns (roll_deg, pitch_deg) shape (B,)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    rad2deg = 180.0 / math.pi
    return roll * rad2deg, pitch * rad2deg


def classify_crash(
    a_pos: torch.Tensor, a_vel: torch.Tensor, a_quat: torch.Tensor,
    z_floor: float, pitch_max_deg: float, roll_max_deg: float, v_max: float,
) -> list[str]:
    """Return per-env sub-cause string. Priority matches env._drone_crash order:
    z_floor → pitch_max → roll_max → v_max → other.
    """
    z = a_pos[:, 2]
    v_norm = a_vel.norm(dim=1)
    roll_deg, pitch_deg = _quat_to_roll_pitch_deg(a_quat)
    out: list[str] = []
    for i in range(a_pos.shape[0]):
        if z[i].item() < z_floor:
            out.append("z_floor")
        elif abs(pitch_deg[i].item()) > pitch_max_deg:
            out.append("pitch_max")
        elif abs(roll_deg[i].item()) > roll_max_deg:
            out.append("roll_max")
        elif v_norm[i].item() > v_max:
            out.append("v_max")
        else:
            out.append("other")
    return out


@torch.no_grad()
def main() -> int:
    args = parse_args()
    _seed_all(args.seed)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    attacker_ckpt = Path(args.attacker_ckpt).resolve()
    defender_ckpt = Path(args.defender_ckpt).resolve()
    cfgs_pkl = (
        Path(args.cfgs_pkl).resolve()
        if args.cfgs_pkl
        else attacker_ckpt.parent / "cfgs.pkl"
    )
    if not cfgs_pkl.is_file():
        raise FileNotFoundError(f"cfgs.pkl not found: {cfgs_pkl}")
    with open(cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)

    env_cfg = dict(cfgs["env_cfg"])
    env_cfg["ego"] = "attacker"
    obs_cfg = cfgs["obs_cfg"]
    reward_cfg = cfgs["reward_cfg"]
    command_cfg = cfgs["command_cfg"]
    drone_yaml = cfgs.get("drone_yaml", args.drone_yaml)

    z_floor = float(env_cfg.get("z_floor", 0.1))
    pitch_max = float(env_cfg.get("pitch_max_deg", 60.0))
    roll_max = float(env_cfg.get("roll_max_deg", 60.0))
    v_max = float(env_cfg.get("v_max", 10.0))

    params = DroneParams(drone_yaml)
    defender = PretrainedDefender(
        num_envs=args.num_envs, ckpt_path=str(defender_ckpt), device=gs.device,
    )
    attacker = PretrainedAttacker(
        num_envs=args.num_envs, ckpt_path=str(attacker_ckpt), device=gs.device,
    )
    env = PursuitEvasion1v1Env(
        num_envs=args.num_envs, params=params,
        env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        defender_policy=defender, show_viewer=False,
    )

    target = int(args.n_episodes)
    counts = Counter()
    crash_subcause = Counter()
    episodes: list[dict] = []
    ep_step_count = torch.zeros(args.num_envs, dtype=torch.int32, device=gs.device)
    n_done_total = 0
    step_count = 0
    max_steps_safety = env.max_episode_length * (target // args.num_envs + 4)

    # Lightweight trajectory buffer: only the LATEST state per env per step.
    # Keep cap on memory: store (T_max, B, k) in CPU after each step.
    buf_a_pos: list[np.ndarray] = []
    buf_a_vel: list[np.ndarray] = []
    buf_d_pos: list[np.ndarray] = []
    buf_done: list[np.ndarray] = []

    env.reset()
    t0 = time.time()
    while n_done_total < target and step_count < max_steps_safety:
        ego_actions = attacker.step(
            ego_state=env._attacker_state_dict(),
            opponent_state=env._defender_state_dict(),
            g_mission=env.g_mission,
        )
        _obs, _rew, done, _extras = env.step(ego_actions)
        ep_step_count += 1

        # snapshot state AFTER step (includes terminal state for done envs)
        a_pos = env.attacker_pos.detach()
        a_vel = env.attacker_vel.detach()
        a_quat = env.attacker_quat.detach()
        d_pos = env.defender_pos.detach()
        buf_a_pos.append(a_pos.cpu().numpy().copy())
        buf_a_vel.append(a_vel.cpu().numpy().copy())
        buf_d_pos.append(d_pos.cpu().numpy().copy())
        buf_done.append(done.detach().cpu().numpy().copy())

        # Classify terminations from THIS step's last_term snapshot.
        done_idx = done.nonzero(as_tuple=False).reshape(-1)
        if done_idx.numel() > 0:
            sub = classify_crash(
                a_pos[done_idx], a_vel[done_idx], a_quat[done_idx],
                z_floor, pitch_max, roll_max, v_max,
            )
            for k in ("mission", "captured", "att_crash",
                      "def_crash", "att_oob", "def_oob", "timeout"):
                counts[k] += int(env.last_term[k].sum().item())

            for j, env_i in enumerate(done_idx.tolist()):
                term = "unknown"
                for k in ("mission", "captured", "att_crash",
                          "def_crash", "att_oob", "def_oob", "timeout"):
                    if bool(env.last_term[k][env_i].item()):
                        term = k
                        break
                episode = {
                    "env_id":     int(env_i),
                    "ep_length":  int(ep_step_count[env_i].item()),
                    "term":       term,
                    "subcause":   sub[j] if term == "att_crash" else "n/a",
                    "final_z":    float(a_pos[env_i, 2].item()),
                    "final_v":    float(a_vel[env_i].norm().item()),
                    "final_dist": float((a_pos[env_i] - d_pos[env_i]).norm().item()),
                }
                if term == "att_crash":
                    crash_subcause[sub[j]] += 1
                episodes.append(episode)

            ep_step_count[done_idx] = 0
            n_done_total += int(done.sum().item())
            attacker.reset(done_idx)

        step_count += 1

    wall_sec = time.time() - t0
    print(f"[dump] {n_done_total} episodes in {step_count} steps ({wall_sec:.1f}s)")

    # Save artifacts
    np.savez_compressed(
        out_dir / "trajectories.npz",
        a_pos=np.stack(buf_a_pos),
        a_vel=np.stack(buf_a_vel),
        d_pos=np.stack(buf_d_pos),
        done=np.stack(buf_done),
    )
    with open(out_dir / "episodes.json", "w") as f:
        json.dump(episodes[:target], f, indent=2)

    summary = {
        "n_episodes":      n_done_total,
        "wall_sec":        round(wall_sec, 2),
        "attacker_ckpt":   str(attacker_ckpt),
        "defender_ckpt":   str(defender_ckpt),
        "cfgs_pkl":        str(cfgs_pkl),
        "termination":     dict(counts),
        "att_crash_subcause": dict(crash_subcause),
        "att_crash_total": counts.get("att_crash", 0),
    }
    if counts.get("att_crash", 0) > 0:
        summary["att_crash_subcause_pct"] = {
            k: round(100.0 * v / counts["att_crash"], 1)
            for k, v in crash_subcause.items()
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Histogram (att_crash subcause + ep_length)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        if crash_subcause:
            keys = list(crash_subcause.keys())
            vals = [crash_subcause[k] for k in keys]
            axes[0].bar(keys, vals, color="tab:red", alpha=0.7)
            axes[0].set_title(f"att_crash sub-cause (n={sum(vals)})")
            axes[0].set_ylabel("count")
        else:
            axes[0].text(0.5, 0.5, "no att_crash", ha="center")
        ep_lens = [e["ep_length"] for e in episodes]
        if ep_lens:
            axes[1].hist(ep_lens, bins=30, color="tab:blue", alpha=0.7)
            axes[1].set_title(f"episode length (n={len(ep_lens)}, mean={np.mean(ep_lens):.0f})")
            axes[1].set_xlabel("steps")
        fig.tight_layout()
        fig.savefig(out_dir / "histogram.png", dpi=120)
        print(f"[dump] saved {out_dir / 'histogram.png'}")
    except Exception as e:
        print(f"[dump] plot skipped: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
