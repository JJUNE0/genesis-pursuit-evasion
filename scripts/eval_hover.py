"""Evaluate trained Hover policy with Genesis viewer.

Usage:
    python scripts/eval_hover.py                       # latest run
    python scripts/eval_hover.py --run logs/hover_rate_v0_<ts>_<sha> --ckpt 200
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs                          # noqa: E402

from agents.ppo_runner import make_runner     # noqa: E402
from envs.hover.cfgs import DEFAULT_DRONE_CONFIG  # noqa: E402
from envs.hover.env import HoverEnv           # noqa: E402
from utils.drone_params import DroneParams    # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hover policy evaluation (viewer)")
    p.add_argument("--run", type=str, default=None,
                   help="log dir (default: latest hover_rate_v0_* under logs/)")
    p.add_argument("--ckpt", type=int, default=None,
                   help="ckpt iteration (default: latest model_*.pt)")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_viewer", action="store_true",
                   help="disable Genesis viewer (headless eval)")
    return p.parse_args()


def _find_latest_run(prefix: str = "hover_rate_v0") -> Path:
    logs = ROOT / "logs"
    if not logs.exists():
        raise FileNotFoundError(f"{logs} not found")
    runs = sorted([d for d in logs.iterdir() if d.is_dir() and d.name.startswith(prefix)],
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no run starting with '{prefix}' in {logs}")
    return runs[-1]


def _find_latest_ckpt(run_dir: Path) -> Path:
    cks = sorted(run_dir.glob("model_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    if not cks:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    return cks[-1]


def main() -> int:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    run_dir = Path(args.run) if args.run else _find_latest_run()
    ckpt = (run_dir / f"model_{args.ckpt}.pt") if args.ckpt is not None else _find_latest_ckpt(run_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")
    print(f"[eval_hover] run = {run_dir}")
    print(f"[eval_hover] ckpt = {ckpt}")

    cfgs_pkl = run_dir / "cfgs.pkl"
    with open(cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)
    env_cfg = cfgs["env_cfg"]
    obs_cfg = cfgs["obs_cfg"]
    reward_cfg = cfgs["reward_cfg"]
    command_cfg = cfgs["command_cfg"]
    train_cfg = cfgs["train_cfg"]
    drone_yaml = cfgs.get("drone_yaml") or str(ROOT / DEFAULT_DRONE_CONFIG)

    env_cfg = dict(env_cfg)
    env_cfg["visualize_target"] = True

    show_viewer = not args.no_viewer

    gs.init(backend=gs.gpu, seed=args.seed, logging_level="warning")
    params = DroneParams(drone_yaml)
    env = HoverEnv(
        num_envs=args.num_envs, params=params,
        env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
        command_cfg=command_cfg, show_viewer=show_viewer,
    )
    runner = make_runner(env, train_cfg, run_dir, device=str(env.device))
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=env.device)

    obs = env.reset()
    n_steps = env.max_episode_length
    print(f"[eval_hover] running {n_steps} steps "
          f"(episode_length_s={env.episode_length_s}s, dt={env.dt}s)")
    print(f"[eval_hover] hover RPM ≈ {(params.mass*params.gravity/4/params.kf)**0.5:.0f}, "
          f"max RPM = {(params.max_thrust_per_motor/params.kf)**0.5:.0f}")
    with torch.no_grad():
        for k in range(n_steps):
            actions = policy(obs)
            # snapshot _propellers_revs BEFORE step → delta = step의 RPM
            revs_before = env.drone._propellers_revs.detach().clone()
            obs, rew, done, extras = env.step(actions)
            if (k + 1) % 100 == 0:
                d = float(env.rel_pos[0].norm().item())
                # delta_revs = revs_after - revs_before. _propellers_revs는
                # `(rev + rpm) % (60/dt)=12000` modulo라 wrap 시 delta < 0.
                # 보정: delta < 0이면 + 12000.
                delta_revs = env.drone._propellers_revs - revs_before
                delta_revs = torch.where(delta_revs < 0, delta_revs + 12000.0, delta_revs)
                rpms_env0 = delta_revs[:, 0].cpu().numpy()
                # |ω| body angular velocity
                w = float(env.base_ang_vel[0].norm().item())
                print(f"  k={k+1:4d}  d={d:.3f}m  |ω|={w:5.2f}  "
                      f"action={env.actions[0].cpu().numpy().round(3)}  "
                      f"rpm={rpms_env0.round(0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
