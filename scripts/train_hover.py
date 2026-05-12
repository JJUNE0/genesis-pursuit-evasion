"""Train HoverEnv (single drone + rate action) via rsl-rl PPO.

Usage:
    python scripts/train_hover.py --num_envs 4096 --max_iterations 500
    python scripts/train_hover.py --action_mode rpm   # baseline 비교

목적: PR-K 인프라 위에서 BodyRateController가 RL 학습 가능한지 검증.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs                             # noqa: E402

from agents.ppo_runner import (                  # noqa: E402
    dump_cfgs_pkl, latest_model_pt, make_run_name, make_runner,
)
from envs.hover.cfgs import DEFAULT_DRONE_CONFIG, get_cfgs   # noqa: E402
from envs.hover.env import HoverEnv              # noqa: E402
from utils.drone_params import DroneParams       # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hover RL training (rate-mode action)")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp_name", type=str, default="hover_rate_v0")
    p.add_argument("--logger", type=str, default="tensorboard",
                   choices=["wandb", "tensorboard"])
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--show_viewer", action="store_true", default=False)
    p.add_argument("--action_mode", type=str, default=None, choices=["rate", "rpm"],
                   help="override env_cfg.action_mode")
    p.add_argument("--drone_yaml", type=str, default=str(ROOT / DEFAULT_DRONE_CONFIG))
    p.add_argument("--train_yaml", type=str,
                   default=str(ROOT / "configs" / "train" / "ppo_hover.yaml"),
                   help="ppo train cfg yaml")
    p.add_argument("--resume", type=str, default=None,
                   help="ckpt path to resume from (file or log dir)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Reproducibility (CLAUDE.md §10)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning",
            performance_mode=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    if args.action_mode is not None:
        env_cfg["action_mode"] = args.action_mode

    params = DroneParams(args.drone_yaml)

    print(f"[train_hover] num_envs={args.num_envs} max_iter={args.max_iterations} "
          f"action_mode={env_cfg['action_mode']}")

    env = HoverEnv(
        num_envs=args.num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )

    # train cfg from yaml; fallback to inline default if file 미존재
    train_cfg = _load_train_cfg(args.train_yaml)

    run_name = make_run_name(args.exp_name)
    log_dir = ROOT / "logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_hover] log_dir = {log_dir}")

    dump_cfgs_pkl(
        log_dir=log_dir,
        env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
        command_cfg=command_cfg, train_cfg=train_cfg,
        drone_yaml=args.drone_yaml,
        ams_stage="hover_rate_v0",
    )

    runner = make_runner(
        env=env, train_cfg=train_cfg, log_dir=log_dir,
        device=str(env.device), run_name=run_name, logger=args.logger,
    )

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_dir():
            found = latest_model_pt(ckpt_path)
            if found is None:
                raise FileNotFoundError(f"no model_*.pt in {ckpt_path}")
            ckpt_path = found
        print(f"[train_hover] resuming from {ckpt_path}")
        runner.load(str(ckpt_path))

    runner.learn(num_learning_iterations=args.max_iterations,
                 init_at_random_ep_len=True)
    return 0


def _load_train_cfg(path: str) -> dict:
    p = Path(path)
    if p.is_file():
        import yaml
        with open(p) as f:
            return yaml.safe_load(f)
    # inline default — 사용자가 yaml 따로 안 만들었을 때 fallback
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3.0e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 32,
        "save_interval": 50,
        "logger": "tensorboard",
        "wandb_project": "genesis-quadrotor",
    }


if __name__ == "__main__":
    sys.exit(main())
