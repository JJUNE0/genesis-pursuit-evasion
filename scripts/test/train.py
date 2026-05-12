"""
통합 드론 RL 학습 스크립트 (Genesis AI + rsl-rl)

사용법:
    python scripts/train.py --task hover --drone nova
    python scripts/train.py --task hover --drone cf2x
    python scripts/train.py --task rates --drone nova -B 8192
    python scripts/train.py --task trajectory --drone nova -v --max_iterations 50
"""

import argparse
import importlib
import os
import pickle
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("rsl-rl-lib>=5.0.0 필요: pip install rsl-rl-lib") from e

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from utils.drone_params import DroneParams

TASK_CHOICES = ["hover", "trajectory", "rates", "racing", "tracking"]
DRONE_CHOICES = ["nova", "cf2x"]
ENV_CLASS_NAMES = {
    "hover": "HoverEnv",
    "trajectory": "TrajectoryEnv",
    "rates": "AttitudeRatesEnv",
    "racing": "RacingEnv",
    "tracking": "TrackingEnv",
}


def get_train_cfg(exp_name: str, task: str = "hover", visual: bool = False) -> dict:
    cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
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
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy", "critic_obs"],
        },
        "num_steps_per_env": 100,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "wandb",
        "wandb_project": "genesis-quadrotor",
    }

    # Tracking with dynamics encoder: add residual_hist obs group
    if task == "tracking" and visual:
        # Phase 5: CNN actor + MLP critic (asymmetric)
        cfg["actor"] = {
            "class_name": "CNNModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "cnn_cfg": {
                "visual_obs": {
                    "output_channels": [32, 64],
                    "kernel_size": [8, 4],
                    "stride": [4, 2],
                    "padding": "none",
                    "activation": "elu",
                    "global_pool": "avg",
                    "flatten": True,
                },
            },
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
        }
        cfg["obs_groups"] = {
            "actor": ["policy", "visual_obs", "residual_hist"],
            "critic": ["policy", "critic_task", "critic_obs", "residual_hist"],
        }
        cfg["algorithm"]["learning_rate"] = 1e-4
    elif task == "tracking":
        cfg["obs_groups"] = {
            "actor": ["policy", "residual_hist"],
            "critic": ["policy", "critic_obs", "residual_hist"],
        }

    return cfg


def load_task_components(task: str, drone: str):
    try:
        cfgs_module = importlib.import_module(f"envs.{task}.{drone}_cfgs")
    except ModuleNotFoundError as exc:
        if exc.name == f"envs.{task}.{drone}_cfgs":
            raise ValueError(f"지원하지 않는 task/drone 조합입니다: task={task}, drone={drone}") from exc
        raise

    env_module = importlib.import_module(f"envs.{task}.env")
    env_class = getattr(env_module, ENV_CLASS_NAMES[task])
    return env_class, cfgs_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hover", choices=TASK_CHOICES)
    parser.add_argument("--drone", type=str, default="nova", choices=DRONE_CHOICES)
    parser.add_argument("-e", "--exp_name", type=str, default=None)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=1001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logger", type=str, default="wandb", choices=["wandb", "tensorboard"])
    parser.add_argument("--visual", action="store_true", default=False,
                        help="Enable Phase 5 visual observation (tracking task only)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path for fine-tuning")
    args = parser.parse_args()

    exp_name = args.exp_name or f"{args.drone}-{args.task}"
    env_class, cfgs_module = load_task_components(args.task, args.drone)
    env_cfg, obs_cfg, reward_cfg, command_cfg = cfgs_module.get_cfgs()
    drone_config = cfgs_module.DRONE_CONFIG

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    params = DroneParams(drone_config)
    hover_rpm = float(np.sqrt(params.mass * params.gravity / (4 * params.kf)))
    print(
        f"[Train] task={args.task}  drone={params.name}  "
        f"hover_rpm={hover_rpm:.1f} RPM  num_envs={args.num_envs}"
    )

    # Phase 5: enable visual observation
    if args.visual and args.task == "tracking":
        env_cfg["use_visual_obs"] = True
        env_cfg["num_rendered_envs"] = args.num_envs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{exp_name}_{timestamp}"
    train_cfg = get_train_cfg(exp_name, task=args.task, visual=args.visual)
    train_cfg["logger"] = args.logger

    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, drone_config, args.task], f)

    env = env_class(
        num_envs=args.num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.resume:
        runner.load(args.resume)
        print(f"[Train] Resumed from checkpoint: {args.resume}")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
