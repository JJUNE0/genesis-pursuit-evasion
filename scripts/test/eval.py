"""
Policy evaluation script.

Usage:
    python scripts/eval.py --task hover --drone nova --ckpt 300
    python scripts/eval.py -e my-exp --task hover --drone cf2x --ckpt 300
    python scripts/eval.py --task rates --drone nova --ckpt 300 --record
"""

import argparse
import importlib
import os
import pickle
import sys
from importlib import metadata

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("rsl-rl-lib>=5.0.0 필요: pip install rsl-rl-lib") from e

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from utils.drone_params import DroneParams

TASK_CHOICES = ["hover", "trajectory", "rates", "racing", "tracking"]
ENV_CLASS_NAMES = {
    "hover": "HoverEnv",
    "trajectory": "TrajectoryEnv",
    "rates": "AttitudeRatesEnv",
    "racing": "RacingEnv",
    "tracking": "TrackingEnv",
}
DRONE_CONFIGS = {
    "nova": "configs/nova/drone_params.yaml",
    "cf2x": "configs/cf2x/drone_params.yaml",
}


def load_saved_cfgs(cfg_path: str):
    with open(cfg_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"지원하지 않는 cfgs.pkl 형식입니다: {type(payload)!r}")

    if len(payload) == 7:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, drone_config, task = payload
    elif len(payload) == 6:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, drone_config = payload
        task = None
    elif len(payload) == 5:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = payload
        drone_config = None
        task = None
    else:
        raise ValueError(f"지원하지 않는 cfgs.pkl 길이입니다: {len(payload)}")

    import pprint
    _pp = pprint.PrettyPrinter(indent=2, width=80)
    print("\n========== Loaded cfgs ==========")
    for name, cfg in [
        ("task",        task),
        ("drone_config",drone_config),
        ("env_cfg",     env_cfg),
        ("obs_cfg",     obs_cfg),
        ("reward_cfg",  reward_cfg),
        ("command_cfg", command_cfg),
        ("train_cfg",   train_cfg),
    ]:
        print(f"\n--- {name} ---")
        _pp.pprint(cfg)
    print("=================================\n")

    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, drone_config, task


def load_env_class(task: str):
    env_module = importlib.import_module(f"envs.{task}.env")
    return getattr(env_module, ENV_CLASS_NAMES[task])


def resolve_task(cli_task: str | None, saved_task: str | None, exp_name: str | None) -> str:
    if cli_task is not None:
        return cli_task
    if saved_task is not None:
        return saved_task
    if exp_name is not None:
        suffix = exp_name.rsplit("-", maxsplit=1)[-1]
        if suffix in TASK_CHOICES:
            return suffix
    return "hover"


def resolve_exp_name(exp_name: str | None, cli_drone: str | None, task: str | None) -> str:
    if exp_name is not None:
        return exp_name
    if cli_drone is None or task is None:
        raise ValueError("--exp_name 없이 실행하려면 --task 와 --drone 을 함께 지정해야 합니다.")
    return f"{cli_drone}-{task}"


def resolve_drone_config(exp_name: str, cli_drone: str | None, saved_drone_config: str | None) -> str:
    if saved_drone_config is not None:
        return saved_drone_config

    if cli_drone is not None:
        return DRONE_CONFIGS[cli_drone]

    for drone_name, config_path in DRONE_CONFIGS.items():
        if exp_name.startswith(f"{drone_name}-"):
            return config_path

    raise ValueError(
        "cfgs.pkl에 drone_config가 저장되어 있지 않습니다. "
        "기존 실험 로그를 평가하려면 --drone {nova,cf2x}를 명시하세요."
    )


def pure_pursuit_policy(env):
    """Scripted pure-pursuit: body-rate commands toward target current position."""
    rel_pos = env.rel_pos_body                                       # (B, 3)
    dist = rel_pos.norm(dim=1, keepdim=True).clamp_min(1e-6)
    direction = rel_pos / dist

    gain_xy, gain_z = 0.5, 0.3
    actions = torch.zeros((env.num_envs, 4), device=env.device)
    actions[:, 0] = direction[:, 2] * gain_z       # collective
    actions[:, 1] = direction[:, 0] * gain_xy      # pitch rate
    actions[:, 2] = -direction[:, 1] * gain_xy     # roll rate
    # yaw rate = 0
    return actions.clamp(-1, 1)


def run_and_log(env, policy_fn, max_steps, is_nn=True):
    """Run a policy and return (ego_traj, tgt_traj) for env 0."""
    obs_dict = env.reset()
    ego_traj, tgt_traj = [], []
    with torch.no_grad():
        for _ in range(max_steps):
            if is_nn:
                actions = policy_fn(obs_dict)
            else:
                actions = policy_fn(env)
                # If action chunking is on, tile to match expected dim
                if hasattr(env, "use_action_chunking") and env.use_action_chunking:
                    actions = actions.repeat(1, env.action_chunk_K)
            obs_dict, _, _, _ = env.step(actions)
            ego_traj.append(env.base_pos[0].cpu().clone())
            tgt_traj.append(env.tgt_pos[0].cpu().clone())
    return torch.stack(ego_traj), torch.stack(tgt_traj)


def plot_pursuit_comparison(rl_ego, rl_tgt, pp_ego, pp_tgt, save_path):
    """Bird's-eye comparison of RL vs pure-pursuit trajectories."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(rl_tgt[:, 0], rl_tgt[:, 1], "r--", label="Target", linewidth=1)
    ax.plot(rl_ego[:, 0], rl_ego[:, 1], "b-", label="RL (lead pursuit)", linewidth=1.5)
    ax.plot(pp_ego[:, 0], pp_ego[:, 1], "g-", label="Pure pursuit", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Bird's-Eye View")
    ax.legend()
    ax.set_aspect("equal")

    rl_len = torch.norm(rl_ego[1:] - rl_ego[:-1], dim=1).sum().item()
    pp_len = torch.norm(pp_ego[1:] - pp_ego[:-1], dim=1).sum().item()
    ratio = rl_len / pp_len if pp_len > 0 else 0
    axes[1].bar(["RL Agent", "Pure Pursuit"], [rl_len, pp_len], color=["#4A90D9", "#6BBF6B"])
    axes[1].set_ylabel("Path Length [m]")
    axes[1].set_title(f"Corner-Cutting Ratio: {ratio:.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Eval] Pursuit comparison saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default=None)
    parser.add_argument("--task", type=str, default=None, choices=TASK_CHOICES)
    parser.add_argument("-B", "--num_envs", type=int, default=1)
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="타임스탬프 폴더명 (예: 20260409_154744). 미지정 시 최신 run 자동 선택",
    )
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--drone", type=str, default=None, choices=sorted(DRONE_CONFIGS))
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--compare_pursuit", action="store_true", default=False,
                        help="Compare RL vs pure-pursuit baseline (tracking task only)")
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--vis", action="store_true", default=False,
                        help="Enable Phase 5 visual observation (tracking task only)")
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    resolved_exp_name = resolve_exp_name(args.exp_name, args.drone, args.task)
    base_dir = "logs"
    if args.run:
        log_dir = os.path.join(base_dir, args.run)
    else:
        runs = sorted(
            [
                d
                for d in (os.listdir(base_dir) if os.path.isdir(base_dir) else [])
                if d.startswith(f"{resolved_exp_name}_")
                and os.path.isdir(os.path.join(base_dir, d))
                and os.path.exists(os.path.join(base_dir, d, "cfgs.pkl"))
            ],
            reverse=True,
        )
        if not runs:
            raise FileNotFoundError(f"학습 로그를 찾을 수 없습니다: {base_dir}/{resolved_exp_name}_*")
        log_dir = os.path.join(base_dir, runs[0])
        print(f"[Eval] 최신 run 선택: {log_dir}")

    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {cfg_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {model_path}")

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, saved_drone_config, saved_task = load_saved_cfgs(cfg_path)
    task = resolve_task(args.task, saved_task, resolved_exp_name)
    env_class = load_env_class(task)
    drone_config = resolve_drone_config(resolved_exp_name, args.drone, saved_drone_config)

    params = DroneParams(drone_config)

    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = {}
    env_cfg = dict(env_cfg)
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60

    if args.vis and task == "tracking":
        env_cfg["use_visual_obs"] = True
        env_cfg["num_rendered_envs"] = args.num_envs

    env = env_class(
        num_envs=args.num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    max_sim_step = env.max_episode_length
    video_path = args.video_path or os.path.join(log_dir, f"eval_ckpt_{args.ckpt}.mp4")

    if args.compare_pursuit and task == "tracking":
        print("[Eval] Running RL policy...")
        rl_ego, rl_tgt = run_and_log(env, policy, max_sim_step, is_nn=True)
        print("[Eval] Running pure-pursuit baseline...")
        pp_ego, pp_tgt = run_and_log(env, pure_pursuit_policy, max_sim_step, is_nn=False)
        plot_path = os.path.join(log_dir, f"pursuit_comparison_ckpt{args.ckpt}.png")
        plot_pursuit_comparison(rl_ego, rl_tgt, pp_ego, pp_tgt, plot_path)
        return

    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, _, _, _ = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(
                save_to_filename=video_path,
                fps=env_cfg["max_visualize_FPS"],
            )
        else:
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
