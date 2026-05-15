"""Phase 1.5 — attacker training against a chosen defender.

Defender modes (TODO.md §1.3):

- ``stationary``  — hover-in-place baseline (easiest curriculum).
- ``random_walk`` — uniform [-1,1]^4 RPM jitter (chaotic baseline).
- ``pretrained`` — load Phase 1.0 산출물 ``actor_v0.pt`` via ``PretrainedDefender``.

Acceptance gates (TODO.md §1.5):

- stationary defender attacker winrate **> 0.9**  (5 seed avg, ≤ 5k iter)
- pretrained defender_v0 attacker winrate **> 0.5** (5 seed avg, ≤ 10k iter)
- random_walk defender winrate is reported only (not a gate).

CLAUDE.md §11: ``--max_iterations >= 100`` is a user-approved action.
Smoke runs default to 5 iter so this script is safe to invoke from CI/QA.

Usage::

    python scripts/train_attacker.py \\
        --defender stationary --num_envs 1024 --max_iterations 5000 --seed 0
    python scripts/train_attacker.py \\
        --defender pretrained --defender_ckpt logs/pe1v1_v0/actor_v0.pt \\
        --num_envs 1024 --max_iterations 10000 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from agents.ppo_runner import (   # noqa: E402
    alias_ckpt,
    dump_cfgs_pkl,
    latest_model_pt,
    load_train_cfg,
    make_run_name,
    make_runner,
    save_final_ckpt,
    save_frozen_actor_snapshot,
)
from envs.pe_1v1.cfgs import (   # noqa: E402
    load_pe_1v1_cfg,
    make_command_cfg,
    make_env_cfg,
    make_obs_cfg,
    make_reward_cfg,
)
from envs.pe_1v1.env import PursuitEvasion1v1Env   # noqa: E402
from envs.pe_1v1.scripted_defender import (   # noqa: E402
    ChaseDefender,
    PretrainedDefender,
    PretrainedDefenderPool,
    RandomWalkDefender,
    StationaryDefender,
)
from utils.drone_params import DroneParams   # noqa: E402
from utils.print_summary import (   # noqa: E402
    print_pe_train_complete,
    print_pe_train_summary,
    print_resume_loaded,
)


DEFENDER_CHOICES = ("stationary", "random_walk", "chase", "pretrained", "pretrained_pool")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1.5 attacker training")
    p.add_argument("--defender", type=str, required=True, choices=DEFENDER_CHOICES,
                   help="Scripted/loaded defender to train against (TODO §1.3).")
    p.add_argument("--defender_ckpt", type=str, default="",
                   help="Required when --defender=pretrained: path to actor_v0.pt.")
    p.add_argument("--defender_ckpt_pool", type=str, nargs="+", default=[],
                   help="Required when --defender=pretrained_pool: K개 ckpt paths "
                        "(Phase D Fictitious Play, 2026-05-07). 각 env가 random "
                        "ckpt에 매핑됨 — 1개 ckpt에 over-fit 못함.")
    p.add_argument("--current_best_defender_ckpt", type=str, default="",
                   help="Phase F (2026-05-10) — pold mixing의 current best ckpt. "
                        "pretrained_pool 모드에서 pool과 별개로 current best 추가. "
                        "(1-pold) 확률로 사용됨.")
    p.add_argument("--pold", type=float, default=1.0,
                   help="Phase F (AMSPB) — pool sampling probability. "
                        "1.0=모든 env가 pool (Phase D 기존), 0.5=AMSPB style mixing. "
                        "current_best_defender_ckpt와 함께 사용.")
    p.add_argument("--use_asymmetric_critic", action="store_true",
                   help="Phase E.1 (CLAUDE.md §7, 2026-05-08). Critic obs에 opponent "
                        "ground truth 추가 (vel+ang_vel+quat=10D). Actor obs는 그대로. "
                        "기본 비활성 (v16 backward compat).")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--max_iterations", type=int, default=5,
                   help="PPO iterations. Plan §10.2: smoke uses 5; stationary≤5k, pretrained≤10k.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--deterministic", action="store_true",
                   help="GPU deterministic algorithms (paper-grade reproducibility). "
                        "Default OFF — 학습 빠름. ON 시 cudnn cache 비활성으로 spawn "
                        "첫 forward 느려질 수 있음.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Override run name (wandb + logs_root subdir). 사용 시 "
                        "timestamp/git_sha 안 붙음. None이면 기존 동작 (phase1.5_attacker_"
                        "<defender>_seed<N>_<ts>_<sha>).")
    p.add_argument("--minimal_log", action="store_true",
                   help="터미널에 'Mean episode {*}' extras print skip. wandb/tensorboard "
                        "기록은 그대로. 깔끔한 console view용.")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="Override wandb project name. None이면 train_yaml의 default 사용.")
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--logger", type=str, default="wandb", choices=["wandb", "tensorboard"])
    p.add_argument("--show_viewer", action="store_true")
    p.add_argument("--wall_clock_budget_min", type=float, default=0.0,
                   help="0 = no budget. Otherwise auto-ckpt + stop after this many minutes.")
    p.add_argument("--iters_per_chunk", type=int, default=None,
                   help="Wall-clock guard granularity. Default = max_iterations "
                        "(no chunking — progress bar runs end-to-end).")
    p.add_argument(
        "--env_yaml", type=str,
        default=str(ROOT / "configs" / "envs" / "pe_1v1_default.yaml"),
    )
    p.add_argument(
        "--train_yaml", type=str,
        default=str(ROOT / "configs" / "train" / "ppo_attacker.yaml"),
    )
    p.add_argument(
        "--drone_yaml", type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    p.add_argument(
        "--logs_root", type=str,
        default=str(ROOT / "logs" / "pe1v1_v0" / "attacker_v0"),
    )
    p.add_argument(
        "--alias_path", type=str,
        default="",
        help="Optional symlink path for the final attacker ckpt; "
             "default = logs_root/../attacker_v0_<defender>.ckpt",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to model_*.pt (또는 .ckpt alias) to resume from. "
             "Loads actor + critic + optimizer via OnPolicyRunner.load(). "
             "Use case: stationary attacker ckpt를 시작점으로 phase 1.5 "
             "fine-tune (pretrained defender 환경).",
    )
    return p.parse_args()


def _seed_all(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 2026-05-13 — opt-in GPU determinism (사용자 spec). default OFF (학습 속도 우선,
    # 기존 ams_v6_highspawn 동작과 동일). ON 시 cudnn cache 비활성 → spawn 첫 forward
    # 느려질 수 있음. paper-grade reproducibility 필요할 때만.
    if deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_defender(
    mode: str, num_envs: int, defender_ckpt: str, device,
    defender_ckpt_pool: list[str] | None = None, seed: int = 0,
    current_best_ckpt: str | None = None, pold: float = 1.0,
):
    if mode == "stationary":
        return StationaryDefender(num_envs=num_envs, device=device)
    if mode == "random_walk":
        return RandomWalkDefender(num_envs=num_envs, device=device)
    if mode == "chase":
        return ChaseDefender(num_envs=num_envs, device=device)
    if mode == "pretrained":
        if not defender_ckpt:
            raise SystemExit(
                "--defender pretrained requires --defender_ckpt <path/to/actor_v0.pt>."
            )
        return PretrainedDefender(
            num_envs=num_envs, ckpt_path=defender_ckpt, device=device,
        )
    if mode == "pretrained_pool":
        if not defender_ckpt_pool:
            raise SystemExit(
                "--defender pretrained_pool requires --defender_ckpt_pool path1 [path2 ...] "
                "(Fictitious Play 다중 ckpt)."
            )
        return PretrainedDefenderPool(
            num_envs=num_envs, ckpt_paths=list(defender_ckpt_pool),
            device=device, sample="uniform", seed=seed,
            current_best_ckpt=current_best_ckpt or None,
            pold=pold,
        )
    raise ValueError(f"Unknown defender mode: {mode!r}")


def main() -> int:
    args = parse_args()
    _seed_all(args.seed, deterministic=args.deterministic)
    if args.minimal_log:
        from agents.ppo_runner import enable_minimal_terminal_log
        enable_minimal_terminal_log()

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")
    device = "cuda" if (args.backend == "gpu" and torch.cuda.is_available()) else "cpu"

    raw_env = load_pe_1v1_cfg(args.env_yaml)
    env_cfg = make_env_cfg(
        raw_env, ego="attacker",
        use_asymmetric_critic=args.use_asymmetric_critic,
    )
    obs_cfg = make_obs_cfg(raw_env)
    reward_cfg = make_reward_cfg(raw_env)
    command_cfg = make_command_cfg(raw_env)

    params = DroneParams(args.drone_yaml)
    defender = build_defender(
        args.defender, args.num_envs, args.defender_ckpt, gs.device,
        defender_ckpt_pool=args.defender_ckpt_pool, seed=args.seed,
        current_best_ckpt=args.current_best_defender_ckpt or None,
        pold=args.pold,
    )

    env = PursuitEvasion1v1Env(
        num_envs=args.num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        defender_policy=defender,
        show_viewer=args.show_viewer,
    )

    train_cfg = load_train_cfg(args.train_yaml)
    # 2026-05-13 — cli --wandb_project가 train_yaml의 default를 override.
    if args.wandb_project:
        train_cfg["wandb_project"] = str(args.wandb_project)

    timestamp = int(time.time())
    if args.run_name:
        run_name = args.run_name   # 2026-05-13 — override (timestamp/sha 없이 깔끔)
    else:
        run_name = make_run_name(
            f"phase1.5_attacker_{args.defender}_seed{args.seed}", timestamp=timestamp,
        )
    log_dir = Path(args.logs_root) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    dump_cfgs_pkl(
        log_dir=log_dir,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        train_cfg=train_cfg,
        drone_yaml=args.drone_yaml,
        ams_stage=f"phase1.5_attacker_{args.defender}",
    )

    print_pe_train_summary(
        title="⚙  PHASE 1.5 — ATTACKER TRAINING",
        ego="attacker",
        opponent_label="defender",
        opponent_mode=args.defender,
        opponent_ckpt=args.defender_ckpt or None,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        seed=args.seed,
        backend=args.backend,
        device=device,
        logger=args.logger,
        show_viewer=args.show_viewer,
        wall_clock_budget_min=args.wall_clock_budget_min,
        iters_per_chunk=args.iters_per_chunk,
        resume=args.resume,
        run_name=run_name,
        log_dir=log_dir,
        drone_yaml=args.drone_yaml,
        env_yaml=args.env_yaml,
        train_yaml=args.train_yaml,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    runner = make_runner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=device,
        run_name=run_name,
        logger=args.logger,
    )

    # PR-D3 — fine-tune from existing ckpt (e.g., stationary → pretrained).
    # OnPolicyRunner.load() restores actor + critic + optimizer + iter counter.
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume ckpt not found: {resume_path}")
        runner.load(str(resume_path))
        print_resume_loaded(role="attacker", path=resume_path)

    # Pre-compute alias paths so intra-train snapshots can update them too.
    alias_path = Path(args.alias_path) if args.alias_path else (
        Path(args.logs_root).parent / f"attacker_v0_{args.defender}.ckpt"
    )
    actor_alias = alias_path.with_name(f"attacker_v0_{args.defender}.pt")

    # Hook frozen-actor snapshot onto rsl-rl's per-save_interval model_*.pt save.
    # OnPolicyRunner.learn() calls runner.save(path) every save_interval iters;
    # wrapping it lets us write attacker_v0.pt at the same cadence WITHOUT
    # breaking the single learn() call (avoids fragmented wandb runs).
    _orig_runner_save = runner.save

    def _save_with_actor_snapshot(path, infos=None):
        _orig_runner_save(path, infos)
        save_frozen_actor_snapshot(runner, log_dir, "attacker_v0.pt", actor_alias)

    runner.save = _save_with_actor_snapshot

    # Default (no --iters_per_chunk): chunk = max_iterations.
    # Set explicitly to enable wall-clock budget granularity.
    budget_sec = args.wall_clock_budget_min * 60.0 if args.wall_clock_budget_min > 0 else None
    chunk_arg = args.iters_per_chunk if args.iters_per_chunk is not None else args.max_iterations
    chunk = max(1, int(chunk_arg))
    remaining = int(args.max_iterations)
    start = time.time()
    # resume 시 init_at_random_ep_len=False — 이미 학습된 정책이라 random start 의미 없음.
    first = (args.resume is None)
    nan_failure = False
    try:
        while remaining > 0:
            n = min(chunk, remaining)
            runner.learn(num_learning_iterations=n, init_at_random_ep_len=first)
            first = False
            remaining -= n
            if budget_sec is not None and (time.time() - start) > budget_sec:
                print(f"[budget] wall_clock_budget_min={args.wall_clock_budget_min} exceeded; stopping.")
                break
    except gs.GenesisException as e:
        print(f"[NaN] genesis sim diverged: {e}", file=sys.stderr)
        print(f"[NaN] preserving last model_*.pt as alias and exiting.", file=sys.stderr)
        nan_failure = True

    # PR-N — ensure final model_<iter>.pt is on disk before downstream
    # orchestrator tries to resume from it (smoke runs with low max_iterations
    # would otherwise miss the save_interval boundary).
    if not nan_failure:
        save_final_ckpt(runner, log_dir)

    final = latest_model_pt(log_dir)
    if final is not None:
        alias_ckpt(final, alias_path)

    # Final frozen-actor save. Intra-loop snapshots already wrote attacker_v0.pt
    # at every save_interval; this re-save guarantees a clean post-train ckpt
    # (and is skipped if the run crashed with NaN weights).
    saved = save_frozen_actor_snapshot(
        runner, log_dir, "attacker_v0.pt", actor_alias,
    )
    actor_path = log_dir / "attacker_v0.pt" if saved else None
    print_pe_train_complete(
        title="✓ ATTACKER TRAINING COMPLETE" if saved else
              "⚠ ATTACKER TRAINING — actor save skipped",
        final_model_pt=final,
        final_model_alias=alias_path if final is not None else None,
        actor_pt=actor_path,
        actor_alias=actor_alias if saved else None,
        skipped_reason="NaN/Inf in actor weights" if not saved else None,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
