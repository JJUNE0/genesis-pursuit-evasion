"""Phase 1.0 — defender pretraining vs random-walk attacker.

Produces ``logs/pe1v1_v0/defender_v0/<timestamp>/...`` checkpoints + a
project-rooted symlink ``logs/pe1v1_v0/defender_v0.ckpt`` pointing at the
final ckpt for PR-C consumption (TODO.md §1.4).

Acceptance gate: capture rate > 0.5 against random-walk attacker within
5k iters (TODO.md §1.4).  This script just runs PPO; capture rate is read
from wandb / tensorboard and reported in the PR body.

Usage::

    python scripts/train_defender.py \\
        --num_envs 1024 --max_iterations 5000 --seed 0 --logger wandb

CLAUDE.md §11: ``--max_iterations >= 100`` requires user pre-approval.
This script accepts the value as a CLI flag — running it yourself is the
approval signal.
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
from envs.pe_1v1.scripted_attacker import (   # noqa: E402
    PretrainedAttacker,
    PretrainedAttackerPool,
    RandomWalkAttacker,
    StationaryAttacker,
)
from utils.drone_params import DroneParams   # noqa: E402
from utils.print_summary import (   # noqa: E402
    print_pe_train_complete,
    print_pe_train_summary,
    print_resume_loaded,
)


ATTACKER_CHOICES = ("stationary", "random_walk", "pretrained", "pretrained_pool")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1.0 defender pretraining")
    p.add_argument("--attacker_mode", type=str, default="stationary",
                   choices=ATTACKER_CHOICES,
                   help="Frozen scripted attacker the defender trains against. "
                        "PR-D default = stationary (random spawn + hover). "
                        "random_walk is chaotic baseline kept for ablation only. "
                        "pretrained loads attacker_v*.pt (PR-N Phase 5 alternating).")
    p.add_argument("--attacker_ckpt", type=str, default="",
                   help="Required when --attacker_mode=pretrained: "
                        "path to attacker_v*.pt (file or directory).")
    p.add_argument("--attacker_ckpt_pool", type=str, nargs="+", default=[],
                   help="Required when --attacker_mode=pretrained_pool: K개 ckpt paths "
                        "(Phase D Fictitious Play, 2026-05-07).")
    p.add_argument("--current_best_attacker_ckpt", type=str, default="",
                   help="Phase F (2026-05-10) — pold mixing의 current best ckpt.")
    p.add_argument("--pold", type=float, default=1.0,
                   help="Phase F (AMSPB) — pool sampling probability. "
                        "1.0=모든 env가 pool, 0.5=AMSPB style mixing.")
    p.add_argument("--use_asymmetric_critic", action="store_true",
                   help="Phase E.1 (CLAUDE.md §7, 2026-05-08). Critic obs에 opponent "
                        "ground truth 추가 (vel+ang_vel+quat=10D). Actor obs는 그대로.")
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--max_iterations", type=int, default=5,
                   help="PPO iterations. Plan §10.2: smoke uses 5; full run uses 5000.")
    p.add_argument("--seed", type=int, required=True)
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
        default=str(ROOT / "configs" / "train" / "ppo_defender.yaml"),
    )
    p.add_argument(
        "--drone_yaml", type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    p.add_argument(
        "--logs_root", type=str,
        default=str(ROOT / "logs" / "pe1v1_v0" / "defender_v0"),
    )
    p.add_argument(
        "--alias_path", type=str,
        default=str(ROOT / "logs" / "pe1v1_v0" / "defender_v0.ckpt"),
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to model_*.pt (또는 .ckpt alias) to resume from. "
             "Loads actor + critic + optimizer via OnPolicyRunner.load(). "
             "Phase 5 AMS-DRL alternating에서 stage 별 ckpt fine-tune 용.",
    )
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    _seed_all(args.seed)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")
    device = "cuda" if (args.backend == "gpu" and torch.cuda.is_available()) else "cpu"

    raw_env = load_pe_1v1_cfg(args.env_yaml)
    env_cfg = make_env_cfg(
        raw_env, ego="defender",
        use_asymmetric_critic=args.use_asymmetric_critic,
    )
    obs_cfg = make_obs_cfg(raw_env)
    reward_cfg = make_reward_cfg(raw_env)
    command_cfg = make_command_cfg(raw_env)

    params = DroneParams(args.drone_yaml)
    if args.attacker_mode == "stationary":
        attacker = StationaryAttacker(num_envs=args.num_envs, device=gs.device)
    elif args.attacker_mode == "random_walk":
        attacker = RandomWalkAttacker(num_envs=args.num_envs, device=gs.device)
    elif args.attacker_mode == "pretrained":
        if not args.attacker_ckpt:
            raise SystemExit(
                "--attacker_mode pretrained requires --attacker_ckpt <path/to/attacker_v*.pt>."
            )
        attacker = PretrainedAttacker(
            num_envs=args.num_envs, ckpt_path=args.attacker_ckpt, device=gs.device,
        )
    elif args.attacker_mode == "pretrained_pool":
        if not args.attacker_ckpt_pool:
            raise SystemExit(
                "--attacker_mode pretrained_pool requires --attacker_ckpt_pool path1 [path2 ...] "
                "(Fictitious Play 다중 ckpt)."
            )
        attacker = PretrainedAttackerPool(
            num_envs=args.num_envs, ckpt_paths=list(args.attacker_ckpt_pool),
            device=gs.device, sample="uniform", seed=args.seed,
            current_best_ckpt=args.current_best_attacker_ckpt or None,
            pold=args.pold,
        )
    else:
        raise ValueError(f"Unknown attacker_mode: {args.attacker_mode!r}")
    print(f"[phase1.0] attacker_mode = {args.attacker_mode}")

    env = PursuitEvasion1v1Env(
        num_envs=args.num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        attacker_policy=attacker,
        show_viewer=args.show_viewer,
    )

    train_cfg = load_train_cfg(args.train_yaml)

    timestamp = int(time.time())
    run_name = make_run_name(f"phase1.0_defender_seed{args.seed}", timestamp=timestamp)
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
        ams_stage="phase1.0_defender",
    )

    print_pe_train_summary(
        title="⚙  PHASE 1.0 — DEFENDER PRETRAINING",
        ego="defender",
        opponent_label="attacker_mode",
        opponent_mode=args.attacker_mode,
        opponent_ckpt=None,
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

    # PR-D3 — fine-tune from existing ckpt (Phase 5 alternating에 활용).
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume ckpt not found: {resume_path}")
        runner.load(str(resume_path))
        print_resume_loaded(role="defender", path=resume_path)

    # Pre-compute alias path so intra-train snapshots can update it too.
    actor_alias = Path(args.alias_path).with_name("defender_v0.pt")

    # Hook frozen-actor snapshot onto rsl-rl's per-save_interval model_*.pt save.
    # OnPolicyRunner.learn() calls runner.save(path) every save_interval iters;
    # wrapping it lets us write defender_v0.pt at the same cadence WITHOUT
    # breaking the single learn() call (avoids fragmented wandb runs).
    _orig_runner_save = runner.save

    def _save_with_actor_snapshot(path, infos=None):
        _orig_runner_save(path, infos)
        save_frozen_actor_snapshot(runner, log_dir, "defender_v0.pt", actor_alias)

    runner.save = _save_with_actor_snapshot

    # Wall-clock-aware learning loop (CLAUDE.md §11).
    # Default behavior (no --iters_per_chunk): chunk = max_iterations so the
    # rsl-rl progress bar runs end-to-end. Set --iters_per_chunk explicitly
    # to enable wall-clock budget granularity.
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
        # capture polishing 시점 두 드론 contact으로 Newton solver NaN. 학습된
        # 결과 (latest model_*.pt) 는 디스크에 살아있으므로 alias 생성 후 정상
        # 종료 — 사용자 매번 재학습 안 하게.
        print(f"[NaN] genesis sim diverged: {e}", file=sys.stderr)
        print(f"[NaN] preserving last model_*.pt as alias and exiting.", file=sys.stderr)
        nan_failure = True

    # PR-N — ensure final model_<iter>.pt is on disk before downstream
    # orchestrator tries to resume from it (smoke runs with low max_iterations
    # would otherwise miss the save_interval boundary).
    if not nan_failure:
        save_final_ckpt(runner, log_dir)

    # Snapshot final ckpt + create stable alias for PR-C consumers.
    final = latest_model_pt(log_dir)
    if final is not None:
        alias_ckpt(final, args.alias_path)

    # Phase 1.5 (PR-C) consumes a frozen defender actor.  Intra-loop snapshots
    # already wrote defender_v0.pt at every save_interval; this re-save
    # guarantees a clean post-train ckpt (skipped if run crashed with NaN).
    saved = save_frozen_actor_snapshot(
        runner, log_dir, "defender_v0.pt", actor_alias,
    )
    actor_path = log_dir / "defender_v0.pt" if saved else None
    print_pe_train_complete(
        title="✓ DEFENDER TRAINING COMPLETE" if saved else
              "⚠ DEFENDER TRAINING — actor save skipped",
        final_model_pt=final,
        final_model_alias=Path(args.alias_path) if final is not None else None,
        actor_pt=actor_path,
        actor_alias=actor_alias if saved else None,
        skipped_reason="NaN/Inf in actor weights" if not saved else None,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
