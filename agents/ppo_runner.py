"""Thin wrapper around ``rsl_rl.runners.OnPolicyRunner`` (rsl-rl >= 5.0).

Phase 1.0 (defender pretraining) and Phase 1.5 (attacker training) both
spin up a *single* ``OnPolicyRunner`` — no two-runner alternating yet
(that's Phase 5's AMS-DRL orchestrator).

Responsibilities collected here so the train scripts stay tiny:

- Load PPO train cfg from YAML and merge runtime overrides (run_name, logger).
- Construct the ``OnPolicyRunner`` and pin ``cfg["obs_groups"]`` from the env.
- Dump ``cfgs.pkl`` so eval can reconstruct env + runner from artifact alone
  (CLAUDE.md §10 reproducibility contract).
- Build wandb-style run names ``<exp>_<timestamp>_<git_sha>(-dirty)``.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from rsl_rl.runners import OnPolicyRunner

from utils.cfg_dict import CfgDict


def load_train_cfg(yaml_path: str | Path) -> dict[str, Any]:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _git_sha(dirty_suffix: bool = True) -> str:
    """Return short git sha + ``-dirty`` if working tree has changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if dirty and dirty_suffix:
            sha = f"{sha}-dirty"
    except Exception:
        pass
    return sha


def make_run_name(exp: str, timestamp: int | None = None) -> str:
    ts = int(timestamp) if timestamp is not None else int(time.time())
    return f"{exp}_{ts}_{_git_sha()}"


def dump_cfgs_pkl(
    log_dir: str | Path,
    env_cfg: dict[str, Any],
    obs_cfg: dict[str, Any],
    reward_cfg: dict[str, Any],
    command_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    drone_yaml: str | Path,
    ams_stage: str,
) -> Path:
    """Persist the full cfg bundle for reproducible eval (CLAUDE.md §10)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "cfgs.pkl"
    with open(out, "wb") as f:
        pickle.dump(
            {
                "env_cfg": env_cfg,
                "obs_cfg": obs_cfg,
                "reward_cfg": reward_cfg,
                "command_cfg": command_cfg,
                "train_cfg": train_cfg,
                "drone_yaml": str(drone_yaml),
                "ams_stage": ams_stage,
            },
            f,
        )
    return out


def make_runner(
    env: Any,
    train_cfg: dict[str, Any],
    log_dir: str | Path,
    device: str,
    *,
    run_name: str | None = None,
    logger: str | None = None,
) -> OnPolicyRunner:
    """Build OnPolicyRunner. Inserts env's obs_groups into train_cfg.

    Wraps the cfg in ``CfgDict`` so the wandb logger's ``cfg.to_dict()``
    path succeeds (rsl-rl 5.x ``WandbSummaryWriter.store_config`` calls it
    on the runner cfg as well as on env_cfg).
    """
    cfg = CfgDict(train_cfg)
    if run_name is not None:
        cfg["run_name"] = run_name
    if logger is not None:
        cfg["logger"] = logger
    cfg["obs_groups"] = env.obs_groups   # rsl-rl 5.2 — env owns the source of truth
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return OnPolicyRunner(env, cfg, str(log_dir), device=device)


def latest_model_pt(log_dir: str | Path) -> Path | None:
    """Return the most recently modified ``model_*.pt`` under log_dir, if any."""
    log_dir = Path(log_dir)
    candidates = sorted(log_dir.rglob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def save_final_ckpt(runner: Any, log_dir: str | Path) -> Path | None:
    """Safety-net: ensure a ``model_<iter>.pt`` exists after ``runner.learn()`` returns.

    rsl-rl 5.x ``OnPolicyRunner.learn()`` writes ``model_<current_iter>.pt`` at
    the end if ``logger.writer is not None`` (verified against rsl_rl source).
    So this helper is a defensive no-op in the common case; it only kicks in
    when (a) the logger was disabled, (b) save_interval+max_iterations hit a
    boundary that skipped the auto-save, or (c) future rsl-rl drift.

    Falls back through three save paths in order:
      1. ``runner.save(str(file_path))``  — rsl-rl 5.x signature (file path arg)
      2. ``runner.save(str(log_dir))``    — older signature (dir arg)
      3. raw ``torch.save({...}, file_path)`` with actor/critic state_dicts

    Returns the resolved file path of the final ckpt, or None if all attempts
    failed (caller should handle as missing artifact).
    """
    import torch

    log_dir = Path(log_dir)
    final_iter = int(getattr(runner, "current_learning_iteration", 0))
    final_path = log_dir / f"model_{final_iter}.pt"
    if final_path.exists() and final_path.stat().st_size > 0:
        return final_path

    # 1) modern rsl-rl signature: save(file_path).
    try:
        runner.save(str(final_path))
        if final_path.exists() and final_path.stat().st_size > 0:
            return final_path
    except TypeError:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[save_final_ckpt] runner.save(file_path) raised: {e!r}", flush=True)

    # 2) older signature variant: save(dir).
    try:
        runner.save(str(log_dir))
        if final_path.exists() and final_path.stat().st_size > 0:
            return final_path
    except TypeError:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[save_final_ckpt] runner.save(dir) raised: {e!r}", flush=True)

    # 3) last-resort raw torch.save.
    try:
        actor = runner.alg.actor
        critic = getattr(runner.alg, "critic", None)
        payload = {
            "model_state_dict": {
                "actor": actor.state_dict(),
                **({"critic": critic.state_dict()} if critic is not None else {}),
            },
            "iter": final_iter,
        }
        torch.save(payload, final_path)
        if final_path.exists() and final_path.stat().st_size > 0:
            return final_path
    except Exception as e:  # noqa: BLE001
        print(f"[save_final_ckpt] raw torch.save fallback failed: {e!r}", flush=True)

    return None


def alias_ckpt(target: str | Path, alias: str | Path) -> None:
    """Create ``alias`` → ``target`` symlink. Replaces existing alias safely."""
    target_p = Path(target).resolve()
    alias_p = Path(alias)
    alias_p.parent.mkdir(parents=True, exist_ok=True)
    if alias_p.is_symlink() or alias_p.exists():
        alias_p.unlink()
    os.symlink(target_p, alias_p)


def save_frozen_actor_snapshot(
    runner: Any,
    log_dir: str | Path,
    actor_filename: str,
    alias_path: str | Path | None = None,
) -> bool:
    """Save a frozen snapshot of the current actor without disturbing training.

    Temporarily flips the live actor to ``eval()`` + ``requires_grad=False``,
    serialises, then restores the original training state. Avoids
    ``copy.deepcopy`` (which fails on actors holding non-leaf tensors from
    the most recent forward pass — e.g., distribution outputs with grad_fn).
    ``torch.save`` itself pickles via ``__reduce_ex__`` which detaches each
    tensor, so saving the live module is safe under the temporary flag flip.

    Skips save when any NaN/Inf is detected (returns False). Creates
    ``alias_path`` symlink to the saved snapshot when provided.

    Used by train_attacker / train_defender to checkpoint the frozen actor
    every ``save_interval`` iterations so that mid-training stops still leave
    a usable ``<role>_v0.pt`` for downstream consumers (PretrainedDefender,
    AMS orchestrator).
    """
    import torch

    actor = runner.alg.actor
    has_nan = any(
        torch.isnan(p).any() or torch.isinf(p).any() for p in actor.parameters()
    )
    if has_nan:
        return False

    was_training = actor.training
    grad_states = [p.requires_grad for p in actor.parameters()]
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    actor_path = Path(log_dir) / actor_filename
    try:
        torch.save(actor, actor_path)
    finally:
        if was_training:
            actor.train()
        for p, g in zip(actor.parameters(), grad_states):
            p.requires_grad_(g)

    if alias_path is not None:
        alias_ckpt(actor_path, alias_path)
    return True


def resolve_actor_ckpt(path: str | Path, prefix: str) -> Path:
    """Resolve a frozen actor ckpt: file → as-is; directory → latest ``<prefix>_v*.pt``.

    Used by PretrainedDefender / PretrainedAttacker so callers can pass
    either a specific ``.pt`` or a project-wide directory and auto-pick
    the most recent stage version (e.g., v0 → v1 → v2 in AMS-DRL).
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob(f"{prefix}_v*.pt"), key=lambda x: x.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(
                f"No {prefix}_v*.pt found under directory {p}. "
                f"Train scripts/train_{prefix}.py first."
            )
        return cands[-1]
    raise FileNotFoundError(f"ckpt path not found: {p}")
