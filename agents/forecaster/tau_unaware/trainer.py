"""Phase 3f — τ-Unaware Forecaster training utilities (TODO §3f, 2026-05-14).

Mirrors Phase 3c trainer (build_train_loader / cosine LR / save+load_ckpt /
validate) for the τ-unaware setup. Validation uses Euler sampling (CFM model)
or direct forward (deterministic baseline).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agents.forecaster.tau_unaware import constants as C
from agents.forecaster.tau_unaware.dataset import (
    TauStripCollator,
    TauUnawareDataset,
    build_dataset_and_collator,
)
from agents.forecaster.tau_unaware.flow import sample_via_euler


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def build_train_loader(
    data_dir: str | Path | list[str | Path],
    batch_size: int = 256,
    history_K: int = C.DEFAULT_HISTORY_K,
    num_workers: int = 0,
    in_memory: bool = False,
    split_seed: int = 42,
    rng_seed: int | None = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    episode_subset: set[int] | None = None,
) -> tuple[TauUnawareDataset, TauStripCollator, DataLoader]:
    dataset, collator = build_dataset_and_collator(
        data_dir=data_dir, mode="train", history_K=history_K, split_seed=split_seed,
        in_memory=in_memory, episode_subset=episode_subset, rng_seed=rng_seed,
    )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=bool(shuffle),
        num_workers=int(num_workers), pin_memory=bool(pin_memory),
        collate_fn=collator, drop_last=True,
    )
    return dataset, collator, loader


def build_val_dataset_and_collator(
    data_dir: str | Path | list[str | Path],
    history_K: int = C.DEFAULT_HISTORY_K,
    in_memory: bool = False,
    split_seed: int = 42,
    rng_seed: int | None = None,
    episode_subset: set[int] | None = None,
) -> tuple[TauUnawareDataset, TauStripCollator]:
    return build_dataset_and_collator(
        data_dir=data_dir, mode="val", history_K=history_K, split_seed=split_seed,
        in_memory=in_memory, episode_subset=episode_subset, rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# LR schedule (reuse Phase 3c)
# ---------------------------------------------------------------------------

def cosine_with_warmup_lr(
    step: int, *, warmup_steps: int, total_steps: int, lr_max: float, lr_min: float,
) -> float:
    if step < warmup_steps:
        return lr_max * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)


# ---------------------------------------------------------------------------
# Validation (per-τ sweep — model is τ-unaware but we eval at fixed τ values)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_dataset: TauUnawareDataset,
    val_collator: TauStripCollator,
    *,
    val_subsample_size: int = 256,
    tau_evals: Iterable[int] = (10, 25, 50),
    num_euler_steps: int = 10,
    device: str | torch.device = "cuda",
    seed: int = 0,
    deterministic: bool = False,
) -> dict[str, float]:
    """Per-τ MSE sweep on a fixed val subsample.

    ``deterministic=True`` switches to direct forward (for the deterministic
    baseline model); otherwise uses Euler sampling.
    """
    if len(val_dataset) == 0:
        return {}
    model.eval()

    rng = np.random.default_rng(int(seed))
    if len(val_dataset) <= int(val_subsample_size):
        chosen = list(range(len(val_dataset)))
    else:
        chosen = rng.choice(len(val_dataset), int(val_subsample_size), replace=False).tolist()
    raw = [val_dataset[i] for i in chosen]

    metrics: dict[str, float] = {}
    for tau_eval in tau_evals:
        tau_eval = int(tau_eval)
        if tau_eval > val_collator.history_K:
            continue
        batch, _ = val_collator(raw, tau_override=tau_eval)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch["target"]                                          # (B, 6)
        context = {
            "attacker_history": batch["attacker_history"],
            "defender_last":    batch["defender_last"],
            "action_history":   batch["action_history"],
        }
        if deterministic:
            pred = model(context)                                         # (B, 6)
        else:
            pred = sample_via_euler(
                model=model, context=context,
                num_steps=int(num_euler_steps), target_dim=int(target.shape[-1]),
            )                                                             # (B, 6)
        mse = F.mse_loss(pred, target).item()
        metrics[f"val/mse_tau{tau_eval}"] = float(mse)

    if metrics:
        metrics["val/avg_mse"] = float(np.mean(list(metrics.values())))

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint save / load (atomic)
# ---------------------------------------------------------------------------

def save_ckpt(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
    }
    if extra:
        payload["extra"] = extra
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_ckpt(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
