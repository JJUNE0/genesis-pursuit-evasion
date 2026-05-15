"""Phase 3c — Forecaster training utilities (TODO §3c, 2026-05-13).

High-level helpers used by ``scripts/train_forecaster.py``:

- ``build_train_loader`` / ``build_val_dataset_and_collator``
- ``cosine_with_warmup_lr`` (lambda LR scheduler)
- ``validate``: 256-sample × 3 τ-eval Euler-20 sweep (avg_last_mse for best ckpt)
- ``save_ckpt`` / ``load_ckpt``: model + optimizer + scheduler state

CLAUDE.md §10: ckpts hold enough state to resume training from any point.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agents.forecaster.dataset import (
    ForecasterDataset,
    Pattern2Collator,
    build_dataset_and_collator,
)
from agents.forecaster.rectified_flow import sample_via_euler


# ---------------------------------------------------------------------------
# Loader builder (DataLoader works because Pattern2Collator is picklable)
# ---------------------------------------------------------------------------

def build_train_loader(
    data_dir: str | Path | list[str | Path],
    batch_size: int = 256,
    tau_max: int = 50,
    num_workers: int = 0,
    in_memory: bool = False,
    split_seed: int = 42,
    rng_seed: int | None = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    episode_subset: set[int] | None = None,
) -> tuple[ForecasterDataset, Pattern2Collator, DataLoader]:
    dataset, collator = build_dataset_and_collator(
        data_dir=data_dir, mode="train", tau_max=tau_max, split_seed=split_seed,
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
    tau_max: int = 50,
    in_memory: bool = False,
    split_seed: int = 42,
    rng_seed: int | None = None,
    episode_subset: set[int] | None = None,
) -> tuple[ForecasterDataset, Pattern2Collator]:
    return build_dataset_and_collator(
        data_dir=data_dir, mode="val", tau_max=tau_max, split_seed=split_seed,
        in_memory=in_memory, episode_subset=episode_subset, rng_seed=rng_seed,
    )


# ---------------------------------------------------------------------------
# LR schedule (linear warmup → cosine decay to lr_min)
# ---------------------------------------------------------------------------

def cosine_with_warmup_lr(
    step: int, *, warmup_steps: int, total_steps: int, lr_max: float, lr_min: float,
) -> float:
    """Compute LR multiplier-free absolute LR (used as ``optimizer.param_groups[*].lr = ...``)."""
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
# Validation (256-sample fixed subsample × 3 τ-eval × Euler 20)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_dataset: ForecasterDataset,
    val_collator: Pattern2Collator,
    *,
    val_subsample_size: int = 256,
    tau_evals: Iterable[int] = (10, 25, 50),
    num_euler_steps: int = 20,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> dict[str, float]:
    """Fixed-subsample val: same 256 (ep, t) pairs evaluated at 3 τ_eval values.

    ``val/avg_last_mse`` is the best-ckpt criterion (deployment quantity =
    last-step prediction).
    """
    if len(val_dataset) == 0:
        return {}
    model.eval()

    # Deterministic subsample.
    rng = np.random.default_rng(int(seed))
    if len(val_dataset) <= int(val_subsample_size):
        chosen = list(range(len(val_dataset)))
    else:
        chosen = rng.choice(len(val_dataset), int(val_subsample_size), replace=False).tolist()
    raw = [val_dataset[i] for i in chosen]

    metrics: dict[str, float] = {}
    for tau_eval in tau_evals:
        tau_eval = int(tau_eval)
        if tau_eval > val_collator.tau_max:
            # Skip — collator's frame-transform helper enforces τ ≤ τ_max via
            # window slicing earlier.
            continue
        batch, _ = val_collator(raw, tau_override=tau_eval)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch["target"]                                          # (B, τ_eval, 6)
        context = {
            "attacker_history": batch["attacker_history"],
            "defender_delayed": batch["defender_delayed"],
            "action_history":   batch["action_history"],
        }
        pred = sample_via_euler(
            model=model, context=context, tau=tau_eval,
            num_steps=int(num_euler_steps), target_dim=int(target.shape[-1]),
        )                                                                 # (B, τ_eval, 6)
        last_mse = F.mse_loss(pred[:, -1, :], target[:, -1, :]).item()
        traj_mse = F.mse_loss(pred, target).item()
        metrics[f"val/last_mse_tau{tau_eval}"] = float(last_mse)
        metrics[f"val/traj_mse_tau{tau_eval}"] = float(traj_mse)

    if metrics:
        last_keys = [k for k in metrics if k.startswith("val/last_mse_tau")]
        traj_keys = [k for k in metrics if k.startswith("val/traj_mse_tau")]
        metrics["val/avg_last_mse"] = float(np.mean([metrics[k] for k in last_keys]))
        metrics["val/avg_traj_mse"] = float(np.mean([metrics[k] for k in traj_keys]))

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_ckpt(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomic save (tmp → rename)."""
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
    """Load weights (and optimizer state if provided). Returns the raw dict."""
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
