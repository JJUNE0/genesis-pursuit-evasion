"""Phase 3f — τ-Unaware Flow Matching forecaster training entry point.

Mirrors ``scripts/train_forecaster.py`` (Phase 3c) but for the τ-unaware
model: no τ embedding, single-step target, fixed-K history. Uses the same
ams_v9 multi-cycle dataset.

Usage::

    python scripts/train_tau_unaware_forecaster.py \\
        --config configs/forecaster/tau_unaware_default.yaml

    # Switch to deterministic-head ablation (Comparison row #6)
    python scripts/train_tau_unaware_forecaster.py \\
        --config configs/forecaster/tau_unaware_default.yaml \\
        --head_type deterministic --log_dir logs/forecaster/tau_unaware_det_v1

    # Smoke
    python scripts/train_tau_unaware_forecaster.py \\
        --config configs/forecaster/tau_unaware_default.yaml \\
        --num_training_steps 5000 --val_every 2500 --save_every 5000 \\
        --log_dir logs/forecaster/tau_unaware_smoke --wandb_mode disabled
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.forecaster.tau_unaware import (   # noqa: E402
    TauUnawareDeterministicModel,
    TauUnawareForecasterModel,
    build_train_loader,
    build_val_dataset_and_collator,
    cfm_loss,
    cosine_with_warmup_lr,
    deterministic_mse_loss,
    save_ckpt,
    set_lr,
    validate,
)


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3f τ-unaware forecaster training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--history_K", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_training_steps", type=int, default=None)
    p.add_argument("--lr_max", type=float, default=None)
    p.add_argument("--lr_min", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--use_bf16", type=lambda s: str(s).lower() != "false", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--in_memory", type=lambda s: str(s).lower() != "false", default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backend", type=str, default=None, choices=("gpu", "cpu"))
    p.add_argument("--head_type", type=str, default=None, choices=("flow", "deterministic"),
                   help="flow = TauUnawareForecasterModel; deterministic = ablation row #6.")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None,
                   choices=("online", "offline", "disabled"))
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def _merge_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    cfg.setdefault("data_dir", "data/forecaster_data_ams_v9_c8")
    cfg.setdefault("history_K", 50)
    cfg.setdefault("batch_size", 256)
    cfg.setdefault("num_training_steps", 100000)
    cfg.setdefault("lr_max", 1.0e-4)
    cfg.setdefault("lr_min", 1.0e-5)
    cfg.setdefault("warmup_steps", 1000)
    cfg.setdefault("weight_decay", 1.0e-4)
    cfg.setdefault("betas", [0.9, 0.999])
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("use_bf16", True)
    cfg.setdefault("num_workers", 0)
    cfg.setdefault("pin_memory", True)
    cfg.setdefault("in_memory", True)
    cfg.setdefault("split_seed", 42)
    cfg.setdefault("val_every", 5000)
    cfg.setdefault("val_subsample_size", 256)
    cfg.setdefault("val_tau_evals", [10, 25, 50])
    cfg.setdefault("val_num_euler_steps", 10)
    cfg.setdefault("save_every", 10000)
    cfg.setdefault("log_every", 100)
    cfg.setdefault("log_dir", "logs/forecaster/tau_unaware_v1")
    cfg.setdefault("seed", 42)
    cfg.setdefault("backend", "gpu")
    cfg.setdefault("head_type", "flow")
    cfg.setdefault("wandb_project", "forecaster")
    cfg.setdefault("wandb_run_name", None)
    cfg.setdefault("wandb_mode", "online")
    cfg.setdefault("model", {})

    overrides = {
        "data_dir": args.data_dir, "history_K": args.history_K,
        "batch_size": args.batch_size, "num_training_steps": args.num_training_steps,
        "lr_max": args.lr_max, "lr_min": args.lr_min,
        "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "use_bf16": args.use_bf16,
        "num_workers": args.num_workers, "in_memory": args.in_memory,
        "val_every": args.val_every, "save_every": args.save_every,
        "log_every": args.log_every, "log_dir": args.log_dir,
        "seed": args.seed, "backend": args.backend, "head_type": args.head_type,
        "wandb_project": args.wandb_project, "wandb_run_name": args.wandb_run_name,
        "wandb_mode": args.wandb_mode,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    cfg["resume"] = args.resume
    return cfg


def _snapshot_config(log_dir: Path, cfg: dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "config.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _init_wandb(cfg: dict[str, Any], log_dir: Path):
    mode = cfg["wandb_mode"]
    if mode == "disabled":
        return None
    import wandb   # noqa: PLC0415
    run_name = cfg["wandb_run_name"] or f"{Path(cfg['log_dir']).name}_seed{cfg['seed']}"
    return wandb.init(
        project=cfg["wandb_project"], name=run_name,
        config=cfg, dir=str(log_dir), mode=mode,
    )


def _wandb_log(run, payload: dict[str, Any], step: int) -> None:
    if run is None:
        return
    run.log(payload, step=int(step))


def _train_iter(loader):
    while True:
        for batch in loader:
            yield batch


def main() -> int:
    args = parse_args()
    cfg = _merge_config(args)

    seed = int(cfg["seed"])
    _seed_all(seed)
    use_gpu = (cfg["backend"] == "gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    use_bf16 = bool(cfg["use_bf16"]) and use_gpu

    log_dir = Path(cfg["log_dir"]).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_config(log_dir, cfg)

    data_dir_repr = (
        f"{len(cfg['data_dir'])} dirs: {cfg['data_dir']}"
        if isinstance(cfg["data_dir"], list) else str(cfg["data_dir"])
    )
    head_type = str(cfg["head_type"])
    print(f"[tau-unaware] data_dir={data_dir_repr}  in_memory={cfg['in_memory']}  "
          f"head_type={head_type}", flush=True)

    train_dataset, train_collator, train_loader = build_train_loader(
        data_dir=cfg["data_dir"], batch_size=int(cfg["batch_size"]),
        history_K=int(cfg["history_K"]), num_workers=int(cfg["num_workers"]),
        in_memory=bool(cfg["in_memory"]), split_seed=int(cfg["split_seed"]),
        rng_seed=seed, shuffle=True, pin_memory=bool(cfg["pin_memory"]),
    )
    val_dataset, val_collator = build_val_dataset_and_collator(
        data_dir=cfg["data_dir"], history_K=int(cfg["history_K"]),
        in_memory=bool(cfg["in_memory"]), split_seed=int(cfg["split_seed"]),
        rng_seed=seed + 1,
    )
    print(f"[tau-unaware] train samples={len(train_dataset)}  val samples={len(val_dataset)}",
          flush=True)

    # Model selection
    model_cfg = cfg.get("model", {}) or {}
    if head_type == "flow":
        model = TauUnawareForecasterModel(history_K=int(cfg["history_K"]), **model_cfg)
    elif head_type == "deterministic":
        # Deterministic baseline now uses SAME encoder + decoder as flow model
        # (peer review fix 2026-05-14) — accepts identical kwargs.
        model = TauUnawareDeterministicModel(history_K=int(cfg["history_K"]), **model_cfg)
    else:
        raise ValueError(f"head_type must be 'flow' or 'deterministic', got {head_type!r}")
    model = model.to(device)
    print(f"[tau-unaware] model params = {model.num_parameters():,}", flush=True)

    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr_max"]),
        weight_decay=float(cfg["weight_decay"]), betas=betas,
    )

    start_step = 0
    if cfg.get("resume"):
        from agents.forecaster.tau_unaware.trainer import load_ckpt   # noqa: PLC0415
        payload = load_ckpt(cfg["resume"], model=model, optimizer=optimizer, map_location=device)
        start_step = int(payload.get("step", 0))
        print(f"[tau-unaware] resumed from {cfg['resume']} at step {start_step}", flush=True)

    wandb_run = _init_wandb(cfg, log_dir)

    total_steps = int(cfg["num_training_steps"])
    warmup = int(cfg["warmup_steps"])
    lr_max = float(cfg["lr_max"]); lr_min = float(cfg["lr_min"])
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    val_every = int(cfg["val_every"])
    save_every = int(cfg["save_every"])
    is_deterministic = (head_type == "deterministic")

    train_iter = _train_iter(train_loader)
    best_metric = float("inf")
    t0 = time.time()
    print(f"[tau-unaware] start training: total_steps={total_steps}  bf16={use_bf16}  "
          f"device={device}", flush=True)

    model.train()
    for step in range(start_step, total_steps):
        lr_now = cosine_with_warmup_lr(
            step, warmup_steps=warmup, total_steps=total_steps,
            lr_max=lr_max, lr_min=lr_min,
        )
        set_lr(optimizer, lr_now)

        batch, tau = next(train_iter)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = (deterministic_mse_loss(model, batch) if is_deterministic
                        else cfm_loss(model, batch))
        else:
            loss = (deterministic_mse_loss(model, batch) if is_deterministic
                    else cfm_loss(model, batch))

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (step % log_every == 0) or (step == total_steps - 1):
            elapsed = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] step {step}/{total_steps} "
                  f"loss={loss.item():.4f} tau={tau} lr={lr_now:.2e} "
                  f"elapsed={elapsed:.1f}s", flush=True)
            _wandb_log(wandb_run, {
                "train/loss": float(loss.item()),
                "train/lr": float(lr_now),
                "train/tau": int(tau),
            }, step=step)

        if (step > 0 and step % val_every == 0) or (step == total_steps - 1):
            val_metrics = validate(
                model=model, val_dataset=val_dataset, val_collator=val_collator,
                val_subsample_size=int(cfg["val_subsample_size"]),
                tau_evals=list(cfg["val_tau_evals"]),
                num_euler_steps=int(cfg["val_num_euler_steps"]),
                device=device, seed=seed, deterministic=is_deterministic,
            )
            avg = val_metrics.get("val/avg_mse", float("inf"))
            print(f"[{time.strftime('%H:%M:%S')}] VAL step {step} avg_mse={avg:.4f} | "
                  + " ".join(f"{k.split('/')[-1]}={v:.4f}"
                             for k, v in val_metrics.items() if k != "val/avg_mse"),
                  flush=True)
            _wandb_log(wandb_run, val_metrics, step=step)
            if avg < best_metric:
                best_metric = avg
                save_ckpt(log_dir / "best.pt", model=model, optimizer=optimizer,
                          step=step, extra={"val_metrics": val_metrics, "config": cfg})
                print(f"[tau-unaware] saved best.pt @ step {step} (avg_mse={avg:.4f})",
                      flush=True)

        if (step > 0 and step % save_every == 0) or (step == total_steps - 1):
            save_ckpt(log_dir / f"step_{step}.pt", model=model, optimizer=optimizer,
                      step=step, extra={"config": cfg})
            save_ckpt(log_dir / "latest.pt", model=model, optimizer=optimizer,
                      step=step, extra={"config": cfg})

    elapsed = time.time() - t0
    print(f"[tau-unaware] DONE — {total_steps} steps in {elapsed:.1f}s. log_dir={log_dir}",
          flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
