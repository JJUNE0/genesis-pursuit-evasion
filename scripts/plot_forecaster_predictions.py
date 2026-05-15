"""Phase 3c — Visualize forecaster predictions vs ground-truth defender trajectories.

Loads a trained forecaster ckpt, samples a few (episode, t) windows from the
validation split, runs Euler-20 sampling at given τ, and renders ground-truth
target vs prediction in the **t-attacker body frame** (where attacker = origin).

Two views per sample:
  - **Top row** (XY plane, top-down): defender ground-truth (blue) + prediction
    (red). Time encoded as marker size + start/end annotated.
  - **Bottom row** (XZ plane, side): same data, different plane.

Body-frame metric coordinates: divides by ``pos_scale`` so axes are meters.

Usage::

    python scripts/plot_forecaster_predictions.py \\
        --ckpt logs/forecaster/ams_v7_c8_v1/best.pt \\
        --data_dir data/forecaster_data_ams_v7_c8 \\
        --tau 25 --num_samples 6 --num_euler_steps 20 \\
        --out /tmp/forecaster_predictions.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.forecaster.dataset import (   # noqa: E402
    Pattern2Collator,
    build_dataset_and_collator,
    load_scales_from_dataset,
)
from agents.forecaster.model import ForecasterModel   # noqa: E402
from agents.forecaster.rectified_flow import sample_via_euler   # noqa: E402
from agents.forecaster.trainer import load_ckpt   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot forecaster predictions vs GT")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--tau", type=int, default=25)
    p.add_argument("--tau_max", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--num_euler_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="/tmp/forecaster_predictions.png")
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    # Build val dataset + collator.
    val_dataset, val_collator = build_dataset_and_collator(
        data_dir=args.data_dir, mode="val", tau_max=int(args.tau_max),
        in_memory=False, split_seed=42, rng_seed=int(args.seed),
    )
    pos_scale, vel_scale, ang_vel_scale = load_scales_from_dataset(args.data_dir)

    # Pick deterministic samples spread across val.
    rng = np.random.default_rng(int(args.seed))
    n = min(int(args.num_samples), len(val_dataset))
    chosen = rng.choice(len(val_dataset), size=n, replace=False).tolist()
    raw_batch = [val_dataset[i] for i in chosen]

    batch, tau = val_collator(raw_batch, tau_override=int(args.tau))
    batch = {k: v.to(device) for k, v in batch.items()}
    target = batch["target"]                      # (B, τ, 6) body+scaled
    context = {k: batch[k] for k in ("attacker_history", "defender_delayed", "action_history")}
    defender_delayed = batch["defender_delayed"]  # (B, 1, 6)  start point t-τ

    # Load model.
    model = ForecasterModel(tau_max=int(args.tau_max)).to(device)
    payload = load_ckpt(args.ckpt, model=model, optimizer=None, map_location=device)
    model.eval()
    step = int(payload.get("step", 0))
    extra = payload.get("extra", {}) or {}
    val_metrics = extra.get("val_metrics", {})

    # Multi-sample stable estimate: average over a few Euler runs (RF is stochastic
    # in initial noise — small variance, but plotting median of 4 reduces visual jitter).
    preds = []
    for _ in range(4):
        preds.append(sample_via_euler(
            model=model, context=context, tau=int(args.tau),
            num_steps=int(args.num_euler_steps), target_dim=int(target.shape[-1]),
        ))
    pred = torch.stack(preds, dim=0).median(dim=0).values   # (B, τ, 6)

    # Move to CPU + unscale to meters.
    target_m = (target.detach().cpu().numpy() / pos_scale)
    pred_m = (pred.detach().cpu().numpy() / pos_scale)
    delayed_m = (defender_delayed.detach().cpu().numpy() / pos_scale)
    # (note: for vel components 3:6 we'd divide by vel_scale — but plots use only pos 0:3)

    # ---- Layout: 2 rows × n cols ----
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.0), squeeze=False)
    fig.suptitle(
        f"Forecaster predictions vs GT  (ckpt step={step}, τ={tau}, num_samples={n})\n"
        + (f"val/avg_last_mse={val_metrics.get('val/avg_last_mse', float('nan')):.4f}  "
           f"val/avg_traj_mse={val_metrics.get('val/avg_traj_mse', float('nan')):.4f}"
           if val_metrics else ""),
        fontsize=10,
    )

    times = np.arange(int(tau))           # 0..τ-1, color encoded
    for col in range(n):
        # Build trajectories starting from delayed point (so plotted line is continuous).
        d_start = delayed_m[col, 0, 0:3]                    # (3,) at t-τ
        gt = np.concatenate([d_start[None, :], target_m[col, :, 0:3]], axis=0)   # (τ+1, 3)
        pr = np.concatenate([d_start[None, :], pred_m[col, :, 0:3]],   axis=0)   # (τ+1, 3)
        t_axis = np.arange(int(tau) + 1)                    # 0..τ

        # ---- Top: XY plane ----
        ax = axes[0, col]
        ax.plot(gt[:, 0], gt[:, 1], "-", color="C0", lw=1.5, label="GT defender")
        ax.plot(pr[:, 0], pr[:, 1], "--", color="C3", lw=1.5, label="forecaster pred")
        ax.scatter(gt[0, 0], gt[0, 1], c="black", s=50, marker="s",
                   zorder=5, label="t-τ start")
        ax.scatter(gt[-1, 0], gt[-1, 1], c="C0", s=50, marker="^",
                   zorder=5, label="GT end")
        ax.scatter(pr[-1, 0], pr[-1, 1], c="C3", s=50, marker="x",
                   zorder=5, label="pred end")
        ax.scatter(0, 0, c="green", s=70, marker="*", zorder=6, label="attacker (t)")
        ax.set_xlabel("body-X (m)")
        ax.set_ylabel("body-Y (m)" if col == 0 else "")
        ax.set_title(f"sample #{col} — XY top-down")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        if col == 0:
            ax.legend(loc="best", fontsize=7)

        # ---- Bottom: XZ plane ----
        ax = axes[1, col]
        ax.plot(gt[:, 0], gt[:, 2], "-", color="C0", lw=1.5)
        ax.plot(pr[:, 0], pr[:, 2], "--", color="C3", lw=1.5)
        ax.scatter(gt[0, 0], gt[0, 2], c="black", s=50, marker="s", zorder=5)
        ax.scatter(gt[-1, 0], gt[-1, 2], c="C0", s=50, marker="^", zorder=5)
        ax.scatter(pr[-1, 0], pr[-1, 2], c="C3", s=50, marker="x", zorder=5)
        ax.scatter(0, 0, c="green", s=70, marker="*", zorder=6)
        ax.set_xlabel("body-X (m)")
        ax.set_ylabel("body-Z (m)" if col == 0 else "")
        ax.set_title("XZ side")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[plot] saved {args.out}  (n={n} samples, τ={tau}, ckpt step={step})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
