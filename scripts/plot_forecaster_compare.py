"""Phase 3c — Compare two forecaster ckpts on the same val samples.

Loads two trained forecaster ckpts (e.g., single-cycle vs multi-cycle) and
overlays their predictions against ground-truth defender trajectories on the
**same** val samples, in the t-attacker body frame.

Two views per sample:
  - Top row (XY top-down): GT (blue solid), ckpt-A pred (red dashed),
    ckpt-B pred (orange dashed). Attacker = origin (green star).
  - Bottom row (XZ side): same overlay.

For OOD evaluation, ``--data_dir`` should point to the dataset the **deployed**
attacker will encounter (typically the multi-cycle ams_v9 data).

Usage::

    python scripts/plot_forecaster_compare.py \\
        --ckpt_a logs/forecaster/ams_v7_c8_v1/best.pt \\
        --label_a "single (v7 c8)" \\
        --ckpt_b logs/forecaster/ams_v9_multi_v1/best.pt \\
        --label_b "multi (v9)" \\
        --data_dir data/forecaster_data_ams_v9_c8 data/forecaster_data_ams_v9_c17 \\
        --tau 25 --num_samples 6 --num_euler_steps 20 \\
        --out /tmp/forecaster_compare_t25.png
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
    build_dataset_and_collator,
    load_scales_from_dataset,
)
from agents.forecaster.model import ForecasterModel   # noqa: E402
from agents.forecaster.rectified_flow import sample_via_euler   # noqa: E402
from agents.forecaster.trainer import load_ckpt   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two forecaster ckpts on same val samples")
    p.add_argument("--ckpt_a", type=str, required=True)
    p.add_argument("--label_a", type=str, default="ckpt A")
    p.add_argument("--ckpt_b", type=str, required=True)
    p.add_argument("--label_b", type=str, default="ckpt B")
    p.add_argument("--data_dir", type=str, nargs="+", required=True,
                   help="One or more data dirs (multi-cycle: pass several).")
    p.add_argument("--tau", type=int, default=25)
    p.add_argument("--tau_max", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--num_euler_steps", type=int, default=20)
    p.add_argument("--num_euler_runs", type=int, default=4,
                   help="Average median over N Euler runs (RF stochastic in init noise).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="/tmp/forecaster_compare.png")
    return p.parse_args()


@torch.no_grad()
def _euler_median(model, context, tau: int, num_steps: int, num_runs: int):
    preds = []
    for _ in range(int(num_runs)):
        preds.append(sample_via_euler(
            model=model, context=context, tau=tau,
            num_steps=int(num_steps), target_dim=6,
        ))
    return torch.stack(preds, dim=0).median(dim=0).values   # (B, τ, 6)


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared error, over (τ, 6). Returns (B,)."""
    return ((pred - target) ** 2).mean(dim=(1, 2))


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    # Single dir if length 1, list otherwise
    data_dirs = args.data_dir if len(args.data_dir) > 1 else args.data_dir[0]

    # Build val dataset + collator (for sample picking and frame transform).
    val_dataset, val_collator = build_dataset_and_collator(
        data_dir=data_dirs, mode="val", tau_max=int(args.tau_max),
        in_memory=False, split_seed=42, rng_seed=int(args.seed),
    )
    pos_scale, _, _ = load_scales_from_dataset(data_dirs)

    rng = np.random.default_rng(int(args.seed))
    n = min(int(args.num_samples), len(val_dataset))
    chosen = rng.choice(len(val_dataset), size=n, replace=False).tolist()
    raw_batch = [val_dataset[i] for i in chosen]

    batch, tau = val_collator(raw_batch, tau_override=int(args.tau))
    batch = {k: v.to(device) for k, v in batch.items()}
    target = batch["target"]                                          # (B, τ, 6)
    context = {k: batch[k] for k in ("attacker_history", "defender_delayed", "action_history")}
    defender_delayed = batch["defender_delayed"]                      # (B, 1, 6)

    # Load both models.
    model_a = ForecasterModel(tau_max=int(args.tau_max)).to(device)
    payload_a = load_ckpt(args.ckpt_a, model=model_a, optimizer=None, map_location=device)
    model_a.eval()
    model_b = ForecasterModel(tau_max=int(args.tau_max)).to(device)
    payload_b = load_ckpt(args.ckpt_b, model=model_b, optimizer=None, map_location=device)
    model_b.eval()

    # Sample predictions (median over N Euler runs).
    pred_a = _euler_median(model_a, context, int(args.tau),
                           int(args.num_euler_steps), int(args.num_euler_runs))
    pred_b = _euler_median(model_b, context, int(args.tau),
                           int(args.num_euler_steps), int(args.num_euler_runs))

    # Per-sample MSE for headline numbers.
    mse_a = _per_sample_mse(pred_a, target).cpu().numpy()
    mse_b = _per_sample_mse(pred_b, target).cpu().numpy()
    last_mse_a = ((pred_a[:, -1, :] - target[:, -1, :]) ** 2).mean(dim=-1).cpu().numpy()
    last_mse_b = ((pred_b[:, -1, :] - target[:, -1, :]) ** 2).mean(dim=-1).cpu().numpy()

    # CPU + unscale to meters (pos channels only; the plot uses pos slices).
    target_m = target.detach().cpu().numpy() / pos_scale
    pred_a_m = pred_a.detach().cpu().numpy() / pos_scale
    pred_b_m = pred_b.detach().cpu().numpy() / pos_scale
    delayed_m = defender_delayed.detach().cpu().numpy() / pos_scale

    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.5), squeeze=False)
    fig.suptitle(
        f"Forecaster A vs B — same val samples  (τ={tau}, n={n})\n"
        f"A = {args.label_a} (step={payload_a.get('step', '?')})  |  "
        f"B = {args.label_b} (step={payload_b.get('step', '?')})  |  "
        f"data = {data_dirs if isinstance(data_dirs, str) else f'{len(data_dirs)} dirs'}\n"
        f"avg per-sample traj MSE — A: {mse_a.mean():.4f}  B: {mse_b.mean():.4f}   |   "
        f"avg last MSE — A: {last_mse_a.mean():.4f}  B: {last_mse_b.mean():.4f}",
        fontsize=9,
    )

    for col in range(n):
        d_start = delayed_m[col, 0, 0:3]
        gt = np.concatenate([d_start[None, :], target_m[col, :, 0:3]], axis=0)
        pa = np.concatenate([d_start[None, :], pred_a_m[col, :, 0:3]], axis=0)
        pb = np.concatenate([d_start[None, :], pred_b_m[col, :, 0:3]], axis=0)

        # ---- Top: XY top-down ----
        ax = axes[0, col]
        ax.plot(gt[:, 0], gt[:, 1], "-", color="C0", lw=1.6, label="GT")
        ax.plot(pa[:, 0], pa[:, 1], "--", color="C3", lw=1.4, label="A pred")
        ax.plot(pb[:, 0], pb[:, 1], "--", color="C1", lw=1.4, label="B pred")
        ax.scatter(gt[0, 0], gt[0, 1], c="black", s=40, marker="s", zorder=5, label="t-τ start")
        ax.scatter(gt[-1, 0], gt[-1, 1], c="C0", s=50, marker="^", zorder=5)
        ax.scatter(pa[-1, 0], pa[-1, 1], c="C3", s=50, marker="x", zorder=5)
        ax.scatter(pb[-1, 0], pb[-1, 1], c="C1", s=50, marker="x", zorder=5)
        ax.scatter(0, 0, c="green", s=70, marker="*", zorder=6, label="attacker (t)")
        ax.set_xlabel("body-X (m)")
        ax.set_ylabel("body-Y (m)" if col == 0 else "")
        ax.set_title(
            f"#{col} — XY\nA last={last_mse_a[col]:.4f}  B last={last_mse_b[col]:.4f}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        if col == 0:
            ax.legend(loc="best", fontsize=6)

        # ---- Bottom: XZ side ----
        ax = axes[1, col]
        ax.plot(gt[:, 0], gt[:, 2], "-", color="C0", lw=1.6)
        ax.plot(pa[:, 0], pa[:, 2], "--", color="C3", lw=1.4)
        ax.plot(pb[:, 0], pb[:, 2], "--", color="C1", lw=1.4)
        ax.scatter(gt[0, 0], gt[0, 2], c="black", s=40, marker="s", zorder=5)
        ax.scatter(gt[-1, 0], gt[-1, 2], c="C0", s=50, marker="^", zorder=5)
        ax.scatter(pa[-1, 0], pa[-1, 2], c="C3", s=50, marker="x", zorder=5)
        ax.scatter(pb[-1, 0], pb[-1, 2], c="C1", s=50, marker="x", zorder=5)
        ax.scatter(0, 0, c="green", s=70, marker="*", zorder=6)
        ax.set_xlabel("body-X (m)")
        ax.set_ylabel("body-Z (m)" if col == 0 else "")
        ax.set_title("XZ side", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[plot] saved {args.out}")
    print(f"  A ({args.label_a}) avg traj MSE = {mse_a.mean():.4f}, avg last = {last_mse_a.mean():.4f}")
    print(f"  B ({args.label_b}) avg traj MSE = {mse_b.mean():.4f}, avg last = {last_mse_b.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
