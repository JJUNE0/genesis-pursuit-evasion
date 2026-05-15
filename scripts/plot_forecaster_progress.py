"""Phase 3c — Plot forecaster training progress from log file (TODO §3c, 2026-05-13).

Parses ``/tmp/forecaster_full_train.log`` (or any log path) and renders:
  - Train loss vs step (log scale)
  - LR schedule
  - Val avg_last_mse + avg_traj_mse
  - Per-τ last_mse (τ=10, 25, 50)

Usage::

    python scripts/plot_forecaster_progress.py /tmp/forecaster_full_train.log \\
        --out /tmp/forecaster_progress.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"step\s+(\d+)/\d+\s+loss=([0-9.eE+-]+)\s+tau=(\d+)\s+lr=([0-9.eE+-]+)"
)
VAL_RE = re.compile(
    r"VAL step (\d+)\s+avg_last_mse=([0-9.eE+-]+)\s+avg_traj_mse=([0-9.eE+-]+)"
    r"\s+last_mse_tau10=([0-9.eE+-]+)\s+last_mse_tau25=([0-9.eE+-]+)"
    r"\s+last_mse_tau50=([0-9.eE+-]+)"
)


def parse_log(path: Path):
    train = []
    val = []
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                step, loss, tau, lr = m.groups()
                train.append((int(step), float(loss), int(tau), float(lr)))
                continue
            m = VAL_RE.search(line)
            if m:
                step, avg_l, avg_t, t10, t25, t50 = m.groups()
                val.append((int(step), float(avg_l), float(avg_t),
                            float(t10), float(t25), float(t50)))
    return train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_path", type=str)
    p.add_argument("--out", type=str, default="/tmp/forecaster_progress.png")
    args = p.parse_args()

    train, val = parse_log(Path(args.log_path))
    if not train:
        print("[plot] no train rows parsed; check log format.")
        return 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_loss, ax_lr, ax_val_avg, ax_val_tau = axes.flatten()

    steps = [r[0] for r in train]
    losses = [r[1] for r in train]
    lrs = [r[3] for r in train]

    # Train loss (log scale)
    ax_loss.plot(steps, losses, color="C0", lw=0.8, alpha=0.6, label="train loss (raw)")
    # Smooth with a window-200 moving avg if enough rows
    if len(losses) >= 50:
        win = max(5, len(losses) // 50)
        ma = []
        for i in range(len(losses)):
            lo = max(0, i - win + 1)
            ma.append(sum(losses[lo:i + 1]) / (i - lo + 1))
        ax_loss.plot(steps, ma, color="C0", lw=1.6, label=f"moving avg (w={win})")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("RF loss (log)")
    ax_loss.set_title("Train RF loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    # LR schedule
    ax_lr.plot(steps, lrs, color="C2")
    ax_lr.set_xlabel("step")
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("LR schedule (warmup → cosine decay)")
    ax_lr.grid(True, alpha=0.3)

    # Val avg curves
    if val:
        v_steps = [r[0] for r in val]
        v_avg_l = [r[1] for r in val]
        v_avg_t = [r[2] for r in val]
        ax_val_avg.plot(v_steps, v_avg_l, "o-", color="C3", label="val/avg_last_mse (best ckpt criterion)")
        ax_val_avg.plot(v_steps, v_avg_t, "s-", color="C1", label="val/avg_traj_mse")
        ax_val_avg.set_xlabel("step")
        ax_val_avg.set_ylabel("MSE")
        ax_val_avg.set_title(f"Validation (n={len(val)} val checkpoints)")
        ax_val_avg.grid(True, alpha=0.3)
        ax_val_avg.legend()

        v_t10 = [r[3] for r in val]
        v_t25 = [r[4] for r in val]
        v_t50 = [r[5] for r in val]
        ax_val_tau.plot(v_steps, v_t10, "o-", color="C0", label="τ=10")
        ax_val_tau.plot(v_steps, v_t25, "s-", color="C2", label="τ=25")
        ax_val_tau.plot(v_steps, v_t50, "^-", color="C3", label="τ=50")
        ax_val_tau.set_xlabel("step")
        ax_val_tau.set_ylabel("last_mse")
        ax_val_tau.set_title("Per-τ validation last_mse (Euler 20 step)")
        ax_val_tau.grid(True, alpha=0.3)
        ax_val_tau.legend()
    else:
        ax_val_avg.set_title("no val rows yet")
        ax_val_tau.set_title("no val rows yet")

    fig.suptitle(f"Forecaster training — {Path(args.log_path).name}  |  "
                 f"{len(train)} train logs, {len(val)} val checkpoints", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"[plot] saved {args.out}  ({len(train)} train, {len(val)} val)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
