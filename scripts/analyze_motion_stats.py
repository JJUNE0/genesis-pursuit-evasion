"""Phase 3f / Paper §1 Intro Motivation — Defender motion vs capture-radius analysis.

Loads ams_v9 multi-cycle rollout dataset and extracts:
  - Defender velocity distribution (per-cycle + aggregate)
  - "How far does defender move during τ-step delay" / capture radius ratio
  - Per-τ "blind region" size (defender displacement) histograms

Output: stats JSON + KDE/histogram plot for paper Intro Figure ("Why Delay is
Critical").

Usage::

    python scripts/analyze_motion_stats.py \\
        --data_dir data/forecaster_data_ams_v9_c{7,8,14,17,20} \\
        --cfgs_pkl logs/phasef/ams_v9/cfgs.pkl \\
        --tau_values 10 25 50 \\
        --num_episodes_per_dir 1000 \\
        --out_dir logs/forecaster/diagnostics/motion_stats
"""

from __future__ import annotations

import argparse
import json
import pickle
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Defender motion stats for Intro Figure")
    p.add_argument("--data_dir", type=str, nargs="+", required=True)
    p.add_argument("--cfgs_pkl", type=str, required=True,
                   help="To pull capture_radius (env_cfg.d_capture) + dt.")
    p.add_argument("--tau_values", type=int, nargs="+", default=[10, 25, 50])
    p.add_argument("--num_episodes_per_dir", type=int, default=1000,
                   help="Subsample to keep stats compute bounded.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


def _episode_paths(data_dir: Path, n: int, seed: int) -> list[Path]:
    paths = sorted(data_dir.glob("episode_*.pt"))
    if len(paths) <= n:
        return paths
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), n, replace=False)
    return [paths[i] for i in sorted(idx)]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load env cfg for capture_radius + dt -------------------------
    with open(args.cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)
    capture_radius = float(cfgs["env_cfg"]["d_capture"])
    dt = float(cfgs["env_cfg"]["dt"])
    print(f"[motion-stats] capture_radius = {capture_radius:.3f} m, dt = {dt:.4f} s",
          flush=True)

    # --- Aggregate across all dirs -----------------------------------
    all_speeds: list[float] = []                                  # m/s, per-step defender speed
    # Per-τ displacement = ||defender_pos[t] - defender_pos[t-τ]|| (m)
    displacements: dict[int, list[float]] = {int(t): [] for t in args.tau_values}

    for d_str in args.data_dir:
        d = Path(d_str)
        eps = _episode_paths(d, int(args.num_episodes_per_dir), int(args.seed))
        print(f"[motion-stats] {d.name}: scanning {len(eps)} episodes", flush=True)
        for ep_path in eps:
            ep = torch.load(ep_path, weights_only=False, map_location="cpu")
            s_d = ep["s_d_world"]                                 # (L, 18) world raw
            L = s_d.shape[0]
            # Per-step defender speed (m/s)
            speeds = s_d[:, 3:6].norm(dim=-1).numpy()             # (L,)
            all_speeds.extend(speeds.tolist())

            # Per-τ displacement (m) — Euclidean distance over τ-step window
            pos = s_d[:, 0:3].numpy()                             # (L, 3)
            for tau in args.tau_values:
                tau = int(tau)
                if L > tau:
                    delta = pos[tau:] - pos[:-tau]                # (L-τ, 3)
                    disp = np.linalg.norm(delta, axis=-1)         # (L-τ,)
                    displacements[tau].extend(disp.tolist())

    speeds_arr = np.array(all_speeds)
    print(f"[motion-stats] total per-step speed samples: {len(speeds_arr):,}", flush=True)

    # --- Stats summary -----------------------------------------------
    stats: dict[str, object] = {
        "capture_radius_m": capture_radius,
        "dt_s": dt,
        "speed_m_per_s": {
            "mean": float(speeds_arr.mean()),
            "std":  float(speeds_arr.std()),
            "p10":  float(np.percentile(speeds_arr, 10)),
            "p50":  float(np.percentile(speeds_arr, 50)),
            "p90":  float(np.percentile(speeds_arr, 90)),
            "max":  float(speeds_arr.max()),
            "n":    int(len(speeds_arr)),
        },
        "per_tau_displacement_m": {},
        "per_tau_ratio_to_capture_radius": {},
    }
    for tau in args.tau_values:
        tau = int(tau)
        d = np.array(displacements[tau])
        ratio = d / capture_radius
        stats["per_tau_displacement_m"][str(tau)] = {
            "n": int(len(d)),
            "mean": float(d.mean()), "std": float(d.std()),
            "p10": float(np.percentile(d, 10)),
            "p50": float(np.percentile(d, 50)),
            "p90": float(np.percentile(d, 90)),
            "max": float(d.max()),
        }
        stats["per_tau_ratio_to_capture_radius"][str(tau)] = {
            "mean": float(ratio.mean()),
            "p50":  float(np.percentile(ratio, 50)),
            "p90":  float(np.percentile(ratio, 90)),
            "frac_exceeding_1": float((ratio > 1).mean()),   # fraction "blind region > capture radius"
            "frac_exceeding_5": float((ratio > 5).mean()),
            "frac_exceeding_10": float((ratio > 10).mean()),
        }

    (out_dir / "motion_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[motion-stats] saved {out_dir/'motion_stats.json'}", flush=True)

    # --- Print summary table -----------------------------------------
    print()
    print(f"{'τ':>4} | {'mean':>7} {'p50':>7} {'p90':>7} {'max':>7} | "
          f"{'ratio_mean':>10} {'%>1×r_cap':>10} {'%>5×r_cap':>10}")
    print("-" * 90)
    for tau in args.tau_values:
        tau = int(tau)
        d = stats["per_tau_displacement_m"][str(tau)]
        r = stats["per_tau_ratio_to_capture_radius"][str(tau)]
        print(f"{tau:>4} | {d['mean']:>7.2f} {d['p50']:>7.2f} {d['p90']:>7.2f} {d['max']:>7.2f} | "
              f"{r['mean']:>10.2f} {100*r['frac_exceeding_1']:>9.1f}% {100*r['frac_exceeding_5']:>9.1f}%")

    # --- Intro Figure: KDE/histogram of (displacement / capture_radius) per τ ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 30, 80)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.tau_values)))
    for tau, color in zip(args.tau_values, colors):
        tau = int(tau)
        d = np.array(displacements[tau])
        ratio = d / capture_radius
        ax.hist(ratio, bins=bins, density=True, alpha=0.5,
                color=color, label=f"τ={tau} ({tau*dt*1000:.0f} ms)")
    # Danger zone: ratio > 1 means defender moves more than capture radius during delay
    ax.axvspan(1.0, 30.0, color="red", alpha=0.08, zorder=0)
    ax.axvline(1.0, color="red", linestyle="--", lw=1.5, alpha=0.7,
               label="r_cap (reactive limit)")
    ax.set_xlabel("Defender displacement during τ-step delay  /  capture radius",
                  fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(
        "Why delay is critical — defender motion overruns reactive interception window",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, min(30, max(np.percentile(np.array(displacements[max(args.tau_values)])
                                              / capture_radius, 99), 5)))
    fig.tight_layout()
    fig_path = out_dir / "intro_motion_motivation.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"[motion-stats] saved {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
