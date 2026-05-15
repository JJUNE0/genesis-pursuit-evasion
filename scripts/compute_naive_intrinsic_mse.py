"""Phase 3f / Paper §5 — Naive constant-velocity forecaster intrinsic MSE.

Measures per-τ MSE of ``naive_constvel`` baseline on val dataset (no learning,
pure math). For F2 fourth line in the forecaster accuracy curve.

Usage::

    python scripts/compute_naive_intrinsic_mse.py \\
        --data_dir data/forecaster_data_ams_v9_c7 \\
                   data/forecaster_data_ams_v9_c8 \\
                   data/forecaster_data_ams_v9_c14 \\
                   data/forecaster_data_ams_v9_c17 \\
                   data/forecaster_data_ams_v9_c20 \\
        --cfgs_pkl logs/phasef/ams_v9/cfgs.pkl \\
        --tau_values 10 25 50 \\
        --num_samples 512 \\
        --out_json logs/forecaster/diagnostics/naive_intrinsic_mse.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.forecaster.baselines.naive_constvel import naive_constvel_predict   # noqa: E402
from agents.forecaster.dataset import load_scales_from_dataset   # noqa: E402
from agents.forecaster.tau_unaware import constants as TUC   # noqa: E402
from agents.forecaster.tau_unaware.dataset import (   # noqa: E402
    build_dataset_and_collator as build_unaware_dataset_and_collator,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Naive const-vel intrinsic MSE")
    p.add_argument("--data_dir", type=str, nargs="+", required=True)
    p.add_argument("--cfgs_pkl", type=str, required=True,
                   help="To pull dt for extrapolation.")
    p.add_argument("--tau_values", type=int, nargs="+", default=[10, 25, 50])
    p.add_argument("--history_K", type=int, default=TUC.DEFAULT_HISTORY_K)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=str, required=True)
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dt + scales
    with open(args.cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)
    dt = float(cfgs["env_cfg"]["dt"])
    data_dirs = args.data_dir if len(args.data_dir) > 1 else args.data_dir[0]
    pos_scale, vel_scale, _ = load_scales_from_dataset(data_dirs)
    print(f"[naive-mse] dt={dt}  pos_scale={pos_scale}  vel_scale={vel_scale}", flush=True)

    # Use τ-unaware val dataset (defender_last + target at single t — same shape).
    dataset, collator = build_unaware_dataset_and_collator(
        data_dir=data_dirs, mode="val", history_K=int(args.history_K),
        in_memory=False, split_seed=42, rng_seed=int(args.seed),
    )
    rng = np.random.default_rng(int(args.seed))
    n = min(int(args.num_samples), len(dataset))
    chosen = rng.choice(len(dataset), n, replace=False).tolist()
    raw = [dataset[i] for i in chosen]

    results: dict[str, dict] = {"per_tau": {}, "meta": {
        "dt": dt, "pos_scale": pos_scale, "vel_scale": vel_scale,
        "n_samples": n, "history_K": int(args.history_K),
    }}
    for tau_eval in args.tau_values:
        tau_eval = int(tau_eval)
        # Build batch with this τ — defender_last is at t-τ (in t-1 attacker frame).
        batch, _ = collator(raw, tau_override=tau_eval)
        target = batch["target"]                                          # (B, 6)
        defender_last = batch["defender_last"]                            # (B, 1, 6)
        # Naive prediction (math only)
        pred = naive_constvel_predict(
            defender_last=defender_last, tau=tau_eval, dt=dt,
            pos_scale=pos_scale, vel_scale=vel_scale,
        )                                                                 # (B, 6)
        mse = F.mse_loss(pred, target).item()
        # Also report identity baseline for comparison
        identity_mse = F.mse_loss(defender_last.squeeze(1), target).item()
        results["per_tau"][str(tau_eval)] = {
            "mse": float(mse),
            "identity_mse_defender_last_to_target": float(identity_mse),
            "improvement_over_identity": float(identity_mse - mse),
        }
        print(f"  τ={tau_eval}: naive_mse={mse:.4f}  identity_mse={identity_mse:.4f}  "
              f"improve={identity_mse - mse:+.4f}", flush=True)

    out_path.write_text(json.dumps(results, indent=2))
    print(f"[naive-mse] saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
