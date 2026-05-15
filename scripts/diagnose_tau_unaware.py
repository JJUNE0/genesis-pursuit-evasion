"""Phase 3f — Diagnostic suite for τ-unaware forecaster (TODO §3f, 2026-05-14).

Runs 4 diagnostics on a trained τ-unaware ckpt:

  D1) Apples-to-apples MSE vs Phase 3c τ-aware reference (same val set,
      same τ values). Uses τ-aware's last-step prediction as comparable
      single-step quantity.
  D2) Shuffled defender_last test — does the model actually use defender_last
      or rely solely on attacker_history? Compares MSE on normal vs row-
      shuffled defender_last batches.
  D3) τ-extreme boundary tests — τ=1 (smallest available) sanity (model should
      excel here, since defender_last ≈ current defender). Optional: τ=0
      override via direct slice (collator forbids τ=0; we build sample manually).
  D4) Variance check — sample N=10 Euler trajectories from the same input;
      report per-τ std. Larger std at larger τ = flow's uncertainty calibration.

Usage::

    python scripts/diagnose_tau_unaware.py \\
        --ckpt_unaware logs/forecaster/tau_unaware_v1/best.pt \\
        --ckpt_aware   logs/forecaster/ams_v9_multi_v1/best.pt \\
        --data_dir data/forecaster_data_ams_v9_c7 \\
                   data/forecaster_data_ams_v9_c8 \\
                   data/forecaster_data_ams_v9_c14 \\
                   data/forecaster_data_ams_v9_c17 \\
                   data/forecaster_data_ams_v9_c20 \\
        --out_dir logs/forecaster/diagnostics/tau_unaware_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.forecaster.dataset import (   # noqa: E402
    build_dataset_and_collator as build_aware_dataset_and_collator,
    load_scales_from_dataset,
)
from agents.forecaster.model import ForecasterModel   # noqa: E402
from agents.forecaster.rectified_flow import sample_via_euler as sample_aware   # noqa: E402
from agents.forecaster.trainer import load_ckpt as load_aware_ckpt   # noqa: E402
from agents.forecaster.tau_unaware import (   # noqa: E402
    TauUnawareDeterministicModel,
    TauUnawareForecasterModel,
    build_dataset_and_collator as build_unaware_dataset_and_collator,
    build_sample_unaware,
    load_ckpt as load_unaware_ckpt,
    sample_via_euler as sample_unaware,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="τ-unaware forecaster diagnostics")
    p.add_argument("--ckpt_unaware", type=str, required=True)
    p.add_argument("--ckpt_aware", type=str, default=None,
                   help="Optional Phase 3c τ-aware ckpt for D1.")
    p.add_argument("--ckpt_det", type=str, default=None,
                   help="Optional τ-unaware deterministic ckpt for sanity.")
    p.add_argument("--data_dir", type=str, nargs="+", required=True)
    p.add_argument("--history_K", type=int, default=50)
    p.add_argument("--tau_evals", type=int, nargs="+", default=[10, 25, 50])
    p.add_argument("--num_samples", type=int, default=512,
                   help="Val samples per (diagnostic, τ) cell.")
    p.add_argument("--num_euler_steps", type=int, default=10)
    p.add_argument("--num_variance_samples", type=int, default=10,
                   help="N for D4 (per-input replicas).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=str, required=True)
    return p.parse_args()


@torch.no_grad()
def _flow_predict_mean(model, context, num_steps: int, num_samples: int = 4) -> torch.Tensor:
    """Median over num_samples Euler runs for stable comparison."""
    preds = [sample_unaware(model=model, context=context, num_steps=num_steps,
                            target_dim=6) for _ in range(num_samples)]
    return torch.stack(preds, dim=0).median(dim=0).values   # (B, 6)


# ---------------------------------------------------------------------------
# D1 — Apples-to-apples vs τ-aware
# ---------------------------------------------------------------------------

@torch.no_grad()
def diag_d1_apples_to_apples(
    model_unaware, model_aware, data_dir, history_K, tau_evals,
    num_samples, num_euler_steps, seed, device,
) -> dict[str, Any]:
    """Same val episode subset, same τ values, single-step MSE for both."""
    print(f"\n=== D1 — apples-to-apples vs τ-aware (n={num_samples}) ===", flush=True)
    out: dict[str, Any] = {"per_tau": {}}

    # Build both datasets — use the SAME val split (split_seed=42) and SAME
    # subset indices so the underlying (ep, t) pairs match between forecasters.
    aware_dataset, aware_collator = build_aware_dataset_and_collator(
        data_dir=data_dir, mode="val", tau_max=history_K, in_memory=False,
        split_seed=42, rng_seed=seed,
    )
    unaware_dataset, unaware_collator = build_unaware_dataset_and_collator(
        data_dir=data_dir, mode="val", history_K=history_K, in_memory=False,
        split_seed=42, rng_seed=seed,
    )
    rng = np.random.default_rng(seed)
    n_aware = len(aware_dataset)
    n_unaware = len(unaware_dataset)
    n = min(num_samples, n_aware, n_unaware)
    print(f"  aware val={n_aware}  unaware val={n_unaware}  → using n={n}", flush=True)

    # Aware: indices need t ≥ τ_max=history_K (already enforced).
    # Unaware: indices need t ∈ [history_K, L-2] (target uses t+1).
    # Take first-N in unaware (its constraint is stricter), then use SAME (ep_id, t)
    # pairs for aware (any t ≥ history_K is valid for aware).
    chosen_unaware = rng.choice(n_unaware, n, replace=False).tolist()
    raw_unaware = [unaware_dataset[i] for i in chosen_unaware]   # list of (dir, ep, t)

    # Find matching indices in aware dataset for the same (dir, ep, t).
    aware_lookup = {tuple(idx): k for k, idx in enumerate(aware_dataset.sample_indices)}
    raw_aware = []
    matched = 0
    for sidx in raw_unaware:
        key = tuple(sidx)
        if key in aware_lookup:
            raw_aware.append(aware_dataset.sample_indices[aware_lookup[key]])
            matched += 1
    print(f"  matched (dir,ep,t) pairs aware↔unaware: {matched}/{n}", flush=True)
    if matched < 8:
        return {"error": "too few matched samples", "matched": matched}

    # Trim unaware list to matched (for symmetry — both use SAME pairs)
    matched_unaware_raw = []
    matched_aware_raw = []
    for sidx in raw_unaware:
        if tuple(sidx) in aware_lookup:
            matched_unaware_raw.append(sidx)
            matched_aware_raw.append(aware_dataset.sample_indices[aware_lookup[tuple(sidx)]])

    for tau_eval in tau_evals:
        tau_eval = int(tau_eval)
        # Unaware MSE
        u_batch, _ = unaware_collator(matched_unaware_raw, tau_override=tau_eval)
        u_batch = {k: v.to(device) for k, v in u_batch.items()}
        u_target = u_batch["target"]
        u_context = {k: u_batch[k] for k in ("attacker_history", "defender_last", "action_history")}
        u_pred = _flow_predict_mean(model_unaware, u_context, num_euler_steps, num_samples=4)
        u_mse = F.mse_loss(u_pred, u_target).item()

        # Aware MSE on its last-step prediction (analog of "current defender estimate")
        a_batch, _ = aware_collator(matched_aware_raw, tau_override=tau_eval)
        a_batch = {k: v.to(device) for k, v in a_batch.items()}
        a_target = a_batch["target"]                       # (B, τ, 6)
        a_context = {k: a_batch[k] for k in ("attacker_history", "defender_delayed", "action_history")}
        a_preds = []
        for _ in range(4):
            a_preds.append(sample_aware(model=model_aware, context=a_context,
                                        tau=tau_eval, num_steps=num_euler_steps,
                                        target_dim=6))
        a_pred = torch.stack(a_preds, dim=0).median(dim=0).values    # (B, τ, 6)
        a_mse_last = F.mse_loss(a_pred[:, -1, :], a_target[:, -1, :]).item()

        out["per_tau"][int(tau_eval)] = {
            "unaware_mse": u_mse,
            "aware_last_mse": a_mse_last,
            "ratio_unaware_over_aware": u_mse / max(1e-9, a_mse_last),
        }
        print(f"  τ={tau_eval}: unaware={u_mse:.4f}  aware_last={a_mse_last:.4f}  "
              f"ratio={u_mse / max(1e-9, a_mse_last):.2f}×", flush=True)
    return out


# ---------------------------------------------------------------------------
# D2 — Shuffled defender_last test
# ---------------------------------------------------------------------------

@torch.no_grad()
def diag_d2_shuffle_defender_last(
    model_unaware, data_dir, history_K, tau_evals,
    num_samples, num_euler_steps, seed, device,
) -> dict[str, Any]:
    """If model relies on defender_last, shuffling rows should hurt MSE."""
    print(f"\n=== D2 — shuffled defender_last test (n={num_samples}) ===", flush=True)
    dataset, collator = build_unaware_dataset_and_collator(
        data_dir=data_dir, mode="val", history_K=history_K, in_memory=False,
        split_seed=42, rng_seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    n = min(num_samples, len(dataset))
    chosen = rng.choice(len(dataset), n, replace=False).tolist()
    raw = [dataset[i] for i in chosen]

    out: dict[str, Any] = {"per_tau": {}}
    for tau_eval in tau_evals:
        tau_eval = int(tau_eval)
        batch, _ = collator(raw, tau_override=tau_eval)
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch["target"]
        # Normal
        ctx_normal = {k: batch[k] for k in ("attacker_history", "defender_last", "action_history")}
        pred_normal = _flow_predict_mean(model_unaware, ctx_normal, num_euler_steps, num_samples=4)
        mse_normal = F.mse_loss(pred_normal, target).item()

        # Shuffled defender_last (row permutation within batch)
        B = batch["defender_last"].shape[0]
        torch.manual_seed(seed + tau_eval)
        perm = torch.randperm(B, device=device)
        ctx_shuffled = dict(ctx_normal)
        ctx_shuffled["defender_last"] = batch["defender_last"][perm]
        pred_shuffled = _flow_predict_mean(model_unaware, ctx_shuffled, num_euler_steps, num_samples=4)
        mse_shuffled = F.mse_loss(pred_shuffled, target).item()

        out["per_tau"][int(tau_eval)] = {
            "mse_normal": mse_normal,
            "mse_shuffled": mse_shuffled,
            "delta": mse_shuffled - mse_normal,
            "ratio_shuffled_over_normal": mse_shuffled / max(1e-9, mse_normal),
        }
        print(f"  τ={tau_eval}: normal={mse_normal:.4f}  shuffled={mse_shuffled:.4f}  "
              f"Δ={mse_shuffled - mse_normal:+.4f}  ratio={mse_shuffled / max(1e-9, mse_normal):.2f}×",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# D3 — τ extremes
# ---------------------------------------------------------------------------

@torch.no_grad()
def diag_d3_extremes(
    model_unaware, data_dir, history_K,
    num_samples, num_euler_steps, seed, device,
) -> dict[str, Any]:
    """τ=1 sanity: defender_last should ≈ current defender (1 step delta)
    → MSE should be smallest among all τ. Also report τ=history_K extreme.
    """
    print(f"\n=== D3 — τ extremes (n={num_samples}) ===", flush=True)
    dataset, collator = build_unaware_dataset_and_collator(
        data_dir=data_dir, mode="val", history_K=history_K, in_memory=False,
        split_seed=42, rng_seed=seed,
    )
    rng = np.random.default_rng(seed + 2)
    n = min(num_samples, len(dataset))
    chosen = rng.choice(len(dataset), n, replace=False).tolist()
    raw = [dataset[i] for i in chosen]

    out: dict[str, Any] = {"per_tau": {}}
    for tau_eval in (1, history_K):
        batch, _ = collator(raw, tau_override=tau_eval)
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch["target"]
        ctx = {k: batch[k] for k in ("attacker_history", "defender_last", "action_history")}
        pred = _flow_predict_mean(model_unaware, ctx, num_euler_steps, num_samples=4)
        mse = F.mse_loss(pred, target).item()

        # Sanity: distance between defender_last and target (no model involved).
        # If τ=1, this distance should be smaller than τ=K (single-step displacement).
        raw_dist = F.mse_loss(batch["defender_last"].squeeze(1), target).item()
        out["per_tau"][int(tau_eval)] = {
            "model_mse": mse,
            "raw_distance_defender_last_to_target": raw_dist,
        }
        print(f"  τ={tau_eval}: model_mse={mse:.4f}  raw|defender_last-target|²={raw_dist:.4f}",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# D4 — Variance check (Flow uncertainty calibration)
# ---------------------------------------------------------------------------

@torch.no_grad()
def diag_d4_variance(
    model_unaware, data_dir, history_K, tau_evals,
    num_samples, num_euler_steps, num_variance_samples, seed, device,
) -> dict[str, Any]:
    """Same input, N Euler runs → per-τ std. Larger std at larger τ = good
    uncertainty calibration."""
    print(f"\n=== D4 — variance check (n={num_samples}, replicas={num_variance_samples}) ===",
          flush=True)
    dataset, collator = build_unaware_dataset_and_collator(
        data_dir=data_dir, mode="val", history_K=history_K, in_memory=False,
        split_seed=42, rng_seed=seed,
    )
    rng = np.random.default_rng(seed + 3)
    n = min(num_samples, len(dataset))
    chosen = rng.choice(len(dataset), n, replace=False).tolist()
    raw = [dataset[i] for i in chosen]

    out: dict[str, Any] = {"per_tau": {}}
    for tau_eval in tau_evals:
        tau_eval = int(tau_eval)
        batch, _ = collator(raw, tau_override=tau_eval)
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = {k: batch[k] for k in ("attacker_history", "defender_last", "action_history")}
        preds = [sample_unaware(model=model_unaware, context=ctx,
                                num_steps=num_euler_steps, target_dim=6)
                 for _ in range(int(num_variance_samples))]
        stack = torch.stack(preds, dim=0)                           # (N, B, 6)
        std_per_dim_per_sample = stack.std(dim=0)                   # (B, 6)
        mean_std = std_per_dim_per_sample.mean().item()             # scalar — averaged across batch+dim
        per_dim_std = std_per_dim_per_sample.mean(dim=0).tolist()   # (6,) — split per output dim
        out["per_tau"][int(tau_eval)] = {
            "mean_std_across_batch_dim": mean_std,
            "per_output_dim_std": per_dim_std,
        }
        print(f"  τ={tau_eval}: mean_std={mean_std:.4f}  "
              f"per_dim_std=[{','.join(f'{x:.4f}' for x in per_dim_std)}]",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = args.data_dir if len(args.data_dir) > 1 else args.data_dir[0]

    # Load τ-unaware model
    model_unaware = TauUnawareForecasterModel(history_K=int(args.history_K)).to(device)
    payload_u = load_unaware_ckpt(args.ckpt_unaware, model=model_unaware,
                                  optimizer=None, map_location=device)
    model_unaware.eval()
    print(f"[diag] τ-unaware ckpt = {args.ckpt_unaware} (step {payload_u.get('step', '?')})",
          flush=True)

    # Load τ-aware model (optional D1)
    model_aware = None
    if args.ckpt_aware:
        model_aware = ForecasterModel(tau_max=int(args.history_K)).to(device)
        payload_a = load_aware_ckpt(args.ckpt_aware, model=model_aware,
                                    optimizer=None, map_location=device)
        model_aware.eval()
        print(f"[diag] τ-aware ckpt = {args.ckpt_aware} (step {payload_a.get('step', '?')})",
              flush=True)

    results: dict[str, Any] = {
        "ckpt_unaware": args.ckpt_unaware,
        "ckpt_aware": args.ckpt_aware,
        "data_dir": list(args.data_dir),
        "history_K": int(args.history_K),
        "tau_evals": list(args.tau_evals),
        "num_samples": int(args.num_samples),
        "num_euler_steps": int(args.num_euler_steps),
    }

    # D1
    if model_aware is not None:
        results["D1_apples_to_apples"] = diag_d1_apples_to_apples(
            model_unaware=model_unaware, model_aware=model_aware,
            data_dir=data_dirs, history_K=int(args.history_K),
            tau_evals=list(args.tau_evals), num_samples=int(args.num_samples),
            num_euler_steps=int(args.num_euler_steps), seed=int(args.seed), device=device,
        )
    else:
        print("\n=== D1 SKIPPED (no --ckpt_aware) ===", flush=True)

    # D2
    results["D2_shuffle_defender_last"] = diag_d2_shuffle_defender_last(
        model_unaware=model_unaware, data_dir=data_dirs, history_K=int(args.history_K),
        tau_evals=list(args.tau_evals), num_samples=int(args.num_samples),
        num_euler_steps=int(args.num_euler_steps), seed=int(args.seed), device=device,
    )

    # D3
    results["D3_extremes"] = diag_d3_extremes(
        model_unaware=model_unaware, data_dir=data_dirs, history_K=int(args.history_K),
        num_samples=int(args.num_samples), num_euler_steps=int(args.num_euler_steps),
        seed=int(args.seed), device=device,
    )

    # D4
    results["D4_variance"] = diag_d4_variance(
        model_unaware=model_unaware, data_dir=data_dirs, history_K=int(args.history_K),
        tau_evals=list(args.tau_evals), num_samples=int(args.num_samples),
        num_euler_steps=int(args.num_euler_steps),
        num_variance_samples=int(args.num_variance_samples),
        seed=int(args.seed), device=device,
    )

    out_path = out_dir / "diagnostics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[diag] DONE — saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
