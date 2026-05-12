"""Compute Nash Proximity Score (NPS) from a cross-eval matrix.

Inputs:
    --root  Directory containing ``a{i}_d{j}.json`` from eval_head_to_head.py.
    --cycles  Space-separated cycle indices (e.g. ``5 6 7 8 9``).

Outputs to stdout:
    1. Defender win-rate matrix M[i,j] (rows = attacker cycle, cols = defender cycle).
    2. μ(M), std(M), Wilson noise floor.
    3. Raw NPS (clip [0,1]) + bootstrap-1000 95% CI on resampled cells.
    4. Noise-corrected NPS (binomial variance subtracted).
    5. Pass/fail vs Nash criteria (NPS_raw ≥ 0.7 AND μ ∈ [0.4, 0.6]).

Formulas (Phase F.1 paper-ready):
    NPS_raw = clip(1 - 2·std(M) / sqrt(μ(1-μ)), 0, 1)
    std_true² = max(0, std(M)² - μ(1-μ)/n_ep)
    NPS_corrected = clip(1 - 2·std_true / sqrt(μ(1-μ)), 0, 1)

Reasoning notes (사용자 critique, 2026-05-11):
    - Cell-level bootstrap (resample 25 matrix entries with replacement)
      is appropriate because each cell's d_win is already a binomial mean.
    - NPS_raw paired with μ ∈ [0.4, 0.6] is the primary claim (reviewer-friendly).
    - NPS_corrected goes to appendix to defend against "noise inflates std" critiques.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _wilson_ci_halfwidth(p: float, n: int) -> float:
    z = 1.96
    return z * float(np.sqrt(p * (1.0 - p) / max(n, 1)))


def _nps(mu: float, std: float) -> float:
    denom = float(np.sqrt(max(mu * (1.0 - mu), 1e-12)))
    return float(np.clip(1.0 - 2.0 * std / denom, 0.0, 1.0))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--cycles", required=True, type=int, nargs="+")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cycles = list(args.cycles)
    k = len(cycles)
    M = np.zeros((k, k), dtype=np.float64)                                    # shape: (k, k)
    n_ep = None
    for i, ci in enumerate(cycles):
        for j, cj in enumerate(cycles):
            f = args.root / f"a{ci}_d{cj}.json"
            d = json.loads(f.read_text())
            M[i, j] = float(d["defender_winrate"])
            if n_ep is None:
                n_ep = int(d["n_episodes"])

    mu = float(M.mean())
    std = float(M.std(ddof=0))
    cell_ci = _wilson_ci_halfwidth(mu, n_ep)
    nps_raw = _nps(mu, std)

    # Noise-corrected
    binomial_var = mu * (1.0 - mu) / max(n_ep, 1)
    std_true_sq = max(0.0, std * std - binomial_var)
    std_true = float(np.sqrt(std_true_sq))
    nps_corr = _nps(mu, std_true)

    # Bootstrap CI on raw NPS (cell-level resample with replacement)
    rng = np.random.default_rng(args.seed)
    flat = M.flatten()
    n_cells = flat.size
    samples = []
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, n_cells, size=n_cells)
        sub = flat[idx]
        mu_s = float(sub.mean())
        std_s = float(sub.std(ddof=0))
        samples.append(_nps(mu_s, std_s))
    ci_lo, ci_hi = np.percentile(samples, [2.5, 97.5])

    print("=" * 64)
    print(f"Cross-eval matrix M[i,j] = defender_winrate(attacker_v{{i}} vs defender_v{{j}})")
    print(f"  rows i (attacker cycle): {cycles}")
    print(f"  cols j (defender cycle): {cycles}")
    print(f"  cells: {k}x{k} = {k*k}, n_episodes/cell = {n_ep}")
    print()
    header = "      " + "  ".join(f"d_v{c:<2d}" for c in cycles)
    print(header)
    for i, ci in enumerate(cycles):
        row = "  ".join(f"{M[i,j]:.3f}" for j in range(k))
        print(f"a_v{ci}  {row}")
    print()
    print(f"μ(M)       = {mu:.4f}   (per-cell Wilson ±{cell_ci:.3f})")
    print(f"std(M)     = {std:.4f}")
    print(f"std_true   = {std_true:.4f}   (binomial noise removed)")
    print()
    print(f"NPS (raw)            = {nps_raw:.3f}   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"NPS (noise-corrected)= {nps_corr:.3f}   (appendix)")
    print()
    mu_ok = 0.4 <= mu <= 0.6
    nps_ok = nps_raw >= 0.7
    verdict = "PASS — Nash range" if (mu_ok and nps_ok) else "FAIL"
    print(f"μ ∈ [0.4, 0.6]?  {mu_ok}")
    print(f"NPS_raw ≥ 0.7?   {nps_ok}")
    print(f"Verdict: {verdict}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
