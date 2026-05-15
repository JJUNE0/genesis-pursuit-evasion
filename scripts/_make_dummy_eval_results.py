"""Generate dummy results.json for plot_paper_figures.py F4-F7 sanity testing.

Synthetic data based on plausible expected outcomes from the eval scripts.
Used ONLY to validate plot scripts before real eval finishes.

Usage::

    python scripts/_make_dummy_eval_results.py --target eval_5fc \\
        --out /tmp/dummy_eval_5fc/results.json
    python scripts/_make_dummy_eval_results.py --target eval_random_delay \\
        --out /tmp/dummy_eval_random_delay/results.json
    python scripts/_make_dummy_eval_results.py --target eval_misspec \\
        --out /tmp/dummy_eval_misspec/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PAIRS = ["a5_d6", "a7_d8", "a8_d9", "a9_d10"]
TAUS = [10, 25, 50]
SEEDS = [42, 0, 1]
FC_TYPES = ["no_fc", "naive_constvel", "tau_aware_flow", "tau_unaware_flow", "tau_unaware_det"]
SCENARIOS = ["fixed_tau50", "random_per_episode", "random_per_step"]


def _rates_for(pair, tau, fc_type, seed) -> dict:
    """Plausible synthetic rates for sanity testing only."""
    rng = np.random.default_rng(hash((pair, tau, fc_type, seed)) % (2**32))
    # Base mission rate by pair (a5_d6 highest, a9_d10 lowest)
    pair_base = {"a5_d6": 0.40, "a7_d8": 0.45, "a8_d9": 0.30, "a9_d10": 0.25}[pair]
    # τ penalty
    tau_penalty = {10: 0.05, 25: 0.10, 50: 0.18}[tau]
    # Forecaster benefit
    fc_recovery = {
        "no_fc": 0.0,
        "naive_constvel": -0.02,    # naive worse than no_fc (per actual data)
        "tau_aware_flow": 0.10,
        "tau_unaware_flow": 0.09,    # almost matches τ-aware
        "tau_unaware_det": 0.06,
    }[fc_type]
    mission = pair_base - tau_penalty + fc_recovery + rng.normal(0, 0.02)
    mission = float(np.clip(mission, 0.0, 1.0))
    capture = float(np.clip(0.50 - mission * 0.4 + rng.normal(0, 0.03), 0, 1))
    oob = float(np.clip(0.10 + tau_penalty * 0.3 + rng.normal(0, 0.02), 0, 1))
    timeout = float(np.clip(1.0 - mission - capture - oob, 0, 1))
    mean_len = float(300 + tau * 1.5 + rng.normal(0, 20))
    return {
        "mission_rate": mission, "capture_rate": capture,
        "oob_crash_rate": oob, "timeout_rate": timeout,
        "mean_ep_length": mean_len, "n_collected": 500,
    }


def make_eval_5fc():
    cells = []
    for seed in SEEDS:
        for pair in PAIRS:
            for tau in TAUS:
                for fc in FC_TYPES:
                    cells.append({
                        "seed": seed, "pair": pair, "tau": tau, "fc_type": fc,
                        "rates": _rates_for(pair, tau, fc, seed),
                        "counts": {"mission": 200, "capture": 200, "oob_crash": 50, "timeout": 50},
                        "n_collected": 500, "elapsed_s": 30.0,
                    })
    return {"args": {"target": "eval_5fc"}, "scales": [0.2, 0.2, 0.31831], "dt": 0.005, "cells": cells}


def make_eval_random_delay():
    cells = []
    fcs = ["no_fc", "naive_constvel", "tau_aware_flow", "tau_unaware_flow"]
    for seed in SEEDS:
        for pair in PAIRS:
            for scen in SCENARIOS:
                for fc in fcs:
                    # τ proxy for plausibility (avg τ across scenario)
                    tau_proxy = {"fixed_tau50": 50, "random_per_episode": 25,
                                 "random_per_step": 25}[scen]
                    rates = _rates_for(pair, tau_proxy, fc, seed)
                    # Random scenarios incur extra penalty for τ-aware (mismatch),
                    # less for τ-unaware (designed for it)
                    if scen != "fixed_tau50":
                        if fc == "tau_aware_flow":
                            rates["mission_rate"] = max(0, rates["mission_rate"] - 0.05)
                        elif fc == "tau_unaware_flow":
                            rates["mission_rate"] = max(0, rates["mission_rate"] - 0.01)
                        elif fc == "no_fc":
                            rates["mission_rate"] = max(0, rates["mission_rate"] - 0.02)
                    cells.append({
                        "seed": seed, "pair": pair, "scenario": scen, "fc_type": fc,
                        "rates": rates,
                        "counts": {"mission": 200, "capture": 200, "oob_crash": 50, "timeout": 50},
                        "n_collected": 500, "elapsed_s": 30.0,
                    })
    return {"args": {"target": "eval_random_delay"}, "scales": [0.2, 0.2, 0.31831], "dt": 0.005, "cells": cells}


def make_eval_misspec():
    cells = []
    for seed in SEEDS:
        for pair in PAIRS:
            for ta in TAUS:
                # τ-aware grid
                for ti in TAUS:
                    rates = _rates_for(pair, ta, "tau_aware_flow", seed)
                    # Misspec penalty: |ta - ti| / 50 → 0..0.8
                    mismatch_pen = abs(ta - ti) / 50 * 0.15
                    rates["mission_rate"] = max(0, rates["mission_rate"] - mismatch_pen)
                    cells.append({
                        "seed": seed, "pair": pair,
                        "tau_actual": ta, "tau_input": ti,
                        "fc_type": "tau_aware_flow",
                        "rates": rates,
                        "counts": {}, "n_collected": 500, "elapsed_s": 30.0,
                    })
                # τ-unaware adjacent
                rates_u = _rates_for(pair, ta, "tau_unaware_flow", seed)
                cells.append({
                    "seed": seed, "pair": pair,
                    "tau_actual": ta, "tau_input": None,
                    "fc_type": "tau_unaware_flow",
                    "rates": rates_u,
                    "counts": {}, "n_collected": 500, "elapsed_s": 30.0,
                })
    return {"args": {"target": "eval_misspec"}, "scales": [0.2, 0.2, 0.31831], "dt": 0.005, "cells": cells}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=["eval_5fc", "eval_random_delay", "eval_misspec"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dispatch = {
        "eval_5fc": make_eval_5fc,
        "eval_random_delay": make_eval_random_delay,
        "eval_misspec": make_eval_misspec,
    }
    data = dispatch[args.target]()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"[dummy] {args.target} → {out}  ({len(data['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
