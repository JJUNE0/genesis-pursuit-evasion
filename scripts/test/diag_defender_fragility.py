"""Phase A diagnostic: defender_v10 fragility test.

Hypothesis discrimination (3-way):
  A. bang-bang policy   — action histogram bimodal at ±1 saturation
  B. attacker OOD       — vs stationary attacker → def_crash ≈ 0
  C. intrinsic fragile  — vs stationary attacker → def_crash > 0.1, |ω| smooth high

Setup:
  - external StationaryAttacker (zero action → hover at spawn)
  - internal PretrainedDefender (v10 ckpt, deterministic — actor.eval())
  - env from c10 cfgs.pkl (P0 reward, Phase A spawn box)
  - 100 episodes, num_envs=32

Outputs:
  - summary print: termination breakdown, action stats, |ω| stats
  - histograms PNG: per-action-dim distribution, |ω| distribution
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from envs.pe_1v1.env import PursuitEvasion1v1Env   # noqa: E402
from envs.pe_1v1.scripted_attacker import StationaryAttacker   # noqa: E402
from envs.pe_1v1.scripted_defender import PretrainedDefender   # noqa: E402
from utils.drone_params import DroneParams   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="defender fragility 3-hypothesis diag")
    p.add_argument("--defender_ckpt", type=str, required=True)
    p.add_argument("--cfgs_pkl", type=str, required=True)
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--drone_yaml", type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    return p.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    with open(args.cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)
    env_cfg = dict(cfgs["env_cfg"])
    env_cfg["ego"] = "attacker"   # external attacker drives ego, defender internal
    obs_cfg = cfgs["obs_cfg"]
    reward_cfg = cfgs["reward_cfg"]
    command_cfg = cfgs["command_cfg"]
    drone_yaml = cfgs.get("drone_yaml", args.drone_yaml)

    params = DroneParams(drone_yaml)
    defender = PretrainedDefender(
        num_envs=args.num_envs, ckpt_path=args.defender_ckpt, device=gs.device,
    )
    attacker = StationaryAttacker(num_envs=args.num_envs, num_actions=4, device=gs.device)

    env = PursuitEvasion1v1Env(
        num_envs=args.num_envs, params=params,
        env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        defender_policy=defender, show_viewer=False,
    )

    counts = Counter()
    n_done = 0
    step_count = 0
    max_steps_safety = env.max_episode_length * (args.n_episodes // args.num_envs + 4)

    # Cumulative buffers (CPU)
    act_buf: list[np.ndarray] = []      # (num_envs, 4) per step
    omega_buf: list[np.ndarray] = []    # (num_envs, 3) per step

    env.reset()
    t0 = time.time()
    while n_done < args.n_episodes and step_count < max_steps_safety:
        ego_actions = attacker.step(
            ego_state=env._attacker_state_dict(),
            opponent_state=env._defender_state_dict(),
            g_mission=env.g_mission,
        )
        _o, _r, done, _e = env.step(ego_actions)
        act_buf.append(env.last_defender_action.detach().cpu().numpy().copy())
        omega_buf.append(env.defender_ang_vel.detach().cpu().numpy().copy())

        for k in ("mission", "captured", "att_crash",
                  "def_crash", "att_oob", "def_oob", "timeout"):
            counts[k] += int(env.last_term[k].sum().item())
        n_done += int(done.sum().item())
        step_count += 1

    wall = time.time() - t0
    actions = np.concatenate(act_buf, axis=0)   # (T*B, 4)
    omegas = np.concatenate(omega_buf, axis=0)  # (T*B, 3)
    omega_norm = np.linalg.norm(omegas, axis=1)  # (T*B,)

    # Per-action-dim stats — bang-bang signature: |action| > 0.9 mass > 50%
    act_abs = np.abs(actions)
    sat_frac = (act_abs > 0.9).mean(axis=0)        # (4,) per dim
    near_zero_frac = (act_abs < 0.1).mean(axis=0)   # (4,)
    act_mean = actions.mean(axis=0)
    act_std = actions.std(axis=0)

    summary = {
        "n_episodes":           n_done,
        "wall_sec":             round(wall, 2),
        "step_count":           step_count,
        "defender_ckpt":        args.defender_ckpt,
        "termination":          dict(counts),
        "rates": {
            k: round(counts[k] / max(1, n_done), 4)
            for k in ("captured", "def_crash", "def_oob", "att_crash", "timeout")
        },
        "action_per_dim": {
            "labels":           ["T_norm", "wx_ref", "wy_ref", "wz_ref"],
            "mean":              [round(float(x), 3) for x in act_mean],
            "std":               [round(float(x), 3) for x in act_std],
            "sat_frac_>0.9":     [round(float(x), 3) for x in sat_frac],
            "near_zero_<0.1":    [round(float(x), 3) for x in near_zero_frac],
        },
        "omega_norm": {
            "mean":   round(float(omega_norm.mean()), 3),
            "median": round(float(np.median(omega_norm)), 3),
            "p90":    round(float(np.percentile(omega_norm, 90)), 3),
            "p99":    round(float(np.percentile(omega_norm, 99)), 3),
            "max":    round(float(omega_norm.max()), 3),
        },
    }

    # Hypothesis verdict
    def_crash_rate = summary["rates"]["def_crash"]
    bang_bang_score = float(sat_frac.mean())   # mean fraction of saturated actions
    omega_p99 = summary["omega_norm"]["p99"]
    verdicts = []
    if bang_bang_score > 0.5:
        verdicts.append(f"A (bang-bang) STRONG: mean sat_frac={bang_bang_score:.2f}")
    elif bang_bang_score > 0.3:
        verdicts.append(f"A (bang-bang) WEAK: mean sat_frac={bang_bang_score:.2f}")
    if def_crash_rate < 0.05:
        verdicts.append(f"B (OOD) supported: stationary def_crash={def_crash_rate:.3f}")
    if def_crash_rate >= 0.05 and omega_p99 > 5.0:
        verdicts.append(
            f"C (fragile) STRONG: stationary def_crash={def_crash_rate:.3f}, |ω| p99={omega_p99:.2f}"
        )
    summary["hypothesis_verdict"] = verdicts

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Histograms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        labels = ["T_norm", "wx_ref", "wy_ref", "wz_ref"]
        for i, lbl in enumerate(labels):
            ax = axes[0, i] if i < 3 else axes[1, 0]
            ax.hist(actions[:, i], bins=50, color="tab:blue", alpha=0.7)
            ax.axvline(-0.9, ls=":", color="r"); ax.axvline(0.9, ls=":", color="r")
            ax.set_title(f"action[{lbl}]  sat={sat_frac[i]:.2f}")
            ax.set_xlim(-1.05, 1.05)
        ax = axes[1, 1]
        ax.hist(omega_norm, bins=50, color="tab:red", alpha=0.7)
        ax.axvline(5.0, ls=":", color="k", label="|ω|=5")
        ax.set_title(f"|ω|  mean={omega_norm.mean():.2f}  p99={omega_p99:.2f}")
        ax.legend()
        ax = axes[1, 2]
        ax.bar(["cap", "def_cr", "def_oob", "att_cr", "to"],
               [summary["rates"][k] for k in ("captured", "def_crash", "def_oob", "att_crash", "timeout")],
               color=["tab:green", "tab:red", "tab:orange", "tab:purple", "tab:gray"], alpha=0.7)
        ax.set_title(f"termination (n={n_done})")
        ax.set_ylim(0, 1)
        fig.suptitle(f"defender fragility diag — {Path(args.defender_ckpt).name}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / "diag.png", dpi=120)
        print(f"[diag] saved {out / 'diag.png'}")
    except Exception as e:
        print(f"[diag] plot skipped: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
