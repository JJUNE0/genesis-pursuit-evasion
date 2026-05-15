"""PR-N — head-to-head evaluation of a pretrained attacker vs a pretrained defender.

Used by ``scripts/train_ams_drl.py`` after each alternating cycle to measure
the Nash gap (attacker_winrate - defender_winrate). Also runnable standalone
to grade a checkpoint pair.

# AMS-DRL terminal classification (Phase 5)
* attacker_win = mission           (attacker reached g_mission)
* defender_win = captured | att_crash | att_oob   (defender intercepted, OR
                                                   attacker crashed/oob)
* draw         = timeout | def_crash | def_oob    (no decisive outcome, OR
                                                   defender removed itself)

# Stat power
``--n_episodes`` is interpreted as the **absolute total** episode count (not
per-env). At p=0.5, n=400 → SE ≈ 0.025, 95% half-width ≈ 0.049 — Wilson CI
also reported in the JSON for proper error bars.

# B2 fix (PR-N CCG round 1)
``ego="attacker"`` so the env's internal ``_attacker_policy`` is None. The
external ``PretrainedAttacker`` produces the ego_actions and is **not** reset
by ``env.reset_idx()``. We manually call ``attacker.reset(done_idx)`` after
each ``env.step()`` to keep its ``last_action`` buffer in sync with the env's
``last_attacker_action`` (which is zeroed on reset). Otherwise stale
last_action values from finished episodes leak into the next attacker obs,
producing garbage actions.

Usage::

    python scripts/eval_head_to_head.py \\
        --attacker_ckpt logs/pe1v1_v0/attacker_v0/<run>/attacker_v0.pt \\
        --defender_ckpt logs/pe1v1_v0/defender_v0/<run>/defender_v0.pt \\
        --n_episodes 400 --num_envs 256 --seed 0 \\
        --output_json /tmp/h2h.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs   # noqa: E402

from envs.pe_1v1.env import PursuitEvasion1v1Env   # noqa: E402
from envs.pe_1v1.scripted_attacker import PretrainedAttacker   # noqa: E402
from envs.pe_1v1.scripted_defender import PretrainedDefender   # noqa: E402
from utils.drone_params import DroneParams   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PR-N head-to-head eval")
    p.add_argument("--attacker_ckpt", type=str, required=True,
                   help="Path to attacker_v*.pt (frozen actor).")
    p.add_argument("--defender_ckpt", type=str, required=True,
                   help="Path to defender_v*.pt (frozen actor).")
    p.add_argument("--cfgs_pkl", type=str, default="",
                   help="Override env cfg source. Default = cfgs.pkl next to attacker_ckpt.")
    p.add_argument("--n_episodes", type=int, default=400,
                   help="Absolute total episode count (NOT per-env). 400 → SE ≈ 0.025 at p=0.5.")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--max_steps_per_episode", type=int, default=0,
                   help="0 = use env.max_episode_length. Otherwise hard cap on rollout length.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--backend", type=str, default="cpu", choices=["gpu", "cpu"])
    p.add_argument("--output_json", type=str, default="",
                   help="If set, write JSON result to this path. Always echoed to stdout.")
    p.add_argument(
        "--drone_yaml", type=str,
        default=str(ROOT / "configs" / "drones" / "nova.yaml"),
    )
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        if dirty:
            sha = f"{sha}-dirty"
    except Exception:
        pass
    return sha


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. ``n==0`` → (0,0)."""
    if n <= 0:
        return (0.0, 0.0)
    if wins < 0 or wins > n:
        raise ValueError(f"wilson_ci: wins={wins} out of [0, n={n}]")
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (round(centre - margin, 4), round(centre + margin, 4))


def _resolve_cfgs(attacker_ckpt: Path, override: str) -> Path:
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"--cfgs_pkl not found: {p}")
        return p
    # resolve symlinks first; cfgs.pkl is co-located with the actor file
    cfgs = attacker_ckpt.resolve().parent / "cfgs.pkl"
    if not cfgs.is_file():
        raise FileNotFoundError(
            f"No cfgs.pkl next to attacker_ckpt ({attacker_ckpt}). "
            "Pass --cfgs_pkl explicitly."
        )
    return cfgs


@torch.no_grad()
def main() -> int:
    args = parse_args()
    _seed_all(args.seed)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    attacker_ckpt = Path(args.attacker_ckpt).resolve()
    defender_ckpt = Path(args.defender_ckpt).resolve()
    if not attacker_ckpt.is_file():
        raise FileNotFoundError(f"--attacker_ckpt not found: {attacker_ckpt}")
    if not defender_ckpt.is_file():
        raise FileNotFoundError(f"--defender_ckpt not found: {defender_ckpt}")

    cfgs_pkl = _resolve_cfgs(attacker_ckpt, args.cfgs_pkl)
    with open(cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)

    env_cfg = cfgs["env_cfg"]
    obs_cfg = cfgs["obs_cfg"]
    reward_cfg = cfgs["reward_cfg"]
    command_cfg = cfgs["command_cfg"]
    drone_yaml = cfgs.get("drone_yaml", args.drone_yaml)

    # ego="attacker" so external PretrainedAttacker drives ego_actions and the
    # env runs PretrainedDefender internally (will get its own reset_idx call).
    env_cfg = dict(env_cfg)
    env_cfg["ego"] = "attacker"

    params = DroneParams(drone_yaml)
    defender = PretrainedDefender(
        num_envs=args.num_envs, ckpt_path=str(defender_ckpt), device=gs.device,
    )
    attacker = PretrainedAttacker(
        num_envs=args.num_envs, ckpt_path=str(attacker_ckpt), device=gs.device,
    )

    env = PursuitEvasion1v1Env(
        num_envs=args.num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        defender_policy=defender,
        show_viewer=False,
    )

    # Counters — per-side terminal categories from env.last_term snapshots.
    counts = {
        "mission": 0, "captured": 0,
        "att_crash": 0, "def_crash": 0,
        "att_oob": 0, "def_oob": 0,
        "timeout": 0,
    }

    # B1 fix — n_episodes is absolute total, NOT × num_envs.
    target_episodes = int(args.n_episodes)
    n_done_total = 0
    step_count = 0
    # 2026-05-13 — action distribution 추적 (사용자 spec). raw action (env clip 전)
    # 모음. attacker는 외부 attacker.step() 결과 ego_actions. defender는 env가
    # 내부 scripted_defender.step() 호출 후 env.last_defender_action에 clipped 저장.
    action_buf_attacker: list[np.ndarray] = []
    action_buf_defender: list[np.ndarray] = []
    max_steps_safety = (
        int(args.max_steps_per_episode) if args.max_steps_per_episode > 0
        else env.max_episode_length
    ) * max(1, target_episodes // env.num_envs + 4)

    env.reset()
    # Per-env step counter for episode-length tracking. Reset on done.
    per_env_step = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
    episode_lengths: list[int] = []
    t0 = time.time()
    while n_done_total < target_episodes and step_count < max_steps_safety:
        # External attacker step (B2 — env doesn't reset our external attacker).
        ego_actions = attacker.step(
            ego_state=env._attacker_state_dict(),
            opponent_state=env._defender_state_dict(),
            g_mission=env.g_mission,
        )
        per_env_step += 1
        _obs, _rew, done, _extras = env.step(ego_actions)
        n_done_total += int(done.sum().item())

        # action 모음 (eval과 동일 schema). raw attacker = ego_actions.
        # defender는 env가 내부 step 후 last_defender_action에 저장.
        action_buf_attacker.append(ego_actions.detach().cpu().numpy().copy())
        action_buf_defender.append(env.last_defender_action.detach().cpu().numpy().copy())

        # B2 — keep attacker.last_action in sync with env.last_attacker_action,
        # which env already zeroed for done envs inside reset_idx.
        done_idx = done.nonzero(as_tuple=False).reshape(-1)
        if done_idx.numel() > 0:
            attacker.reset(done_idx)
            # Record episode lengths + reset per-env counter
            episode_lengths.extend(per_env_step[done_idx].cpu().tolist())
            per_env_step[done_idx] = 0

        counts["mission"]   += int(env.last_term["mission"].sum().item())
        counts["captured"]  += int(env.last_term["captured"].sum().item())
        counts["att_crash"] += int(env.last_term["att_crash"].sum().item())
        counts["def_crash"] += int(env.last_term["def_crash"].sum().item())
        counts["att_oob"]   += int(env.last_term["att_oob"].sum().item())
        counts["def_oob"]   += int(env.last_term["def_oob"].sum().item())
        counts["timeout"]   += int(env.last_term["timeout"].sum().item())
        # Per-condition crash counts (env exposes them via last_term["att_crash_*"]
        # / "def_crash_*"). startswith underscore-suffix to avoid double-counting
        # the aggregate "att_crash" / "def_crash".
        for k, v in env.last_term.items():
            if k.startswith(("att_crash_", "def_crash_")):
                counts.setdefault(k, 0)
                counts[k] += int(v.sum().item())

        step_count += 1

    wall_sec = time.time() - t0

    # 2026-05-13 — action distribution per role (사용자 spec).
    # raw value 저장 (env clip 전). saturation % = |clipped|>0.95.
    def _action_dist_summary(actions_np: np.ndarray) -> dict:
        labels = ["T_norm", "wx_ref", "wy_ref", "wz_ref"]
        out = {}
        for i, label in enumerate(labels):
            col = actions_np[:, i]
            clipped = np.clip(col, -1.0, 1.0)
            out[label] = {
                "mean":        float(clipped.mean()),
                "std":         float(clipped.std()),
                "min":         float(clipped.min()),
                "max":         float(clipped.max()),
                "sat_gt95":    float((np.abs(clipped) > 0.95).mean()),
                "raw_abs_max": float(np.abs(col).max()),
            }
        return out

    actions_att = np.concatenate(action_buf_attacker, axis=0) if action_buf_attacker else np.zeros((0, 4))
    actions_def = np.concatenate(action_buf_defender, axis=0) if action_buf_defender else np.zeros((0, 4))
    action_dist_attacker = _action_dist_summary(actions_att) if actions_att.size else {}
    action_dist_defender = _action_dist_summary(actions_def) if actions_def.size else {}

    # AMS-DRL convention.
    attacker_wins = counts["mission"]
    defender_wins = counts["captured"] + counts["att_crash"] + counts["att_oob"]
    draws = counts["timeout"] + counts["def_crash"] + counts["def_oob"]
    n_eff = max(1, n_done_total)

    # Episode-length stats (added 2026-05-14 for spawn-stress evaluation).
    if episode_lengths:
        ep_lens_arr = np.array(episode_lengths, dtype=np.int64)
        ep_len_stats = {
            "mean": float(ep_lens_arr.mean()),
            "min":  int(ep_lens_arr.min()),
            "p10":  float(np.percentile(ep_lens_arr, 10)),
            "p50":  float(np.percentile(ep_lens_arr, 50)),
            "p90":  float(np.percentile(ep_lens_arr, 90)),
            "max":  int(ep_lens_arr.max()),
            "std":  float(ep_lens_arr.std()),
        }
    else:
        ep_len_stats = {"mean": 0.0, "min": 0, "p10": 0.0, "p50": 0.0,
                        "p90": 0.0, "max": 0, "std": 0.0}

    result = {
        "attacker_winrate": round(attacker_wins / n_eff, 4),
        "defender_winrate": round(defender_wins / n_eff, 4),
        "draw_rate":        round(draws / n_eff, 4),
        "n_episodes":       n_done_total,
        "episode_length":   ep_len_stats,
        "wilson_ci_95": {
            "attacker": list(_wilson_ci(attacker_wins, n_done_total)),
            "defender": list(_wilson_ci(defender_wins, n_done_total)),
            "draw":     list(_wilson_ci(draws, n_done_total)),
        },
        "breakdown": dict(counts),
        "seed": int(args.seed),
        "attacker_ckpt": str(attacker_ckpt),
        "defender_ckpt": str(defender_ckpt),
        "cfgs_pkl": str(cfgs_pkl),
        "git_sha": _git_sha(),
        "num_envs": int(args.num_envs),
        "backend": args.backend,
        "wall_sec": round(wall_sec, 2),
        "action_dist_attacker": action_dist_attacker,
        "action_dist_defender": action_dist_defender,
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        out = Path(args.output_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[h2h] wrote {out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
