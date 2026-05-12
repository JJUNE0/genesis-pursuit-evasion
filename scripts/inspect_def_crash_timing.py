"""Diagnose def_crash exploit pattern in 1v1 PE.

Runs frozen attacker_v0 vs frozen defender_v0 for N episodes, records per-step
trajectory metadata, and outputs aggregated statistics:

* event distribution (mission / captured / att_crash / def_crash / oob / timeout)
* def_crash terminal-step histogram (early-spike vs late?)
* mission terminal-step histogram (for comparison)
* per-event max attacker velocity (does attacker accelerate before def_crash?)
* per-event mean d_to_goal at terminal (was attacker heading to box?)

Used to confirm whether the "defender 자살 유도" exploit is genuine before
committing to ego_aware termination implementation.

Usage::

    python scripts/inspect_def_crash_timing.py \\
        --attacker_ckpt logs/.../attacker_v0.pt \\
        --defender_ckpt logs/.../defender_v0.pt \\
        --n_episodes 1000 --num_envs 256 --seed 0 \\
        --output_json /tmp/def_crash_timing.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
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
    p = argparse.ArgumentParser(description="Inspect def_crash exploit timing")
    p.add_argument("--attacker_ckpt", type=str, required=True)
    p.add_argument("--defender_ckpt", type=str, required=True)
    p.add_argument("--cfgs_pkl", type=str, default="",
                   help="Override env cfg source. Default = cfgs.pkl next to attacker_ckpt.")
    p.add_argument("--n_episodes", type=int, default=1000,
                   help="Absolute total episode count.")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="cpu", choices=["gpu", "cpu"])
    p.add_argument("--output_json", type=str, default="/tmp/def_crash_timing.json")
    p.add_argument("--drone_yaml", type=str,
                   default=str(ROOT / "configs" / "drones" / "nova.yaml"))
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_cfgs(attacker_ckpt: Path, override: str) -> Path:
    if override:
        return Path(override)
    return attacker_ckpt.resolve().parent / "cfgs.pkl"


@torch.no_grad()
def main() -> int:
    args = parse_args()
    _seed_all(args.seed)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, seed=args.seed, logging_level="warning")

    attacker_ckpt = Path(args.attacker_ckpt).resolve()
    defender_ckpt = Path(args.defender_ckpt).resolve()
    cfgs_pkl = _resolve_cfgs(attacker_ckpt, args.cfgs_pkl)
    with open(cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)

    env_cfg = dict(cfgs["env_cfg"])
    env_cfg["ego"] = "attacker"  # external PretrainedAttacker drives ego
    obs_cfg = cfgs["obs_cfg"]
    reward_cfg = cfgs["reward_cfg"]
    command_cfg = cfgs["command_cfg"]
    drone_yaml = cfgs.get("drone_yaml", args.drone_yaml)

    params = DroneParams(drone_yaml)
    defender = PretrainedDefender(
        num_envs=args.num_envs, ckpt_path=str(defender_ckpt), device=gs.device,
    )
    attacker = PretrainedAttacker(
        num_envs=args.num_envs, ckpt_path=str(attacker_ckpt), device=gs.device,
    )
    env = PursuitEvasion1v1Env(
        num_envs=args.num_envs, params=params,
        env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
        command_cfg=command_cfg, defender_policy=defender, show_viewer=False,
    )

    device = gs.device
    B = args.num_envs

    # Per-env trackers (reset on done)
    step_in_ep = torch.zeros(B, dtype=torch.long, device=device)
    max_v_attacker = torch.zeros(B, dtype=torch.float32, device=device)
    init_d_ad = torch.zeros(B, dtype=torch.float32, device=device)  # at episode start
    init_d_goal = torch.zeros(B, dtype=torch.float32, device=device)
    has_init = torch.zeros(B, dtype=torch.bool, device=device)

    # Per-event log
    events: list[dict] = []
    target_episodes = args.n_episodes
    step_count = 0
    safety_max_steps = env.max_episode_length * (target_episodes // B + 4)

    env.reset()
    t0 = time.time()

    while len(events) < target_episodes and step_count < safety_max_steps:
        # init buffers on freshly-reset envs
        new_starts = ~has_init
        if new_starts.any():
            init_d_ad[new_starts] = env.dist_to_attacker_defender[new_starts].clone()
            init_d_goal[new_starts] = env.dist_to_goal[new_starts].clone()
            has_init[new_starts] = True

        # Frozen attacker step
        ego_actions = attacker.step(
            ego_state=env._attacker_state_dict(),
            opponent_state=env._defender_state_dict(),
            g_mission=env.g_mission,
        )
        _obs, _rew, done, _info = env.step(ego_actions)
        step_in_ep += 1

        # Track max velocity over episode
        v = env.attacker_vel.norm(dim=1)
        max_v_attacker = torch.maximum(max_v_attacker, v)

        # Reset external attacker buffers for done envs
        done_idx = done.nonzero(as_tuple=False).reshape(-1)
        if done_idx.numel() > 0:
            # Read last_alive_state snapshot (taken before reset_idx wipes
            # pos/vel/quat for done envs). env.{attacker,defender}_* would be
            # post-reset spawn values for those envs.
            from envs.base_pe_env import quat_to_euler_deg as _q2e  # lazy import
            las = env.last_alive_state
            d_pos = las["defender_pos"]
            d_vel = las["defender_vel"]
            d_quat = las["defender_quat"]
            d_eul = _q2e(d_quat)
            d_speed = d_vel.norm(dim=1)
            attacker.reset(done_idx)
            term = env.last_term

            # Final per-env measurements (at terminal step)
            for idx in done_idx.tolist():
                event_label = None
                for k in ("mission", "captured", "att_crash", "def_crash",
                         "att_oob", "def_oob", "timeout"):
                    if bool(term[k][idx].item()):
                        event_label = k
                        break
                if event_label is None:
                    event_label = "unknown"

                # Defender crash sub-condition breakdown
                def_z       = float(d_pos[idx, 2].item())
                def_speed   = float(d_speed[idx].item())
                def_roll    = float(d_eul[idx, 0].abs().item())
                def_pitch   = float(d_eul[idx, 1].abs().item())
                crash_subs = []
                if def_z < env.z_floor:        crash_subs.append("z_below")
                if def_speed > env.v_max:      crash_subs.append("speed_high")
                if def_roll > env.roll_max_deg:    crash_subs.append("roll_bad")
                if def_pitch > env.pitch_max_deg:  crash_subs.append("pitch_bad")

                events.append({
                    "event": event_label,
                    "term_step": int(step_in_ep[idx].item()),
                    "max_v_attacker": float(max_v_attacker[idx].item()),
                    "init_d_ad": float(init_d_ad[idx].item()),
                    "init_d_goal": float(init_d_goal[idx].item()),
                    "term_d_ad": float(env.dist_to_attacker_defender[idx].item()),
                    "term_d_goal": float(env.dist_to_goal[idx].item()),
                    "def_z": def_z,
                    "def_speed": def_speed,
                    "def_roll": def_roll,
                    "def_pitch": def_pitch,
                    "crash_subs": crash_subs,
                })

            # Reset trackers for done envs
            step_in_ep[done_idx] = 0
            max_v_attacker[done_idx] = 0.0
            has_init[done_idx] = False

        step_count += 1

    wall_sec = time.time() - t0
    print(f"\n[done] {len(events)} episodes in {wall_sec:.1f}s ({step_count} steps)")

    # ── Aggregate Statistics ──────────────────────────────────────────
    n = len(events)

    # Event distribution
    event_counts: dict[str, int] = {}
    for e in events:
        event_counts[e["event"]] = event_counts.get(e["event"], 0) + 1

    print(f"\n=== Event distribution ({n} episodes) ===")
    for k in sorted(event_counts, key=lambda x: -event_counts[x]):
        c = event_counts[k]
        print(f"  {k:>10s}: {c:>5d} ({c/n*100:5.1f}%)")

    # Per-event timing analysis
    def _by_event(label: str) -> list[dict]:
        return [e for e in events if e["event"] == label]

    def _stats(arr: list[float], label: str, dt: float = 0.005):
        if not arr:
            print(f"  {label}: no samples")
            return
        a = np.asarray(arr)
        print(f"  {label}: n={len(a)}, mean={a.mean():.2f}, "
              f"median={np.median(a):.2f}, min={a.min():.2f}, max={a.max():.2f}"
              + (f" ({a.mean()*dt:.2f}s avg)" if dt else ""))

    def _hist(arr: list[float], bins: list[float], label: str):
        if not arr:
            return
        a = np.asarray(arr)
        hist, _ = np.histogram(a, bins=bins)
        print(f"  {label} histogram:")
        for i, c in enumerate(hist):
            pct = c / len(a) * 100
            bar = "▇" * int(pct / 3)
            print(f"    [{bins[i]:>4.0f}, {bins[i+1]:>4.0f}): "
                  f"{c:>4d} ({pct:5.1f}%) {bar}")

    print(f"\n=== def_crash analysis ===")
    dcr = _by_event("def_crash")
    if dcr:
        _stats([e["term_step"] for e in dcr], "term_step")
        _hist([e["term_step"] for e in dcr],
              [0, 50, 100, 200, 300, 500, 800, 1000, 1600], "term_step")
        _stats([e["max_v_attacker"] for e in dcr], "max_v_attacker (m/s)", dt=None)
        _stats([e["init_d_ad"] for e in dcr], "init_d_ad (m)", dt=None)
        _stats([e["term_d_goal"] for e in dcr], "term_d_goal (m)", dt=None)

        # Crash sub-condition breakdown — which trigger fired most?
        sub_counts: dict[str, int] = {}
        for e in dcr:
            for s in e.get("crash_subs", []):
                sub_counts[s] = sub_counts.get(s, 0) + 1
        if sub_counts:
            print(f"  crash sub-conditions (multi-fire OK):")
            for k in sorted(sub_counts, key=lambda x: -sub_counts[x]):
                c = sub_counts[k]
                print(f"    {k:>12s}: {c:>4d} ({c/len(dcr)*100:5.1f}%)")
        # Defender state distribution at terminal
        _stats([e["def_speed"] for e in dcr], "def_speed (m/s)", dt=None)
        _stats([e["def_roll"]  for e in dcr], "def_roll  (deg)", dt=None)
        _stats([e["def_pitch"] for e in dcr], "def_pitch (deg)", dt=None)
        _stats([e["def_z"]     for e in dcr], "def_z     (m)",   dt=None)

    print(f"\n=== mission analysis ===")
    msn = _by_event("mission")
    if msn:
        _stats([e["term_step"] for e in msn], "term_step")
        _stats([e["max_v_attacker"] for e in msn], "max_v_attacker (m/s)", dt=None)
    else:
        print("  no mission terminations observed")

    print(f"\n=== captured analysis ===")
    cap = _by_event("captured")
    if cap:
        _stats([e["term_step"] for e in cap], "term_step")
    else:
        print("  no captured terminations observed")

    # Velocity comparison
    print(f"\n=== Max attacker velocity by terminal event ===")
    for k in sorted(event_counts, key=lambda x: -event_counts[x]):
        vs = [e["max_v_attacker"] for e in _by_event(k)]
        if vs:
            a = np.asarray(vs)
            print(f"  {k:>10s}: mean={a.mean():.2f} m/s, max={a.max():.2f} "
                  f"(>5m/s: {(a > 5).mean()*100:.1f}%, >7m/s: {(a > 7).mean()*100:.1f}%)")

    # Verdict heuristic
    print(f"\n=== Diagnostic verdict ===")
    dcr_steps = np.asarray([e["term_step"] for e in dcr]) if dcr else np.asarray([])
    msn_steps = np.asarray([e["term_step"] for e in msn]) if msn else np.asarray([])
    if dcr_steps.size:
        early_pct = (dcr_steps < 200).mean() * 100
        print(f"  def_crash <200 step: {early_pct:.1f}%")
        if msn_steps.size:
            print(f"  mean def_crash step ({dcr_steps.mean():.0f}) vs "
                  f"mean mission step ({msn_steps.mean():.0f})")
        if early_pct > 50:
            print("  → def_crash 대부분 episode 초반 → exploit 가능성 강함")
        else:
            print("  → def_crash 분산됨 → natural side-effect 가능성")

    # Save JSON
    out = {
        "n_episodes": n,
        "wall_sec": round(wall_sec, 2),
        "event_counts": event_counts,
        "events": events,
        "attacker_ckpt": str(attacker_ckpt),
        "defender_ckpt": str(defender_ckpt),
        "cfgs_pkl": str(cfgs_pkl),
        "seed": args.seed,
        "num_envs": args.num_envs,
    }
    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
