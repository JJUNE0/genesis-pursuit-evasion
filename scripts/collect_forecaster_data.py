"""Phase 3b — Forecaster data collection (TODO §3b, 2026-05-13).

학습된 attacker/defender ckpt 를 frozen 으로 사용해 PE rollout 수행, 매 step
의 raw world-frame state + env actor obs + actions + g_mission 을 per-episode
``.pt`` 파일로 저장. Forecaster (§3.4) 학습용 supervised dataset 생성.

CLAUDE.md §1: env 코드 한 줄도 수정하지 않음. ``env.attacker_pos`` 등 기존
attribute 만 읽고, world-state 18D 합성은 본 스크립트 내부에서 수행.
CLAUDE.md §10: ``cfgs.pkl`` 강제 (ams 학습 분포와 정확 일치). ``env_yaml_override``
는 명시적 옵션 + warning.

Pattern: ``scripts/eval_head_to_head.py`` 그대로. ego="attacker" + env-injected
``PretrainedDefender`` + external ``PretrainedAttacker``. env.step(a_a)만 호출;
defender action 은 env 내부 처리 후 ``env.last_defender_action`` 에 저장됨.

Schema (per-episode ``.pt``):
    s_a_world (L, 18)   : pos(3)+vel_world(3)+rot_flat(9)+ang_vel(3) raw
    s_d_world (L, 18)   : 동일 (defender)
    s_a_obs   (L, 31)   : env _attacker_obs() 그대로 (scaled body frame)
    a_a       (L, 4)    : attacker action (post env clip)
    a_d       (L, 4)    : defender action (post env clip)
    g_mission (L, 3)    : world frame raw
    + metadata (episode_length, termination_reason, seed, ckpt paths)

Usage::

    python scripts/collect_forecaster_data.py \\
        --config configs/forecaster/data_collection_default.yaml

    # CLI override:
    python scripts/collect_forecaster_data.py \\
        --attacker_ckpt logs/.../attacker_v16.pt \\
        --defender_ckpt logs/.../defender_v16.pt \\
        --num_episodes 100 --num_envs 32 --output_dir /tmp/smoke \\
        --seed 42 --backend cpu
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs  # noqa: E402

from envs.base_pe_env import quat_to_rotmat_flat  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S_WORLD_DIM = 18          # pos(3) + vel_world(3) + rot_flat(9) + ang_vel(3)
G_MISSION_DIM = 3
NUM_ACTIONS = 4
TERMINATION_REASONS = (
    "mission_success", "captured", "out_of_bounds", "timeout",
)
PROGRESS_FILENAME = "progress.json"
INDEX_FILENAME = "index.json"
METADATA_FILENAME = "metadata.json"


# ---------------------------------------------------------------------------
# Seeding + Cfg loading
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_cfgs_pkl(attacker_ckpt: Path, cfgs_pkl_source: str | Path) -> Path:
    """``auto`` = ckpt parent (or grandparent) / cfgs.pkl; otherwise treat as path."""
    if str(cfgs_pkl_source) == "auto":
        cands = [
            attacker_ckpt.resolve().parent / "cfgs.pkl",
            attacker_ckpt.resolve().parent.parent / "cfgs.pkl",
        ]
        for c in cands:
            if c.is_file():
                return c
        raise FileNotFoundError(
            f"cfgs.pkl auto-resolve failed near {attacker_ckpt}; checked: "
            f"{[str(c) for c in cands]}"
        )
    p = Path(cfgs_pkl_source)
    if not p.is_file():
        raise FileNotFoundError(f"cfgs_pkl_source not found: {p}")
    return p


def load_cfgs(
    attacker_ckpt: Path,
    cfgs_pkl_source: str | Path = "auto",
    env_yaml_override: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve + load cfgs.pkl. Optionally override env_cfg from a yaml.

    Returns ``(cfgs_pkl_path, cfgs_dict)`` where cfgs_dict has keys
    ``env_cfg, obs_cfg, reward_cfg, command_cfg, drone_yaml, ...`` (matching
    ``agents.ppo_runner.dump_cfgs_pkl``).
    """
    cfgs_pkl = _resolve_cfgs_pkl(attacker_ckpt, cfgs_pkl_source)
    with open(cfgs_pkl, "rb") as f:
        cfgs = pickle.load(f)

    if env_yaml_override:
        print(
            f"[collect] ⚠ env_yaml_override active ({env_yaml_override}); "
            f"data distribution may differ from ckpt training cfg.",
            flush=True,
        )
        from envs.pe_1v1.cfgs import (  # noqa: PLC0415
            load_pe_1v1_cfg,
            make_command_cfg,
            make_env_cfg,
            make_obs_cfg,
            make_reward_cfg,
        )
        raw = load_pe_1v1_cfg(env_yaml_override)
        use_async = bool(cfgs["env_cfg"].get("use_asymmetric_critic", False))
        cfgs["env_cfg"] = make_env_cfg(raw, ego="attacker", use_asymmetric_critic=use_async)
        cfgs["obs_cfg"] = make_obs_cfg(raw)
        cfgs["reward_cfg"] = make_reward_cfg(raw)
        cfgs["command_cfg"] = make_command_cfg(raw)
    return cfgs_pkl, cfgs


# ---------------------------------------------------------------------------
# Env build (eval_head_to_head pattern)
# ---------------------------------------------------------------------------

def build_env_from_cfgs(
    cfgs: dict[str, Any],
    num_envs: int,
    defender_policy,
    tau_delay: int = 0,
    drone_yaml: str | Path | None = None,
):
    """Mirror ``scripts/eval_head_to_head.py`` env build.

    - ``ego="attacker"`` (external attacker drives ego_actions).
    - ``defender_policy`` injected (PretrainedDefender or StationaryDefender).
    - ``tau_delay`` 강제 set (forecaster data collection 은 항상 0).
    """
    from envs.pe_1v1.env import PursuitEvasion1v1Env  # noqa: PLC0415
    from utils.drone_params import DroneParams  # noqa: PLC0415

    env_cfg = dict(cfgs["env_cfg"])
    env_cfg["ego"] = "attacker"
    env_cfg["tau_delay"] = int(tau_delay)

    drone = (
        str(drone_yaml) if drone_yaml is not None
        else cfgs.get("drone_yaml") or str(ROOT / "configs" / "drones" / "nova.yaml")
    )
    params = DroneParams(drone)

    return PursuitEvasion1v1Env(
        num_envs=num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=cfgs["obs_cfg"],
        reward_cfg=cfgs["reward_cfg"],
        command_cfg=cfgs["command_cfg"],
        defender_policy=defender_policy,
        show_viewer=False,
    )


# ---------------------------------------------------------------------------
# State extraction (env attribute → 18D world raw)
# ---------------------------------------------------------------------------

def build_world_state_batched(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize 18D world-frame raw state for attacker + defender.

    Layout (18) = pos(3) + vel_world(3) + rot_flat(9) + ang_vel(3).
    - pos:       world frame, raw (env.{attacker,defender}_pos)
    - vel_world: world frame, raw, NO scale (env.{...}_vel — set from
                 drone.get_vel() which Genesis returns in world frame).
    - rot_flat:  quat_to_rotmat_flat(env.{...}_quat) → 9D rotation matrix flat.
    - ang_vel:   body frame, raw (env._update_state already rotates).

    Returns ``(s_a_world, s_d_world)`` each shape ``(B, 18)``.
    """
    pos_a = env.attacker_pos                                              # (B, 3) world
    vel_a = env.attacker_vel                                              # (B, 3) world
    rot_a = quat_to_rotmat_flat(env.attacker_quat)                        # (B, 9)
    ang_a = env.attacker_ang_vel                                          # (B, 3) body
    s_a_world = torch.cat([pos_a, vel_a, rot_a, ang_a], dim=1)            # (B, 18)

    pos_d = env.defender_pos                                              # (B, 3) world
    vel_d = env.defender_vel                                              # (B, 3) world
    rot_d = quat_to_rotmat_flat(env.defender_quat)                        # (B, 9)
    ang_d = env.defender_ang_vel                                          # (B, 3) body
    s_d_world = torch.cat([pos_d, vel_d, rot_d, ang_d], dim=1)            # (B, 18)

    assert s_a_world.shape[1] == S_WORLD_DIM
    assert s_d_world.shape[1] == S_WORLD_DIM
    return s_a_world, s_d_world


# ---------------------------------------------------------------------------
# Termination classification
# ---------------------------------------------------------------------------

def _last_term_get(last_term: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    return last_term.get(key)


def classify_termination(last_term: dict[str, torch.Tensor], env_i: int) -> str:
    """Map env per-env termination flags → 4-class string.

    Priority: mission > captured > crash/oob (→ "out_of_bounds") > timeout.
    Any side's crash or oob collapses to "out_of_bounds" (training envelope hit).
    """
    if bool(last_term["mission"][env_i]):
        return "mission_success"
    if bool(last_term["captured"][env_i]):
        return "captured"
    # Per-side flags (preferred) → fall back to aggregate crash/oob.
    crash_keys = ("att_crash", "def_crash")
    oob_keys = ("att_oob", "def_oob")
    for k in crash_keys + oob_keys:
        t = _last_term_get(last_term, k)
        if t is not None and bool(t[env_i]):
            return "out_of_bounds"
    for k in ("crash", "oob"):
        t = _last_term_get(last_term, k)
        if t is not None and bool(t[env_i]):
            return "out_of_bounds"
    if bool(last_term["timeout"][env_i]):
        return "timeout"
    # done was True but no flag matched — should not happen; default to timeout.
    return "timeout"


# ---------------------------------------------------------------------------
# Episode IO
# ---------------------------------------------------------------------------

def save_episode(
    output_dir: Path,
    episode_id: int,
    buffer: dict[str, list[torch.Tensor]],
    termination: str,
    seed: int,
    attacker_ckpt_path: str,
    defender_ckpt_path: str,
) -> Path:
    """Stack per-step CPU tensors → save .pt. Returns the file path."""
    L = len(buffer["s_a_world"])
    if L == 0:
        raise ValueError(f"save_episode: empty buffer for episode {episode_id}")

    payload = {
        "s_a_world":         torch.stack(buffer["s_a_world"]),     # (L, 18)
        "s_d_world":         torch.stack(buffer["s_d_world"]),     # (L, 18)
        "s_a_obs":           torch.stack(buffer["s_a_obs"]),       # (L, D_obs)
        "a_a":               torch.stack(buffer["a_a"]),           # (L, 4)
        "a_d":               torch.stack(buffer["a_d"]),           # (L, 4)
        "g_mission":         torch.stack(buffer["g_mission"]),     # (L, 3)
        "episode_length":    int(L),
        "termination_reason": termination,
        "seed":              int(seed),
        "attacker_ckpt_path": str(attacker_ckpt_path),
        "defender_ckpt_path": str(defender_ckpt_path),
    }
    path = output_dir / f"episode_{episode_id:06d}.pt"
    # Write atomically (tmp → rename) so crash mid-save doesn't leave a
    # half-written file in the index.
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def _compute_index_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {
            "mean_length": 0.0, "min_length": 0, "max_length": 0,
            "termination_dist": {k: 0.0 for k in TERMINATION_REASONS},
        }
    lengths = [e["length"] for e in episodes]
    counts: dict[str, int] = defaultdict(int)
    for e in episodes:
        counts[e["termination"]] += 1
    total = len(episodes)
    return {
        "mean_length": sum(lengths) / total,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "termination_dist": {k: counts.get(k, 0) / total for k in TERMINATION_REASONS},
    }


def write_index(output_dir: Path, episodes: list[dict[str, Any]]) -> Path:
    """Atomic write of index.json (full rewrite each flush)."""
    idx = {
        "total_episodes": len(episodes),
        "episodes": episodes,
        "stats": _compute_index_stats(episodes),
    }
    path = output_dir / INDEX_FILENAME
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=2)
    os.replace(tmp, path)
    return path


def write_metadata(
    output_dir: Path,
    config: dict[str, Any],
    cfgs_pkl: Path,
    elapsed_seconds: float,
    total_episodes: int,
) -> Path:
    meta = {
        "total_episodes": int(total_episodes),
        "attacker_ckpt": str(config["attacker_ckpt"]),
        "defender_ckpt": str(config["defender_ckpt"]),
        "cfgs_pkl": str(cfgs_pkl),
        "tau_delay": int(config.get("tau_delay", 0)),
        "num_envs": int(config["num_envs"]),
        "seed": int(config["seed"]),
        "collection_date": datetime.now().isoformat(timespec="seconds"),
        "elapsed_time_seconds": round(float(elapsed_seconds), 2),
        "env_yaml_override": (
            str(config["env_yaml_override"]) if config.get("env_yaml_override") else None
        ),
    }
    path = output_dir / METADATA_FILENAME
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)
    return path


def save_progress(output_dir: Path, collected_count: int) -> Path:
    progress = {
        "collected_count": int(collected_count),
        "last_save_time": datetime.now().isoformat(timespec="seconds"),
    }
    path = output_dir / PROGRESS_FILENAME
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, path)
    return path


def load_progress(output_dir: Path) -> int:
    """Resume crash recovery: read collected_count from progress.json (0 if absent)."""
    p = output_dir / PROGRESS_FILENAME
    if not p.is_file():
        return 0
    try:
        with open(p) as f:
            return int(json.load(f).get("collected_count", 0))
    except Exception as e:  # noqa: BLE001
        print(f"[collect] ⚠ progress.json unreadable ({e}); restarting from 0.")
        return 0


def load_index(output_dir: Path) -> list[dict[str, Any]]:
    """Load existing index.json episodes list (for resume)."""
    p = output_dir / INDEX_FILENAME
    if not p.is_file():
        return []
    try:
        with open(p) as f:
            return list(json.load(f).get("episodes", []))
    except Exception as e:  # noqa: BLE001
        print(f"[collect] ⚠ index.json unreadable ({e}); starting fresh index.")
        return []


# ---------------------------------------------------------------------------
# Rollout — core collection loop
# ---------------------------------------------------------------------------

def collect_episodes(
    env,
    attacker,
    *,
    num_episodes: int,
    output_dir: Path,
    seed: int,
    attacker_ckpt_path: str,
    defender_ckpt_path: str,
    max_episode_steps: int = 0,
    log_every_n_episodes: int = 100,
    checkpoint_progress_every: int = 1000,
    start_count: int = 0,
    existing_episodes: list[dict[str, Any]] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Rollout the env until ``num_episodes`` episodes are collected.

    Parameters
    ----------
    env : PursuitEvasion1v1Env
        Already built; ``ego="attacker"`` and defender injected.
    attacker : ScriptedPolicy
        ``step(ego_state, opponent_state, g_mission) -> (B, 4)``. Either
        ``PretrainedAttacker`` or ``StationaryAttacker`` (test) etc.
    output_dir : Path
        Episode .pt files + index.json + progress.json will be written here.
    start_count : int
        Resume from this index (number of episodes already collected).
    existing_episodes : list[dict] | None
        Pre-existing index entries (for resume).

    Returns ``(final_collected, episodes_index)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_envs = int(env.num_envs)
    max_steps_per_ep = (
        int(max_episode_steps) if int(max_episode_steps) > 0
        else int(env.max_episode_length)
    )
    # Safety bound on outer loop — prevents infinite spin if something is off.
    # collected per outer iter ≤ num_envs; total iters ≤ ceil(remaining / 1) * max_steps_per_ep.
    safety_step_budget = (
        max(1, (num_episodes - start_count) // max(1, num_envs) + 4)
        * max_steps_per_ep
    )

    buffers: list[dict[str, list[torch.Tensor]]] = [
        defaultdict(list) for _ in range(num_envs)
    ]
    episodes_index: list[dict[str, Any]] = list(existing_episodes or [])
    collected = int(start_count)

    obs = env.reset()
    # PretrainedAttacker.last_action — keep in sync with env's per-env reset.
    if hasattr(attacker, "reset"):
        attacker.reset(torch.arange(num_envs, device=env.device))

    t0 = time.time()
    step_idx = 0
    last_log_at = collected

    while collected < num_episodes and step_idx < safety_step_budget:
        # ---- Build t-step world state (BEFORE step) ----
        s_a_world_b, s_d_world_b = build_world_state_batched(env)         # (B, 18) each
        s_a_obs_b = obs["policy"]                                         # (B, D_obs)
        g_mission_b = env.g_mission                                       # (B, 3)

        # ---- Attacker action ----
        with torch.no_grad():
            ego_actions = attacker.step(
                ego_state=env._attacker_state_dict(),
                opponent_state=env._defender_state_dict(),
                g_mission=env.g_mission,
            )                                                             # (B, 4)

        # ---- CPU-side snapshots (single batched transfer, then split) ----
        s_a_world_cpu = s_a_world_b.detach().to("cpu")
        s_d_world_cpu = s_d_world_b.detach().to("cpu")
        s_a_obs_cpu   = s_a_obs_b.detach().to("cpu")
        g_mission_cpu = g_mission_b.detach().to("cpu")
        ego_actions_cpu = ego_actions.detach().to("cpu")

        for env_i in range(num_envs):
            buf = buffers[env_i]
            buf["s_a_world"].append(s_a_world_cpu[env_i].clone())
            buf["s_d_world"].append(s_d_world_cpu[env_i].clone())
            buf["s_a_obs"].append(s_a_obs_cpu[env_i].clone())
            buf["g_mission"].append(g_mission_cpu[env_i].clone())
            buf["a_a"].append(ego_actions_cpu[env_i].clone())

        # ---- Step ----
        obs, _rew, done, _extras = env.step(ego_actions)

        # ---- a_d (post-step; env applied & clipped defender action) ----
        last_def_cpu = env.last_defender_action.detach().to("cpu")
        for env_i in range(num_envs):
            buffers[env_i]["a_d"].append(last_def_cpu[env_i].clone())

        # ---- Termination snapshot (last_term survives env's reset wipe) ----
        last_term_cpu: dict[str, torch.Tensor] = {}
        for k in ("mission", "captured", "crash", "oob",
                  "att_crash", "def_crash", "att_oob", "def_oob",
                  "timeout"):
            v = env.last_term.get(k)
            if v is not None:
                last_term_cpu[k] = v.detach().to("cpu")

        # ---- Keep external attacker's last_action in sync with env's reset ----
        done_idx = done.nonzero(as_tuple=False).reshape(-1)
        if done_idx.numel() > 0 and hasattr(attacker, "reset"):
            attacker.reset(done_idx)

        # ---- Flush done episodes ----
        done_cpu = done.detach().to("cpu")
        for env_i in range(num_envs):
            if not bool(done_cpu[env_i]):
                continue
            if collected >= num_episodes:
                # Drop trailing in-flight buffers we don't need.
                buffers[env_i] = defaultdict(list)
                continue
            termination = classify_termination(last_term_cpu, env_i)
            ep_seed = int(seed) + collected
            try:
                save_episode(
                    output_dir=output_dir,
                    episode_id=collected,
                    buffer=buffers[env_i],
                    termination=termination,
                    seed=ep_seed,
                    attacker_ckpt_path=attacker_ckpt_path,
                    defender_ckpt_path=defender_ckpt_path,
                )
            except ValueError:
                # Empty buffer (shouldn't happen since at least one step ran).
                buffers[env_i] = defaultdict(list)
                continue
            L = len(buffers[env_i]["s_a_world"])
            episodes_index.append({
                "id": collected,
                "length": L,
                "termination": termination,
                "file": f"episode_{collected:06d}.pt",
            })
            buffers[env_i] = defaultdict(list)
            collected += 1

            # Periodic flushes (drive by collected, not step_idx, so log/ckpt
            # cadence is in episode units regardless of multi-env race).
            if collected - last_log_at >= log_every_n_episodes:
                _log_progress(collected, num_episodes, episodes_index, t0)
                last_log_at = collected
            if collected % max(1, checkpoint_progress_every) == 0:
                write_index(output_dir, episodes_index)
                save_progress(output_dir, collected)

        step_idx += 1

    # ---- Final flush ----
    write_index(output_dir, episodes_index)
    save_progress(output_dir, collected)
    return collected, episodes_index


def _log_progress(
    collected: int, target: int, episodes: list[dict[str, Any]], t0: float,
) -> None:
    elapsed = time.time() - t0
    pct = 100.0 * collected / max(1, target)
    if episodes:
        recent = episodes[-min(len(episodes), 200):]
        avg_len = sum(e["length"] for e in recent) / len(recent)
        term_counts: dict[str, int] = defaultdict(int)
        for e in recent:
            term_counts[e["termination"]] += 1
        denom = float(len(recent))
        term_dist = {k: term_counts.get(k, 0) / denom for k in TERMINATION_REASONS}
        term_str = " ".join(f"{k[:4]}={v:.2f}" for k, v in term_dist.items())
    else:
        avg_len = 0.0
        term_str = "(no episodes yet)"
    print(
        f"[collect] {collected}/{target} ({pct:.1f}%) — "
        f"avg_len={avg_len:.1f} term[{term_str}] elapsed={elapsed:.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# main() — CLI entrypoint
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge yaml defaults with CLI overrides. CLI > yaml > hard defaults."""
    cfg: dict[str, Any] = {
        "attacker_ckpt": "",
        "defender_ckpt": "",
        "cfgs_pkl_source": "auto",
        "env_yaml_override": None,
        "num_episodes": 20000,
        "num_envs": 1024,
        "max_episode_steps": 0,
        "tau_delay": 0,
        "output_dir": "data/forecaster_data_v1/",
        "seed": 42,
        "backend": "gpu",
        "log_every_n_episodes": 100,
        "checkpoint_progress_every": 1000,
    }
    if args.config:
        cfg.update(_load_yaml_config(args.config))

    # CLI overrides — only override when explicitly set (non-default).
    overrides = {
        "attacker_ckpt": args.attacker_ckpt,
        "defender_ckpt": args.defender_ckpt,
        "cfgs_pkl_source": args.cfgs_pkl_source,
        "env_yaml_override": args.env_yaml_override,
        "num_episodes": args.num_episodes,
        "num_envs": args.num_envs,
        "max_episode_steps": args.max_episode_steps,
        "tau_delay": args.tau_delay,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "backend": args.backend,
        "log_every_n_episodes": args.log_every_n_episodes,
        "checkpoint_progress_every": args.checkpoint_progress_every,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3b forecaster data collection")
    p.add_argument("--config", type=str, default=None,
                   help="YAML defaults; CLI overrides take precedence.")
    p.add_argument("--attacker_ckpt", type=str, default=None)
    p.add_argument("--defender_ckpt", type=str, default=None)
    p.add_argument("--cfgs_pkl_source", type=str, default=None,
                   help="'auto' or explicit path. Default: yaml or 'auto'.")
    p.add_argument("--env_yaml_override", type=str, default=None,
                   help="Override env_cfg from this yaml (NOT recommended; warning logged).")
    p.add_argument("--num_episodes", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--max_episode_steps", type=int, default=None,
                   help="0 = use env.max_episode_length.")
    p.add_argument("--tau_delay", type=int, default=None,
                   help="Forecaster collection ought to be 0 (default).")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backend", type=str, default=None, choices=("gpu", "cpu"))
    p.add_argument("--log_every_n_episodes", type=int, default=None)
    p.add_argument("--checkpoint_progress_every", type=int, default=None)
    return p.parse_args()


def _build_pretrained_attacker(num_envs: int, attacker_ckpt: str, device):
    from envs.pe_1v1.scripted_attacker import PretrainedAttacker  # noqa: PLC0415
    return PretrainedAttacker(
        num_envs=num_envs, ckpt_path=attacker_ckpt, device=device,
    )


def _build_pretrained_defender(num_envs: int, defender_ckpt: str, device):
    from envs.pe_1v1.scripted_defender import PretrainedDefender  # noqa: PLC0415
    return PretrainedDefender(
        num_envs=num_envs, ckpt_path=defender_ckpt, device=device,
    )


def main() -> int:
    args = parse_args()
    config = _merge_config(args)

    # --- Preflight ------------------------------------------------------
    if not config["attacker_ckpt"]:
        raise SystemExit("[collect] --attacker_ckpt (or config) required.")
    if not config["defender_ckpt"]:
        raise SystemExit("[collect] --defender_ckpt (or config) required.")
    attacker_ckpt = Path(config["attacker_ckpt"]).resolve()
    defender_ckpt = Path(config["defender_ckpt"]).resolve()
    if not attacker_ckpt.is_file():
        raise SystemExit(f"[collect] attacker_ckpt not found: {attacker_ckpt}")
    if not defender_ckpt.is_file():
        raise SystemExit(f"[collect] defender_ckpt not found: {defender_ckpt}")

    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    _seed_all(seed)

    backend = gs.gpu if config["backend"] == "gpu" else gs.cpu
    gs.init(backend=backend, seed=seed, logging_level="warning")

    # --- Cfg + env build -----------------------------------------------
    cfgs_pkl, cfgs = load_cfgs(
        attacker_ckpt=attacker_ckpt,
        cfgs_pkl_source=config["cfgs_pkl_source"],
        env_yaml_override=config["env_yaml_override"],
    )
    print(f"[collect] cfgs.pkl = {cfgs_pkl}", flush=True)

    defender = _build_pretrained_defender(
        num_envs=int(config["num_envs"]), defender_ckpt=str(defender_ckpt), device=gs.device,
    )
    env = build_env_from_cfgs(
        cfgs=cfgs, num_envs=int(config["num_envs"]), defender_policy=defender,
        tau_delay=int(config["tau_delay"]),
    )
    attacker = _build_pretrained_attacker(
        num_envs=int(config["num_envs"]), attacker_ckpt=str(attacker_ckpt), device=gs.device,
    )

    # --- Resume support -------------------------------------------------
    existing = load_index(output_dir)
    progress_count = load_progress(output_dir)
    start_count = min(progress_count, len(existing))
    if start_count > 0:
        print(
            f"[collect] resuming from {start_count} episodes "
            f"(progress.json={progress_count}, index.json={len(existing)}).",
            flush=True,
        )
        existing = existing[:start_count]
    if start_count >= int(config["num_episodes"]):
        print(f"[collect] already at target ({start_count} ≥ {config['num_episodes']}); exiting.")
        return 0

    # --- Collect --------------------------------------------------------
    print(
        f"[collect] target={config['num_episodes']} num_envs={config['num_envs']} "
        f"tau_delay={config['tau_delay']} output_dir={output_dir}",
        flush=True,
    )
    t0 = time.time()
    collected, episodes_index = collect_episodes(
        env=env, attacker=attacker,
        num_episodes=int(config["num_episodes"]),
        output_dir=output_dir,
        seed=seed,
        attacker_ckpt_path=str(attacker_ckpt),
        defender_ckpt_path=str(defender_ckpt),
        max_episode_steps=int(config["max_episode_steps"]),
        log_every_n_episodes=int(config["log_every_n_episodes"]),
        checkpoint_progress_every=int(config["checkpoint_progress_every"]),
        start_count=start_count,
        existing_episodes=existing,
    )
    elapsed = time.time() - t0
    write_metadata(
        output_dir=output_dir, config=config, cfgs_pkl=cfgs_pkl,
        elapsed_seconds=elapsed, total_episodes=collected,
    )
    print(
        f"[collect] DONE — collected {collected} episodes in {elapsed:.1f}s. "
        f"output_dir={output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
