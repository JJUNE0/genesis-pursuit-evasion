"""Config builders for PursuitEvasion1v1Env.

The single source of truth is the YAML at configs/envs/pe_1v1_default.yaml.
This module loads it and returns the four dicts used by the env constructor:
``env_cfg``, ``obs_cfg``, ``reward_cfg``, ``command_cfg``.

CLAUDE.md §1: cfg schema 변경은 사용자 명시 승인 필수.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ---- Observation dimensions (Phase 1 fixed; see plan §4.1) -----------------
# s^a       = pos(3) + vel(3) + rot_mat(9) + ang_vel(3) = 18
# s^d_kin   = pos(3) + vel(3)                            = 6
# s^d       = pos(3) + vel(3) + rot_mat(9) + ang_vel(3) = 18
# s^a_kin   = pos(3) + vel(3)                            = 6
# g_mission = 3
# last_action = 4   (PR-D: action smoothness signal — hover env 차용)
S_FULL_DIM = 18
S_KIN_DIM = 6
G_MISSION_DIM = 3
LAST_ACTION_DIM = 4

ATTACKER_OBS_DIM = S_FULL_DIM + S_KIN_DIM + G_MISSION_DIM + LAST_ACTION_DIM   # 31
DEFENDER_OBS_DIM = S_FULL_DIM + S_KIN_DIM + LAST_ACTION_DIM                   # 28

# Phase 3d (TODO §3d, 2026-05-12) — BPQL-style asymmetric critic privileged
# obs [Kim et al. 2023 NeurIPS]. Critic receives ground-truth current
# defender kinematics (no delay) in attacker body frame relative form:
#   rel_pos_d_GT(3) + rel_vel_d_GT(3) = 6
# matching the actor's s_d_kin slot exactly — so the critic sees the *same*
# representation as the actor, only with the τ-delay removed. Belief
# projection (BPQL Prop. 3.2) keeps critic value targets in the original
# state space rather than the augmented one, sidestepping augmented-state
# blowup as τ grows.
# Supersedes PR-E.1 (2026-05-08) which used vel(3)+ang_vel(3)+quat(4)=10D —
# that path was extra dynamics info, not BPQL belief projection.
CRITIC_PRIV_DIM = 6

NUM_ACTIONS = 4   # per-prop scaled-hover RPM, a ∈ [-1, 1]^4

# obs scaling defaults (PR-D — hover env 차용). Final scaling values come
# from YAML; these are the documented defaults.
DEFAULT_OBS_SCALES = {
    "pos":     0.2,        # 1 / 5.0  (bound_xyz)
    "vel":     0.2,        # 1 / 5.0
    "ang_vel": 0.31831,    # 1 / π
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_yaml_path() -> Path:
    return _project_root() / "configs" / "envs" / "pe_1v1_default.yaml"


def load_pe_1v1_cfg(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """Load the raw YAML cfg dict.

    Returns the raw dict (sim / termination / spawn / g_mission / reward / action).
    Use ``make_env_cfg`` etc. to convert to the legacy 4-dict layout that
    rsl-rl OnPolicyRunner expects.
    """
    p = Path(yaml_path) if yaml_path is not None else default_yaml_path()
    with open(p) as f:
        return yaml.safe_load(f)


VALID_GAME_MODES = ("mission", "pure_pe")


def make_env_cfg(
    raw: dict[str, Any], ego: str = "attacker",
    use_asymmetric_critic: bool = False,
) -> dict[str, Any]:
    sim = raw["sim"]
    term = raw["termination"]
    action = raw.get("action", {})
    if ego not in ("attacker", "defender"):
        raise ValueError(f"ego must be 'attacker' or 'defender', got {ego!r}")
    # action_mode: "rpm" (legacy per-prop scaled-hover) or "rate"
    # ([T_norm, wx_ref, wy_ref, wz_ref] body-rate PD inner loop).
    rate_ctrl_cfg = action.get("rate_controller", {}) or {}
    return {
        "ego": ego,
        "dt": float(sim["dt"]),
        "substeps": int(sim["substeps"]),
        "episode_length_s": float(sim["episode_length_s"]),
        "num_actions": NUM_ACTIONS,
        "clip_actions": float(action.get("clip_actions", 1.0)),
        "simulate_action_latency": bool(action.get("simulate_action_latency", False)),
        "action_latency_steps": int(action.get("action_latency_steps", 0)),
        "action_mode": str(action.get("action_mode", "rpm")),
        "rate_controller": dict(rate_ctrl_cfg),
        # termination
        "d_capture": float(term["d_capture"]),
        "g_mission_radius": float(term["g_mission_radius"]),
        "pitch_max_deg": float(term["pitch_max_deg"]),
        "roll_max_deg": float(term["roll_max_deg"]),
        # 2026-05-04 — when False, roll/pitch threshold no longer terminates
        # episode (only z_floor / v_max / oob remain hard crash triggers).
        # attitude_penalty regulates instead. Default True = legacy.
        "crash_on_attitude": bool(term.get("crash_on_attitude", True)),
        # 2026-05-04 — when False, |v|>v_max no longer terminates episode.
        # velocity_penalty (safety_penalty) regulates. Default True = legacy.
        "crash_on_speed": bool(term.get("crash_on_speed", True)),
        "z_floor": float(term["z_floor"]),
        "v_max": float(term["v_max"]),
        "bound_xyz": float(term["bound_xyz"]),
        # per-axis override (fallback to bound_xyz when not specified)
        "bound_x": float(term.get("bound_x", term["bound_xyz"])),
        "bound_y": float(term.get("bound_y", term["bound_xyz"])),
        "bound_z": float(term.get("bound_z", term["bound_xyz"])),
        # Phase E.1 (CLAUDE.md §7) — asymmetric critic flag (default False = backward compat).
        "use_asymmetric_critic": bool(use_asymmetric_critic),
        # Phase F (2026-05-10) — game mode. "mission" (기존 g_mission 추구 + 회피 동시) or
        # "pure_pe" (mission 비활성, attacker는 도망만 — AMSPB 직접 비교용).
        # default = "mission" (backward compat).
        "game_mode": _validate_game_mode(raw.get("game_mode", "mission")),
        # Phase 3a (TODO §3a, 2026-05-12) — τ-step defender obs delay (BPQL
        # augmented state). Actor reads defender state from τ steps ago via
        # the ring buffer in BasePursuitEvasionEnv. default 0 = no delay
        # (backward compat). Variants 2-5 set this per-variant yaml.
        "tau_delay": int(raw.get("tau_delay", 0)),
    }


def _validate_game_mode(mode: str) -> str:
    if mode not in VALID_GAME_MODES:
        raise ValueError(
            f"game_mode must be one of {VALID_GAME_MODES}, got {mode!r}"
        )
    return mode


def make_obs_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    obs_scales = dict(DEFAULT_OBS_SCALES)
    yaml_scales = raw.get("obs_scales") or {}
    obs_scales.update({k: float(v) for k, v in yaml_scales.items()})
    return {
        "attacker_obs_dim": ATTACKER_OBS_DIM,
        "defender_obs_dim": DEFENDER_OBS_DIM,
        "s_full_dim": S_FULL_DIM,
        "s_kin_dim": S_KIN_DIM,
        "g_mission_dim": G_MISSION_DIM,
        "last_action_dim": LAST_ACTION_DIM,
        "obs_scales": obs_scales,
    }


def make_reward_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    """Pass through reward magnitudes from YAML.

    See TODO.md §1.2 표 for the expected keys.
    """
    return dict(raw["reward"])


def make_command_cfg(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "spawn": raw["spawn"],
        "g_mission": raw["g_mission"],
    }


def num_actions_for(_role: str) -> int:
    """Action dim is the same (4 per-prop) regardless of role."""
    return NUM_ACTIONS


def policy_obs_dim_for(role: str) -> int:
    if role == "attacker":
        return ATTACKER_OBS_DIM
    if role == "defender":
        return DEFENDER_OBS_DIM
    raise ValueError(f"role must be 'attacker' or 'defender', got {role!r}")
