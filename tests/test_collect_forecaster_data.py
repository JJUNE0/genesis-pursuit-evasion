"""Phase 3b — Tests for ``scripts/collect_forecaster_data`` (TODO §3b, 2026-05-13).

4 tests:
  1. End-to-end small-scale collection (10 episodes, num_envs=4, stationary).
  2. ``metadata.json`` + ``index.json`` schema completeness.
  3. ``cfgs.pkl`` auto-resolution + load.
  4. Variable-length episode handling (RandomWalkAttacker → diverse termination).

The session-scoped CPU Genesis init from ``tests/conftest.py`` is reused.
We construct local PE envs with ``episode_length_s=0.05`` (10 steps) so each
test stays under a few seconds.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import genesis as gs  # noqa: E402

from scripts.collect_forecaster_data import (  # noqa: E402
    S_WORLD_DIM,
    TERMINATION_REASONS,
    _resolve_cfgs_pkl,
    classify_termination,
    collect_episodes,
    load_cfgs,
    write_metadata,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_short_pe_env(num_envs: int = 4, episode_length_s: float = 0.05):
    """Build a tiny PE env (ego=attacker + StationaryDefender) for fast tests."""
    from envs.pe_1v1.cfgs import (
        load_pe_1v1_cfg,
        make_command_cfg,
        make_env_cfg,
        make_obs_cfg,
        make_reward_cfg,
    )
    from envs.pe_1v1.env import PursuitEvasion1v1Env
    from envs.pe_1v1.scripted_defender import StationaryDefender
    from utils.drone_params import DroneParams

    raw = load_pe_1v1_cfg()
    raw["sim"]["episode_length_s"] = float(episode_length_s)
    env_cfg = make_env_cfg(raw, ego="attacker")
    env_cfg["tau_delay"] = 0
    obs_cfg = make_obs_cfg(raw)
    reward_cfg = make_reward_cfg(raw)
    command_cfg = make_command_cfg(raw)

    drone_yaml = ROOT / "configs" / "drones" / "nova.yaml"
    params = DroneParams(str(drone_yaml))
    defender = StationaryDefender(num_envs=num_envs, device=gs.device)
    return PursuitEvasion1v1Env(
        num_envs=num_envs,
        params=params,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        defender_policy=defender,
        show_viewer=False,
    )


# ---------------------------------------------------------------------------
# Test 1 — end-to-end small-scale collection
# ---------------------------------------------------------------------------

def test_collect_10_episodes_end_to_end(tmp_path):
    """10 episode 수집, 파일 구조 + per-episode schema 검증."""
    from envs.pe_1v1.scripted_attacker import StationaryAttacker

    env = _make_short_pe_env(num_envs=4, episode_length_s=0.05)
    attacker = StationaryAttacker(num_envs=4, device=gs.device)

    out_dir = tmp_path / "data"
    collected, episodes = collect_episodes(
        env=env, attacker=attacker,
        num_episodes=10,
        output_dir=out_dir,
        seed=0,
        attacker_ckpt_path="<test>",
        defender_ckpt_path="<test>",
        log_every_n_episodes=1000,
        checkpoint_progress_every=1000,
    )
    assert collected >= 10

    # File structure
    assert (out_dir / "index.json").exists()
    assert (out_dir / "progress.json").exists()
    assert (out_dir / "episode_000000.pt").exists()
    assert (out_dir / "episode_000009.pt").exists()

    # Data shape on first episode
    ep0 = torch.load(out_dir / "episode_000000.pt", weights_only=False)
    L = ep0["episode_length"]
    assert L >= 1
    assert ep0["s_a_world"].shape == (L, S_WORLD_DIM)
    assert ep0["s_d_world"].shape == (L, S_WORLD_DIM)
    assert ep0["a_a"].shape == (L, 4)
    assert ep0["a_d"].shape == (L, 4)
    assert ep0["g_mission"].shape == (L, 3)
    assert ep0["termination_reason"] in TERMINATION_REASONS
    # s_a_obs: env _attacker_obs() — 31D under current schema, but tolerate
    # changes by asserting only the leading dim matches L.
    assert ep0["s_a_obs"].shape[0] == L
    assert ep0["s_a_obs"].dim() == 2
    # Per-episode deterministic seed = base_seed + episode_id.
    assert ep0["seed"] == 0
    # Ckpt paths persisted.
    assert ep0["attacker_ckpt_path"] == "<test>"
    assert ep0["defender_ckpt_path"] == "<test>"


# ---------------------------------------------------------------------------
# Test 2 — metadata + index completeness
# ---------------------------------------------------------------------------

def test_metadata_and_index_complete(tmp_path):
    """``metadata.json`` + ``index.json`` 모든 required field 존재."""
    from envs.pe_1v1.scripted_attacker import StationaryAttacker

    env = _make_short_pe_env(num_envs=4, episode_length_s=0.05)
    attacker = StationaryAttacker(num_envs=4, device=gs.device)

    out_dir = tmp_path / "data"
    collected, _ = collect_episodes(
        env=env, attacker=attacker,
        num_episodes=5,
        output_dir=out_dir,
        seed=42,
        attacker_ckpt_path="att.pt",
        defender_ckpt_path="def.pt",
        log_every_n_episodes=1000,
        checkpoint_progress_every=1000,
    )
    write_metadata(
        output_dir=out_dir,
        config={
            "attacker_ckpt": "att.pt",
            "defender_ckpt": "def.pt",
            "tau_delay": 0,
            "num_envs": 4,
            "seed": 42,
            "env_yaml_override": None,
        },
        cfgs_pkl=Path("dummy_cfgs.pkl"),
        elapsed_seconds=1.5,
        total_episodes=collected,
    )

    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["tau_delay"] == 0
    assert "attacker_ckpt" in meta
    assert "defender_ckpt" in meta
    assert "cfgs_pkl" in meta
    assert meta["env_yaml_override"] is None
    assert meta["num_envs"] == 4
    assert meta["seed"] == 42
    assert "collection_date" in meta
    assert "elapsed_time_seconds" in meta
    assert meta["total_episodes"] == collected

    with open(out_dir / "index.json") as f:
        idx = json.load(f)
    assert idx["total_episodes"] == collected
    assert len(idx["episodes"]) == collected
    assert "mean_length" in idx["stats"]
    assert "min_length" in idx["stats"]
    assert "max_length" in idx["stats"]
    assert "termination_dist" in idx["stats"]
    for k in TERMINATION_REASONS:
        assert k in idx["stats"]["termination_dist"]
    # Per-episode index entries have required keys.
    for e in idx["episodes"]:
        assert {"id", "length", "termination", "file"} <= set(e)
        assert e["termination"] in TERMINATION_REASONS


# ---------------------------------------------------------------------------
# Test 3 — cfgs.pkl resolution + load (with env_yaml_override behavior)
# ---------------------------------------------------------------------------

def test_cfgs_pkl_auto_resolution(tmp_path):
    """``_resolve_cfgs_pkl(..., 'auto')`` finds sibling cfgs.pkl and load_cfgs round-trips."""
    fake_dir = tmp_path / "cycle_016"
    fake_dir.mkdir()
    fake_cfgs = {
        "env_cfg": {
            "dt": 0.005,
            "tau_delay": 5,
            "ego": "attacker",
            "episode_length_s": 5.0,
            "use_asymmetric_critic": False,
        },
        "obs_cfg": {"attacker_obs_dim": 31},
        "reward_cfg": {"some_key": 1.0},
        "command_cfg": {"spawn": {}, "g_mission": {}},
        "drone_yaml": "drone.yaml",
        "train_cfg": {},
        "ams_stage": "test",
    }
    with open(fake_dir / "cfgs.pkl", "wb") as f:
        pickle.dump(fake_cfgs, f)

    fake_attacker_ckpt = fake_dir / "attacker_v16.pt"
    fake_attacker_ckpt.write_bytes(b"dummy")

    # auto resolve → finds sibling cfgs.pkl
    resolved = _resolve_cfgs_pkl(fake_attacker_ckpt, "auto")
    assert resolved.resolve() == (fake_dir / "cfgs.pkl").resolve()

    # load_cfgs returns the cached env_cfg verbatim (no override)
    cfgs_pkl, cfgs = load_cfgs(
        fake_attacker_ckpt, cfgs_pkl_source="auto", env_yaml_override=None,
    )
    assert cfgs_pkl.resolve() == (fake_dir / "cfgs.pkl").resolve()
    assert cfgs["env_cfg"]["dt"] == 0.005
    assert cfgs["env_cfg"]["tau_delay"] == 5
    assert cfgs["env_cfg"]["ego"] == "attacker"
    assert cfgs["obs_cfg"]["attacker_obs_dim"] == 31

    # Explicit path also works
    resolved2 = _resolve_cfgs_pkl(fake_attacker_ckpt, str(fake_dir / "cfgs.pkl"))
    assert resolved2 == fake_dir / "cfgs.pkl"

    # Missing → FileNotFoundError (informative)
    bad_dir = tmp_path / "lonely"
    bad_dir.mkdir()
    bad_ckpt = bad_dir / "att.pt"
    bad_ckpt.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        _resolve_cfgs_pkl(bad_ckpt, "auto")


# ---------------------------------------------------------------------------
# Test 4 — variable-length episodes via RandomWalk attacker
# ---------------------------------------------------------------------------

def test_variable_episode_lengths(tmp_path):
    """RandomWalk attacker → diverse trajectories → variable episode lengths."""
    from envs.pe_1v1.scripted_attacker import RandomWalkAttacker

    env = _make_short_pe_env(num_envs=4, episode_length_s=0.05)
    attacker = RandomWalkAttacker(num_envs=4, device=gs.device)

    out_dir = tmp_path / "data"
    collected, _ = collect_episodes(
        env=env, attacker=attacker,
        num_episodes=20,
        output_dir=out_dir,
        seed=123,
        attacker_ckpt_path="<rw>",
        defender_ckpt_path="<test>",
        log_every_n_episodes=1000,
        checkpoint_progress_every=1000,
    )
    assert collected >= 20

    with open(out_dir / "index.json") as f:
        idx = json.load(f)
    lengths = [e["length"] for e in idx["episodes"]]
    assert len(lengths) == collected
    assert min(lengths) >= 1
    # max_episode_length comes from episode_length_s/dt; cap is a soft check.
    assert max(lengths) <= int(env.max_episode_length)

    # Termination distribution sums to 1 (within float tolerance).
    term_dist = idx["stats"]["termination_dist"]
    s = sum(term_dist[k] for k in TERMINATION_REASONS)
    assert abs(s - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Extra unit: classify_termination priority order
# ---------------------------------------------------------------------------

def test_classify_termination_priority():
    """mission > captured > crash/oob > timeout."""
    last_term = {
        "mission":   torch.tensor([True, False, False, False]),
        "captured":  torch.tensor([True, True,  False, False]),  # env 0 also captured
        "att_crash": torch.tensor([False, False, True, False]),
        "def_crash": torch.tensor([False, False, False, False]),
        "att_oob":   torch.tensor([False, False, False, False]),
        "def_oob":   torch.tensor([False, False, False, False]),
        "crash":     torch.tensor([False, False, True,  False]),
        "oob":       torch.tensor([False, False, False, False]),
        "timeout":   torch.tensor([False, False, False, True]),
    }
    assert classify_termination(last_term, 0) == "mission_success"
    assert classify_termination(last_term, 1) == "captured"
    assert classify_termination(last_term, 2) == "out_of_bounds"
    assert classify_termination(last_term, 3) == "timeout"
