"""Phase 1 shape contract — obs / action / reward / done shapes & dtypes.

CLAUDE.md §4 mandates explicit shape comments and dtype consistency. These
tests pin those contracts so any future refactor that drifts the obs layout
fails loudly here instead of silently in PPO.
"""

from __future__ import annotations

import torch

import genesis as gs

from envs.pe_1v1.cfgs import ATTACKER_OBS_DIM, NUM_ACTIONS


def test_attacker_obs_shape_and_dtype(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    obs = env.get_observations()
    assert "policy" in obs.keys()
    pol = obs["policy"]
    assert pol.shape == (env.num_envs, ATTACKER_OBS_DIM)
    assert pol.dtype == gs.tc_float
    assert pol.device.type == env.device.type
    assert torch.isfinite(pol).all()


def test_critic_obs_groups_phase1(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    # Phase 1 wiring: critic reads "policy" only — no critic_priv built.
    assert env.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    obs = env.get_observations()
    assert "critic_priv" not in obs.keys()


def test_step_returns_correct_shapes(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    actions = torch.zeros((env.num_envs, NUM_ACTIONS), device=env.device, dtype=gs.tc_float)
    obs, rew, done, extras = env.step(actions)
    assert obs["policy"].shape == (env.num_envs, ATTACKER_OBS_DIM)
    assert rew.shape == (env.num_envs,)
    assert done.shape == (env.num_envs,)
    assert rew.dtype == gs.tc_float
    assert done.dtype == torch.bool
    assert torch.isfinite(rew).all()
    assert "episode" in extras
    assert "time_outs" in extras


def test_action_clip_handles_out_of_range(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    # large + NaN values — env must nan_to_num and clip to [-1, 1]
    actions = torch.full((env.num_envs, NUM_ACTIONS), 5.0, device=env.device, dtype=gs.tc_float)
    actions[0, 0] = float("nan")
    obs, rew, done, extras = env.step(actions)
    assert torch.isfinite(env.actions).all()
    assert env.actions.abs().max() <= env.clip_actions + 1e-6


def test_internal_state_buffer_shapes(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    B = env.num_envs
    assert env.attacker_pos.shape == (B, 3)
    assert env.attacker_vel.shape == (B, 3)
    assert env.attacker_quat.shape == (B, 4)
    assert env.attacker_ang_vel.shape == (B, 3)
    assert env.defender_pos.shape == (B, 3)
    assert env.defender_vel.shape == (B, 3)
    assert env.defender_quat.shape == (B, 4)
    assert env.defender_ang_vel.shape == (B, 3)
    assert env.g_mission.shape == (B, 3)
    for t in (
        env.attacker_pos, env.defender_pos, env.attacker_vel, env.defender_vel,
        env.attacker_quat, env.defender_quat, env.g_mission,
    ):
        assert t.dtype == gs.tc_float
