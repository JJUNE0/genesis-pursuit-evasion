"""Each of the five termination conditions (mission / captured / crash /
oob / timeout) must independently trigger ``done`` and route to the
correct ``extras["episode"]`` rate counter."""

from __future__ import annotations

import torch

import genesis as gs


def _zero_actions(env) -> torch.Tensor:
    return torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float)


def test_mission_reached_alone_triggers_attacker_win(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    B = env.num_envs
    # attacker AT g_mission, defender far away
    g = torch.zeros((B, 3), device=env.device, dtype=gs.tc_float)
    g[:, 2] = 1.0
    far = g.clone()
    far[:, 0] = 4.5
    env.attacker_drone.set_pos(g, zero_velocity=True)
    env.defender_drone.set_pos(far, zero_velocity=True)
    env.attacker_pos[:] = g
    env.defender_pos[:] = far
    env.g_mission[:] = g
    env._update_pe_state()
    env.last_dist_to_goal[:] = env.dist_to_goal

    _, _, done, extras = env.step(_zero_actions(env))
    assert done.all().item()
    assert extras["episode"]["attacker_win_rate"] == 1.0
    assert extras["episode"]["captured_rate"] == 0.0
    assert extras["episode"]["crash_rate"] == 0.0


def test_captured_alone_triggers_defender_win(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    B = env.num_envs
    pos = torch.zeros((B, 3), device=env.device, dtype=gs.tc_float)
    pos[:, 2] = 1.0
    env.attacker_drone.set_pos(pos, zero_velocity=True)
    env.defender_drone.set_pos(pos, zero_velocity=True)
    env.attacker_pos[:] = pos
    env.defender_pos[:] = pos
    far_goal = pos.clone()
    far_goal[:, 0] = 4.0
    env.g_mission[:] = far_goal
    env._update_pe_state()
    env.last_dist_to_goal[:] = env.dist_to_goal

    _, _, done, extras = env.step(_zero_actions(env))
    assert done.all().item()
    assert extras["episode"]["captured_rate"] == 1.0
    assert extras["episode"]["attacker_win_rate"] == 0.0
    # attacker reward should be R_captured = -1.0 dominated
    assert (env.rew_attacker_buf < -0.5).all()
    assert (env.rew_defender_buf > 0.5).all()


def test_crash_below_floor_triggers(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    B = env.num_envs
    # place attacker below z_floor — crash
    bad = torch.zeros((B, 3), device=env.device, dtype=gs.tc_float)
    bad[:, 2] = env.z_floor - 0.5   # below floor
    safe = bad.clone()
    safe[:, 2] = 1.0
    safe[:, 0] = 4.5
    env.attacker_drone.set_pos(bad, zero_velocity=True)
    env.defender_drone.set_pos(safe, zero_velocity=True)
    env.attacker_pos[:] = bad
    env.defender_pos[:] = safe
    far_goal = safe.clone()
    far_goal[:, 0] = -4.0
    env.g_mission[:] = far_goal
    env._update_pe_state()
    env.last_dist_to_goal[:] = env.dist_to_goal

    _, _, done, extras = env.step(_zero_actions(env))
    assert done.all().item()
    assert extras["episode"]["crash_rate"] == 1.0
    assert extras["episode"]["attacker_win_rate"] == 0.0
    assert extras["episode"]["captured_rate"] == 0.0


def test_oob_triggers(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    B = env.num_envs
    # x outside bound_xyz
    bad = torch.zeros((B, 3), device=env.device, dtype=gs.tc_float)
    bad[:, 0] = env.bound_xyz + 1.0   # outside workspace
    bad[:, 2] = 1.0
    safe = bad.clone()
    safe[:, 0] = 0.0
    env.attacker_drone.set_pos(bad, zero_velocity=True)
    env.defender_drone.set_pos(safe, zero_velocity=True)
    env.attacker_pos[:] = bad
    env.defender_pos[:] = safe
    g = safe.clone()
    g[:, 0] = -2.0
    env.g_mission[:] = g
    env._update_pe_state()
    env.last_dist_to_goal[:] = env.dist_to_goal

    _, _, done, extras = env.step(_zero_actions(env))
    assert done.all().item()
    # oob and crash bucket overlap if z criteria also hit; we only care that
    # at least one of {oob, crash} fired and mission/captured did not.
    assert extras["episode"]["attacker_win_rate"] == 0.0
    assert extras["episode"]["captured_rate"] == 0.0
    assert (extras["episode"]["oob_rate"] + extras["episode"]["crash_rate"]) == 1.0


def test_timeout_triggers_after_max_steps(pe_env_attacker_stationary):
    """Force episode_length_buf to max - 1 and step once → timeout fires."""
    env = pe_env_attacker_stationary
    env.reset()
    B = env.num_envs
    # Spawn drones far apart so neither mission_reached nor captured triggers.
    a = torch.zeros((B, 3), device=env.device, dtype=gs.tc_float)
    a[:, 2] = 1.0
    d = a.clone()
    d[:, 0] = 4.5
    g = a.clone()
    g[:, 0] = -4.5  # far from attacker
    env.attacker_drone.set_pos(a, zero_velocity=True)
    env.defender_drone.set_pos(d, zero_velocity=True)
    env.attacker_pos[:] = a
    env.defender_pos[:] = d
    env.g_mission[:] = g
    env._update_pe_state()
    env.last_dist_to_goal[:] = env.dist_to_goal
    # Push counter so the next ``step`` increments to >= max_episode_length.
    env.episode_length_buf[:] = env.max_episode_length - 1

    _, _, done, extras = env.step(_zero_actions(env))
    assert done.all().item()
    assert extras["episode"]["timeout_rate"] == 1.0
    assert extras["episode"]["attacker_win_rate"] == 0.0
