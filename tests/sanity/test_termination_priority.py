"""TODO.md §1.2 priority rule — mission_reached + captured 동시 trigger
시 attacker win 으로 분류되어야 한다.

Force the attacker to be inside the goal radius AND inside the capture
radius simultaneously, run one step with zero action, then check that the
priority-resolved buffers + episode extras report attacker_win.
"""

from __future__ import annotations

import torch

import genesis as gs


def _force_collision_at_goal(env) -> None:
    """Place attacker, defender, and g_mission at the same point."""
    B = env.num_envs
    pos = torch.zeros((B, 3), device=env.device, dtype=gs.tc_float)
    pos[:, 2] = 1.0
    env.attacker_drone.set_pos(pos, zero_velocity=True)
    env.defender_drone.set_pos(pos, zero_velocity=True)
    # mirror buffers (Genesis read-back will overwrite after step)
    env.attacker_pos[:] = pos
    env.defender_pos[:] = pos
    env.g_mission[:] = pos
    # Reset distance snapshot so progress reward is well-defined this step.
    env._update_pe_state()
    env.last_dist_to_goal[:] = env.dist_to_goal


def test_mission_priority_over_capture(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    _force_collision_at_goal(env)

    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float)
    _, _, done, extras = env.step(actions)

    # All envs terminated this step.
    assert done.all().item()
    # Priority: every env should be classified as attacker_win, not captured.
    assert extras["episode"]["attacker_win_rate"] == 1.0
    assert extras["episode"]["captured_rate"] == 0.0


def test_attacker_reward_is_mission_win_when_priority_resolved(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    _force_collision_at_goal(env)

    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float)
    env.step(actions)

    # Attacker reward should be dominated by R_mission_win = +1.0; dense
    # corrections are bounded by ±0.002 progress + ±0.0001 smooth + existential.
    # If priority were broken (captured won), reward would be ≈ R_captured = -1.0.
    assert (env.rew_attacker_buf > 0.5).all()
    assert (env.rew_defender_buf < -0.5).all()


def test_zero_sum_terminal_under_priority(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.reset()
    _force_collision_at_goal(env)

    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device, dtype=gs.tc_float)
    env.step(actions)

    # Dense terms cancel: attacker R_progress + defender R_block = 0,
    # action_smooth equal magnitude at zero-action so equal & nonpositive on
    # both sides; existential is attacker-only.  Their sum should be exactly:
    #   (R_mission_win + R_mission_failed) = 0   (terminal zero-sum)
    # + (R_progress + R_block) = 0                (dense zero-sum)
    # + (R_smooth_a + R_smooth_d) = 2 * R_smooth (both equal, nonpositive)
    # + R_existential                             (attacker only, negative)
    # so total < 0 but the *terminal* component must zero exactly.  We test
    # the looser overall property here; the strict zero-sum-of-terminals
    # is checked in test_reward_zero_sum.
    total = env.rew_attacker_buf + env.rew_defender_buf
    # No NaN / Inf and bounded magnitude.
    assert torch.isfinite(total).all()
    assert total.abs().max() < 0.05   # only existential + double smooth contribute
