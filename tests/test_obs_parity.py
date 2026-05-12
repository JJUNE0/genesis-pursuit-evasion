"""PR-N parity tests — scripted obs builders vs env obs.

The PR-N (Phase 5 alternating co-training) frozen policies (``PretrainedAttacker``,
``PretrainedDefender``) build their actor inputs from raw state dicts via
``_build_attacker_policy_obs`` / ``_build_defender_policy_obs``. The obs schema
must match ``env._attacker_obs`` / ``env._defender_obs`` **bit-for-bit**, or
the loaded actor sees a different distribution than it was trained on and
produces garbage actions.

This test asserts ``torch.allclose(builder_output, env_output, atol=1e-5)``
after stepping the env 2+ times so neither last_action nor quat is identity-trivial.
"""

from __future__ import annotations

import torch

from envs.pe_1v1.scripted_attacker import _build_attacker_policy_obs
from envs.pe_1v1.scripted_defender import _build_defender_policy_obs


def _step_to_nontrivial(env, num_steps: int = 3) -> None:
    """Run a few env steps with random actions so last_action != 0 and quat
    has drifted from identity (drone has rotated under random RPM input)."""
    env.reset()
    for _ in range(num_steps):
        actions = (torch.rand(
            (env.num_envs, env.num_actions), device=env.device,
        ) * 2.0 - 1.0)
        env.step(actions)


def test_attacker_obs_parity(pe_env_attacker_stationary):
    """_build_attacker_policy_obs == env._attacker_obs after 3 random steps."""
    env = pe_env_attacker_stationary
    _step_to_nontrivial(env, num_steps=3)

    env_obs = env._attacker_obs()
    built_obs = _build_attacker_policy_obs(
        ego_state=env._attacker_state_dict(),
        opponent_state=env._defender_state_dict(),
        g_mission=env.g_mission,
        last_action=env.last_attacker_action,
        pos_scale=env.pos_scale,
        vel_scale=env.vel_scale,
        ang_vel_scale=env.ang_vel_scale,
    )
    assert env_obs.shape == built_obs.shape == (env.num_envs, 31)
    # Both go through nan_to_num eventually — env_obs is raw, built has it.
    # In a well-behaved sim step both are finite, so allclose holds without
    # explicit nan_to_num on env_obs.
    assert torch.allclose(env_obs, built_obs, atol=1e-5), (
        "attacker obs schema mismatch:\n"
        f"  env_obs[0]   = {env_obs[0]}\n"
        f"  built_obs[0] = {built_obs[0]}\n"
        f"  max abs diff = {(env_obs - built_obs).abs().max().item():.3e}"
    )

    # Also assert non-trivial state was reached (sanity).
    assert env.last_attacker_action.abs().sum().item() > 0, (
        "fixture did not produce non-zero last_attacker_action"
    )


def test_defender_obs_parity(pe_env_defender_stationary):
    """_build_defender_policy_obs == env._defender_obs after 3 random steps."""
    env = pe_env_defender_stationary
    _step_to_nontrivial(env, num_steps=3)

    env_obs = env._defender_obs()
    built_obs = _build_defender_policy_obs(
        ego_state=env._defender_state_dict(),
        opponent_state=env._attacker_state_dict(),
        last_action=env.last_defender_action,
        pos_scale=env.pos_scale,
        vel_scale=env.vel_scale,
        ang_vel_scale=env.ang_vel_scale,
    )
    assert env_obs.shape == built_obs.shape == (env.num_envs, 28)
    assert torch.allclose(env_obs, built_obs, atol=1e-5), (
        "defender obs schema mismatch:\n"
        f"  env_obs[0]   = {env_obs[0]}\n"
        f"  built_obs[0] = {built_obs[0]}\n"
        f"  max abs diff = {(env_obs - built_obs).abs().max().item():.3e}"
    )

    assert env.last_defender_action.abs().sum().item() > 0, (
        "fixture did not produce non-zero last_defender_action"
    )


# Note: PretrainedAttacker.step() raises ValueError when g_mission is None
# (explicit guard at envs/pe_1v1/scripted_attacker.py:218). That contract is
# covered in tests/test_scripted_policies.py alongside other PretrainedDefender
# / PretrainedAttacker guard tests; this file stays focused on the obs-schema
# parity that gates a clean PR-N merge.
