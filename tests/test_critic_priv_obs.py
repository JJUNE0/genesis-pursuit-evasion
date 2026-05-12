"""Phase 3d — BPQL asymmetric critic privileged obs 단위 테스트.

Supersedes the PR-E.1 (2026-05-08) 10D test which pinned the now-deprecated
``vel(3)+ang_vel(3)+quat(4)`` spec. TODO §3d / CLAUDE.md §7.

What this test pins:
  1. CRITIC_PRIV_DIM constant equals 6 (BPQL body-frame rel kinematics).
  2. obs_groups branching when use_asymmetric_critic=True.
  3. ``critic_priv`` key present in TensorDict with shape (B, 6).
  4. Content = ``rel_pos_GT(3) + rel_vel_GT(3)`` in attacker body frame.
  5. τ=0 → critic_priv equals the actor's s_d_kin slice bit-for-bit
     (same representation, delay collapses).
  6. τ>0 → critic_priv stays GT current, actor's s_d_kin is delayed
     (the BPQL contrast that makes critic useful).
  7. Backward compat: use_asymmetric_critic=False → no critic_priv key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import genesis as gs

ROOT = Path(__file__).resolve().parents[1]


def _build_env(
    use_asymmetric: bool, ego: str = "attacker", tau_delay: int = 0,
    num_envs: int = 4, defender_kind: str = "stationary",
):
    from envs.pe_1v1.cfgs import (
        load_pe_1v1_cfg,
        make_command_cfg,
        make_env_cfg,
        make_obs_cfg,
        make_reward_cfg,
    )
    from envs.pe_1v1.env import PursuitEvasion1v1Env
    from envs.pe_1v1.scripted_attacker import StationaryAttacker
    from envs.pe_1v1.scripted_defender import RandomWalkDefender, StationaryDefender
    from utils.drone_params import DroneParams

    raw = load_pe_1v1_cfg()
    raw["tau_delay"] = int(tau_delay)
    env_cfg = make_env_cfg(raw, ego=ego, use_asymmetric_critic=use_asymmetric)
    obs_cfg = make_obs_cfg(raw)
    reward_cfg = make_reward_cfg(raw)
    command_cfg = make_command_cfg(raw)

    drone_yaml = ROOT / "configs" / "drones" / "nova.yaml"
    params = DroneParams(str(drone_yaml))

    if ego == "attacker":
        if defender_kind == "random":
            defender = RandomWalkDefender(num_envs=num_envs, device=gs.device)
        else:
            defender = StationaryDefender(num_envs=num_envs, device=gs.device)
        kwargs = {"defender_policy": defender}
    else:
        attacker = StationaryAttacker(num_envs=num_envs, device=gs.device)
        kwargs = {"attacker_policy": attacker}

    return PursuitEvasion1v1Env(
        num_envs=num_envs, params=params, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False, **kwargs,
    )


# ---------------------------------------------------------------------------
# cfg-only
# ---------------------------------------------------------------------------

def test_make_env_cfg_default_no_asymmetric():
    """make_env_cfg default = False (backward compat)."""
    from envs.pe_1v1.cfgs import load_pe_1v1_cfg, make_env_cfg
    raw = load_pe_1v1_cfg()
    cfg = make_env_cfg(raw, ego="attacker")
    assert cfg["use_asymmetric_critic"] is False


def test_make_env_cfg_asymmetric_true():
    from envs.pe_1v1.cfgs import load_pe_1v1_cfg, make_env_cfg
    raw = load_pe_1v1_cfg()
    cfg = make_env_cfg(raw, ego="attacker", use_asymmetric_critic=True)
    assert cfg["use_asymmetric_critic"] is True


def test_critic_priv_dim_constant():
    """Phase 3d: 10D (vel+ang_vel+quat) → 6D (rel_pos+rel_vel body frame)."""
    from envs.pe_1v1.cfgs import CRITIC_PRIV_DIM
    assert CRITIC_PRIV_DIM == 6


# ---------------------------------------------------------------------------
# Integration — build envs once per τ/asymmetry combo
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env_attacker_asym_tau0(_gs_init):
    return _build_env(use_asymmetric=True, ego="attacker", tau_delay=0)


@pytest.fixture(scope="module")
def env_attacker_asym_tau5(_gs_init):
    return _build_env(use_asymmetric=True, ego="attacker", tau_delay=5)


@pytest.fixture(scope="module")
def env_attacker_asym_tau5_randomwalk(_gs_init):
    """τ=5 + RandomWalkDefender — guarantees defender displacement so
    critic_priv (GT current) and actor s_d_kin (delayed) diverge unconditionally.
    """
    return _build_env(
        use_asymmetric=True, ego="attacker", tau_delay=5,
        defender_kind="random",
    )


def test_env_use_asymmetric_critic_flag(env_attacker_asym_tau0):
    env = env_attacker_asym_tau0
    assert env.use_asymmetric_critic is True


def test_obs_groups_includes_critic_priv(env_attacker_asym_tau0):
    env = env_attacker_asym_tau0
    assert env.obs_groups == {"actor": ["policy"], "critic": ["policy", "critic_priv"]}


def test_get_observations_has_critic_priv_key(env_attacker_asym_tau0):
    """TensorDict carries 'critic_priv' key with shape (B, 6)."""
    from envs.pe_1v1.cfgs import CRITIC_PRIV_DIM
    env = env_attacker_asym_tau0
    env.reset()
    td = env.get_observations()
    assert "policy" in td.keys()
    assert "critic_priv" in td.keys()
    assert td["critic_priv"].shape == (env.num_envs, CRITIC_PRIV_DIM)


def test_critic_priv_obs_no_nan(env_attacker_asym_tau0):
    env = env_attacker_asym_tau0
    env.reset()
    td = env.get_observations()
    assert torch.isfinite(td["critic_priv"]).all()


def test_critic_priv_content_body_frame_rel(env_attacker_asym_tau0):
    """critic_priv = rel_pos_GT + rel_vel_GT in attacker body frame."""
    from envs.base_pe_env import rotate_to_body_frame
    env = env_attacker_asym_tau0
    env.reset()
    td = env.get_observations()
    cp = td["critic_priv"]                                       # (B, 6)

    expected_rel_pos = rotate_to_body_frame(
        env.defender_pos - env.attacker_pos, env.attacker_quat,
    ) * env.pos_scale
    expected_rel_vel = rotate_to_body_frame(
        env.defender_vel - env.attacker_vel, env.attacker_quat,
    ) * env.vel_scale

    assert torch.allclose(cp[:, :3], expected_rel_pos, atol=1e-5)
    assert torch.allclose(cp[:, 3:], expected_rel_vel, atol=1e-5)


def test_critic_priv_tau0_equals_actor_sd_kin(env_attacker_asym_tau0):
    """τ=0 (no delay) — critic_priv == actor obs s_d_kin slice [18:24].

    At τ=0 the delay buffer holds current state, so the actor's body-frame
    rel kinematics are computed from the same (current) defender pose as
    critic_priv. Both should be bit-for-bit identical.
    """
    env = env_attacker_asym_tau0
    env.reset()
    td = env.get_observations()
    cp = td["critic_priv"]                                       # (B, 6)
    actor_sd_kin = td["policy"][:, 18:24]                        # (B, 6)
    assert torch.allclose(cp, actor_sd_kin, atol=1e-6)


def test_critic_priv_tau5_differs_from_actor_sd_kin(
    env_attacker_asym_tau5_randomwalk,
):
    """τ>0, moving defender — critic_priv (GT current) ≠ actor s_d_kin (delayed).

    Uses RandomWalkDefender so the defender accumulates a non-trivial
    displacement within a few env steps, guaranteeing the GT/delayed contrast
    rather than relying on a conditional assertion (Stationary defender +
    zero action would degenerate to a no-op).

    Pins:
      - delayed defender state (buffer[0]) is genuinely stale (≠ current).
      - critic_priv equals current-state body-frame rel kinematics.
      - critic_priv != actor s_d_kin under the BPQL asymmetry.
    """
    from envs.base_pe_env import rotate_to_body_frame
    env = env_attacker_asym_tau5_randomwalk
    env.reset()
    # Sample non-zero attacker actions so attacker pose also drifts — the
    # BPQL contrast lives across both the delayed defender AND the rotation
    # frame change; we want both to be visible.
    torch.manual_seed(0)
    for _ in range(5):
        action = torch.rand(
            (env.num_envs, env.num_actions), device=env.device,
        ) * 0.4 - 0.2
        env.step(action)
    td = env.get_observations()
    cp = td["critic_priv"]
    actor_sd_kin = td["policy"][:, 18:24]

    # 1. Buffer's oldest slot must NOT equal current state — guarantees the
    # input contrast that BPQL is supposed to exploit.
    delayed = env.defender_state_history[0]
    current = torch.cat([env.defender_pos, env.defender_vel], dim=1)
    assert not torch.allclose(delayed, current, atol=1e-3), (
        "RandomWalkDefender should have produced detectable drift in 5 steps"
    )

    # 2. critic_priv = current-state body-frame rel kinematics (BPQL spec).
    expected_rel_pos = rotate_to_body_frame(
        env.defender_pos - env.attacker_pos, env.attacker_quat,
    ) * env.pos_scale
    expected_rel_vel = rotate_to_body_frame(
        env.defender_vel - env.attacker_vel, env.attacker_quat,
    ) * env.vel_scale
    assert torch.allclose(cp[:, :3], expected_rel_pos, atol=1e-5)
    assert torch.allclose(cp[:, 3:], expected_rel_vel, atol=1e-5)

    # 3. Unconditional BPQL contrast: actor's delayed view differs from
    # critic's GT view by more than numerical noise.
    assert not torch.allclose(cp, actor_sd_kin, atol=1e-3), (
        "critic_priv (GT current) should differ from actor s_d_kin (delayed) "
        "under non-trivial defender motion"
    )


def test_backward_compat_no_asymmetric(pe_env_attacker_stationary):
    """fixture (use_asymmetric_critic=False default) — no critic_priv key."""
    env = pe_env_attacker_stationary
    assert env.use_asymmetric_critic is False
    assert env.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    env.reset()
    td = env.get_observations()
    assert "critic_priv" not in td.keys()
