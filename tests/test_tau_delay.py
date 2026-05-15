"""Phase 3a — τ-delay ring buffer 단위 테스트 (TODO §3a, 2026-05-12).

What this test pins:
  1. cfgs.make_env_cfg default tau_delay=0, explicit tau_delay value pass-through.
  2. BasePursuitEvasionEnv.__init__ builds defender_state_history of shape
     (max(1, τ+1), B, 6) and exposes self.tau_delay.
  3. tau_delay < 0 → ValueError.
  4. After reset(), every slot of defender_state_history equals current state
     (warmup fill — no zero garbage).
  5. After one step, [-1] holds fresh state and prior entries shift toward
     [0] (oldest end).
  6. τ=0 backward-compat: actor obs s_d_kin slot identical to the
     ground-truth current-state body-frame relative kinematics.
  7. τ>0: while episode_length < τ, actor obs s_d_kin slot uses the spawn
     state (oldest in buffer) — not the current state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import genesis as gs

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_env(tau_delay: int, num_envs: int = 4):
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
    raw["tau_delay"] = int(tau_delay)
    env_cfg = make_env_cfg(raw, ego="attacker")
    obs_cfg = make_obs_cfg(raw)
    reward_cfg = make_reward_cfg(raw)
    command_cfg = make_command_cfg(raw)

    drone_yaml = ROOT / "configs" / "drones" / "nova.yaml"
    params = DroneParams(str(drone_yaml))

    defender = StationaryDefender(num_envs=num_envs, device=gs.device)
    return PursuitEvasion1v1Env(
        num_envs=num_envs, params=params, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        defender_policy=defender, show_viewer=False,
    )


# ---------------------------------------------------------------------------
# cfg-only tests (no Scene build)
# ---------------------------------------------------------------------------

def test_make_env_cfg_default_tau_delay_zero():
    """Default yaml + make_env_cfg ⇒ tau_delay=0 (backward compat)."""
    from envs.pe_1v1.cfgs import load_pe_1v1_cfg, make_env_cfg
    raw = load_pe_1v1_cfg()
    raw.pop("tau_delay", None)
    cfg = make_env_cfg(raw, ego="attacker")
    assert cfg["tau_delay"] == 0


def test_make_env_cfg_explicit_tau_delay():
    from envs.pe_1v1.cfgs import load_pe_1v1_cfg, make_env_cfg
    raw = load_pe_1v1_cfg()
    raw["tau_delay"] = 25
    cfg = make_env_cfg(raw, ego="attacker")
    assert cfg["tau_delay"] == 25


def test_default_yaml_has_tau_delay_key():
    """pe_1v1_default.yaml carries an explicit tau_delay: 0 marker."""
    from envs.pe_1v1.cfgs import load_pe_1v1_cfg
    raw = load_pe_1v1_cfg()
    assert raw.get("tau_delay") == 0


# ---------------------------------------------------------------------------
# Heavy env tests (module-scope so Scene build is paid once per τ).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env_tau0(_gs_init):
    return _build_env(tau_delay=0, num_envs=4)


@pytest.fixture(scope="module")
def env_tau5(_gs_init):
    return _build_env(tau_delay=5, num_envs=4)


def test_env_exposes_tau_delay_attr(env_tau0, env_tau5):
    assert env_tau0.tau_delay == 0
    assert env_tau5.tau_delay == 5


def test_buffer_shape_tau_0(env_tau0):
    """τ=0 → buf_size=1, shape (1, B, 6)."""
    env = env_tau0
    assert env.defender_state_history.shape == (1, env.num_envs, 6)


def test_buffer_shape_tau_5(env_tau5):
    """τ=5 → buf_size=6, shape (6, B, 6)."""
    env = env_tau5
    assert env.defender_state_history.shape == (6, env.num_envs, 6)


def test_negative_tau_raises(_gs_init):
    """Defensive: tau_delay < 0 must surface immediately at construction."""
    with pytest.raises(ValueError, match="tau_delay"):
        _build_env(tau_delay=-1, num_envs=2)


def test_warmup_fill_after_reset(env_tau5):
    """After reset(), every slot = current spawn state (no zero garbage)."""
    env = env_tau5
    env.reset()
    hist = env.defender_state_history          # (6, B, 6)
    current = torch.cat([env.defender_pos, env.defender_vel], dim=1)  # (B, 6)
    # All slots match current spawn — broadcast comparison.
    for slot in range(hist.shape[0]):
        assert torch.allclose(hist[slot], current, atol=1e-6), (
            f"slot {slot} not filled to current at reset"
        )


def test_history_push_after_step(env_tau5):
    """Single step: [-1] holds fresh state, prior slots shift toward [0]."""
    env = env_tau5
    env.reset()
    snapshot = env.defender_state_history.clone()         # (6, B, 6) all = spawn
    spawn_slot = snapshot[0].clone()                       # (B, 6)

    action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    env.step(action)

    hist = env.defender_state_history
    new_state = torch.cat([env.defender_pos, env.defender_vel], dim=1)  # (B, 6)
    # After 1 step: newest at [-1] = current, [0] still = spawn (τ-1=4 steps to go).
    assert torch.allclose(hist[-1], new_state, atol=1e-6)
    assert torch.allclose(hist[0], spawn_slot, atol=1e-6), \
        "history[0] should still equal spawn after only 1 step (τ=5)"


def test_tau_0_actor_obs_uses_current_state(env_tau0):
    """τ=0 backward compat — actor obs s_d_kin matches current GT body-frame."""
    from envs.base_pe_env import rotate_to_body_frame
    env = env_tau0
    env.reset()
    # Build expected body-frame relative kinematics from CURRENT state.
    expected_rel_pos = rotate_to_body_frame(
        env.defender_pos - env.attacker_pos, env.attacker_quat,
    ) * env.pos_scale
    expected_rel_vel = rotate_to_body_frame(
        env.defender_vel - env.attacker_vel, env.attacker_quat,
    ) * env.vel_scale

    obs = env._attacker_obs()
    # Slot layout (env.py _attacker_obs): pos(3)+vel(3)+rot(9)+ang(3)=18, then
    # rel_pos_d(3)+rel_vel_d(3) at indices [18:24].
    sd_kin = obs[:, 18:24]
    assert torch.allclose(sd_kin[:, :3], expected_rel_pos, atol=1e-5)
    assert torch.allclose(sd_kin[:, 3:], expected_rel_vel, atol=1e-5)


def test_tau_5_actor_obs_uses_delayed_state(env_tau5):
    """τ=5, episode_length=1: actor obs s_d_kin uses spawn state (oldest).

    Computed body-frame rel uses the *delayed* defender pos but *current*
    attacker pose (per Phase 3a spec). Compare to obs slice [18:24].
    """
    from envs.base_pe_env import rotate_to_body_frame
    env = env_tau5
    env.reset()
    spawn_def_pos = env.defender_pos.clone()
    spawn_def_vel = env.defender_vel.clone()

    action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    env.step(action)

    # After 1 step, history[0] still = spawn defender state.
    delayed_pos = env.defender_state_history[0, :, :3]
    delayed_vel = env.defender_state_history[0, :, 3:]
    assert torch.allclose(delayed_pos, spawn_def_pos, atol=1e-6)
    assert torch.allclose(delayed_vel, spawn_def_vel, atol=1e-6)

    expected_rel_pos = rotate_to_body_frame(
        delayed_pos - env.attacker_pos, env.attacker_quat,
    ) * env.pos_scale
    expected_rel_vel = rotate_to_body_frame(
        delayed_vel - env.attacker_vel, env.attacker_quat,
    ) * env.vel_scale

    obs = env._attacker_obs()
    sd_kin = obs[:, 18:24]
    assert torch.allclose(sd_kin[:, :3], expected_rel_pos, atol=1e-5)
    assert torch.allclose(sd_kin[:, 3:], expected_rel_vel, atol=1e-5)


def test_multi_env_shape_consistency(_gs_init):
    """Buffer scales linearly along B; 16-env build keeps (τ+1, 16, 6)."""
    env = _build_env(tau_delay=3, num_envs=16)
    assert env.defender_state_history.shape == (4, 16, 6)
    env.reset()
    action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    env.step(action)
    # Buffer still well-formed (no NaN/Inf) after a real step.
    assert torch.isfinite(env.defender_state_history).all()


# ---------------------------------------------------------------------------
# Backward-compat regression guards (2026-05-13).
# Triggered by docs/notes/prompt_for_tau_delay_claude.md — going through
# the history buffer at tau=0 perturbed cuDNN op ordering and broke ams_v6
# seed reproducibility (a_win 60-72% → 3-5% on cycle 6-8). These tests pin
# that the tau=0 path *never touches* the buffer.
# ---------------------------------------------------------------------------

def test_tau0_push_history_is_noop(env_tau0):
    """tau=0 path: _push_defender_history must NOT mutate the buffer.

    The buffer is allocated to all zeros and stays zero — confirming no
    tensor ops happen at all when tau_delay=0. This is the regression guard
    for the ams_v6 trajectory breakage.
    """
    env = env_tau0
    env.reset()
    # After reset_idx warmup-fill (which itself is guarded by tau>0), the
    # buffer for tau=0 should remain at its initial zeros.
    assert torch.all(env.defender_state_history == 0), (
        "tau=0 buffer should remain untouched by reset_idx warmup fill"
    )
    action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    env.step(action)
    assert torch.all(env.defender_state_history == 0), (
        "tau=0 buffer should remain untouched by step()._push_defender_history"
    )


def test_tau0_actor_obs_bypasses_history(env_tau0):
    """tau=0 actor obs reads self.defender_pos / self.defender_vel directly,
    NOT defender_state_history[0]. Verified by zeroing the buffer and
    confirming obs is still meaningful (would be wrong if read from buffer)."""
    env = env_tau0
    env.reset()
    # Sanity check: buffer is zeros (per the previous test's invariant).
    assert torch.all(env.defender_state_history == 0)
    obs = env._attacker_obs()
    sd_kin = obs[:, 18:24]
    # If actor read from the zero buffer, rel_pos_d would equal
    # rotate_to_body_frame(-attacker_pos, attacker_quat) * pos_scale, NOT
    # the real defender-relative kinematics. Check it matches real state.
    from envs.base_pe_env import rotate_to_body_frame
    expected_rel_pos = rotate_to_body_frame(
        env.defender_pos - env.attacker_pos, env.attacker_quat,
    ) * env.pos_scale
    assert torch.allclose(sd_kin[:, :3], expected_rel_pos, atol=1e-5), (
        "actor obs at tau=0 must come from self.defender_pos, not buffer"
    )
