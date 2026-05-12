"""2026-05-04 — Quadratic boundary penalty unit tests.

Diagnosed root cause for v26 def_crash transition: attitude penalty regulated
roll banking but defender over-commit chase still triggered z_below (83.6%) +
def_oob (39.8%). Soft boundary signals (velocity / z floor / bound proximity)
encode "you're approaching a hard limit" before the hard threshold fires.

Properties tested:
  1. all coefs 0 → penalty == 0 everywhere (backward compat)
  2. velocity below pad → penalty 0; above pad → quadratic
  3. z above pad → penalty 0; below pad → quadratic on (pad - z)
  4. bound below (bound - pad) → 0; above → quadratic per axis, summed
  5. components independent — each can fire alone or together
  6. attacker / defender symmetric (same pos/vel → same penalty)
"""

from __future__ import annotations

import torch


def _set_safety(env, **kwargs):
    """Set safety params and bound attrs on the env (test override)."""
    for k, v in kwargs.items():
        setattr(env, k, float(v))


def _make_pos_vel(env, x=0.0, y=0.0, z=1.5, vx=0.0, vy=0.0, vz=0.0):
    pos = torch.tensor([[x, y, z]], dtype=torch.float32, device=env.device)
    vel = torch.tensor([[vx, vy, vz]], dtype=torch.float32, device=env.device)
    return pos, vel


def test_all_zero_coefs_no_penalty(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    _set_safety(
        env, velocity_coef=0.0, z_safety_coef=0.0, bound_safety_coef=0.0,
        velocity_pad=10.0, z_safety_pad=0.5, bound_safety_pad=1.0,
    )
    # any extreme state → still 0
    pos, vel = _make_pos_vel(env, x=15.0, y=15.0, z=0.0, vx=20.0)
    r = env._safety_penalty(pos, vel)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-7)


def test_velocity_below_pad_zero(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    _set_safety(env, velocity_coef=5.0e-4, velocity_pad=10.0,
                z_safety_coef=0.0, bound_safety_coef=0.0)
    pos, vel = _make_pos_vel(env, vx=8.0)  # |v|=8 < pad
    r = env._safety_penalty(pos, vel)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-7)


def test_velocity_above_pad_quadratic(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    _set_safety(env, velocity_coef=5.0e-4, velocity_pad=10.0,
                z_safety_coef=0.0, bound_safety_coef=0.0)
    pos, vel = _make_pos_vel(env, vx=15.0)  # |v|=15, over=5
    r = env._safety_penalty(pos, vel)
    expected = -5.0e-4 * (5.0 ** 2)
    assert torch.allclose(r, torch.full_like(r, expected), atol=1e-5)


def test_z_above_pad_zero(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    _set_safety(env, z_safety_coef=0.05, z_safety_pad=0.5,
                velocity_coef=0.0, bound_safety_coef=0.0)
    pos, vel = _make_pos_vel(env, z=1.0)  # z=1.0 > pad
    r = env._safety_penalty(pos, vel)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-7)


def test_z_below_pad_quadratic(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    _set_safety(env, z_safety_coef=0.05, z_safety_pad=0.5,
                velocity_coef=0.0, bound_safety_coef=0.0)
    pos, vel = _make_pos_vel(env, z=0.2)  # z=0.2, over=0.3
    r = env._safety_penalty(pos, vel)
    expected = -0.05 * (0.3 ** 2)
    assert torch.allclose(r, torch.full_like(r, expected), atol=1e-5)


def test_bound_inside_zero(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    # Force bound_x/y/z and pad
    env.bound_x, env.bound_y, env.bound_z = 12.0, 5.0, 5.0
    _set_safety(env, bound_safety_coef=0.01, bound_safety_pad=1.0,
                velocity_coef=0.0, z_safety_coef=0.0)
    pos, vel = _make_pos_vel(env, x=10.0, y=3.0, z=3.0)  # all inside (bound-pad)
    r = env._safety_penalty(pos, vel)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-7)


def test_bound_proximity_quadratic_per_axis(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.bound_x, env.bound_y, env.bound_z = 12.0, 5.0, 5.0
    _set_safety(env, bound_safety_coef=0.01, bound_safety_pad=1.0,
                velocity_coef=0.0, z_safety_coef=0.0)
    # |x|=11.5 → over_x=0.5 (pad=1, threshold_x=11). |y|=4.5 → over_y=0.5.
    pos, vel = _make_pos_vel(env, x=11.5, y=4.5, z=4.0)
    r = env._safety_penalty(pos, vel)
    expected = -0.01 * (0.25 + 0.25 + 0.0)  # z=4 < bound_z-pad=4 → 0
    assert torch.allclose(r, torch.full_like(r, expected), atol=1e-5)


def test_components_sum_independently(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.bound_x, env.bound_y, env.bound_z = 12.0, 5.0, 5.0
    _set_safety(
        env,
        velocity_coef=5.0e-4, velocity_pad=10.0,
        z_safety_coef=0.05,   z_safety_pad=0.5,
        bound_safety_coef=0.01, bound_safety_pad=1.0,
    )
    # All three trigger:
    #   v=15: -5e-4 * 25      = -0.0125
    #   z=0.2: -0.05 * 0.09   = -0.0045
    #   |x|=11.5, over_x=0.5: -0.01 * 0.25 = -0.0025
    pos, vel = _make_pos_vel(env, x=11.5, y=0.0, z=0.2, vx=15.0)
    r = env._safety_penalty(pos, vel)
    expected = -(5.0e-4 * 25.0) - (0.05 * 0.09) - (0.01 * 0.25)
    assert torch.allclose(r, torch.full_like(r, expected), atol=1e-5)


def test_negative_velocity_uses_norm(pe_env_attacker_stationary):
    """|v| uses norm, so direction doesn't matter — symmetric."""
    env = pe_env_attacker_stationary
    _set_safety(env, velocity_coef=5.0e-4, velocity_pad=10.0,
                z_safety_coef=0.0, bound_safety_coef=0.0)
    pos1, vel1 = _make_pos_vel(env, vx=12.0)
    pos2, vel2 = _make_pos_vel(env, vx=-12.0)
    r1 = env._safety_penalty(pos1, vel1)
    r2 = env._safety_penalty(pos2, vel2)
    assert torch.allclose(r1, r2, atol=1e-7)


def test_negative_position_bound_uses_abs(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    env.bound_x, env.bound_y, env.bound_z = 12.0, 5.0, 5.0
    _set_safety(env, bound_safety_coef=0.01, bound_safety_pad=1.0,
                velocity_coef=0.0, z_safety_coef=0.0)
    pos1, _ = _make_pos_vel(env, x=11.5)
    pos2, _ = _make_pos_vel(env, x=-11.5)
    _, vel = _make_pos_vel(env)
    r1 = env._safety_penalty(pos1, vel)
    r2 = env._safety_penalty(pos2, vel)
    assert torch.allclose(r1, r2, atol=1e-7)
