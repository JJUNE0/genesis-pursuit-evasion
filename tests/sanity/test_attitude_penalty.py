"""2026-05-03 — Quadratic attitude penalty unit tests.

Diagnosed root cause for 100% def_crash via roll_bad in phase_m wide-spawn ams:
chase reward only encodes distance, not attitude safety margin → defender
banks into roll>60° crash threshold. Soft penalty in [pad, threshold] band
so policy self-regulates.

Properties tested (per outline):
  1. inside band (|angle| <= pad)             → r_attitude == 0
  2. at threshold (|roll| == 60°, pad=50)     → -coef · (10°)² = -coef · 100
  3. mixed (|roll|=55°, |pitch|=30°)          → uses max → -coef · 25
  4. attacker / defender symmetric (same quat → same penalty)
  5. attitude_coef == 0 → backward compat (penalty zero everywhere)
"""

from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float = 0.0,
                        device: str = "cuda") -> torch.Tensor:
    """Inverse of envs.base_pe_env.quat_to_euler_deg.

    Returns wxyz quat (Genesis convention) for ZYX intrinsic Tait-Bryan
    (yaw, pitch, roll) — the same order ``quat_to_euler_deg`` decodes.
    """
    r = math.radians(roll_deg) * 0.5
    p = math.radians(pitch_deg) * 0.5
    y = math.radians(yaw_deg) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return torch.tensor([[qw, qx, qy, qz]], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Tests — using attacker fixture (env._attitude_penalty is a method on it)
# ---------------------------------------------------------------------------

def _set_attitude_params(env, pad_deg: float, coef: float) -> None:
    env.attitude_pad_deg = pad_deg
    env.attitude_coef = coef


def test_inside_band_zero(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    _set_attitude_params(env, pad_deg=50.0, coef=5.0e-4)
    # Roll 30°, pitch 20° → both inside [-50°, 50°] band.
    quat = _quat_from_euler_deg(30.0, 20.0, device=env.device)
    r = env._attitude_penalty(quat)
    assert r.shape == (1,)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-7)


def test_at_threshold_quadratic_value(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    pad, coef = 50.0, 5.0e-4
    _set_attitude_params(env, pad_deg=pad, coef=coef)
    # roll = 60° → over = 10° → -coef · 100
    quat = _quat_from_euler_deg(60.0, 0.0, device=env.device)
    r = env._attitude_penalty(quat)
    expected = -coef * (60.0 - pad) ** 2
    assert torch.allclose(r, torch.full_like(r, expected), atol=1e-4)


def test_max_of_roll_pitch(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    pad, coef = 50.0, 5.0e-4
    _set_attitude_params(env, pad_deg=pad, coef=coef)
    # |roll|=55° (over=5), |pitch|=30° (inside) → max = 55° → over = 5°
    quat = _quat_from_euler_deg(55.0, 30.0, device=env.device)
    r = env._attitude_penalty(quat)
    expected = -coef * (55.0 - pad) ** 2
    assert torch.allclose(r, torch.full_like(r, expected), atol=1e-4)


def test_negative_angle_uses_abs(pe_env_attacker_stationary):
    env = pe_env_attacker_stationary
    pad, coef = 50.0, 5.0e-4
    _set_attitude_params(env, pad_deg=pad, coef=coef)
    # |roll|=55° via negative roll = -55° → same penalty as +55°
    q_pos = _quat_from_euler_deg(+55.0, 0.0, device=env.device)
    q_neg = _quat_from_euler_deg(-55.0, 0.0, device=env.device)
    r_pos = env._attitude_penalty(q_pos)
    r_neg = env._attitude_penalty(q_neg)
    assert torch.allclose(r_pos, r_neg, atol=1e-4)


def test_attacker_defender_symmetric(pe_env_attacker_stationary):
    """Same quat → same penalty regardless of which side computes it.

    `_attitude_penalty` is a pure function of (quat, pad, coef); test that
    feeding identical quats yields identical scalars.
    """
    env = pe_env_attacker_stationary
    _set_attitude_params(env, pad_deg=50.0, coef=5.0e-4)
    quat = _quat_from_euler_deg(58.0, 12.0, device=env.device)
    r1 = env._attitude_penalty(quat)
    r2 = env._attitude_penalty(quat)
    assert torch.allclose(r1, r2, atol=1e-7)


def test_coef_zero_backward_compat(pe_env_attacker_stationary):
    """coef=0 → penalty is zero for any quat. Required for legacy yaml that
    do not declare attitude_pad_deg / attitude_coef.
    """
    env = pe_env_attacker_stationary
    _set_attitude_params(env, pad_deg=50.0, coef=0.0)
    for roll, pitch in [(0.0, 0.0), (60.0, 0.0), (55.0, 55.0), (89.0, 0.0)]:
        quat = _quat_from_euler_deg(roll, pitch, device=env.device)
        r = env._attitude_penalty(quat)
        assert torch.allclose(r, torch.zeros_like(r), atol=1e-7), (
            f"coef=0 must yield 0 for all quats; got {r} at roll={roll}, pitch={pitch}"
        )


def test_quadratic_growth(pe_env_attacker_stationary):
    """r(60°) / r(55°) ≈ (10°)² / (5°)² = 4. Confirms quadratic shape, not linear."""
    env = pe_env_attacker_stationary
    _set_attitude_params(env, pad_deg=50.0, coef=5.0e-4)
    r55 = env._attitude_penalty(_quat_from_euler_deg(55.0, 0.0, device=env.device))
    r60 = env._attitude_penalty(_quat_from_euler_deg(60.0, 0.0, device=env.device))
    ratio = (r60.item() / r55.item()) if r55.item() != 0 else float("inf")
    assert abs(ratio - 4.0) < 0.05, f"quadratic ratio expected ~4.0, got {ratio:.3f}"
