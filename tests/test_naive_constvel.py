"""Tests for ``agents/forecaster/baselines/naive_constvel`` (Phase 3f, 2026-05-15).

Math correctness:
  - constant-velocity extrapolation result == manual computation
  - scale unscale + rescale round-trip preserves identity at τ=0... well, τ≥1 only
  - vel slot unchanged
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.forecaster.baselines.naive_constvel import (   # noqa: E402
    NaiveConstVelForecaster,
    naive_constvel_predict,
)


def test_naive_predict_shape():
    """Returns (B, 6) for (B, 1, 6) input."""
    B = 5
    defender_last = torch.randn(B, 1, 6)
    out = naive_constvel_predict(
        defender_last=defender_last, tau=10, dt=0.005,
        pos_scale=0.2, vel_scale=0.2,
    )
    assert out.shape == (B, 6)
    assert torch.isfinite(out).all()


def test_naive_predict_math():
    """Manual computation of constant-velocity extrapolation."""
    pos_scale = 0.2; vel_scale = 0.2; dt = 0.005; tau = 25

    # Hand-craft: pos_m=[1,2,3], vel_mps=[0.5, -1.0, 2.0]
    pos_m = torch.tensor([[1.0, 2.0, 3.0]])
    vel_mps = torch.tensor([[0.5, -1.0, 2.0]])
    defender_last = torch.cat([pos_m * pos_scale, vel_mps * vel_scale], dim=-1).unsqueeze(1)
    # shape: (1, 1, 6)

    out = naive_constvel_predict(
        defender_last=defender_last, tau=tau, dt=dt,
        pos_scale=pos_scale, vel_scale=vel_scale,
    )    # (1, 6) scaled output
    # Unscale to physical units
    pred_pos_m = out[0, 0:3] / pos_scale
    pred_vel_mps = out[0, 3:6] / vel_scale

    # Expected: pos + tau*dt*vel; vel unchanged
    expected_pos = pos_m[0] + (tau * dt) * vel_mps[0]
    expected_vel = vel_mps[0]
    assert torch.allclose(pred_pos_m, expected_pos, atol=1e-6), (
        f"pos mismatch: {pred_pos_m} vs {expected_pos}"
    )
    assert torch.allclose(pred_vel_mps, expected_vel, atol=1e-6), (
        f"vel mismatch (should be unchanged): {pred_vel_mps} vs {expected_vel}"
    )


def test_naive_zero_vel_returns_pos():
    """If vel=0, predicted pos should == input pos exactly."""
    pos_scale = 0.2; vel_scale = 0.2
    pos_m = torch.tensor([[5.0, -3.0, 1.5]])
    vel_mps = torch.zeros((1, 3))
    defender_last = torch.cat([pos_m * pos_scale, vel_mps * vel_scale], dim=-1).unsqueeze(1)

    out = naive_constvel_predict(
        defender_last=defender_last, tau=50, dt=0.005,
        pos_scale=pos_scale, vel_scale=vel_scale,
    )
    pred_pos_m = out[0, 0:3] / pos_scale
    assert torch.allclose(pred_pos_m, pos_m[0], atol=1e-6)


def test_class_wrapper_matches_function():
    """NaiveConstVelForecaster forward matches functional API."""
    fc = NaiveConstVelForecaster(dt=0.005, pos_scale=0.2, vel_scale=0.2)
    B = 4
    defender_last = torch.randn(B, 1, 6)
    out_class = fc({"defender_last": defender_last}, tau=25)
    out_fn = naive_constvel_predict(defender_last, tau=25, dt=0.005,
                                     pos_scale=0.2, vel_scale=0.2)
    assert torch.allclose(out_class, out_fn, atol=1e-9)
    assert fc.num_parameters() == 0


def test_invalid_inputs():
    """tau<1 and wrong shape raise ValueError."""
    import pytest
    defender_last = torch.randn(2, 1, 6)
    with pytest.raises(ValueError):
        naive_constvel_predict(defender_last, tau=0, dt=0.005,
                                pos_scale=0.2, vel_scale=0.2)
    with pytest.raises(ValueError):
        naive_constvel_predict(torch.randn(2, 6), tau=10, dt=0.005,
                                pos_scale=0.2, vel_scale=0.2)
