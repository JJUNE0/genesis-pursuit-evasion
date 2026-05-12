"""Body-rate PD inner-loop controller — sanity tests.

Validates the action-scaling, mixer, and PD math without requiring a full
Genesis scene build (the controller is pure-tensor, batched on num_envs).

Three cases (per task spec):
    1. action = [0,0,0,0]      → all motors at hover RPM (no torque).
    2. action = [+1, 0, 0, 0]  → all motors equal, RPM > hover (T = 2·m·g).
    3. action = [0, +1, 0, 0]  → +roll torque → +y motors (left) > -y motors.

Genesis only needs ``gs.init`` to expose ``gs.tc_float`` / ``gs.device`` —
the session-scoped autouse ``_gs_init`` in ``tests/conftest.py`` covers that.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import genesis as gs

from utils.drone_params import DroneParams
from utils.rate_controller import BodyRateController, build_mixer_matrix


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def nova_params():
    return DroneParams(str(_project_root() / "configs" / "drones" / "nova.yaml"))


@pytest.fixture(scope="module")
def rate_ctrl(nova_params):
    return BodyRateController(
        num_envs=4, dt=0.005, params=nova_params, device=gs.device,
    )


def _hover_rpm(params: DroneParams) -> float:
    return math.sqrt(
        params.mass * params.gravity / (params.n_propellers * params.kf)
    )


def test_zero_action_yields_hover_rpm(rate_ctrl, nova_params):
    """case 1: a = [0,0,0,0] → 모든 모터 hover RPM."""
    rate_ctrl.reset()
    B = rate_ctrl.num_envs
    action = torch.zeros((B, 4), device=rate_ctrl.device, dtype=gs.tc_float)
    ang_vel = torch.zeros((B, 3), device=rate_ctrl.device, dtype=gs.tc_float)

    rpms = rate_ctrl.step(action, ang_vel)

    # All motors equal across envs and props.
    assert rpms.shape == (B, nova_params.n_propellers)
    assert torch.allclose(rpms, rpms[:1, :1].expand_as(rpms))

    # Equal to hover RPM (sqrt(m*g / (n*kf))).
    expected = _hover_rpm(nova_params)
    assert torch.allclose(
        rpms, torch.full_like(rpms, expected), rtol=1e-4, atol=1e-2,
    ), f"expected {expected:.2f} RPM, got {float(rpms[0, 0]):.2f}"


def test_max_thrust_action_yields_uniform_higher_rpm(rate_ctrl, nova_params):
    """case 2: a = [+1, 0, 0, 0] → 모든 모터 equal, RPM = √2 · hover_rpm.

    T_norm = +1 → T_total = 2·m·g, torque = 0 → 각 모터 thrust = m·g/2 =
    2 · hover_thrust_per_motor → RPM = √2 · hover_rpm.
    """
    rate_ctrl.reset()
    B = rate_ctrl.num_envs
    action = torch.zeros((B, 4), device=rate_ctrl.device, dtype=gs.tc_float)
    action[:, 0] = 1.0
    ang_vel = torch.zeros((B, 3), device=rate_ctrl.device, dtype=gs.tc_float)

    rpms = rate_ctrl.step(action, ang_vel)

    # All motors equal (no torque).
    assert torch.allclose(rpms, rpms[:1, :1].expand_as(rpms))

    hover_rpm = _hover_rpm(nova_params)
    expected = math.sqrt(2.0) * hover_rpm
    # Cap by physical max (Nova nominal max RPM > √2·hover for default params).
    expected = min(expected, rate_ctrl.max_rpm)

    assert (rpms > hover_rpm).all(), \
        f"rpms ({float(rpms[0, 0]):.1f}) should exceed hover ({hover_rpm:.1f})"
    assert torch.allclose(
        rpms, torch.full_like(rpms, expected), rtol=1e-4, atol=1e-2,
    )


def test_positive_roll_creates_asymmetric_thrust(rate_ctrl, nova_params):
    """case 3: a = [0, +1, 0, 0] → +roll torque → +y(좌) 모터가 -y(우) 모터보다 크다."""
    rate_ctrl.reset()
    B = rate_ctrl.num_envs
    action = torch.zeros((B, 4), device=rate_ctrl.device, dtype=gs.tc_float)
    action[:, 1] = 1.0   # wx_ref = +max_body_rate (= +π rad/s)
    ang_vel = torch.zeros((B, 3), device=rate_ctrl.device, dtype=gs.tc_float)

    rpms = rate_ctrl.step(action, ang_vel)

    # Re-derive per-motor thrust: f = kf · ω².
    thrusts = rate_ctrl.kf * (rpms ** 2)        # shape: (B, n_props)

    # Identify left (+y) vs right (-y) motors from URDF/yaml positions.
    pos_y = torch.tensor(
        nova_params.prop_positions[:, 1],
        device=rate_ctrl.device, dtype=gs.tc_float,
    )
    left_mask = pos_y > 0
    right_mask = pos_y < 0
    assert left_mask.sum() > 0 and right_mask.sum() > 0, \
        "nova.yaml prop_positions degenerate"

    f_left_mean = thrusts[:, left_mask].mean(dim=1)
    f_right_mean = thrusts[:, right_mask].mean(dim=1)

    assert (f_left_mean > f_right_mean).all(), (
        f"+roll torque should lift +y side: f_left={float(f_left_mean[0]):.4f}, "
        f"f_right={float(f_right_mean[0]):.4f}"
    )
    # Asymmetric: NOT all-equal.
    assert not torch.allclose(rpms, rpms[:1, :1].expand_as(rpms))


def test_mixer_round_trip_identity(nova_params):
    """forward · inverse = I → 우리 inverse가 self-consistent."""
    import numpy as np
    M = build_mixer_matrix(nova_params)         # (4, n)
    if M.shape[0] == M.shape[1]:
        M_inv = np.linalg.inv(M)
    else:
        M_inv = np.linalg.pinv(M)
    err = np.max(np.abs(M @ M_inv - np.eye(M.shape[0])))
    assert err < 1e-9, f"mixer round-trip error {err:.3e}"


def test_reset_zeroes_last_ang_vel(rate_ctrl):
    """reset(envs_idx) 가 해당 슬롯의 last_ang_vel만 0으로 만든다."""
    rate_ctrl.reset()
    B = rate_ctrl.num_envs
    # Spike last_ang_vel via a step with non-zero ang_vel.
    action = torch.zeros((B, 4), device=rate_ctrl.device, dtype=gs.tc_float)
    ang_vel = torch.full((B, 3), 0.5, device=rate_ctrl.device, dtype=gs.tc_float)
    rate_ctrl.step(action, ang_vel)
    assert (rate_ctrl.last_ang_vel != 0).any()

    # Partial reset
    idx = torch.tensor([0, 2], device=rate_ctrl.device, dtype=torch.long)
    rate_ctrl.reset(idx)
    assert (rate_ctrl.last_ang_vel[idx] == 0).all()
    keep = torch.tensor([1, 3], device=rate_ctrl.device, dtype=torch.long)
    assert (rate_ctrl.last_ang_vel[keep] != 0).any()

    # Full reset
    rate_ctrl.reset()
    assert (rate_ctrl.last_ang_vel == 0).all()


def test_negative_thrust_clamped_to_zero(rate_ctrl, nova_params):
    """T_norm = -1 → T_total = 0 → 모든 모터 RPM = 0 (torque = 0 가정)."""
    rate_ctrl.reset()
    B = rate_ctrl.num_envs
    action = torch.zeros((B, 4), device=rate_ctrl.device, dtype=gs.tc_float)
    action[:, 0] = -1.0
    ang_vel = torch.zeros((B, 3), device=rate_ctrl.device, dtype=gs.tc_float)
    rpms = rate_ctrl.step(action, ang_vel)
    assert torch.allclose(rpms, torch.zeros_like(rpms))
