"""Phase E.2 — ForecasterTrainer 단위 테스트 (2026-05-08)."""

from __future__ import annotations

import math

import pytest
import torch

from agents.forecaster.model import ForecasterModel
from agents.forecaster.replay_buffer import ForecasterReplayBuffer
from agents.forecaster.trainer import ForecasterTrainer


def _make_trainer(
    capacity=200, K=10, H=5, s_a_dim=18, s_d_dim=18, na=4,
    d_model=64, lr=3e-4,
):
    model = ForecasterModel(
        s_a_dim=s_a_dim, s_d_dim=s_d_dim, num_actions=na,
        history_K=K, horizon_H=H, d_model=d_model, nhead=2, num_layers=2,
    )
    buffer = ForecasterReplayBuffer(
        capacity=capacity, history_K=K, horizon_H=H,
        s_a_dim=s_a_dim, s_d_dim=s_d_dim, num_actions=na,
        device="cpu", seed=0,
    )
    return ForecasterTrainer(model=model, buffer=buffer, lr=lr, device="cpu")


def _fake_rollout(T, B, s_a_dim=18, s_d_dim=18, na=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    s_a = torch.randn(T, B, s_a_dim, generator=g)
    s_d = torch.randn(T, B, s_d_dim, generator=g)
    action = torch.randn(T, B, na, generator=g).clamp(-1, 1)
    return s_a, s_d, action


def test_collect_too_short_rollout():
    """T < K+H+1 → 0 windows pushed."""
    tr = _make_trainer()
    s_a, s_d, a = _fake_rollout(T=10, B=4)   # K=10, H=5, need T >= 16
    n = tr.collect_from_rollout(s_a, s_d, a)
    assert n == 0
    assert len(tr.buffer) == 0


def test_collect_normal_rollout():
    """T = K+H+1 → 1 window per env. T larger → more windows."""
    tr = _make_trainer()
    K, H = 10, 5
    T, B = 30, 4
    s_a, s_d, a = _fake_rollout(T=T, B=B)
    n = tr.collect_from_rollout(s_a, s_d, a)
    expected = (T - K - H) * B   # n_windows × B = 15 × 4 = 60
    assert n == expected
    assert len(tr.buffer) == expected


def test_collect_invalid_shape():
    tr = _make_trainer()
    s_a = torch.randn(20, 4)   # 2D not 3D
    s_d = torch.randn(20, 4, 18)
    a = torch.randn(20, 4, 4)
    with pytest.raises(ValueError, match="3D"):
        tr.collect_from_rollout(s_a, s_d, a)


def test_train_step_underfilled_buffer():
    """buffer < batch_size → NaN return, no error."""
    tr = _make_trainer()
    loss = tr.train_step(batch_size=256)
    assert math.isnan(loss)
    assert tr.train_step_count == 0


def test_train_step_returns_loss():
    """Sufficient buffer → train_step returns float loss."""
    tr = _make_trainer()
    s_a, s_d, a = _fake_rollout(T=30, B=8)
    tr.collect_from_rollout(s_a, s_d, a)
    loss = tr.train_step(batch_size=32)
    assert isinstance(loss, float)
    assert loss > 0
    assert tr.train_step_count == 1


def test_train_step_loss_decreases():
    """100 train steps on small fixed buffer → loss should decrease."""
    tr = _make_trainer(lr=1e-3)
    s_a, s_d, a = _fake_rollout(T=30, B=8)
    tr.collect_from_rollout(s_a, s_d, a)
    initial_loss = tr.train_step(batch_size=32)
    for _ in range(99):
        tr.train_step(batch_size=32)
    final_loss = tr.last_loss
    # 100 steps overfit small buffer → loss should drop notably.
    assert final_loss < initial_loss * 0.7, \
        f"loss did not decrease enough: {initial_loss} → {final_loss}"


def test_eval_mse_no_grad():
    """eval_mse doesn't update weights."""
    tr = _make_trainer()
    s_a, s_d, a = _fake_rollout(T=30, B=8)
    tr.collect_from_rollout(s_a, s_d, a)
    # Snapshot weights
    pre = [p.clone() for p in tr.model.parameters()]
    _ = tr.eval_mse(batch_size=32)
    post = list(tr.model.parameters())
    for a_w, b_w in zip(pre, post):
        assert torch.allclose(a_w, b_w), "eval_mse should not update weights"


def test_predict_shape():
    """predict() returns (B, H, s_d_dim)."""
    tr = _make_trainer()
    K, H = 10, 5
    B = 4
    s_a = torch.randn(B, K + 1, 18)
    s_d = torch.randn(B, K + 1, 18)
    a = torch.randn(B, K, 4)
    pred = tr.predict(s_a, s_d, a)
    assert pred.shape == (B, H, 18)


def test_predict_no_grad_required_input():
    """predict accepts non-grad-tracked inputs."""
    tr = _make_trainer()
    s_a = torch.randn(2, 10, 18)
    s_d = torch.randn(2, 10, 18)
    a = torch.randn(2, 10, 4)
    pred = tr.predict(s_a, s_d, a)
    assert pred.requires_grad is False  # @torch.no_grad


def test_state_summary():
    tr = _make_trainer()
    summ = tr.state_summary()
    assert summ["buffer_size"] == 0
    assert summ["train_steps"] == 0
    assert summ["last_loss"] is None

    s_a, s_d, a = _fake_rollout(T=30, B=8)
    tr.collect_from_rollout(s_a, s_d, a)
    tr.train_step(batch_size=32)
    summ = tr.state_summary()
    assert summ["buffer_size"] > 0
    assert summ["train_steps"] == 1
    assert summ["last_loss"] is not None
    assert isinstance(summ["last_loss"], float)
