"""Phase E.2 — Forecaster ReplayBuffer 단위 테스트 (2026-05-08)."""

from __future__ import annotations

import pytest
import torch

from agents.forecaster.replay_buffer import ForecasterBatch, ForecasterReplayBuffer


def _make_buf(capacity=100, K=10, H=5, s_a_dim=18, s_d_dim=18, num_actions=4):
    return ForecasterReplayBuffer(
        capacity=capacity, history_K=K, horizon_H=H,
        s_a_dim=s_a_dim, s_d_dim=s_d_dim, num_actions=num_actions,
        device="cpu", seed=0,
    )


def _fake_batch(B, K, H, s_a, s_d, na):
    return (
        torch.randn(B, K + 1, s_a),
        torch.randn(B, K + 1, s_d),
        torch.randn(B, K, na),
        torch.randn(B, H, s_d),
    )


def test_init_empty():
    buf = _make_buf()
    assert len(buf) == 0
    assert not buf.is_full()


def test_invalid_init():
    with pytest.raises(ValueError, match="capacity"):
        ForecasterReplayBuffer(0, 5, 3, 1, 1, 1, "cpu")
    with pytest.raises(ValueError, match="history_K"):
        ForecasterReplayBuffer(10, 0, 3, 1, 1, 1, "cpu")
    with pytest.raises(ValueError, match="horizon_H"):
        ForecasterReplayBuffer(10, 5, 0, 1, 1, 1, "cpu")


def test_sample_empty_raises():
    buf = _make_buf()
    with pytest.raises(RuntimeError, match="empty"):
        buf.sample(4)


def test_add_increases_size():
    buf = _make_buf(capacity=100, K=10, H=5, s_a_dim=18, s_d_dim=18, num_actions=4)
    s_a, s_d, a, s_d_f = _fake_batch(20, 10, 5, 18, 18, 4)
    buf.add(s_a, s_d, a, s_d_f)
    assert len(buf) == 20
    assert buf.ptr == 20


def test_add_fills_to_capacity():
    buf = _make_buf(capacity=50)
    s_a, s_d, a, s_d_f = _fake_batch(50, 10, 5, 18, 18, 4)
    buf.add(s_a, s_d, a, s_d_f)
    assert len(buf) == 50
    assert buf.is_full()


def test_add_wraps_fifo():
    """Add 2× capacity → wrap-around, size=capacity, ptr cycles."""
    buf = _make_buf(capacity=50)
    s_a, s_d, a, s_d_f = _fake_batch(40, 10, 5, 18, 18, 4)
    buf.add(s_a, s_d, a, s_d_f)
    assert len(buf) == 40
    s_a2, s_d2, a2, s_d_f2 = _fake_batch(30, 10, 5, 18, 18, 4)
    buf.add(s_a2, s_d2, a2, s_d_f2)   # 40+30=70, wrap
    assert len(buf) == 50   # capped
    assert buf.is_full()
    # ptr = (40 + 30) % 50 = 20
    assert buf.ptr == 20


def test_sample_shapes():
    buf = _make_buf(capacity=100, K=10, H=5, s_a_dim=18, s_d_dim=18, num_actions=4)
    s_a, s_d, a, s_d_f = _fake_batch(50, 10, 5, 18, 18, 4)
    buf.add(s_a, s_d, a, s_d_f)
    batch = buf.sample(8)
    assert isinstance(batch, ForecasterBatch)
    assert batch.s_a_history.shape == (8, 11, 18)
    assert batch.s_d_history.shape == (8, 11, 18)
    assert batch.action_history.shape == (8, 10, 4)
    assert batch.s_d_future.shape == (8, 5, 18)


def test_sample_values_in_buffer():
    """Sample returned values are from the buffer (not random new tensors)."""
    buf = _make_buf(capacity=10, K=2, H=2, s_a_dim=3, s_d_dim=3, num_actions=2)
    s_a = torch.arange(10 * 3 * 3, dtype=torch.float32).reshape(10, 3, 3)
    s_d = torch.arange(10 * 3 * 3, dtype=torch.float32).reshape(10, 3, 3) + 1000
    a = torch.arange(10 * 2 * 2, dtype=torch.float32).reshape(10, 2, 2)
    s_d_f = torch.arange(10 * 2 * 3, dtype=torch.float32).reshape(10, 2, 3) + 2000
    buf.add(s_a, s_d, a, s_d_f)
    batch = buf.sample(5)
    # 모든 sample 값이 원본 buffer (즉 0..2999)에서 옴을 검증
    assert (batch.s_a_history >= 0).all() and (batch.s_a_history < 90).all()
    assert (batch.s_d_history >= 1000).all() and (batch.s_d_history < 1090).all()
    assert (batch.s_d_future >= 2000).all() and (batch.s_d_future < 2060).all()


def test_invalid_shape_raises():
    buf = _make_buf(capacity=10, K=2, H=2, s_a_dim=3, s_d_dim=3, num_actions=2)
    with pytest.raises(AssertionError):
        buf.add(
            torch.zeros(5, 99, 3),    # wrong K dim
            torch.zeros(5, 3, 3),
            torch.zeros(5, 2, 2),
            torch.zeros(5, 2, 3),
        )


def test_sample_invalid_batch_size():
    buf = _make_buf()
    s_a, s_d, a, s_d_f = _fake_batch(5, 10, 5, 18, 18, 4)
    buf.add(s_a, s_d, a, s_d_f)
    with pytest.raises(ValueError):
        buf.sample(0)
    with pytest.raises(ValueError):
        buf.sample(-1)
