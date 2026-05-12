"""Phase E.2 — ForecasterModel Transformer 단위 테스트 (2026-05-08)."""

from __future__ import annotations

import pytest
import torch

from agents.forecaster.model import ForecasterModel


def _make_model(K=10, H=5, s_a=18, s_d=18, na=4, d_model=64):
    return ForecasterModel(
        s_a_dim=s_a, s_d_dim=s_d, num_actions=na,
        history_K=K, horizon_H=H, d_model=d_model, nhead=2, num_layers=2,
    )


def test_forward_shape_K_input():
    """Input length K → (B, H, s_d_dim) output."""
    m = _make_model()
    B = 8
    s_a = torch.randn(B, 10, 18)
    s_d = torch.randn(B, 10, 18)
    a = torch.randn(B, 10, 4)
    out = m(s_a, s_d, a)
    assert out.shape == (B, 5, 18)


def test_forward_shape_K_plus_1_input():
    """Buffer는 K+1 길이로 저장 — 모델이 last K 사용."""
    m = _make_model()
    B = 8
    s_a = torch.randn(B, 11, 18)   # K+1
    s_d = torch.randn(B, 11, 18)
    a = torch.randn(B, 10, 4)      # K
    out = m(s_a, s_d, a)
    assert out.shape == (B, 5, 18)


def test_invalid_history_length_raises():
    m = _make_model()
    B = 4
    s_a_bad = torch.randn(B, 7, 18)   # neither K nor K+1
    s_d_bad = torch.randn(B, 7, 18)
    a = torch.randn(B, 10, 4)
    with pytest.raises(ValueError, match="seq_len"):
        m(s_a_bad, s_d_bad, a)


def test_num_parameters_small():
    """사용자 권장: 작은 모델 (2 layer, 64 hidden) ~50K params 이내."""
    m = _make_model(d_model=64, K=10, H=5)
    n = m.num_parameters()
    assert n < 200_000, f"forecaster too large: {n} params"
    assert n > 10_000, f"forecaster suspiciously small: {n} params"


def test_gradient_flow():
    """forward → loss → backward 정상 흐름."""
    m = _make_model()
    B = 4
    s_a = torch.randn(B, 10, 18, requires_grad=False)
    s_d = torch.randn(B, 10, 18, requires_grad=False)
    a = torch.randn(B, 10, 4, requires_grad=False)
    target = torch.randn(B, 5, 18)
    pred = m(s_a, s_d, a)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    # 모든 trainable param의 grad가 존재 + 일부는 non-zero
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any((g != 0).any() for g in grads)


def test_output_initially_small():
    """xavier(gain=0.1) + zero bias → 초기 prediction 크기 작음 (~unit norm 이하)."""
    m = _make_model()
    m.eval()
    B = 4
    s_a = torch.randn(B, 10, 18)
    s_d = torch.randn(B, 10, 18)
    a = torch.randn(B, 10, 4)
    with torch.no_grad():
        out = m(s_a, s_d, a)
    # Norm per sample, per step — should be O(1) or smaller initially
    norms = out.norm(dim=-1)   # (B, H)
    assert norms.mean() < 5.0, f"forecaster output too large at init: mean={norms.mean()}"


def test_eval_mode_deterministic():
    """eval mode + 동일 input → 동일 output (dropout off)."""
    m = _make_model()
    m.eval()
    B = 2
    s_a = torch.randn(B, 10, 18)
    s_d = torch.randn(B, 10, 18)
    a = torch.randn(B, 10, 4)
    with torch.no_grad():
        out1 = m(s_a, s_d, a)
        out2 = m(s_a, s_d, a)
    assert torch.allclose(out1, out2)
