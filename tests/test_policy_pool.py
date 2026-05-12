"""Phase D — Fictitious Play 다중 ckpt pool 단위 테스트.

Plan: docs/plans/phase_c_curriculum_scratch.md (Phase D 섹션, 2026-05-07).

What this test pins:
  1. PretrainedDefenderPool / PretrainedAttackerPool 클래스 import / 생성.
  2. K=1 ckpt → drop-in compat (단일 ckpt PretrainedDefender와 동일 step interface).
  3. K=2+ ckpt 분산 — env_to_ckpt mapping이 K 범위 안에 있고 num_envs 일치.
  4. reset(envs_idx) 후 last_action 0 + env_to_ckpt 재할당.
  5. step output shape (num_envs, num_actions) + value range [-1, 1].

Real ckpts 필요 — logs/pe1v1_v0/curriculum_v9/stage_4/attacker_v10.pt 등 v16 결과.
ckpt 없으면 ``pytest.skip`` (CI에서는 ckpt 없을 수 있음).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import genesis as gs

ROOT = Path(__file__).resolve().parents[1]


def _find_attacker_ckpts() -> list[Path]:
    """Look for v16 stage 4 attacker ckpts (Phase D smoke material)."""
    base = ROOT / "logs" / "pe1v1_v0" / "curriculum_v9" / "stage_4"
    if not base.is_dir():
        return []
    return sorted(base.glob("attacker_v*.pt"))


def _find_defender_ckpts() -> list[Path]:
    base = ROOT / "logs" / "pe1v1_v0" / "curriculum_v9" / "stage_4"
    if not base.is_dir():
        return []
    return sorted(base.glob("defender_v*.pt"))


def test_pool_classes_importable():
    """Pool 클래스가 import 가능 (구문/이름 회귀 가드)."""
    from envs.pe_1v1.scripted_attacker import PretrainedAttackerPool
    from envs.pe_1v1.scripted_defender import PretrainedDefenderPool
    assert PretrainedDefenderPool is not None
    assert PretrainedAttackerPool is not None


def test_defender_pool_empty_ckpts_raises():
    """빈 ckpt list 거부."""
    from envs.pe_1v1.scripted_defender import PretrainedDefenderPool
    with pytest.raises(ValueError, match="at least 1 ckpt"):
        PretrainedDefenderPool(num_envs=4, ckpt_paths=[], device="cpu")


def test_attacker_pool_empty_ckpts_raises():
    from envs.pe_1v1.scripted_attacker import PretrainedAttackerPool
    with pytest.raises(ValueError, match="at least 1 ckpt"):
        PretrainedAttackerPool(num_envs=4, ckpt_paths=[], device="cpu")


def test_defender_pool_unsupported_sample_raises():
    from envs.pe_1v1.scripted_defender import PretrainedDefenderPool
    with pytest.raises(ValueError, match="not supported"):
        PretrainedDefenderPool(
            num_envs=4, ckpt_paths=["dummy"], device="cpu", sample="pfsp",
        )


@pytest.mark.parametrize("K", [1, 2, 3])
def test_defender_pool_step_shape_and_range(_gs_init, K):
    """K개 ckpt로 pool 생성 → step 출력 shape + value range 검증."""
    from envs.pe_1v1.scripted_defender import PretrainedDefenderPool
    ckpts = _find_defender_ckpts()
    if len(ckpts) < K:
        pytest.skip(f"need ≥{K} defender ckpts in curriculum_v9/stage_4/")
    selected = ckpts[:K]

    num_envs = 8
    pool = PretrainedDefenderPool(
        num_envs=num_envs, ckpt_paths=selected, device=gs.device, sample="uniform", seed=42,
    )
    # env_to_ckpt 검증
    assert pool.env_to_ckpt.shape == (num_envs,)
    assert int(pool.env_to_ckpt.min()) >= 0
    assert int(pool.env_to_ckpt.max()) < K

    # fake state with shape (num_envs, *) — only structural correctness.
    def _fake_state(B):
        return {
            "pos": torch.zeros((B, 3), device=gs.device),
            "vel": torch.zeros((B, 3), device=gs.device),
            "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * B, device=gs.device),
            "ang_vel": torch.zeros((B, 3), device=gs.device),
        }

    action = pool.step(_fake_state(num_envs), _fake_state(num_envs), g_mission=None)
    assert action.shape == (num_envs, 4)
    assert (action >= -1.0).all() and (action <= 1.0).all()
    # last_action 추적 확인
    assert torch.allclose(pool.last_action, action)


@pytest.mark.parametrize("K", [1, 2])
def test_attacker_pool_step_shape_and_range(_gs_init, K):
    from envs.pe_1v1.scripted_attacker import PretrainedAttackerPool
    ckpts = _find_attacker_ckpts()
    if len(ckpts) < K:
        pytest.skip(f"need ≥{K} attacker ckpts in curriculum_v9/stage_4/")
    selected = ckpts[:K]

    num_envs = 8
    pool = PretrainedAttackerPool(
        num_envs=num_envs, ckpt_paths=selected, device=gs.device, sample="uniform", seed=7,
    )
    assert pool.env_to_ckpt.shape == (num_envs,)
    assert int(pool.env_to_ckpt.min()) >= 0
    assert int(pool.env_to_ckpt.max()) < K

    def _fake_state(B):
        return {
            "pos": torch.zeros((B, 3), device=gs.device),
            "vel": torch.zeros((B, 3), device=gs.device),
            "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * B, device=gs.device),
            "ang_vel": torch.zeros((B, 3), device=gs.device),
        }

    g_mission = torch.zeros((num_envs, 3), device=gs.device)
    action = pool.step(_fake_state(num_envs), _fake_state(num_envs), g_mission=g_mission)
    assert action.shape == (num_envs, 4)
    assert (action >= -1.0).all() and (action <= 1.0).all()


def test_attacker_pool_g_mission_required(_gs_init):
    """attacker obs는 g_mission 필요 — None이면 ValueError."""
    from envs.pe_1v1.scripted_attacker import PretrainedAttackerPool
    ckpts = _find_attacker_ckpts()
    if not ckpts:
        pytest.skip("need attacker ckpt")
    pool = PretrainedAttackerPool(
        num_envs=4, ckpt_paths=[ckpts[0]], device=gs.device,
    )

    def _fake_state(B):
        return {
            "pos": torch.zeros((B, 3), device=gs.device),
            "vel": torch.zeros((B, 3), device=gs.device),
            "quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * B, device=gs.device),
            "ang_vel": torch.zeros((B, 3), device=gs.device),
        }

    with pytest.raises(ValueError, match="requires g_mission"):
        pool.step(_fake_state(4), _fake_state(4), g_mission=None)


def test_defender_pool_reset_zeros_last_action_and_reassigns(_gs_init):
    """reset(envs_idx) 후 last_action[envs_idx]=0 + env_to_ckpt 재할당."""
    from envs.pe_1v1.scripted_defender import PretrainedDefenderPool
    ckpts = _find_defender_ckpts()
    if len(ckpts) < 2:
        pytest.skip("need ≥2 defender ckpts")
    pool = PretrainedDefenderPool(
        num_envs=8, ckpt_paths=ckpts[:2], device=gs.device, seed=0,
    )
    pool.last_action[:] = 0.5
    before_assignment = pool.env_to_ckpt.clone()

    envs_idx = torch.tensor([0, 1, 2], device=gs.device)
    pool.reset(envs_idx)
    # last_action 영향 받은 envs는 0
    assert torch.allclose(pool.last_action[envs_idx], torch.zeros_like(pool.last_action[envs_idx]))
    # 영향 안 받은 envs는 그대로
    assert torch.allclose(pool.last_action[3:], torch.full_like(pool.last_action[3:], 0.5))
    # env_to_ckpt — 재할당된 envs는 [0, K) 범위 안 (값 자체는 random이라 동일할 수도)
    K = len(pool.actors)
    assert (pool.env_to_ckpt[envs_idx] >= 0).all()
    assert (pool.env_to_ckpt[envs_idx] < K).all()
    # 영향 안 받은 envs 그대로
    assert torch.equal(pool.env_to_ckpt[3:], before_assignment[3:])
