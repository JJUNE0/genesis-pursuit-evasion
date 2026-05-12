"""Pure-CPU smokes for scripted attacker / defender (no Genesis sim).

The ``ScriptedPolicy`` interface only requires returning shape (B, 4) tensors
in [-1, 1]; these tests pin that contract without spinning up Genesis.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from envs.pe_1v1.cfgs import DEFENDER_OBS_DIM
from envs.pe_1v1.scripted_attacker import RandomWalkAttacker
from envs.pe_1v1.scripted_defender import (
    PretrainedDefender,
    RandomWalkDefender,
    StationaryDefender,
)


def _fake_state(B: int) -> dict[str, torch.Tensor]:
    state = {
        "pos": torch.zeros((B, 3)),
        "vel": torch.zeros((B, 3)),
        "quat": torch.zeros((B, 4)),
        "ang_vel": torch.zeros((B, 3)),
    }
    state["quat"][:, 0] = 1.0
    return state


class _DummyDefenderActor(nn.Module):
    """Smallest module that satisfies the PretrainedDefender load contract.

    Module-level class so ``torch.save(actor)`` produces a pickle that the
    test process can re-import.
    """

    def __init__(self, obs_dim: int = DEFENDER_OBS_DIM, num_actions: int = 4):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.linear = nn.Linear(self.obs_dim, num_actions)

    def forward(self, td):  # type: ignore[no-untyped-def]
        return torch.tanh(self.linear(td["policy"]))


def test_random_walk_attacker_shape_range_finite():
    B = 32
    pol = RandomWalkAttacker(num_envs=B, device="cpu")
    pol.reset(torch.arange(B))
    a = pol.step(_fake_state(B), _fake_state(B), torch.zeros((B, 3)))
    assert a.shape == (B, 4)
    assert torch.isfinite(a).all()
    assert (a >= -1.0).all() and (a <= 1.0).all()


def test_random_walk_attacker_non_degenerate():
    B = 32
    pol = RandomWalkAttacker(num_envs=B, device="cpu")
    a1 = pol.step(_fake_state(B), _fake_state(B), torch.zeros((B, 3)))
    a2 = pol.step(_fake_state(B), _fake_state(B), torch.zeros((B, 3)))
    # consecutive samples must differ
    assert (a1 - a2).abs().max() > 1e-3


def test_stationary_defender_zero_action():
    B = 16
    pol = StationaryDefender(num_envs=B, device="cpu")
    a = pol.step(_fake_state(B), _fake_state(B), None)
    assert a.shape == (B, 4)
    assert (a == 0).all()


def test_random_walk_defender_shape_range():
    B = 16
    pol = RandomWalkDefender(num_envs=B, device="cpu")
    a = pol.step(_fake_state(B), _fake_state(B), None)
    assert a.shape == (B, 4)
    assert (a >= -1.0).all() and (a <= 1.0).all()


def test_pretrained_defender_missing_ckpt_raises():
    """No defender_v0.ckpt yet → constructor must refuse loudly."""
    with pytest.raises(FileNotFoundError):
        PretrainedDefender(num_envs=4, ckpt_path="/nonexistent.ckpt", device="cpu")


def test_pretrained_defender_loads_and_steps(tmp_path):
    """Save a minimal actor → PretrainedDefender loads it → forward yields (B,4)."""
    pytest.importorskip(
        "rsl_rl", reason="PretrainedDefender ckpt unpickle requires rsl_rl import",
    )
    ckpt = tmp_path / "actor_v0.pt"
    torch.save(_DummyDefenderActor(), ckpt)

    B = 8
    pol = PretrainedDefender(num_envs=B, ckpt_path=ckpt, device="cpu")
    a = pol.step(_fake_state(B), _fake_state(B), None)
    assert a.shape == (B, 4)
    assert torch.isfinite(a).all()
    assert (a >= -1.0).all() and (a <= 1.0).all()
    # Frozen — params must not require grad after load.
    assert all(not p.requires_grad for p in pol.actor.parameters())


def test_pretrained_defender_obs_dim_mismatch_raises(tmp_path):
    pytest.importorskip("rsl_rl")
    ckpt = tmp_path / "wrong_dim.pt"
    torch.save(_DummyDefenderActor(obs_dim=DEFENDER_OBS_DIM + 5), ckpt)
    with pytest.raises(ValueError, match="obs_dim"):
        PretrainedDefender(num_envs=4, ckpt_path=ckpt, device="cpu")
