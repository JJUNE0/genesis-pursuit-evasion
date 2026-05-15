"""Phase 3f — Tests for τ-unaware forecaster (TODO §3f, 2026-05-14).

5 tests:
  1. Collator shape consistency (model-visible tensors are τ-independent shape).
  2. **τ leak test** — same (ep, t) sampled with two different τ values must
     produce identical attacker_history, action_history, target tensors;
     only defender_last differs (older defender state for larger τ).
  3. Frame transform parity vs env _attacker_obs() on real episode (target
     equals env's s_a_obs[t][18:24] byte-equal modulo float noise).
  4. CFM loss decreases on a single fixed (context, target) overfit.
  5. Euler sampling shape + finiteness.

Test 2 is the **most important** — it directly verifies the τ-unaware property
that paper main contribution depends on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.forecaster.dataset import load_scales_from_dataset   # noqa: E402
from agents.forecaster.tau_unaware import (   # noqa: E402
    TauStripCollator,
    TauUnawareDataset,
    TauUnawareDeterministicModel,
    TauUnawareForecasterModel,
    build_dataset_and_collator,
    build_sample_unaware,
    cfm_loss,
    sample_via_euler,
)

REAL_DATA_DIR = ROOT / "data" / "forecaster_data_ams_v9_c8"


# ---------------------------------------------------------------------------
# Synthetic episode helpers (mirror tests/test_forecaster.py pattern)
# ---------------------------------------------------------------------------

def _identity_episode(L: int = 200, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(int(seed))
    pos_a = torch.randn(L, 3, generator=g) * 2.0
    vel_a = torch.randn(L, 3, generator=g) * 0.5
    rot_a = torch.eye(3).reshape(1, 9).expand(L, 9).clone()
    ang_a = torch.randn(L, 3, generator=g) * 0.1
    s_a_world = torch.cat([pos_a, vel_a, rot_a, ang_a], dim=-1)

    pos_d = torch.randn(L, 3, generator=g) * 2.0
    vel_d = torch.randn(L, 3, generator=g) * 0.5
    rot_d = torch.eye(3).reshape(1, 9).expand(L, 9).clone()
    ang_d = torch.randn(L, 3, generator=g) * 0.1
    s_d_world = torch.cat([pos_d, vel_d, rot_d, ang_d], dim=-1)

    return {
        "s_a_world": s_a_world,
        "s_d_world": s_d_world,
        "s_a_obs": torch.zeros(L, 31),
        "a_a": (torch.rand(L, 4, generator=g) * 2 - 1),
        "a_d": (torch.rand(L, 4, generator=g) * 2 - 1),
        "g_mission": torch.randn(1, 3, generator=g).expand(L, 3).clone(),
        "episode_length": int(L),
        "termination_reason": "mission_success",
        "seed": int(seed),
        "attacker_ckpt_path": "<test>",
        "defender_ckpt_path": "<test>",
    }


def _make_synthetic_dataset_dir(tmp_path: Path, num_episodes: int = 6, L: int = 200) -> Path:
    out = tmp_path / "forecaster_data_synthetic"
    out.mkdir(parents=True, exist_ok=True)
    episodes_index = []
    for i in range(num_episodes):
        ep = _identity_episode(L=L, seed=i)
        torch.save(ep, out / f"episode_{i:06d}.pt")
        episodes_index.append({
            "id": i, "length": L, "termination": "mission_success",
            "file": f"episode_{i:06d}.pt",
        })
    (out / "index.json").write_text(json.dumps({
        "total_episodes": num_episodes,
        "episodes": episodes_index,
        "stats": {"mean_length": L, "min_length": L, "max_length": L,
                  "termination_dist": {"mission_success": 1.0, "captured": 0.0,
                                       "out_of_bounds": 0.0, "timeout": 0.0}},
    }))
    (out / "metadata.json").write_text(json.dumps({
        "total_episodes": num_episodes,
        "attacker_ckpt": "<test>", "defender_ckpt": "<test>",
        "cfgs_pkl": "<missing>", "tau_delay": 0, "num_envs": 4, "seed": 0,
        "env_yaml_override": None,
    }))
    return out


# ---------------------------------------------------------------------------
# Test 1 — Collator shape (τ-independent for model-visible tensors)
# ---------------------------------------------------------------------------

def test_collator_shapes(tmp_path):
    """All model-visible tensors have shape that does NOT depend on τ."""
    data_dir = _make_synthetic_dataset_dir(tmp_path, num_episodes=8, L=120)
    dataset, collator = build_dataset_and_collator(
        data_dir=data_dir, mode="train", history_K=50, split_seed=0,
        in_memory=False, rng_seed=123,
    )
    raw = [dataset[i] for i in range(4)]

    for tau_val in (1, 10, 25, 50):
        batch, tau = collator(raw, tau_override=tau_val)
        assert tau == tau_val
        # τ-independent shapes (CRITICAL):
        assert batch["attacker_history"].shape == (4, 50, 18)
        assert batch["defender_last"].shape == (4, 1, 6)
        assert batch["action_history"].shape == (4, 50, 4)
        assert batch["target"].shape == (4, 6)


# ---------------------------------------------------------------------------
# Test 2 — τ LEAK TEST (paper main contribution depends on this)
# ---------------------------------------------------------------------------

def test_tau_leak_invariant(tmp_path):
    """Same (ep, t) at different τ → only defender_last differs.

    Verifies that:
      - attacker_history, action_history, target are byte-equal across τ
        (independent of τ — model can't infer τ from these)
      - defender_last DOES differ (older defender state for larger τ;
        this is the only τ-dependent quantity, hidden from model as a value)
    """
    data_dir = _make_synthetic_dataset_dir(tmp_path, num_episodes=4, L=200)
    dataset, collator = build_dataset_and_collator(
        data_dir=data_dir, mode="train", history_K=50, split_seed=0,
        in_memory=False, rng_seed=42,
    )
    # Use the same 2 (ep, t) samples at different τ.
    raw = [dataset[0], dataset[1]]

    batch_tau1, _ = collator(raw, tau_override=1)
    batch_tau25, _ = collator(raw, tau_override=25)
    batch_tau50, _ = collator(raw, tau_override=50)

    # τ-independent quantities: byte-equal across τ
    for key in ("attacker_history", "action_history", "target"):
        assert torch.equal(batch_tau1[key], batch_tau25[key]), (
            f"{key} differs between τ=1 and τ=25 — τ leak!"
        )
        assert torch.equal(batch_tau1[key], batch_tau50[key]), (
            f"{key} differs between τ=1 and τ=50 — τ leak!"
        )

    # defender_last MUST differ between distinct τ (its value reflects τ).
    assert not torch.equal(batch_tau1["defender_last"], batch_tau25["defender_last"]), (
        "defender_last identical across τ=1, τ=25 — τ has no effect on data!"
    )
    assert not torch.equal(batch_tau1["defender_last"], batch_tau50["defender_last"]), (
        "defender_last identical across τ=1, τ=50 — τ has no effect on data!"
    )


# ---------------------------------------------------------------------------
# Test 3 — Frame transform parity vs env _attacker_obs() (real episode)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REAL_DATA_DIR / "episode_000000.pt").is_file(),
    reason="real Phase 3b dataset not available",
)
def test_frame_transform_matches_env():
    """For τ=1, defender_last (= s^d at t-1 in (t-1)-attacker frame) and
    attacker_history[-1] (= attacker self at t-1) check against the env's
    s_a_obs at t-1 (which uses (t-1)-attacker frame for both quantities)."""
    episode = torch.load(REAL_DATA_DIR / "episode_000000.pt", weights_only=False)
    L = episode["episode_length"]
    if L <= 60:
        pytest.skip(f"episode too short ({L}) for K=50 history test")

    pos_scale, vel_scale, ang_vel_scale = load_scales_from_dataset(REAL_DATA_DIR)

    K = 50
    t = min(100, L - 2)   # need t+1 ≤ L
    tau = 1

    s_a_world = episode["s_a_world"][t - K : t]                      # (K, 18)
    s_d_world_last = episode["s_d_world"][t - tau : t - tau + 1]     # (1, 18)
    s_d_world_target = episode["s_d_world"][t : t + 1]               # (1, 18)
    a_a = episode["a_a"][t - K : t]                                  # (K, 4)

    sample = build_sample_unaware(
        s_a_world_window=s_a_world,
        s_d_world_last=s_d_world_last,
        s_d_world_target=s_d_world_target,
        a_a_window=a_a,
        pos_scale=pos_scale, vel_scale=vel_scale, ang_vel_scale=ang_vel_scale,
    )

    # attacker_history[-1] should equal env's s_a_obs[t-1][0:18] (reference frame is t-1).
    expected_attacker = episode["s_a_obs"][t - 1, 0:18]
    actual_attacker = sample["attacker_history"][-1]
    assert torch.allclose(actual_attacker, expected_attacker, atol=1e-4), (
        f"attacker[-1] vs env s_a_obs[t-1][0:18] mismatch: max diff = "
        f"{(actual_attacker - expected_attacker).abs().max().item():.2e}"
    )

    # defender_last (τ=1: defender at t-1 in (t-1)-attacker frame) should equal
    # env's s_a_obs[t-1][18:24] (env at t-1 sees current defender = defender at t-1).
    expected_def = episode["s_a_obs"][t - 1, 18:24]
    actual_def = sample["defender_last"][0]
    assert torch.allclose(actual_def, expected_def, atol=1e-4), (
        f"defender_last vs env s_a_obs[t-1][18:24] mismatch: max diff = "
        f"{(actual_def - expected_def).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — CFM loss overfit
# ---------------------------------------------------------------------------

def test_cfm_loss_overfit():
    """Single fixed (context, target) overfit — loss should drop by > 50%."""
    torch.manual_seed(0)
    K = 50
    model = TauUnawareForecasterModel(
        d_model=128, n_heads=4, n_enc_layers=2, n_dec_layers=2,
        history_K=K, dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = 4
    g = torch.Generator().manual_seed(0)
    one_attacker = torch.randn(1, K, 18, generator=g)
    one_defender = torch.randn(1, 1, 6, generator=g)
    one_action = torch.randn(1, K, 4, generator=g)
    one_target = torch.randn(1, 6, generator=g)
    batch = {
        "attacker_history": one_attacker.expand(B, -1, -1).contiguous(),
        "defender_last":    one_defender.expand(B, -1, -1).contiguous(),
        "action_history":   one_action.expand(B, -1, -1).contiguous(),
        "target":           one_target.expand(B, -1).contiguous(),
    }

    initial = []
    for _ in range(10):
        with torch.no_grad():
            initial.append(cfm_loss(model, batch).item())
    initial_avg = float(np.mean(initial))

    for _ in range(500):
        loss = cfm_loss(model, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final = []
    for _ in range(10):
        with torch.no_grad():
            final.append(cfm_loss(model, batch).item())
    final_avg = float(np.mean(final))

    assert final_avg < initial_avg * 0.5, (
        f"CFM loss did not decrease enough: initial avg {initial_avg:.4f} → "
        f"final avg {final_avg:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Euler sampling shape + finiteness
# ---------------------------------------------------------------------------

def test_euler_sampling_shape():
    torch.manual_seed(42)
    K = 50
    model = TauUnawareForecasterModel(
        d_model=64, n_heads=2, n_enc_layers=1, n_dec_layers=1, history_K=K,
    )
    model.eval()

    B = 4
    context = {
        "attacker_history": torch.randn(B, K, 18),
        "defender_last":    torch.randn(B, 1, 6),
        "action_history":   torch.randn(B, K, 4),
    }
    pred = sample_via_euler(model=model, context=context, num_steps=10)
    assert pred.shape == (B, 6)
    assert torch.isfinite(pred).all()


# ---------------------------------------------------------------------------
# Test 6 (bonus) — Deterministic head ablation model works
# ---------------------------------------------------------------------------

def test_deterministic_head_forward():
    """TauUnawareDeterministicModel forward returns (B, 6) finite."""
    K = 50
    model = TauUnawareDeterministicModel(
        d_model=64, n_heads=2, n_enc_layers=1, history_K=K,
    )
    model.eval()
    B = 4
    context = {
        "attacker_history": torch.randn(B, K, 18),
        "defender_last":    torch.randn(B, 1, 6),
        "action_history":   torch.randn(B, K, 4),
    }
    with torch.no_grad():
        out = model(context)
    assert out.shape == (B, 6)
    assert torch.isfinite(out).all()
