"""Phase 3c — Forecaster module tests (TODO §3c, 2026-05-13).

5 tests:
  1. DataLoader shape consistency (Pattern 2 collator).
  2. Frame transformation parity vs env _attacker_obs() (real episode).
  3. Model forward pass shape.
  4. RF loss decreases on overfit smoke (proves trainable).
  5. Euler sampling shape + finite output.

Test 2 uses a real episode from data/forecaster_data_ams_v7_c8/episode_000000.pt
(checked: target[-1] should equal env's s_a_obs[t][18:24] byte-equal modulo
float noise, by construction).
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

from agents.forecaster.dataset import (  # noqa: E402
    Pattern2Collator,
    build_dataset_and_collator,
    build_sample,
    load_scales_from_dataset,
)
from agents.forecaster.model import ForecasterModel  # noqa: E402
from agents.forecaster.rectified_flow import rf_loss, sample_via_euler  # noqa: E402

REAL_DATA_DIR = ROOT / "data" / "forecaster_data_ams_v7_c8"


# ---------------------------------------------------------------------------
# Synthetic episode helpers
# ---------------------------------------------------------------------------

def _identity_episode(L: int = 200, seed: int = 0) -> dict:
    """Synthetic episode with identity rotation (so frame transforms are well-defined)."""
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
        "s_a_obs": torch.zeros(L, 31),    # not used by these synthetic tests
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
    # metadata.json with cfgs_pkl pointing to a nonexistent path → defaults
    (out / "metadata.json").write_text(json.dumps({
        "total_episodes": num_episodes,
        "attacker_ckpt": "<test>", "defender_ckpt": "<test>",
        "cfgs_pkl": "<missing>", "tau_delay": 0, "num_envs": 4, "seed": 0,
        "env_yaml_override": None,
    }))
    return out


# ---------------------------------------------------------------------------
# Test 1 — DataLoader shape consistency
# ---------------------------------------------------------------------------

def test_dataloader_shapes(tmp_path):
    """Pattern 2 collator returns expected tensor shapes for arbitrary τ."""
    data_dir = _make_synthetic_dataset_dir(tmp_path, num_episodes=8, L=120)
    dataset, collator = build_dataset_and_collator(
        data_dir=data_dir, mode="train", tau_max=50, split_seed=0,
        in_memory=False, rng_seed=123,
    )
    # Sample 4 indices
    raw = [dataset[i] for i in range(4)]
    batch, tau = collator(raw)

    assert 1 <= tau <= 50
    assert batch["attacker_history"].shape == (4, tau + 1, 18)
    assert batch["defender_delayed"].shape == (4, 1, 6)
    assert batch["action_history"].shape == (4, tau, 4)
    assert batch["target"].shape == (4, tau, 6)

    # tau_override sanity (validation path)
    for tau_eval in (1, 10, 50):
        batch_v, tau_v = collator(raw, tau_override=tau_eval)
        assert tau_v == tau_eval
        assert batch_v["attacker_history"].shape == (4, tau_eval + 1, 18)
        assert batch_v["target"].shape == (4, tau_eval, 6)


# ---------------------------------------------------------------------------
# Test 2 — Frame transformation parity with env _attacker_obs() (real episode)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REAL_DATA_DIR / "episode_000000.pt").is_file(),
    reason="real Phase 3b dataset not available",
)
def test_frame_transform_matches_env():
    """For any τ, target[-1] (defender at t in t-attacker frame) ==
    env-recorded s_a_obs[t][18:24] byte-equal modulo fp tolerance.

    Also checks attacker_history[-1] (attacker self at t) ≈ s_a_obs[t][0:18].
    Manual computation verifies defender_delayed[0] (t-τ defender in t-attacker
    frame) from world-frame inputs.
    """
    episode = torch.load(REAL_DATA_DIR / "episode_000000.pt", weights_only=False)
    L = episode["episode_length"]
    if L <= 20:
        pytest.skip(f"episode too short ({L}) for τ=20 test")

    pos_scale, vel_scale, ang_vel_scale = load_scales_from_dataset(REAL_DATA_DIR)

    t = min(100, L - 1)
    tau = 20

    s_a_world = episode["s_a_world"][t - tau : t + 1]                  # (τ+1, 18)
    s_d_world = episode["s_d_world"][t - tau : t + 1]                  # (τ+1, 18)
    a_a = episode["a_a"][t - tau : t]                                  # (τ, 4)

    sample = build_sample(
        s_a_world_window=s_a_world, s_d_world_window=s_d_world,
        a_a_window=a_a, pos_scale=pos_scale, vel_scale=vel_scale,
        ang_vel_scale=ang_vel_scale,
    )

    # --- A) attacker_history[-1] (18D) ≈ env s_a_obs[t][0:18] -----------
    expected_attacker = episode["s_a_obs"][t, 0:18]
    actual_attacker = sample["attacker_history"][-1]
    assert torch.allclose(actual_attacker, expected_attacker, atol=1e-4), (
        f"attacker self mismatch: max diff = "
        f"{(actual_attacker - expected_attacker).abs().max().item():.2e}"
    )

    # --- B) target[-1] (6D defender at t in t-attacker) ≈ env s_a_obs[t][18:24] ---
    expected_def_at_t = episode["s_a_obs"][t, 18:24]
    actual_def_at_t = sample["target"][-1]
    assert torch.allclose(actual_def_at_t, expected_def_at_t, atol=1e-4), (
        f"target[-1] vs env defender at t mismatch: max diff = "
        f"{(actual_def_at_t - expected_def_at_t).abs().max().item():.2e}"
    )

    # --- C) Manual computation of defender_delayed[0] -------------------
    attacker_pos_t = s_a_world[-1, 0:3]
    attacker_vel_t = s_a_world[-1, 3:6]
    attacker_rot_t = s_a_world[-1, 6:15].reshape(3, 3)

    pos_d_at_t_minus_tau = s_d_world[0, 0:3]
    vel_d_at_t_minus_tau = s_d_world[0, 3:6]
    rel_pos_w = pos_d_at_t_minus_tau - attacker_pos_t                  # (3,)
    rel_vel_w = vel_d_at_t_minus_tau - attacker_vel_t                  # (3,)
    rel_pos_b_manual = attacker_rot_t.T @ rel_pos_w * pos_scale        # (3,)
    rel_vel_b_manual = attacker_rot_t.T @ rel_vel_w * vel_scale        # (3,)
    expected_def_delayed = torch.cat([rel_pos_b_manual, rel_vel_b_manual])

    actual_def_delayed = sample["defender_delayed"][0]
    assert torch.allclose(actual_def_delayed, expected_def_delayed, atol=1e-5), (
        f"defender_delayed[0] manual mismatch: max diff = "
        f"{(actual_def_delayed - expected_def_delayed).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Model forward pass shape
# ---------------------------------------------------------------------------

def test_model_forward_shape():
    """ForecasterModel returns (B, τ, target_dim) for arbitrary B, τ."""
    model = ForecasterModel(d_model=128, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                            tau_max=50)
    model.eval()

    for B, tau in [(4, 25), (2, 1), (3, 50)]:
        context = {
            "attacker_history": torch.randn(B, tau + 1, 18),
            "defender_delayed": torch.randn(B, 1, 6),
            "action_history": torch.randn(B, tau, 4),
        }
        x_t = torch.randn(B, tau, 6)
        t = torch.rand(B, 1, 1)
        with torch.no_grad():
            v_pred = model(x_t=x_t, t=t, context=context, tau=tau)
        assert v_pred.shape == (B, tau, 6), f"B={B} τ={tau} → {v_pred.shape}"
        assert torch.isfinite(v_pred).all()


# ---------------------------------------------------------------------------
# Test 4 — RF loss decreases on overfit smoke
# ---------------------------------------------------------------------------

def test_rf_loss_overfit():
    """Single fixed (context, target) overfit — loss should drop substantially.

    With random context AND random target, Bayes-optimal loss is the noise
    floor (~1.0) since model can't predict v_target = target − noise from
    unrelated context. With a SINGLE shared (context, target) for the batch,
    optimal v_pred = (target − x_t)/(1−t) is a deterministic function of
    (x_t, t) → small model can memorize → loss → 0.
    """
    torch.manual_seed(0)
    model = ForecasterModel(d_model=128, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                            tau_max=30, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B, tau = 4, 12
    g = torch.Generator().manual_seed(0)
    one_attacker = torch.randn(1, tau + 1, 18, generator=g)
    one_defender = torch.randn(1, 1, 6, generator=g)
    one_action = torch.randn(1, tau, 4, generator=g)
    one_target = torch.randn(1, tau, 6, generator=g)
    batch = {
        "attacker_history": one_attacker.expand(B, -1, -1).contiguous(),
        "defender_delayed": one_defender.expand(B, -1, -1).contiguous(),
        "action_history":   one_action.expand(B, -1, -1).contiguous(),
        "target":           one_target.expand(B, -1, -1).contiguous(),
    }

    # Track loss (averaged over 10 stochastic noise/t draws).
    initial_losses = []
    for _ in range(10):
        with torch.no_grad():
            initial_losses.append(rf_loss(model, batch, tau).item())
    initial_avg = float(np.mean(initial_losses))

    for _ in range(500):
        loss = rf_loss(model, batch, tau)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final_losses = []
    for _ in range(10):
        with torch.no_grad():
            final_losses.append(rf_loss(model, batch, tau).item())
    final_avg = float(np.mean(final_losses))

    assert final_avg < initial_avg * 0.5, (
        f"RF loss did not decrease enough: initial avg {initial_avg:.4f} → "
        f"final avg {final_avg:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Euler sampling shape + finiteness
# ---------------------------------------------------------------------------

def test_euler_sampling_shape():
    """sample_via_euler returns (B, τ, target_dim) with finite values."""
    torch.manual_seed(42)
    model = ForecasterModel(d_model=64, n_heads=2, n_enc_layers=1, n_dec_layers=1,
                            tau_max=30)
    model.eval()

    B, tau = 4, 25
    context = {
        "attacker_history": torch.randn(B, tau + 1, 18),
        "defender_delayed": torch.randn(B, 1, 6),
        "action_history":   torch.randn(B, tau, 4),
    }
    pred = sample_via_euler(model=model, context=context, tau=tau, num_steps=20)
    assert pred.shape == (B, tau, 6)
    assert torch.isfinite(pred).all()
    assert not torch.isnan(pred).any()


# ---------------------------------------------------------------------------
# Test 6 — multi-dir dataset merges samples from multiple cycles
# ---------------------------------------------------------------------------

def test_multi_dir_dataset_merge(tmp_path):
    """ForecasterDataset accepts list[data_dir] and merges samples across dirs.

    Verifies:
      - len(multi) == sum(len(single_i)) up to per-dir 95/5 split alignment
      - sample_indices are 3-tuples (dir_idx, ep_id, t)
      - load_episode routes to correct dir
      - collator handles 3-tuple samples and produces correct shapes
    """
    from agents.forecaster.dataset import ForecasterDataset, build_dataset_and_collator

    dir_a = _make_synthetic_dataset_dir(tmp_path / "a", num_episodes=8, L=120)
    dir_b = _make_synthetic_dataset_dir(tmp_path / "b", num_episodes=6, L=120)

    ds_multi = ForecasterDataset(
        data_dir=[dir_a, dir_b], tau_max=50, mode="train", split_seed=0,
    )
    ds_a = ForecasterDataset(data_dir=dir_a, tau_max=50, mode="train", split_seed=0)
    ds_b = ForecasterDataset(data_dir=dir_b, tau_max=50, mode="train", split_seed=0 + 1000)
    # Note: per-dir split uses split_seed + dir_idx*1000, so ds_b matches dir_b in multi.

    assert len(ds_multi) == len(ds_a) + len(ds_b)
    assert len(ds_multi.data_dirs) == 2

    # 3-tuple format
    sample = ds_multi[0]
    assert len(sample) == 3
    dir_idx, ep_id, t = sample
    assert dir_idx in (0, 1)
    assert t >= ds_multi.tau_max

    # Both dirs represented
    dir_idxs = {ds_multi[i][0] for i in range(len(ds_multi))}
    assert dir_idxs == {0, 1}

    # Collator works end-to-end on multi-dir samples
    _, collator = build_dataset_and_collator(
        data_dir=[dir_a, dir_b], mode="train", tau_max=50, split_seed=0,
        in_memory=False, rng_seed=42,
    )
    raw = [ds_multi[i] for i in (0, len(ds_multi) // 2, len(ds_multi) - 1)]
    batched, tau = collator(raw, tau_override=10)
    assert tau == 10
    assert batched["attacker_history"].shape == (3, 11, 18)
    assert batched["target"].shape == (3, 10, 6)


# ---------------------------------------------------------------------------
# Test 7 (bonus) — validate() helper produces all expected keys
# ---------------------------------------------------------------------------

def test_validate_returns_all_metrics(tmp_path):
    """validate() over a synthetic val dataset returns all per-τ + averaged keys."""
    from agents.forecaster.trainer import build_val_dataset_and_collator, validate

    data_dir = _make_synthetic_dataset_dir(tmp_path, num_episodes=4, L=120)
    val_dataset, val_collator = build_val_dataset_and_collator(
        data_dir=data_dir, tau_max=50, in_memory=False, split_seed=0, rng_seed=0,
    )
    if len(val_dataset) == 0:
        pytest.skip("val split empty for tiny synthetic dataset")

    model = ForecasterModel(d_model=64, n_heads=2, n_enc_layers=1, n_dec_layers=1,
                            tau_max=50)
    metrics = validate(
        model=model, val_dataset=val_dataset, val_collator=val_collator,
        val_subsample_size=8, tau_evals=(10, 25, 50), num_euler_steps=4,
        device="cpu", seed=0,
    )
    for tau in (10, 25, 50):
        assert f"val/last_mse_tau{tau}" in metrics
        assert f"val/traj_mse_tau{tau}" in metrics
    assert "val/avg_last_mse" in metrics
    assert "val/avg_traj_mse" in metrics
