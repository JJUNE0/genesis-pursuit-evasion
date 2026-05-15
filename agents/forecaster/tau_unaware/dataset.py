"""Phase 3f — τ-Unaware Forecaster Dataset (TODO §3f, 2026-05-14).

Reads the same per-episode .pt files as Phase 3c (τ-aware) but constructs
samples that **never expose τ to the model**:

  - ``attacker_history``: fixed K=50 step real-time attacker self state
  - ``defender_last``:    a single ``s^d`` snapshot from t-τ (staleness HIDDEN)
  - ``action_history``:   fixed K step attacker actions
  - ``target``:           single-step ``s^d_t`` (current defender)

The τ value (∈ [1, K]) varies per batch but is **never written into any
model-visible tensor**: input shape, sequence length, position encoding —
all τ-independent. This is the core "τ-unaware" property.

CLAUDE.md §1: this module is a NEW addition; the Phase 3c
``agents/forecaster/dataset.py`` stays untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from agents.forecaster.dataset import load_scales_from_dataset   # reuse
from agents.forecaster.tau_unaware import constants as C


# ---------------------------------------------------------------------------
# Frame transform — single-step target version of build_sample
# ---------------------------------------------------------------------------

def build_sample_unaware(
    s_a_world_window: torch.Tensor,    # shape: (K, 18) — attacker world raw [t-K..t-1]
    s_d_world_last: torch.Tensor,      # shape: (1, 18) — defender world raw at t-τ
    s_d_world_target: torch.Tensor,    # shape: (1, 18) — defender world raw at t (target)
    a_a_window: torch.Tensor,          # shape: (K, 4)
    pos_scale: float,
    vel_scale: float,
    ang_vel_scale: float,
) -> dict[str, torch.Tensor]:
    """Build one τ-unaware training sample.

    Frame: every relative quantity is expressed in **t-attacker body frame**
    using attacker pose at t-1 (last available real-time attacker step).
    Mirrors Phase 3c ``build_sample`` formulas; only difference is single-step
    target + last-observed defender (vs multi-step).

    Returns dict::
        attacker_history : (K, 18) — env-style per-step attacker self obs
        defender_last    : (1, 6)  — defender at t-τ in (t-1)-attacker body frame
        action_history   : (K, 4)  — verbatim
        target           : (6,)    — defender at t in (t-1)-attacker body frame
    """
    K = s_a_world_window.shape[0]

    # ---- Per-step attacker self (env _attacker_obs() s_a layout) ----
    pos_k = s_a_world_window[:, 0:3] * pos_scale                        # (K, 3) world * scale
    vel_world_k = s_a_world_window[:, 3:6]                              # (K, 3) world raw
    rot_k = s_a_world_window[:, 6:15].reshape(K, 3, 3)                  # (K, 3, 3) body→world
    ang_k = s_a_world_window[:, 15:18] * ang_vel_scale                  # (K, 3) body * scale
    vel_body_k = torch.einsum("kji,kj->ki", rot_k, vel_world_k)         # (K, 3) body
    vel_k = vel_body_k * vel_scale
    rot_flat_k = s_a_world_window[:, 6:15]                              # (K, 9)
    attacker_history = torch.cat([pos_k, vel_k, rot_flat_k, ang_k], dim=-1)  # (K, 18)

    # ---- Reference frame: attacker pose at t-1 (last real-time step) ----
    attacker_pos_ref = s_a_world_window[-1, 0:3]                        # (3,) world
    attacker_vel_ref = s_a_world_window[-1, 3:6]                        # (3,) world
    attacker_rot_ref = s_a_world_window[-1, 6:15].reshape(3, 3)         # (3, 3) body→world

    def _to_body_ref(v_world: torch.Tensor) -> torch.Tensor:            # (..., 3) → (..., 3)
        # einsum 'ji,nj->ni' = R_ref^T @ each row vector
        return torch.einsum("ji,nj->ni", attacker_rot_ref, v_world)

    # ---- defender_last (at t-τ, in ref-attacker body frame) ----
    pos_d_last = s_d_world_last[0:1, 0:3]                                # (1, 3) world
    vel_d_last = s_d_world_last[0:1, 3:6]                                # (1, 3) world
    rel_pos_b_last = _to_body_ref(pos_d_last - attacker_pos_ref) * pos_scale   # (1, 3)
    rel_vel_b_last = _to_body_ref(vel_d_last - attacker_vel_ref) * vel_scale   # (1, 3)
    defender_last = torch.cat([rel_pos_b_last, rel_vel_b_last], dim=-1)        # (1, 6)

    # ---- target (at t, in ref-attacker body frame) ----
    pos_d_t = s_d_world_target[0:1, 0:3]                                 # (1, 3) world
    vel_d_t = s_d_world_target[0:1, 3:6]                                 # (1, 3) world
    rel_pos_b_t = _to_body_ref(pos_d_t - attacker_pos_ref) * pos_scale   # (1, 3)
    rel_vel_b_t = _to_body_ref(vel_d_t - attacker_vel_ref) * vel_scale   # (1, 3)
    target = torch.cat([rel_pos_b_t, rel_vel_b_t], dim=-1).squeeze(0)    # (6,)

    return {
        "attacker_history": attacker_history,
        "defender_last": defender_last,
        "action_history": a_a_window,
        "target": target,
    }


# ---------------------------------------------------------------------------
# TauUnawareDataset
# ---------------------------------------------------------------------------

class TauUnawareDataset(Dataset):
    """Per-(episode, t) sample index for τ-unaware forecaster training.

    Differences from Phase 3c ``ForecasterDataset``:
      - Sample shape is ``(dir_idx, ep_id, t)``: τ is decided in the collator.
      - Valid range is ``t ∈ [history_K, L-1]`` (history must fit; τ value is
        sampled within ``[1, history_K]`` by the collator and bounded by t).

    Args:
        data_dir:    single path or list of paths (multi-cycle merge).
        history_K:   fixed history window length seen by the model (default 50).
        mode:        'train' or 'val'. 95/5 episode-level split (per-dir, deterministic).
        split_seed:  deterministic split seed.
        in_memory:   preload all selected episodes into RAM.
        episode_subset: optional set of episode ids to further restrict (test convenience).
    """

    def __init__(
        self,
        data_dir: str | Path | list[str | Path],
        history_K: int = C.DEFAULT_HISTORY_K,
        mode: str = "train",
        split_seed: int = 42,
        in_memory: bool = False,
        episode_subset: set[int] | None = None,
    ):
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        if isinstance(data_dir, (str, Path)):
            self.data_dirs: list[Path] = [Path(data_dir)]
        else:
            self.data_dirs = [Path(d) for d in data_dir]
        if not self.data_dirs:
            raise ValueError("TauUnawareDataset: empty data_dir list")
        self.data_dir = self.data_dirs[0]   # backwards-compat alias
        self.history_K = int(history_K)
        self.mode = mode

        # Build (dir_idx, ep_id, t) sample list. Enforce t ∈ [history_K, L-1].
        self.sample_indices: list[tuple[int, int, int]] = []
        active_ids_per_dir: list[set[int]] = []

        for dir_idx, d in enumerate(self.data_dirs):
            index_path = d / "index.json"
            if not index_path.is_file():
                raise FileNotFoundError(f"index.json missing under {d}")
            index = json.loads(index_path.read_text())
            episodes = index["episodes"]

            rng = np.random.default_rng(int(split_seed) + dir_idx * 1000)
            ep_ids_all = np.array([int(e["id"]) for e in episodes])
            rng.shuffle(ep_ids_all)
            n_val = max(1, int(0.05 * len(ep_ids_all)))
            val_ids = set(int(x) for x in ep_ids_all[:n_val].tolist())
            train_ids = set(int(x) for x in ep_ids_all[n_val:].tolist())
            active_ids = val_ids if mode == "val" else train_ids
            if episode_subset is not None:
                active_ids = active_ids & set(int(x) for x in episode_subset)
            active_ids_per_dir.append(active_ids)

            for e in episodes:
                ep_id = int(e["id"])
                if ep_id not in active_ids:
                    continue
                L = int(e["length"])
                # Need t ∈ [history_K, L-1]. Also need t-τ ≥ 0 and t+1 ≤ L for
                # any τ ∈ [1, history_K]. t ≥ history_K satisfies both:
                # t - history_K ≥ 0 and target slice [t, t+1] requires t+1 ≤ L.
                if L > self.history_K:
                    self.sample_indices.extend(
                        (dir_idx, ep_id, t) for t in range(self.history_K, L - 1)
                    )

        self._memory_cache: dict[tuple[int, int], dict[str, Any]] = {}
        if in_memory:
            for dir_idx, active in enumerate(active_ids_per_dir):
                self._preload(dir_idx, active)

    # ------------------------------------------------------------------
    def _episode_path(self, dir_idx: int, ep_id: int) -> Path:
        return self.data_dirs[dir_idx] / f"episode_{ep_id:06d}.pt"

    def load_episode(self, dir_idx: int, ep_id: int) -> dict[str, Any]:
        key = (int(dir_idx), int(ep_id))
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached
        return torch.load(
            self._episode_path(int(dir_idx), int(ep_id)),
            weights_only=False, map_location="cpu",
        )

    def _preload(self, dir_idx: int, ep_ids: set[int]) -> None:
        for ep_id in sorted(ep_ids):
            self._memory_cache[(int(dir_idx), int(ep_id))] = torch.load(
                self._episode_path(int(dir_idx), int(ep_id)),
                weights_only=False, map_location="cpu",
            )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        return self.sample_indices[idx]


# ---------------------------------------------------------------------------
# TauStripCollator — samples τ but never lets it leak into the model input
# ---------------------------------------------------------------------------

class TauStripCollator:
    """Per-batch τ sampler that strips τ from the model-visible tensors.

    For each batch:
      1. Sample τ ~ U[1, history_K]  (or fixed via tau_override at val time).
      2. For each (dir_idx, ep_id, t): slice
           attacker_history = s^a[t-K:t]              # K steps, τ-independent
           defender_last    = s^d[t-τ:t-τ+1]          # 1 step at t-τ
           action_history   = a^a[t-K:t]              # K steps
           target           = s^d[t:t+1]              # current defender
      3. Frame-transform → t-attacker body frame.
      4. Stack → batched dict.

    The stacked tensors **do not encode τ** anywhere:
      - attacker_history shape: (B, K, 18) — independent of τ
      - defender_last shape:    (B, 1, 6)  — independent of τ
      - action_history shape:   (B, K, 4)  — independent of τ
      - target shape:           (B, 6)     — independent of τ

    Only the *value* of ``defender_last`` reflects τ (older defender state for
    larger τ). Test 2 in ``tests/test_tau_unaware_forecaster.py`` verifies this
    invariant rigorously by sampling two τ for the same (ep, t) and comparing.
    """

    def __init__(
        self,
        dataset: TauUnawareDataset,
        history_K: int = C.DEFAULT_HISTORY_K,
        scales: tuple[float, float, float] | None = None,
        rng_seed: int | None = None,
    ):
        self.dataset = dataset
        self.history_K = int(history_K)
        self.scales = (
            tuple(scales) if scales is not None
            else load_scales_from_dataset(dataset.data_dirs)
        )
        self._gen = torch.Generator()
        if rng_seed is not None:
            self._gen.manual_seed(int(rng_seed))

    def sample_tau(self) -> int:
        return int(torch.randint(1, self.history_K + 1, (1,), generator=self._gen).item())

    def __call__(
        self,
        batch: list[tuple[int, int, int]],
        tau_override: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        tau = int(tau_override) if tau_override is not None else self.sample_tau()
        if tau < 1 or tau > self.history_K:
            raise ValueError(f"tau out of [1, {self.history_K}]: {tau}")

        pos_scale, vel_scale, ang_vel_scale = self.scales
        K = self.history_K
        samples: list[dict[str, torch.Tensor]] = []
        for sample_idx in batch:
            if len(sample_idx) == 3:
                dir_idx, ep_id, t = sample_idx
            else:
                ep_id, t = sample_idx
                dir_idx = 0
            episode = self.dataset.load_episode(int(dir_idx), int(ep_id))
            t = int(t)
            # Slices (all τ-independent shapes; τ only shifts defender_last source idx)
            s_a_world = episode["s_a_world"][t - K : t]                    # (K, 18)
            s_d_world_last = episode["s_d_world"][t - tau : t - tau + 1]   # (1, 18)
            s_d_world_target = episode["s_d_world"][t : t + 1]             # (1, 18)
            a_a = episode["a_a"][t - K : t]                                # (K, 4)
            samples.append(build_sample_unaware(
                s_a_world_window=s_a_world,
                s_d_world_last=s_d_world_last,
                s_d_world_target=s_d_world_target,
                a_a_window=a_a,
                pos_scale=pos_scale, vel_scale=vel_scale, ang_vel_scale=ang_vel_scale,
            ))

        batched = {
            "attacker_history": torch.stack([s["attacker_history"] for s in samples]),  # (B, K, 18)
            "defender_last":    torch.stack([s["defender_last"]    for s in samples]),  # (B, 1, 6)
            "action_history":   torch.stack([s["action_history"]   for s in samples]),  # (B, K, 4)
            "target":           torch.stack([s["target"]           for s in samples]),  # (B, 6)
        }
        return batched, tau


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_dataset_and_collator(
    data_dir: str | Path | list[str | Path],
    mode: str,
    history_K: int = C.DEFAULT_HISTORY_K,
    split_seed: int = 42,
    in_memory: bool = False,
    episode_subset: set[int] | None = None,
    rng_seed: int | None = None,
) -> tuple[TauUnawareDataset, TauStripCollator]:
    dataset = TauUnawareDataset(
        data_dir=data_dir, history_K=history_K, mode=mode,
        split_seed=split_seed, in_memory=in_memory,
        episode_subset=episode_subset,
    )
    collator = TauStripCollator(
        dataset=dataset, history_K=history_K, rng_seed=rng_seed,
    )
    return dataset, collator
