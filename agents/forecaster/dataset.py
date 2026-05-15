"""Phase 3c — Forecaster dataset + Pattern 2 collator (TODO §3c, 2026-05-13).

Reads per-episode .pt files emitted by ``scripts/collect_forecaster_data.py``
and yields ``(episode_id, t)`` indices. The Pattern 2 collator picks **one
τ for the whole batch** and slices ``[t-τ, t+1]`` windows out of each episode,
transforming world-frame raw state into the **t-attacker body frame** to
mirror ``envs.pe_1v1.env.PursuitEvasion1v1Env._attacker_obs()`` exactly.

Spec (CLAUDE.md §3 / §4):
- Variable names follow paper symbols (s^a, s^d, τ, ...).
- Every tensor expression carries a shape comment.
- Frame convention is **t-attacker body frame** for relative quantities,
  matching env obs (so target[-1] equals env's s_a_obs[t][18:24] byte-equal
  modulo float noise).
- Dataset valid range: ``t ∈ [τ_max, L-1]`` (worst-case, ~12.5% data loss).
- 95/5 episode-level train/val split (deterministic via split_seed).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Scale loading from dataset metadata → cfgs.pkl
# ---------------------------------------------------------------------------

def load_scales_from_dataset(
    data_dir: str | Path | list[str | Path],
) -> tuple[float, float, float]:
    """Return (pos_scale, vel_scale, ang_vel_scale) from the dataset's source cfgs.pkl.

    Accepts a single dir or a list (multi-cycle dataset). For a list, scales
    are read from the **first** dir; the remaining dirs are checked for
    matching scales and a warning is printed on mismatch.

    metadata.json points to the cfgs.pkl used at collection time; that pickle
    contains ``obs_cfg.obs_scales`` (DEFAULT_OBS_SCALES merged with yaml
    overrides). Falls back to library defaults if cfgs.pkl is unreachable.
    """
    if isinstance(data_dir, (str, Path)):
        dirs = [Path(data_dir)]
    else:
        dirs = [Path(d) for d in data_dir]
    if not dirs:
        raise ValueError("load_scales_from_dataset: empty data_dir list")

    def _read_one(d: Path) -> tuple[float, float, float]:
        meta_path = d / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"dataset metadata.json missing: {meta_path}")
        meta = json.loads(meta_path.read_text())
        cfgs_pkl_path = Path(meta["cfgs_pkl"])
        if not cfgs_pkl_path.is_file():
            return 0.2, 0.2, 0.31831  # DEFAULT_OBS_SCALES fallback
        with open(cfgs_pkl_path, "rb") as f:
            cfgs = pickle.load(f)
        scales = cfgs["obs_cfg"]["obs_scales"]
        return float(scales["pos"]), float(scales["vel"]), float(scales["ang_vel"])

    head = _read_one(dirs[0])
    for d in dirs[1:]:
        other = _read_one(d)
        if not all(abs(a - b) < 1e-9 for a, b in zip(head, other)):
            print(f"[forecaster] ⚠ obs_scales mismatch between {dirs[0]} ({head}) "
                  f"and {d} ({other}); using {dirs[0]} values.", flush=True)
    return head


# ---------------------------------------------------------------------------
# Frame transform — env _attacker_obs() parity (CRITICAL)
# ---------------------------------------------------------------------------

def build_sample(
    s_a_world_window: torch.Tensor,    # shape: (τ+1, 18) — attacker world raw [t-τ..t]
    s_d_world_window: torch.Tensor,    # shape: (τ+1, 18) — defender world raw [t-τ..t]
    a_a_window: torch.Tensor,          # shape: (τ, 4)    — attacker actions [t-τ..t-1]
    pos_scale: float,
    vel_scale: float,
    ang_vel_scale: float,
) -> dict[str, torch.Tensor]:
    """Transform a per-episode world-frame window → t-attacker body frame sample.

    Mirrors env._attacker_obs() formulas exactly (envs/pe_1v1/env.py:388):
    - ``pos_a`` (attacker self): world * pos_scale (env keeps attacker's own
      position in world frame intentionally).
    - ``vel_a``: rotate(world_vel, quat_at_step) * vel_scale  (per-step body).
    - ``rot_flat``: 9D rotation matrix flat (body→world), pass through.
    - ``ang_vel``: env stores body-frame, just * ang_vel_scale.
    - ``rel_pos_d`` / ``rel_vel_d``: rotate world-rel into **t-attacker** body
      frame (NOT per-step body — defender is expressed from t-attacker's POV).
    - ``target[k]`` for k ∈ [0, τ): defender at t-τ+1+k in t-attacker frame.
      target[-1] equals env s_a_obs[t][18:24] by construction.

    Returns dict::
        attacker_history : (τ+1, 18) — env-style per-step attacker self obs
        defender_delayed : (1, 6)    — defender at t-τ in t-attacker frame
        action_history   : (τ, 4)    — attacker actions verbatim
        target           : (τ, 6)    — defender at t-τ+1..t in t-attacker frame
    """
    K = s_a_world_window.shape[0]                                     # τ+1

    # ---- Per-step attacker self (env _attacker_obs() s_a layout) ----
    pos_k = s_a_world_window[:, 0:3] * pos_scale                       # (K, 3) world * scale
    vel_world_k = s_a_world_window[:, 3:6]                             # (K, 3) world raw
    rot_k = s_a_world_window[:, 6:15].reshape(K, 3, 3)                 # (K, 3, 3) body→world
    ang_k = s_a_world_window[:, 15:18] * ang_vel_scale                 # (K, 3) body * scale

    # vel body at step k = R_k^T @ world_vel_k  → einsum 'kji,kj->ki'
    vel_body_k = torch.einsum("kji,kj->ki", rot_k, vel_world_k)        # (K, 3)
    vel_k = vel_body_k * vel_scale                                     # (K, 3) body * scale
    rot_flat_k = s_a_world_window[:, 6:15]                             # (K, 9)
    attacker_history = torch.cat([pos_k, vel_k, rot_flat_k, ang_k], dim=-1)  # (K, 18)

    # ---- t-attacker pose (used for defender / target rel transforms) ----
    attacker_pos_t = s_a_world_window[-1, 0:3]                         # (3,) world
    attacker_vel_t = s_a_world_window[-1, 3:6]                         # (3,) world
    attacker_rot_t = s_a_world_window[-1, 6:15].reshape(3, 3)          # (3, 3) body→world

    # Helper: world vec (..., 3) → t-attacker body (..., 3) = R_t^T @ vec
    # einsum 'ji,nj->ni' equals (n × R^T)^T per row, i.e. R_t^T applied to each row.
    def _to_body_t(v_world: torch.Tensor) -> torch.Tensor:             # (..., 3) -> (..., 3)
        return torch.einsum("ji,nj->ni", attacker_rot_t, v_world)

    # ---- defender_delayed: at index 0 (= step t-τ), in t-attacker body frame ----
    pos_d_delayed = s_d_world_window[0:1, 0:3]                         # (1, 3) world
    vel_d_delayed = s_d_world_window[0:1, 3:6]                         # (1, 3) world
    rel_pos_b_d = _to_body_t(pos_d_delayed - attacker_pos_t) * pos_scale   # (1, 3)
    rel_vel_b_d = _to_body_t(vel_d_delayed - attacker_vel_t) * vel_scale   # (1, 3)
    defender_delayed = torch.cat([rel_pos_b_d, rel_vel_b_d], dim=-1)   # (1, 6)

    # ---- target: defender at indices 1..τ in t-attacker body frame ----
    pos_d_tgt = s_d_world_window[1:, 0:3]                              # (τ, 3) world
    vel_d_tgt = s_d_world_window[1:, 3:6]                              # (τ, 3) world
    rel_pos_b_t = _to_body_t(pos_d_tgt - attacker_pos_t) * pos_scale   # (τ, 3)
    rel_vel_b_t = _to_body_t(vel_d_tgt - attacker_vel_t) * vel_scale   # (τ, 3)
    target = torch.cat([rel_pos_b_t, rel_vel_b_t], dim=-1)             # (τ, 6)

    return {
        "attacker_history": attacker_history,
        "defender_delayed": defender_delayed,
        "action_history": a_a_window,
        "target": target,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ForecasterDataset(Dataset):
    """Per-(episode, t) sample index for forecaster training.

    Supports a **single dir** (legacy) or a **list of dirs** (multi-cycle):
    each dir contributes its own episodes, and the dataset transparently merges
    them. Internally indices are 3-tuples ``(dir_idx, ep_id, t)``; the collator
    routes ``load_episode(dir_idx, ep_id)`` through the right dir.

    Args:
        data_dir: single path **or** list of paths (one per cycle / source).
        tau_max:  enforce ``t ≥ tau_max`` so all batched τ are valid.
        mode:     ``'train'`` or ``'val'``. 95/5 episode-level split (per dir).
        split_seed: deterministic split seed (same seed → consistent split).
        in_memory:  if True, preload all selected episodes from every dir into
                    RAM. Big speed-up; for 5×4000 ep budget ~6 GB.
        episode_subset: optional iterable of episode ids to further restrict
                        (used by tests / smoke runs). Applied **uniformly to
                        every dir** — primarily a single-dir test convenience.
    """

    def __init__(
        self,
        data_dir: str | Path | list[str | Path],
        tau_max: int = 50,
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
            raise ValueError("ForecasterDataset: empty data_dir list")
        # Backwards-compat alias for code that still reads ``dataset.data_dir``.
        self.data_dir = self.data_dirs[0]
        self.tau_max = int(tau_max)
        self.mode = mode

        # Build sample index across all dirs.
        # sample_indices: list of (dir_idx, ep_id, t)
        self.sample_indices: list[tuple[int, int, int]] = []
        # active_ids_per_dir: dir_idx → set of ep_ids in this mode (for in-memory preload)
        active_ids_per_dir: list[set[int]] = []

        for dir_idx, d in enumerate(self.data_dirs):
            index_path = d / "index.json"
            if not index_path.is_file():
                raise FileNotFoundError(f"index.json missing under {d}")
            index = json.loads(index_path.read_text())
            episodes = index["episodes"]

            # Per-dir 95/5 split (deterministic). Independent split per dir
            # → consistent val coverage even when adding/removing dirs later.
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
                if L > self.tau_max:
                    self.sample_indices.extend(
                        (dir_idx, ep_id, t) for t in range(self.tau_max, L)
                    )

        # In-memory cache keyed by (dir_idx, ep_id).
        self._memory_cache: dict[tuple[int, int], dict[str, Any]] = {}
        if in_memory:
            for dir_idx, active in enumerate(active_ids_per_dir):
                self._preload(dir_idx, active)

    # ------------------------------------------------------------------
    # IO + cache (multi-dir aware)
    # ------------------------------------------------------------------

    def _episode_path(self, dir_idx: int, ep_id: int) -> Path:
        return self.data_dirs[dir_idx] / f"episode_{ep_id:06d}.pt"

    def load_episode(self, dir_idx: int, ep_id: int) -> dict[str, Any]:
        """Cached or on-demand episode load (routed through dir_idx)."""
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

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        # Returns (dir_idx, ep_id, t); collator does the frame transform.
        return self.sample_indices[idx]


# ---------------------------------------------------------------------------
# Pattern 2 Collator — single τ per batch
# ---------------------------------------------------------------------------

class Pattern2Collator:
    """Collator that samples a single τ per batch and builds the model inputs.

    Pattern 2 (CLAUDE.md §3.4 v10): a single ``τ ∈ [1, τ_max]`` is drawn for
    every call; all batch items use the same τ so the encoder/decoder
    sequence lengths match.

    Pass ``tau_override`` to fix τ (used by validation per-τ sweep).
    """

    def __init__(
        self,
        dataset: ForecasterDataset,
        tau_max: int = 50,
        scales: tuple[float, float, float] | None = None,
        rng_seed: int | None = None,
    ):
        self.dataset = dataset
        self.tau_max = int(tau_max)
        # Multi-dir aware: pass the full list of dirs so cross-dir mismatch is checked.
        self.scales = (
            tuple(scales) if scales is not None
            else load_scales_from_dataset(dataset.data_dirs)
        )
        # Dedicated generator so worker forks get reproducible-yet-distinct streams.
        self._gen = torch.Generator()
        if rng_seed is not None:
            self._gen.manual_seed(int(rng_seed))

    def sample_tau(self) -> int:
        return int(torch.randint(1, self.tau_max + 1, (1,), generator=self._gen).item())

    def __call__(
        self,
        batch: list[tuple[int, int, int]],
        tau_override: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        tau = int(tau_override) if tau_override is not None else self.sample_tau()
        if tau < 1 or tau > self.tau_max:
            raise ValueError(f"tau out of [1, {self.tau_max}]: {tau}")

        pos_scale, vel_scale, ang_vel_scale = self.scales
        samples: list[dict[str, torch.Tensor]] = []
        for sample_idx in batch:
            # Backward-compat: accept legacy 2-tuple (ep_id, t) by routing to dir_idx=0.
            if len(sample_idx) == 3:
                dir_idx, ep_id, t = sample_idx
            else:
                ep_id, t = sample_idx
                dir_idx = 0
            episode = self.dataset.load_episode(int(dir_idx), int(ep_id))
            t = int(t)
            s_a_world = episode["s_a_world"][t - tau : t + 1]                   # (τ+1, 18)
            s_d_world = episode["s_d_world"][t - tau : t + 1]                   # (τ+1, 18)
            a_a = episode["a_a"][t - tau : t]                                   # (τ, 4)
            samples.append(build_sample(
                s_a_world_window=s_a_world,
                s_d_world_window=s_d_world,
                a_a_window=a_a,
                pos_scale=pos_scale,
                vel_scale=vel_scale,
                ang_vel_scale=ang_vel_scale,
            ))

        batched = {
            "attacker_history": torch.stack([s["attacker_history"] for s in samples]),  # (B, τ+1, 18)
            "defender_delayed": torch.stack([s["defender_delayed"] for s in samples]),  # (B, 1, 6)
            "action_history":   torch.stack([s["action_history"]   for s in samples]),  # (B, τ, 4)
            "target":           torch.stack([s["target"]           for s in samples]),  # (B, τ, 6)
        }
        return batched, tau


# ---------------------------------------------------------------------------
# Convenience builder used by training script + tests
# ---------------------------------------------------------------------------

def build_dataset_and_collator(
    data_dir: str | Path | list[str | Path],
    mode: str,
    tau_max: int = 50,
    split_seed: int = 42,
    in_memory: bool = False,
    episode_subset: set[int] | None = None,
    rng_seed: int | None = None,
) -> tuple[ForecasterDataset, Pattern2Collator]:
    dataset = ForecasterDataset(
        data_dir=data_dir, tau_max=tau_max, mode=mode,
        split_seed=split_seed, in_memory=in_memory,
        episode_subset=episode_subset,
    )
    collator = Pattern2Collator(
        dataset=dataset, tau_max=tau_max, rng_seed=rng_seed,
    )
    return dataset, collator
