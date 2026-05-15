"""Phase 3f — τ-Unaware Forecaster constants (TODO §3f, 2026-05-14)."""

# History window length (model input). Independent of τ; τ ∈ [1, DEFAULT_HISTORY_K].
DEFAULT_HISTORY_K = 50

# Per-modality input dimensions (match Phase 3c forecaster).
ATTACKER_DIM = 18    # pos(3) + vel(3) + rot_flat(9) + ang_vel(3)
DEFENDER_DIM = 6     # rel_pos(3) + rel_vel(3)
ACTION_DIM = 4
TARGET_DIM = 6       # single-step defender state in body frame
