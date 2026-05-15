"""Phase 3f — τ-Unaware Flow Matching Forecaster module.

NEW package — does NOT modify Phase 3c τ-aware forecaster (CLAUDE.md §1).
"""

from agents.forecaster.tau_unaware import constants
from agents.forecaster.tau_unaware.dataset import (
    TauStripCollator,
    TauUnawareDataset,
    build_dataset_and_collator,
    build_sample_unaware,
)
from agents.forecaster.tau_unaware.flow import (
    cfm_loss,
    deterministic_mse_loss,
    sample_via_euler,
)
from agents.forecaster.tau_unaware.model import (
    TauUnawareDeterministicModel,
    TauUnawareForecasterModel,
)
from agents.forecaster.tau_unaware.trainer import (
    build_train_loader,
    build_val_dataset_and_collator,
    cosine_with_warmup_lr,
    load_ckpt,
    save_ckpt,
    set_lr,
    validate,
)

__all__ = [
    "TauStripCollator",
    "TauUnawareDataset",
    "TauUnawareDeterministicModel",
    "TauUnawareForecasterModel",
    "build_dataset_and_collator",
    "build_sample_unaware",
    "build_train_loader",
    "build_val_dataset_and_collator",
    "cfm_loss",
    "constants",
    "cosine_with_warmup_lr",
    "deterministic_mse_loss",
    "load_ckpt",
    "sample_via_euler",
    "save_ckpt",
    "set_lr",
    "validate",
]
