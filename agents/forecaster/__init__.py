"""Phase 3c — Forecaster module (RF + Chi_Transformer encoder-decoder)."""

from agents.forecaster.dataset import (
    ForecasterDataset,
    Pattern2Collator,
    build_dataset_and_collator,
    build_sample,
    load_scales_from_dataset,
)
from agents.forecaster.model import ForecasterModel, SinusoidalEmbedding
from agents.forecaster.rectified_flow import rf_loss, sample_via_euler
from agents.forecaster.trainer import (
    build_train_loader,
    build_val_dataset_and_collator,
    cosine_with_warmup_lr,
    load_ckpt,
    save_ckpt,
    set_lr,
    validate,
)

__all__ = [
    "ForecasterDataset",
    "ForecasterModel",
    "Pattern2Collator",
    "SinusoidalEmbedding",
    "build_dataset_and_collator",
    "build_sample",
    "build_train_loader",
    "build_val_dataset_and_collator",
    "cosine_with_warmup_lr",
    "load_ckpt",
    "load_scales_from_dataset",
    "rf_loss",
    "sample_via_euler",
    "save_ckpt",
    "set_lr",
    "validate",
]
