"""``dict`` subclass exposing ``.to_dict()`` for rsl-rl 5.x wandb logger.

rsl-rl's ``WandbSummaryWriter.store_config`` calls ``env_cfg.to_dict()``
first (legged_gym convention), then falls back to ``asdict(env_cfg)``
(dataclass convention). Plain dict raises ``AttributeError`` on the first
and ``TypeError`` on the second — the wandb logger then aborts learning.

This wrapper preserves the ``dict`` interface (so existing key/value access
in env code stays identical) while satisfying the wandb path. The
tensorboard logger never calls ``.to_dict()`` and is unaffected.
"""

from __future__ import annotations


class CfgDict(dict):
    """dict that also exposes ``.to_dict()``."""

    def to_dict(self) -> dict:
        return dict(self)
