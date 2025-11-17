"""Callback that periodically persists prototype registries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl


class PrototypeCheckpointCallback(pl.Callback):
    """Saves the model's ``prototype_registry`` at a configurable cadence."""

    def __init__(self, cache_dir: str, interval: int = 1, enabled: bool = True) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.interval = max(1, int(interval))
        self.enabled = enabled

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return
        if (trainer.current_epoch + 1) % self.interval != 0:
            return
        self._checkpoint(pl_module, suffix=f"epoch{trainer.current_epoch + 1}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return
        self._checkpoint(pl_module, suffix="final")

    # ------------------------------------------------------------------
    def _checkpoint(self, pl_module: pl.LightningModule, *, suffix: str) -> None:
        registry = getattr(pl_module, "prototype_registry", None)
        if registry is None:
            return
        target = self.cache_dir / f"prototypes-{suffix}.pt"
        registry.checkpoint(target)
