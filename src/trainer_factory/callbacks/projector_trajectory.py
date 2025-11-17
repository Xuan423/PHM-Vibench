"""Lightning callback that records projector trajectories for each stage."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

from ...task_factory.utils.projector_diag import snapshot_projector


class ProjectorTrajectoryLogger(pl.Callback):
    """Snapshot projector weights at the start/end of each training stage."""

    def __init__(
        self,
        *,
        snapshot_dir: Path,
        stage_name: str,
        compare_snapshot: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.snapshot_dir = Path(snapshot_dir)
        self.stage_name = stage_name
        self.compare_snapshot = compare_snapshot
        self.start_snapshot: Optional[str] = None
        self.latest_snapshot: Optional[str] = None

    # ------------------------------------------------------------------
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        manifest = self._capture(pl_module, event="start", compare_manifest=self.compare_snapshot)
        if manifest:
            self.start_snapshot = manifest

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        reference = self.compare_snapshot or self.start_snapshot
        manifest = self._capture(pl_module, event="end", compare_manifest=reference)
        if manifest:
            self.latest_snapshot = manifest

    # ------------------------------------------------------------------
    def _capture(
        self,
        pl_module: pl.LightningModule,
        *,
        event: str,
        compare_manifest: Optional[str],
    ) -> Optional[str]:
        network = getattr(pl_module, "network", pl_module)
        return snapshot_projector(
            network,
            self.snapshot_dir,
            stage=self.stage_name,
            event=event,
            compare_manifest=compare_manifest,
        )
