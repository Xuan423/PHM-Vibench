"""Diagnostics callback for TF_MultiProtoDG-style interpretable exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl


class InterpretableTFDiagnosticsCallback(pl.Callback):
    def __init__(self, output_dir: str | Path) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.diagnostics_dir = self.output_dir / "diagnostics"
        self.feature_map_json = self.diagnostics_dir / "feature_map.json"
        self.prototype_cards_json = self.diagnostics_dir / "prototype_cards.json"
        self.prototype_health_json = self.diagnostics_dir / "prototype_health.json"
        self.context_json = self.diagnostics_dir / "diagnostics_context.json"

    def _enabled(self, pl_module: pl.LightningModule) -> bool:
        network = getattr(pl_module, "network", None)
        return bool(
            getattr(network, "export_diagnostics", False)
            and hasattr(network, "export_diagnostics_payload")
            and hasattr(network, "reset_diagnostics")
        )

    def _append_json(self, path: Path, record: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    data = loaded
            except Exception:
                data = []
        data.append(record)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_feature_map_once(self, payload: Dict[str, Any]) -> None:
        if self.feature_map_json.exists():
            return
        feature_map = payload.get("feature_map")
        if feature_map is None:
            return
        self.feature_map_json.parent.mkdir(parents=True, exist_ok=True)
        self.feature_map_json.write_text(
            json.dumps(feature_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_context_once(self, payload: Dict[str, Any]) -> None:
        if self.context_json.exists():
            return
        context = {
            "variant_id": payload.get("variant_id", "baseline"),
            "active_components": payload.get("active_components", {}),
            "active_feature_count": payload.get("active_feature_count"),
        }
        self.context_json.parent.mkdir(parents=True, exist_ok=True)
        self.context_json.write_text(
            json.dumps(context, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _export_stage(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if not self._enabled(pl_module):
            return
        payload = pl_module.network.export_diagnostics_payload(stage, int(getattr(trainer, "current_epoch", 0)))
        if not payload:
            return
        self._write_feature_map_once(payload)
        self._write_context_once(payload)
        cards = payload.get("prototype_cards")
        health = payload.get("prototype_health")
        record = {
            "epoch": payload.get("epoch"),
            "stage": payload.get("stage"),
            "variant_id": payload.get("variant_id", "baseline"),
            "active_components": payload.get("active_components", {}),
            "active_feature_count": payload.get("active_feature_count"),
        }
        if cards is not None:
            self._append_json(self.prototype_cards_json, {**record, "items": cards})
        if health is not None:
            self._append_json(self.prototype_health_json, {**record, "items": health})
        pl_module.network.reset_diagnostics(stage)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._enabled(pl_module):
            pl_module.network.reset_diagnostics("train")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._enabled(pl_module):
            pl_module.network.reset_diagnostics("val")

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._enabled(pl_module):
            pl_module.network.reset_diagnostics("test")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._export_stage(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._export_stage(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._export_stage(trainer, pl_module, "test")
