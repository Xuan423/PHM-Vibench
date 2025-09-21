"""Lightning callbacks for contrastive/few-shot telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import torch
import pytorch_lightning as pl


DEFAULT_METRIC_KEYS: Sequence[str] = (
    "train_total_loss",
    "train_cls_loss",
    "train_contrastive_loss",
    "train_acc",
    "val_total_loss",
    "val_cls_loss",
    "val_contrastive_loss",
    "val_acc",
)


def _iter_loggers(trainer: pl.Trainer) -> Iterator[pl.loggers.Logger]:
    logger_collection = getattr(trainer, "loggers", None)
    if logger_collection:
        for logger in logger_collection:
            yield logger
        return

    logger = getattr(trainer, "logger", None)
    if logger is None:
        return
    try:
        for item in logger:  # LoggerCollection is iterable
            yield item
        return
    except TypeError:
        yield logger


def _to_plain_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and callable(value.item):  # torch, numpy scalars
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_value(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _to_plain_value(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {k: _to_plain_value(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _flatten_dict(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            items.update(_flatten_dict(value, new_key))
        else:
            items[new_key] = value
    return items


def _collect_diagnostics(module: pl.LightningModule) -> Dict[str, float]:
    diagnostics: Dict[str, float] = {}

    embedding_stats = getattr(module, "embedding_stats", None)
    if isinstance(embedding_stats, Mapping):
        for key, value in embedding_stats.items():
            plain = _to_plain_value(value)
            if isinstance(plain, (int, float)):
                diagnostics[f"embedding_{key}"] = float(plain)

    latest_rep = getattr(module, "latest_representation", None)
    if isinstance(latest_rep, torch.Tensor):
        try:
            rep = latest_rep.detach().float()
            if rep.ndim >= 2:
                diagnostics.setdefault("embedding_norm", rep.norm(dim=-1).mean().item())
            else:
                diagnostics.setdefault("embedding_norm", rep.norm().item())
        except Exception:  # pragma: no cover - defensive
            pass

    grad_norm = getattr(module, "latest_grad_norm", None)
    plain_grad = _to_plain_value(grad_norm)
    if isinstance(plain_grad, (int, float)):
        diagnostics.setdefault("grad_norm", float(plain_grad))

    return diagnostics


@dataclass
class ContrastiveTelemetryCallback(pl.Callback):
    """Callback that mirrors contrastive/few-shot telemetry to active loggers."""

    summary_payload: MutableMapping[str, Any]
    metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS
    enabled: bool = True

    def __post_init__(self) -> None:
        self.summary_payload = {k: _to_plain_value(v) for k, v in self.summary_payload.items()}
        self._flat_summary = _flatten_dict(self.summary_payload)
        self._summary_logged = False
        self._latest_diagnostics: Dict[str, float] = {}

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # pragma: no cover - exercised in integration
        if not self.enabled or not self._flat_summary:
            return
        self._log_summary(trainer)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return
        metrics: Dict[str, float] = {}
        callback_metrics = getattr(trainer, "callback_metrics", {})
        for key in self.metric_keys:
            if key in callback_metrics:
                plain = _to_plain_value(callback_metrics[key])
                if isinstance(plain, (int, float)):
                    metrics[key] = float(plain)

        diagnostics = _collect_diagnostics(pl_module)
        self._latest_diagnostics = {k: v for k, v in diagnostics.items() if isinstance(v, (int, float))}

        if not metrics:
            return

        epoch = getattr(trainer, "current_epoch", 0)
        payload = {f"metrics.{k}": v for k, v in metrics.items()}
        payload["metrics.epoch"] = epoch

        for logger in _iter_loggers(trainer):
            try:
                logger.log_metrics(payload, step=epoch)
            except Exception:  # pragma: no cover - defensive
                continue
            if self._latest_diagnostics:
                self._log_hyperparams(logger, {"latest_diagnostics": self._latest_diagnostics, "latest_epoch": epoch})

    def _log_summary(self, trainer: pl.Trainer) -> None:
        if self._summary_logged:
            return
        for logger in _iter_loggers(trainer):
            self._log_hyperparams(logger, self.summary_payload)
        self._summary_logged = True

    def _log_hyperparams(self, logger: pl.loggers.Logger, payload: Dict[str, Any]) -> None:
        plain_payload = {k: _to_plain_value(v) for k, v in payload.items()}
        if hasattr(logger, 'log_hyperparams'):
            try:
                logger.log_hyperparams(plain_payload)
            except Exception:  # pragma: no cover - defensive
                pass
        experiment = getattr(logger, 'experiment', None)
        if experiment is not None:
            config_obj = getattr(experiment, 'config', None)
            if config_obj is not None:
                try:
                    config_obj.update(plain_payload, allow_val_change=True)
                except Exception:  # pragma: no cover - best effort
                    pass


__all__ = ["ContrastiveTelemetryCallback"]
