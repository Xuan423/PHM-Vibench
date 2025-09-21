"""Contrastive-aware trainer factory built on top of the default trainer wiring."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

try:  # SwanLab is optional in some environments
    from swanlab.integration.pytorch_lightning import SwanLabLogger  # type: ignore
except Exception:  # pragma: no cover - optional dependency in CI
    SwanLabLogger = None  # type: ignore

from .Default_trainer import Prune_callback, create_early_stopping_callback
from .callbacks import ContrastiveTelemetryCallback
from .callbacks.contrastive_logging import DEFAULT_METRIC_KEYS


def _is_main_process() -> bool:
    """Detect whether the current process is responsible for external logging."""
    try:
        return int(os.environ.get("LOCAL_RANK", "0")) == 0
    except ValueError:  # pragma: no cover - defensive fallback
        return True


def trainer(args_e, args_t, args_d, path):
    """Build a PyTorch Lightning trainer tuned for contrastive experiments."""
    callbacks, _ = _build_callbacks(args_t, path)
    loggers = _build_loggers(args_e, args_t, path)

    if _should_enable_contrastive_logging(args_t):
        summary_payload = _build_summary_payload(args_t, args_d)
        metric_keys = _resolve_metric_keys(args_t)
        callbacks.append(
            ContrastiveTelemetryCallback(
                summary_payload=summary_payload,
                metric_keys=metric_keys,
                enabled=_is_main_process(),
            )
        )

    accelerator = 'cpu' if getattr(args_t, 'device', None) == 'cpu' else 'auto'
    if not getattr(args_t, 'log_every_n_steps', None):
        args_t.log_every_n_steps = 50

    return pl.Trainer(
        callbacks=callbacks,
        accelerator=accelerator,
        max_epochs=args_t.num_epochs,
        devices=args_t.gpus,
        logger=loggers,
        log_every_n_steps=args_t.log_every_n_steps,
        check_val_every_n_epoch=getattr(args_t, 'check_val_every_n_epoch', 1),
        val_check_interval=getattr(args_t, 'val_check_interval', 1.0),
        strategy='ddp_find_unused_parameters_true' if getattr(args_t, 'gpus', 1) > 1 else 'auto',
    )


def _build_callbacks(args_t, path: str) -> Tuple[List[pl.Callback], str]:
    """Create checkpoint/early stopping callbacks using YAML-configured monitor."""
    monitor_metric, mode, save_top_k = _resolve_checkpoint_config(args_t)

    checkpoint_filename = f"model-{{epoch:02d}}-{{{monitor_metric}:.4f}}"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        filename=checkpoint_filename,
        save_top_k=save_top_k,
        mode=mode,
        dirpath=path,
    )

    callbacks: List[pl.Callback] = [checkpoint_callback]

    if getattr(args_t, 'pruning', False):
        prune_callback = Prune_callback(args_t)
        if prune_callback is not None:
            callbacks.append(prune_callback)

    if getattr(args_t, 'early_stopping', True):
        # Reuse the default helper but inject our resolved monitor metric.
        args_with_monitor = deepcopy(args_t)
        setattr(args_with_monitor, 'monitor', monitor_metric)
        early_stopping = create_early_stopping_callback(args_with_monitor)
        callbacks.append(early_stopping)

    return callbacks, monitor_metric


def _resolve_checkpoint_config(args_t) -> Tuple[str, str, int]:
    """Infer monitor metric, optimization mode, and save_top_k from config."""
    callbacks_cfg = getattr(args_t, 'callbacks', None)

    monitor = getattr(args_t, 'monitor', None)
    mode = getattr(args_t, 'mode', None)
    save_top_k = getattr(args_t, 'save_top_k', 1)

    if callbacks_cfg is not None:
        monitor = getattr(callbacks_cfg, 'checkpoint_monitor', monitor)
        mode = getattr(callbacks_cfg, 'checkpoint_mode', getattr(callbacks_cfg, 'mode', mode))
        save_top_k = getattr(callbacks_cfg, 'save_top_k', save_top_k)

    if not monitor:
        monitor = 'val_total_loss'

    if not mode:
        mode = 'min' if 'loss' in monitor else 'max'

    return monitor, mode, save_top_k


def _build_loggers(args_e, args_t, path: str):
    """Construct Lightning loggers, guarding external services on non-main processes."""
    loggers = [CSVLogger(path, name="logs")]
    main_process = _is_main_process()

    project_name = getattr(args_e, 'project', getattr(args_t, 'project', 'vbench'))

    use_wandb = getattr(args_t, 'wandb', getattr(args_e, 'wandb', False))
    wandb_mode = os.environ.get('WANDB_MODE', '').lower()
    wandb_offline = True if wandb_mode == 'disabled' else False
    if use_wandb and main_process:
        loggers.append(
            WandbLogger(
                project=project_name,
                save_dir=path,
                offline=wandb_offline,
            )
        )

    use_swanlab = getattr(args_t, 'swanlab', getattr(args_e, 'swanlab', False))
    if use_swanlab and main_process and SwanLabLogger is not None:
        loggers.append(
            SwanLabLogger(
                project=project_name,
            )
        )

    return loggers


def _should_enable_contrastive_logging(args_t) -> bool:
    logging_cfg = getattr(args_t, 'logging', None)
    if logging_cfg is None:
        return False
    return bool(getattr(logging_cfg, 'contrastive_metrics', False))


def _resolve_metric_keys(args_t) -> Sequence[str]:
    logging_cfg = getattr(args_t, 'logging', None)
    if logging_cfg is None:
        return DEFAULT_METRIC_KEYS
    custom_keys = getattr(logging_cfg, 'contrastive_metric_keys', None)
    if not custom_keys:
        return DEFAULT_METRIC_KEYS
    return tuple(str(key) for key in custom_keys)


def _build_summary_payload(args_t, args_d) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    dataset_section = {}
    for key in ('target_system_id', 'target_domain_id', 'source_domain_id'):
        value = getattr(args_t, key, None)
        if value is not None:
            dataset_section[key] = _to_plain(value)
    metadata_file = getattr(args_d, 'metadata_file', None)
    if metadata_file:
        dataset_section['metadata_file'] = metadata_file
    if dataset_section:
        summary['dataset'] = dataset_section

    few_shot_cfg = getattr(args_t, 'few_shot', None)
    if few_shot_cfg and getattr(few_shot_cfg, 'enabled', False):
        summary['few_shot'] = _extract_known_keys(
            few_shot_cfg,
            (
                'enabled',
                'sampler',
                'systems_per_episode',
                'domains_per_episode',
                'classes_per_domain',
                'support_per_class',
                'query_per_class',
                'episodes_per_epoch',
                'preserve_labels',
                'warn_on_shortfall',
            ),
        )

    task_contrastive = getattr(args_t, 'contrastive', None)
    if task_contrastive:
        summary['contrastive_task'] = _extract_known_keys(
            task_contrastive,
            (
                'loss_weight',
                'temperature',
                'mode',
                'normalize_embeddings',
            ),
        )

    model_cfg = getattr(args_t, 'model', None)
    if model_cfg is not None:
        model_contrastive = getattr(model_cfg, 'contrastive', None)
        if model_contrastive:
            summary['contrastive_model'] = _extract_known_keys(
                model_contrastive,
                (
                    'enabled',
                    'projection_hidden',
                    'projection_dim',
                    'temperature',
                    'loss_weight',
                    'mode',
                ),
            )

    summary['logging'] = {
        'contrastive_metrics': bool(getattr(getattr(args_t, 'logging', None), 'contrastive_metrics', False))
    }

    return summary


def _extract_known_keys(namespace, keys: Sequence[str]) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    for key in keys:
        if hasattr(namespace, key):
            value = getattr(namespace, key)
            plain_value = _to_plain(value)
            if plain_value is not None:
                extracted[key] = plain_value
    return extracted


def _to_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'item') and callable(value.item):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return None
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if hasattr(value, '__dict__'):
        return {k: _to_plain(v) for k, v in vars(value).items() if not k.startswith('_')}
    return value


__all__ = ["trainer"]
